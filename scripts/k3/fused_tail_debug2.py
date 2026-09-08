#!/usr/bin/env python3
"""Bisect the barrier_first score corruption: 3 score compute modes x
barrier on/off, single call per compile. The kernel below is a verbatim
copy of the shipped _fused_out_proj_ar_kernel with only the score
epilogue switched."""

import os
import sys
import time

if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

PACK, LANES = 2, 128
HID = 7168
NUM_DEVICES = 32
TILES = HID // (PACK * LANES)
EPS = 1e-5


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _full_barrier(my_id, num_devices: int):
    """All-to-all barrier across every device on the mesh axis."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(
            barrier_sem,
            1,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, num_devices - 1)


def _signal_round(sem, my_id, num_devices: int):
    """Signal every peer once (its copies from me have landed)."""
    for i in range(1, num_devices):
        pl.semaphore_signal(
            sem,
            1,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )


def _dbg_kernel(x_ref, w_ref, resid_ref, fw_ref, rs_ref, ag_ref, o_ref,
                inv_ref, score_ref, payload_vmem, recv_vmem, red_vmem,
                asm_vmem, local_sem, send_sem, recv_sem, arr1, arr2,
                fw_stage_vmem, *, num_devices: int, axis_name: str, tiles: int,
                slot: int, barrier_first: int, eps: float,
                steady_signals: bool, score_mode: int):
    my_id = lax.axis_index(axis_name)

    # Read fw BEFORE the entry barrier: with barrier_first set, a late
    # fw_ref[...] read (in the tail epilogue below) returns wrong values --
    # deterministic ~4.3x score corruption on the check's call 0 while
    # hidden/inv_rms stay exact (job 791-793). Reading it up front into a
    # value sidesteps whatever the collective barrier does to late input-ref
    # reads; the read is free either way.
    fw_vals = fw_ref[...].astype(jnp.float32)

    if barrier_first:
        _full_barrier(my_id, num_devices)

    # The o_proj partial, stored packed as [tiles, 2, 128] in VMEM. w is the
    # rank's row-parallel weight shard in the model's layout [k_shard,
    # width] (contracting dim FIRST -- that is how the torch_tpu
    # RowParallelLinear stores it, measured on device: the armed layer's
    # weight is [384, 7168] at TP=32).
    acc = jax.lax.dot_general(x_ref[...],
                              w_ref[...],
                              dimension_numbers=(((1, ), (0, )), ((), ())),
                              preferred_element_type=jnp.float32)
    payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

    # Phase 1 -- reduce-scatter. Push the tile peer ``+i`` owns to it
    # (offset-rem pattern: every rank receives exactly num_devices-1
    # copies -- tile-major sends leave ownerless ranks at zero receives and
    # deadlock). It lands in that peer's slot ``my_id``; a peer's own slot
    # is never written, so the reduce below can sum all slots (its own
    # contribution comes from payload_vmem).
    copies = []
    for i in range(1, num_devices):
        dst = lax.rem(my_id + i, num_devices)
        tile = lax.rem(dst, tiles)
        copy = pltpu.make_async_remote_copy(
            src_ref=payload_vmem.at[tile],
            dst_ref=rs_ref.at[slot, my_id],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(dst, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    # Delivered before the signal, so a peer that observes N-1 signals may
    # read its receive buffer. The round is UNCONDITIONAL: a fired-but-
    # unwatched semaphore is a hard device fault on this stack ("Semaphore
    # has a nonzero value upon exit from a Mosaic kernel", K3 warmup).
    # The signal round + wait gate the owner's read of its receive buffer.
    # They are UNCONDITIONAL when present: a fired-but-unwatched semaphore
    # is a hard device fault on this stack ("Semaphore has a nonzero value
    # upon exit from a Mosaic kernel", K3 warmup job 558), and for an
    # ownerless rank the wait only means "every peer's phase-1 deliveries
    # landed". With steady_signals=False the sender-side copy waits are the
    # whole protocol (see the wrapper's docstring for the safety argument).
    if steady_signals:
        _signal_round(arr1, my_id, num_devices)
        pl.semaphore_wait(arr1, num_devices - 1)

    # Owners reduce their tile. The local stage copy is the only DMA inside
    # a conditional -- remote DMA inside pl.when deadlocks on this stack
    # (probe job 485), local is fine. The owner also adds its tile's slice of
    # the residual stream: the residual is replicated across ranks, so the
    # add is exact and rides the owner-reduce instead of the assembly tail
    # (bench job 790; ref indexing by my_id -- an array dynamic_slice is
    # unimplemented in this lowering).
    @pl.when(my_id < tiles)
    def _():
        stage = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                      dst_ref=recv_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        red = payload_vmem[my_id]
        for j in range(num_devices):
            red = red + recv_vmem[j]
        red_vmem[...] = red + resid_ref[my_id]

    # Phase 2 -- all-gather. Fully uniform: no remote DMA inside a
    # conditional. Ownerless ranks broadcast their (garbage) red_vmem into
    # the trash slot ``tiles`` of everyone's assembly buffer, never read.
    out_slot = lax.min(my_id, tiles)
    local = pltpu.make_async_copy(src_ref=red_vmem,
                                  dst_ref=ag_ref.at[slot, out_slot],
                                  sem=local_sem)
    local.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, out_slot],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    local.wait()
    for copy in copies:
        copy.wait()
    # Assembly once every phase-2 tile has landed locally (or, with
    # steady_signals=False, once our own sends prove ours landed).
    if steady_signals:
        _signal_round(arr2, my_id, num_devices)
        pl.semaphore_wait(arr2, num_devices - 1)
    stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot, 0:tiles],
                                  dst_ref=asm_vmem,
                                  sem=local_sem)
    stage.start()
    stage.wait()
    # The result goes out in the natural [1, width] layout -- NOT the packed
    # [tiles, 2, 128] form: the packed custom-call result shifted XLA's
    # global layout assignment and materialized the [7168,384] MoE
    # weight-staging transposes (the Batch W contagion term, ~16 ms/window,
    # jobs 581/593). A boring [1, width] result doesn't propagate.
    #
    # Tail epilogue (bench jobs 787/790): the AttentionResidual prefix
    # branch that follows the AR in the model -- inv_rms and the folded
    # score -- needs only (hidden, fw), so both reduces ride the assembly.
    # hidden already carries the residual (owner-side add above).
    hidden = asm_vmem[...]
    h32 = hidden.astype(jnp.float32)
    inv = lax.rsqrt(jnp.sum(h32 * h32) / (tiles * PACK * LANES) + eps)
    inv_ref[...] = jnp.broadcast_to(inv, (1, ))
    if score_mode == 0:
        score = jnp.sum(h32 * fw_vals)
    elif score_mode == 1:
        # MXU path: [1, W] @ [W, 1] matmul instead of the VPU reduce.
        score = jax.lax.dot_general(h32.reshape(1, -1),
                                    fw_vals.reshape(-1, 1),
                                    dimension_numbers=(((1, ), (0, )), ((),
                                                                        ())),
                                    preferred_element_type=jnp.float32)[0, 0]
    else:
        # Fresh DMA restage of fw at the epilogue.
        c = pltpu.make_async_copy(src_ref=fw_ref,
                                  dst_ref=fw_stage_vmem,
                                  sem=local_sem)
        c.start()
        c.wait()
        score = jnp.sum(h32 * fw_stage_vmem[...])
    score_ref[...] = jnp.broadcast_to(score * inv, (1, ))
    o_ref[...] = hidden.reshape(o_ref.shape)


def dbg_call(x, w, resid, fw, rs, ag, *, barrier_first, score_mode):
    kernel = functools.partial(_dbg_kernel,
                               num_devices=NUM_DEVICES,
                               axis_name="tp",
                               tiles=TILES,
                               slot=0,
                               barrier_first=barrier_first,
                               eps=EPS,
                               steady_signals=False,
                               score_mode=score_mode)
    scratch = [
        pltpu.VMEM((TILES, PACK, LANES), jnp.bfloat16),
        pltpu.VMEM((NUM_DEVICES, PACK, LANES), jnp.bfloat16),
        pltpu.VMEM((PACK, LANES), jnp.bfloat16),
        pltpu.VMEM((TILES, PACK, LANES), jnp.bfloat16),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.REGULAR,
        pltpu.SemaphoreType.REGULAR,
    ]
    scratch.append(pltpu.VMEM((TILES, PACK, LANES), jnp.float32))
    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(
            (1, HID), jnp.bfloat16), jax.ShapeDtypeStruct(
                (1, ), jnp.float32), jax.ShapeDtypeStruct((1, ), jnp.float32)),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            collective_id=8 if barrier_first else None,
            has_side_effects=False,
            disable_semaphore_checks=True),
    )(x, w, resid, fw, rs, ag)


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    x = jax.jit(lambda: jax.random.normal(jax.random.key(3),
                                          (1, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P(None, "tp")))()
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1),
                                          (HID, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()
    resid = jax.jit(lambda: jax.random.normal(jax.random.key(5),
                                              (1, HID), jnp.bfloat16),
                    out_shardings=NamedSharding(mesh, P()))()
    fw = jax.jit(lambda: jax.random.normal(jax.random.key(7),
                                           (HID, ), jnp.float32),
                 out_shardings=NamedSharding(mesh, P()))()
    rs = jax.jit(lambda: jnp.zeros(
        (2, NUM_DEVICES, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()
    ag = jax.jit(lambda: jnp.zeros((2, TILES + 1, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()

    ref = jax.jit(lambda x, w, resid, fw: (jnp.dot(x.astype(
        jnp.float32), w.astype(jnp.float32)) + resid.astype(jnp.float32), ),
                  out_shardings=NamedSharding(mesh, P()))
    (hid_ref, ) = ref(x, w, resid, fw)
    inv_ref = float(lax.rsqrt(jnp.sum(hid_ref * hid_ref) / HID + EPS))
    score_ref = float(jnp.sum(hid_ref * fw) * inv_ref)

    for score_mode in (0, 1, 2):
        for barrier_first in (0, 1):

            def fn(x, w, resid, fw, rs, ag, m=score_mode, b=barrier_first):

                def body(xl, wl, residl, fwl, rs, ag):
                    return dbg_call(xl,
                                    wl,
                                    residl.reshape(TILES, PACK, LANES),
                                    fwl.reshape(TILES, PACK, LANES),
                                    rs,
                                    ag,
                                    barrier_first=b,
                                    score_mode=m)

                return jax.shard_map(body,
                                     mesh=mesh,
                                     in_specs=(P(None, "tp"), P("tp", None),
                                               P(), P(), P(), P()),
                                     out_specs=(P(), P(), P()),
                                     check_vma=False)(x, w, resid, fw, rs, ag)

            hid, inv, score = jax.jit(fn)(x, w, resid, fw, rs, ag)
            jax.block_until_ready((hid, inv, score))
            hid_err = float(
                jnp.max(jnp.abs(hid.astype(jnp.float32) - hid_ref)) /
                jnp.max(jnp.abs(hid_ref)))
            _stamp(
                rank, f"mode={score_mode} barrier={barrier_first}: "
                f"hid_err {hid_err:.3g} | inv {float(inv[0]):.6g} vs "
                f"{inv_ref:.6g} | score {float(score[0]):.6g} vs "
                f"{score_ref:.6g}")
    _stamp(rank, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
