#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What does a *variadic* two-shot cost, before writing one?

XLA does not concatenate K3's latent-MoE pair. Its combiner emits a single
all-reduce with two operands -- ``all-reduce bf16[1,3584],bf16[1,7168]``, visible
in any fusion-probe trace -- so it amortises one collective's synchronisation
across two buffers while moving no data to join them. Our two-shot cannot do
that today: it scatters over a flat ``(N, 1, numel/N)`` view, and the reshape
into that view is a physical retile (112 lanes against a 128-lane register),
which is where job 151's ``data formatting`` went. Three attempts at removing the
concatenation failed because the concatenation was never the cost.

The obvious fix is to copy XLA's shape: keep the two operands separate, in their
natural layout, and issue *two* remote copies per peer from one kernel, sharing
one barrier. No join, no reshape. Two things could sink it, and both are cheaper
to measure than to discover:

  1. **Unaligned dynamic offsets.** Operand a's chunk starts at ``peer * 112``
     and b's at ``peer * 224``; neither is a multiple of the 128-lane register,
     and HBM arrays are tiled too. ``causal_conv1d`` slices an HBM ref at a
     dynamic offset (``hbm_ref.at[pl.ds(b_start, size)]``) so the primitive
     exists, but not that an *unaligned* one lowers.

  2. **Twice the DMAs.** 2(N-1) remote copies per phase instead of N-1. The
     ablation put the all-gather at 4.65 us for 31 copies -- about 0.15 us each,
     issue-dominated rather than byte-dominated at this payload -- so doubling
     could cost ~4.6 us/layer against the ~3.3 us of formatting it would save.
     On paper that is break-even at best.

So this measures the DMA pattern alone: barrier, N-1 (or 2(N-1)) reduce-scatter
sends, wait, the same again for the all-gather, wait. No arithmetic, no
workspace threading, nothing but the traffic under test. Three variants:

    joined            1 copy/peer of ca+cb, sourced from a reshaped VMEM ref
                      -- today's structure, the baseline
    split_aligned     2 copies/peer at 128 and 256, dynamic *aligned* offsets
                      into 1-D HBM refs -- isolates the cost of doubling
    split_unaligned   2 copies/peer at 112 and 224, K3's real widths
                      -- isolates alignment, and answers (1) by running at all

joined vs split_aligned prices question 2. split_aligned vs split_unaligned
prices question 1. If split_unaligned fails to lower, that is the answer to (1)
and the design needs whole-tile chunking instead.

VERDICT (job 153, 32 devices) -- the design is not reachable, and question 2
never got asked:

    variant           copies/phase   us/call
    joined                      31    381.99
    split_aligned                -    FAILED  MosaicError: Slice sizes along
    split_unaligned              -    FAILED  tiled dimensions must be aligned
                                              to tiling

Both split variants fail identically, *including the aligned one*. So 128-lane
alignment is not the bar: a 1-D bf16 array is tiled in two dimensions
(sublanes x lanes, ~16 x 128 = 2048 elements), and every per-device chunk here
is 112-256 elements -- an order of magnitude below one tile. No choice of
operand width fixes that; K3 would need ~18x padding to make each chunk a whole
tile.

This is why the reshape to ``(N, 1, cols)`` in ``_build_two_shot`` exists and is
not optional. It puts the device index on the *leading* dimension, which is not
tiled, so ``.at[peer]`` is a legal slice. The relayout that costs ~3.3 us/layer
is the admission price for expressing a device-chunked collective in Pallas at
all, not an oversight.

And it is the precise mechanism behind the XLA asymmetry: XLA's collective
engine chunks by byte range, which has no tiling constraint; Mosaic's DMA
operates on tiled arrays, which does. Copying XLA's variadic shape is sound by
analogy and blocked by the framework.

NOTE ON THE TIMING HARNESS: the 381.99 us above is host dispatch, not device
time -- the same collective measures ~12.5 us in fusion_probe.py. Each call
here does its own block_until_ready, so the slope between iters-lo and iters-hi
cancels fixed setup but not per-call host overhead. Chain the calls inside one
jitted function before trusting any number this script prints. It did not matter
for the question that got answered (compiles or does not), but it would for the
one that did not.

Run (32 devices, 4 hosts):
    sbatch scripts/debug/slurm_probe_variadic_dma.sbatch
"""

import argparse
import os
import time

import jax
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import \
    _full_barrier

VARIANTS = ("joined", "split_aligned", "split_unaligned")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=3584)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def _joined_kernel(x_ref, rs_ref, ag_ref, o_ref, rs_o, ag_o, mine, sem, s1s,
                   s1r, s2s, s2r, *, num_devices: int, axis_name: str):
    """One copy per peer, sourced from the reshaped (N, 1, C) VMEM view."""
    del rs_ref, ag_ref
    my_id = lax.axis_index(axis_name)
    _full_barrier(my_id, num_devices, double_barrier=True)

    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        c = pltpu.make_async_remote_copy(src_ref=x_ref.at[peer],
                                         dst_ref=rs_o.at[i - 1],
                                         send_sem=s1s,
                                         recv_sem=s1r,
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH)
        c.start()
        copies.append(c)
    for c in copies:
        c.wait()

    mine[...] = x_ref[0]
    copies = []
    for i in range(1, num_devices):
        c = pltpu.make_async_remote_copy(src_ref=mine,
                                         dst_ref=ag_o.at[my_id],
                                         send_sem=s2s,
                                         recv_sem=s2r,
                                         device_id=(lax.rem(
                                             my_id + i, num_devices), ),
                                         device_id_type=pl.DeviceIdType.MESH)
        c.start()
        copies.append(c)
    for c in copies:
        c.wait()
    o_ref[0] = 1


def _split_kernel(a_ref, b_ref, rsa_ref, rsb_ref, aga_ref, agb_ref, o_ref,
                  rsa_o, rsb_o, aga_o, agb_o, mine_a, mine_b, sem, s1s, s1r,
                  s2s, s2r, *, num_devices: int, axis_name: str, ca: int,
                  cb: int):
    """Two copies per peer, sourced from 1-D HBM refs at dynamic offsets.

    This is the pattern under test: the operands keep their natural layout and
    are never joined, so the only thing that grows is the number of DMAs.
    """
    del rsa_ref, rsb_ref, aga_ref, agb_ref
    my_id = lax.axis_index(axis_name)
    _full_barrier(my_id, num_devices, double_barrier=True)

    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copies.append(
            pltpu.make_async_remote_copy(src_ref=a_ref.at[pl.ds(peer * ca,
                                                                ca)],
                                         dst_ref=rsa_o.at[i - 1],
                                         send_sem=s1s,
                                         recv_sem=s1r,
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH))
        copies[-1].start()
        copies.append(
            pltpu.make_async_remote_copy(src_ref=b_ref.at[pl.ds(peer * cb,
                                                                cb)],
                                         dst_ref=rsb_o.at[i - 1],
                                         send_sem=s1s,
                                         recv_sem=s1r,
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH))
        copies[-1].start()
    for c in copies:
        c.wait()

    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copies.append(
            pltpu.make_async_remote_copy(src_ref=mine_a,
                                         dst_ref=aga_o.at[my_id],
                                         send_sem=s2s,
                                         recv_sem=s2r,
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH))
        copies[-1].start()
        copies.append(
            pltpu.make_async_remote_copy(src_ref=mine_b,
                                         dst_ref=agb_o.at[my_id],
                                         send_sem=s2s,
                                         recv_sem=s2r,
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH))
        copies[-1].start()
    for c in copies:
        c.wait()
    o_ref[0] = 1


def build_joined(mesh, axis, n, cols, dtype, cid):
    import functools

    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    kernel = functools.partial(_joined_kernel, num_devices=n, axis_name=axis)

    def call(x, rs, ag):
        return pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct(
                (1, ), jnp.int32), jax.ShapeDtypeStruct(rs.shape, dtype),
                       jax.ShapeDtypeStruct(ag.shape, dtype)),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            input_output_aliases={
                1: 1,
                2: 2
            },
            scratch_shapes=[
                pltpu.VMEM((1, cols), dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
            compiler_params=pltpu.CompilerParams(
                collective_id=cid,
                has_side_effects=True,
                disable_semaphore_checks=True),
        )(x, rs, ag)

    def fn(x, rs, ag):
        return jax.shard_map(call,
                             mesh=mesh,
                             in_specs=(P(), P(), P()),
                             out_specs=(P(), P(), P()),
                             check_vma=False)(x, rs, ag)

    return jax.jit(fn)


def build_split(mesh, axis, n, ca, cb, dtype, cid):
    import functools

    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    kernel = functools.partial(_split_kernel,
                               num_devices=n,
                               axis_name=axis,
                               ca=ca,
                               cb=cb)

    def call(a, b, rsa, rsb, aga, agb):
        return pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct(
                (1, ), jnp.int32), jax.ShapeDtypeStruct(rsa.shape, dtype),
                       jax.ShapeDtypeStruct(rsb.shape, dtype),
                       jax.ShapeDtypeStruct(aga.shape, dtype),
                       jax.ShapeDtypeStruct(agb.shape, dtype)),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 6,
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            input_output_aliases={
                2: 1,
                3: 2,
                4: 3,
                5: 4
            },
            scratch_shapes=[
                pltpu.VMEM((ca, ), dtype),
                pltpu.VMEM((cb, ), dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
            compiler_params=pltpu.CompilerParams(
                collective_id=cid,
                has_side_effects=True,
                disable_semaphore_checks=True),
        )(a, b, rsa, rsb, aga, agb)

    def fn(a, b, rsa, rsb, aga, agb):
        return jax.shard_map(call,
                             mesh=mesh,
                             in_specs=(P(), ) * 6,
                             out_specs=(P(), ) * 5,
                             check_vma=False)(a, b, rsa, rsb, aga, agb)

    return jax.jit(fn)


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # tpu_dist initialises the device mesh; gloo carries the pickled gather,
    # which tpu_dist cannot. Same split as probe_pallas_collective.py.
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    def lead(msg):
        if rank == 0:
            print(msg, flush=True)

    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from torch_tpu._internal.distributed import tpu_distributed

    torch.empty(1, device="tpu")
    devs = jax.devices()
    mine = {
        "rank": rank,
        "global_device_id": int(tpu_distributed.global_device_id()),
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, mine, group=gloo)
    rank_to_device = {f["rank"]: f["global_device_id"] for f in gathered}
    by_id = {int(getattr(d, "id", i)): d for i, d in enumerate(devs)}
    ordered = np.asarray([by_id[rank_to_device[r]] for r in range(world_size)])
    mesh = Mesh(ordered, axis_names=("tp", ))

    n = world_size
    dtype = jnp.bfloat16
    ca_real, cb_real = args.latent // n, args.hidden // n
    ca_pad, cb_pad = ((ca_real + 127) // 128) * 128, (
        (cb_real + 127) // 128) * 128
    cols = ca_real + cb_real

    lead(f"world_size={n}  latent={args.latent} hidden={args.hidden}")
    lead(f"per-device: real a={ca_real} b={cb_real} (joined {cols}); "
         f"aligned a={ca_pad} b={cb_pad}")
    lead(f"\n{'variant':>16} {'copies/phase':>13} {'us/call':>10}  note")

    cid = 8
    for variant in args.variants.split(","):
        variant = variant.strip()
        if not variant:
            continue
        try:
            if variant == "joined":
                x = jnp.zeros((n, 1, cols), dtype)
                rs = jnp.zeros((n - 1, 1, cols), dtype)
                ag = jnp.zeros((n, 1, cols), dtype)
                fn = build_joined(mesh, "tp", n, cols, dtype, cid)
                bufs = (x, rs, ag)
                per_phase = n - 1
            else:
                ca, cb = ((ca_pad, cb_pad) if variant == "split_aligned" else
                          (ca_real, cb_real))
                a = jnp.zeros((n * ca, ), dtype)
                b = jnp.zeros((n * cb, ), dtype)
                rsa = jnp.zeros((n - 1, ca), dtype)
                rsb = jnp.zeros((n - 1, cb), dtype)
                aga = jnp.zeros((n, ca), dtype)
                agb = jnp.zeros((n, cb), dtype)
                fn = build_split(mesh, "tp", n, ca, cb, dtype, cid)
                bufs = (a, b, rsa, rsb, aga, agb)
                per_phase = 2 * (n - 1)
            cid += 1

            def once(bufs=bufs, fn=fn, variant=variant):
                out = fn(*bufs)
                jax.block_until_ready(out)

            for _ in range(args.warmup):
                once()

            def chain(k):
                best = None
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    for _ in range(k):
                        once()
                    dt = time.perf_counter() - t0
                    best = dt if best is None else min(best, dt)
                return best

            lo = chain(args.iters_lo)
            hi = chain(args.iters_hi)
            us = ((hi - lo) / (args.iters_hi - args.iters_lo)) * 1e6
            lead(f"{variant:>16} {per_phase:>13} {us:>10.2f}")
        except Exception as e:  # noqa: BLE001
            lead(f"{variant:>16} {'-':>13} {'FAILED':>10}  "
                 f"{type(e).__name__}: {str(e)[:160]}")

    dist.barrier(group=gloo)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
