#!/usr/bin/env python3
"""Microbenchmark for the fused o_proj+AR kernel at the K3 KDA T=1 shape.

The e2e arm (job 569) puts the fused op at ~20.7 us/call x 69 KDA layers
~= 1.43 ms/step -- the whole regression. This bench iterates on the kernel
at its production shape (x [1,384], w [384,7168] per rank, 32 devices) with
fast-mutating VARIANTS, each numerics-checked against the psum reference
and timed as an 8-call alternating-slot chain (the model's call pattern),
plus an optional xprof trace per variant for device-time ground truth.

Variants:
  baseline   -- the shipped kernel (kernels/collectives/
                fused_out_proj_all_reduce.py), imported.
  directout  -- AG assembly DMAs ag -> o_ref directly (skips the asm_vmem
                VMEM->VMEM hop).
  treereduce -- tree reduction over the 32 receive slots (5 dependent
                levels) instead of the serial 32-add chain.
  pipehalf   -- split the dot into two 3584-wide halves; the first half's
                RS sends go out while the second half computes. Uses the
                probe-proven tile-major receive-uniform mapping (jobs
                497/501/502): tile t's owners are {t, t+28}, self-copies go
                to the trash slot, ranks 28-31 receive duplicated tiles.
  combo      -- all three.

BENCH_VARIANTS selects a comma list; BENCH_PROFILE=1 captures traces into
$BENCH_ROOT/traces/<variant>/ (needs the probe venv's libtpu for the LLO
flags; wall-clock timings work in either venv).
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts. Join on every task.
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

try:
    from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import (
        LANES, PACK, fused_out_proj_all_reduce)
except ImportError:
    # The package __init__ pulls torch; the probe venv (used for LLO tracing)
    # has only jax. The kernel module is pure jax -- load it by path.
    import importlib.util

    _path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src/vllm_torchtpu/kernels/collectives/fused_out_proj_all_reduce.py")
    _spec = importlib.util.spec_from_file_location("fused_out_proj_all_reduce",
                                                   _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LANES, PACK = _mod.LANES, _mod.PACK
    fused_out_proj_all_reduce = _mod.fused_out_proj_all_reduce

WIDTH, K_SHARD = 7168, 384  # K3 KDA o_proj per-rank shapes at TP=32
NUM_DEVICES = 32
TILES = WIDTH // (PACK * LANES)  # 28
CALLS = 8
RS_SLOTS = NUM_DEVICES + 1  # 33rd slot: trash for the tile-major pattern


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _full_barrier(my_id, num_devices):
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(barrier_sem,
                            1,
                            device_id=(lax.rem(my_id + i, num_devices), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(sem,
                                1,
                                device_id=(lax.rem(my_id + i, num_devices), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem, num_devices - 1)


def _signal_round(sem, my_id, num_devices):
    for i in range(1, num_devices):
        pl.semaphore_signal(sem,
                            1,
                            device_id=(lax.rem(my_id + i, num_devices), ),
                            device_id_type=pl.DeviceIdType.MESH)


def _fused_kernel_variant(x_ref, w_ref, rs_ref, ag_ref, o_ref, payload_vmem,
                          recv_vmem, red_vmem, asm_vmem, local_sem, send_sem,
                          recv_sem, arr1, arr2, *, num_devices, axis_name,
                          tiles, slot, barrier_first, variant):
    """The shipped kernel with trace-time variant switches."""
    my_id = lax.axis_index(axis_name)
    pipehalf = "pipehalf" in variant
    treereduce = "treereduce" in variant
    directout = "directout" in variant
    # Phase-isolation timing variants (numerics wrong by construction):
    # dotonly = dot + payload store only; rsonly = phase 1 only.
    dotonly = "dotonly" in variant
    rsonly = "rsonly" in variant
    nosig = "nosig" in variant

    if barrier_first:
        _full_barrier(my_id, num_devices)

    if dotonly:
        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (0, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)
        o_ref[...] = payload_vmem[...]
        return

    def dot_half(w_block):
        acc = jax.lax.dot_general(x_ref[...],
                                  w_block,
                                  dimension_numbers=(((1, ), (0, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        return acc.astype(o_ref.dtype)

    def send_rem(t_lo, t_hi):
        """The shipped offset-rem pattern, restricted to dsts whose owned
        tile is in [t_lo, t_hi) -- used by the non-pipehalf path only."""
        del t_lo, t_hi
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
        return copies

    def send_tile_major(t_lo, t_hi):
        """Tile-major receive-uniform sends (probe's pipehalf pattern): tile
        t goes to ranks {t, t+28}; a sender's copy to itself is redirected to
        the trash slot. Ranks 28-31 receive duplicated tiles 0-3."""
        copies = []
        for t in range(t_lo, t_hi):
            dsts = (t, t + tiles) if t + tiles < num_devices else (t, )
            for d in dsts:
                copy = pltpu.make_async_remote_copy(
                    src_ref=payload_vmem.at[t],
                    dst_ref=rs_ref.at[slot,
                                      lax.select(d ==
                                                 my_id, num_devices, my_id)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(d, ),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
        return copies

    if pipehalf:
        half_tiles = tiles // 2
        half_cols = WIDTH // 2
        payload_vmem[0:half_tiles] = dot_half(w_ref[:, :half_cols]).reshape(
            half_tiles, PACK, LANES)
        copies = send_tile_major(0, half_tiles)
        payload_vmem[half_tiles:tiles] = dot_half(
            w_ref[:, half_cols:]).reshape(half_tiles, PACK, LANES)
        copies += send_tile_major(half_tiles, tiles)
    else:
        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (0, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)
        copies = send_rem(0, tiles)
    for copy in copies:
        copy.wait()
    # nosig: barrier-free (Lamport-style, cf. two_shot_all_reduce's
    # steady_barrier=False -- verified exact in jobs 118/119): sender-side
    # copy.wait() proves delivery; cross-call safety comes from the slot
    # rotation + barrier_first. UNSOUND under rank-divergent call patterns
    # (silent wrong numbers) -- the bench's per-call perturbed inputs are
    # the skew canary.
    if nosig and rsonly:
        o_ref[...] = payload_vmem[...]
        return
    if rsonly:
        # Phase 1 only: keep the signal round so the DMA/semaphore protocol
        # is identical to the full kernel, then write the payload out.
        _signal_round(arr1, my_id, num_devices)
        pl.semaphore_wait(arr1, num_devices - 1)
        o_ref[...] = payload_vmem[...]
        return
    if not nosig:
        _signal_round(arr1, my_id, num_devices)
        pl.semaphore_wait(arr1, num_devices - 1)

    @pl.when(my_id < tiles)
    def _():
        stage = pltpu.make_async_copy(src_ref=rs_ref.at[slot, 0:num_devices],
                                      dst_ref=recv_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        red = payload_vmem[my_id]
        if treereduce:
            # Pairwise tree: 32 serial VPU adds -> 5 dependent levels.
            accs = [red] + [recv_vmem[j] for j in range(num_devices)]
            while len(accs) > 1:
                nxt = []
                for j in range(0, len(accs) - 1, 2):
                    nxt.append(accs[j] + accs[j + 1])
                if len(accs) % 2:
                    nxt.append(accs[-1])
                accs = nxt
            red = accs[0]
        else:
            for j in range(num_devices):
                red = red + recv_vmem[j]
        red_vmem[...] = red

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
    if not nosig:
        _signal_round(arr2, my_id, num_devices)
        pl.semaphore_wait(arr2, num_devices - 1)

    if directout:
        # Stage straight into the output ref: skips the asm_vmem VMEM->VMEM
        # hop.
        stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot, 0:tiles],
                                      dst_ref=o_ref,
                                      sem=local_sem)
        stage.start()
        stage.wait()
    else:
        stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot, 0:tiles],
                                      dst_ref=asm_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        o_ref[...] = asm_vmem[...]


def _oneshot_kernel(x_ref, w_ref, ag_ref, o_ref, payload_vmem, recv_vmem,
                    local_sem, send_sem, recv_sem, arr1, *, num_devices,
                    axis_name, slot, barrier_first):
    """One-shot all-reduce: every rank pushes its WHOLE [1, WIDTH] partial
    into slot my_id of every rank's HBM gather buffer (31 remote + 1 local
    copy of 14 KB), one signal round, then stage and reduce all 32
    partials locally. 27x the two-shot's egress but ONE exchange phase --
    the two-shot's phase-1 protocol overhead is ~13 us at this shape, so
    the trade is measured, not assumed.
    """
    my_id = lax.axis_index(axis_name)

    if barrier_first:
        _full_barrier(my_id, num_devices)

    acc = jax.lax.dot_general(x_ref[...],
                              w_ref[...],
                              dimension_numbers=(((1, ), (0, )), ((), ())),
                              preferred_element_type=jnp.float32)
    payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

    local = pltpu.make_async_copy(src_ref=payload_vmem,
                                  dst_ref=ag_ref.at[slot, my_id],
                                  sem=local_sem)
    local.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=payload_vmem,
            dst_ref=ag_ref.at[slot, my_id],
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
    _signal_round(arr1, my_id, num_devices)
    pl.semaphore_wait(arr1, num_devices - 1)

    stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    stage.start()
    stage.wait()
    # recv_vmem is [num_devices, PACK*LANES] flat rows of the partials.
    red = recv_vmem[0, 0]
    for j in range(1, num_devices):
        red = red + recv_vmem[j, 0]
    o_ref[...] = red.reshape(o_ref.shape)


def fused_oneshot(x, w, ag, *, slot, barrier_first, axis_name, num_devices):
    (out, ) = pl.pallas_call(
        functools.partial(_oneshot_kernel,
                          num_devices=num_devices,
                          axis_name=axis_name,
                          slot=slot,
                          barrier_first=barrier_first),
        out_shape=(jax.ShapeDtypeStruct((1, WIDTH), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((1, WIDTH), x.dtype),
            pltpu.VMEM((num_devices, 1, WIDTH), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=8 if barrier_first else None,
            has_side_effects=False,
            disable_semaphore_checks=True),
    )(x, w, ag)
    return out


def fused_variant(x, w, rs, ag, *, slot, barrier_first, axis_name, num_devices,
                  variant):
    (out, ) = pl.pallas_call(
        functools.partial(_fused_kernel_variant,
                          num_devices=num_devices,
                          axis_name=axis_name,
                          tiles=TILES,
                          slot=slot,
                          barrier_first=barrier_first,
                          variant=variant),
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.VMEM((PACK, LANES), x.dtype),
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
        ],
        compiler_params=pltpu.CompilerParams(
            # The barrier semaphore needs a collective id; a kernel without
            # one must not be given one (probe job 489).
            collective_id=8 if barrier_first else None,
            has_side_effects=False,
            disable_semaphore_checks=True),
    )(x, w, rs, ag)
    return out


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True

    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    chosen = os.environ.get(
        "BENCH_VARIANTS",
        "baseline,directout,treereduce,pipehalf,combo").split(",")
    timing_only = {"dotonly", "rsonly"}

    def make_fn(variant):
        if variant == "combo":
            variant = "pipehalf,treereduce,directout"

        def fn(x, w, rs, ag):

            def body(xl, wl, rs, ag):
                outs = []
                for k in range(CALLS):
                    xk = xl + jnp.asarray(k, xl.dtype)
                    if variant == "baseline":
                        out = fused_out_proj_all_reduce(
                            xk,
                            wl,
                            rs,
                            ag,
                            slot=k % 2,
                            barrier_first=1 if k == 0 else 0,
                            axis_name="tp",
                            num_devices=NUM_DEVICES,
                            has_side_effects=False)
                    elif variant == "oneshot":
                        out = fused_oneshot(xk,
                                            wl,
                                            ag,
                                            slot=k % 2,
                                            barrier_first=1 if k == 0 else 0,
                                            axis_name="tp",
                                            num_devices=NUM_DEVICES)
                    elif variant == "xla":
                        out = lax.psum(
                            jax.lax.dot_general(
                                xk,
                                wl,
                                dimension_numbers=(((1, ), (0, )), ((), ())),
                                preferred_element_type=jnp.float32).astype(
                                    xl.dtype), "tp")
                    else:
                        out = fused_variant(xk,
                                            wl,
                                            rs,
                                            ag,
                                            slot=k % 2,
                                            barrier_first=1 if k == 0 else 0,
                                            axis_name="tp",
                                            num_devices=NUM_DEVICES,
                                            variant=variant)
                    outs.append(out.reshape(1, WIDTH))
                return outs

            return jax.shard_map(body,
                                 mesh=mesh,
                                 in_specs=(P(None, "tp"), P("tp",
                                                            None), P(), P()),
                                 out_specs=[P()] * CALLS,
                                 check_vma=False)(x, w, rs, ag)

        return fn

    # x: [1, 12288] global, sharded along the contracting dim -> [1, 384]
    # per rank. w: [12288, 7168] global (contracting-dim first, the model's
    # layout), sharded along dim 0 -> [384, 7168] per rank.
    x = jax.jit(
        lambda: jax.random.normal(jax.random.key(3),
                                  (1, K_SHARD * NUM_DEVICES), jnp.bfloat16),
        out_shardings=NamedSharding(mesh, P(None, "tp")))()
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1), (
        K_SHARD * NUM_DEVICES, WIDTH), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("tp", None)))()

    def make_ws(rs_slots):
        return jax.jit(lambda:
                       (jnp.zeros((2, rs_slots, PACK, LANES), jnp.bfloat16),
                        jnp.zeros((2, TILES + 1, PACK, LANES), jnp.bfloat16)),
                       out_shardings=(NamedSharding(mesh, P()),
                                      NamedSharding(mesh, P())))()

    # The shipped kernel stages rs[slot] whole; give it exactly 32 slots.
    ws = {
        "shipped": make_ws(NUM_DEVICES),
        "variant": make_ws(RS_SLOTS),
    }
    rs_ship, ag_ship = ws["shipped"]
    rs_var, ag_var = ws["variant"]
    # One-shot's gather buffer: 2 slots x 32 partials x [1, WIDTH].
    ag_os = jax.jit(lambda: jnp.zeros(
        (2, NUM_DEVICES, 1, WIDTH), jnp.bfloat16),
                    out_shardings=NamedSharding(mesh, P()))()

    wants = [
        jax.jit(lambda x, w, k=k: jnp.dot(
            (x + jnp.asarray(k, x.dtype)).astype(jnp.float32),
            w.astype(jnp.float32)),
                out_shardings=NamedSharding(mesh, P()))(x, w)
        for k in range(CALLS)
    ]

    def check_body(out, want):
        err = jnp.max(jnp.abs(out.astype(jnp.float32) - want))
        scale = jnp.max(jnp.abs(want))
        return lax.pmax(err / jnp.maximum(scale, 1e-6), "tp")

    checker = jax.jit(
        jax.shard_map(check_body,
                      mesh=mesh,
                      in_specs=(P(), P()),
                      out_specs=P(),
                      check_vma=False))

    prof_root = os.environ.get("BENCH_PROFILE")
    for name in chosen:
        if name == "baseline":
            rs, ag = rs_ship, ag_ship
        elif name in ("oneshot", "xla"):
            rs, ag = rs_ship, ag_os
        else:
            rs, ag = rs_var, ag_var
        fn = jax.jit(make_fn(name))
        t0 = time.time()
        compiled = fn.lower(x, w, rs, ag).compile()
        _stamp(rank, f"{name}: lowered+compiled in {time.time() - t0:.1f}s")
        outs = compiled(x, w, rs, ag)
        jax.block_until_ready(outs)
        if name in timing_only:
            _stamp(rank, f"{name}: (timing only, numerics n/a)")
        else:
            worst = 0.0
            for k in range(CALLS):
                rel = float(checker(outs[k], wants[k]))
                worst = max(worst, rel)
            _stamp(rank, f"{name}: max rel err = {worst:.3g}")
            ok = ok and worst < 0.05

        for _ in range(5):
            outs = compiled(x, w, rs, ag)
        jax.block_until_ready(outs)
        t0 = time.time()
        for _ in range(50):
            outs = compiled(x, w, rs, ag)
        jax.block_until_ready(outs)
        _stamp(
            rank, f"{name}: {(time.time() - t0) / 50 / CALLS * 1e6:.2f} "
            "us/call (8-call chain, host dispatch amortized)")

        if prof_root:
            from jax import profiler
            tdir = os.path.join(prof_root, f"{name}/rank{rank}")
            profiler.start_trace(tdir)
            for _ in range(30):
                outs = compiled(x, w, rs, ag)
            jax.block_until_ready(outs)
            profiler.stop_trace()
            _stamp(rank, f"{name}: trace captured -> {tdir}")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
