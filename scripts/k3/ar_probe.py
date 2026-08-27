"""Isolate the collective: one-shot vs two-shot all-reduce at the mega kernel's
exact tail payload ([T, 3584] bf16 per device), nothing else in the kernel.

Why: the fused full-layer tail costs ~17 us more than the unfused tail in
lockstep, and up-ring depth sweeps plateau at depth ~3, so the deficit is the
AR implementation, not the up_proj stream. This probe measures the two AR
protocols head to head.

    srun env PYTHONPATH=src .venv/bin/python scripts/k3/ar_probe.py
"""
import functools
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import (
    make_two_shot_workspace, two_shot_all_reduce_ws)

T, L = 8, 3584


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


def _oneshot_kernel(x_ref, redws_in, step_in, o_ref, redws_out, step_out,
                    stage_vmem, recv_vmem, sem_local, s_send, s_recv, *,
                    num_devices, axis_name):
    del redws_in, step_in
    my_id = lax.axis_index(axis_name)
    step = step_out[0]
    slot = lax.rem(step, 2)
    stage_vmem[...] = x_ref[...]
    st = pltpu.make_async_copy(stage_vmem, o_ref, sem_local)
    st.start()
    st.wait()

    @pl.when(step == 0)
    def _():
        _full_barrier(my_id, num_devices)

    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        c = pltpu.make_async_remote_copy(src_ref=o_ref,
                                         dst_ref=redws_out.at[slot, my_id],
                                         send_sem=s_send.at[slot],
                                         recv_sem=s_recv.at[slot],
                                         device_id=(peer, ),
                                         device_id_type=pl.DeviceIdType.MESH)
        c.start()
        copies.append(c)
    local = pltpu.make_async_copy(o_ref, redws_out.at[slot, my_id], sem_local)
    local.start()

    @pl.when(step != 0)
    def _():
        _full_barrier(my_id, num_devices)

    for c in copies:
        c.wait()
    local.wait()

    pull = pltpu.make_async_copy(redws_out.at[slot], recv_vmem, sem_local)
    pull.start()
    pull.wait()
    acc = recv_vmem[0].astype(jnp.float32)
    for i in range(1, num_devices):
        acc = acc + recv_vmem[i].astype(jnp.float32)
    # write the reduced value back over the output (a DMA: o_ref is HBM) so
    # the chain stays honest; safe to overwrite after my sends have completed
    stage_vmem[...] = acc.astype(stage_vmem.dtype)
    fin = pltpu.make_async_copy(stage_vmem, o_ref, sem_local)
    fin.start()
    fin.wait()
    step_out[0] = step + 1


def oneshot_ar(x, redws, step, *, axis_name, num_devices):
    nd = num_devices
    ANY = pl.BlockSpec(memory_space=pl.ANY)
    o, rw, st = pl.pallas_call(
        functools.partial(_oneshot_kernel, num_devices=nd,
                          axis_name=axis_name),
        grid=(1, ),
        out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype),
                   jax.ShapeDtypeStruct(redws.shape, redws.dtype),
                   jax.ShapeDtypeStruct(step.shape, step.dtype)),
        in_specs=[
            pl.BlockSpec(x.shape, lambda j: (0, 0)), ANY,
            pl.BlockSpec(memory_space=pltpu.SMEM)
        ],
        out_specs=[ANY, ANY, pl.BlockSpec(memory_space=pltpu.SMEM)],
        scratch_shapes=[
            pltpu.VMEM((T, L), x.dtype),
            pltpu.VMEM((nd, T, L), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        input_output_aliases={
            1: 1,
            2: 2
        },
        compiler_params=pltpu.CompilerParams(has_side_effects=False,
                                             collective_id=21),
    )(x, redws, step)
    return o, rw, st


def main():
    nd = len(jax.devices())
    AXIS = "dev"
    mesh = Mesh(jax.devices(), (AXIS, ))

    def rep(t):
        return jax.device_put(t, NamedSharding(mesh, P()))

    key = jax.random.key(0)
    x = rep(jax.random.normal(key, (T, L), jnp.bfloat16))
    redws0 = rep(jnp.zeros((2, nd, T, L), jnp.bfloat16))
    step0 = rep(jnp.zeros((1, ), jnp.int32))
    ks_rows, ks_cols = T * nd, L // nd
    ks_ws = tuple(
        rep(t)
        for t in make_two_shot_workspace((ks_rows, ks_cols), jnp.bfloat16, nd))

    def tie(v, prev):
        return (v if prev is None else v +
                (prev.reshape(-1)[0] * 1e-30).astype(v.dtype))

    def chain_oneshot(x, redws, step, depth):
        acc = None
        for _ in range(depth):
            v, redws, step = oneshot_ar(tie(x, acc),
                                        redws,
                                        step,
                                        axis_name=AXIS,
                                        num_devices=nd)
            acc = v if acc is None else acc + v
        return acc, redws, step

    def chain_twoshot(x, rs, ag, st, depth):
        acc = None
        for _ in range(depth):
            red, rs, ag, st = two_shot_all_reduce_ws(tie(x, acc).reshape(
                ks_rows, ks_cols),
                                                     rs,
                                                     ag,
                                                     st,
                                                     axis_name=AXIS,
                                                     num_devices=nd,
                                                     collective_id=3,
                                                     has_side_effects=False)
            v = red.reshape(T, L)
            acc = v if acc is None else acc + v
        return acc, rs, ag, st

    def slope(make, lo, hi, reps=20):

        def once(depth):
            fn, a = make(depth)
            jax.block_until_ready(fn(*a))
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*a))
                best = min(best, time.perf_counter() - t0)
            return best

        return (once(hi) - once(lo)) / (hi - lo)

    pspec = (P(), )

    def mk_oneshot(depth):

        def fn(xx, rw, st):
            return chain_oneshot(xx, rw, st, depth)

        jf = jax.jit(
            jax.shard_map(fn,
                          mesh=mesh,
                          in_specs=pspec * 3,
                          out_specs=(P(), P(), P()),
                          check_vma=False))
        return jf, (x, redws0, step0)

    def mk_twoshot(depth):

        def fn(xx, rs, ag, st):
            return chain_twoshot(xx, rs, ag, st, depth)

        jf = jax.jit(
            jax.shard_map(fn,
                          mesh=mesh,
                          in_specs=pspec * 4,
                          out_specs=(P(), ) * 4,
                          check_vma=False))
        return jf, (x.reshape(ks_rows, ks_cols), ) + ks_ws

    print(f"device {jax.devices()[0].device_kind}  nd={nd}  "
          f"payload {T*L*2/1024:.0f} KiB/device")
    for label, mk in (("one-shot", mk_oneshot), ("two-shot", mk_twoshot)):
        # 5 trials of the slope, report median
        vals = sorted(slope(mk, 4, 16) for _ in range(5))
        med = vals[len(vals) // 2]
        print(f"{label:10} {med*1e6:8.2f} us/op  "
              f"(spread +-{(vals[-1]-vals[0])/2*1e6:.2f})")


if __name__ == "__main__":
    main()
