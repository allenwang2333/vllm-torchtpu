#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Can one kernel reduce K3's two MoE tensors without concatenating them?

``probe_variadic_dma.py`` concluded the design was unreachable: both of its
split variants died on ``MosaicError: Slice sizes along tiled dimensions must be
aligned to tiling``, *including* the 128-aligned one, which looked like proof
that no dynamic per-peer offset can lower at all.

That conclusion was an artifact of its layout. It sourced the operands from
**1-D** arrays (``jnp.zeros((n * ca,))``) and sliced them with ``pl.ds()``. A 1-D
bf16 array is tiled 16x128 = 2048 elements, so neither 112 nor 128 is aligned --
the "aligned" arm was never aligned in the dimension that mattered.

Production's two-shot does something different and demonstrably works: it holds a
**2-D** ``(n, cols)`` ref and uses ``x_ref.at[peer]`` -- a dynamic *index*, which
squeezes the dimension away, rather than a ``pl.ds()`` slice of it. That is the
pattern this probe applies to two operands at once.

Why it matters: at the MoE site production currently pays
``torch.cat`` -> ``(1, 10752)`` -> ``reshape(32, 336)`` -> kernel ->
``reshape(1, 10752)`` -> two slices. If one kernel can take ``(32, 112)`` and
``(32, 224)`` directly, the concat and both reshapes disappear and each tensor
keeps the layout its producer already emits.

Arms
----
``joined_floor``
    One two-shot over a ``(32, 336)`` input that is *already* in kernel layout.
    Not achievable in production -- it is the floor the others are measured
    against.
``joined_cat``
    ``jnp.concatenate`` the two operands to ``(32, 336)``, one two-shot, split
    back. Prices the join alone, with no reshape.
``joined_prod``
    The production path: concatenate to ``(1, 10752)``, reshape to ``(32, 336)``
    inside the jit, reduce, reshape back, slice. Prices join + reshape.
``variadic``
    Two operands, one kernel, one barrier, two reduce-scatters and two
    all-gathers. No join, no reshape.

All arms are checked against ``psum`` on both operands independently, so an arm
cannot pass by reducing only one of them or by reducing a subset.

Run (32 devices, 4 hosts):
    sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_variadic_2d.sbatch
"""

import argparse
import functools
import os
import sys
import time

ARMS = ("prod", "two_calls", "variadic")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--widths",
                   default="3584x7168",
                   help="Comma-separated AxB total element counts. Default is "
                   "K3's latent 3584 + hidden 7168 = 10752.")
    p.add_argument("--n-lo", type=int, default=200)
    p.add_argument("--n-hi", type=int, default=800)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--multihost", action="store_true")
    return p.parse_args(argv)


ARGS = _parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

IS_LEAD = jax.process_index() == 0
AXIS = "x"


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw)
        sys.stdout.flush()


def _load_kernels():
    import importlib.util
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "src", "vllm_torchtpu", "kernels", "collectives",
                        "low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("_llar", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


llar = _load_kernels()


def _barrier(my_id, n):
    sem = pltpu.get_barrier_semaphore()
    for i in range(1, n):
        pl.semaphore_signal(sem,
                            1,
                            device_id=(lax.rem(my_id + i, n), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, n - 1)

    @functools.partial(pl.run_scoped, s2=pltpu.SemaphoreType.REGULAR)
    def _(s2):
        for i in range(1, n):
            pl.semaphore_signal(s2,
                                1,
                                device_id=(lax.rem(my_id + i, n), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(s2, n - 1)


def _rc(src, dst, ss, rs, peer):
    return pltpu.make_async_remote_copy(src_ref=src,
                                        dst_ref=dst,
                                        send_sem=ss,
                                        recv_sem=rs,
                                        device_id=(peer, ),
                                        device_id_type=pl.DeviceIdType.MESH)


def _variadic_kernel(a_ref, b_ref, oa_ref, ob_ref, rsa, rsb, aga, agb, s1a_s,
                     s1a_r, s1b_s, s1b_r, s2a_s, s2a_r, s2b_s, s2b_r, *,
                     n: int):
    """Two operands, one launch, one barrier: two RS phases and two AG phases.

    Every ref is **3-D** ``(n, chunk, cols)`` and every per-peer access indexes
    the leading dimension. That is load-bearing, and it is what the first attempt
    at this got wrong. A 2-D VMEM ref has both dims tiled -- bf16 tiles as
    ``(8,128)`` with ``(2,1)`` packing -- so taking one row out of ``(7, 448)``
    fails with "size at dimension 0 is divisible by the tile dimension 8", even
    for a *static* index. Production sidesteps this by reshaping to
    ``(n, chunk, cols)`` before the pallas_call (see two_shot_all_reduce_ws's
    final line), which puts the peer index on an untiled dimension.

    So the reshape is not removable and never was about ``rows % n == 0``. The
    concat is the removable part, which is what this kernel tests.
    """
    my_id = lax.axis_index(AXIS)
    _barrier(my_id, n)

    cps = []
    for i in range(1, n):
        peer = lax.rem(my_id + i, n)
        cps.append(_rc(a_ref.at[peer], rsa.at[i - 1], s1a_s, s1a_r, peer))
        cps.append(_rc(b_ref.at[peer], rsb.at[i - 1], s1b_s, s1b_r, peer))
    for c in cps:
        c.start()
    for c in cps:
        c.wait()

    acc_a = a_ref[my_id]
    for i in range(n - 1):
        acc_a = acc_a + rsa[i]
    acc_b = b_ref[my_id]
    for i in range(n - 1):
        acc_b = acc_b + rsb[i]
    aga[my_id] = acc_a
    agb[my_id] = acc_b

    cps = []
    for i in range(1, n):
        peer = lax.rem(my_id + i, n)
        cps.append(_rc(aga.at[my_id], aga.at[my_id], s2a_s, s2a_r, peer))
        cps.append(_rc(agb.at[my_id], agb.at[my_id], s2b_s, s2b_r, peer))
    for c in cps:
        c.start()
    for c in cps:
        c.wait()

    oa_ref[...] = aga[...]
    ob_ref[...] = agb[...]


def variadic(a, b, n):
    """``a``/``b`` are ``(1, na)`` / ``(1, nb)``; reshaped in, reshaped out."""
    na, nb = a.shape[1], b.shape[1]
    ca, cb = na // n, nb // n
    a3 = a.reshape(n, 1, ca)
    b3 = b.reshape(n, 1, cb)
    oa, ob = pl.pallas_call(
        functools.partial(_variadic_kernel, n=n),
        out_shape=(jax.ShapeDtypeStruct(a3.shape, a.dtype),
                   jax.ShapeDtypeStruct(b3.shape, b.dtype)),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        scratch_shapes=[
            pltpu.VMEM((n - 1, 1, ca), a.dtype),
            pltpu.VMEM((n - 1, 1, cb), b.dtype),
            pltpu.VMEM((n, 1, ca), a.dtype),
            pltpu.VMEM((n, 1, cb), b.dtype),
        ] + [pltpu.SemaphoreType.DMA] * 8,
        compiler_params=pltpu.CompilerParams(collective_id=9,
                                             has_side_effects=True,
                                             disable_semaphore_checks=True),
    )(a3, b3)
    return oa.reshape(a.shape), ob.reshape(b.shape)


def one_step(arm, a, b, n):
    """One all-reduce of the pair, from the ``(1, w)`` layout producers emit."""

    def two_shot(v):
        rs, ag, step = llar.make_two_shot_workspace(
            (n, v.shape[1] * v.shape[0] // n), v.dtype, n)
        flat = v.reshape(n, v.size // n)
        out, _, _, _ = llar.two_shot_all_reduce_ws(flat,
                                                   rs,
                                                   ag,
                                                   step,
                                                   axis_name=AXIS,
                                                   num_devices=n,
                                                   collective_id=9)
        return out.reshape(v.shape)

    if arm == "variadic":
        return variadic(a, b, n)
    if arm == "two_calls":
        # What upstream does without the concat patch: one collective each, so
        # two kernel launches. The concat exists to avoid this.
        return two_shot(a), two_shot(b)
    # prod: concatenate to (1, na+nb), one collective, slice back.
    na = a.size
    joined = jnp.concatenate([a.reshape(-1), b.reshape(-1)])[None, :]
    out = two_shot(joined)[0]
    return out[:na].reshape(a.shape), out[na:].reshape(b.shape)


def build_mesh():
    return Mesh(np.asarray(sorted(jax.devices(), key=lambda d: d.id)),
                axis_names=(AXIS, ))


def make_chain(arm, n, mesh):
    scale = 1.0 / n

    def body(la, lb, n_steps):
        a, b = la[0], lb[0]

        def one(_, c):
            x, y = one_step(arm, c[0], c[1], n)
            return (x * scale).astype(a.dtype), (y * scale).astype(b.dtype)

        a, b = lax.fori_loop(0, n_steps[0], one, (a, b))
        return a[None], b[None]

    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None, None), P(AXIS, None,
                                                       None), P(AXIS)),
                      out_specs=(P(AXIS, None, None), P(AXIS, None, None)),
                      check_vma=False))


def make_verify(arm, n, mesh):

    def body(la, lb):
        a, b = la[0], lb[0]
        ga, gb = one_step(arm, a, b, n)
        wa = lax.psum(a, axis_name=AXIS).astype(jnp.float32)
        wb = lax.psum(b, axis_name=AXIS).astype(jnp.float32)
        ra = jnp.max(jnp.abs(ga.astype(jnp.float32) - wa)) / jnp.maximum(
            jnp.max(jnp.abs(wa)), 1e-6)
        rb = jnp.max(jnp.abs(gb.astype(jnp.float32) - wb)) / jnp.maximum(
            jnp.max(jnp.abs(wb)), 1e-6)
        return lax.pmax(jnp.maximum(ra, rb), axis_name=AXIS)

    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS, None, None), P(AXIS, None, None)),
                      out_specs=P(),
                      check_vma=False))


def time_call(fn, args, repeats, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(repeats):
        multihost_utils.sync_global_devices("probe")
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append(time.perf_counter() - t0)
    local = float(np.median(s))
    if jax.process_count() == 1:
        return local
    return float(
        np.max(
            np.asarray(multihost_utils.process_allgather(np.asarray([local
                                                                     ])))))


def main():
    dtype = jnp.bfloat16
    n = jax.device_count()
    mesh = build_mesh()
    shard = NamedSharding(mesh, P(AXIS, None, None))
    scalar = NamedSharding(mesh, P(AXIS))

    def n_arr(k):
        return jax.jit(lambda kk=int(k): jnp.full((n, ), kk, jnp.int32),
                       out_shardings=scalar)()

    rprint(f"== {n} devices, {jax.process_count()} process(es), "
           f"chain {ARGS.n_lo} vs {ARGS.n_hi}")
    for spec in ARGS.widths.split(","):
        na, nb = (int(v) for v in spec.strip().split("x"))
        if na % n or nb % n:
            rprint(f"\n== skip {spec}: both widths must divide {n}")
            continue
        ca, cb = na // n, nb // n
        kib = (na + nb) * 2 / 1024
        rprint(f"\n== a=(1,{na}) b=(1,{nb}) -> ({n},1,{ca}) + ({n},1,{cb}), "
               f"{kib:.1f} KiB total")
        rprint(f"   lane alignment: {ca} % 128 = {ca % 128}, "
               f"{cb} % 128 = {cb % 128}, joined {ca + cb} % 128 = "
               f"{(ca + cb) % 128}")
        a = jax.jit(lambda w=na: jax.random.normal(jax.random.key(0),
                                                   (n, 1, w), dtype),
                    out_shardings=shard)()
        b = jax.jit(lambda w=nb: jax.random.normal(jax.random.key(1),
                                                   (n, 1, w), dtype),
                    out_shardings=shard)()
        rprint(f"{'arm':>16} {'us':>9} {'verify':>10}   note")
        base = {}
        for arm in [x.strip() for x in ARGS.arms.split(",") if x.strip()]:
            try:
                err = "skipped"
                if not ARGS.skip_verify:
                    e = float(make_verify(arm, n, mesh)(a, b))
                    err = f"{e:.2e}" + (" BAD" if e > 0.05 else "")
                fn = make_chain(arm, n, mesh)
                t_lo = time_call(fn, (a, b, n_arr(ARGS.n_lo)), ARGS.repeats,
                                 ARGS.warmup)
                t_hi = time_call(fn, (a, b, n_arr(ARGS.n_hi)), ARGS.repeats,
                                 ARGS.warmup)
                us = (t_hi - t_lo) / (ARGS.n_hi - ARGS.n_lo) * 1e6
                base[arm] = us
                note = ""
                if arm != "prod" and "prod" in base:
                    d = base["prod"] - us
                    note = (f"{'saves' if d > 0 else 'COSTS'} "
                            f"{abs(d):.2f} vs prod")
                rprint(f"{arm:>16} {us:>9.2f} {err:>10}   {note}")
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).replace("\n", " ")[:700]
                rprint(f"{arm:>16} {'--':>9} {'--':>10}   FAILED: {msg}")
        if "prod" in base and "variadic" in base:
            rprint(f"\n   variadic vs production path: "
                   f"{base['prod'] - base['variadic']:+.2f} us "
                   f"(positive = variadic wins)")


if __name__ == "__main__":
    main()
