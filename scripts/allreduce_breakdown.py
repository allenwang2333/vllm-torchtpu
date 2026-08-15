#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where does the one-shot all-reduce's latency floor actually go?

The one-shot kernel bottoms out around 6.4 us on 8 devices for a 2 KiB payload,
which is far above any plausible wire time. This strips the kernel down one
layer at a time so the floor can be attributed instead of guessed:

    copy      Pallas launch + VMEM copy. No comms at all -- the launch floor.
    bar1      one all-to-all barrier, then copy.
    bar2      the shipped double barrier, then copy.
    dma1bar   one barrier + DMA + reduce  (== one_shot with double_barrier=False)
    full      the shipped kernel: double barrier + DMA + reduce.

Differences give the decomposition:
    launch      = copy
    barrier #1  = bar1 - copy
    barrier #2  = bar2 - bar1
    DMA+reduce  = dma1bar - bar1
and ``full`` vs ``bar2 + (dma1bar - bar1)`` says whether they serialize.

There is deliberately no barrier-free DMA variant. Removing *every* barrier is
not merely racy in its values -- a peer's remote DMA can land after the
receiving kernel has already exited, which corrupts semaphore state and takes
down the whole slice with SLICE_FAILURE_SW_INJECT_ERROR. One barrier is the
floor. ``dma1bar`` is still unsafe to ship (a fast peer can overwrite a buffer
the previous iteration is reducing) but stays within one invocation, so it
times cleanly.

Usage:
    python scripts/allreduce_breakdown.py --devices 8 --rows 1,4,16
"""

import argparse
import functools
import os
import time

VARIANTS = ["copy", "bar1", "bar2", "dma1bar", "full"]


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--rows", default="1,4,16")
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--sc-offload-bytes", type=int, default=67108864)
    return p.parse_args()


ARGS = _parse_args()

if ARGS.sc_offload_bytes is not None:
    _extra = " ".join(f"--xla_tpu_sparse_core_{n}_offload_min_size_in_bytes="
                      f"{ARGS.sc_offload_bytes}"
                      for n in ("all_reduce", "all_gather"))
    os.environ["LIBTPU_INIT_ARGS"] = (
        _extra + " " + os.environ.get("LIBTPU_INIT_ARGS", "")).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

AXIS = "x"


def _barrier(my_id, n, double):
    sem = pltpu.get_barrier_semaphore()
    for i in range(1, n):
        pl.semaphore_signal(sem,
                            1,
                            device_id=(lax.rem(my_id + i, n), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, n - 1)
    if not double:
        return

    @functools.partial(pl.run_scoped, sem2=pltpu.SemaphoreType.REGULAR)
    def _(sem2):
        for i in range(1, n):
            pl.semaphore_signal(sem2,
                                1,
                                device_id=(lax.rem(my_id + i, n), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem2, n - 1)


def _kernel(x_ref, o_ref, recv_ref, send_sem, recv_sem, *, n, variant):
    my_id = lax.axis_index(AXIS)

    if variant in ("bar1", "dma1bar"):
        _barrier(my_id, n, double=False)
    elif variant in ("bar2", "full"):
        _barrier(my_id, n, double=True)

    if variant in ("dma1bar", "full"):
        copies = []
        for i in range(1, n):
            c = pltpu.make_async_remote_copy(
                src_ref=x_ref,
                dst_ref=recv_ref.at[i - 1],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(lax.rem(my_id + i, n), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            c.start()
            copies.append(c)
        for c in copies:
            c.wait()
        acc = x_ref[...]
        for i in range(n - 1):
            acc = acc + recv_ref[i]
        o_ref[...] = acc
    else:
        o_ref[...] = x_ref[...]


def make_op(variant: str, n: int):

    def call(x):
        return pl.pallas_call(
            functools.partial(_kernel, n=n, variant=variant),
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((n - 1, *x.shape), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                has_side_effects=True,
                # copy/dma variants never touch the barrier semaphore, but we
                # keep collective_id set so every variant shares one launch
                # configuration and the differences stay attributable.
                allow_collective_id_without_custom_barrier=True,
            ),
        )(x)

    return call


def make_chain(variant: str, mesh: Mesh, n: int, n_ops: int):
    op = make_op(variant, n)
    scale = 1.0 / n

    def body(local_x):
        v = local_x[0]
        for _ in range(n_ops):
            v = op(v) * scale
        return v[None]

    spec = P(AXIS, None, None)
    return jax.jit(
        shard_map(body,
                  mesh=mesh,
                  in_specs=spec,
                  out_specs=spec,
                  check_rep=False))


def time_fn(fn, x, repeats, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(x))
    s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        s.append(time.perf_counter() - t0)
    return float(np.median(s))


def main():
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[ARGS.dtype]
    itemsize = np.dtype(ARGS.dtype).itemsize
    n = ARGS.devices
    variants = [v.strip() for v in ARGS.variants.split(",") if v.strip()]

    mesh = Mesh(
        mesh_utils.create_device_mesh((n, ), devices=jax.devices()[:n]),
        (AXIS, ))
    sharding = NamedSharding(mesh, P(AXIS, None, None))

    print(
        f"jax {jax.__version__} | {n} devices | {jax.devices()[0].device_kind}"
    )
    print(
        f"dtype={ARGS.dtype} cols={ARGS.cols} "
        f"slope {ARGS.iters_lo}->{ARGS.iters_hi}, median of {ARGS.repeats}\n")
    print(f"{'rows':>6} {'KiB':>8} " + "".join(f"{v:>10}" for v in variants))

    lat_by_rows = {}
    for rows in (int(r) for r in ARGS.rows.split(",")):
        x = jax.jit(
            lambda: jax.random.normal(  # noqa: B023
                jax.random.key(0), (n, rows, ARGS.cols), dtype),
            out_shardings=sharding)()
        lat = {}
        for v in variants:
            try:
                lo = make_chain(v, mesh, n, ARGS.iters_lo)
                hi = make_chain(v, mesh, n, ARGS.iters_hi)
                t_lo = time_fn(lo, x, ARGS.repeats, ARGS.warmup)
                t_hi = time_fn(hi, x, ARGS.repeats, ARGS.warmup)
                d = (t_hi - t_lo) / (ARGS.iters_hi - ARGS.iters_lo)
                lat[v] = d if d > 0 else None
            except Exception as e:  # noqa: BLE001
                print(f"  !! {v} rows={rows}: {type(e).__name__}: {e}")
                lat[v] = None
        lat_by_rows[rows] = lat
        cells = "".join(
            f"{lat[v] * 1e6:>10.2f}" if lat.get(v) else f"{'-':>10}"
            for v in variants)
        print(f"{rows:>6} {rows * ARGS.cols * itemsize / 1024:>8.2f} {cells}")

    print("\nattribution (us):")
    for rows, lat in lat_by_rows.items():

        def d(a, b):
            return (lat.get(a) or 0) - (lat.get(b) or 0)

        parts = []
        if lat.get("copy"):
            parts.append(f"launch={lat['copy'] * 1e6:.2f}")
        if lat.get("bar1") and lat.get("copy"):
            parts.append(f"barrier1={d('bar1', 'copy') * 1e6:.2f}")
        if lat.get("bar2") and lat.get("bar1"):
            parts.append(f"barrier2={d('bar2', 'bar1') * 1e6:.2f}")
        if lat.get("dma1bar") and lat.get("bar1"):
            parts.append(f"dma+reduce={d('dma1bar', 'bar1') * 1e6:.2f}")
        if lat.get("full"):
            parts.append(f"full={lat['full'] * 1e6:.2f}")
        print(f"  rows={rows:<5} " + "  ".join(parts))


if __name__ == "__main__":
    main()
