#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness stress test for the persistent-workspace one-shot all-reduce.

This kernel issues its barrier *after* the sends, so a peer can be writing our
workspace while we are still reducing the previous round. That is safe only if
the double-buffering of both the data slots and the DMA semaphores is exactly
right, and the failure mode is timing-dependent: earlier variants passed short
smoke tests and only died under a longer sweep.

So: long chains, many shapes, many repeats, values checked every time against
``jax.lax.psum``. Run this to a clean exit before trusting any latency number.

Values are checked with a non-idempotent chain. The obvious chain
``x -> psum(x)/N`` computes a mean, and the mean of a mean is the same mean, so
it would still look correct if entire rounds were skipped or duplicated. Adding
a per-round twist makes the result depend on the number of rounds *and* their
order, so a dropped, doubled, or stale round shows up as a value mismatch.

Usage:
    python scripts/allreduce_ws_stress.py                     # default sweep
    python scripts/allreduce_ws_stress.py --rounds 200 --repeats 50
"""

import argparse
import importlib.util
import os
import sys


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--rows", default="1,2,4,8,16,32,64,128")
    p.add_argument("--cols", type=int, default=1024)
    # float32 by default: the kernel sums N-1 peer contributions in a
    # different order than psum, so in bfloat16 the two legitimately diverge,
    # and that divergence grows with rounds and device count (0.023 at 8
    # devices, 0.054 at 32, for an 8-round chain). A fixed tolerance therefore
    # reports drift as corruption. float32 makes the oracle exact -- a real
    # race shows up as a gross mismatch, not a few percent.
    p.add_argument("--dtype",
                   default="float32",
                   choices=["bfloat16", "float32"])
    p.add_argument("--rounds",
                   default="1,2,3,8,64,200",
                   help="Chain lengths to test. Include odd and even: the "
                   "parity of the final round differs.")
    p.add_argument("--repeats",
                   type=int,
                   default=20,
                   help="Executions per (shape, chain length). Each execution "
                   "restarts the counter, exercising the round-0 path.")
    p.add_argument("--tol",
                   type=float,
                   default=None,
                   help="Default: 1e-4 for float32; for bfloat16, scaled as "
                   "0.01*sqrt(rounds*devices) to track accumulation "
                   "drift rather than flag it as corruption.")
    # default None, not False: False *forces* the overlap and defeats the
    # kernel's auto-fallback, which is how a "safe" config still crashed.
    p.add_argument("--barrier-first",
                   action="store_true",
                   default=None,
                   help="Diagnostic: barrier before the sends every round, "
                   "giving up the overlap. Isolates the workspace "
                   "machinery from the overlap itself.")
    # Default None so the kernel's own default (checks disabled, required for
    # the overlap) applies. A store_true default of False would silently
    # re-enable the check and take down the slice -- which it did.
    p.add_argument("--kernel",
                   default="oneshot",
                   choices=["oneshot", "twoshot"],
                   help="Which workspace kernel to stress.")
    p.add_argument("--enable-sem-checks",
                   action="store_true",
                   default=None,
                   help="Re-enable Mosaic's exit-time semaphore zero check. "
                   "Expected to kill the slice with the overlap on; kept "
                   "to reproduce the bug.")
    p.add_argument("--force-overlap",
                   action="store_true",
                   help="Force barrier_first=False, bypassing auto-fallback.")
    p.add_argument("--multihost",
                   action="store_true",
                   help="Join a multi-process slice. Auto-enabled under a "
                   "multi-task SLURM step.")
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
from jax.experimental import mesh_utils  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# Must precede any device use. Without this every task grabs the local devices
# independently and the slice deadlocks.
if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

IS_LEAD = jax.process_index() == 0
AXIS = "x"


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw)


def _load_kernels():
    path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
            "/src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("low_latency_all_reduce",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


llar = _load_kernels()


def make_chain(use_ws: bool, mesh: Mesh, n: int, rounds: int, shape, dtype):
    """Chain of `rounds` all-reduces with a per-round twist.

    The twist (a round-dependent scale) makes the result depend on how many
    rounds ran and in what order, so a skipped or duplicated round is visible
    in the values -- unlike a plain mean-of-means chain.
    """

    def body(local_x):
        v = local_x[0]
        if use_ws and ARGS.kernel == "twoshot":
            rs, ag, step = llar.make_two_shot_workspace(shape, dtype, n)
            for j in range(rounds):
                v, rs, ag, step = llar.two_shot_all_reduce_ws(
                    v,
                    rs,
                    ag,
                    step,
                    axis_name=AXIS,
                    num_devices=n,
                    barrier_first=bool(ARGS.barrier_first),
                    disable_semaphore_checks=(False if ARGS.enable_sem_checks
                                              else True))
                v = v * (1.0 / n) + jnp.asarray(j % 3, dtype) * 0.25
        elif use_ws:
            ws, step = llar.make_workspace(shape, dtype, n)
            for j in range(rounds):
                v, ws, step = llar.one_shot_all_reduce_ws(
                    v,
                    ws,
                    step,
                    axis_name=AXIS,
                    num_devices=n,
                    barrier_first=(False if ARGS.force_overlap else
                                   ARGS.barrier_first),
                    disable_semaphore_checks=(False if ARGS.enable_sem_checks
                                              else True))
                v = v * (1.0 / n) + jnp.asarray(j % 3, dtype) * 0.25
        else:
            for j in range(rounds):
                v = jax.lax.psum(v, axis_name=AXIS)
                v = v * (1.0 / n) + jnp.asarray(j % 3, dtype) * 0.25
        return v[None]

    spec = P(AXIS, None, None)
    return jax.jit(
        shard_map(body,
                  mesh=mesh,
                  in_specs=spec,
                  out_specs=spec,
                  check_rep=False))


def main():
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[ARGS.dtype]
    n = ARGS.devices
    mesh = Mesh(
        mesh_utils.create_device_mesh((n, ), devices=jax.devices()[:n]),
        (AXIS, ))
    sharding = NamedSharding(mesh, P(AXIS, None, None))

    rprint(
        f"jax {jax.__version__} | {n} devices | {jax.devices()[0].device_kind}"
    )
    rprint(f"dtype={ARGS.dtype} cols={ARGS.cols} repeats={ARGS.repeats}\n")

    failures = 0
    checks = 0
    for rows in (int(r) for r in ARGS.rows.split(",")):
        shape = (rows, ARGS.cols)
        x = jax.jit(
            lambda: jax.random.normal(  # noqa: B023
                jax.random.key(0), (n, rows, ARGS.cols), dtype),
            out_shardings=sharding)()
        for rounds in (int(r) for r in ARGS.rounds.split(",")):
            got_fn = make_chain(True, mesh, n, rounds, shape, dtype)
            want = make_chain(False, mesh, n, rounds, shape, dtype)(x)

            def rel(a, b):
                a = a.astype(jnp.float32)
                b = b.astype(jnp.float32)
                return (jnp.max(jnp.abs(a - b)) /
                        jnp.maximum(jnp.max(jnp.abs(b)), 1e-6))

            rel_fn = jax.jit(rel, out_shardings=NamedSharding(mesh, P()))
            worst = 0.0
            for _ in range(ARGS.repeats):
                worst = max(worst, float(rel_fn(got_fn(x), want)))
                checks += 1
            if ARGS.tol is not None:
                tol = ARGS.tol
            elif ARGS.dtype == "float32":
                tol = 1e-4
            else:
                tol = 0.01 * (rounds * n)**0.5
            ok = worst <= tol
            failures += 0 if ok else 1
            rprint(f"  rows={rows:<5} rounds={rounds:<5} "
                   f"max_rel_err={worst:.3g} tol={tol:.3g}  "
                   f"{'ok' if ok else 'FAIL'}")

    rprint(f"\n{checks} executions checked, {failures} failing configs")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
