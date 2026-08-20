#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness stress for the variadic (two-operand) workspace all-reduce.

``probe_variadic_2d.py`` proved the *DMA pattern* reduces both operands
correctly, but it used scratch-VMEM destinations and barriered every round. The
production kernel, :func:`two_shot_all_reduce_variadic_ws`, is a different
animal: it keeps four persistent HBM workspaces plus a round counter, aliases
them call-to-call, and moves the barrier after its phase-1 sends so it overlaps
the transfer. That overlap is only safe if the double-buffering of every data
slot *and* every DMA semaphore is exactly right -- and now there are twice as
many of each, on two independent operand widths.

The failure mode is timing-dependent and shape-dependent: a peer writing slot
``k%2`` of operand b while we still read it for round ``k-2``. A single-round
check cannot see it, which is why this runs chains.

Two things make a wrong answer visible rather than plausible:

* **A per-round twist.** The obvious chain ``x -> psum(x)/N`` computes a mean,
  and a mean of a mean is the same mean, so a dropped or duplicated round would
  still look right. A round-dependent offset makes the result depend on how many
  rounds ran and in what order.
* **Asymmetric operands.** ``a`` and ``b`` are given different widths and
  different data, so a kernel that reduced one operand twice, or crossed the two
  workspaces, cannot produce matching values for both.

float32 by default: the kernel sums N-1 peer contributions in a different order
than psum, and in bfloat16 that drift grows with rounds and device count, so a
fixed tolerance would report drift as corruption.

Usage:
    python scripts/allreduce_variadic_stress.py --devices 8
    python scripts/allreduce_variadic_stress.py --multihost --devices 32 \
        --widths 3584x7168 --rounds 1,2,3,8,64,200
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
    p.add_argument(
        "--widths",
        default="3584x7168,7168x3584",
        help="Comma-separated AxB element counts. The reversed pair "
        "is deliberate: it catches a kernel that assumes the first "
        "operand is the narrower one. Equal widths are absent "
        "because the kernel now refuses them -- it corrupts both "
        "operands there, unexplained; see its ValueError and "
        "scripts/debug/probe_variadic_bug.py.")
    p.add_argument("--rows",
                   default="1",
                   help="Token buckets. K3 decodes at 1.")
    p.add_argument("--dtype",
                   default="float32",
                   choices=["bfloat16", "float32"])
    p.add_argument("--rounds", default="1,2,3,8,64,200")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--no-cse-guard",
                   action="store_true",
                   help="Drop the optimization_barrier that keeps the two "
                   "operands' workspaces from being CSE'd together. Use to "
                   "reproduce the equal-width failure.")
    p.add_argument("--no-steady-barrier", action="store_true")
    p.add_argument("--barrier-first", action="store_true")
    p.add_argument("--multihost", action="store_true")
    return p.parse_args()


ARGS = _parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

AXIS = "x"
IS_LEAD = jax.process_index() == 0


def rprint(*a, **kw):
    if IS_LEAD:
        print(*a, **kw)
        sys.stdout.flush()


def _load_kernels():
    path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
            "/src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("llar", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


llar = _load_kernels()


def make_check(mesh, n, rounds, sa, sb, dtype, collective_id: int):
    """Run both chains and reduce the comparison to a replicated scalar.

    The comparison happens *on device* and comes back as a single number. Pulling
    the arrays to the host instead works on one process and fails on four --
    ``jax.device_get`` refuses an array spanning non-addressable devices -- which
    is exactly how this passed at 8 devices and died at 32 (job 267). Reducing
    with ``pmax`` inside the shard_map means the host only ever fetches a scalar,
    and it also makes every rank agree on the verdict.

    Both chains carry the same per-round twist, and a *different* twist per
    operand, so a kernel that reduced one operand twice or swapped the two
    workspaces cannot match on both.
    """

    def _twist(a, b, j):
        a = a * (1.0 / n) + jnp.asarray(j % 3, dtype) * 0.25
        b = b * (1.0 / n) + jnp.asarray((j + 1) % 3, dtype) * 0.5
        return a, b

    def body(la, lb):
        a0, b0 = la[0], lb[0]

        a, b = a0, b0
        rsa, rsb, aga, agb, step = llar.make_two_shot_variadic_workspace(
            int(np.prod(sa)), int(np.prod(sb)), dtype, n)
        if not ARGS.no_cse_guard:
            # When the two operands have equal width, rsa/rsb (and aga/agb) are
            # identical shape *and* identical value (zeros), so XLA is free to
            # CSE each pair into one buffer -- which input_output_aliases then
            # aliases twice, collapsing the two operands' workspaces onto each
            # other. That is the hypothesis for why 3584x3584 fails
            # deterministically at 32 devices (ca=cb=112) while 3584x7168 and
            # 7168x3584 pass. optimization_barrier forces them to stay distinct.
            rsa, rsb, aga, agb, step = jax.lax.optimization_barrier(
                (rsa, rsb, aga, agb, step))
        for j in range(rounds):
            a, b, rsa, rsb, aga, agb, step = (
                llar.two_shot_all_reduce_variadic_ws(
                    a,
                    b,
                    rsa,
                    rsb,
                    aga,
                    agb,
                    step,
                    axis_name=AXIS,
                    num_devices=n,
                    collective_id=collective_id,
                    barrier_first=bool(ARGS.barrier_first),
                    steady_barrier=not ARGS.no_steady_barrier))
            a, b = _twist(a, b, j)

        ra, rb = a0, b0
        for j in range(rounds):
            ra = jax.lax.psum(ra, axis_name=AXIS)
            rb = jax.lax.psum(rb, axis_name=AXIS)
            ra, rb = _twist(ra, rb, j)

        err = jnp.asarray(0.0, jnp.float32)
        for got, want in ((a, ra), (b, rb)):
            got = got.astype(jnp.float32)
            want = want.astype(jnp.float32)
            rel = jnp.max(jnp.abs(got - want)) / jnp.maximum(
                jnp.max(jnp.abs(want)), 1e-6)
            err = jnp.maximum(err, rel)

        # Keep the final workspaces live. This is the caller contract, not a
        # nicety: with the outputs dead *and* the two operands the same width,
        # rsa/rsb have identical shapes and XLA's buffer assignment is free to
        # give them one allocation, so both operands scatter into the same
        # workspace. That is exactly and only what failed -- 3584x3584 at
        # rounds=1, where a single call means nothing consumes the returned
        # workspaces, while every rounds>=2 case threads them into the next call
        # and passes. An optimization_barrier on the *inputs* changed the error
        # without fixing it, because the freedom is in output assignment.
        # Production always threads them (see get_moe_variadic_op's copy_ back).
        err = jax.lax.optimization_barrier((err, rsa, rsb, aga, agb, step))[0]
        return jax.lax.pmax(err, axis_name=AXIS)

    spec = P(AXIS, None, None)
    return jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(spec, spec),
                      out_specs=P(),
                      check_vma=False))


def main():
    dtype = jnp.float32 if ARGS.dtype == "float32" else jnp.bfloat16
    n = ARGS.devices
    devices = sorted(jax.devices(), key=lambda d: d.id)[:n]
    if len(devices) < n:
        raise SystemExit(f"need {n} devices, saw {len(devices)}")
    mesh = Mesh(np.asarray(devices), axis_names=(AXIS, ))
    shard = NamedSharding(mesh, P(AXIS, None, None))
    tol = ARGS.tol if ARGS.tol is not None else (
        1e-4 if ARGS.dtype == "float32" else 0.05)

    rprint(f"== variadic stress: {n} devices, {jax.process_count()} "
           f"process(es), {ARGS.dtype}, tol {tol:g}")
    rprint(f"== widths={ARGS.widths} rows={ARGS.rows} rounds={ARGS.rounds} "
           f"repeats={ARGS.repeats} "
           f"steady_barrier={not ARGS.no_steady_barrier}")

    total = 0
    bad = []
    import itertools
    cids = itertools.count(1)
    for spec in ARGS.widths.split(","):
        wa, wb = (int(v) for v in spec.strip().split("x"))
        for rows in (int(r) for r in ARGS.rows.split(",")):
            sa, sb = (rows, wa), (rows, wb)
            if (rows * wa) % n or (rows * wb) % n:
                rprint(f"  {spec} rows={rows}: skip, not divisible by {n}")
                continue
            a = jax.jit(lambda s=sa: jax.random.normal(jax.random.key(0),
                                                       (n, *s), dtype),
                        out_shardings=shard)()
            b = jax.jit(lambda s=sb: jax.random.normal(jax.random.key(1),
                                                       (n, *s), dtype),
                        out_shardings=shard)()
            for rounds in (int(r) for r in ARGS.rounds.split(",")):
                # A distinct collective_id per config, because they share a
                # process. When a chain ends, peers exit at different times: a
                # device still in round k signals a peer's barrier semaphore for
                # a round that peer has already passed, and those counts persist.
                # The next config's step-0 barrier then releases early on a
                # stale count. That is what failed 3584x3584 rounds=1 (and only
                # its first round) at 32 devices while 200-round chains passed.
                # Production is not exposed: the dispatcher hands every call
                # site its own id from _next_collective_id.
                cid = next(cids)
                check = make_check(mesh, n, rounds, sa, sb, dtype, cid)
                worst = 0.0
                for _ in range(ARGS.repeats):
                    worst = max(worst, float(check(a, b)))
                    total += 1
                ok = worst <= tol
                if not ok:
                    bad.append((spec, rows, rounds, worst))
                rprint(f"  a={sa} b={sb} rounds={rounds:>3} cid={cid:>3}  "
                       f"max_rel_err={worst:.2e} tol={tol:g}  "
                       f"{'ok' if ok else 'FAIL'}")

    rprint(f"\n{total} executions checked, {len(bad)} failing configs")
    if bad:
        for spec, rows, rounds, err in bad:
            rprint(f"  FAIL {spec} rows={rows} rounds={rounds} err={err:.3e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
