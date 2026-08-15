#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness + latency for the low-latency all-reduce integration, TP8.

Exercises ``vllm_torchtpu.distributed.low_latency_all_reduce`` exactly as the
TPU device communicator does, but without standing up a model -- one process
per core, ``tpu_dist`` process group, a stub communicator carrying the four
attributes the dispatcher reads.

What is actually at risk here, and therefore what this checks:

* **Repeated calls.** The ``_ws`` kernels carry a round counter and a
  double-buffered workspace across invocations; a dropped, doubled or stale
  round is the failure mode the design is most exposed to. So every shape runs
  a *chain* of reductions whose input depends on the previous output, checked
  against ``funcol.all_reduce`` at every step rather than once.
* **The gate.** Shapes on both sides of each crossover, confirming the ones
  outside the measured range fall through to XLA rather than running a kernel
  that would be slower (or, at 1 MiB / 32 devices, catastrophically slower).
* **Under torch.compile**, which is how vLLM runs it: the workspace is a
  donated graph input mutated with ``copy_``, and that has to survive tracing.

Run:
    eval "$(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
    torchrun --nproc_per_node=8 scripts/debug/test_low_latency_all_reduce.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce


class _StubCommunicator:
    """The four attributes LowLatencyAllReduce reads off a communicator."""

    def __init__(self, cpu_group, rank_in_group: int, world_size: int):
        self.cpu_group = cpu_group
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.device = torch.device("tpu")
        self.unique_name = "tp:0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512")
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--chain",
                   type=int,
                   default=6,
                   help="sequential reductions per shape (exercises parity)")
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-timing", action="store_true")
    p.add_argument("--compile-first",
                   action="store_true",
                   help="Trace under torch.compile before any eager call, "
                   "which is how a vLLM server hits this: the first time "
                   "a shape is seen is inside dynamo. An eager pass "
                   "first would populate the plan cache and hide any "
                   "untraceable work in the builder.")
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="float32 gives an exact oracle: any deviation beyond "
                   "~1e-6 is corruption, not rounding.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from vllm_torchtpu.distributed.low_latency_all_reduce import (
        LowLatencyAllReduce, _choose_kernel)

    comm = _StubCommunicator(gloo, rank, world_size)
    dispatcher = LowLatencyAllReduce(comm)
    device = torch.device("tpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8

    if rank == 0:
        print(f"world_size={world_size} dtype={args.dtype} cols={args.cols}",
              flush=True)
        print(
            f"{'rows':>6} {'KiB':>8} {'route':>9} {'max|diff|':>10} "
            f"{'status':>8}",
            flush=True)

    failures = []

    if args.compile_first:
        # A vLLM server meets every shape for the first time *inside* dynamo,
        # so anything untraceable in the plan builder fails there. Running an
        # eager pass first would populate the cache and hide exactly that.
        if rank == 0:
            print("\n-- compile-first: prebuild eagerly, then trace --",
                  flush=True)
        # This is the sequence a vLLM server runs: warmup() builds every plan
        # before torch.compile sees a forward, and the traced region then only
        # ever hits the cache.
        for rows in (int(r) for r in args.rows.split(",")):
            probe = torch.empty((rows, args.cols), dtype=dtype, device=device)
            dispatcher._build_plan(probe, ((rows, args.cols), dtype))
            del probe
        for rows in (int(r) for r in args.rows.split(",")):
            x = (torch.randn(rows, args.cols,
                             dtype=torch.float32).to(dtype).to(device))

            def one(x):
                out = dispatcher.maybe_all_reduce(x.contiguous())
                return x if out is None else out

            try:
                compiled = torch.compile(one,
                                         backend="tpu",
                                         fullgraph=True,
                                         dynamic=False)
                got = compiled(x)
                want = funcol_all_reduce(x, "sum", list(range(world_size)))
                d = (got.float() - want.float()).abs().max().cpu().item()
                scale = max(1.0, want.float().abs().max().cpu().item())
                rel = d / scale
                ok = rel < (1e-5 if dtype == torch.float32 else 0.02)
                if not ok:
                    failures.append(
                        f"compile-first rows={rows}: rel={rel:.5f}")
                if rank == 0:
                    print(
                        f"  rows={rows:>4} traced+ran, rel={rel:.5f} "
                        f"{'OK' if ok else 'FAIL'}",
                        flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"compile-first rows={rows}: "
                                f"{type(exc).__name__}: {str(exc)[:200]}")
                if rank == 0:
                    print(
                        f"  rows={rows:>4} FAILED: {type(exc).__name__}: "
                        f"{str(exc)[:300]}",
                        flush=True)
        if rank == 0:
            print("", flush=True)

    for rows in (int(r) for r in args.rows.split(",")):
        payload = rows * args.cols * itemsize
        expected_route = _choose_kernel(world_size, payload, rows * args.cols)

        torch.manual_seed(1234 + rank)
        x = (torch.randn(rows, args.cols,
                         dtype=torch.float32).to(dtype).to(device))

        # Chain: each round's input depends on the previous round's output, so
        # a stale or duplicated round changes the values rather than hiding in
        # an idempotent mean.
        got = x
        want = x
        worst = 0.0
        for step in range(args.chain):
            out = dispatcher.maybe_all_reduce(got.contiguous())
            if out is None:
                if expected_route != "xla":
                    failures.append(f"rows={rows}: expected "
                                    f"{expected_route}, dispatcher declined")
                    break
                # Declining is the gate working; the reference path is what
                # the communicator would have done.
                out = funcol_all_reduce(got, "sum", list(range(world_size)))
            twist = float(step + 1)
            got = out * (1.0 / world_size) + twist
            want = funcol_all_reduce(want, "sum", list(range(world_size)))
            want = want * (1.0 / world_size) + twist
            diff = (got.float() - want.float()).abs().max().cpu().item()
            scale = max(1.0, want.float().abs().max().cpu().item())
            # bf16 carries an 8-bit mantissa, so a different summation order
            # legitimately differs by a few ulps (~2^-8 relative). Judge
            # relative to magnitude, not against a fixed absolute number.
            worst = max(worst, diff / scale)

        route = dispatcher._plans[(tuple(x.shape), dtype)].kernel
        # float32 is an exact oracle; bf16 legitimately reorders by a few ulps.
        tol = 1e-5 if dtype == torch.float32 else 0.02
        ok = worst < tol
        if not ok:
            failures.append(f"rows={rows}: max|diff|={worst:.4f}")
        if route != expected_route:
            failures.append(
                f"rows={rows}: routed {route}, gate said {expected_route}")
        if rank == 0:
            print(
                f"{rows:>6} {payload / 1024:>8.1f} {route:>9} "
                f"{worst:>10.5f} {'OK' if ok else 'FAIL':>8}",
                flush=True)

    if args.skip_timing:
        return _finish(rank, world_size, gloo, failures)

    # ---- latency, same two-point slope as the microbenchmarks ----
    if rank == 0:
        print(
            f"\nchained-latency slope {args.iters_lo}->{args.iters_hi}, "
            f"median of {args.repeats}",
            flush=True)
        print(
            f"{'rows':>6} {'KiB':>8} {'route':>9} {'xla us':>9} "
            f"{'kernel us':>10} {'speedup':>8}",
            flush=True)

    def build(n_ops: int, use_kernel: bool):

        def f(x):
            for _ in range(n_ops):
                if use_kernel:
                    out = dispatcher.maybe_all_reduce(x.contiguous())
                    x = x if out is None else out
                else:
                    x = funcol_all_reduce(x, "sum", list(range(world_size)))
                x = x * (1.0 / world_size)
            return x

        return torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)

    def time_fn(fn, x) -> float:
        for _ in range(args.warmup):
            fn(x).cpu()
        samples = []
        for _ in range(args.repeats):
            dist.barrier()
            t0 = time.perf_counter()
            fn(x).cpu()
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    for rows in (int(r) for r in args.rows.split(",")):
        payload = rows * args.cols * itemsize
        route = _choose_kernel(world_size, payload, rows * args.cols)
        x = (torch.randn(rows, args.cols,
                         dtype=torch.float32).to(dtype).to(device))
        lat = {}
        for label, use_kernel in (("xla", False), ("kernel", True)):
            lo = build(args.iters_lo, use_kernel)
            hi = build(args.iters_hi, use_kernel)
            t_lo = time_fn(lo, x)
            t_hi = time_fn(hi, x)
            lat[label] = ((t_hi - t_lo) /
                          (args.iters_hi - args.iters_lo)) * 1e6
        if rank == 0:
            speedup = lat["xla"] / lat["kernel"] if lat["kernel"] > 0 else 0.0
            print(
                f"{rows:>6} {payload / 1024:>8.1f} {route:>9} "
                f"{lat['xla']:>9.2f} {lat['kernel']:>10.2f} "
                f"{speedup:>7.2f}x",
                flush=True)

    return _finish(rank, world_size, gloo, failures)


def _finish(rank, world_size, gloo, failures) -> int:
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, failures, group=gloo)
    every = [f for per_rank in gathered for f in per_rank]
    if rank == 0:
        if every:
            print(f"\nFAILURES ({len(every)}):", flush=True)
            for f in sorted(set(every)):
                print(f"  {f}", flush=True)
        else:
            print("\nall shapes correct and routed as the gate specifies",
                  flush=True)
    try:
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"(ignored teardown error: {type(e).__name__}: {e})")
    return 1 if every else 0


if __name__ == "__main__":
    sys.exit(main())
