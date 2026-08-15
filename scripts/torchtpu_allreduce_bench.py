#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""torch_tpu all-reduce latency microbenchmark -- one process per TPU device.

Companion to ``allreduce_latency_bench.py`` (which measures the same thing
through JAX).  This one goes through the stack vLLM actually uses:
``torch.distributed`` functional collectives on the ``tpu_dist`` backend,
captured by ``torch.compile(backend="tpu")``.  Running both answers whether
torchtpu adds any overhead of its own on top of the XLA collective, or whether
the two are the same op and the latency is XLA's to fix.

Same two-point slope methodology: time a chain of ``n_lo`` then ``n_hi``
data-dependent all-reduces in one compiled graph and take the slope, so
dispatch, host sync and D2H all cancel.

Launch (single host, 8 devices):

    eval "$(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
    torchrun --nproc_per_node="$WORLD_SIZE" scripts/torchtpu_allreduce_bench.py
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--op", default="allreduce", choices=["allreduce", "noop"])
    p.add_argument("--interleave-matmul",
                   action="store_true",
                   help="Insert a cols x cols matmul before each op, so the "
                   "graph looks like a real TP layer instead of a bare "
                   "chain of collectives.")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def build_chain(n_ops: int, world_size: int, op: str = "allreduce"):
    """Compiled graph of ``n_ops`` chained ops.

    The 1/N scale keeps values bounded and, more importantly, makes each
    all-reduce consume the previous one's result so the compiler cannot CSE
    them into a single collective.

    ``op="noop"`` swaps the all-reduce for a pure-compute op of the same shape.
    That is the control: torch_tpu's ``force_collectives_output`` FX pass makes
    every collective a submodule output, so a chain of N collectives can become
    N separately dispatched executables. Comparing against the compute-only
    chain separates per-executable dispatch cost from the collective itself.
    """
    scale = 1.0 / world_size

    def f(x, w=None):
        for _ in range(n_ops):
            if w is not None:
                # Model-like graph: real compute between collectives, so a
                # host-side dispatch that merely fails to *overlap* shows up
                # differently from one that is genuinely serialized.
                x = torch.matmul(x, w)
            if op == "allreduce":
                x = funcol_all_reduce(x, "sum", list(
                    range(world_size))) * scale
            else:
                x = x * scale
        return x

    # dynamic=False is required: once the sweep hits its second shape dynamo
    # would otherwise generalize to symbolic shapes, which the TPU backend
    # rejects outright.
    return torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)


def time_fn(fn, x, repeats: int, warmup: int, w=None) -> float:
    """Median wall time in seconds; ``.cpu()`` forces execution and sync."""
    args = (x, ) if w is None else (x, w)
    for _ in range(warmup):
        fn(*args).cpu()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        t0 = time.perf_counter()
        fn(*args).cpu()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8
    device = torch.device("tpu")

    if rank == 0:
        print(f"torch {torch.__version__} | world_size={world_size} | "
              f"dtype={args.dtype} cols={args.cols}")
        print(
            f"slope over {args.iters_lo}->{args.iters_hi} chained all-reduces,"
            f" median of {args.repeats}\n")
        print(f"{'rows':>6} {'KiB':>9} {'torchtpu us':>13}")

    results = []
    for rows in (int(r) for r in args.rows.split(",")):
        x = torch.randn(rows, args.cols,
                        dtype=torch.float32).to(dtype).to(device)
        lo = build_chain(args.iters_lo, world_size, args.op)
        hi = build_chain(args.iters_hi, world_size, args.op)
        w = (torch.randn(args.cols, args.cols, dtype=torch.float32).to(
            dtype).to(device)) if args.interleave_matmul else None
        t_lo = time_fn(lo, x, args.repeats, args.warmup, w)
        t_hi = time_fn(hi, x, args.repeats, args.warmup, w)
        latency = (t_hi - t_lo) / (args.iters_hi - args.iters_lo)
        payload = rows * args.cols * itemsize
        if rank == 0:
            print(f"{rows:>6} {payload / 1024:>9.2f} {latency * 1e6:>13.2f}"
                  f"  (T{args.iters_lo}={t_lo * 1e3:.2f}ms "
                  f"T{args.iters_hi}={t_hi * 1e3:.2f}ms)")
        results.append({
            "rows": rows,
            "cols": args.cols,
            "payload_bytes": payload,
            "impl": "torchtpu",
            "latency_us": latency * 1e6,
        })

    if rank == 0 and args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"wrote {args.csv}")

    # torch_tpu's distributed runtime does not always tear down cleanly and can
    # fail the process *after* every measurement is already in hand. Don't let
    # that turn a good run into a nonzero exit.
    try:
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"(ignored teardown error: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
