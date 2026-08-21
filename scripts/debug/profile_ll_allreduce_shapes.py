#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-occurrence device self time for the all-reduce, at K3's real shapes.

Step 0 of the low-latency all-reduce post-mortem. The K3 A/B measured ~26 us
per one-shot call in a production trace, against 15.14 us for "the same"
payload in the JAX microbenchmark (docs/low_latency_allreduce.md section 2b,
16 KiB at 32 devices). Those two numbers do not denote the same quantity, and
three differences could account for the gap:

  1. **Metric.** The bench reports a two-point wall-clock *slope* -- the
     marginal cost of one more collective, with every fixed cost cancelled by
     construction. The trace reports device *self time* per HLO op. This
     script reports both, from the same run.
  2. **Shape.** The doc's 16 KiB row is 16x1024. K3 reduces 2x3584 -- the same
     bytes, one eighth the rows, and the kernel grids over rows.
  3. **Stack.** The bench is pure JAX; production is JAX-under-torch_tpu-under-
     torch.compile. This script runs the production path.

This measures 1-3 together. What it deliberately does *not* reproduce is
arrival skew: a chained collective leaves the devices in lockstep, whereas a
real decode step puts ~100 us of MoE compute with per-rank expert imbalance
between them. If the numbers here reproduce production, skew is not the story
and the shape is; if they do not, skew is what is left.

The XLA arm is the control that makes this decisive. Production XLA at 2x7168
cost 20.1 us/occ. If this harness reads ~20 us there too, it is faithful and
the kernel's 26 us is a real property of the shape. If it reads well under
that, production carries a per-collective tax this harness cannot see -- and
the finding is that the kernel is more exposed to it than XLA's ring.

``--force-kernel`` builds a one-shot kernel even where the gate would route to
XLA. That is how 2x7168 gets measured, which is the number the "wire both
widths to kernels" follow-up lives or dies on: the pair has to beat the 34.0 us
that XLA's *fused* [2,3584]+[2,7168] collective costs today.

Run (32 devices, 4 hosts):
    sbatch scripts/debug/slurm_profile_ll_allreduce_32.sbatch
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
from torch_tpu._internal.profiler import TpuProfilerConfig

from vllm_torchtpu import profiler_trace


class _StubCommunicator:
    """The attributes LowLatencyAllReduce reads off a real communicator."""

    def __init__(self, cpu_group, device_group, rank_in_group: int,
                 world_size: int):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.device = torch.device("tpu")
        self.unique_name = "tp:0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes",
                   default="2x3584,2x7168,16x1024,16x3584,16x7168",
                   help="'ROWSxCOLS' payloads. K3 decode reduces 2x3584 and "
                   "2x7168 at concurrency 1; 16x1024 is the doc's 16 KiB "
                   "row, kept as a methodology control.")
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--profile-reps",
                   type=int,
                   default=3,
                   help="Traced executions of the hi chain. Each carries "
                   "iters_hi collectives, so this is already well sampled.")
    p.add_argument("--force-kernel",
                   action="store_true",
                   help="Build a kernel even where the gate routes to XLA, "
                   "so shapes past the crossover are measurable.")
    p.add_argument("--gap-dim",
                   type=int,
                   default=0,
                   help="Put a DxD matmul between consecutive collectives. 0 "
                   "(default) chains them back-to-back, which is what the "
                   "microbenchmarks do and what a decode step does NOT: there "
                   "a collective is separated from the next by a layer of "
                   "compute. Back-to-back calls leave every peer already "
                   "sitting at the barrier, so a barrier-heavy kernel is "
                   "measured at its most favourable. 2048 is roughly 20 us "
                   "of matmul here, 4096 roughly 150 us.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Each shape compiles the same chain code object four times (lo/hi x
    # xla/kernel), so the default limit of 8 aborts on the third shape. These
    # recompiles are the measurement, not a performance bug.
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from vllm_torchtpu.distributed.low_latency_all_reduce import (
        LowLatencyAllReduce, _choose_kernel, _Plan)

    comm = _StubCommunicator(gloo, dist.group.WORLD, rank, world_size)
    dispatcher = LowLatencyAllReduce(comm)
    # No get_entry_op here: the shipped entry op took one_shot's (ws, step)
    # pair, and two_shot carries (rs, ag, step). Each shape below builds its
    # own equivalent 4-arg op instead.

    device = torch.device("tpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8
    shapes = []
    for tok in args.shapes.split(","):
        r, _, c = tok.strip().lower().partition("x")
        shapes.append((int(r), int(c)))

    def lead(*a):
        if rank == 0:
            print(*a, flush=True)

    # Optional compute between collectives. gap_eps holds 0.0 but is a runtime
    # tensor, so the matmul survives DCE while the reduced values do not move.
    gap_a = gap_eps = None
    if args.gap_dim:
        gap_a = torch.randn(args.gap_dim, args.gap_dim,
                            dtype=torch.float32).to(dtype).to(device)
        gap_eps = torch.zeros((), dtype=torch.float32).to(device)

    lead(f"world_size={world_size} dtype={args.dtype} "
         f"force_kernel={args.force_kernel}")
    lead(f"slope {args.iters_lo}->{args.iters_hi}, median of {args.repeats}\n")
    lead(f"{'shape':>10} {'KiB':>8} {'gate':>9} "
         f"{'xla slope':>10} {'ker slope':>10} "
         f"{'xla abs':>9} {'ker abs':>9} {'speedup':>8}")

    def time_fn(fn, x) -> float:
        for _ in range(args.warmup):
            fn(x).cpu()
        samples = []
        for _ in range(args.repeats):
            # Line the ranks up so one straggler's dispatch does not land
            # inside another's timed region.
            dist.barrier()
            t0 = time.perf_counter()
            fn(x).cpu()
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    def profile_chain(fn, x, tag: str) -> None:
        """Trace a warm chain through the profiler production uses.

        Per-rank capture directories, exactly as PhaseBasedProfiler lays them
        out, so scripts/allreduce_profile_stats.py and scripts/k3/diff_arms.py
        read these the same way they read a server's traces.
        """
        run_dir = os.path.join(args.profile_dir, tag)
        cap_dir = profiler_trace.rank_capture_dir(run_dir, rank)
        os.makedirs(cap_dir, exist_ok=True)
        fn(x).cpu()
        dist.barrier()
        cfg = TpuProfilerConfig(
            run_dir=cap_dir,
            host_tracer_level=2,
            device_tracer_level=1,
            python_tracer_level=1,
            experimental_options={
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=cap_dir, use_gzip=True),
                experimental_config=cfg,
        ):
            for _ in range(args.profile_reps):
                fn(x).cpu()
        dist.barrier()

    failures = []
    for rows, cols in shapes:
        payload = rows * cols * itemsize
        gate = _choose_kernel(world_size, payload, rows * cols)

        # Build the plan before anything is traced -- the same order warmup()
        # uses, and the only order that works: registering or building inside
        # a traced region fails in dynamo.
        probe = torch.empty((rows, cols), dtype=dtype, device=device)
        plan = dispatcher._build_plan(probe, ((rows, cols), dtype))
        want = args.kernel_type
        if (plan.kernel != want) and args.force_kernel:
            try:
                # _build_plan short-circuits the mesh lookup when the gate
                # already said "xla", so self._mesh is unset for exactly the
                # shapes worth forcing. The builders read it directly.
                if dispatcher._get_mesh() is None:
                    raise RuntimeError("no mesh; cannot force a kernel")
                # _build_two_shot reshapes to (N, numel//N) internally, so
                # the rows % N == 0 limit in the docs does not apply on this
                # path -- only numel % N == 0, which 2x10752 (21504 = 32 x 672)
                # satisfies.
                if (rows * cols) % world_size:
                    raise RuntimeError(f"numel {rows * cols} not divisible by "
                                       f"{world_size}")
                op, buffers = dispatcher._build_two_shot((rows, cols), dtype)
                plan = _Plan(want, op, buffers, (rows, cols), dtype)
                dispatcher._plans[((rows, cols), dtype)] = plan
            except Exception as e:  # noqa: BLE001
                lead(f"  !! forced {want} {rows}x{cols} failed: "
                     f"{type(e).__name__}: {e}")
                plan = _Plan("xla", None, ())
        del probe
        if plan.kernel == "xla":
            lead(f"{f'{rows}x{cols}':>10} {payload / 1024:>8.1f} {gate:>9} "
                 f"  (no kernel built; pass --force-kernel to measure one)")
            continue
        # two_shot carries three buffers (rs, ag, step). The shipped entry op
        # took the one_shot pair, so drive it through an equivalent 4-arg op.
        # Same contract as get_entry_op: opaque to dynamo, buffers as arguments
        # rather than captures, mutation undeclared because declaring it makes
        # AOTAutograd add them to the graph outputs and the runtime wrapper then
        # expects more outputs than the artifact returns.
        rs, ag, step = plan.buffers
        ts_name = f"vllm_torchtpu::ll_two_shot_probe_{rows}_{cols}"

        def _ts_impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
                     step: torch.Tensor) -> torch.Tensor:
            out, nrs, nag, nstep = plan.op(x, rs, ag, step)
            rs.copy_(nrs)
            ag.copy_(nag)
            step.copy_(nstep)
            return out

        ts_op = torch.library.custom_op(ts_name, _ts_impl, mutates_args=())
        ts_op.register_fake(lambda x, rs, ag, step: torch.empty_like(x))

        def kernel_reduce(t):
            return ts_op(t.contiguous(), rs, ag, step)

        def build(n_ops: int, use_kernel: bool):

            def f(x):
                for _ in range(n_ops):
                    if use_kernel:
                        x = kernel_reduce(x)
                    else:
                        x = funcol_all_reduce(x, "sum",
                                              list(range(world_size)))
                    x = x * (1.0 / world_size)
                    if gap_a is not None:
                        # Separate this collective from the next by real
                        # compute, on the dependency chain rather than beside
                        # it: the matmul consumes this round's x and its result
                        # feeds the next collective, exactly as a decoder layer
                        # sits between two all-reduces.
                        #
                        # It must depend on x. A loop-invariant `gap_a @ gap_a`
                        # is identical every iteration, so XLA hoists it out by
                        # CSE and the chain stays back-to-back -- measured, and
                        # the slope did not move.
                        #
                        # gap_eps is a runtime tensor holding 0.0, so the result
                        # cannot be folded away but the values reaching the next
                        # collective are bit-unchanged.
                        g = (gap_a * x.sum().to(gap_a.dtype)) @ gap_a
                        x = x + (g.sum() * gap_eps).to(x.dtype)
                return x

            return torch.compile(f,
                                 backend="tpu",
                                 fullgraph=True,
                                 dynamic=False)

        x = (torch.randn(rows, cols, dtype=torch.float32).to(dtype).to(device))

        # Correctness first: a fast wrong kernel is not a measurement.
        one = torch.compile(lambda t: kernel_reduce(t),
                            backend="tpu",
                            fullgraph=True,
                            dynamic=False)
        got = one(x)
        want = funcol_all_reduce(x, "sum", list(range(world_size)))
        d = (got.float() - want.float()).abs().max().cpu().item()
        scale = max(1.0, want.float().abs().max().cpu().item())
        # bf16 sums N-1 peers in a different order than the reference, and that
        # drift grows with device count -- 0.024 at 32 devices is reordering,
        # not corruption. Scale the tolerance the way the JAX bench does rather
        # than fixing it at a number that only holds at 8 devices.
        tol = 1e-5 if dtype == torch.float32 else 0.01 * world_size**0.5
        if d / scale >= tol:
            failures.append(
                f"{rows}x{cols}: rel={d / scale:.5f} tol={tol:.5f}")

        slope, absolute = {}, {}
        for label, use_kernel in (("xla", False), ("kernel", True)):
            lo = build(args.iters_lo, use_kernel)
            hi = build(args.iters_hi, use_kernel)
            t_lo = time_fn(lo, x)
            t_hi = time_fn(hi, x)
            slope[label] = ((t_hi - t_lo) /
                            (args.iters_hi - args.iters_lo)) * 1e6
            absolute[label] = (t_hi / args.iters_hi) * 1e6
            if args.profile_dir:
                profile_chain(hi, x, f"{label}_{rows}x{cols}")

        sp = (slope["xla"] /
              slope["kernel"] if slope["kernel"] > 0 else float("nan"))
        lead(f"{f'{rows}x{cols}':>10} {payload / 1024:>8.1f} {gate:>9} "
             f"{slope['xla']:>10.2f} {slope['kernel']:>10.2f} "
             f"{absolute['xla']:>9.2f} {absolute['kernel']:>9.2f} "
             f"{sp:>7.2f}x")

    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, failures, group=gloo)
    every = [f for per_rank in gathered for f in per_rank]
    if rank == 0 and every:
        print("\nCORRECTNESS FAILURES:", flush=True)
        for f in every:
            print(f"  {f}", flush=True)
    dist.destroy_process_group()
    return 1 if every else 0


if __name__ == "__main__":
    raise SystemExit(main())
