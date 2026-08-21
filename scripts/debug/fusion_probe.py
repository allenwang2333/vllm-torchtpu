#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K3's collective *topology* in a two-minute job, so fusion is measurable.

Every microbenchmark in this repo before this one measured a collective with no
neighbours, and that is precisely what made them mispredict production. They
were accurate about the collective itself -- two-shot's 12.67 us/occ became
13.32 in production, a 5% tax -- and blind to everything arising from context:

  * XLA's CRS combiner fusing two adjacent reductions, worth 18 us/layer, the
    single largest effect in the whole investigation. It needs *two* reductions
    to appear at all, and no bench had two.
  * The two K3 sites sharing the shape [2, 7168], so wiring "the shape" to a
    kernel captures both and splits the fused pair. A bench has one call site
    by construction, so this could only be discovered on K3 -- at a 45-minute
    bring-up per attempt.

So this reproduces the topology rather than a single call:

    p = all_reduce(h)                    [2, 7168]   standalone, post-attention
    h2 = h + gap(p)                                  compute, so p cannot fuse
    a = all_reduce(latent(h2))           [2, 3584]   \\ independent of each other
    b = all_reduce(shared(h2))           [2, 7168]   / -> the combiner's pair

``a`` and ``b`` are independent of one another and both downstream of ``p``,
which is what makes the pair fusable and ``p`` not: exactly K3's arrangement,
where the attention output feeds a MoE block whose latent and shared branches
diverge. The producers in front of the reductions are real matmuls, so the
operands arrive rather than sitting resident.

Arms, and what each answers:

  xla         all three through XLA. Expect ONE standalone [2,7168] plus ONE
              fused [2,3584]+[2,7168] -- the control that proves the probe
              reproduces the combiner at all. If this arm shows two separate
              collectives, the probe is wrong and nothing below means anything.
  shape7168   kernel wired to the *shape* [2,7168], capturing p and b both.
              Reproduces the +5.5% K3 regression (job 120) in two minutes.
  p_only      kernel on p alone, pair left to XLA. The arrangement K3 could not
              be made to run (the dispatcher routes by shape, and the site
              patch that tried to fix it halted the SparseCore) -- but trivial
              here, because this script chooses its own call sites. This is the
              number the post-attention plan lives or dies on.
  concat      kernel on p, and ONE kernel over the concatenated pair as
              [2,10752]. The target arrangement: keeps the amortisation the
              combiner was providing and uses the faster collective for it.
              Turns the ~27 us projection into a measurement.

``--intra-chip-latent`` is the Tier-2 probe for the ``[2,3584]`` site wait, the
largest single number in the investigation: that reduction costs 17.31 us
isolated and 32.66 in production, +88%, while the *larger* ``[2,7168]`` pays only
+8%. A smaller payload cannot take longer on the same network, so ~15 us of it is
the collective blocked at its barrier rather than working -- and it is uniform
across ranks (CV 1.5%, so not skew) and implementation-independent (XLA +88%,
one-shot +85%), which is why no kernel touches it. At 92 layers it is ~1.4 ms per
decode step, roughly 8% of TPOT, and the control pays it too.

A plain matmul producer does not reproduce it: this probe's standalone
``[2,3584]`` reads 17.60, i.e. the isolated cost. The K3 trace says why -- its
operand arrives via an async ``copy-done``, and there are all-gathers at
``bf16[1,2,1792]`` / ``bf16[2,2,1792]`` right beside the reduction, where
1792 = 3584/2. That is the latent projection sharded across a chip's two cores
and gathered before the TP reduce. So the flag models the real producer chain:
a half-width matmul, an intra-chip all-gather over a 2-rank subgroup, then the
TP all-reduce.

**Measured, and the intra-chip gather is exonerated.** Paired runs (132 plain,
133 with the flag):

    collective us/occ        plain   intra-chip   K3 production
    standalone [2,3584]      17.74        16.28           32.66
    fused pair               21.16        18.85           33.97
    intra-chip all-gather        -      2.0-2.3         1.8-1.9

The reduction got *faster* with the sharded-and-gathered producer, not slower,
and nowhere near production's 32.66. The model is not too weak to show the
effect either -- its all-gather reproduces K3's within 0.4 us. So the wait does
not come from the gather-to-reduce handoff, and the
``TPU_LATENT_PROJ_INTRA_CHIP_TP=0`` arm that would have tested the same
hypothesis on K3 is unnecessary.

What is left is upstream of anything modelled here. The reduction is the first
consumer of the latent-*MoE* result, so its operand waits on the expert
computation -- routing, dispatch, the EP combine across chips, and whatever
SparseCore does in that path -- not on a projection. Uniform per-rank self time
(CV 1.5%) says each rank blocks the same amount on a local dependency rather
than on a straggler peer, which fits an operand that simply is not ready ~15 us
after the collective is scheduled. Modelling that needs a real MoE layer
(Tier 3), not a producer stub.

**Read the collective rows from the trace, not the us/layer slope.** Across two
runs of identical code (jobs 129 and 130) the per-collective self time
reproduced to within 0.07 us while the whole-layer slope moved up to 7 us for
the same arm:

    arm         129 layer  130 layer   129 coll  130 coll
    p_only         106.27     113.42      33.22     33.29
    concat         103.98     103.98      28.60     28.47
    shape7168      116.35     119.12      43.80     43.72

The slope subtracts two wall-clock medians and so carries the producers' jitter
across four hosts; the collective rows are device self time for the ops actually
under test. On collectives the arms are shape7168 +4.84, p_only -5.59, concat
-10.41 us/layer against xla. Quote those.

Run (32 devices, 4 hosts):
    sbatch --partition=ici-2x2x4-b scripts/debug/fusion_probe.sbatch
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device and backends
from torch.distributed._functional_collectives import \
    all_gather_tensor as funcol_all_gather
from torch.distributed._functional_collectives import \
    all_reduce as funcol_all_reduce
from torch_tpu._internal.profiler import TpuProfilerConfig

from vllm_torchtpu import profiler_trace

ARMS = ("xla", "shape7168", "p_only", "concat")


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
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--latent", type=int, default=3584)
    p.add_argument("--rows",
                   type=int,
                   default=2,
                   help="Decode token count. K3 at concurrency 1 reduces 2.")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--iters-lo", type=int, default=8)
    p.add_argument("--iters-hi",
                   type=int,
                   default=24,
                   help="Layers per chain. Each carries three reductions, so "
                   "24 is 72 collectives -- plenty, and short enough that the "
                   "producers do not dominate the trace.")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--gap-dim",
                   type=int,
                   default=1024,
                   help="Width of the matmul separating p from the pair. Its "
                   "job is to put real work on the dependency chain so p "
                   "cannot join the pair's fusion group.")
    p.add_argument(
        "--check-layers",
        type=int,
        default=4,
        help="Layers in the correctness chain. More than one on "
        "purpose: the workspace slot parity alternates per call, so "
        "a single call cannot catch a workspace that goes stale "
        "between calls.")
    p.add_argument(
        "--intra-chip-latent",
        action="store_true",
        help="Produce the latent operand the way K3 does: the projection "
        "sharded across a chip's two cores and all-gathered before the TP "
        "reduce, instead of one replicated matmul. This is the Tier-2 probe "
        "for the [2,3584] site wait -- see the module docstring.")
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--profile-reps", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Each arm compiles the chain twice (lo/hi); the default limit of 8 aborts
    # partway through four arms.
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)
    gloo = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from vllm_torchtpu.distributed.low_latency_all_reduce import (
        LowLatencyAllReduce, donation_enabled)

    comm = _StubCommunicator(gloo, dist.group.WORLD, rank, world_size)
    dispatcher = LowLatencyAllReduce(comm)
    donate = donation_enabled()

    device = torch.device("tpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    H, L, R = args.hidden, args.latent, args.rows

    def lead(*a):
        if rank == 0:
            print(*a, flush=True)

    if dispatcher._get_mesh() is None:
        lead("!! no mesh; cannot build any kernel")
        dist.destroy_process_group()
        return 2

    # The three payloads, plus the concatenated pair. two-shot reshapes to
    # (N, numel/N) internally, so the constraint is numel % world_size == 0.
    shapes = {
        "p": (R, H),
        "a": (R, L),
        "b": (R, H),
        # The pair concatenated: a's numel plus b's, viewed as R rows. At K3's
        # widths that is 2x10752 = 21504 = 32 x 672, so it divides.
        "ab": (R, L + H),
    }
    for name, (r, c) in shapes.items():
        if (r * c) % world_size:
            lead(f"!! {name} {r}x{c}: numel {r * c} not divisible by "
                 f"{world_size}")
            dist.destroy_process_group()
            return 2

    lead(f"world_size={world_size} dtype={args.dtype} rows={R} "
         f"hidden={H} latent={L}")
    lead(f"payloads: p={R * H * 2 / 1024:.1f} KiB  a={R * L * 2 / 1024:.1f} "
         f"KiB  b={R * H * 2 / 1024:.1f} KiB  "
         f"ab={R * (L + H) * 2 / 1024:.1f} KiB")
    lead(f"slope {args.iters_lo}->{args.iters_hi} layers, "
         f"median of {args.repeats}\n")

    # One kernel op per distinct payload. Built before anything is traced --
    # the same order warmup() uses, and the only one that works, since
    # building inside a traced region fails in dynamo.
    ops: dict[str, tuple] = {}
    for name in ("p", "a", "b", "ab"):
        try:
            op, buffers = dispatcher._build_two_shot(shapes[name], dtype)
            ops[name] = (op, buffers)
        except Exception as e:  # noqa: BLE001
            lead(f"!! could not build two_shot for {name} {shapes[name]}: "
                 f"{type(e).__name__}: {e}")

    def kernel_reduce(name):
        """A torch op for one payload, wired like the production entry op."""
        op, buffers = ops[name]
        rs, ag, step = buffers
        oname = f"vllm_torchtpu::fusion_probe_{name}_{shapes[name][1]}"

        def _impl(x: torch.Tensor, rs: torch.Tensor, ag: torch.Tensor,
                  step: torch.Tensor) -> torch.Tensor:
            out, nrs, nag, nstep = op(x, rs, ag, step)
            # Mirrors the production entry op, donation included, so what this
            # measures is what production would pay. See donation_enabled().
            if not donate:
                rs.copy_(nrs)
                ag.copy_(nag)
                step.copy_(nstep)
            return out

        top = torch.library.custom_op(oname, _impl, mutates_args=())
        top.register_fake(lambda x, rs, ag, step: torch.empty_like(x))

        def call(t):
            return top(t.contiguous(), rs, ag, step)

        return call

    kcall = {name: kernel_reduce(name) for name in ops}

    def xla_reduce(t):
        return funcol_all_reduce(t, "sum", list(range(world_size)))

    # The intra-chip group, for --intra-chip-latent. Adjacent ranks share a
    # chip: the dispatcher's rank-in-group -> device-id map came out
    # [8, 9, 12, 13, ...], so rank pairs land on consecutive device ids, which
    # are the two cores of one chip.
    #
    # Every rank creates every group, in the same order, and keeps its own.
    # new_group is collective -- a rank that builds only its own pair deadlocks
    # the others -- and the uniform order is what keeps the per-rank graphs
    # structurally identical, which this repo needs for the collectives to get
    # matching XLA channel ids.
    chip_group = None
    if args.intra_chip_latent:
        if world_size % 2:
            lead(f"!! --intra-chip-latent needs an even world_size, "
                 f"got {world_size}")
            dist.destroy_process_group()
            return 2
        for base in range(0, world_size, 2):
            grp = dist.new_group(ranks=[base, base + 1])
            if rank // 2 == base // 2:
                chip_group = grp
        lead(f"intra-chip latent: 2-core subgroups, latent shard "
             f"{R}x{L // 2}, gathered to {R}x{L}")

    # Producers. Real matmuls so each reduction's operand arrives rather than
    # being resident, which is the other thing the old benches never modelled.
    g = torch.Generator().manual_seed(1234)
    w_latent_full = torch.randn(H, L, generator=g, dtype=torch.float32)
    if args.intra_chip_latent:
        # Column-sharded across the chip's two cores, so this core computes
        # half the latent width and the gather reassembles it. Seeded
        # identically on every rank, then sliced, so the two halves compose
        # back into exactly the replicated weight -- which is what lets the
        # gathered result be compared against the unsharded reference.
        half = L // 2
        which = rank % 2
        w_latent = (w_latent_full[:, which * half:(which + 1) *
                                  half].contiguous().to(dtype).to(device))
    else:
        w_latent = w_latent_full.to(dtype).to(device)
    w_shared = (torch.randn(H, H, generator=g,
                            dtype=torch.float32).to(dtype).to(device))
    w_back = (torch.randn(L, H, generator=g,
                          dtype=torch.float32).to(dtype).to(device))
    gap_w = (torch.randn(H, args.gap_dim, generator=g,
                         dtype=torch.float32).to(dtype).to(device))
    gap_w2 = (torch.randn(args.gap_dim, H, generator=g,
                          dtype=torch.float32).to(dtype).to(device))
    inv = 1.0 / world_size

    def build(n_layers: int, arm: str):

        def layer(h):
            # p -- the standalone reduction, post-attention's analogue.
            # Every kernel arm must route p through the kernel, or its arm total is
            # not comparable: a missing entry here sent concat's p to XLA
            # and made the arm look 4.3 us/layer worse than it was.
            if arm in ("shape7168", "p_only", "concat"):
                p = kcall["p"](h)
            else:
                p = xla_reduce(h)
            p = p * inv

            # Compute between p and the pair, on the dependency chain. Without
            # this the scheduler is free to put all three reductions in one
            # fusion group, which is not K3's graph.
            h2 = h + ((p @ gap_w) @ gap_w2)

            # a and b -- independent of each other, both downstream of p.
            if chip_group is not None:
                # K3's real producer chain: a half-width matmul on each of the
                # chip's two cores, then an intra-chip all-gather to reassemble
                # the latent, and only then the TP all-reduce. The reduce
                # therefore waits on the gather's copy rather than reading a
                # resident tensor, which is the arrangement the +88% site wait
                # is suspected to come from.
                lat_shard = h2 @ w_latent
                lat = funcol_all_gather(lat_shard, 1, chip_group)
            else:
                lat = h2 @ w_latent
            sh = h2 @ w_shared
            if arm == "concat":
                # One reduction over both, which is what the combiner was
                # doing for XLA and what the kernel has to do itself.
                pair = torch.cat([lat.reshape(-1), sh.reshape(-1)])
                red = kcall["ab"](pair.reshape(R, -1))
                flat = red.reshape(-1)
                a = flat[:R * L].reshape(R, L) * inv
                b = flat[R * L:].reshape(R, H) * inv
            else:
                if arm == "shape7168":
                    # Wired by shape, so b is captured along with p and the
                    # pair can no longer fuse. This is the K3 regression.
                    a = xla_reduce(lat)
                    b = kcall["b"](sh)
                else:
                    a = xla_reduce(lat)
                    b = xla_reduce(sh)
                a = a * inv
                b = b * inv
            return (a @ w_back) + b

        def f(h):
            for _ in range(n_layers):
                h = layer(h)
            return h

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

    def profile_chain(fn, x, tag: str) -> None:
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

    h0 = (torch.randn(R, H, dtype=torch.float32).to(dtype).to(device))

    # Correctness: every arm against the xla arm, over a chain of several
    # layers rather than one.
    #
    # The length matters. The _ws kernels alternate workspace slots on a round
    # counter, so a single call only ever exercises slot 0 and round 0's
    # barrier placement. A workspace that goes stale between calls -- the exact
    # risk donation introduces -- gives the right answer on call 1 and the
    # wrong one on call 2, which a one-layer check cannot see. Several layers
    # put both parities and the steady-state barrier on the critical path.
    ref = build(args.check_layers, "xla")(h0).float()
    ref_scale = max(1.0, ref.abs().max().cpu().item())
    tol = 1e-5 if dtype == torch.float32 else 0.02 * world_size**0.5
    failures = []
    for arm in [a for a in args.arms.split(",") if a.strip() and a != "xla"]:
        try:
            got = build(args.check_layers, arm)(h0).float()
        except Exception as e:  # noqa: BLE001
            failures.append(
                f"{arm}: build/run failed: {type(e).__name__}: {e}")
            continue
        d = (got - ref).abs().max().cpu().item() / ref_scale
        if d >= tol:
            failures.append(f"{arm}: rel={d:.5f} tol={tol:.5f} over "
                            f"{args.check_layers} layers")

    lead(f"{'arm':>12} {'us/layer':>10} {'vs xla':>9}")
    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        try:
            lo, hi = build(args.iters_lo, arm), build(args.iters_hi, arm)
            t_lo, t_hi = time_fn(lo, h0), time_fn(hi, h0)
            per = ((t_hi - t_lo) / (args.iters_hi - args.iters_lo)) * 1e6
        except Exception as e:  # noqa: BLE001
            lead(f"{arm:>12} !! {type(e).__name__}: {e}")
            continue
        results[arm] = per
        base = results.get("xla")
        delta = f"{per - base:+9.2f}" if base and arm != "xla" else "        --"
        lead(f"{arm:>12} {per:>10.2f} {delta}")
        if args.profile_dir:
            profile_chain(hi, h0, arm)

    lead("\nRead the per-arm collective counts from the trace, not just the "
         "times: the xla arm must show ONE fused [2,3584]+[2,7168] op, or the "
         "probe is not exercising the combiner and the rest is meaningless.")

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
