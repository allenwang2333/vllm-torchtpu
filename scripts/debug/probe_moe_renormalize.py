#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is MoE routing's ``renormalize`` sum worth 254 us/step, and can a fold fix it?

The barrier-free K3 profile (rank0, `results/iso-bar3-free`) accounts for every
microsecond of the 14.90 ms decode period. 697 us of it is device idle *inside*
the main graph -- gaps between ops, not op execution -- and the single largest
attributable family is:

    reduce -> fusion   254.3 us/step   92 occurrences   2.76 us each

xprof's framework-op and source-code planes pin all 644 of those gaps (92/step x
7 steps) on one line, `moe_routing.py:275`:

    topk_weights = topk_weights / torch.clamp(
        topk_weights.sum(dim=-1, keepdim=True), min=1e-20)

`.sum(dim=-1)` becomes a standalone `reduce` HLO and its consumer `fusion` sits
idle ~2.8 us waiting for it. On a `(1, 16)` tensor there is no work here at all
-- it is cross-lane reduction latency plus the sync that hands a scalar to the
next fusion. K3 runs it 92 times per step, once per MoE layer.

The hypothesis is that replacing the reduction with slices and adds keeps the
whole thing inside one elementwise fusion and never emits a `reduce`. That is a
hypothesis and not obviously true: at topk=16 a linear chain is 16 slices plus
15 adds, so if XLA does *not* fuse them the cure is worse than the disease. Hence
this probe rather than a direct edit -- an e2e A/B costs a 4-node slice and a
~25-minute compile, and this costs one device and two minutes.

Arms:

  sum     the current code. Control.
  fold    halving pairwise fold: (...,16) -> 8 -> 4 -> 2 -> 1. 4 adds, depth 4.
  chain   left-to-right accumulate. 15 adds, depth 15. Included because if
          `fold` wins and `chain` also wins, the mechanism is "no reduce HLO";
          if only `fold` wins, it is depth; if neither, the premise is wrong.
  late    keeps `.sum()` but moves the *division* past the expert combine.
          Sum_i (w_i/S) E_i == (1/S) Sum_i w_i E_i, so the reduce no longer
          blocks the dispatch -- it computes alongside the combine matmul and
          only its result is needed at the end. Removes the stall instead of
          shortening it. Differs from `sum` in bf16 rounding (divide after
          accumulate, not before), so it is a real but not free candidate.
  none    renormalize deleted outright. Not a candidate -- it changes the math --
          but it is the *upper bound* on what any rewrite can recover, so it
          tells us whether a winning arm captured the whole 254 us or a sliver.

Read the `us/layer` column against `none`. The prediction under test is that
`none` recovers ~2.8 us/layer and a working arm lands near it.

Shape realism: the routing tail's cost depends on topk (16) and not on the
expert count, since the reduce is over the gathered `(1, topk)` weights. Expert
count is taken from the pruned-K3 config (224) purely to keep the router matmul
honest; the real model's count changes the topk/sort cost, which is not what is
being measured and is identical across arms.

No collectives are involved -- every rank measures its own device and they
never talk -- but `ici-2x2x4` will not allocate a partial slice, and torch_tpu
cannot bring up a 4-host topology from one process. So this runs the standard
32-process layout and reports rank 0. Ranks are independent replicates, which is
free extra confidence in the result.

    sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_moe_renormalize.sbatch
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  -- registers the 'tpu' device

ARMS = ("sum", "fold", "chain", "late", "none")


def _fold_sum(w: torch.Tensor) -> torch.Tensor:
    """Sum the last dim by repeated halving. log2(k) depth, no `reduce` HLO.

    Reassociates, so ~1 ulp from `.sum()`; below bf16 resolution downstream.
    """
    n = w.shape[-1]
    while n > 1:
        half = n // 2
        if n % 2:
            # Odd width: add the aligned halves and carry the orphan lane.
            w = torch.cat(
                [w[..., :half] + w[..., half:2 * half], w[..., 2 * half:]],
                dim=-1)
            n = half + 1
        else:
            w = w[..., :half] + w[..., half:]
            n = half
    return w


def _chain_sum(w: torch.Tensor) -> torch.Tensor:
    """Sum the last dim left to right. k-1 depth, no `reduce` HLO."""
    total = w[..., 0:1]
    for i in range(1, w.shape[-1]):
        total = total + w[..., i:i + 1]
    return total


def _renorm(arm: str, w: torch.Tensor):
    """Return (weights, post_divisor). `post_divisor` is applied after combine."""
    if arm == "none":
        return w, None
    if arm in ("sum", "late"):
        total = w.sum(dim=-1, keepdim=True)
    elif arm == "fold":
        total = _fold_sum(w)
    elif arm == "chain":
        total = _chain_sum(w)
    else:
        raise ValueError(arm)
    total = torch.clamp(total, min=1e-20)
    if arm == "late":
        return w, total
    return w / total, None


def build(arm: str, routers, combs, biases, topk: int):
    """K3's classic top-k routing tail, once per layer, threaded through `h`.

    Each layer's router logits come from the running hidden state and the
    renormalized weights feed back into it, so the 92 tails cannot be CSE'd into
    one and cannot be hoisted out of the chain.
    """

    def f(h):
        for w_router, w_comb, bias in zip(routers, combs, biases):
            scores = (h @ w_router).float().sigmoid()  # noaux_tc
            _, topk_ids = torch.topk(scores + bias, k=topk, dim=-1)
            topk_weights = scores.gather(-1, topk_ids)
            topk_weights, post = _renorm(arm, topk_weights)
            combined = topk_weights.to(h.dtype) @ w_comb
            if post is not None:
                combined = combined / post.to(h.dtype)
            h = h + combined
        return h

    return f


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=92)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--experts", type=int, default=224)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--arms", default=",".join(ARMS))
    args = p.parse_args()

    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="tpu_dist",
                                rank=rank,
                                world_size=world_size)

    def lead(*a):
        if rank == 0:
            print(*a, flush=True)

    device = torch.device("tpu")
    dtype = torch.bfloat16
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) *
                scale).to(dtype).to(device)

    routers = [
        rnd(args.hidden, args.experts, scale=0.02) for _ in range(args.layers)
    ]
    combs = [
        rnd(args.topk, args.hidden, scale=0.02) for _ in range(args.layers)
    ]
    biases = [rnd(args.experts, scale=0.1).float() for _ in range(args.layers)]
    h0 = rnd(1, args.hidden)

    lead(f"layers={args.layers} topk={args.topk} experts={args.experts} "
         f"hidden={args.hidden} repeats={args.repeats}")

    results = {}
    outs = {}
    for arm in args.arms.split(","):
        fn = torch.compile(build(arm, routers, combs, biases, args.topk),
                           backend="tpu",
                           fullgraph=True,
                           dynamic=False)
        for _ in range(args.warmup):
            out = fn(h0).cpu()
        outs[arm] = out.float()
        samples = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn(h0).cpu()
            samples.append((time.perf_counter() - t0) * 1e6)
        results[arm] = (float(np.median(samples)), float(np.std(samples)))
        lead(f"  {arm:6s} median={results[arm][0]:9.1f} us  "
             f"sd={results[arm][1]:6.1f}")

    # Equivalence: `fold` must match `sum` to bf16 rounding, `late` may differ
    # slightly (divide after accumulate), `none` and `chain` are reference
    # points. A `fold` that does not match is a wrong answer, not a fast one.
    if "sum" in outs:
        ref = outs["sum"]
        scale = ref.abs().max().clamp(min=1e-30)
        lead("\nmax |arm - sum| relative to max|sum|:")
        for arm, o in outs.items():
            lead(f"  {arm:6s} {((o - ref).abs().max() / scale).item():.3e}")

    base = results.get("sum", (float("nan"), 0))[0]
    none = results.get("none", (float("nan"), 0))[0]
    lead(f"\n{'arm':6s} {'total us':>10} {'vs sum':>10} {'us/layer':>10} "
         f"{'% of headroom':>14}")
    headroom = base - none
    for arm in args.arms.split(","):
        med = results[arm][0]
        d = base - med
        pct = (100.0 * d / headroom) if headroom > 0 else float("nan")
        lead(f"{arm:6s} {med:10.1f} {d:+10.1f} {d/args.layers:+10.3f} "
             f"{pct:13.0f}%")
    lead(f"\nheadroom (sum - none) = {headroom:.1f} us "
         f"= {headroom/args.layers:.3f} us/layer")
    lead("profile said 2.76 us/layer of reduce->fusion gap")

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
