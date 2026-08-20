#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Can RMSNorm's variance reduce stop stalling its consumer, without a kernel?

Second item out of the K3 within-graph idle budget (697 us/step, of which the
MoE renormalize sum was the largest at 254 us and is now folded). This one:

    fusion [mean]  layernorm.py:18   dur 0.60      variance = x.pow(2).mean(-1)
      gap 0.730 / 0.736 / 0.736                    <- constant, ~320x per step
    fusion [mul]   layernorm.py:21                 consumer

212 us/step, and the constancy of the gap says fixed cross-lane-reduce result
latency rather than a data-dependent wait -- in one instance the `copy-start` it
appears to wait on was issued 17 us earlier, so the copy is not the blocker.

The MoE fix does not transfer. There the reduce was over topk=16 lanes and a
halving fold turned it into slices and adds. Here it is over 7168 elements,
which genuinely needs cross-lane reduction, and a 13-level fold would be worse
for the same reason `chain` lost there.

So the arms test ways to move the reduction off the VPU's cross-lane path
entirely, none of which need a Pallas kernel:

  ref        `x.pow(2).mean(dim=-1, keepdim=True)`. Control.
  matmul     `x @ x.mT / D`. A dot product on the MXU instead of a cross-lane
             VPU reduction -- different unit, different latency, and the MXU is
             idle here at batch 1.
  sublane    view (1, D) as (D/128, 128), sum the leading dim (elementwise
             across sublanes, no lane movement), then reduce the last 128.
             Trades one 7168-wide cross-lane reduce for one 128-wide one.
  twostage   like `sublane` but with the second stage as a matmul.
  none       variance replaced by a constant. Not a candidate -- wrong math --
             but the upper bound on what any rewrite can recover.

A Pallas kernel is the obvious other answer and is deliberately not an arm here:
it is a much larger change, and it should only be written if these cheap
rewrites fail *and* the bound says there is enough left to pay for a launch. The
pallas_call floor measured in K3's own graph is 0.37-1.0 us against the ~1.46 us
(0.60 mean + 0.73 stall + 0.13 mul) a kernel would replace, so the margin is
thin enough that the bound matters.

RESULT: all three rewrites are a dead end, and the probe itself is only
partly trustworthy. Keep both findings.

    norms=512, hidden=7168, 60 repeats
      ref       2121.5 us      control (run twice, 2123.6 / 2121.5)
      matmul    2209.9         -0.173 us/norm  (worse)
      sublane   2130.9         -0.018 us/norm  (noise)
      none      2116.5         +0.010 us/norm  <- headroom collapsed

`matmul`, `sublane` and `twostage` all produce **bit-identical** output to `ref`
*and* identical timing, which means XLA canonicalises them to the same HLO. The
reduction cannot be moved off the cross-lane path by rewriting it at the torch
level -- that closes the whole cheap-rewrite family, and is the durable result
here.

The headroom number, though, is invalid: `none` came out the same speed as
`ref`. At 512 chained norms this probe reads 512 x 7168 x 2 B = 7.3 MB of norm
weights per pass and becomes HBM-bandwidth-bound, so a 0.73 us scheduling stall
hides completely. At 32 norms the same probe reported 1.405 us/norm of headroom,
which is fixed-dispatch-dominated in the other direction. Neither N models K3,
where the norms are interleaved with matmuls and attention -- and a stall that
only appears when the scheduler has nothing else to issue cannot be reproduced
by a chain of norms with no neighbours. This is the exact failure
`fusion_probe.py` documents at the top of its own docstring.

If the stall is ever attacked again, the probe has to reproduce K3's
neighbourhood, not the norm in isolation.

Superseded in priority: the 186 of these 513 norms that sit inside
`AttentionResidual` are part of a block costing ~1.1 ms/step, where the norm is
a minor term next to a 13-element softmax billed at 1.33 us x 372/step. Fuse
that block rather than the norm.

Run:
    srun --jobid=<dev alloc> --overlap --ntasks=1 \\
        env PYTHONPATH=$PWD/src .venv/bin/python -u \\
        scripts/debug/probe_rmsnorm_variance.py
"""
import argparse
import os
import time

import numpy as np
import torch
import torch_tpu  # noqa: F401  -- registers the 'tpu' device

ARMS = ("ref", "matmul", "sublane", "twostage", "none")


def _variance(arm: str, x: torch.Tensor, ones, d: int):
    """Mean of squares along the last dim, shaped (..., 1)."""
    if arm == "ref":
        return x.pow(2).mean(dim=-1, keepdim=True)
    if arm == "matmul":
        return (x @ x.transpose(-1, -2)) / d
    if arm == "sublane":
        return x.view(x.shape[0], -1, 128).pow(2).sum(dim=1).sum(
            dim=-1, keepdim=True) / d
    if arm == "twostage":
        partial = x.view(x.shape[0], -1, 128).pow(2).sum(dim=1)
        return (partial @ ones) / d
    raise ValueError(arm)


def build(arm: str, weights, biases, eps: float, d: int, ones):
    """N chained RMSNorms at K3's hidden size, threaded so none can be CSE'd."""

    def f(h):
        for w, bias in zip(weights, biases):
            x = h.float()
            if arm == "none":
                var = torch.ones_like(x[..., :1])
            else:
                var = _variance(arm, x, ones, d)
            x = x * torch.rsqrt(var + eps)
            h = (x.to(w.dtype) * w).to(h.dtype) + bias
        return h

    return f


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--norms", type=int, default=512)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--arms", default=",".join(ARMS))
    args = p.parse_args()

    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024

    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("tpu")
    dtype = torch.bfloat16
    d = args.hidden
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) *
                scale).to(dtype).to(device)

    weights = [rnd(d, scale=0.05) for _ in range(args.norms)]
    biases = [rnd(1, d, scale=0.01) for _ in range(args.norms)]
    ones = torch.ones(128, 1, dtype=torch.float32).to(device)
    h0 = rnd(1, d)

    def lead(*a):
        if rank == 0:
            print(*a, flush=True)

    lead(f"norms={args.norms} hidden={d} repeats={args.repeats}")

    results, outs = {}, {}
    for arm in args.arms.split(","):
        fn = torch.compile(build(arm, weights, biases, 1e-5, d, ones),
                           backend="tpu",
                           fullgraph=True,
                           dynamic=False)
        for _ in range(args.warmup):
            out = fn(h0).cpu()
        outs[arm] = out.float()
        s = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn(h0).cpu()
            s.append((time.perf_counter() - t0) * 1e6)
        results[arm] = (float(np.median(s)), float(np.std(s)))
        lead(f"  {arm:9s} median={results[arm][0]:9.1f} us  "
             f"sd={results[arm][1]:6.1f}")

    if "ref" in outs:
        ref = outs["ref"]
        scale = ref.abs().max().clamp(min=1e-30)
        lead("\nmax |arm - ref| relative to max|ref|:")
        for arm, o in outs.items():
            lead(f"  {arm:9s} {((o - ref).abs().max() / scale).item():.3e}")

    base = results.get("ref", (float("nan"), 0))[0]
    none = results.get("none", (float("nan"), 0))[0]
    headroom = base - none
    lead(f"\n{'arm':9s} {'total us':>10} {'vs ref':>9} {'us/norm':>9} "
         f"{'% headroom':>11}")
    for arm in args.arms.split(","):
        med = results[arm][0]
        delta = base - med
        pct = (100.0 * delta / headroom) if headroom > 0 else float("nan")
        lead(f"{arm:9s} {med:10.1f} {delta:+9.1f} {delta/args.norms:+9.3f} "
             f"{pct:10.0f}%")
    lead(f"\nheadroom (ref - none) = {headroom:.1f} us = "
         f"{headroom/args.norms:.3f} us/norm")
    lead("profile said 0.73 us/norm of post-reduce stall, ~320 norms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
