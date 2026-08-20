#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is tallax's top-k faster than XLA's at K3's MoE routing shape?

K3's router does `torch.topk(scores_for_choice, k=16, dim=-1)` over all 896
experts, once per MoE layer. XLA lowers that to a full **sort**, and the K3
profile bills it at:

    topk   92 ops/step   438.4 us/step   4.54-4.82 us/op   moe_routing.py:270

2.95% of the 14.90 ms decode period, spent sorting 896 elements to select 16.
A separate 296 us/step of `sort` (184 ops inside a `custom_kernel` scope) is the
expert-permutation sort feeding the grouped matmul -- a different problem, not
addressed here.

tallax (github.com/oliverdutton/tallax) is a Pallas top-k library for TPU. Its
headline numbers are for LLM sampling -- vocab 262K, k=64 -- which is three
orders of magnitude away from 896, so they do not transfer. The relevant cell of
its own table is `(b=16, 1024)` at k=16, which reports 2.8x, and two caveats
apply even to that: it is measured against `jax.lax.approx_max_k` rather than
the exact `top_k` XLA gives us, and at b=128 the same cell falls to 1.0x, so the
batch trend is unclear and b=1 -- our case -- is unmeasured.

Hence this probe. Arms:

  xla_topk      `jax.lax.top_k`. The baseline `torch.topk` actually lowers to.
  tallax_dnf    `tallax.tax.top_k`, divide-and-filter. Exact.
  tallax_bito   `tallax.tax.bitonic_top_k`. Exact.
  xla_approx    `jax.lax.approx_max_k`. Not a candidate (approximate, and it is
                the op with the known missed-element bug) but it is the baseline
                tallax's published table uses, so including it makes their 2.8x
                comparable with what we measure.

RESULT: tallax is slower than XLA at this shape, at every batch and both
dtypes. n=896, k=16, 92 chained calls, TPU7x, us per call:

    float32        b=1              b=8              b=64
    xla_topk       3.060            3.155            3.189
    tallax_dnf     5.552  (0.55x)   7.519  (0.42x)  12.493  (0.26x)
    tallax_bito    4.910  (0.62x)   6.060  (0.52x)   7.163  (0.45x)
    xla_approx     3.947  (0.78x)   4.163  (0.76x)   4.204  (0.76x)

bfloat16 is within 2% of float32 on every cell. All tallax arms match
`jax.lax.top_k`'s values exactly (max|dv| = 0), so they are correct, just
slower. `xla_approx` does *not* match (2.2e-2 at b=8, 2.7e-1 at b=64), an
independent reproduction of the approximation weakness tallax's README
describes.

Three things this says:

1. XLA's `top_k` is already good here and nearly flat in batch, 3.06 -> 3.19 us
   from b=1 to b=64. There is not much to win.
2. `approx_max_k` is *slower* than exact `top_k` at n=896 (0.76-0.83x). tallax's
   published 2.8x at `(b=16, 1024)` k=16 is measured against `approx_max_k`, so
   its baseline is one that loses to `top_k` at this size -- which is most of why
   that number does not transfer. It is not the whole story though: the residual
   gap is presumably v5e vs TPU7x, plus 896 padding to 1024, plus their kernel
   being timed standalone rather than 92-deep in a loop.
3. Divide-and-filter degrades with batch here (0.55x -> 0.26x), the opposite of
   the trend in their table. The algorithm's advantage is asymptotic in n: at 896
   with the auto-chosen 128 bins there are 7 elements per bin, far too few for
   binning to pay for itself. Their headline numbers are at vocab 262K.

Not pursued further, but the honest caveats on the negative result: `num_bins`,
`bins_topm_schedule` and `block_token` were left at their auto-computed values
(128 bins, schedule (4,5)) rather than tuned, and the divide-and-filter arm runs
only under the `pltpu.repeat` shim below. Neither seems likely to close a 1.6-3.8x
gap, but neither was ruled out.

Where tallax probably *would* pay in this stack is sampling, not routing. K3
currently samples with argmax (`sample_from_logits_argmax`, 10.4 us/step), but a
top-k/top-p sampler runs k~64 over a ~164K vocab -- exactly the regime the
library is built for and benchmarks against vLLM in.

One measurement note that matters more than the arms. A single top-k is ~5 us
against a host dispatch floor of ~1.1 ms, so timing one call measures dispatch
and nothing else. Every arm therefore runs 92 chained calls inside one jit --
mirroring K3's 92 MoE layers -- with the carry threaded through the returned
values so no call can be CSE'd or hoisted. The per-iteration perturbation is
identical across arms, so it cancels in the comparison.

Setup -- tallax is not vendored, clone it next to the repo and point
PYTHONPATH at it:

    git clone --depth 1 https://github.com/oliverdutton/tallax.git /tmp/tallax
    srun --jobid=<dev alloc> --overlap --ntasks=1 \
        env PYTHONPATH=/tmp/tallax .venv/bin/python -u \
        scripts/debug/probe_topk_tallax.py --dtype float32 --batches 1,8,64
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ARMS = ("xla_topk", "tallax_dnf", "tallax_bito", "xla_approx")


def _install_pltpu_compat() -> bool:
    """tallax pins jax==0.8.0; `pltpu.repeat` is gone by our 0.10.2.

    Its one use in the top-k path tiles a 128-wide array along the lane axis to
    reach the packed width, so concatenating copies is the same operation. This
    is patched here rather than in the vendored tree so it is unambiguously our
    shim, and so any result from the divide-and-filter arm is read with that
    caveat attached.
    """
    from jax.experimental.pallas import tpu as pltpu
    if hasattr(pltpu, "repeat"):
        return False
    pltpu.repeat = lambda x, repeats, axis: jnp.concatenate([x] * repeats,
                                                            axis=axis)
    return True


def _topk_fn(arm: str, k: int):
    from tallax import tax
    if arm == "xla_topk":
        return lambda x: lax.top_k(x, k)
    if arm == "xla_approx":
        return lambda x: lax.approx_max_k(x, k)
    if arm == "tallax_dnf":
        return lambda x: tax.top_k(x, k)
    if arm == "tallax_bito":
        return lambda x: tax.bitonic_top_k(x, k)
    raise ValueError(arm)


def build(arm: str, k: int, layers: int):
    f = _topk_fn(arm, k)

    def chain(x):

        def body(_, carry):
            out = f(carry)
            vals = out[0] if isinstance(out, (tuple, list)) else out
            # Thread the result back so the 92 calls cannot collapse into one.
            return carry + vals[..., :1].astype(carry.dtype)

        return lax.fori_loop(0, layers, body, x)

    return jax.jit(chain)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=896)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--layers", type=int, default=92)
    p.add_argument("--batches", default="1,8,64")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    dt = getattr(jnp, args.dtype)
    shimmed = _install_pltpu_compat()
    print(f"pltpu.repeat shim installed: {shimmed}")
    print(
        f"device={jax.devices()[0].device_kind}  n={args.experts} "
        f"k={args.k} layers={args.layers} dtype={args.dtype}",
        flush=True)

    for batch in [int(b) for b in args.batches.split(",")]:
        rng = np.random.default_rng(0)
        x0 = jnp.asarray(rng.standard_normal((batch, args.experts)), dtype=dt)
        print(f"\n=== batch={batch}  shape=({batch}, {args.experts}) ===",
              flush=True)
        ref = None
        rows = []
        for arm in args.arms.split(","):
            try:
                fn = build(arm, args.k, args.layers)
                for _ in range(args.warmup):
                    out = fn(x0)
                out.block_until_ready()
            except Exception as e:  # noqa: BLE001 -- report and continue
                msg = str(e).replace("\n", " ")[:150]
                print(f"  {arm:12s} FAILED: {type(e).__name__}: {msg}",
                      flush=True)
                continue
            s = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn(x0).block_until_ready()
                s.append((time.perf_counter() - t0) * 1e6)
            mn, med = float(np.min(s)), float(np.median(s))
            rows.append((arm, mn, med))
            # Exactness against the XLA baseline, on the top-k values.
            single = jax.jit(lambda z, a=arm: _topk_fn(a, args.k)(z))(x0)
            vals = single[0] if isinstance(single, (tuple, list)) else single
            vals = jnp.sort(vals.astype(jnp.float32), axis=-1)
            if arm == "xla_topk":
                ref = vals
                err = 0.0
            else:
                err = float(jnp.abs(vals - ref).max()) if ref is not None \
                    else float("nan")
            print(
                f"  {arm:12s} min={mn:9.1f} med={med:9.1f} us  "
                f"per_call={mn/args.layers:6.3f} us  max|dv|={err:.3e}",
                flush=True)
        if rows:
            base = next((m for a, m, _ in rows if a == "xla_topk"), None)
            if base:
                print("  -- speedup vs xla_topk (min) --", flush=True)
                for a, m, _ in rows:
                    print(
                        f"     {a:12s} {base/m:5.2f}x   "
                        f"delta={(base-m)/args.layers:+7.3f} us/layer",
                        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
