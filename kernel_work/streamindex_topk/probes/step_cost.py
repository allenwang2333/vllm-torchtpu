# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probe: is there a fixed cost per INNER LOOP ITERATION, and how big is it?

Why this probe exists
---------------------
Step 1's roofline gate says `decode_prod` is at 55% of its modelled floor,
inside the 50-80% band where `write-tpu-kernels` says a kernel is worth
writing. That verdict divides bytes by a rate and carries no per-step term.

`references/03-pipelining.md` prescribes a different floor -- `sum_steps [512 ns
+ bytes_step / 3.48e12]` -- and says in as many words that a floor quoted
without a step count understates a kernel whose steps are small. The 512 ns was
fitted on `pallas_call` GRID steps [verified: j-5855c2a5-aef4-4a61-b161]. This
kernel's iteration is not a grid step: it is a `lax.fori_loop` inside one kernel
body (streamindex_topk.py:418-420), and each iteration hand-rolls its own DMAs.

Which unit the 512 ns attaches to changes the verdict, and it changes it across
the line that ends the exercise:

    unit of the fixed term      steps   floor      achieved
    no step term (report.md)      --    41.7 us      55%   -> write a kernel
    pallas_call grid step         16    46.8 us      61%   -> write a kernel
    inner (bq, bkv) iteration     64    71.3 us      93%   -> DO NOT
    individual page DMA          512   300.6 us      refuted by the 76.42 us

Two briefers flagged the extrapolation independently and neither would settle
it, correctly: it is a measurement.

The design, and why it is single-variable
-----------------------------------------
Sweep `bkv_p` (KV pages per block) at a decode shape, holding EVERYTHING the
competing explanations depend on constant. With `max_model_len = 8192` the
sequence is exactly 8 pages, so:

    bkv_p           1     2     4     8
    num_bkv         8     4     2     1     <- inner iterations per grid step
    inner total   128    64    32    16     <- x16 grid steps
    page DMAs     512   512   512   512     <- num_bkv * seq_batch * bkv_p * grid
    KV bytes    134.2 134.2 134.2 134.2 MB  <- num_bq = 1, so read exactly once
    grid steps     16    16    16    16
    sublanes       64    64    64    64     <- align_to(8, bkv_p) == 8 for all
    einsum FLOPs  same  same  same  same

The page-DMA count is constant because `_fetch_bkv` issues `bkv_p` separate
page DMAs per fetch (streamindex_topk.py:177-194) and the fetch count falls by
the same factor the per-fetch DMA count rises. That is what makes this probe
able to do what neither briefer could: a per-DMA fixed cost contributes a
CONSTANT here and cannot produce a slope, while a per-inner-step fixed cost
produces a slope of exactly F per iteration.

    time(arm) = F * inner_total + (everything else, constant)

So the fitted slope IS the per-inner-iteration fixed cost. If F is near 512 ns,
128 -> 16 iterations saves 112 * 512 ns = 57 us of a 76 us kernel and is
unmissable. If F is near zero the four arms are flat and report.md's gate
stands.

`max_model_len = 8192` is NOT the production 9216. That is deliberate: at 9216
the sequence is 9 pages and `align_to(9, bkv_p)` moves 80 -> 96 -> 128 sublanes
across the ladder, growing the scores buffer 60% and confounding the slope with
scores traffic. 8192 pins it. The arm at `bkv_p = 2` is therefore comparable to
`decode_prod` in structure but not equal to it in absolute time, and no number
here should be quoted as a `decode_prod` measurement.

Second result, for free: `bkv_p` is on the alternative list of three separate
area briefers (partitioning B1, pipelining 8/9). Whatever the slope turns out
to be, this is also a measurement of that candidate.

Sign convention: percentages are changes in TIME, negative is faster.
"""

import argparse
import json
import statistics
import sys

import jax

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments.microbench import _time  # noqa: E402

LADDER = [1, 2, 4, 8]
REPS = 10
INNER = 20


def probe_regime(max_num_seqs: int) -> S.Regime:
    """The ladder's shape at a chosen batch size.

    `max_num_seqs` is varied ONLY to cross-check the fit: it scales the grid
    (`max_num_seqs // decode_req_batch_size`) and therefore scales
    `inner_total` at every rung by the same factor. A per-inner-iteration cost
    F must come back the same from both runs; a fit that only reproduces at one
    batch size is fitting something else.
    """
    return S.Regime(
        name=f"decode_8k_flat_ms{max_num_seqs}",
        phase="decode",
        max_num_seqs=max_num_seqs,
        num_tokens=max_num_seqs,
        kv_len=8192,
        max_model_len=8192,
        note="probe-only shape. 8192 (not the production 9216) so that "
             "num_sublanes_total is 64 at every bkv_p on the ladder; at 9216 "
             "it would move 80->96->128 and confound the step-count slope "
             "with scores traffic.",
    )


def arm(impl, bkv_p: int, d, regime: S.Regime, reps=REPS):
    """One point on the ladder. `d` is shared so inputs are identical."""
    k = S.INDEX_TOPK
    call = jax.jit(lambda q, w, c, s, p, cu, dist: impl.streamindex_topk(
        q, w, c, s, p, cu, dist,
        k=k,
        compression_ratio=S.COMPRESSION_RATIO,
        # Only entry [0] is reached at decode (streamindex_topk.py:757). The
        # other two are held at 2 so num_sublanes_total takes its max over a
        # set whose align_to(8, .) is 8 either way.
        num_kv_pages_per_block=(bkv_p, 2, 2),
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    ))
    args = (d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
            d["page_indices"], d["cu_q_lens"], d["distribution"])
    ts = _time(call, *args, reps=reps, inner=INNER)
    med = statistics.median(ts)
    spread = (max(ts) - min(ts)) / med * 100.0

    num_bkv = max(1, S.cdiv(regime.kv_len, S.PAGE_SIZE * bkv_p))
    grid = regime.max_num_seqs // S.DECODE_REQ_BATCH_SIZE
    inner_total = num_bkv * grid
    page_dmas = num_bkv * S.DECODE_REQ_BATCH_SIZE * bkv_p * grid
    bkv_vmem_mib = 2 * S.DECODE_REQ_BATCH_SIZE * bkv_p * S.PAGE_SIZE * 256 / 2**20

    return dict(
        bkv_p=bkv_p,
        num_bkv=num_bkv,
        grid_steps=grid,
        inner_total=inner_total,
        page_dmas=page_dmas,
        sublanes=S.num_sublanes_total(regime.max_model_len,
                                      bkv_ps=(bkv_p, 2, 2)),
        bkv_scratch_MiB=bkv_vmem_mib,
        full_ms=med * 1e3,
        spread_pct=spread,
    )


def fit(rows):
    """Least-squares slope of full_ms against inner_total.

    Two unknowns, four points: `t = F * inner_total + C`. F is the per-inner-
    iteration fixed cost this probe exists to measure. Reported in ns.
    """
    xs = [r["inner_total"] for r in rows]
    ys = [r["full_ms"] * 1e6 for r in rows]  # ns
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    # R^2, so a scattered fit cannot be quoted as a clean linear one.
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl",
                    default="vllm_torchtpu.kernels.deepseek_v4.streamindex_topk")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="scales the grid, hence inner_total at every rung. "
                         "Cross-check only -- F must not depend on it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import importlib
    impl = importlib.import_module(a.impl)

    # Built ONCE and shared by every arm: the arms must differ in `bkv_p` and
    # in nothing else, including the random page permutation.
    regime = probe_regime(a.max_num_seqs)
    print(f"regime: {regime.name}  kv_len={regime.kv_len} "
          f"max_model_len={regime.max_model_len} "
          f"max_num_seqs={regime.max_num_seqs} seed={a.seed}", flush=True)
    d = I.build(regime, seed=a.seed)

    rows = []
    for bkv_p in LADDER:
        try:
            r = arm(impl, bkv_p, d, regime, reps=a.reps)
        except Exception as exc:  # noqa: BLE001
            r = dict(bkv_p=bkv_p, error=repr(exc)[:400])
        rows.append(r)
        print(json.dumps(r), flush=True)

    ok = [r for r in rows if "error" not in r]
    print(f"\n=== step-cost ladder ({regime.name}, kv 8192, "
          f"max_model_len 8192) ===")
    hdr = (f"{'bkv_p':>6}{'num_bkv':>9}{'inner':>7}{'pageDMA':>9}{'subl':>6}"
           f"{'bkvMiB':>8}{'full_ms':>10}{'spread%':>9}{'d time %':>10}")
    print(hdr)
    print("-" * len(hdr))
    base = next((r["full_ms"] for r in ok if r["bkv_p"] == 2), None)
    for r in rows:
        if "error" in r:
            print(f"{r['bkv_p']:>6}  ERROR {r['error'][:70]}")
            continue
        dt = ((r["full_ms"] - base) / base * 100.0) if base else float("nan")
        print(f"{r['bkv_p']:>6}{r['num_bkv']:>9}{r['inner_total']:>7}"
              f"{r['page_dmas']:>9}{r['sublanes']:>6}"
              f"{r['bkv_scratch_MiB']:>8.1f}{r['full_ms']:>10.4f}"
              f"{r['spread_pct']:>9.2f}{dt:>10.2f}")
    print("\n`d time %` is against the bkv_p=2 arm, the shipped setting. "
          "Negative is faster.")
    print("Page DMAs and KV bytes are constant across arms BY CONSTRUCTION, "
          "so a per-DMA fixed cost cannot produce a slope here; only a "
          "per-inner-iteration cost can.")

    if len(ok) >= 3:
        slope, intercept, r2 = fit(ok)
        print(f"\nfit: full_time_ns = {slope:.1f} * inner_total + "
              f"{intercept:.0f}   (R^2 = {r2:.4f})")
        print(f"  => per-inner-iteration fixed cost F = {slope:.1f} ns")
        print("  Reference's grid-step term, for comparison: 512 ns "
              "[verified: j-5855c2a5-aef4-4a61-b161]")
        print("\nReading it:")
        print("  F near 512 ns  -> the inner-step reading is right, decode is "
              "near 93% of its floor, and the roofline gate FAILS.")
        print("  F near 0       -> the fixed term does not attach to the "
              "fori_loop iteration; report.md's 55% stands.")
        print("  in between     -> quote F, recompute the floor with it, and "
              "say which side of 80% the result lands on.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, regime=regime.name,
                           max_num_seqs=a.max_num_seqs, seed=a.seed),
                      f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
