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
"""Option B: test the linearity its shipping estimate assumes, and price bucketing.

Two questions, both at `max_model_len = 66560`.

(a) Is the estimate's linearity real?
-------------------------------------
`results.md` section 5c fits time against live width `w` in sublanes from
**two points per arm**:

    incumbent   1.92125 + 0.005595*w    from mixed_confound 2.27933 @ w=64
                                         and mixed_64k      4.78598 @ w=512
    B           0.16422 + 0.006996*w    from mixed_confound 0.61196 @ w=64
                                         and mixed_64k      3.74624 @ w=512

Its own text says the linearity is untested -- two points cannot test the form
they were fitted with -- and flags B's fitted slope coming out *steeper* than
the incumbent's (0.006996 vs 0.005595) as unexplained and mildly suspicious.
That whole section 5c bucketing table is derived from those two lines, so if the
form is wrong the table is wrong.

This measures **both arms** at six live `kv_len` values and refits. The two ends
of the ladder are not similar to section 5c's anchors, they **are** them:
`kv_len = 8192` reproduces `mixed_confound` field for field and `kv_len = 65536`
reproduces `mixed_64k`, so the four published numbers are a direct cross-check on
this run before any refit is read.

(b) What does bucketing actually cost?
--------------------------------------
B was measured with the driver passing the **exact** `kv_len` as the bucket,
which its own caveat 4 correctly calls a ceiling: a shipping version must round
the width up to one of a small set of jit-static buckets so the recompile count
stays bounded. So every live `kv_len` is also measured with the width rounded
**up** to the next bucket on two ladders -- powers of two, and 8192-granularity.

The claim under test, from section 5c: *coarsening the ladder costs only the
byte term and not the large half*, because every bucket at or below
`MAX_SLICE_WORDS = 65536` keeps `_pick_partition` at P = 1 while the incumbent's
`max_model_len`-derived width of 67584 is stuck at P = 2.

**That P claim is read back directly, not inferred from the timing.**
`impl.sparsecore_topk` is replaced, in each implementation's own module
namespace, by a shim that records the `(b, n, k)` actually handed to it and the
`P` that the real `sparsecore_topk._pick_partition` returns for that triple,
then delegates to the real function. The shim runs at trace time only and adds
nothing to any measured step.

Nothing in `instruments/` is edited. Both arms go through the read-only
`microbench.run_regime`, and the variant is adapted with `_rs_support.bind`
exactly as `variants/right-size/run_perf.py` does, so the timing loop, the input
build, `inner`, `reps` and `warmup` are the same code for every arm.

Sign convention: every percentage is a change in TIME, `(new - old)/old`.
Negative is faster. A `speedup x` column sits alongside and is a different
number (>1 is faster).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "variants", "right-size")))

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as SCT  # noqa: E402

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import microbench as MB  # noqa: E402

import _rs_support as RS  # noqa: E402

MAX_MODEL_LEN = 66560
MAX_NUM_SEQS = 6            # as in mixed_64k and mixed_confound
KV_LADDER = [8192, 16384, 24576, 32768, 49152, 65536]
POW2_BUCKETS = [8192, 16384, 32768, 65536]
GRAN8K_BUCKETS = [8192 * i for i in range(1, 9)]   # 8192 .. 65536

# section 5c's published two-point fits, for comparison only.
PUB = {"incumbent": (1.92125, 0.005595), "B": (0.16422, 0.006996)}
PUB_POINTS = {                     # (arm, w) -> published ms
    ("incumbent", 64): 2.27933, ("incumbent", 512): 4.78598,
    ("B", 64): 0.61196, ("B", 512): 3.74624,
}

_PICKS = []


def install_probe(impl):
    """Replace `impl.sparsecore_topk` with a recording shim. Returns the original.

    The shim reads the shape the kernel really passes and asks the real
    `_pick_partition` what it decides for it, so neither `b`, `n` nor `P` is
    inferred. Trace-time only.
    """
    real = impl.sparsecore_topk

    def shim(scores, k, row_lengths=None, **kw):
        b, n = scores.shape
        p = SCT._pick_partition(b, n, k)
        _PICKS.append(dict(b=int(b), n=int(n), k=int(k), P=int(p),
                           max_slice_words=SCT.MAX_SLICE_WORDS,
                           p_min=SCT._cdiv(int(n), SCT.MAX_SLICE_WORDS)))
        return real(scores, k, row_lengths=row_lengths, **kw)

    impl.sparsecore_topk = shim
    return real


def regime_for(kv_len: int) -> S.Regime:
    """The ladder rung. At kv_len 8192 this is `mixed_confound` field for field,
    and at 65536 it is `mixed_64k`; both are asserted below rather than
    asserted in prose."""
    return S.Regime(
        name=f"mixed_mml66560_kv{kv_len}",
        phase="mixed",
        max_num_seqs=MAX_NUM_SEQS,
        num_tokens=S.PROD_MAX_NUM_BATCHED_TOKENS,
        kv_len=kv_len,
        max_model_len=MAX_MODEL_LEN,
        note="bucket-ladder rung; identical construction to mixed_confound "
             "(kv 8192) and mixed_64k (kv 65536).",
    )


def _same_shape(a: S.Regime, b: S.Regime) -> bool:
    return (a.phase, a.max_num_seqs, a.num_tokens, a.kv_len,
            a.max_model_len) == (b.phase, b.max_num_seqs, b.num_tokens,
                                 b.kv_len, b.max_model_len)


def ceil_to(x, ladder):
    for v in ladder:
        if v >= x:
            return v
    return ladder[-1]


def b_width_cols(bucket: int) -> int:
    """The variant's own arithmetic, streamindex_topk_rs.py:757-758."""
    pps = S.pages_per_seq(MAX_MODEL_LEN)
    live = min(bucket, pps * S.PAGE_SIZE)
    return max(S.cdiv(live, 128), S.cdiv(S.INDEX_TOPK, 128)) * 128


def fit_linear(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return dict(a=a, b=b, r2=(1 - ss_res / ss_tot if ss_tot else float("nan")),
                rms_resid=(ss_res / n) ** 0.5,
                resid=[(x, y, a * x + b, y - (a * x + b))
                       for x, y in zip(xs, ys)],
                max_abs_resid=max(abs(y - (a * x + b))
                                  for x, y in zip(xs, ys)))


def measure(impl, regime, ee, reps, label, bucket, cols):
    del _PICKS[:]
    try:
        row = MB.run_regime(impl, regime, early_exit=ee, reps=reps,
                            measure_topk=False)
    except Exception as exc:  # noqa: BLE001
        row = dict(regime=regime.name, error=repr(exc)[:2000])
    row.update(arm=label, bucket=bucket, scores_cols=cols,
               live_sublanes=S.cdiv(regime.kv_len, 128),
               early_exit=ee, picks=list(_PICKS))
    print(json.dumps(row), flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default=",".join(str(x) for x in KV_LADDER))
    ap.add_argument("--reps", type=int, default=MB.REPS)
    ap.add_argument("--early-exit", default="true",
                    choices=["false", "true"],
                    help="section 5c was taken through run_perf.py, whose "
                         "default is true; keep it true to stay comparable")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    ee = (a.early_exit == "true")

    kvs = [int(x) for x in a.kv.split(",") if x]

    inc = RS.load_incumbent()
    var = RS.load_variant()
    install_probe(inc)
    install_probe(var)

    inc_cols = S.num_sublanes_total(MAX_MODEL_LEN) * 128
    print(f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
          f"num_tokens={S.PROD_MAX_NUM_BATCHED_TOKENS} early_exit={ee} "
          f"reps={a.reps}", flush=True)
    print(f"incumbent scores width (from max_model_len) = {inc_cols} cols "
          f"= {inc_cols // 128} sublanes, at every rung", flush=True)
    print(f"MAX_SLICE_WORDS = {SCT.MAX_SLICE_WORDS}", flush=True)

    # The two anchors are the published shapes, checked rather than claimed.
    for kv, other in ((8192, RS.CONFOUND), (65536,
                                            S.REGIMES_BY_NAME["mixed_64k"])):
        if kv in kvs:
            same = _same_shape(regime_for(kv), other)
            print(f"anchor check: kv={kv} rung is field-for-field "
                  f"{other.name}: {same}", flush=True)
            assert same, f"rung kv={kv} is not {other.name}"

    rows = []
    for kv in kvs:
        r = regime_for(kv)
        p2 = ceil_to(kv, POW2_BUCKETS)
        g8 = ceil_to(kv, GRAN8K_BUCKETS)
        print(f"\n--- kv_len={kv} (live {S.cdiv(kv,128)} sublanes); "
              f"buckets: exact={kv} pow2={p2} gran8k={g8} ---", flush=True)

        rows.append(measure(inc, r, ee, a.reps, "incumbent", None, inc_cols))

        # dedupe: at four of the six rungs the three buckets coincide.
        seen = {}
        for tag, bucket in (("exact", kv), ("pow2", p2), ("gran8k", g8)):
            if bucket in seen:
                seen[bucket].append(tag)
                continue
            seen[bucket] = [tag]
        for bucket, tags in seen.items():
            cols = b_width_cols(bucket)
            rows.append(measure(RS.bind(var, kv_len_bucket=bucket), r, ee,
                                a.reps, "B[" + "+".join(tags) + "]", bucket,
                                cols))

    def get(kv, pred):
        return next((x for x in rows
                     if x.get("kv_len") == kv and "error" not in x
                     and pred(x)), None)

    # ---- the direct P read-back -------------------------------------------
    print("\n=== (b) P read-back, direct from `_pick_partition` on the "
          "(b, n, k) the kernel actually passes ===")
    hdr = (f"{'kv_len':>8}{'arm':<16}{'bucket':>8}{'cols n':>9}{'b':>7}"
           f"{'k':>6}{'p_min':>7}{'P':>4}")
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        if "error" in x:
            continue
        if not x["picks"]:
            print(f"{x['kv_len']:>8}{x['arm']:<16}"
                  f"{str(x['bucket']):>8}{x['scores_cols']:>9}"
                  f"      -     -      -   -   (no trace recorded: the inner "
                  f"jit cache was hit; P is unchanged from the row above with "
                  f"the same n)")
            continue
        for p in x["picks"][:1]:
            print(f"{x['kv_len']:>8}{x['arm']:<16}{str(x['bucket']):>8}"
                  f"{p['n']:>9}{p['b']:>7}{p['k']:>6}{p['p_min']:>7}"
                  f"{p['P']:>4}")
    inc_ps = sorted({p["P"] for x in rows if x.get("arm") == "incumbent"
                     for p in x["picks"]})
    b_ps = sorted({p["P"] for x in rows
                   if str(x.get("arm", "")).startswith("B[")
                   for p in x["picks"]})
    print(f"\nincumbent P values observed: {inc_ps}   "
          f"B P values observed: {b_ps}")
    print("The section 5c claim is that the incumbent is stuck at P=2 at every "
          "width while every B bucket <= 65536 keeps P=1.")

    # ---- (b) the bucketing table ------------------------------------------
    print("\n=== (b) measured: B under each bucket ladder, vs the incumbent "
          "===")
    print("Delta time % = (B - incumbent) / incumbent. NEGATIVE IS FASTER.")
    hdr2 = (f"{'kv_len':>8}{'w':>5}{'ladder':<9}{'bucket':>8}{'B cols':>8}"
            f"{'inc ms':>10}{'inc sp%':>9}{'B ms':>10}{'B sp%':>8}"
            f"{'d time %':>10}{'speedup x':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    table = []
    for kv in kvs:
        ir = get(kv, lambda x: x["arm"] == "incumbent")
        if not ir:
            continue
        for tag, ladder in (("exact", None), ("pow2", POW2_BUCKETS),
                            ("gran8k", GRAN8K_BUCKETS)):
            bucket = kv if ladder is None else ceil_to(kv, ladder)
            br = get(kv, lambda x, _b=bucket: x["bucket"] == _b)
            if not br:
                continue
            old, new = ir["full_ms"], br["full_ms"]
            rec = dict(kv_len=kv, w=S.cdiv(kv, 128), ladder=tag,
                       bucket=bucket, b_cols=br["scores_cols"],
                       inc_ms=old, inc_spread=ir["full_spread_pct"],
                       b_ms=new, b_spread=br["full_spread_pct"],
                       d_time_pct=(new - old) / old * 100.0,
                       speedup_x=old / new)
            table.append(rec)
            print(f"{kv:>8}{rec['w']:>5}{tag:<9}{bucket:>8}"
                  f"{rec['b_cols']:>8}{old:>10.5f}{rec['inc_spread']:>9.2f}"
                  f"{new:>10.5f}{rec['b_spread']:>8.2f}"
                  f"{rec['d_time_pct']:>10.2f}{rec['speedup_x']:>11.3f}")
    print("\nA delta at or below the spread of the two arms it is a "
          "difference of cannot support a conclusion. Both spreads are in the "
          "row so that is checkable without leaving the table.")

    # ---- cross-check against the published anchors ------------------------
    print("\n=== cross-check: this run's two anchors against section 5c ===")
    hdr3 = f"{'arm':<12}{'w':>5}{'published ms':>14}{'this run ms':>13}{'d time %':>10}"
    print(hdr3)
    print("-" * len(hdr3))
    for (armname, w), pub in sorted(PUB_POINTS.items()):
        kv = w * 128
        if kv not in kvs:
            continue
        r = get(kv, (lambda x: x["arm"] == "incumbent") if armname ==
                "incumbent" else (lambda x, _k=kv: x["bucket"] == _k))
        if not r:
            continue
        d = (r["full_ms"] - pub) / pub * 100.0
        print(f"{armname:<12}{w:>5}{pub:>14.5f}{r['full_ms']:>13.5f}{d:>10.2f}")
    print("Negative is faster. This is a reproduction check across jobs, not "
          "an A/B; a few percent either way is drift, a large gap means the "
          "two runs are not measuring the same thing.")

    # ---- (a) the refit -----------------------------------------------------
    print("\n=== (a) refit of t against live width w in sublanes, six points "
          "per arm ===")
    fits = {}
    for armname, sel in (("incumbent", lambda x: x["arm"] == "incumbent"),
                         ("B", lambda x: x.get("bucket") == x.get("kv_len"))):
        pts = [(S.cdiv(kv, 128), get(kv, sel)) for kv in kvs]
        pts = [(w, r["full_ms"]) for w, r in pts if r]
        if len(pts) < 2:
            print(f"{armname}: too few points")
            continue
        f = fit_linear([p[0] for p in pts], [p[1] for p in pts])
        fits[armname] = f
        pb, pa = PUB["incumbent" if armname == "incumbent" else "B"]
        print(f"\n{armname}:  t = {f['b']:.5f} + {f['a']:.6f}*w   "
              f"(R^2 = {f['r2']:.5f}, rms resid = {f['rms_resid']*1e3:.1f} us, "
              f"max |resid| = {f['max_abs_resid']*1e3:.1f} us)")
        print(f"    section 5c two-point fit was: {pb:.5f} + {pa:.6f}*w")
        print(f"    {'w':>5}{'measured ms':>13}{'fitted ms':>12}"
              f"{'resid ms':>11}{'resid % of meas':>17}")
        for w, y, yh, rr in f["resid"]:
            print(f"    {w:>5}{y:>13.5f}{yh:>12.5f}{rr:>11.5f}"
                  f"{rr/y*100.0:>17.2f}")

    if "incumbent" in fits and "B" in fits:
        si, sb = fits["incumbent"]["a"], fits["B"]["a"]
        print(f"\nSLOPE COMPARISON -- the thing section 5c flagged as "
              f"suspicious:")
        print(f"  refitted incumbent slope {si:.6f} ms/sublane, "
              f"refitted B slope {sb:.6f} ms/sublane")
        print(f"  B steeper than the incumbent? {sb > si}   "
              f"(published two-point answer was True: 0.006996 > 0.005595)")
        print(f"  ratio B/incumbent = {sb/si:.3f} "
              f"(published ratio was {0.006996/0.005595:.3f})")

        # Redo section 5c's bucketing table from the refitted lines.
        print("\n=== section 5c's bucketing table, RECOMPUTED from the "
              "six-point refit -- `modelled`, not measured ===")
        print("Mean live context of a 65536-token prefill in 1024-token "
              "chunks is ~33280 tokens; section 5c's mean w per ladder is "
              "reused unchanged so only the fit changes.")
        hdr4 = (f"{'bucketing':<34}{'mean w':>8}{'B ms':>9}{'inc ms':>9}"
                f"{'d time %':>10}")
        print(hdr4)
        print("-" * len(hdr4))
        for label, mw in (("exact kv_len (a ceiling)", 260),
                          ("8192-token granularity (8 buckets)", 292),
                          ("powers of two (4 buckets)", 346)):
            bm = fits["B"]["b"] + fits["B"]["a"] * mw
            im = fits["incumbent"]["b"] + fits["incumbent"]["a"] * mw
            print(f"{label:<34}{mw:>8}{bm:>9.3f}{im:>9.3f}"
                  f"{(bm-im)/im*100.0:>10.1f}")
        print("Published section 5c figures for the same three rows were "
              "-41.3, -34.6, -23.4.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, table=table,
                           fits={k: {kk: vv for kk, vv in v.items()
                                     if kk != "resid"}
                                 for k, v in fits.items()},
                           inc_cols=inc_cols,
                           max_slice_words=SCT.MAX_SLICE_WORDS), f, indent=2)
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
