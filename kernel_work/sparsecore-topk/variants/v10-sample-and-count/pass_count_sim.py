"""v10 (sample-and-count): how many count passes does the correction loop need?

`design.md` §5 records the v10 pass count as the node's one open term and as
**host-simulable -- no job needed**.  This is that simulation.

What is being counted
---------------------
A *count pass* is one full walk of the row's live keys that evaluates
`count(key >= T)` for one or more scalar thresholds `T` and returns the counts
in scalar registers.  It performs no memory scatter, which is why
`probes/15-mandatory-v3-v10` measured it at 0.191x the dense L0 histogram.  The
open question is how many of them the correction loop needs before it can hand
the emit pass a threshold it is allowed to emit against.

Why the loop cannot stop early
------------------------------
The emit pass may only emit `key > T` plus a quota drawn from `key == T`
(`sparsecore_topk.py` `emit_body`).  Filling a quota out of a *band* of
distinct values would return the wrong set, so the correction loop must
converge on a `T` with `count(> T) <= k <= count(>= T)` -- i.e. on the exact
k-th order statistic.  There is no approximate stopping point that stays exact.
That is `design.md` §9 Plan E risk row (c) and it is why this file simulates
convergence, not a fixed iteration budget.

Distributional assumption -- NAMED, not silently chosen
-------------------------------------------------------
Bisection cost is `log2(bracket_width / gap)` in monotone-int32 key space, so it
is a property of the *score distribution*, not of the length distribution.  A
run at rho=1 on `standard_normal` would measure a shape production never sees.

This file therefore reuses `probes/13-level-count-refit/level_sim.py`'s score
generator verbatim -- `make_kv` + `make_scores`, which reproduce
`streamindex_topk.py:326-338` (fp8 q/k, relu, signed bf16 head weights, sum over
64 heads, kscale) -- together with its `prefill_lengths` (E/n = 0.40005) and
`decode_lengths` (E/n = 0.850).  That generator was validated in probes/13
against probes/05's published table.  `alpha=0.0` and 4 replicate KV draws
follow probes/13 STEP 4.

`--dist` re-runs the whole thing on the unvalidated synthetic rows
(`random` / `uniform` / `heavy_ties`, rho=1) so the sensitivity of the answer to
the assumption is on the record rather than assumed away.

Self-validation
---------------
`--validate` runs the identical search machinery at ARITY=256 starting from the
full key range, which *is* the incumbent's radix walk (L0 over the top 8 bits,
then 8-bit refinement levels).  It must reproduce probes/13's level counts.  If
it does not, nothing else this file prints is worth reading.

Aggregation
-----------
Kernel cost is set by the critical subcore, not the mean row: `_crit` from
level_sim (max over subcores of the sum of its own rows -- there is no
inter-subcore barrier).  Per-row correction work is `passes * eff_len`, so the
figure that prices the arm is
    P_eff = crit(passes * eff) / crit(eff)
which is the correction cost expressed in units of one full-density count pass,
i.e. directly multipliable by the 6.48 us/pass of `design.md` §4a.
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_L13 = os.path.abspath(
    os.path.join(_HERE, "..", "..", "probes", "13-level-count-refit",
                 "level_sim.py"))

_spec = importlib.util.spec_from_file_location("level_sim", _L13)
LS = importlib.util.module_from_spec(_spec)
sys.modules["level_sim"] = LS
_spec.loader.exec_module(LS)

K, N = LS.K, LS.N
KV_ROWS = LS.KV_ROWS
SEED = LS.SEED
HEAD_DIM, N_HEAD = LS.HEAD_DIM, LS.N_HEAD
NSUB = LS.NUM_SUBCORES_TOTAL

KEY_MIN = -(1 << 31)
KEY_SUP = 1 << 31  # one past the largest int32 key; count(>= KEY_SUP) == 0
LANES = 16
SAMPLE_STRIDE = 16  # every 16th VECTOR, i.e. 1/16 of the row


# ------------------------------------------------------------------ search --
def _sample_bracket(keys_sorted, samp_keys, k, eff):
    """L0 sample histogram -> the bucket [lo, hi) the k-th key is *estimated*
    to live in.  Unverified: the correction loop probes both edges."""
    s = samp_keys.size
    if s == 0:
        return KEY_MIN, KEY_SUP
    # target rank within the sample, scaled by the sampling fraction
    ks = int(min(max(1, round(k * s / eff)), s))
    buck = (samp_keys >> 24).astype(np.int64)  # -128 .. 127
    # cumulative count at or above each bucket's lower edge, walked from the top
    order = np.sort(buck)
    # count(bucket >= b) for the smallest b with count >= ks
    uniq, cnt = np.unique(order, return_counts=True)
    above = np.cumsum(cnt[::-1])[::-1]  # above[i] = #samples with bucket >= uniq[i]
    hit = np.flatnonzero(above >= ks)
    b = int(uniq[hit[-1]]) if hit.size else int(uniq[0])
    return b << 24, (b + 1) << 24


def _passes_for_row(keys_sorted, samp_keys, k, eff, arity, rule="bisect"):
    """Number of count passes to converge on an emittable threshold.

    Invariant: cnt(>= LO) >= k > cnt(>= HI).  Converged when some probed T has
    cnt(>= T) == k (emit `>= T`, no quota), or when HI == LO + 1 (emit `> LO`
    plus a quota from `== LO`, which is exactly the incumbent's emit).
    """
    n = keys_sorted.size

    def cnt(t):
        # keys_sorted ascending; count of keys >= t
        return n - int(np.searchsorted(keys_sorted, t, side="left"))

    LO, HI = KEY_MIN, KEY_SUP
    cLO, cHI = n, 0
    if cLO == k:
        return 0
    passes = 0
    slots = max(1, arity - 1)

    lo0, hi0 = _sample_bracket(keys_sorted, samp_keys, k, eff)
    # The sample bracket is an ESTIMATE.  Both edges are probed before it is
    # trusted; if either fails, the invariant simply keeps the wider bound and
    # the search degrades to bisection of the surviving interval.
    seeds = [hi0, lo0]  # popped from the end: lo0 first
    if samp_keys.size == 0:
        seeds = []  # validation mode: no sample, pure radix walk from scratch
    elif slots > 2:
        extra = slots - 2
        w = hi0 - lo0
        seeds = ([lo0 + (j * w) // (extra + 1) for j in range(extra, 0, -1)]
                 + seeds)

    while True:
        if HI - LO <= 1:
            return passes
        cands = []
        while seeds and len(cands) < slots:
            s = int(seeds.pop())
            if LO < s < HI:
                cands.append(s)
        if not cands:
            if rule == "illinois" and passes % 2 == 0:
                # false position on the count.  Every other pass is a plain
                # bisection, which bounds this at 2x bisection --
                # unguarded regula falsi stalls on a step function, and count
                # is one.
                span = HI - LO
                dlo, dhi = float(cLO - k), float(k - cHI)
                t = (LO + int(span * dlo / (dlo + dhi))
                     if dlo + dhi > 0 else LO + span // 2)
                cands = [min(max(t, LO + 1), HI - 1)]
            else:
                cands = [LO + (j * (HI - LO)) // arity for j in range(1, arity)]
        cands = sorted({int(c) for c in cands if LO < int(c) < HI})
        if not cands:
            return passes
        passes += 1
        for c, cc in ((c, cnt(c)) for c in cands):
            if cc == k:
                return passes
            if cc > k:
                if c > LO:
                    LO, cLO = c, cc
            else:
                if c < HI:
                    HI, cHI = c, cc
        if passes > 96:
            raise RuntimeError("correction loop did not converge")


def _samp_positions(eff):
    """Element positions the sample pass visits: every SAMPLE_STRIDE-th vector.

    A systematic cluster sample over the whole row, not a contiguous prefix --
    scores are attention scores over KV positions and carry positional
    structure, so a contiguous 1/16 block would be biased.
    """
    nvec = -(-eff // LANES)
    v = np.arange(0, nvec, SAMPLE_STRIDE, dtype=np.int64)
    p = (v[:, None] * LANES + np.arange(LANES, dtype=np.int64)[None, :]).ravel()
    return p[p < eff]


# ------------------------------------------------------------------- case ---
def run_case(scores, lengths, k, arities, rule="bisect", full_range=False):
    b, n = scores.shape
    waves = b / NSUB
    eff = np.minimum(np.asarray(lengths, np.int64), n)
    out = {}
    keys_all = LS.monotone_key(scores.view(np.int32))
    for arity in arities:
        pv = np.zeros(b, np.float64)
        keep = 0
        for r in range(b):
            e = int(eff[r])
            kr = keys_all[r, :e]
            if e <= k:
                keep += 1
                continue
            ks = np.sort(kr)
            if full_range:
                # incumbent-equivalent: no sample, start from the full range
                sk = np.zeros(0, np.int32)
            else:
                sk = kr[_samp_positions(e)]
            pv[r] = _passes_for_row(ks, sk, k, e, arity, rule)
        efff = eff.astype(np.float64)
        work = pv * efff
        out[arity] = {
            "passes": pv,
            "keep_all": keep / b,
            "mean": float(pv.mean()),
            "mean_walking": float(pv[pv > 0].mean()) if (pv > 0).any() else 0.0,
            "p50": float(np.percentile(pv, 50)),
            "p90": float(np.percentile(pv, 90)),
            "p99": float(np.percentile(pv, 99)),
            "max": float(pv.max()),
            "P_eff": float(LS._crit(work, waves) / LS._crit(efff, waves)),
            "P_eff_mean": float(work.sum() / efff.sum()),
            "hist": np.bincount(pv.astype(np.int64)).tolist(),
        }
    return out


def _prod_scores(b, phase, rep):
    rng = np.random.default_rng(SEED + 1000 * rep)
    kq, ks = LS.make_kv(KV_ROWS, HEAD_DIM, 0.0, rng)
    sc = np.full((b, N), -np.inf, np.float32)
    sc[:, :KV_ROWS] = LS.make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)
    lens = (LS.prefill_lengths(b, rng)
            if phase == "prefill" else LS.decode_lengths(b, rng))
    return sc, lens


def _fmt(tag, s):
    return (f"{tag:<30} mean={s['mean']:6.3f}  walking={s['mean_walking']:6.3f}"
            f"  p50={s['p50']:5.1f} p90={s['p90']:5.1f} p99={s['p99']:5.1f}"
            f" max={s['max']:5.1f}  P_eff={s['P_eff']:6.3f}"
            f"  keep_all={s['keep_all']*100:4.1f}%")


# ------------------------------------------------------------------ prices --
# design.md §4a per-op price sheet, harness denominator 189.98 us, already
# rho-weighted and crit-aggregated.  Quoted, not re-derived here.
HARNESS = 189.98
HIST0 = 33.91      # dense L0 scatter
REF_HIST = 20.08   # refinement scatters, 1.570 crit levels
SWEEPS = 15.63     # zero_priv + fold_priv + scan_glob + find_bucket, ALL levels
CNT_PASS = 0.191 * HIST0          # probes/15 Probe A: cnt/scat = 20.681/108.217
SWEEP_PER_LEVEL = SWEEPS / 2.570  # incumbent pays 1 + 1.570 levels of sweeps
SCALE_PROD = 1.219                # harness -> production, design.md

# --- marginal cost of the 2nd..m-th threshold of one count pass -------------
# Measured statically from the AOT bundle dumps (`bundle_stats.py`), in the
# PRODUCTION kernel at the production shape -- not in a toy.  The innermost
# count-pass loop body, at a fixed unroll of 4 (so 4 vectors per iteration):
#     arity 2  (1 threshold)  .LBB2_50 .. .Ltmp27   11 bundles
#     arity 4  (3 thresholds) .LBB2_63 .. .Ltmp27   30 bundles
# b(m) = a + c*m  =>  c = 9.5, a = 1.5 bundles per 4 vectors.  A count pass is
# issue-bound (probes/15 put it at the read-only walk floor), so scheduled
# bundles are the right currency and one extra threshold costs c/(a+c) of a
# whole one-threshold pass.
BUNDLES_1THR = 11.0
BUNDLES_3THR = 30.0
_C = (BUNDLES_3THR - BUNDLES_1THR) / 2.0          # 9.5 bundles per threshold
_A = BUNDLES_1THR - _C                            # 1.5 bundles of walk overhead
MARGINAL_FRAC = _C / BUNDLES_1THR                 # 0.864 of a 1-threshold pass
MARGINAL_PER_THRESH = MARGINAL_FRAC * CNT_PASS    # 5.59 us, rho-weighted
# arity 16 is measurably SUBLINEAR: .LBB2_34 .. .Ltmp26 is 60 bundles for 2
# vectors x 15 thresholds = 120 per 4 vectors, against a linear 1.5+9.5*15=144.
# Used below as the optimistic bound, never as the headline.
MARGINAL_PER_THRESH_SC16 = ((120.0 - _A) / 15.0 / BUNDLES_1THR) * CNT_PASS


def price(P_eff, extra_thresh=0.0, marginal_per_thresh=MARGINAL_PER_THRESH):
    """v10 whole-kernel cost at the harness denominator.

    Removed : hist0 + refinement hist + all but ONE level of V4 sweeps.
    Added   : a 1/16 sample scatter + one level of sweeps + P_eff count passes.
    """
    removed = HIST0 + REF_HIST + SWEEPS
    sample = HIST0 / 16.0 + SWEEP_PER_LEVEL
    corr = P_eff * (CNT_PASS + extra_thresh * marginal_per_thresh)
    tot = HARNESS - removed + sample + corr
    return tot, (tot - HARNESS) / HARNESS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dist", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ARITIES = (2, 4, 8, 16)

    print("=" * 100)
    print("v10 sample-and-count: correction-pass count")
    print(f"score generator = probes/13 level_sim.make_scores (alpha=0.0), "
          f"{args.reps} reps, seed {SEED}")
    print(f"k={K} n={N} sample=1/{SAMPLE_STRIDE} of the row, "
          f"systematic over vectors")
    print("=" * 100)

    if args.validate:
        print("\n--- VALIDATION: same search machinery at ARITY=256 from the "
              "full 32-bit range")
        print("    == the incumbent's radix walk.  Must reproduce probes/13 "
              "(prefill crit 1.570 refinement levels -> 2.570 total).")
        for phase, b in (("prefill", 1024), ("decode", 64)):
            acc, ref = [], []
            for rep in range(args.reps):
                sc, ln = _prod_scores(b, phase, rep)
                acc.append(run_case(sc, ln, K, (256,), full_range=True)[256])
                ref.append(LS.run_case(sc, ln, K, 256))
            m = float(np.mean([a["mean_walking"] for a in acc]))
            c = float(np.mean([a["P_eff"] for a in acc]))
            rm = float(np.mean([x["mean_walking"] for x in ref]))
            rc = float(np.mean([x["crit"] for x in ref]))
            print(f"    {phase:<8} b={b:<5} refinement levels  "
                  f"this file: mean(walking)={m - 1:.3f} crit={c - 1:.3f}   "
                  f"level_sim: mean(walking)={rm:.3f} crit={rc:.3f}")
        print()

    results = {}
    for phase, b in (("prefill", 1024), ("decode", 64)):
        print(f"\n### {phase}  b={b}  "
              f"rho={'0.40005' if phase == 'prefill' else '0.850'}")
        per = {a: [] for a in ARITIES}
        ill = []
        for rep in range(args.reps):
            sc, ln = _prod_scores(b, phase, rep)
            r = run_case(sc, ln, K, ARITIES)
            for a in ARITIES:
                per[a].append(r[a])
            ill.append(run_case(sc, ln, K, (2,), rule="illinois")[2])
        for a in ARITIES:
            s = {kk: float(np.mean([x[kk] for x in per[a]]))
                 for kk in ("mean", "mean_walking", "p50", "p90", "p99", "max",
                            "P_eff", "P_eff_mean", "keep_all")}
            print(_fmt(f"arity={a} ({int(math.log2(a))} bit/pass)", s))
            results[(phase, a)] = s
        s = {kk: float(np.mean([x[kk] for x in ill]))
             for kk in ("mean", "mean_walking", "p50", "p90", "p99", "max",
                        "P_eff", "P_eff_mean", "keep_all")}
        print(_fmt("arity=2 illinois (interp)", s))
        results[(phase, "illinois")] = s
        # pass-count histogram of the plan-as-written arm
        h = np.zeros(40, np.float64)
        for x in per[2]:
            hh = np.array(x["hist"], np.float64)
            h[:hh.size] += hh / args.reps
        nz = np.flatnonzero(h)
        print("   plan-as-written (arity=2) pass histogram, rows per rep:")
        print("     " + "  ".join(f"{i}:{h[i]:.0f}" for i in nz))

    print("\n" + "=" * 100)
    print("WHOLE-KERNEL PRICE (harness denominator 189.98 us; design.md §4a "
          "rows quoted, not re-derived)")
    print("  removed: hist0 33.91 + refinement hist 20.08 + sweeps 15.63 = "
          "69.62   added: sample 2.12+6.08 = 8.20")
    print("  break-even P_eff = (69.62 - 8.20) / 6.48 = "
          f"{(HIST0+REF_HIST+SWEEPS-(HIST0/16+SWEEP_PER_LEVEL))/CNT_PASS:.2f} "
          "count passes  (probes/15 records 8.0 on a tighter accounting)")
    print("=" * 100)
    print(f"{'arm':<34}{'P_eff':>8}{'us/pass':>9}{'total us':>10}"
          f"{'harness %':>11}{'prod %':>9}")
    for a in ARITIES:
        s = results[("prefill", a)]
        et = a - 2  # thresholds beyond the first
        tot, d = price(s["P_eff"], et)
        print(f"{'arity=' + str(a):<34}{s['P_eff']:>8.3f}"
              f"{CNT_PASS + et * MARGINAL_PER_THRESH:>9.2f}{tot:>10.1f}"
              f"{d * 100:>+10.2f}%{d * SCALE_PROD * 100:>+8.2f}%")
    s = results[("prefill", "illinois")]
    tot, d = price(s["P_eff"], 0)
    print(f"{'arity=2 illinois':<34}{s['P_eff']:>8.3f}{CNT_PASS:>9.2f}"
          f"{tot:>10.1f}{d * 100:>+10.2f}%{d * SCALE_PROD * 100:>+8.2f}%")
    print("  design.md §5 v10 prediction: -12.2% harness / -14.9% production")

    print("\n" + "-" * 100)
    print("SENSITIVITY to the marginal cost of an extra threshold "
          f"(headline {MARGINAL_PER_THRESH:.2f} us = {MARGINAL_FRAC:.3f} x a "
          "1-threshold pass, from AOT bundles)")
    print("-" * 100)
    print(f"{'us/extra threshold':<34}" +
          "".join(f"{'arity=' + str(a):>12}" for a in ARITIES))
    cases = [("0.111  probes/15 null, 'free'", 0.111),
             ("2.35   pre-bundle guess", 2.35),
             (f"{MARGINAL_PER_THRESH_SC16:.2f}   sc16 sublinear fit",
              MARGINAL_PER_THRESH_SC16),
             (f"{MARGINAL_PER_THRESH:.2f}   MEASURED (sc2 vs sc4)",
              MARGINAL_PER_THRESH)]
    for lab, mpt in cases:
        row = ""
        for a in ARITIES:
            _, d = price(results[("prefill", a)]["P_eff"], a - 2, mpt)
            row += f"{d * 100:>+11.2f}%"
        print(f"{lab:<34}{row}")
    print("  arity=2 carries NO extra threshold, so its number is invariant "
          "to this whole table.")

    if args.dist:
        print("\n" + "=" * 100)
        print("SENSITIVITY TO THE DISTRIBUTIONAL ASSUMPTION (rho=1 synthetics, "
              "probes/05 `synth`)")
        print("=" * 100)
        print("NAME COLLISION, deliberate and load-bearing: `random` and "
              "`heavy_ties` here are")
        print("probes/05's generators, NOT the frozen test's `kind`s of the "
              "same name.  probes/05")
        print("`heavy_ties` is linspace(-2,2,17); the test's is "
              "integers(0,17) -- 17 distinct")
        print("values either way, but one straddles zero and the other does "
              "not, so they occupy")
        print("different L0 buckets.  The gate's own kinds are the next "
              "block.")
        for dist in ("random", "uniform", "heavy_ties"):
            rng = np.random.default_rng(20260904)
            sc = LS.synth(dist, 1024, N, rng)
            r = run_case(sc, np.full(1024, N), K, (2, 16))
            for a in (2, 16):
                print(_fmt(f"{dist:<12} arity={a}", r[a]))

        print("\n" + "=" * 100)
        print("THE FROZEN GATE'S OWN KINDS, transcribed by IMPORTING the test "
              "module")
        print("=" * 100)
        print("Not retyped and not approximated: `_scores` is called from "
              "the instrument itself, so")
        print("no transcription error is possible.  Ids confirmed by "
              "`pytest --collect-only`, not by")
        print("reading parametrize decorators.  Rows are full (rho=1), which "
              "is what the test uses.")
        spec = importlib.util.spec_from_file_location(
            "frozen_test",
            os.path.abspath(os.path.join(_HERE, "..", "..", "instruments",
                                         "test_dsv4_sparsecore_topk.py")))
        ft = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ft)
        for kind in ("strict_desc", "random", "heavy_ties"):
            rng = np.random.default_rng(ft.SEED)
            sc = ft._scores(kind, 1024, N, rng)
            w = np.sort(sc[0])[::-1][:K]
            r = run_case(sc, np.full(1024, N), K, (2, 16))
            for a in (2, 16):
                print(_fmt(f"{kind:<12} arity={a}", r[a]))
            print(f"    negative winners in row 0: {int((w < 0).sum())}/{K}"
                  f"   distinct values in row: {len(np.unique(sc[0]))}")

    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
