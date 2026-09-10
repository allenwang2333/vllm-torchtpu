"""probe 24 -- the composed candidate, measured whole against the incumbent.

WHAT THIS IS
============
A thin wrapper around `rank.py` (md5 e0bde4c103d277aa933fa585c1c49603), the
harness that already executed clean as j-66c7e331-fa41-461e-8686.  It is a
wrapper and not a rewrite on purpose: the pooling, the two brackets, the
eager `pl.kernel` spy, the HLO scope-name check, the numpy-oracle smoke and
the md5 gate are all code that has already run once on this hardware and
produced a result the coordinator accepted.  Re-typing them to change an arm
table would throw that away.

It does exactly three things `rank.py` does not:

  1. Swaps the arm table for {incumbent, composed}, keeps the three nulls, and
     keeps all three cells including the rho=1 control (which earned its place
     in probe 23 by catching v4ragged's by-construction null).

  2. STAGE 8 -- the ONE-SIGNED NULL CORRECTION.  probe 23's floor is the range
     of the four incumbent slots but its delta is measured against `base`, one
     particular slot.  When the three nulls come out one-signed -- as they did
     in D.decodeX|inside, at -0.205 / -0.211 / -0.103 us -- `base` is sitting
     at an extreme of its own floor, so a candidate at the incumbent centroid
     shows a delta of up to a full floor width and reads as a win.  Four
     verdicts flipped to null on that basis.  The fix is to measure the delta
     against the MEDIAN of the four incumbent slots while keeping the range as
     the floor.  Both numbers are reported; the corrected one governs.

  3. STAGE 9 -- THREE-ESTIMATOR AGREEMENT.  `us_median` is the pre-registered
     read, but the harness also stores the pooled slope (`us`) and the unpaired
     estimate (`us_unpaired`).  A call is RESOLVED only if all three agree in
     sign AND all three clear their own corrected floor.  In probe 23 this is
     what separated the six stable prefill rows from four rows of noise.

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not reconcile the composed number against the sum of v2 + v4c2048 +
v12.  Coordinator's instruction, and it is the right one: the composed number
comes from this one job, and if it lands short of the parts, that shortfall is
the finding -- overlapping arms are exactly what step 4 exists to detect.  The
prior per-arm deltas are printed for orientation, clearly labelled as a
DIFFERENT JOB, and no sum, expected value or residual is computed anywhere.

ACTIVATION AND GATING
=====================
  - The composed md5 is REQUIRED, from SCTOPK_FINAL_MD5.  Unset is a hard
    refusal, not a default: a default would let a stale subject through.
  - The scope suffix is SCTOPK_FINAL_SUFFIX, default `_final`.  It is tested as
    a SUBSTRING, never `endswith` -- probe 23 found that v2/v2big PREPEND their
    tag while v4/v11/v12 append it, so an `endswith` checker is convention-
    dependent and this one must not be.
  - Foreign-suffix detection is left non-fatal here.  The composed kernel merges
    three variants and may legitimately carry their tags in its scope name; an
    abort on that would be a false negative on a correct build.  STAGE 8 prints
    any legacy tag it sees as a NOTE.  The fatal gates remain the md5 and the
    `_final` substring in the `pl.kernel` name= kwarg read in rank.py STAGE 3.
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rank  # noqa: E402  the vetted harness, md5-gated by the job script

BASE_MD5 = "44353b347c96e0eedbd742c710d4540a"
LEGACY_SUFFIXES = ["_v11retile", "_f32in", "_v4ragged", "_v4c2048",
                   "_v2r1", "_v2bigscratch", "_sc2"]

# probe 23, j-66c7e331-fa41-461e-8686.  ORIENTATION ONLY -- a DIFFERENT JOB.
# Printed so the composed number can be read in context.  Never summed.
PRIOR = {
    ("P.prefillX", "whole"):  {"v2": -37.052, "v4c2048": -11.588, "v12": -25.149},
    ("P.prefillX", "inside"): {"v2": -37.400, "v4c2048": -11.189, "v12": -0.081},
    ("D.decodeX", "whole"):   {"v2": -3.265,  "v4c2048": -0.267,  "v12": -0.893},
    ("D.decodeX", "inside"):  {"v2": -4.067,  "v4c2048": -0.334,  "v12": -0.252},
}

INCUMBENT_SLOTS = ["base", "#null_a", "#null_b", "#null_c"]
ESTIMATORS = ["us_median", "us", "us_unpaired"]


def _install_arms():
    md5 = os.environ.get("SCTOPK_FINAL_MD5", "").strip().lower()
    if len(md5) != 32 or any(c not in "0123456789abcdef" for c in md5):
        print("FINAL_GATE_FAIL: SCTOPK_FINAL_MD5 is unset or not a 32-hex "
              "md5. Refusing to run -- a default here would let a stale "
              "subject through the gate it exists to close.", flush=True)
        return None, None
    suf = os.environ.get("SCTOPK_FINAL_SUFFIX", "_final")
    rank.ARMS[:] = [
        ("base", "base", BASE_MD5, None, {}, {}, {}),
        ("final", "final", md5, suf, {}, {}, {}),
    ]
    # Only `_final` is policed as foreign, so a bare incumbent name carrying it
    # still aborts (that is the check that proves `base` is not the composite),
    # while the composite carrying a legacy tag does not (see the docstring).
    rank.ALL_SUFFIXES[:] = [suf]
    rank.RHO1_ARMS = {"base", "final", "#null_a", "#null_b", "#null_c"}
    return md5, suf


def _slots(raw, cell, br, est):
    out = {}
    for t in INCUMBENT_SLOTS:
        r = raw.get(f"{cell}|{br}|{t}")
        if r is not None:
            out[t] = r[est]
    return out


def stage8_9(outdir, md5, suf):
    path = os.path.join(outdir, "ranking.json")
    with open(path) as fh:
        res = json.load(fh)
    raw = res["raw"]
    log = lambda s: print(s, flush=True)  # noqa: E731

    log("")
    log("=" * 74)
    log("STAGE 8  ONE-SIGNED NULL CORRECTION")
    log("=" * 74)
    log("  rank.py measures every delta against `base`, one incumbent slot,")
    log("  while the floor is the RANGE of all four. If the three nulls come")
    log("  out one-signed, `base` is at an extreme of its own floor and a")
    log("  candidate sitting at the incumbent centroid reads as a win of up")
    log("  to a full floor width. Corrected delta is against the MEDIAN of the")
    log("  four incumbent slots; the floor stays the range. probe 23 flipped")
    log("  four verdicts on exactly this.")
    log("")

    diag = {}
    cells = sorted({k.split("|")[0] for k in raw})
    for cell in cells:
        for br in ("whole", "inside"):
            sl = _slots(raw, cell, br, "us_median")
            if len(sl) < 4:
                continue
            base = sl["base"]
            nulls = [sl[t] - base for t in INCUMBENT_SLOTS[1:]]
            signs = {(1 if d > 0 else -1 if d < 0 else 0) for d in nulls}
            one_signed = len(signs) == 1 and 0 not in signs
            vals = list(sl.values())
            med = statistics.median(vals)
            rng = max(vals) - min(vals)
            pos = sorted(vals).index(base) if base in sorted(vals) else -1
            diag[f"{cell}|{br}"] = {
                "slots": sl, "null_deltas_vs_base": nulls,
                "one_signed": one_signed, "incumbent_median": med,
                "floor_range_us": rng, "base_us": base,
                "base_offset_from_median_us": base - med,
                "base_rank_in_slots": pos}
            flag = ("ONE-SIGNED -- base is biased, corrected delta governs"
                    if one_signed else "mixed sign, base is not obviously biased")
            log(f"  {cell:12s} {br:6s} nulls vs base "
                f"{nulls[0]:+7.3f} {nulls[1]:+7.3f} {nulls[2]:+7.3f}  "
                f"floor {rng:6.3f}  base-median {base - med:+7.3f}  {flag}")

    log("")
    log("  NOTE on scope names actually observed (non-fatal, see docstring):")
    for tag, a in sorted(res.get("arm_check_hlo", {}).items()):
        legacy = sorted({o for o in LEGACY_SUFFIXES
                         if any(o in s for s in a.get("names", []))})
        log(f"    {tag:10s} names={a.get('names')}"
            + (f"  legacy tags present: {legacy}" if legacy else ""))

    log("")
    log("=" * 74)
    log("STAGE 9  THREE-ESTIMATOR AGREEMENT -- the composed candidate")
    log("=" * 74)
    log("  RESOLVED requires all three estimators to agree in sign AND each to")
    log("  clear its own corrected floor. us_median alone is the pre-registered")
    log("  read but it is not sufficient; in probe 23 it disagreed with the")
    log("  pooled slope on four calls and every one of them was noise.")
    log("")

    verdicts = {}
    for cell in cells:
        for br in ("whole", "inside"):
            if f"{cell}|{br}" not in diag:
                continue
            nm = f"{cell}|{br}|final"
            if nm not in raw:
                continue
            row = {}
            ok_sign = set()
            n_clear = 0
            for est in ESTIMATORS:
                sl = _slots(raw, cell, br, est)
                vals = list(sl.values())
                med = statistics.median(vals)
                rng = max(vals) - min(vals)
                v = raw[nm][est]
                d_med = v - med
                d_base = v - sl["base"]
                clear = abs(d_med) > rng
                n_clear += int(clear)
                ok_sign.add(1 if d_med > 0 else -1 if d_med < 0 else 0)
                # The margin is reported because "resolved by 0.7% of the
                # floor on its weakest estimator" and "resolved" read very
                # differently to someone deciding what to build on. The rule
                # is not changed by the margin; the margin is printed beside
                # the verdict so the reader can weight it.
                margin = abs(d_med) - rng
                row[est] = {"final_us": v, "incumbent_median_us": med,
                            "delta_vs_median_us": d_med,
                            "delta_vs_base_us": d_base,
                            "floor_us": rng, "clears_floor": clear,
                            "margin_over_floor_us": margin,
                            "margin_pct_of_floor": (100.0 * margin / rng)
                            if rng > 0 else float("inf")}
            agree = len(ok_sign) == 1 and 0 not in ok_sign
            resolved = agree and n_clear == 3
            sgn = next(iter(ok_sign)) if agree else 0
            worst = min(row[e]["margin_pct_of_floor"] for e in ESTIMATORS)
            marginal = resolved and worst < 10.0
            verdict = ("RESOLVED FASTER" if resolved and sgn < 0 else
                       "RESOLVED SLOWER" if resolved and sgn > 0 else
                       "NOT RESOLVED -- estimators disagree in sign"
                       if not agree else
                       f"NOT RESOLVED -- only {n_clear}/3 clear the floor")
            best = max(row[e]["margin_pct_of_floor"] for e in ESTIMATORS)
            reading = ""
            if marginal:
                verdict += (f"  [MARGINAL: weakest estimator clears the floor "
                            f"by only {worst:.1f}% of it]")
                # The DECOMPOSITION is more informative than the flag. One
                # estimator on the line while the others clear widely is a
                # different object from three clustered at the boundary: the
                # first says the call is probably real and that this estimator
                # is the conservative one, the second says the effect itself is
                # borderline. Same flag, opposite readings.
                reading = (
                    f"one estimator on the line at {worst:.1f}% while another "
                    f"clears by {best:.1f}% -- the call is probably REAL and "
                    f"the weak estimator is the conservative one here, NOT an "
                    f"effect sitting on the boundary"
                    if best >= 50.0 else
                    f"all three estimators are near the boundary "
                    f"({worst:.1f}%..{best:.1f}%) -- the EFFECT itself is "
                    f"borderline, not just one estimator")
            verdicts[f"{cell}|{br}"] = {"per_estimator": row, "agree": agree,
                                        "n_clear": n_clear,
                                        "resolved": resolved,
                                        "marginal": marginal,
                                        "worst_margin_pct_of_floor": worst,
                                        "best_margin_pct_of_floor": best,
                                        "margin_reading": reading,
                                        "verdict": verdict}
            log(f"  --- {cell} / {br} ---")
            for est in ESTIMATORS:
                r = row[est]
                log(f"    {est:12s} final {r['final_us']:9.3f}  "
                    f"inc_median {r['incumbent_median_us']:9.3f}  "
                    f"d_med {r['delta_vs_median_us']:+8.3f}  "
                    f"d_base {r['delta_vs_base_us']:+8.3f}  "
                    f"floor {r['floor_us']:6.3f}  "
                    f"margin {r['margin_over_floor_us']:+7.3f} us "
                    f"({r['margin_pct_of_floor']:+7.1f}% of floor)  "
                    f"{'clears' if r['clears_floor'] else 'INSIDE FLOOR'}")
            log(f"    => {verdict}")
            if reading:
                log(f"       READING: {reading}")
            pr = PRIOR.get((cell, br))
            if pr:
                log("    prior single arms, probe 23, A DIFFERENT JOB, "
                    "orientation only:")
                log("      " + "  ".join(f"{k} {v:+.3f}" for k, v in pr.items())
                    + "   [NO SUM IS COMPUTED -- if the composite lands short "
                      "of the parts, that gap is the finding]")
            log("")

    res["one_signed_diagnostic"] = diag
    res["composed_verdicts"] = verdicts
    res["final_md5"] = md5
    res["final_suffix"] = suf
    res["prior_single_arms_orientation_only"] = {
        f"{c}|{b}": v for (c, b), v in PRIOR.items()}
    res["reconciliation_policy"] = (
        "The composed number comes from this job alone. No sum of the "
        "individual arms is computed anywhere in this file.")
    with open(os.path.join(outdir, "final.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=str)
    log(f"wrote {os.path.join(outdir, 'final.json')}")
    log("FINAL_DONE")
    return 0


def main():
    md5, suf = _install_arms()
    if md5 is None:
        return 2
    print(f"FINAL_ARMS_OK composed md5={md5} suffix={suf} "
          f"(incumbent 4 slots: base + 3 nulls)", flush=True)
    rc = rank.main()
    if rc != 0:
        print(f"RANK_FAILED rc={rc} -- STAGE 8/9 skipped", flush=True)
        return rc
    if "--dry-run" in sys.argv:
        print("DRY_RUN_OK (STAGE 8/9 skipped, they need a ranking.json)",
              flush=True)
        return 0
    i = sys.argv.index("--outdir")
    return stage8_9(sys.argv[i + 1], md5, suf)


if __name__ == "__main__":
    raise SystemExit(main())
