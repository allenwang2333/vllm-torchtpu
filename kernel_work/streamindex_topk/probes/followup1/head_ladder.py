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
"""Split the head-sum bundle: what in it is linear in `NUM_HEADS`, and what is not.

The term this exists to decompose
---------------------------------
`probes/step3_decider_RESULT.md` shard 3 `measured (j-b54e9716-3db5-4f1f-b8d4)`
put the MIXED inner step at 8.42 us and decomposed it as

    MXU at K=128                                        1.084 us  (14.8%)
    relu + `w` + scale + two masks + `where`            2.401 us  (32.7%)
    32-head sum + output store + fixed per-grid-step    3.858 us  (52.5%)

The last line is the largest term in the step and it is the one term no built
variant touches. That shard measured the **bundle** and said in as many words
that the split of it into reduction, store and fixed per-step cost is
`unmodelled`. This probe models it, by the only means that separates the three:
move `NUM_HEADS` and watch which part moves.

    t(H) = a * H + b

    a * H  -- the reduction arithmetic. Reading and summing H rows of
              (BQ_SZ, BKV_SZ) f32 is exactly proportional to H.
    b      -- the store of the reduced result (BQ_SZ x BKV_SZ x 4 = 1 MiB per
              step, the same at every H) plus the fixed per-grid-step cost.

THIS IS A DECOMPOSITION, NOT AN OPTIMISATION
--------------------------------------------
`NUM_HEADS` is `config.json: index_n_heads = 32`. A kernel at H = 4 computes a
**different function** of different inputs and is not shippable, cannot be made
shippable, and no row of this table is a win. The deliverable is `a` and `b`, so
that options E1, E2 and F -- all of which change how the head reduction is done
at the real H = 32 -- can be priced before any of them is built.

Construction
------------
The kernel is not rewritten. `probes/vpu_mxu_strike.py` already contains the
mirror of `compute_scores` (streamindex_topk.py:304-354) that produced the
3.858 us, and its extents are module globals. This script loads that file by
path and **rebinds** `NUM_HEADS` around each arm, so the body, the BlockSpecs,
the output store and the timing loop are literally the same code at every rung.
Nothing in `probes/vpu_mxu_strike.py` is edited and nothing in `instruments/`
is touched.

Four arms per rung, and why each is here
----------------------------------------
  `mxu/sum` K=128   the 3.858 us bundle itself, at the real K. Its H-dependence
                    still contains the matmul, whose FLOPs are 2*BQ*H*BKV*K and
                    therefore also proportional to H -- so this arm alone
                    cannot separate reduction from MXU.
  `mxu/sum` K=512   with the above, gives a per-rung K-slope, hence an
                    MXU-removed intercept `rest_sum(H)`. Reported, but see the
                    caveat: the prior job's two-point K slope implied 2242
                    Tops/s, at or above fp8 MXU peak, so the intercept method
                    is known to be shaky.
  `bcast/sum`       THE CLEAN ARM. `s` is built by broadcasting a runtime
                    (1, BKV_SZ) ref instead of by the einsum, so there is **no
                    matmul at all** and the H-dependence is the reduction (plus
                    the broadcast's own materialisation, which is also O(H) and
                    is named as a contaminant rather than ignored).
  `mxu/full` K=128  the whole step, for the share arithmetic only.

Every `sum`-epilogue row is a CONTROL ARM in `vpu_mxu_strike.py`'s sense: it is
engineered to be subtracted and its absolute time is not a finding on its own.
Here it is also the quantity under test, which is a different use of the same
number; both readings are printed.

Sign convention: every percentage is a change in TIME, `(new - old)/old`.
Negative is faster. This file prints no cross-arm percentages -- an H ladder has
no faster and slower, only a slope.
"""

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

_HERE = os.path.dirname(os.path.abspath(__file__))
_STRIKE = os.path.normpath(os.path.join(_HERE, "..", "vpu_mxu_strike.py"))

LADDER = [4, 8, 16, 32, 64]
REAL_H = 32           # config.json index_n_heads
PUBLISHED_REST_SUM = 3.858   # us/step, j-b54e9716-3db5-4f1f-b8d4
PUBLISHED_STEP = 8.42        # us/step, j-5f8f4a08-20b4-4fa0-bfc3


def _load_strike():
    spec = importlib.util.spec_from_file_location("vpu_mxu_strike", _STRIKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VM = _load_strike()


def fit_linear(xs, ys):
    """Least squares `y = a*x + b` with R^2 and per-point residuals.

    Returned so a scattered fit cannot be quoted as a clean linear one: the
    residuals are printed next to the points they belong to.
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    resid = [(x, y, a * x + b, y - (a * x + b)) for x, y in zip(xs, ys)]
    rms = (ss_res / n) ** 0.5
    return dict(a=a, b=b, r2=r2, rms_resid=rms, resid=resid,
                max_abs_resid=max(abs(r[3]) for r in resid))


def one_arm(h, gen, epi, k):
    rec = dict(num_heads=h, gen=gen, epilogue=epi, k=k, grid=VM.GRID,
               bq_sz=VM.BQ_SZ, bkv_sz=VM.BKV_SZ,
               is_control=(epi == "sum"),
               s_vmem_MiB=VM.BQ_SZ * h * VM.BKV_SZ * 4 / 2 ** 20,
               out_store_MiB=VM.BQ_SZ * VM.BKV_SZ * 4 / 2 ** 20)
    if gen == "mxu":
        rec["mxu_flops_per_step"] = 2 * VM.BQ_SZ * h * VM.BKV_SZ * k
    try:
        fn, args = VM.build(gen, epi, k)
        med, spread = VM._time(fn, *args)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:900]
        print(json.dumps(rec), flush=True)
        return rec
    rec.update(total_s=med, us_per_step=med / VM.GRID * 1e6, spread_pct=spread)
    print(json.dumps(rec), flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default=",".join(str(h) for h in LADDER))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--k-hi", type=int, default=512,
                    help="the second rung of the per-H K ladder, used only to "
                         "form an MXU-removed intercept")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    # More reps and a bigger batch than `vpu_mxu_strike.py`'s defaults: two of
    # that job's rows came back at 550% and 41% spread, and this probe's whole
    # output is a slope through five points. Rebinding the module's timing
    # globals keeps the loop itself identical.
    VM.REPS, VM.BATCH, VM.WARMUP = a.reps, a.batch, a.warmup

    heads = [int(x) for x in a.heads.split(",") if x]
    print(f"head ladder {heads}; grid={VM.GRID} bq_sz={VM.BQ_SZ} "
          f"bkv_sz={VM.BKV_SZ} head_dim={VM.HEAD_DIM} k_hi={a.k_hi}; "
          f"reps={VM.REPS} batch={VM.BATCH} warmup={VM.WARMUP}", flush=True)
    print("DECOMPOSITION PROBE. A smaller NUM_HEADS computes a DIFFERENT "
          "FUNCTION and is not shippable. No row below is a win.", flush=True)

    rows = []
    for h in heads:
        old = VM.NUM_HEADS
        VM.NUM_HEADS = h
        try:
            for gen, epi, k in (("mxu", "sum", VM.HEAD_DIM),
                                ("mxu", "sum", a.k_hi),
                                ("mxu", "full", VM.HEAD_DIM),
                                ("bcast", "sum", VM.HEAD_DIM)):
                rows.append(one_arm(h, gen, epi, k))
        finally:
            VM.NUM_HEADS = old

    def get(h, gen, epi, k):
        return next((r for r in rows
                     if r["num_heads"] == h and r["gen"] == gen
                     and r["epilogue"] == epi and r["k"] == k
                     and "error" not in r), None)

    # ---- raw table ---------------------------------------------------------
    print(f"\n=== raw arms: us per inner step, grid={VM.GRID}, "
          f"(bq_sz={VM.BQ_SZ}, bkv_sz={VM.BKV_SZ}) ===")
    hdr = (f"{'H':>4}{'gen':<8}{'epi':<7}{'K':>6}{'us/step':>11}"
           f"{'spread%':>10}{'s MiB':>9}{'GFLOP/step':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['num_heads']:>4}{r['gen']:<8}{r['epilogue']:<7}"
                  f"{r['k']:>6}   ERROR {r['error'][:70]}")
            continue
        fl = r.get("mxu_flops_per_step", 0) / 1e9
        print(f"{r['num_heads']:>4}{r['gen']:<8}{r['epilogue']:<7}{r['k']:>6}"
              f"{r['us_per_step']:>11.4f}{r['spread_pct']:>10.2f}"
              f"{r['s_vmem_MiB']:>9.1f}{fl:>12.3f}")

    # ---- MXU-removed intercept per rung ------------------------------------
    print("\n=== per-rung MXU-removed intercept (from the K ladder) ===")
    print(f"slope_H = (t_sum(K={a.k_hi}) - t_sum(K={VM.HEAD_DIM})) / "
          f"({a.k_hi} - {VM.HEAD_DIM});  mxu(H) = slope_H * {VM.HEAD_DIM};  "
          f"rest_sum(H) = t_sum(K={VM.HEAD_DIM}) - mxu(H)")
    hdr2 = (f"{'H':>4}{'t_sum(Klo)':>12}{'t_sum(Khi)':>12}{'mxu us':>10}"
            f"{'rest_sum us':>13}{'mxu Tops/s':>12}")
    print(hdr2)
    print("-" * len(hdr2))
    rest = {}
    for h in heads:
        lo, hi = get(h, "mxu", "sum", VM.HEAD_DIM), get(h, "mxu", "sum",
                                                        a.k_hi)
        if not lo or not hi:
            continue
        slope = (hi["us_per_step"] - lo["us_per_step"]) / (a.k_hi -
                                                           VM.HEAD_DIM)
        mxu = slope * VM.HEAD_DIM
        rest[h] = lo["us_per_step"] - mxu
        tops = ((2 * VM.BQ_SZ * h * VM.BKV_SZ * VM.HEAD_DIM) / (mxu * 1e-6)
                / 1e12) if mxu > 0 else float("nan")
        print(f"{h:>4}{lo['us_per_step']:>12.4f}{hi['us_per_step']:>12.4f}"
              f"{mxu:>10.4f}{rest[h]:>13.4f}{tops:>12.1f}")
    print("A `mxu Tops/s` at or above fp8 peak means the two-point K slope is "
          "not measuring the matmul; treat that rung's `rest_sum` as "
          "unreliable and read the bcast fit instead.")

    # ---- the three fits ----------------------------------------------------
    series = {}
    xs_bs = [h for h in heads if get(h, "bcast", "sum", VM.HEAD_DIM)]
    series["bcast_sum"] = (xs_bs, [get(h, "bcast", "sum",
                                       VM.HEAD_DIM)["us_per_step"]
                                   for h in xs_bs])
    xs_ms = [h for h in heads if get(h, "mxu", "sum", VM.HEAD_DIM)]
    series["mxu_sum_raw"] = (xs_ms, [get(h, "mxu", "sum",
                                         VM.HEAD_DIM)["us_per_step"]
                                     for h in xs_ms])
    xs_r = sorted(rest)
    series["mxu_sum_mxu_removed"] = (xs_r, [rest[h] for h in xs_r])
    xs_f = [h for h in heads if get(h, "mxu", "full", VM.HEAD_DIM)]
    series["mxu_full"] = (xs_f, [get(h, "mxu", "full",
                                     VM.HEAD_DIM)["us_per_step"]
                                 for h in xs_f])

    fits = {}
    print("\n=== fits: t(H) = a*H + b, us per inner step ===")
    print("`a` is the term LINEAR in NUM_HEADS (the reduction arithmetic).")
    print("`b` is the term INDEPENDENT of NUM_HEADS (the 1 MiB output store "
          "plus the fixed per-grid-step cost).")
    for name, (xs, ys) in series.items():
        f = fit_linear(xs, ys)
        fits[name] = f
        if not f:
            print(f"\n{name}: too few points to fit")
            continue
        at_real = f["a"] * REAL_H + f["b"]
        lin_share = (f["a"] * REAL_H / at_real * 100.0) if at_real else float(
            "nan")
        print(f"\n{name}:  t(H) = {f['a']:.6f}*H + {f['b']:.4f}   "
              f"(R^2 = {f['r2']:.5f}, rms resid = {f['rms_resid']:.4f} us, "
              f"max |resid| = {f['max_abs_resid']:.4f} us)")
        print(f"    at H = {REAL_H}: linear part {f['a']*REAL_H:.4f} us, "
              f"constant part {f['b']:.4f} us, total {at_real:.4f} us  "
              f"-> linear is {lin_share:.1f}% of it")
        print(f"    {'LINEAR-IN-H DOMINATES' if lin_share > 50 else 'CONSTANT DOMINATES'}"
              f" at the real H = {REAL_H}")
        print(f"    {'H':>5}{'measured':>11}{'fitted':>11}{'resid':>10}")
        for x, y, yh, rr in f["resid"]:
            print(f"    {x:>5}{y:>11.4f}{yh:>11.4f}{rr:>10.4f}")

    # ---- linearity actually tested, not assumed ----------------------------
    print("\n=== is it linear? fit on the low rungs, predict the top rung ===")
    for name, (xs, ys) in series.items():
        if len(xs) < 4:
            print(f"{name}: fewer than 4 points, extrapolation test skipped")
            continue
        lo_x, lo_y = xs[:-1], ys[:-1]
        f_lo = fit_linear(lo_x, lo_y)
        pred = f_lo["a"] * xs[-1] + f_lo["b"]
        err = (ys[-1] - pred) / ys[-1] * 100.0 if ys[-1] else float("nan")
        print(f"{name}: fit on H={lo_x} -> t({xs[-1]}) predicted "
              f"{pred:.4f} us, measured {ys[-1]:.4f} us, "
              f"error {err:+.2f}% of the measurement")

    # ---- reconciliation with the published bundle --------------------------
    print("\n=== against the published bundle ===")
    print(f"published `32-head sum + output store + fixed per-step` = "
          f"{PUBLISHED_REST_SUM} us/step  measured (j-b54e9716-3db5-4f1f-b8d4)")
    print(f"published whole MIXED inner step = {PUBLISHED_STEP} us/step  "
          f"measured (j-5f8f4a08-20b4-4fa0-bfc3)")
    for name in ("bcast_sum", "mxu_sum_raw", "mxu_sum_mxu_removed"):
        f = fits.get(name)
        if not f:
            continue
        at_real = f["a"] * REAL_H + f["b"]
        d = (at_real - PUBLISHED_REST_SUM) / PUBLISHED_REST_SUM * 100.0
        print(f"  {name} at H={REAL_H}: {at_real:.4f} us  "
              f"({d:+.2f}% vs the published bundle)")
    m32 = get(REAL_H, "mxu", "sum", VM.HEAD_DIM)
    if m32:
        print(f"  direct measurement, mxu/sum at H={REAL_H}, K={VM.HEAD_DIM}: "
              f"{m32['us_per_step']:.4f} us ({m32['spread_pct']:.2f}% spread)")

    print("\nREMINDER: a smaller NUM_HEADS computes a different function. "
          "No row here is shippable and none is a speedup. The deliverable is "
          "`a` and `b`.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows,
                           fits={k: (v and {kk: vv for kk, vv in v.items()
                                            if kk != "resid"})
                                 for k, v in fits.items()},
                           rest_sum=rest, heads=heads,
                           reps=VM.REPS, batch=VM.BATCH), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
