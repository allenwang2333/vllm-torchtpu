#!/usr/bin/env python3
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
"""Step 4: the COMPOSED candidate against the untouched incumbent, one job.

The deliverable of this file is not a sum. Two variants each measured at -20%
do not compose to -40%, and which of additive / sub-additive / lossy this one
is, is the question step 4 exists to answer. So every arm is timed by the
READ-ONLY `instruments/microbench.py`, in the SAME PROCESS, on the SAME shapes,
back to back:

    1  incumbent          shipped kernel, packed cache, tuple (2,2,2)
    2  B                  Option B alone, exactly as step 3 measured it
    3  BA                 the composed file: B + Option A
    4  BAE                the composed file with Option E's tuple (4,2,4)
    5  incumbent_repeat   DRIFT CONTROL -- identical code to arm 1

Arm 2 is the ATTRIBUTION arm. Without it a composed number that comes in below
B's individual number is uninterpretable: it could be A costing something, or
it could be the machine. With it, `BA - B` is A's contribution IN COMPOSITION
and `B - B_step3` is the cross-job reproduction of B itself.

Arm 5 spans the whole regime's work, so its delta is a conservative upper bound
on drift over the four arms it brackets. It is a NOISE FLOOR, not a result: an
arm-to-arm delta smaller than |drift| is not a finding. Note that drift is a
larger quantity than the within-arm `spread_pct`, which only sees jitter inside
one timing loop.

SIGN CONVENTION (binding). Every percentage below is a change in TIME:
`(new - old)/old * 100`, so NEGATIVE IS FASTER. The signed number appears only
in a column named `dtime%`. A `speedup` column (>1 is faster) sits beside it,
and both sit in the same row as the two absolute times they came from.

EARLY EXIT. `early_exit=False` is the primary setting because that is what
every step-3 number was taken at, and step 4's job is to compare against those.
`--early-exit-also` re-runs a regime at `True`, the checked-in call-site value
(`shapes.CALLSITE_ENABLE_EARLY_EXIT`), to check the win survives it.

Usage:
    python3 run_perf.py --regimes mixed_64k,mixed_confound --json out.json
"""

import argparse
import json
import sys

import _support as SUP


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--early-exit-also", default="",
                    help="comma list of regimes to ALSO run at early_exit=True")
    ap.add_argument("--bucket-mode", default="static",
                    choices=("static", "exact", "kvlen"),
                    help="static = what ships; exact = the unreachable "
                         "per-step ceiling. See _support.py.")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    from kernel_work.streamindex_topk.instruments import microbench as MB

    SUP.banner()
    incumbent = SUP.load_incumbent()
    b_only = SUP.load_b_only()
    composed = SUP.load_composed()
    narrow = SUP.load_narrow_inputs()

    also = {n for n in a.early_exit_also.split(",") if n}
    rows = []

    for name in [n for n in a.regimes.split(",") if n]:
        r = SUP.regime(name)
        bucket = SUP.bucket_for(r, composed, a.bucket_mode)
        acct = SUP.lane_accounting(r, bucket)
        print(f"\n[{name}] {r.phase} kv_len={r.kv_len} "
              f"max_model_len={r.max_model_len}", flush=True)
        print(f"[{name}] bucket_mode={a.bucket_mode} bucket={bucket}  "
              f"incumbent scores width={acct['incumbent_cols']} cols  "
              f"B width={acct['buffer_cols']} cols  "
              f"written={acct['written_sublanes']}/{acct['buffer_sublanes']} "
              f"sublanes  unwritten={acct['unwritten_sublanes']}  "
              f"tail_store={acct['tail_store_emitted']}", flush=True)

        ee_list = [False] + ([True] if name in also else [])
        for ee in ee_list:

            def timed(label, impl, *, split, tup):
                ctx_in = (narrow.patched_inputs() if split else SUP.nullctx())
                ctx_t = SUP.tuple_bound(tup)
                try:
                    with ctx_t, ctx_in:
                        row = MB.run_regime(impl, r, early_exit=ee,
                                            reps=a.reps, measure_topk=True)
                except Exception as exc:  # noqa: BLE001
                    row = dict(regime=name, error=repr(exc)[:2000])
                row["arm"] = label
                row["early_exit"] = ee
                row["kv_len_bucket"] = bucket
                row["bucket_mode"] = a.bucket_mode
                row["tuple"] = list(tup)
                rows.append(row)
                print(json.dumps(row), flush=True)
                return row

            arm_inc = timed("incumbent", incumbent, split=False, tup=(2, 2, 2))
            arm_b = timed("B", SUP.bind(b_only, kv_len_bucket=bucket),
                          split=False, tup=(2, 2, 2))
            arm_ba = timed("BA", SUP.bind(composed, kv_len_bucket=bucket),
                           split=True, tup=(2, 2, 2))
            arm_bae = timed("BAE", SUP.bind(composed, kv_len_bucket=bucket),
                            split=True, tup=(4, 2, 4))
            arm_drift = timed("incumbent_repeat", incumbent, split=False,
                              tup=(2, 2, 2))

            if "error" in arm_inc:
                continue
            base = arm_inc["full_ms"]

            def d(arm):
                if "error" in arm:
                    return None, None
                return ((arm["full_ms"] - base) / base * 100.0,
                        base / arm["full_ms"])

            db, sb = d(arm_b)
            dba, sba = d(arm_ba)
            dbae, sbae = d(arm_bae)
            ddrift, _ = d(arm_drift)

            # A's contribution IN COMPOSITION, and E's on top of it. Both are
            # relative to the arm they are stacked on, not to the incumbent --
            # that is the whole attribution.
            a_in_comp = (None if (db is None or dba is None) else
                         (arm_ba["full_ms"] - arm_b["full_ms"])
                         / arm_b["full_ms"] * 100.0)
            e_in_comp = (None if (dba is None or dbae is None) else
                         (arm_bae["full_ms"] - arm_ba["full_ms"])
                         / arm_ba["full_ms"] * 100.0)

            s = dict(kind="ab", regime=name, early_exit=ee, reps=a.reps,
                     inner=arm_inc.get("inner"), kv_len_bucket=bucket,
                     bucket_mode=a.bucket_mode,
                     lane_accounting=acct,
                     incumbent_ms=base,
                     incumbent_spread_pct=arm_inc["full_spread_pct"],
                     B_ms=arm_b.get("full_ms"),
                     B_spread_pct=arm_b.get("full_spread_pct"),
                     BA_ms=arm_ba.get("full_ms"),
                     BA_spread_pct=arm_ba.get("full_spread_pct"),
                     BAE_ms=arm_bae.get("full_ms"),
                     BAE_spread_pct=arm_bae.get("full_spread_pct"),
                     incumbent_repeat_ms=arm_drift.get("full_ms"),
                     B_dtime_pct=db, B_speedup_x=sb,
                     BA_dtime_pct=dba, BA_speedup_x=sba,
                     BAE_dtime_pct=dbae, BAE_speedup_x=sbae,
                     drift_pct=ddrift,
                     A_in_composition_pct=a_in_comp,
                     E_in_composition_pct=e_in_comp)
            rows.append(s)
            print(json.dumps(s), flush=True)

    print("\n=== COMPOSED vs INCUMBENT: full call, median ms, "
          "one TensorCore ===")
    print("dtime% = (arm - incumbent)/incumbent * 100.  NEGATIVE IS FASTER.")
    print("drift% is the second incumbent arm against the first: the noise "
          "floor. An |dtime%| below it is not a finding.")
    hdr = (f"{'regime':<22}{'ee':<6}{'inc_ms':>10}{'B_ms':>10}{'BA_ms':>10}"
           f"{'BAE_ms':>10}{'B_d%':>8}{'BA_d%':>8}{'BAE_d%':>8}"
           f"{'A|B%':>8}{'E|BA%':>8}{'drift%':>8}")
    print(hdr)
    print("-" * len(hdr))

    def f(x, w=8, p=2):
        return f"{'--':>{w}}" if x is None else f"{x:>{w}.{p}f}"

    for r in rows:
        if r.get("kind") != "ab":
            continue
        print(f"{r['regime']:<22}{str(r['early_exit']):<6}"
              f"{f(r['incumbent_ms'], 10, 3)}{f(r['B_ms'], 10, 3)}"
              f"{f(r['BA_ms'], 10, 3)}{f(r['BAE_ms'], 10, 3)}"
              f"{f(r['B_dtime_pct'])}{f(r['BA_dtime_pct'])}"
              f"{f(r['BAE_dtime_pct'])}{f(r['A_in_composition_pct'])}"
              f"{f(r['E_in_composition_pct'])}{f(r['drift_pct'])}")

    print("\nspreads (within-arm, %):")
    for r in rows:
        if r.get("kind") != "ab":
            continue
        print(f"  {r['regime']:<22}{str(r['early_exit']):<6}"
              f"inc {f(r['incumbent_spread_pct'], 6)}  "
              f"B {f(r['B_spread_pct'], 6)}  "
              f"BA {f(r['BA_spread_pct'], 6)}  "
              f"BAE {f(r['BAE_spread_pct'], 6)}")

    errs = [r for r in rows if "error" in r]
    for r in errs:
        print(f"\nERROR arm={r.get('arm')} regime={r['regime']} "
              f"ee={r.get('early_exit')}:\n{r['error']}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    print(f"\nPERF DONE errors={len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
