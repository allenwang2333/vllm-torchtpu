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
"""Option B perf: incumbent vs right-size, same process, same instrument.

Both arms go through `instruments/microbench.run_regime`, which is read-only.
The variant is handed to it through `_rs_support.bind`, which binds the two
extra static kwargs the instrument's fixed kwargs list cannot pass. Everything
that could move the number -- `inner`, `reps`, `warmup`, the input build, the
timing loop -- is therefore literally the same code for both arms, in the same
process, on the same inputs.

The `topk` control arm is disabled here (`measure_topk=False`). It would run
`sparsecore_topk` at `S.num_sublanes_total(regime.max_model_len)` for BOTH
arms, because that width is computed from the regime rather than from the
implementation -- so it cannot see B's change and would only cost time. B's
effect on the SparseCore side is measured explicitly in `run_topk_width.py`.

Sign convention: `d_time_pct = (variant - incumbent) / incumbent * 100`.
Negative is faster.
"""

import argparse
import json
import sys

from kernel_work.streamindex_topk.instruments import microbench as MB

import _rs_support as RS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--reps", type=int, default=MB.REPS)
    ap.add_argument("--early-exit", default="true",
                    choices=["false", "true", "both"])
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    inc = RS.load_incumbent()
    var = RS.load_variant()
    arms = {"false": [False], "true": [True], "both": [False, True]}[
        a.early_exit]

    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        r = RS.regime(name)
        w_inc, w_rs = RS.widths(r)
        for ee in arms:
            pair = {}
            for arm, impl in (("incumbent", inc),
                              ("right_size",
                               RS.bind(var, kv_len_bucket=r.kv_len))):
                try:
                    row = MB.run_regime(impl, r, early_exit=ee, reps=a.reps,
                                        measure_topk=False)
                except Exception as exc:  # noqa: BLE001
                    row = dict(regime=name, error=repr(exc)[:2000])
                row["arm"] = arm
                row["early_exit"] = ee
                row["scores_cols"] = w_inc if arm == "incumbent" else w_rs
                rows.append(row)
                pair[arm] = row
                print(json.dumps(row), flush=True)

            if "error" in pair["incumbent"] or "error" in pair["right_size"]:
                continue
            old = pair["incumbent"]["full_ms"]
            new = pair["right_size"]["full_ms"]
            delta = dict(
                regime=name, early_exit=ee, kind="delta",
                incumbent_ms=old, right_size_ms=new,
                incumbent_spread_pct=pair["incumbent"]["full_spread_pct"],
                right_size_spread_pct=pair["right_size"]["full_spread_pct"],
                incumbent_cols=w_inc, right_size_cols=w_rs,
                d_time_pct=(new - old) / old * 100.0,
                speedup_x=old / new,
                inner=pair["incumbent"]["inner"],
            )
            rows.append(delta)
            print(json.dumps(delta), flush=True)

    print("\n=== Option B: full-call time, single TensorCore ===")
    print("d time % = (right_size - incumbent) / incumbent; NEGATIVE IS FASTER")
    hdr = (f"{'regime':<16}{'ee':<6}{'cols_inc':>9}{'cols_rs':>9}"
           f"{'inc_ms':>10}{'inc_sp%':>9}{'rs_ms':>10}{'rs_sp%':>9}"
           f"{'d time %':>10}{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("kind") != "delta":
            continue
        print(f"{r['regime']:<16}{str(r['early_exit']):<6}"
              f"{r['incumbent_cols']:>9}{r['right_size_cols']:>9}"
              f"{r['incumbent_ms']:>10.5f}{r['incumbent_spread_pct']:>9.2f}"
              f"{r['right_size_ms']:>10.5f}{r['right_size_spread_pct']:>9.2f}"
              f"{r['d_time_pct']:>10.2f}{r['speedup_x']:>11.3f}")
    for r in rows:
        if "error" in r:
            print(f"\nERROR [{r['regime']}/{r.get('arm')}]: {r['error']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
