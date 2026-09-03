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
"""Off-pod reader: `hlo_stats` device self time -> the per-config A/B table.

Runs against a synced `cdk job sync-outputs` directory.  The bucketing is
`probes/bdt_hlo_stats.py`'s, which is `probes/R2.md` 1's, and is reused
verbatim by import rather than reimplemented:

  * **full op**  = the sum of `Avg. self time (us)` over the TensorCore rows.
    `probes/baseline-device-time.md` 1 established that this is the whole
    call: the TensorCore is *blocked* for the SparseCore's duration, so the
    `sc_topk...call-done` row already carries stage B and the two planes are
    not additive.
  * **stage A**  = the `StreamIdxTC*` rows (the `-hm` suffix Option C adds is
    matched by the same regex).
  * **stage B**  = the `sc_topk*` rows on TensorCore.
  * **residual** = everything else on TensorCore (the `-inf` init, bitcasts,
    slices, the `cond` on the arms that still have one).
  * the SparseCore plane is reported separately and never added.

Partition check: `sum(buckets) - TC total` must be 0, and `rows_kept ==
rows_total`.  R2's first dump silently lost rows to `--top-n 60`; a truncated
dump is the failure this catches, and any arm that fails either check is
marked and excluded from the table rather than quietly averaged in.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "probes"))


def _load_bdt():
    spec = importlib.util.spec_from_file_location(
        "bdt_hlo_stats", os.path.join(_PROBES, "bdt_hlo_stats.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bdt_hlo_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


BDT = _load_bdt()

ARMS = ("inc", "bkvp", "d1", "bkvp_d1", "merged")


def one(pb: str) -> tuple[str, dict]:
    parts = pb.split(os.sep)
    key = "unknown"
    if "plugins" in parts:
        key = parts[parts.index("plugins") - 1]
    with tempfile.TemporaryDirectory() as td:
        try:
            res = BDT.analyse(BDT.dump(pb, td))
        except Exception as err:  # noqa: BLE001
            res = {"error": repr(err)[:400]}
    res["xplane"] = pb
    return key, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    pbs = []
    for root, _d, files in os.walk(a.synced):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    if not pbs:
        raise SystemExit(f"no *.xplane.pb under {a.synced}")
    print(f"{len(pbs)} xplane(s)")

    raw: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for key, res in ex.map(one, sorted(pbs)):
            raw[key] = res
            tc = res.get("tensorcore_total_us")
            print(f"{key:<34} "
                  f"TC={tc if tc is None else f'{tc:9.3f}'} "
                  f"A={res.get('buckets_us', {}).get('stage_a', float('nan')):9.3f} "
                  f"B={res.get('buckets_us', {}).get('stage_b_on_tc', float('nan')):9.3f} "
                  f"res={res.get('buckets_us', {}).get('residual', float('nan')):8.3f} "
                  f"part={res.get('partition_residual_us', float('nan')):+.6f} "
                  f"rows={res.get('rows_kept')}/{res.get('rows_total')} "
                  f"SC={res.get('plane_totals_us', {}).get('SparseCore', 0.0):.3f}",
                  flush=True)

    # ---- reshape into config x arm -------------------------------------
    table: dict[str, dict] = {}
    for key, res in raw.items():
        if "__" not in key:
            print(f"WARN: cannot parse '{key}' as <config>__<arm>")
            continue
        cfg, arm = key.rsplit("__", 1)
        entry = table.setdefault(cfg, {})
        ok = ("error" not in res
              and res.get("rows_complete") is True
              and abs(res.get("partition_residual_us", 1.0)) < 1e-6)
        entry[arm] = {
            "full_us": res.get("tensorcore_total_us"),
            "stage_a_us": res.get("buckets_us", {}).get("stage_a"),
            "stage_b_us": res.get("buckets_us", {}).get("stage_b_on_tc"),
            "residual_us": res.get("buckets_us", {}).get("residual"),
            "sparsecore_us": res.get("plane_totals_us", {}).get("SparseCore"),
            "rows_kept": res.get("rows_kept"),
            "rows_total": res.get("rows_total"),
            "partition_residual_us": res.get("partition_residual_us"),
            "checks_pass": ok,
            "error": res.get("error"),
            "xplane": res.get("xplane"),
        }

    def pct(new, base):
        if new is None or base in (None, 0):
            return None
        return 100.0 * (new / base - 1.0)

    for cfg, arms in table.items():
        base = arms.get("inc")
        if not base:
            continue
        for arm, v in arms.items():
            for field in ("full", "stage_a", "stage_b", "residual"):
                v[f"{field}_pct_vs_inc"] = pct(v.get(f"{field}_us"),
                                               base.get(f"{field}_us"))

    payload = {"raw": raw, "table": table}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"wrote {a.out}")

    lines = []
    hdr = ("| config | arm | full µs | Δfull | stage A µs | ΔA | stage B µs |"
           " ΔB | residual µs | SC µs | checks |")
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"\
          " ---: | :--- |"
    lines += [hdr, sep]

    def fnum(x, fmt="{:.3f}"):
        return "—" if x is None else fmt.format(x)

    for cfg in sorted(table):
        for arm in ARMS:
            v = table[cfg].get(arm)
            if v is None:
                continue
            lines.append(
                f"| {cfg} | `{arm}` | {fnum(v['full_us'])} | "
                f"{fnum(v.get('full_pct_vs_inc'), '{:+.2f}%')} | "
                f"{fnum(v['stage_a_us'])} | "
                f"{fnum(v.get('stage_a_pct_vs_inc'), '{:+.2f}%')} | "
                f"{fnum(v['stage_b_us'])} | "
                f"{fnum(v.get('stage_b_pct_vs_inc'), '{:+.2f}%')} | "
                f"{fnum(v['residual_us'])} | {fnum(v['sparsecore_us'])} | "
                f"{'ok' if v['checks_pass'] else 'FAILED'} |")
    md = "\n".join(lines)
    print()
    print(md)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md + "\n")
        print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
