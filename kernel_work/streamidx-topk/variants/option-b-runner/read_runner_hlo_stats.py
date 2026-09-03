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
"""Off-pod reader for the Option B runner A/B: `hlo_stats` device self time.

The bucketing is `probes/bdt_hlo_stats.py`'s, reused by import rather than
reimplemented.  One difference in interpretation, and it matters:

  * **`residual` now contains the KV-cache scatter.**  The traced unit here is
    the caller, not the kernel, so the `-inf` init, the fp8 bitcast, the pad,
    the `searchsorted` and the scatter all land in `residual`.  Option B adds
    the `concatenate` of the G per-chunk top-k blocks to that bucket too, and
    that concatenate is one of the costs the harness never charged.  Read
    `residual` as "caller overhead", not as "noise".

  * `full` (the TensorCore plane total) is still the whole call: the
    TensorCore is *blocked* for the SparseCore's duration, so the
    `sc_topk...call-done` row already carries stage B.  The SparseCore plane
    is reported separately and never added.

Baseline is **`bG1`**, not `inc`: `bG1` is the merged module (bkv_p ladder, D1,
Option C on prefill) unchunked, so dividing by it isolates Option B from the
three changes it is being taken *with*.  `inc` is reported in the same table
so the compound number is visible too.

Partition check: `sum(buckets) - TC total` must be 0 and `rows_kept ==
rows_total`.  An arm failing either is marked FAILED and excluded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
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

ARM_ORDER = ("inc", "bG1", "bG2", "bG4", "bG8")
_ARM_RE = re.compile(r"^(inc|bG\d+)$")


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
            print(
                f"{key:<34} TC={tc if tc is None else f'{tc:9.3f}'} "
                f"A={res.get('buckets_us', {}).get('stage_a', float('nan')):9.3f} "
                f"B={res.get('buckets_us', {}).get('stage_b_on_tc', float('nan')):9.3f} "
                f"res={res.get('buckets_us', {}).get('residual', float('nan')):8.3f} "
                f"part={res.get('partition_residual_us', float('nan')):+.6f} "
                f"rows={res.get('rows_kept')}/{res.get('rows_total')} "
                f"SC={res.get('plane_totals_us', {}).get('SparseCore', 0.0):.3f}",
                flush=True)

    table: dict[str, dict] = {}
    for key, res in raw.items():
        if "__" not in key:
            print(f"WARN: cannot parse '{key}' as <config>__<arm>")
            continue
        cfg, arm = key.rsplit("__", 1)
        if not _ARM_RE.match(arm):
            print(f"WARN: unexpected arm '{arm}' in '{key}'")
        ok = ("error" not in res and res.get("rows_complete") is True
              and abs(res.get("partition_residual_us", 1.0)) < 1e-6)
        table.setdefault(cfg, {})[arm] = {
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

    for _cfg, arms in table.items():
        for base_arm, suffix in (("bG1", "vs_bG1"), ("inc", "vs_inc")):
            base = arms.get(base_arm)
            if not base:
                continue
            for _arm, v in arms.items():
                for field in ("full", "stage_a", "stage_b", "residual"):
                    v[f"{field}_pct_{suffix}"] = pct(v.get(f"{field}_us"),
                                                     base.get(f"{field}_us"))

    payload = {"raw": raw, "table": table}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"wrote {a.out}")

    def fnum(x, fmt="{:.3f}"):
        return "—" if x is None else fmt.format(x)

    lines = [
        "| config | arm | full µs | Δfull vs bG1 | Δfull vs inc | stage A µs |"
        " ΔA vs bG1 | stage B µs | ΔB vs bG1 | residual µs | Δres vs bG1 |"
        " SC µs | checks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        " ---: | ---: | :--- |",
    ]
    for cfg in sorted(table):
        arms = table[cfg]
        for arm in list(ARM_ORDER) + sorted(set(arms) - set(ARM_ORDER)):
            v = arms.get(arm)
            if v is None:
                continue
            lines.append(
                f"| {cfg} | `{arm}` | {fnum(v['full_us'])} | "
                f"{fnum(v.get('full_pct_vs_bG1'), '{:+.2f}%')} | "
                f"{fnum(v.get('full_pct_vs_inc'), '{:+.2f}%')} | "
                f"{fnum(v['stage_a_us'])} | "
                f"{fnum(v.get('stage_a_pct_vs_bG1'), '{:+.2f}%')} | "
                f"{fnum(v['stage_b_us'])} | "
                f"{fnum(v.get('stage_b_pct_vs_bG1'), '{:+.2f}%')} | "
                f"{fnum(v['residual_us'])} | "
                f"{fnum(v.get('residual_pct_vs_bG1'), '{:+.2f}%')} | "
                f"{fnum(v['sparsecore_us'])} | "
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
