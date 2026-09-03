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
"""Off-pod reader: `hlo_stats` device self time for the KV-scatter A/B.

Runs against a synced `cdk job sync-outputs` directory.  The dump and the
bucketing are `probes/bdt_hlo_stats.py`'s, reused by import rather than
reimplemented, with one bucket added.

## Buckets

| bucket     | rows | what it is |
| ---        | ---  | --- |
| `stage_a`  | `StreamIdxTC*`             | the scoring Pallas kernel |
| `stage_b`  | `sc_topk*`                 | the SparseCore top-k, as charged to the TensorCore (the TC is blocked for its duration) |
| `triad`    | `reshape* / *scatter* / copy* / bitcast* / transpose*` | the KV-cache write and any relayout around it -- the thing this exercise is about |
| `residual` | everything else on the TensorCore | the `-inf` init, the fp8 bitcast, `searchsorted`, the pad |

`triad` is carved out of what `bdt_hlo_stats` calls `residual`; the two
together are that module's `residual`, so nothing is double counted.  The
partition check below is what enforces that.

## Checks, and what a failure means

* `partition_residual` = `sum(buckets) - TensorCore total` must be `< 1e-6`.
* `rows_kept == rows_total`: a dump truncated by `--top-n` silently loses rows,
  which is how an earlier probe in this exercise under-reported a total.

An arm failing either is printed as `FAILED` and must not be quoted.

The SparseCore plane is reported separately and never added to the TensorCore
total -- the two planes run concurrently.

Usage:
    python read_kvs_hlo_stats.py <synced-dir> --out kvs_hlo_stats.json \\
        --md kvs_hlo_stats.md
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

STAGE_A = re.compile(r"StreamIdxTC")
STAGE_B = re.compile(r"sc_topk|sparsecore_topk|SparseCoreTopK")
TRIAD = re.compile(r"reshape|scatter|^copy|bitcast|transpose", re.I)

ARM_ORDER = ("base", "v128", "v128rt", "scat256", "scat128", "scat128rt")
CFG_ORDER = ("pre1024", "dec64")


def analyse(d: dict) -> dict:
    data = d["tools"]["hlo_stats"]["data"]
    rows = data["rows"]
    out = {
        "rows_total": data.get("rows_total"),
        "rows_kept": data.get("rows_kept"),
        "rows_complete": data.get("rows_total") == data.get("rows_kept"),
    }
    planes: dict[str, float] = {}
    buckets = {"stage_a": 0.0, "stage_b": 0.0, "triad": 0.0, "residual": 0.0}
    detail: dict[str, list] = {k: [] for k in buckets}
    detail["sparsecore"] = []
    for r in rows:
        core = r.get("TPU core type", "?")
        us = float(r.get("Avg. self time (us)", 0.0) or 0.0)
        name = str(r.get("HLO op name", ""))
        occ = r.get("Occurrences")
        planes[core] = planes.get(core, 0.0) + us
        if core != "TensorCore":
            detail["sparsecore"].append([name, us, occ])
            continue
        if STAGE_A.search(name):
            b = "stage_a"
        elif STAGE_B.search(name):
            b = "stage_b"
        elif TRIAD.search(name):
            b = "triad"
        else:
            b = "residual"
        buckets[b] += us
        detail[b].append([name, us, occ])
    tc = planes.get("TensorCore", 0.0)
    out.update(
        plane_totals_us=planes,
        tensorcore_total_us=tc,
        buckets_us=buckets,
        partition_residual_us=sum(buckets.values()) - tc,
        # Every triad row, not a top-N: the whole point is to name the ops.
        triad_rows=sorted(detail["triad"], key=lambda x: -x[1]),
        top_rows={
            k: sorted(v, key=lambda x: -x[1])[:10]
            for k, v in detail.items() if k != "triad"
        },
    )
    return out


def one(pb: str) -> tuple[str, dict]:
    parts = pb.split(os.sep)
    key = "unknown"
    if "plugins" in parts:
        key = parts[parts.index("plugins") - 1]
    with tempfile.TemporaryDirectory() as td:
        try:
            res = analyse(BDT.dump(pb, td, top_n=2000))
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
            b = res.get("buckets_us", {})
            print(f"{key:<26} TC={res.get('tensorcore_total_us', float('nan')):9.3f} "
                  f"A={b.get('stage_a', float('nan')):9.3f} "
                  f"B={b.get('stage_b', float('nan')):8.3f} "
                  f"triad={b.get('triad', float('nan')):8.3f} "
                  f"res={b.get('residual', float('nan')):7.3f} "
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
        ok = ("error" not in res and res.get("rows_complete") is True
              and abs(res.get("partition_residual_us", 1.0)) < 1e-6)
        b = res.get("buckets_us", {})
        table.setdefault(cfg, {})[arm] = {
            "full_us": res.get("tensorcore_total_us"),
            "stage_a_us": b.get("stage_a"),
            "stage_b_us": b.get("stage_b"),
            "triad_us": b.get("triad"),
            "residual_us": b.get("residual"),
            "sparsecore_us": res.get("plane_totals_us", {}).get("SparseCore"),
            "rows_kept": res.get("rows_kept"),
            "rows_total": res.get("rows_total"),
            "partition_residual_us": res.get("partition_residual_us"),
            "checks_pass": ok,
            "error": res.get("error"),
            "triad_rows": res.get("triad_rows"),
            "xplane": res.get("xplane"),
        }

    def delta(new, base):
        if new is None or base is None:
            return None
        return new - base

    for _cfg, arms in table.items():
        # Scatter-only arms are compared with `scat256`, full arms with `base`.
        for group, base_arm in ((("base", "v128", "v128rt"), "base"),
                                (("scat256", "scat128", "scat128rt"),
                                 "scat256")):
            base = arms.get(base_arm)
            if not base:
                continue
            for arm in group:
                v = arms.get(arm)
                if v is None:
                    continue
                v["baseline_arm"] = base_arm
                for f in ("full", "stage_a", "stage_b", "triad", "residual"):
                    v[f"{f}_delta_us"] = delta(v.get(f"{f}_us"),
                                               base.get(f"{f}_us"))

    payload = {"raw": raw, "table": table}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"wrote {a.out}")

    def fnum(x, fmt="{:.3f}"):
        return "—" if x is None else fmt.format(x)

    lines = [
        "| config | arm | vs | full µs | Δfull | stage A µs | ΔA | stage B µs"
        " | triad µs | Δtriad | residual µs | SC µs | checks |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        " ---: | ---: | :--- |",
    ]
    for cfg in [c for c in CFG_ORDER if c in table] + sorted(
            set(table) - set(CFG_ORDER)):
        arms = table[cfg]
        for arm in list(ARM_ORDER) + sorted(set(arms) - set(ARM_ORDER)):
            v = arms.get(arm)
            if v is None:
                continue
            lines.append(
                f"| {cfg} | `{arm}` | `{v.get('baseline_arm', '—')}` | "
                f"{fnum(v['full_us'])} | {fnum(v.get('full_delta_us'), '{:+.3f}')} | "
                f"{fnum(v['stage_a_us'])} | {fnum(v.get('stage_a_delta_us'), '{:+.3f}')} | "
                f"{fnum(v['stage_b_us'])} | "
                f"{fnum(v['triad_us'])} | {fnum(v.get('triad_delta_us'), '{:+.3f}')} | "
                f"{fnum(v['residual_us'])} | {fnum(v['sparsecore_us'])} | "
                f"{'ok' if v['checks_pass'] else 'FAILED'} |")

    lines.append("")
    lines.append("### Every op in the `triad` bucket, per arm")
    lines.append("")
    lines.append("| config | arm | HLO op name | self µs | occurrences |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for cfg in [c for c in CFG_ORDER if c in table] + sorted(
            set(table) - set(CFG_ORDER)):
        for arm in list(ARM_ORDER) + sorted(set(table[cfg]) - set(ARM_ORDER)):
            v = table[cfg].get(arm)
            if not v:
                continue
            for row in (v.get("triad_rows") or []):
                lines.append(f"| {cfg} | `{arm}` | `{row[0]}` | {row[1]:.3f} |"
                             f" {row[2]} |")
            if not v.get("triad_rows"):
                lines.append(f"| {cfg} | `{arm}` | *(none)* | 0.000 | 0 |")

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
