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
"""Off-pod reader for the D1-lite job: `hlo_stats` device self time -> tables.

Runs against a synced `cdk job sync-outputs` directory.  Nothing here is new
machinery: the dump call and the coarse A/B/residual split are
`probes/bdt_hlo_stats.py`'s, imported; the fine per-bucket split (`cond`,
`metadata`, `prefetch_c3`, ...) is `option-d-hoist/analyze_hlo_stats.py`'s,
imported.  Reusing both by import is what makes the D1-lite numbers directly
comparable to Option D's table.

Two gates, both enforced rather than assumed, and an arm that fails either is
marked FAILED and excluded from the headline rather than quietly averaged in:

  * `rows_kept == rows_total` -- a `--top-n`-truncated table looks exactly like
    a complete one.
  * `|sum(buckets) - TensorCore total| < 1e-6` -- what makes the split a
    verified partition of the call rather than a selection of rows.

The SparseCore plane is reported and never added: the TensorCore is blocked for
the SparseCore's duration, so the `sc_topk...call-done` TensorCore row already
carries stage B.

Usage:
    python read_d1lite_hlo_stats.py <synced-dir> --dumps <dir> \
        --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VARIANTS = os.path.abspath(os.path.join(_HERE, os.pardir))
_PROBES = os.path.abspath(os.path.join(_VARIANTS, os.pardir, "probes"))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BDT = _load("bdt_hlo_stats", os.path.join(_PROBES, "bdt_hlo_stats.py"))
FINE = _load("analyze_hlo_stats",
             os.path.join(_VARIANTS, "option-d-hoist", "analyze_hlo_stats.py"))

ARM_ORDER = ("inc", "dlctl", "dl", "dlc", "d1")


def _key_of(pb: str) -> str:
    parts = pb.split(os.sep)
    if "plugins" in parts:
        return parts[parts.index("plugins") - 1]
    return "unknown"


def _layers_of(tag: str) -> int:
    for token in tag.split("_"):
        if token.startswith("L") and token[1:].isdigit():
            return int(token[1:])
    return 1


def one(args):
    pb, dumps_root, top_n = args
    key = _key_of(pb)
    outdir = os.path.join(dumps_root, key)
    os.makedirs(outdir, exist_ok=True)
    try:
        doc = BDT.dump(pb, outdir, top_n=top_n)
    except Exception as err:  # noqa: BLE001
        return key, {"error": f"dump: {repr(err)[:300]}", "xplane": pb}
    res = BDT.analyse(doc)
    res["xplane"] = pb
    res["dump_dir"] = outdir
    return key, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--dumps", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--items", action="store_true")
    # `--top-n` must exceed the row count of the biggest table.  A 21-layer
    # `lax.cond` module emits ~1000 distinct HLO ops; the 500 default
    # silently truncates, and a truncated table still passes the coarse
    # partition check because the check sums only the rows it kept.  The
    # `rows_kept == rows_total` gate is what catches it.
    ap.add_argument("--top-n", type=int, default=4000)
    a = ap.parse_args()

    dumps_root = a.dumps or os.path.join(a.synced, "_dumps")
    os.makedirs(dumps_root, exist_ok=True)

    pbs = []
    for root, _d, files in os.walk(a.synced):
        if os.path.abspath(root).startswith(os.path.abspath(dumps_root)):
            continue
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    if not pbs:
        raise SystemExit(f"no *.xplane.pb under {a.synced}")
    print(f"{len(pbs)} xplane(s)")

    raw: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for key, res in ex.map(one, [(p, dumps_root, a.top_n) for p in sorted(pbs)]):
            raw[key] = res
            if "error" in res:
                print(f"{key:<40} ERROR {res['error']}")
                continue
            b = res.get("buckets_us", {})
            print(f"{key:<40} TC={res.get('tensorcore_total_us', 0.0):10.3f} "
                  f"A={b.get('stage_a', 0.0):9.3f} "
                  f"B={b.get('stage_b_on_tc', 0.0):9.3f} "
                  f"res={b.get('residual', 0.0):9.3f} "
                  f"part={res.get('partition_residual_us', float('nan')):+.6f} "
                  f"rows={res.get('rows_kept')}/{res.get('rows_total')} "
                  f"SC={res.get('plane_totals_us', {}).get('SparseCore', 0.0):.3f}",
                  flush=True)

    # ---- fine buckets, via option-d's summarize -------------------------
    fine: dict[str, dict] = {}
    for key, res in raw.items():
        if "error" in res:
            continue
        dumped = None
        for root, _d, files in os.walk(res["dump_dir"]):
            for f in files:
                if f.endswith(".json"):
                    with open(os.path.join(root, f)) as fh:
                        if "hlo_stats" in json.load(fh).get("tools", {}):
                            dumped = os.path.join(root, f)
        if dumped is None:
            continue
        tag = key.rsplit("__", 1)[0]
        try:
            fine[key] = FINE.summarize(key, dumped, _layers_of(tag))
        except SystemExit as err:
            fine[key] = {"error": str(err)}

    # ---- config x arm ----------------------------------------------------
    table: dict[str, dict] = {}
    for key, res in raw.items():
        if "__" not in key:
            print(f"WARN: cannot parse {key!r} as <config>__<arm>")
            continue
        cfg, arm = key.rsplit("__", 1)
        layers = _layers_of(cfg)
        f = fine.get(key, {})
        ok = ("error" not in res and res.get("rows_complete") is True
              and abs(res.get("partition_residual_us", 1.0)) < 1e-6
              and "error" not in f
              and abs(f.get("bucket_sum_minus_tc_total", 1.0)) < 1e-6)
        b = res.get("buckets_us", {})
        table.setdefault(cfg, {})[arm] = {
            "layers": layers,
            "full_us": res.get("tensorcore_total_us"),
            "stage_a_us": b.get("stage_a"),
            "stage_b_us": b.get("stage_b_on_tc"),
            "residual_us": b.get("residual"),
            "sparsecore_us": res.get("plane_totals_us", {}).get("SparseCore"),
            "rows_kept": res.get("rows_kept"),
            "rows_total": res.get("rows_total"),
            "partition_residual_us": res.get("partition_residual_us"),
            "fine_partition_check": f.get("bucket_sum_minus_tc_total"),
            "fine_per_layer": f.get("buckets_us_per_layer"),
            "fine_row_counts": {k: len(v)
                                for k, v in (f.get("items") or {}).items()},
            "checks_pass": ok,
            "error": res.get("error") or f.get("error"),
            "xplane": res.get("xplane"),
        }

    def per_layer(v, field):
        x = v.get(field)
        return None if x is None else x / v["layers"]

    def dpct(new, base):
        if new is None or base in (None, 0):
            return None
        return 100.0 * (new / base - 1.0)

    def fnum(x, fmt="{:.3f}"):
        return "—" if x is None else fmt.format(x)

    lines: list[str] = []
    for cfg in sorted(table):
        arms = table[cfg]
        base = arms.get("inc")
        lines.append(f"\n#### `{cfg}`  (layers = {_layers_of(cfg)})\n")
        lines.append("| arm | full µs/layer | Δfull vs inc | stage A µs/layer |"
                     " stage B µs/layer | residual µs/layer | Δresidual µs/layer"
                     " | SC µs/prog | rows | partition | checks |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                     " :--- | ---: | :--- |")
        for arm in list(ARM_ORDER) + sorted(set(arms) - set(ARM_ORDER)):
            v = arms.get(arm)
            if v is None:
                continue
            fl = per_layer(v, "full_us")
            rs = per_layer(v, "residual_us")
            b_fl = per_layer(base, "full_us") if base else None
            b_rs = per_layer(base, "residual_us") if base else None
            lines.append(
                f"| `{arm}` | {fnum(fl)} | {fnum(dpct(fl, b_fl), '{:+.2f}%')} |"
                f" {fnum(per_layer(v, 'stage_a_us'))} |"
                f" {fnum(per_layer(v, 'stage_b_us'))} | {fnum(rs)} |"
                f" {fnum(None if (rs is None or b_rs is None) else rs - b_rs, '{:+.3f}')} |"
                f" {fnum(v['sparsecore_us'])} |"
                f" {v['rows_kept']}/{v['rows_total']} |"
                f" {fnum(v['partition_residual_us'], '{:+.1e}')} |"
                f" {'ok' if v['checks_pass'] else 'FAILED'} |")

        # fine buckets
        keys = [k for k in FINE.ORDER
                if any((arms[x].get("fine_per_layer") or {}).get(k)
                       for x in arms)]
        if keys:
            lines.append("")
            lines.append("| bucket µs/layer | " +
                         " | ".join(f"`{x}`" for x in ARM_ORDER
                                    if x in arms) + " |")
            lines.append("| --- |" + " ---: |" * len(
                [x for x in ARM_ORDER if x in arms]))
            for k in keys:
                row = []
                for x in ARM_ORDER:
                    if x not in arms:
                        continue
                    fp = arms[x].get("fine_per_layer") or {}
                    n = (arms[x].get("fine_row_counts") or {}).get(k)
                    row.append(f"{fp.get(k, 0.0):.3f}"
                               + (f" ({n})" if n else ""))
                lines.append(f"| `{k}` | " + " | ".join(row) + " |")

    md = "\n".join(lines)
    print(md)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md + "\n")
        print(f"\nwrote {a.md}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"table": table, "raw": raw, "fine": fine}, fh, indent=1,
                      default=str)
        print(f"wrote {a.json}")

    if a.items:
        for key in sorted(fine):
            s = fine[key]
            if "items" not in s:
                continue
            print(f"\n### {key} -- every TensorCore row, by bucket\n")
            for k, rows in s["items"].items():
                print(f"  [{k}] {s['buckets_us_per_prog'][k]:.3f} us/prog "
                      f"({len(rows)} rows)")
                for us, name, fw in rows[:12]:
                    print(f"      {us:9.3f}  {name[:64]:64} | {fw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
