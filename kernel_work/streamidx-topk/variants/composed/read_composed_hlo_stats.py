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
"""Off-pod reader for the composed-candidate job: `hlo_stats` -> tables.

Runs against a synced `cdk job sync-outputs` directory.  The dump call and the
coarse stage-A / stage-B / residual split are `probes/bdt_hlo_stats.py`'s,
imported; the fine per-bucket split is `option-d-hoist/analyze_hlo_stats.py`'s,
imported.  Reusing both by import is what keeps these numbers comparable with
Option C's, Option D's and D1-lite's tables.

## The instrument

**`hlo_stats` device self time, TensorCore rows only.**  Device self time
carries no host dispatch floor, so nothing is subtracted from it -- per-call
wall clock is `max(device, host)`, not `device + host`, and subtracting a floor
produced numbers wrong by up to 10x earlier in this exercise.

The SparseCore plane is reported in its own column and **never added in**.  For
an offloaded op the TensorCore row is the TensorCore's blocking wait and the
SparseCore row is the SparseCore's concurrent execution -- the same interval,
counted twice.

## The two gates

An arm failing either is marked `FAILED` and excluded from every derived
number rather than quietly averaged in:

  * `rows_kept == rows_total`.  A `--top-n`-truncated table looks exactly like
    a complete one *and still passes the partition check*, because the check
    sums only the rows it kept.  A 21-layer program with 21 `lax.cond`s emits
    822-961 rows against a 500 default, so `--top-n` defaults to 4000 here.
  * `|partition_residual| < 1e-6` on the coarse split and
    `|bucket_sum - TC total| < 1e-6` on the fine one.  This is what makes the
    split a verified partition of the call rather than a selection of rows.

## What it computes

1. a per-trace table (`<config>[_r<n>]__<arm>`), every arm against `inc`;
2. a per-config aggregate over replicates, with the **within-job spread**;
3. the **three-way composition residual**

       R = Δ(c_dl) - [ Δ(c) + Δ(dl) ]      where   Δ(x) = x - inc

   Every one of the four terms is measured independently against `inc` in the
   same job.  None is defined as a leftover: an earlier version of this
   exercise defined one arm's term as a difference of two others, which made
   that arm cancel algebraically and produced an additivity check that could
   not fail.  `R` here can fail, and the spread from (2) is the yardstick that
   says whether it did.

Usage:
    python read_composed_hlo_stats.py <synced-dir> --dumps <dir> \
        --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
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

ARM_ORDER = ("inc", "ctl", "c", "dl", "c_dl")
BASE_ARM = "inc"
# The four metrics the brief asks for, at every level.
METRICS = ("full_us", "stage_a_us", "stage_b_us", "residual_us")
REP_RE = re.compile(r"_r\d+$")


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


def _base_cfg(cfg: str) -> str:
    return REP_RE.sub("", cfg)


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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _stdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / (len(xs) - 1))


def fnum(x, fmt="{:.3f}"):
    return "—" if x is None else fmt.format(x)


def dpct(new, base):
    if new is None or base in (None, 0):
        return None
    return 100.0 * (new / base - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--dumps", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--items", action="store_true")
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
        for key, res in ex.map(
                one, [(p, dumps_root, a.top_n) for p in sorted(pbs)]):
            raw[key] = res
            if "error" in res:
                print(f"{key:<46} ERROR {res['error']}")
                continue
            b = res.get("buckets_us", {})
            print(f"{key:<46} TC={res.get('tensorcore_total_us', 0.0):10.3f} "
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

    # ---- per-trace table -------------------------------------------------
    # `table[cfg][arm]`, where `cfg` still carries the `_r<n>` replicate token.
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
        if v is None:
            return None
        x = v.get(field)
        return None if x is None else x / v["layers"]

    # ---- aggregate over replicates --------------------------------------
    # `agg[base_cfg][arm][metric] = {mean, stdev, spread_pct, n, values}` in
    # **microseconds per layer**.  A trace that failed a gate contributes
    # nothing; `n` says how many did contribute.
    agg: dict[str, dict] = {}
    for cfg, arms in table.items():
        base = _base_cfg(cfg)
        for arm, v in arms.items():
            if not v["checks_pass"]:
                continue
            slot = agg.setdefault(base, {}).setdefault(arm, {})
            slot.setdefault("layers", v["layers"])
            for m in METRICS:
                slot.setdefault(m, []).append(per_layer(v, m))
            slot.setdefault("reps", []).append(cfg)

    summary: dict[str, dict] = {}
    for base, arms in agg.items():
        for arm, slot in arms.items():
            out = {"layers": slot["layers"], "reps": sorted(slot["reps"]),
                   "n": len(slot["reps"])}
            for m in METRICS:
                vals = slot[m]
                mu, sd = _mean(vals), _stdev(vals)
                clean = [x for x in vals if x is not None]
                out[m] = {
                    "mean_us_per_layer": mu,
                    "stdev_us_per_layer": sd,
                    "n": len(clean),
                    "values": clean,
                    # Spread as a percentage of the mean: the yardstick the
                    # composition residual is judged against.
                    "spread_pct": (None if (mu in (None, 0) or len(clean) < 2)
                                   else 100.0 * (max(clean) - min(clean)) / mu),
                }
            summary.setdefault(base, {})[arm] = out

    # ---- the three-way composition residual ------------------------------
    # For each metric:  D(x) = mean(x) - mean(inc)   [us/layer]
    #                   R    = D(c_dl) - (D(c) + D(dl))
    # `R` is only defined where all four arms passed their gates.
    comp: dict[str, dict] = {}
    for base, arms in summary.items():
        if not all(x in arms for x in ("inc", "c", "dl", "c_dl")):
            comp[base] = {"error": "not all of inc/c/dl/c_dl passed the gates",
                          "arms_present": sorted(arms)}
            continue
        row = {}
        for m in METRICS:
            inc = arms["inc"][m]["mean_us_per_layer"]
            d = {x: (arms[x][m]["mean_us_per_layer"] - inc)
                 for x in ("ctl", "c", "dl", "c_dl") if x in arms}
            if inc in (None, 0):
                continue
            r = d["c_dl"] - (d["c"] + d["dl"])
            # The error bar: the largest within-job spread among the four arms,
            # expressed in us/layer, added in quadrature over the three
            # independent measurements that enter R.
            sds = [arms[x][m]["stdev_us_per_layer"] for x in
                   ("inc", "c", "dl", "c_dl")
                   if arms[x][m]["stdev_us_per_layer"] is not None]
            # R = c_dl - c - dl + inc : four independent means.
            err = (math.sqrt(sum(s**2 for s in sds)) if len(sds) == 4 else None)
            spreads = [arms[x][m]["spread_pct"] for x in
                       ("inc", "ctl", "c", "dl", "c_dl")
                       if x in arms and arms[x][m]["spread_pct"] is not None]
            row[m] = {
                "inc_us_per_layer": inc,
                "delta_us_per_layer": d,
                "delta_pct": {k: 100.0 * v / inc for k, v in d.items()},
                "sum_of_parts_us_per_layer": d["c"] + d["dl"],
                "composition_residual_us_per_layer": r,
                "composition_residual_pct_of_inc": 100.0 * r / inc,
                "residual_error_bar_us_per_layer": err,
                "max_within_job_spread_pct": max(spreads) if spreads else None,
                "inside_error_bar": (None if err is None else abs(r) <= err),
                "additivity_realised_pct": (
                    None if (d["c"] + d["dl"]) == 0
                    else 100.0 * d["c_dl"] / (d["c"] + d["dl"])),
            }
        comp[base] = row

    # ---- markdown --------------------------------------------------------
    lines: list[str] = []

    lines.append("## Per-trace table\n")
    lines.append("One row per traced program.  `µs/layer` is the whole "
                 "`streamindex_topk` call's TensorCore device self time "
                 "divided by the number of chained layers in the program.\n")
    for cfg in sorted(table):
        arms = table[cfg]
        base = arms.get(BASE_ARM)
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
            b_fl = per_layer(base, "full_us")
            b_rs = per_layer(base, "residual_us")
            lines.append(
                f"| `{arm}` | {fnum(fl)} | {fnum(dpct(fl, b_fl), '{:+.2f}%')} |"
                f" {fnum(per_layer(v, 'stage_a_us'))} |"
                f" {fnum(per_layer(v, 'stage_b_us'))} | {fnum(rs)} |"
                f" {fnum(None if (rs is None or b_rs is None) else rs - b_rs, '{:+.3f}')} |"
                f" {fnum(v['sparsecore_us'])} |"
                f" {v['rows_kept']}/{v['rows_total']} |"
                f" {fnum(v['partition_residual_us'], '{:+.1e}')} |"
                f" {'ok' if v['checks_pass'] else 'FAILED'} |")

        keys = [k for k in FINE.ORDER
                if any((arms[x].get("fine_per_layer") or {}).get(k)
                       for x in arms)]
        if keys:
            present = [x for x in ARM_ORDER if x in arms]
            lines.append("")
            lines.append("| bucket µs/layer | " +
                         " | ".join(f"`{x}`" for x in present) + " |")
            lines.append("| --- |" + " ---: |" * len(present))
            for k in keys:
                row = []
                for x in present:
                    fp = arms[x].get("fine_per_layer") or {}
                    n = (arms[x].get("fine_row_counts") or {}).get(k)
                    row.append(f"{fp.get(k, 0.0):.3f}"
                               + (f" ({n})" if n else ""))
                lines.append(f"| `{k}` | " + " | ".join(row) + " |")

    lines.append("\n\n## Replicate aggregate and within-job spread\n")
    lines.append("`n` is the number of replicate traces that passed both "
                 "gates.  `spread` is `(max - min) / mean` over those "
                 "replicates -- the loop was re-entered for each, so this is "
                 "run-to-run variation, not within-loop jitter.\n")
    for base in sorted(summary):
        arms = summary[base]
        if max(v["n"] for v in arms.values()) < 2:
            continue
        lines.append(f"\n#### `{base}`\n")
        lines.append("| arm | n | full µs/layer | sd | spread | Δfull vs inc |"
                     " stage A µs/layer | stage B µs/layer |"
                     " residual µs/layer |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                     " ---: |")
        inc = arms.get(BASE_ARM)
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            v = arms[arm]
            f = v["full_us"]
            b = inc["full_us"]["mean_us_per_layer"] if inc else None
            lines.append(
                f"| `{arm}` | {v['n']} | {fnum(f['mean_us_per_layer'])} |"
                f" {fnum(f['stdev_us_per_layer'])} |"
                f" {fnum(f['spread_pct'], '{:.3f}%')} |"
                f" {fnum(dpct(f['mean_us_per_layer'], b), '{:+.2f}%')} |"
                f" {fnum(v['stage_a_us']['mean_us_per_layer'])} |"
                f" {fnum(v['stage_b_us']['mean_us_per_layer'])} |"
                f" {fnum(v['residual_us']['mean_us_per_layer'])} |")

    lines.append("\n\n## Composition residual\n")
    lines.append("`Δ(x) = mean(x) − mean(inc)`, µs/layer, each arm measured "
                 "independently against `inc` in this job.  "
                 "`R = Δ(c_dl) − [Δ(c) + Δ(dl)]`.  `R < 0` means the two "
                 "changes together beat the sum of their parts; `R > 0` means "
                 "they overlap.  The error bar is the four arms' replicate "
                 "standard deviations added in quadrature; where there are no "
                 "replicates it is `—` and `R` has no error bar in this job.\n")
    for base in sorted(comp):
        row = comp[base]
        lines.append(f"\n#### `{base}`\n")
        if "error" in row:
            lines.append(f"*{row['error']}* (arms present: "
                         f"{row.get('arms_present')})")
            continue
        lines.append("| metric | inc µs/layer | Δctl | Δc | Δdl | Δc_dl |"
                     " Δc+Δdl | R | R as % of inc | error bar | inside? |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                     " ---: | ---: | :--- |")
        for m in METRICS:
            if m not in row:
                continue
            r = row[m]
            d = r["delta_us_per_layer"]
            inside = r["inside_error_bar"]
            lines.append(
                f"| {m} | {fnum(r['inc_us_per_layer'])} |"
                f" {fnum(d.get('ctl'), '{:+.3f}')} |"
                f" {fnum(d.get('c'), '{:+.3f}')} |"
                f" {fnum(d.get('dl'), '{:+.3f}')} |"
                f" {fnum(d.get('c_dl'), '{:+.3f}')} |"
                f" {fnum(r['sum_of_parts_us_per_layer'], '{:+.3f}')} |"
                f" {fnum(r['composition_residual_us_per_layer'], '{:+.3f}')} |"
                f" {fnum(r['composition_residual_pct_of_inc'], '{:+.3f}%')} |"
                f" {fnum(r['residual_error_bar_us_per_layer'], '±{:.3f}')} |"
                f" {'—' if inside is None else ('yes' if inside else 'NO')} |")

    md = "\n".join(lines)
    print(md)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md + "\n")
        print(f"\nwrote {a.md}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"table": table, "summary": summary, "composition": comp,
                       "raw": raw, "fine": fine}, fh, indent=1, default=str)
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
