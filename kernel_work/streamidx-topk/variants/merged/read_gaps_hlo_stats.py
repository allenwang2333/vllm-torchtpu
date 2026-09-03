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
"""Off-pod reader for the gaps probe: `hlo_stats` device self time -> tables.

The dump and the bucketing are `read_merged_hlo_stats.one()`'s, imported
rather than reimplemented, which is what makes a number here comparable to one
in `RESULTS.md`: full op = the TensorCore plane total, stage A = the
`StreamIdxTC*` rows (the `-hm` suffix Option C adds matches the same regex),
stage B = the `sc_topk*` rows on TensorCore, residual = the rest.  The
SparseCore plane is reported separately and never added.

Every arm is gated on `|partition_residual| < 1e-6` and
`rows_kept == rows_total` before it enters a table.  An arm that fails either
is printed with its numbers and excluded from the arithmetic, not quietly
averaged in.

Joining is by the `tag` recorded in `trace_meta.<plan>.json`, not by parsing
the directory name.  E1 and E3 both measure the production shape -- on
purpose, on two different chips -- so the config name alone is not unique.

Usage:
    python read_gaps_hlo_stats.py <synced-dir> --out gaps.json --md gaps.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RM = _load(os.path.join(_HERE, "read_merged_hlo_stats.py"),
           "read_merged_hlo_stats")

FIELDS = ("full", "stage_a", "stage_b", "residual")


def checks_pass(res: dict) -> bool:
    return ("error" not in res
            and res.get("rows_complete") is True
            and abs(res.get("partition_residual_us", 1.0)) < 1e-6)


def flatten(res: dict) -> dict:
    b = res.get("buckets_us", {})
    return {
        "full_us": res.get("tensorcore_total_us"),
        "stage_a_us": b.get("stage_a"),
        "stage_b_us": b.get("stage_b_on_tc"),
        "residual_us": b.get("residual"),
        "sparsecore_us": res.get("plane_totals_us", {}).get("SparseCore"),
        "rows_kept": res.get("rows_kept"),
        "rows_total": res.get("rows_total"),
        "partition_residual_us": res.get("partition_residual_us"),
        "checks_pass": checks_pass(res),
        "error": res.get("error"),
        "xplane": res.get("xplane"),
    }


def fnum(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return 100.0 * (new / base - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    # ---- meta ----------------------------------------------------------
    metas, runs = {}, {}
    for root, _d, files in os.walk(a.synced):
        for f in files:
            if f.startswith("trace_meta.") and f.endswith(".json"):
                with open(os.path.join(root, f)) as fh:
                    m = json.load(fh)
                metas[m.get("plan", f)] = m
                for r in m.get("runs", []):
                    if r.get("tag"):
                        runs[r["tag"]] = r
    print(f"{len(metas)} meta file(s), {len(runs)} recorded run(s)")
    for plan, m in sorted(metas.items()):
        print(f"  plan {plan}: jax={m.get('jax')} md5_ok={m.get('md5_ok')} "
              f"runs={len(m.get('runs', []))}")
        for name, info in m.get("modules", {}).items():
            print(f"    {name:<32} md5={info['md5']} ok={info['ok']}")

    # ---- xplanes -------------------------------------------------------
    pbs = []
    for root, _d, files in os.walk(a.synced):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    if not pbs:
        raise SystemExit(f"no *.xplane.pb under {a.synced}")
    print(f"{len(pbs)} xplane(s)")

    raw = {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for tag, res in ex.map(RM.one, sorted(pbs)):
            raw[tag] = res
            print(f"{tag:<44} "
                  f"TC={fnum(res.get('tensorcore_total_us'), '{:9.3f}')} "
                  f"A={fnum(res.get('buckets_us', {}).get('stage_a'), '{:9.3f}')} "
                  f"B={fnum(res.get('buckets_us', {}).get('stage_b_on_tc'), '{:9.3f}')} "
                  f"res={fnum(res.get('buckets_us', {}).get('residual'), '{:8.3f}')} "
                  f"part={fnum(res.get('partition_residual_us'), '{:+.6f}')} "
                  f"rows={res.get('rows_kept')}/{res.get('rows_total')} "
                  f"pass={checks_pass(res)}", flush=True)

    rows = {}
    for tag, res in raw.items():
        rec = runs.get(tag, {})
        rows[tag] = {"tag": tag, **{k: rec.get(k) for k in (
            "plan", "experiment", "config", "arm", "rep", "phase", "ctx",
            "max_model_len", "num_seqs", "q_len", "spec", "scan", "check",
            "num_sublanes_total_for_tuple", "bkv_p_ladder")},
            **flatten(res)}
        if not rec:
            print(f"WARN: no meta record for tag {tag}")

    out_md = []

    def emit(line=""):
        out_md.append(line)
        print(line)

    # ---- raw per-trace table -------------------------------------------
    emit("## Every trace")
    emit()
    emit("| plan | config | arm | rep | full us | stage A us | stage B us | "
         "residual us | SC us | part | rows | checks |")
    emit("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
         " ---: | :--- |")
    for tag in sorted(rows):
        v = rows[tag]
        emit(f"| {v['plan']} | {v['config']} | `{v['arm']}` | {v['rep']} | "
             f"{fnum(v['full_us'])} | {fnum(v['stage_a_us'])} | "
             f"{fnum(v['stage_b_us'])} | {fnum(v['residual_us'])} | "
             f"{fnum(v['sparsecore_us'])} | "
             f"{fnum(v['partition_residual_us'], '{:+.6f}')} | "
             f"{v['rows_kept']}/{v['rows_total']} | "
             f"{'ok' if v['checks_pass'] else 'FAILED'} |")
    emit()

    # ---- correctness ----------------------------------------------------
    emit("## Index check vs `inc` (same inputs, before any timing)")
    emit()
    emit("| plan | config | arm | index_identical | array_identical | "
         "rows_mismatched / rows |")
    emit("| --- | --- | --- | :--- | :--- | ---: |")
    seen = set()
    for tag in sorted(rows):
        v = rows[tag]
        key = (v["plan"], v["config"], v["arm"])
        if key in seen or not v.get("check"):
            continue
        seen.add(key)
        c = v["check"]
        if c.get("is_reference"):
            emit(f"| {v['plan']} | {v['config']} | `{v['arm']}` | "
                 f"(reference) | (reference) | -- |")
            continue
        if c.get("error"):
            emit(f"| {v['plan']} | {v['config']} | `{v['arm']}` | ERROR | "
                 f"{c['error'][:60]} | -- |")
            continue
        emit(f"| {v['plan']} | {v['config']} | `{v['arm']}` | "
             f"{c['index_identical']} | {c['array_identical']} | "
             f"{c['rows_mismatched']}/{c['rows_total']} |")
    emit()

    payload = {"metas": metas, "rows": rows, "analysis": {}}

    def group(pred):
        g = {}
        for v in rows.values():
            if pred(v) and v["checks_pass"]:
                g.setdefault((v["config"], v["arm"]), []).append(v)
        return g

    # ---- E1: within-job replicate --------------------------------------
    e1 = group(lambda v: v["experiment"] == "E1")
    if e1:
        emit("## E1 -- within-job replicate at the production config")
        emit()
        emit("| arm | n | mean full us | min | max | spread % of mean | "
             "sd % | mean stage A us | stage A spread % |")
        emit("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        e1a = {}
        for (cfg, arm), vs in sorted(e1.items()):
            f = sorted(v["full_us"] for v in vs)
            sa = sorted(v["stage_a_us"] for v in vs)
            mean = statistics.fmean(f)
            spread = 100.0 * (f[-1] - f[0]) / mean
            sd = 100.0 * statistics.stdev(f) / mean if len(f) > 1 else 0.0
            sam = statistics.fmean(sa)
            saspread = 100.0 * (sa[-1] - sa[0]) / sam
            e1a[arm] = {"n": len(f), "mean_full_us": mean, "min_full_us": f[0],
                        "max_full_us": f[-1], "spread_pct": spread,
                        "sd_pct": sd, "values_us": f,
                        "mean_stage_a_us": sam,
                        "stage_a_spread_pct": saspread}
            emit(f"| `{arm}` | {len(f)} | {mean:.3f} | {f[0]:.3f} | "
                 f"{f[-1]:.3f} | {spread:.3f}% | {sd:.3f}% | {sam:.3f} | "
                 f"{saspread:.3f}% |")
        worst = max(e1a.values(), key=lambda d: d["spread_pct"])
        emit()
        emit(f"Largest within-job spread across arms: "
             f"**{worst['spread_pct']:.3f}%** of that arm's mean.")
        # additivity residual recomputed from the E1 means
        if {"inc", "bkvp", "d1", "merged"} <= set(e1a):
            base = e1a["inc"]["mean_full_us"]
            d = {k: e1a[k]["mean_full_us"] - base for k in e1a}
            emit()
            emit("| quantity | us | % of `inc` |")
            emit("| --- | ---: | ---: |")
            for k in sorted(d):
                emit(f"| d({k}) | {d[k]:+.3f} | {100 * d[k] / base:+.3f}% |")
            if "bkvp_d1" in d:
                r2 = d["bkvp_d1"] - d["bkvp"] - d["d1"]
                emit(f"| two-way residual d(bkvp_d1)-d(bkvp)-d(d1) | "
                     f"{r2:+.3f} | {100 * r2 / base:+.3f}% |")
            if "c" in d and "bkvp_d1" in d:
                r3 = d["merged"] - (d["bkvp"] + d["d1"] + d["c"])
                emit(f"| three-way residual d(merged)-[d(bkvp)+d(d1)+d(c)] | "
                     f"{r3:+.3f} | {100 * r3 / base:+.3f}% |")
        emit()
        payload["analysis"]["E1"] = e1a

    # ---- E2: the genuine three-way residual ----------------------------
    e2 = group(lambda v: v["experiment"] == "E2")
    if e2:
        emit("## E2 -- Option C alone, and the genuine three-way residual")
        emit()
        by_cfg = {}
        for (cfg, arm), vs in e2.items():
            by_cfg.setdefault(cfg, {})[arm] = statistics.fmean(
                v["full_us"] for v in vs)
        sa_by_cfg = {}
        for (cfg, arm), vs in e2.items():
            sa_by_cfg.setdefault(cfg, {})[arm] = statistics.fmean(
                v["stage_a_us"] for v in vs)
        emit("| config | `inc` us | d(bkvp) | d(d1) | d(c) | d(merged) | "
             "sum of parts | three-way residual | as % of `inc` | "
             "old two-way residual |")
        emit("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
             " ---: |")
        e2a = {}
        for cfg in sorted(by_cfg):
            m = by_cfg[cfg]
            if "inc" not in m:
                continue
            base = m["inc"]
            d = {k: v - base for k, v in m.items()}
            row = {"inc_us": base, "delta_us": d,
                   "delta_pct": {k: 100 * v / base for k, v in d.items()}}
            if {"bkvp", "d1", "c", "merged"} <= set(d):
                s = d["bkvp"] + d["d1"] + d["c"]
                r3 = d["merged"] - s
                row.update(sum_of_parts_us=s, residual3_us=r3,
                           residual3_pct_of_inc=100 * r3 / base)
                r2 = (d["bkvp_d1"] - d["bkvp"] - d["d1"]
                      if "bkvp_d1" in d else None)
                row["residual2_us"] = r2
                emit(f"| `{cfg}` | {base:.3f} | {d['bkvp']:+.3f} | "
                     f"{d['d1']:+.3f} | {d['c']:+.3f} | {d['merged']:+.3f} | "
                     f"{s:+.3f} | **{r3:+.3f}** | "
                     f"**{100 * r3 / base:+.3f}%** | "
                     f"{fnum(r2, '{:+.3f}')} |")
            e2a[cfg] = row
        emit()
        emit("`c` and `cctl`, in full (full op / stage A):")
        emit()
        emit("| config | arm | full us | d full % vs inc | stage A us | "
             "d stage A % vs inc |")
        emit("| --- | --- | ---: | ---: | ---: | ---: |")
        for cfg in sorted(by_cfg):
            base = by_cfg[cfg].get("inc")
            sbase = sa_by_cfg[cfg].get("inc")
            for arm in ("inc", "cctl", "c", "d1", "bkvp", "bkvp_d1",
                        "merged"):
                if arm not in by_cfg[cfg]:
                    continue
                emit(f"| `{cfg}` | `{arm}` | {by_cfg[cfg][arm]:.3f} | "
                     f"{fnum(pct(by_cfg[cfg][arm], base), '{:+.3f}%')} | "
                     f"{sa_by_cfg[cfg][arm]:.3f} | "
                     f"{fnum(pct(sa_by_cfg[cfg][arm], sbase), '{:+.3f}%')} |")
        emit()
        payload["analysis"]["E2"] = e2a

    # ---- E3: ctx decoupled from max_model_len --------------------------
    e3 = group(lambda v: v["experiment"] == "E3")
    if e3:
        emit("## E3 -- runtime ctx swept under a fixed max_model_len 9216")
        emit()
        by_ctx = {}
        for (cfg, arm), vs in e3.items():
            ctx = vs[0]["ctx"]
            by_ctx.setdefault(ctx, {})[arm] = {
                "full_us": statistics.fmean(v["full_us"] for v in vs),
                "stage_a_us": statistics.fmean(v["stage_a_us"] for v in vs),
                "stage_b_us": statistics.fmean(v["stage_b_us"] for v in vs),
                "residual_us": statistics.fmean(v["residual_us"] for v in vs),
                "scan": vs[0]["scan"],
                "config": cfg,
            }
        emit("| ctx | arm | bkv_p | num_bkv | scanned pos | scanned/inc | "
             "full us | d full % | stage A us | stage B us | residual us | "
             "d full % per scanned-pos ratio |")
        emit("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
             " ---: | ---: | ---: |")
        e3a = {}
        for ctx in sorted(by_ctx):
            m = by_ctx[ctx]
            base = m.get("inc", {}).get("full_us")
            inc_scan = m.get("inc", {}).get("scan", {}).get(
                "scanned_positions")
            for arm in ("inc", "inc_nee", "bkvp", "d1", "bkvp_d1", "merged"):
                if arm not in m:
                    continue
                v = m[arm]
                sc = v["scan"]
                ratio = (sc["scanned_positions"] / inc_scan
                         if inc_scan else None)
                dp = pct(v["full_us"], base)
                norm = (dp / ratio if dp is not None and ratio else None)
                emit(f"| {ctx} | `{arm}` | {sc['bkv_p']} | {sc['num_bkv']} | "
                     f"{sc['scanned_positions']} | "
                     f"{fnum(ratio, '{:.3f}')} | {v['full_us']:.3f} | "
                     f"{fnum(dp, '{:+.3f}%')} | {v['stage_a_us']:.3f} | "
                     f"{v['stage_b_us']:.3f} | {v['residual_us']:.3f} | "
                     f"{fnum(norm, '{:+.3f}')} |")
                e3a.setdefault(ctx, {})[arm] = {
                    **{k: v[k] for k in ("full_us", "stage_a_us",
                                         "stage_b_us", "residual_us")},
                    "scan": sc, "scanned_ratio_vs_inc": ratio,
                    "delta_full_pct_vs_inc": dp,
                    "delta_full_pct_per_scan_ratio": norm,
                }
        emit()
        emit("Scoring-path-only view: every arm against `d1` (the incumbent "
             "with the cond gone), which is on the scoring path at every "
             "context. `inc` and `bkvp` take the early-exit fast path at "
             "ctx <= 2048 and are not comparable there.")
        emit()
        emit("| ctx | arm | full us | d full % vs `d1` | stage A us | "
             "d stage A % vs `d1` |")
        emit("| ---: | --- | ---: | ---: | ---: | ---: |")
        for ctx in sorted(by_ctx):
            m = by_ctx[ctx]
            b = m.get("d1", {}).get("full_us")
            sb = m.get("d1", {}).get("stage_a_us")
            for arm in ("inc", "inc_nee", "bkvp", "d1", "bkvp_d1", "merged"):
                if arm not in m:
                    continue
                v = m[arm]
                emit(f"| {ctx} | `{arm}` | {v['full_us']:.3f} | "
                     f"{fnum(pct(v['full_us'], b), '{:+.3f}%')} | "
                     f"{v['stage_a_us']:.3f} | "
                     f"{fnum(pct(v['stage_a_us'], sb), '{:+.3f}%')} |")
                if ctx in e3a and arm in e3a[ctx]:
                    e3a[ctx][arm]["delta_full_pct_vs_d1"] = pct(
                        v["full_us"], b)
                    e3a[ctx][arm]["delta_stage_a_pct_vs_d1"] = pct(
                        v["stage_a_us"], sb)
        emit()
        payload["analysis"]["E3"] = e3a

    failed = [t for t, v in rows.items() if not v["checks_pass"]]
    emit(f"Arms failing the partition / row-completeness gate: "
         f"{len(failed)} of {len(rows)}"
         + (f" -- {failed}" if failed else ""))

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"wrote {a.out}")
    if a.md:
        with open(a.md, "w") as fh:
            fh.write("\n".join(out_md) + "\n")
        print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
