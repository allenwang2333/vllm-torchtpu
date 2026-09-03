#!/usr/bin/env python3
"""Read `hlo_stats` device self time out of the traces `probe_bdt_trace.py`
wrote, and bucket it the way `probes/R2.md` §1 does.

Runs **off-pod**, against a synced `cdk job sync-outputs` directory.  For each
arm it shells out to the xprof dumper, then:

  * sums `Avg. self time (us)` over the **TensorCore** rows -- that is the
    device time of one call;
  * buckets those rows into stage A (`StreamIdxTC-*`), stage B on TC
    (`sc_topk*`) and residual;
  * applies R2's **partition check**: `sum(buckets) - TC total` must be ~0,
    which is what catches a truncated dump.  R2's first dump silently lost
    rows to `--top-n 60`; `--top-n 500` plus an explicit
    `rows_kept == rows_total` assertion is the fix, and both are checked here;
  * reports the **SparseCore** plane separately and does **not** add it to the
    TensorCore total.  The two planes run concurrently; R2 excluded the
    SparseCore row for exactly this reason and so does this.

Usage:
    python bdt_hlo_stats.py <synced-outputs-dir> [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
DUMPER_PROJECT = os.environ.get(
    "BDT_DUMPER_PROJECT",
    os.path.join(os.path.dirname(REPO), "kernel5_copy", ".claude", "skills",
                 "xprof-dump", "xprof-dumper"))

STAGE_A = re.compile(r"StreamIdxTC")
STAGE_B = re.compile(r"sc_topk|sparsecore_topk|SparseCoreTopK")


def dump(pb: str, outdir: str, top_n: int = 500) -> dict:
    cmd = ["uv", "run", "--project", DUMPER_PROJECT, "xprof-dumper", "dump",
           pb, "--tools", "hlo_stats", "--top-n", str(top_n),
           "--trace-sample", "0", "--out-dir", outdir]
    subprocess.run(cmd, check=True, cwd=DUMPER_PROJECT,
                   stdout=subprocess.DEVNULL)
    for root, _d, files in os.walk(outdir):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f)) as fh:
                    d = json.load(fh)
                if "hlo_stats" in d.get("tools", {}):
                    return d
    raise SystemExit(f"no hlo_stats json under {outdir}")


def analyse(d: dict) -> dict:
    data = d["tools"]["hlo_stats"]["data"]
    rows = data["rows"]
    out = {
        "rows_total": data.get("rows_total"),
        "rows_kept": data.get("rows_kept"),
        "rows_complete": data.get("rows_total") == data.get("rows_kept"),
    }
    planes: dict[str, float] = {}
    buckets = {"stage_a": 0.0, "stage_b_on_tc": 0.0, "residual": 0.0}
    detail: dict[str, list] = {"stage_a": [], "stage_b_on_tc": [],
                               "residual": [], "sparsecore": []}
    for r in rows:
        core = r.get("TPU core type", "?")
        us = float(r.get("Avg. self time (us)", 0.0) or 0.0)
        name = str(r.get("HLO op name", ""))
        planes[core] = planes.get(core, 0.0) + us
        if core != "TensorCore":
            detail["sparsecore"].append([name, us])
            continue
        if STAGE_A.search(name):
            b = "stage_a"
        elif STAGE_B.search(name):
            b = "stage_b_on_tc"
        else:
            b = "residual"
        buckets[b] += us
        detail[b].append([name, us])
    tc = planes.get("TensorCore", 0.0)
    out.update(
        plane_totals_us=planes,
        tensorcore_total_us=tc,
        buckets_us=buckets,
        partition_residual_us=sum(buckets.values()) - tc,
        top_rows={k: sorted(v, key=lambda x: -x[1])[:8]
                  for k, v in detail.items()},
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("synced")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pbs = []
    for root, _d, files in os.walk(a.synced):
        for f in files:
            if f.endswith(".xplane.pb"):
                pbs.append(os.path.join(root, f))
    if not pbs:
        raise SystemExit(f"no *.xplane.pb under {a.synced}")

    res = {}
    for pb in sorted(pbs):
        # .../bdt/trace/<arm>/plugins/profile/<ts>/<host>.xplane.pb
        parts = pb.split(os.sep)
        arm = "unknown"
        if "plugins" in parts:
            arm = parts[parts.index("plugins") - 1]
        with tempfile.TemporaryDirectory() as td:
            try:
                res[arm] = analyse(dump(pb, td))
                res[arm]["xplane"] = pb
            except Exception as e:
                res[arm] = {"error": repr(e), "xplane": pb}
        r = res[arm]
        print(f"{arm:24s} TC={r.get('tensorcore_total_us', float('nan')):9.3f}us "
              f"A={r.get('buckets_us', {}).get('stage_a', float('nan')):9.3f} "
              f"B={r.get('buckets_us', {}).get('stage_b_on_tc', float('nan')):8.3f} "
              f"res={r.get('buckets_us', {}).get('residual', float('nan')):8.3f} "
              f"part={r.get('partition_residual_us', float('nan')):+.6f} "
              f"rows={r.get('rows_kept')}/{r.get('rows_total')} "
              f"SC={r.get('plane_totals_us', {}).get('SparseCore', 0.0):.3f}",
              flush=True)
    blob = json.dumps(res, indent=2, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(blob)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
