#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-TensorCore device time from the xprof traces the phase 3 benches save.

    p3_xplane_device_time.py OUT.csv TRACE_ROOT [--window T0 T1] [--match SUBSTR]
        [--calls-per-step N]

TRACE_ROOT holds one directory per trace (the benches' <out_dir>/trace/<name>/ or
<out_dir>/xprof/), each with plugins/profile/*/*.xplane.pb. Runs in the VM's
xprof venv (the `xprof` package); generalises
~/result/data/qwen35-agentx-p2/e1/bench_device_time.py.

Per trace and TensorCore plane ("/device:TPU:n", SparseCore planes skipped) it
reads the "XLA Ops" line and reports:

  busy_ms          union of op intervals (device busy time) inside the window
  ops_ms           sum of op durations (overlapping async ops counted twice)
  span_ms          first op start to last op end
  n_ops
  match_ms, n_match, match_ms_per_step
                   for ops whose name contains --match (e.g. the megakernel
                   custom call, a collective, "fused_ep"), grouped
                   --calls-per-step at a time

Window times are seconds since the epoch (the benches record t0 / t1 of every
round and window in result.json); without a window the whole trace counts.
Spans here are program-level wall spans on the device, so under lockstep
workloads they include collective arrival skew; use them as device time, not
as compute time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys


def _planes(path):
    from xprof import profile_data as pd

    prof = pd.ProfileData.from_file(path)
    for plane in prof.planes:
        if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
            continue
        for ln in plane.lines:
            if ln.name == "XLA Ops":
                yield plane.name, ln


def _union_ms(intervals):
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e6


def analyze(trace_dir: str, window, match: str, calls_per_step: int) -> list[dict]:
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    rows = []
    lo = window[0] * 1e9 if window else None
    hi = window[1] * 1e9 if window else None
    for f in files:
        for plane, ln in _planes(f):
            ev = [(e.start_ns, e.start_ns + e.duration_ns, e.name) for e in ln.events]
            if lo is not None:
                ev = [x for x in ev if x[0] >= lo and x[1] <= hi]
            if not ev:
                continue
            row = {
                "trace": os.path.basename(trace_dir.rstrip("/")),
                "file": os.path.basename(f),
                "core": plane,
                "n_ops": len(ev),
                "busy_ms": round(_union_ms([(s, e) for s, e, _ in ev]), 4),
                "ops_ms": round(sum(e - s for s, e, _ in ev) / 1e6, 4),
                "span_ms": round((max(e for _, e, _ in ev) - min(s for s, _, _ in ev)) / 1e6, 4),
            }
            if match:
                m = sorted((s, e) for s, e, n in ev if match in n.split(" = ")[0])
                row["n_match"] = len(m)
                row["match_ms"] = round(sum(e - s for s, e in m) / 1e6, 4)
                n_steps = len(m) // max(calls_per_step, 1)
                row["match_ms_per_step"] = (
                    round(sum(e - s for s, e in m[: n_steps * calls_per_step]) / n_steps / 1e6, 4)
                    if n_steps else ""
                )
                gaps = [(m[i + 1][0] - m[i][1]) / 1e6 for i in range(len(m) - 1)]
                row["match_gap_ms_median"] = round(statistics.median(gaps), 4) if gaps else ""
            rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("trace_root")
    ap.add_argument("--window", nargs=2, type=float, default=None, metavar=("T0", "T1"))
    ap.add_argument("--match", default="")
    ap.add_argument("--calls-per-step", type=int, default=1)
    a = ap.parse_args(argv)
    dirs = [a.trace_root] if glob.glob(os.path.join(a.trace_root, "**", "*.xplane.pb"), recursive=True) \
        and not any(os.path.isdir(os.path.join(a.trace_root, d, "plugins")) for d in os.listdir(a.trace_root)) \
        else sorted(d for d in glob.glob(os.path.join(a.trace_root, "*")) if os.path.isdir(d))
    rows = []
    for d in dirs:
        rows += analyze(d, a.window, a.match, a.calls_per_step)
        if rows:
            print(rows[-1], flush=True)
    if not rows:
        print("no TPU planes found", file=sys.stderr)
        return 1
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("trace", "file", "core"), k))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # Per-trace summary across cores: mean, min, max of busy_ms.
    by_trace: dict[str, list] = {}
    for r in rows:
        by_trace.setdefault(r["trace"], []).append(r["busy_ms"])
    for t, v in by_trace.items():
        print(f"{t}: cores={len(v)} busy_ms mean={statistics.mean(v):.3f} "
              f"min={min(v):.3f} max={max(v):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
