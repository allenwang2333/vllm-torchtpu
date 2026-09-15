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
"""What share of device time one kernel owns, read out of an xplane trace.

Sums every XLA op's duration on every TPU device plane and prints the biggest
by share, plus the total for whichever ops match `--match`. A kernel's share of
step time is the ceiling on what optimizing it can be worth end to end, so this
is the number that decides whether a campaign against it is worth running.

    python3 bench/xplane_op_share.py --dir <profile-dir> --match fused_conv1d

Runs anywhere jax imports -- CPU is fine -- so the trace can be captured on the
TPU job and read afterwards.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import jax


def op_durations(path: pathlib.Path) -> dict[str, tuple[int, int]]:
    """name -> (total duration ns, call count), over the TPU device planes."""
    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    profile = jax.profiler.ProfileData.from_file(str(path))
    for plane in profile.planes:
        if (not plane.name.startswith("/device:TPU:")
                or "SparseCore" in plane.name):
            continue
        for line in plane.lines:
            if line.name != "XLA Ops":
                continue
            for event in line.events:
                entry = totals[event.name]
                entry[0] += int(event.duration_ns)
                entry[1] += 1
    return {k: (v[0], v[1]) for k, v in totals.items()}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--match", default="kda")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    traces = sorted(pathlib.Path(args.dir).rglob("*.xplane.pb"))
    if not traces:
        print(f"no *.xplane.pb under {args.dir}", file=sys.stderr)
        return 1

    pattern = re.compile(args.match, re.IGNORECASE)
    per_trace = []
    for trace in traces:
        ops = op_durations(trace)
        total = sum(d for d, _ in ops.values())
        if not total:
            print(f"{trace}: no XLA op events", file=sys.stderr)
            continue
        matched = {k: v for k, v in ops.items() if pattern.search(k)}
        rows = sorted(ops.items(), key=lambda kv: -kv[1][0])[:args.top]
        per_trace.append({
            "trace": str(trace),
            "total_device_ns": total,
            "matched_ns": sum(d for d, _ in matched.values()),
            "matched_share": sum(d for d, _ in matched.values()) / total,
            "matched_ops": {k: {"ns": v[0], "calls": v[1]}
                            for k, v in sorted(matched.items(),
                                               key=lambda kv: -kv[1][0])},
            "top_ops": [{"op": k, "ns": v[0], "calls": v[1],
                         "share": v[0] / total} for k, v in rows],
        })

    if not per_trace:
        return 1
    summary = {
        "match": args.match,
        "traces": per_trace,
        # Per rank, because the step costs the max over ranks rather than the
        # mean: a share averaged across ranks hides the straggler.
        "share_min": min(t["matched_share"] for t in per_trace),
        "share_max": max(t["matched_share"] for t in per_trace),
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
