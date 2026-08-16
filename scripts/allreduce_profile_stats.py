#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-occurrence device self time for the all-reduce microbenchmark traces.

  python scripts/allreduce_profile_stats.py <profile-dir>

``allreduce_latency_bench.py --profile-dir`` writes one trace per
(impl, shape, devices). This reports, for each, the device self time of the
collective op divided by its occurrence count -- **the same metric, from the
same xprof tool, as scripts/k3/diff_arms.py reports for a production server**.

Why this exists: the bench's headline number is a two-point slope, which
cancels every fixed cost by construction. That is the right way to isolate a
collective's marginal cost, and the wrong way to predict what it will cost on a
model's critical path. When a kernel that measured 15.14 us on the bench shows
26 us in a production trace, the first question is whether the two numbers even
denote the same quantity. This makes them denote the same quantity.
"""
import collections
import glob
import json
import os
import re
import sys

from xprof.convert import raw_to_tool_data

T = "Total self time (us)"
OCC = "#Occurrences"
CAT = "HLO op category"
TXT = "HLO op text"
NAME = "HLO op name"

# Everything the two implementations can lower to. The Pallas kernels appear as
# a custom-call rather than a collective, which is exactly why a category-level
# read of a production trace can look like all-reduce time "disappearing".
COLLECTIVE_CATS = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
                   "collective-permute", "custom-call")


def load(trace_dir):
    """Rank-0 xplane for one traced chain."""
    paths = sorted(glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True))
    if not paths:
        return None
    data = raw_to_tool_data.xspace_to_tool_data([paths[0]], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    return [
        dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]
    ]


def summarize(rows):
    """(total us, {(cat, shape): [us, occ]}) for the collective-ish ops."""
    total = sum(float(r[T]) for r in rows)
    ops = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        cat = str(r[CAT])
        if cat not in COLLECTIVE_CATS:
            continue
        text = str(r[TXT] or "")
        shapes = sorted(set(re.findall(r"(?:bf16|f32)\[[\d,]+\]", text)))
        key = (cat, ",".join(shapes[:2]) or str(r[NAME] or "?")[:24])
        ops[key][0] += float(r[T])
        ops[key][1] += int(r[OCC] or 0)
    return total, ops


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = sys.argv[1]
    dirs = sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no per-run trace directories under {root}")

    print(f"{'run':<26} {'category':<16} {'payload':<26} {'ms':>8} "
          f"{'occ':>7} {'us/occ':>8}")
    for d in dirs:
        rows = load(d)
        if rows is None:
            print(f"{os.path.basename(d):<26} (no xplane)")
            continue
        _, ops = summarize(rows)
        if not ops:
            print(f"{os.path.basename(d):<26} (no collective ops found)")
            continue
        for (cat, shapes), (t, occ) in sorted(ops.items(),
                                              key=lambda kv: -kv[1][0]):
            per = t / occ if occ else 0.0
            print(f"{os.path.basename(d):<26} {cat:<16} {shapes[:26]:<26} "
                  f"{t / 1e3:8.2f} {occ:7d} {per:8.2f}")


if __name__ == "__main__":
    main()
