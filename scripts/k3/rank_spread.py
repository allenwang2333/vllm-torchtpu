#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Across-rank spread of a collective's device self time.

  python scripts/k3/rank_spread.py <results_dir> [<results_dir> ...]

Separates two explanations for a collective costing more in production than in
an isolated harness.

**Arrival skew.** Ranks reach the collective at different times -- in K3,
because hierarchical EP gives each chip a different expert load. A barrier
makes every rank leave together, so the rank that arrived first waits longest
and the rank that arrived last barely waits at all. The cost is real but it is
*imbalance*, and its signature is a wide spread of self time **across ranks**
for the same op.

**Uniform cost.** Contention for ICI bandwidth, or simply a slower collective,
raises every rank's self time together. Signature: high mean, narrow spread.

So: wide across-rank spread => skew; narrow => not skew. This distinguishes
them without another cluster run, since the phased profiler already captured
all 32 ranks.
"""
import collections
import glob
import json
import re
import sys

from xprof.convert import raw_to_tool_data

T = "Total self time (us)"
OCC = "#Occurrences"
CAT = "HLO op category"
TXT = "HLO op text"
NAME = "HLO op name"


def rank_of(path):
    m = re.search(r"rank(\d+)_", path)
    return int(m.group(1)) if m else None


def load(path):
    data = raw_to_tool_data.xspace_to_tool_data([path], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    return [
        dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]
    ]


def bucket(rows):
    """{label: (total_us, occ)} for the collectives and the Pallas kernel."""
    out = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        cat = str(r[CAT])
        name = str(r[NAME] or "")
        t, occ = float(r[T]), int(r[OCC] or 0)
        if "shard_map" in name:
            label = "kernel (pallas one_shot)"
        elif cat == "all-reduce":
            shapes = sorted(set(re.findall(r"bf16\[[\d,]+\]", str(r[TXT]))))
            label = "all-reduce " + ",".join(shapes[:2])
        else:
            continue
        out[label][0] += t
        out[label][1] += occ
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for results_dir in sys.argv[1:]:
        paths = sorted(
            glob.glob(f"{results_dir}/traces/decode_only/plugins/profile/*/"
                      f"rank*.xplane.pb"))
        if not paths:
            print(f"{results_dir}: no traces")
            continue
        per_rank = {}
        for p in paths:
            rk = rank_of(p)
            if rk is None:
                continue
            try:
                per_rank[rk] = bucket(load(p))
            except Exception as e:  # noqa: BLE001
                print(f"  !! rank {rk}: {type(e).__name__}: {e}")

        print(f"\n=== {results_dir}  ({len(per_rank)} ranks) ===")
        print(f"{'op':<34} {'ranks':>6} {'mean':>8} {'min':>8} {'max':>8} "
              f"{'spread':>8} {'cv%':>6}")
        labels = sorted({k for b in per_rank.values() for k in b})
        for label in labels:
            vals = []
            for rk, b in per_rank.items():
                t, occ = b.get(label, [0.0, 0])
                if occ:
                    vals.append(t / occ)
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            lo, hi = min(vals), max(vals)
            var = sum((v - mean)**2 for v in vals) / len(vals)
            cv = 100.0 * var**0.5 / mean if mean else 0.0
            print(f"{label[:34]:<34} {len(vals):>6} {mean:>8.2f} {lo:>8.2f} "
                  f"{hi:>8.2f} {hi - lo:>8.2f} {cv:>6.1f}")


if __name__ == "__main__":
    main()
