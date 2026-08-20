#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-rank self time of each low-latency all-reduce, from a production trace.

Job 154's trace says the post-attention kernel costs 12.77 us -- matching its
12.83 us microbenchmark almost exactly -- while the latent-MoE pair kernel costs
25.16 us against a 12.98 us microbenchmark. Same kernel family, same trace, so
the 1.94x inflation is a property of the *call site*.

Arrival skew is the standing suspicion, and it has a signature this can test.
If ranks reach the collective at different times, the early arrivers block in it
and the last arriver does not, so the kernel's self time varies *widely across
ranks* while (preceding compute + collective) stays roughly constant. If instead
the kernel is uniformly ~25 us on every rank, the time is real work or wire
time, not waiting.

Reads every rank's xplane rather than rank 0's -- which is the whole point, and
what ``allreduce_profile_stats.py`` cannot do, since it loads ``paths[0]``.

    python scripts/debug/llar_rank_spread.py <trace-dir>
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

from xprof.convert import raw_to_tool_data

T = "Total self time (us)"
OCC = "#Occurrences"
CAT = "HLO op category"
TXT = "HLO op text"
NAME = "HLO op name"


def rank_of(path):
    m = re.search(r"rank(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def rows_for(path):
    data = raw_to_tool_data.xspace_to_tool_data([path], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    return [
        dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]
    ]


# The Pallas kernels lower to *anonymous* custom-calls (custom-call.611 and so
# on), so the call site is not recoverable from the op name -- only from the
# payload shape in the op text. K3 at compile size 1: 7168 wide is the
# post-attention reduce, 10752 the concatenated latent-MoE pair, and the
# reshaped (32,224)/(32,336) forms are the same tensors after _build_two_shot's
# (N, numel/N) view.
SITE_SHAPES = {
    "post-attention": ("[1,7168]", "[32,224]"),
    "moe-pair": ("[1,10752]", "[32,336]"),
    "moe-latent": ("[1,3584]", "[32,112]"),
}


def llar_ops(rows):
    """{label: (us_per_occ, occurrences)} keyed by call site."""
    out = {}
    for r in rows:
        cat = str(r[CAT] or "")
        if cat != "custom-call":
            continue
        occ = int(r[OCC] or 0)
        if not occ:
            continue
        text = str(r[TXT] or "")
        for label, needles in SITE_SHAPES.items():
            if any(nd in text for nd in needles):
                us, n = out.get(label, (0.0, 0))
                out[label] = (us + float(r[T] or 0), n + occ)
                break
    return {k: (v[0] / v[1], v[1]) for k, v in out.items() if v[1]}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    paths = sorted(glob.glob(f"{sys.argv[1]}/**/*.xplane.pb", recursive=True),
                   key=rank_of)
    if not paths:
        raise SystemExit(f"no xplane under {sys.argv[1]}")
    print(f"reading {len(paths)} rank trace(s)")

    per_rank = collections.defaultdict(dict)
    for path in paths:
        r = rank_of(path)
        try:
            for label, (us, occ) in llar_ops(rows_for(path)).items():
                per_rank[label][r] = (us, occ)
        except Exception as exc:  # noqa: BLE001
            print(f"  rank {r}: failed ({str(exc)[:70]})")

    for label in sorted(per_rank):
        vals = per_rank[label]
        us = [v[0] for v in vals.values()]
        occ = [v[1] for v in vals.values()]
        print(f"\n== {label}: {len(us)} ranks, "
              f"{occ[0] if occ else 0} occurrences/rank")
        print(f"   min {min(us):7.2f}   median {statistics.median(us):7.2f}   "
              f"max {max(us):7.2f}   spread {max(us) - min(us):6.2f} "
              f"({max(us)/min(us):.2f}x)")
        if len(us) > 1:
            print(f"   stdev {statistics.stdev(us):6.2f}  "
                  f"({statistics.stdev(us)/statistics.mean(us)*100:.1f}% of "
                  f"mean)")
        ranked = sorted(vals.items(), key=lambda kv: -kv[1][0])
        print("   slowest ranks: " + ", ".join(f"r{r}={v[0]:.1f}"
                                               for r, v in ranked[:6]))
        print("   fastest ranks: " + ", ".join(f"r{r}={v[0]:.1f}"
                                               for r, v in ranked[-6:]))

    print("\nRead: a wide spread across ranks means the kernel is absorbing "
          "arrival skew\n(early arrivers block, the last arriver does not). A "
          "tight spread means the time\nis real work or wire time.")


if __name__ == "__main__":
    main()
