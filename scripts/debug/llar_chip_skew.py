#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-chip arrival skew at each low-latency all-reduce call site.

Answers why K3's post-MoE all-reduce looks ~2x its microbenchmark while the
post-attention one matches: it does not. It is waiting.

In an all-to-all collective every participant blocks until the last one arrives,
so within a layer the *fastest* participant is the straggler -- it never waits --
and ``max - min`` across participants is the spread in arrival times. That makes
the skew directly readable from a production trace without instrumenting
anything.

Chips, not ranks, are the unit. Under ``TPU_MOE_HIERARCHICAL_EP`` expert
parallelism runs *between* chips and tensor parallelism *within* one, so both
cores of a chip hold identical expert sets and should arrive together; the
imbalance is 16-way, not 32-way.

Two trace-reading traps, both of which produced confidently wrong answers first:

* **xprof caches combined op-stats keyed on the session directory, not the
  file.** Passing individual xplane paths from one session returns the same
  rank-combined aggregate every time, silently -- which showed up as a per-layer
  correlation of exactly +1.000 between "different" ranks. Give each xplane its
  own session path. Symlinks are enough and avoid copying gigabytes:

      W=results/chipiso
      for f in <trace>/plugins/profile/*/*.xplane.pb; do
        r=$(basename $f | sed 's/_.*//')
        mkdir -p $W/$r/plugins/profile/s1
        ln -sf "$(realpath $f)" $W/$r/plugins/profile/s1/
      done
      python scripts/debug/llar_chip_skew.py $W

* **The kernels are anonymous custom-calls** (``%shard_map.539``), one HLO op per
  layer, so the call site is recoverable only from the workspace shape in the op
  text -- ``bf16[32,1,336]`` for K3's concatenated MoE pair, ``bf16[32,1,224]``
  for the post-attention reduce. Op names carry nothing, which is why
  ``allreduce_profile_stats.py`` cannot attribute them.

Findings, job 154 (K3, compile size 1, 32 devices, 16 chips, EP=16 x TP=2):

                       per-chip median      per-layer skew (max-min)
    MoE pair           23.6 - 26.2 us       median 16.01 us
    post-attention     12.1 - 13.1 us       median  1.53 us

  All 16 chips are the fastest for at least one layer and every chip's median
  sits inside a 2.56 us band, so there is no persistent placement imbalance --
  no chip drew a hot expert set. The skew is per-layer routing.

  Average excess over the straggler is ~9 us (per-chip median 25.08 against
  per-layer min median 16.13), so ~828 us of a 15.31 ms step, about 5.4%, is
  spent waiting at this one collective. That is several times what any of the
  kernel-level changes can deliver, and it is a load-balance/overlap problem.

  It also means barrier removal cannot help at this site: the recv wait blocks on
  the same stragglers the barrier did.
"""
import collections
import glob
import json
import re
import statistics as st
import sys

from xprof.convert import raw_to_tool_data

K = "Total self time (us)"
OCC = "#Occurrences"
CAT = "HLO op category"
TXT = "HLO op text"
NAME = "HLO op name"

SITES = (("moe-pair", "bf16[32,1,336]"), ("post-attention", "bf16[32,1,224]"))


def chip_of(rank, cores_per_chip=2):
    """Chips are consecutive rank pairs.

    Verified against chip_topology's log line on this slice: chip0=[0,1],
    chip8=[2,3], chip13=[14,15] and so on -- the *device ids* pair up even/odd
    even though chip indices (ordered by mesh coords) do not follow rank order.
    """
    return rank // cores_per_chip


def parse(path):
    data = raw_to_tool_data.xspace_to_tool_data([path], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    rows = [
        dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]
    ]
    out = {}
    for label, pat in SITES:
        out[label] = {
            str(r[NAME]): float(r[K]) / int(r[OCC])
            for r in rows if str(r[CAT]) == "custom-call" and int(r[OCC] or 0)
            and pat in str(r[TXT] or "")
        }
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    paths = sorted(glob.glob(f"{sys.argv[1]}/**/*.xplane.pb", recursive=True))
    if not paths:
        raise SystemExit(f"no xplane under {sys.argv[1]}")

    per = collections.defaultdict(dict)
    for p in paths:
        m = re.search(r"rank(\d+)_", p)
        if not m:
            continue
        rank = int(m.group(1))
        try:
            for label, ops in parse(p).items():
                if ops:
                    per[label][rank] = ops
        except Exception as exc:  # noqa: BLE001
            print(f"  rank {rank}: {str(exc)[:70]}")
        print(f"parsed rank {rank} (chip {chip_of(rank)})", flush=True)

    for label, _ in SITES:
        d = per.get(label)
        if not d or len(d) < 2:
            print(f"\n== {label}: need >=2 ranks, got {len(d or {})}")
            continue
        ranks = sorted(d)
        common = sorted(set.intersection(*[set(d[r]) for r in ranks]))
        print(f"\n{'=' * 66}\n== {label}: {len(ranks)} chips x {len(common)} "
              f"layers")
        meds = sorted(
            (chip_of(r), st.median(list(d[r].values()))) for r in ranks)
        print("   per-chip median: " + "  ".join(f"c{c}={m:.1f}"
                                                 for c, m in meds))
        allmed = [m for _, m in meds]
        print(f"   across chips: min {min(allmed):.2f} "
              f"median {st.median(allmed):.2f} max {max(allmed):.2f} "
              f"spread {max(allmed) - min(allmed):.2f}")
        lo = [min(d[r][op] for r in ranks) for op in common]
        hi = [max(d[r][op] for r in ranks) for op in common]
        skew = [hi_v - lo_v for hi_v, lo_v in zip(hi, lo)]
        print(f"   per-layer min across chips (the straggler, so ~= true "
              f"kernel cost): median {st.median(lo):.2f} "
              f"(range {min(lo):.2f}-{max(lo):.2f})")
        print(f"   per-layer max across chips: median {st.median(hi):.2f}")
        print(f"   per-layer SKEW (max-min):   median {st.median(skew):.2f}")
        excess = st.median(allmed) - st.median(lo)
        print(f"   average excess over the straggler: {excess:.2f} us/layer "
              f"-> {excess * len(common):.0f} us/step across {len(common)} "
              f"layers")
        argmin = collections.Counter(
            chip_of(min(ranks, key=lambda r: d[r][op])) for op in common)
        print("   fastest chip per layer (i.e. the straggler), top 6: " +
              ", ".join(f"c{c}x{n}" for c, n in argmin.most_common(6)))
        print(f"   distinct chips ever fastest: {len(argmin)}/{len(ranks)}"
              f"  <- spread over all chips means no persistent placement "
              f"imbalance")


if __name__ == "__main__":
    main()
