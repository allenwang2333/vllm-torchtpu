"""Summarize one arm's rank-0 device time for a single profiled phase.

  python scripts/k3/summarize_phase.py <results_dir> [prefill_only|decode_only]

diff_arms.py answers "what moved between two arms". This answers the question
that comes first for a new workload: "where does the time go at all". It reads
the same rank-0 xplane that diff_arms.py does and reports total device self
time, the top ops, per-category totals, and every collective with its
PER-OCCURRENCE cost -- the last because a collective that is cheap once and
runs 500 times is a different problem from one that is expensive once.

Caveat carried over from the runbook: per-op self times are hardware counters
and stay valid under the profiler, but the profiler inflates step time ~50x, so
never read wall-clock or inter-step gaps out of a trace.
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
NAME = "HLO op name"
FW = "Framework op name"
TXT = "HLO op text"

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
               "collective-permute", "collective-permute-start",
               "collective-permute-done")


def load(results_dir, phase):
    pattern = f"{results_dir}/traces/{phase}/plugins/profile/*/rank0_*.xplane.pb"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no rank0 {phase} trace under {results_dir}")
    if len(paths) > 1:
        print(f"note: {len(paths)} rank0 captures for {phase}; using {paths[0]}")
    data = raw_to_tool_data.xspace_to_tool_data([paths[0]], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    return [dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]]


def main():
    results_dir = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else "prefill_only"
    rows = load(results_dir, phase)

    total = sum(float(r[T]) for r in rows)
    print(f"{results_dir}  phase={phase}")
    print(f"rank0 device self time: {total/1e3:.2f} ms over "
          f"{sum(int(r[OCC] or 0) for r in rows)} op occurrences\n")

    print("=== top 25 ops by total self time ===")
    print(f"  {'ms':>9} {'%':>6} {'occ':>7} {'us/occ':>9}  {'category':<20} name")
    for r in sorted(rows, key=lambda r: -float(r[T]))[:25]:
        t, occ = float(r[T]), int(r[OCC] or 0)
        per = t / occ if occ else 0.0
        name = str(r.get(NAME) or "?")
        fw = str(r.get(FW) or "")
        label = f"{name}" + (f"  [{fw}]" if fw and fw != name else "")
        print(f"  {t/1e3:9.3f} {100*t/total:6.1f} {occ:7d} {per:9.2f}  "
              f"{str(r[CAT]):<20} {label[:70]}")

    by_cat = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        c = by_cat[str(r[CAT])]
        c[0] += float(r[T])
        c[1] += int(r[OCC] or 0)
    print("\n=== by category ===")
    print(f"  {'ms':>9} {'%':>6} {'occ':>7}  category")
    for cat, (t, occ) in sorted(by_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t/1e3:9.3f} {100*t/total:6.1f} {occ:7d}  {cat}")

    coll = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        cat = str(r[CAT])
        if cat not in COLLECTIVES:
            continue
        shapes = sorted(set(re.findall(r"(?:bf16|f32|s32|u32|f8e4m3fn)\[[\d,]+\]",
                                       str(r[TXT] or ""))))
        coll[(cat, ",".join(shapes[:2]) or "?")][0] += float(r[T])
        coll[(cat, ",".join(shapes[:2]) or "?")][1] += int(r[OCC] or 0)
    if coll:
        c_total = sum(v[0] for v in coll.values())
        c_occ = sum(v[1] for v in coll.values())
        print(f"\n=== collectives: {c_total/1e3:.2f} ms "
              f"({100*c_total/total:.1f}% of device) over {c_occ} occurrences ===")
        print(f"  {'ms':>9} {'occ':>7} {'us/occ':>9}  {'category':<24} payload")
        for (cat, shapes), (t, occ) in sorted(coll.items(), key=lambda kv: -kv[1][0]):
            per = t / occ if occ else 0.0
            print(f"  {t/1e3:9.3f} {occ:7d} {per:9.2f}  {cat:<24} {shapes[:44]}")


if __name__ == "__main__":
    main()
