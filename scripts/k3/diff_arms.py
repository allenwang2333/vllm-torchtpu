"""Diff two arms' rank-0 decode HLO stats.

  python diff_arms.py <baseline_results_dir> <candidate_results_dir>

Written for the intra-chip latent-projection A/B, so it reports three things:
  1. the projection GEMMs, matched by weight shape (full vs halved shard);
  2. every collective, with per-occurrence cost -- an added collective is the
     first thing to suspect whenever a bandwidth win fails to show up;
  3. category totals, to catch anything else that moved.

Compare only arms whose servers were started with the same flags. Device self
time is meaningless across a change in async scheduling or compile buckets.
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

# Latent projection weights: full replicated, and the halved column shards.
PROJ_SHAPES = {
    "up_proj   full  [7168,3584]": "bf16[7168,3584]",
    "up_proj   shard [3584,3584]": "bf16[3584,3584]",
    "down_proj full  [3584,7168]": "bf16[3584,7168]",
    "down_proj shard [1792,7168]": "bf16[1792,7168]",
}


def load(results_dir):
    pattern = f"{results_dir}/traces/decode_only/plugins/profile/*/rank0_*.xplane.pb"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no rank0 decode_only trace under {results_dir}")
    data = raw_to_tool_data.xspace_to_tool_data([paths[0]], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    return [
        dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]
    ]


def summarize(rows):
    total = sum(float(r[T]) for r in rows)
    by_cat = collections.defaultdict(float)
    proj = collections.defaultdict(lambda: [0.0, 0])
    # Collectives bucketed by (category, payload shape) so a 2-rank gather of
    # [2,1792] is distinguishable from a 32-rank reduce of [2,7168].
    coll = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        t, occ = float(r[T]), int(r[OCC] or 0)
        cat = str(r[CAT])
        by_cat[cat] += t
        text = str(r[TXT] or "")
        for label, shape in PROJ_SHAPES.items():
            if shape in text:
                proj[label][0] += t
                proj[label][1] += occ
        if cat in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
                   "collective-permute"):
            shapes = sorted(set(re.findall(r"bf16\[[\d,]+\]", text)))
            key = (cat, ",".join(shapes[:2]) or "?")
            coll[key][0] += t
            coll[key][1] += occ
    return total, by_cat, proj, coll


def main():
    base_dir, cand_dir = sys.argv[1], sys.argv[2]
    b_total, b_cat, b_proj, b_coll = summarize(load(base_dir))
    c_total, c_cat, c_proj, c_coll = summarize(load(cand_dir))

    print(f"baseline  {base_dir}")
    print(f"candidate {cand_dir}\n")
    d = c_total - b_total
    print(f"Total rank0 decode device self time: base={b_total/1e3:.2f} ms  "
          f"cand={c_total/1e3:.2f} ms  delta={d/1e3:+.2f} ms "
          f"({100*d/b_total:+.1f}%)\n")

    print("=== latent projections (by weight shape) ===")
    print(f"  {'':28} {'base ms':>9} {'occ':>6} {'cand ms':>9} {'occ':>6}")
    for label in PROJ_SHAPES:
        bt, bo = b_proj.get(label, [0.0, 0])
        ct, co = c_proj.get(label, [0.0, 0])
        if bo or co:
            print(f"  {label:28} {bt/1e3:9.2f} {bo:6d} {ct/1e3:9.2f} {co:6d}")
    bsum = sum(v[0] for v in b_proj.values())
    csum = sum(v[0] for v in c_proj.values())
    print(f"  {'TOTAL projections':28} {bsum/1e3:9.2f} {'':6} "
          f"{csum/1e3:9.2f}        delta={(csum-bsum)/1e3:+.2f} ms")

    print("\n=== collectives (per-occurrence cost is the point) ===")
    print(f"  {'arm':<5} {'category':<16} {'payload':<28} {'ms':>8} "
          f"{'occ':>6} {'us/occ':>8}")
    for arm, coll in (("base", b_coll), ("cand", c_coll)):
        for (cat, shapes), (t, occ) in sorted(coll.items(),
                                              key=lambda kv: -kv[1][0])[:6]:
            per = t / occ if occ else 0.0
            print(f"  {arm:<5} {cat:<16} {shapes[:28]:<28} {t/1e3:8.2f} "
                  f"{occ:6d} {per:8.1f}")
    b_coll_total = sum(v[0] for v in b_coll.values())
    c_coll_total = sum(v[0] for v in c_coll.values())
    b_coll_occ = sum(v[1] for v in b_coll.values())
    c_coll_occ = sum(v[1] for v in c_coll.values())
    print(
        f"  ALL collectives: base={b_coll_total/1e3:.2f} ms ({b_coll_occ} occ)"
        f"  cand={c_coll_total/1e3:.2f} ms ({c_coll_occ} occ)"
        f"  delta={(c_coll_total-b_coll_total)/1e3:+.2f} ms")

    print("\n=== by category (largest absolute change) ===")
    keys = set(b_cat) | set(c_cat)
    print(
        f"  {'category':<22} {'base ms':>9} {'cand ms':>9} {'delta':>9} {'%':>8}"
    )
    for k in sorted(
            keys,
            key=lambda k: -abs(c_cat.get(k, 0.) - b_cat.get(k, 0.)))[:12]:
        bv, cv = b_cat.get(k, 0.0), c_cat.get(k, 0.0)
        pct = (100 * (cv - bv) / bv) if bv else float("inf")
        print(f"  {k[:22]:<22} {bv/1e3:9.2f} {cv/1e3:9.2f} "
              f"{(cv-bv)/1e3:+9.2f} {pct:+7.1f}%")


if __name__ == "__main__":
    main()
