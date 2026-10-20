"""Union size U of each request's queries' CSA2 top-512 sets, from topk_tap dumps.

usage: python union_stats.py <raw dir> [--json out.json] [--group-rows 6]
Rows of one request are starts[b]..starts[b+1]; for prefill (one request, 2048 rows)
--group-rows N splits the chunk into consecutive N-query groups instead.
"""

import argparse
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("raw")
p.add_argument("--json")
p.add_argument("--group-rows", type=int, default=0)
a = p.parse_args()

per_layer = {}
checks = []
for f in sorted(glob.glob(os.path.join(a.raw, "*.npz"))):
    z = np.load(f)
    layers, rec, pos, starts = z["layers"], z["records"], z["positions"], z["starts"]
    tokens = int(z["tokens"])
    ref = 0
    # Group-sharing check: same group's layers should carry identical records.
    for x, y in ((2, 3), (20, 21)):
        if x in layers and y in layers:
            i, j = list(layers).index(x), list(layers).index(y)
            checks.append(
                dict(
                    file=os.path.basename(f),
                    pair=[x, y],
                    identical=bool((rec[i] == rec[j]).all()),
                    positions_equal=bool((pos[i] == pos[j]).all()),
                )
            )
    for li, layer in enumerate(layers):
        if (pos[li] == -2).all():
            continue
        st = starts[li]
        if a.group_rows:
            bounds = [
                (s, min(s + a.group_rows, tokens))
                for s in range(0, tokens, a.group_rows)
            ]
        else:
            n = (
                int(np.argmax(st[1:] < st[:-1])) + 1
                if (st[1:] < st[:-1]).any()
                else len(st)
            )
            st = st[:n]
            bounds = [
                (int(st[b]), int(st[b + 1]))
                for b in range(len(st) - 1)
                if st[b + 1] > st[b]
            ]
        d = per_layer.setdefault(
            int(layer),
            dict(
                U=[], valid=[], pair_adj=[], pair_all=[], prog=[], rows=[], straddle=[]
            ),
        )
        for s, e in bounds:
            sets = [set(int(v) for v in rec[li, t] if v >= 0) for t in range(s, e)]
            if not sets:
                continue
            u = set().union(*sets)
            d["U"].append(len(u))
            d["rows"].append(e - s)
            d["valid"].append(float(np.mean([len(x) for x in sets])))
            adj = [
                len(sets[k] & sets[k + 1]) / max(1, len(sets[k]))
                for k in range(len(sets) - 1)
            ]
            allp = [
                len(sets[k] & sets[m]) / max(1, len(sets[k]))
                for k in range(len(sets))
                for m in range(k + 1, len(sets))
            ]
            d["pair_adj"] += adj
            d["pair_all"] += allp
            prog, acc = [], set()
            for x in sets:
                acc |= x
                prog.append(len(acc))
            d["prog"].append(prog)
            p0 = int(pos[li, s])
            anchor = max(0, p0 - 127) // 128 * 128
            d["straddle"].append(int(int(pos[li, e - 1]) >= anchor + 256))

out = dict(checks=checks, layers={})
print(
    "group-sharing checks:",
    "all identical" if all(c["identical"] for c in checks) else checks,
)
COLUMNS = (
    "layer",
    "n",
    "rows",
    "valid",
    "U mean",
    "U p50",
    "U p90",
    "U max",
    "U/512",
    "adj ov",
    "all ov",
    "strad",
)
print(" ".join(f"{c:>7}" for c in COLUMNS), " union growth per added query")
for layer, d in sorted(per_layer.items()):
    U = np.array(d["U"])
    prog = np.array([x for x in d["prog"] if len(x) == max(len(y) for y in d["prog"])])
    row = dict(
        n=len(U),
        rows=int(np.median(d["rows"])),
        valid_mean=float(np.mean(d["valid"])),
        U_mean=float(U.mean()),
        U_p10=float(np.percentile(U, 10)),
        U_p50=float(np.median(U)),
        U_p90=float(np.percentile(U, 90)),
        U_max=int(U.max()),
        U_over_512_mean=float(U.mean() / 512),
        pair_adjacent_overlap=float(np.mean(d["pair_adj"])),
        pair_all_overlap=float(np.mean(d["pair_all"])),
        straddle_frac=float(np.mean(d["straddle"])),
        growth=[float(v) for v in prog.mean(0)] if len(prog) else [],
        hist=np.histogram(
            U, bins=[0, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 3072, 1 << 20]
        )[0].tolist(),
    )
    out["layers"][layer] = row
    cells = (
        layer,
        row["n"],
        row["rows"],
        row["valid_mean"],
        row["U_mean"],
        row["U_p50"],
        row["U_p90"],
        row["U_max"],
        row["U_over_512_mean"],
        row["pair_adjacent_overlap"],
        row["pair_all_overlap"],
        row["straddle_frac"],
    )
    print(
        " ".join(f"{c:>7.2f}" if isinstance(c, float) else f"{c:>7}" for c in cells),
        "",
        " ".join(f"{v:.0f}" for v in row["growth"][:8]),
    )
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
