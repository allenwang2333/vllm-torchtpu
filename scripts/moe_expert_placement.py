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
"""Build a TPU_EXPERT_PLACEMENT table from MOE_EXPERT_COUNT_TAP_DIR snapshots.

    moe_expert_placement.py --shards 16 --layers 3:60 [--spare 1] \\
        --out placement.json TAP_DIR [TAP_DIR ...]

Each rank's load is its last snapshot minus its `--skip`-th (1-based; 0 keeps
everything), which drops warmup and compile traffic; ranks are then summed.

The fused EP kernel streams every visited expert's weights once and computes
its rows a tile at a time, so a slot's cost is modelled as
`base + tile_us * (ceil(rows / tile_rows) - 1)` on its expected rows per step,
and a layer is as slow as its most expensive rank. Per layer, `--spare` extra
slots per rank go to replicas of the costliest experts (a replica takes an
even share of its expert's rows), slots are assigned costliest-first to the
cheapest rank that does not already hold that expert, and pairwise swaps off
the costliest rank follow. Prints the modelled straggler of the checkpoint's
contiguous placement and of the built one.
"""

import argparse
import collections
import glob
import json
import math
import os
import re

import numpy as np

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_counts(dirs, skip):
    """{layer name: (mean [E] per step, variance [E] per step)} summed over ranks."""
    by_rank = collections.defaultdict(list)
    for d in dirs:
        pattern = os.path.join(d, "**", "counts_ep*_*.npz")
        for path in glob.glob(pattern, recursive=True):
            m = re.search(r"counts_ep(\d+)_(\d+)\.npz$", path)
            by_rank[(d, int(m.group(1)))].append((int(m.group(2)), path))
    if not by_rank:
        raise SystemExit(f"no counts_ep*_*.npz under {dirs}")
    mean, var, names = None, None, None
    for key, snaps in sorted(by_rank.items()):
        snaps.sort()
        if len(snaps) <= skip:
            raise SystemExit(f"{key}: {len(snaps)} snapshots, need more than {skip}")
        last = np.load(snaps[-1][1])
        c = last["counts"].astype(np.int64)
        if skip:
            c = c - np.load(snaps[skip - 1][1])["counts"].astype(np.int64)
        steps = np.maximum(c[:, 2, :1], 1)
        mu = c[:, 0] / steps
        v = np.maximum(c[:, 1] / steps - mu * mu, 0)
        layer_names = [str(n) for n in last["layer_names"]]
        if names is None:
            names, mean, var = layer_names, mu, v
        else:
            assert names == layer_names, key
            mean, var = mean + mu, var + v
    return {n: (mean[i], var[i]) for i, n in enumerate(names)}


class CostModel:
    def __init__(self, base, tile_us, tile_rows):
        self.base, self.tile_us, self.tile_rows = base, tile_us, tile_rows

    def slot(self, rows):
        return self.base + self.tile_us * max(0, math.ceil(rows / self.tile_rows) - 1)


def place(mu, shards, spare, model):
    """Per rank, the experts it holds (len experts / shards + spare each)."""
    experts = mu.size
    width = experts // shards + spare
    reps = np.ones(experts, int)
    for _ in range(shards * spare):
        e = max(
            range(experts), key=lambda e: (model.slot(mu[e] / reps[e]), mu[e] / reps[e])
        )
        if reps[e] == shards:
            raise SystemExit("an expert would need more replicas than ranks")
        reps[e] += 1
    cost = {e: model.slot(mu[e] / reps[e]) for e in range(experts)}
    order = sorted(
        (e for e in range(experts) for _ in range(reps[e])), key=lambda e: -cost[e]
    )
    held = [[] for _ in range(shards)]
    total = np.zeros(shards)
    for e in order:
        free = [r for r in range(shards) if len(held[r]) < width and e not in held[r]]
        r = min(free, key=lambda r: total[r])
        held[r].append(e)
        total[r] += cost[e]
    for _ in range(experts * 4):
        hi = int(np.argmax(total))
        best = None
        for lo in range(shards):
            if lo == hi:
                continue
            for i, a in enumerate(held[hi]):
                for j, b in enumerate(held[lo]):
                    d = cost[a] - cost[b]
                    if d <= 0 or a in held[lo] or b in held[hi]:
                        continue
                    new_max = max(total[hi] - d, total[lo] + d)
                    if new_max < total[hi] and (best is None or new_max < best[0]):
                        best = (new_max, lo, i, j, d)
        if best is None:
            break
        _, lo, i, j, d = best
        held[hi][i], held[lo][j] = held[lo][j], held[hi][i]
        total[hi] -= d
        total[lo] += d
    return held, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=1, help="snapshots to drop")
    ap.add_argument(
        "--layers",
        required=True,
        help="first:last decoder layers to place (a drafter's MoE layers are "
        "numbered after the target's)",
    )
    ap.add_argument("--spare", type=int, default=0, help="replica slots per rank")
    ap.add_argument("--base-us", type=float, default=10.7)
    ap.add_argument("--tile-us", type=float, default=12.0)
    ap.add_argument("--tile-rows", type=int, default=128)
    a = ap.parse_args()
    model = CostModel(a.base_us, a.tile_us, a.tile_rows)
    counts = load_counts(a.dirs, a.skip)
    lo, hi = (int(x) for x in a.layers.split(":"))
    table, experts, before, after = {}, None, [], []
    for name, (mu, _) in counts.items():
        m = _LAYER_RE.search(name)
        if not m or not lo <= int(m.group(1)) <= hi:
            continue
        layer = int(m.group(1))
        experts = mu.size
        slots = experts // a.shards
        linear = [
            sum(model.slot(x) for x in mu[r * slots : (r + 1) * slots])
            for r in range(a.shards)
        ]
        held, cost = place(mu, a.shards, a.spare, model)
        placed = [sum(cost[e] for e in h) for h in held]
        before.append(max(linear))
        after.append(max(placed))
        print(
            f"layer {layer:3d} rows/step {mu.sum():8.1f} hottest expert "
            f"{mu.max() / mu.mean():5.1f}x mean | modelled straggler FFN us: "
            f"contiguous {max(linear):6.1f} placed {max(placed):6.1f} "
            f"(rank mean {np.mean(placed):6.1f})"
        )
        table[str(layer)] = held
    out = {"experts": experts, "shards": a.shards, "layers": table}
    if a.spare:
        out["slots_per_shard"] = experts // a.shards + a.spare
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(
        f"wrote {a.out}: {len(table)} layers, modelled straggler FFN summed over "
        f"layers {sum(before) / 1e3:.2f} -> {sum(after) / 1e3:.2f} ms/step"
    )


if __name__ == "__main__":
    main()
