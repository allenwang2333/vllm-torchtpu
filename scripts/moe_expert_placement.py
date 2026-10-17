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

    moe_expert_placement.py --shards 16 --out placement.json TAP_DIR [TAP_DIR ...]

Each rank's load is its last snapshot minus its `--skip`-th (1-based; 0 keeps
everything), which drops warmup and compile traffic; ranks are then summed.
Per layer, experts are assigned to ranks longest-first onto the least loaded
rank that still has a free slot, then improved by pairwise swaps between the
most and the least loaded rank. Prints, per layer, the max/mean rank load of
the checkpoint's contiguous placement and of the built one.
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_counts(dirs, skip):
    """{layer name: (mean [E] per step, variance [E] per step)} summed over ranks."""
    by_rank = collections.defaultdict(list)
    for d in dirs:
        for path in glob.glob(
            os.path.join(d, "**", "counts_ep*_*.npz"), recursive=True
        ):
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


def balance(load, shards):
    experts = load.size
    slots = experts // shards
    held = [[] for _ in range(shards)]
    total = np.zeros(shards)
    for e in np.argsort(-load, kind="stable"):
        r = min(
            (r for r in range(shards) if len(held[r]) < slots), key=lambda r: total[r]
        )
        held[r].append(int(e))
        total[r] += load[e]
    for _ in range(experts * 4):
        hi, lo = int(np.argmax(total)), int(np.argmin(total))
        gap = total[hi] - total[lo]
        best = None
        for i, a in enumerate(held[hi]):
            for j, b in enumerate(held[lo]):
                d = load[a] - load[b]
                if 0 < d < gap and (best is None or abs(gap - 2 * d) < best[0]):
                    best = (abs(gap - 2 * d), i, j, d)
        if best is None or best[0] >= gap:
            break
        _, i, j, d = best
        held[hi][i], held[lo][j] = held[lo][j], held[hi][i]
        total[hi] -= d
        total[lo] += d
    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=1, help="snapshots to drop")
    ap.add_argument("--layers", default="", help="first:last decoder layers to place")
    a = ap.parse_args()
    counts = load_counts(a.dirs, a.skip)
    lo, hi = (int(x) for x in a.layers.split(":")) if a.layers else (0, 1 << 30)
    table = {}
    for name, (mu, var) in counts.items():
        m = _LAYER_RE.search(name)
        if not m or not lo <= int(m.group(1)) <= hi:
            continue
        layer = int(m.group(1))
        experts = mu.size
        slots = experts // a.shards
        linear = mu.reshape(a.shards, slots).sum(1)
        held = balance(mu, a.shards)
        placed = np.array([mu[h].sum() for h in held])
        sd = np.sqrt(np.array([var[h].sum() for h in held]))
        print(
            f"layer {layer:3d} rows/step {mu.sum():8.1f} rank max/mean "
            f"linear {linear.max() / linear.mean():.3f} placed "
            f"{placed.max() / placed.mean():.3f} | hottest expert "
            f"{mu.max() / mu.mean():.2f}x mean | rank sd/mean "
            f"{sd.mean() / placed.mean():.3f}"
        )
        table[str(layer)] = held
    with open(a.out, "w") as f:
        json.dump({"experts": experts, "shards": a.shards, "layers": table}, f)
    print(f"wrote {a.out}: {len(table)} layers")


if __name__ == "__main__":
    main()
