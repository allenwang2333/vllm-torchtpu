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
"""Expert placement tables for N shards from recorded decode routing.

Produces the JSON schema `vllm_torchtpu.expert_placement` reads (shards,
experts, owned_per_core, slots_per_core, copies, extra_slots, layers ->
cores) for any shard count that divides the expert count: 4 (EP4 x TP2 on
8 ranks), 8, 16 (EP16, or EP8 x TP2 on 16 ranks), 32 (EP32, or EP16 x TP2
on 32 ranks).

Objective (the same as the 8-shard agentx_corr tables): the replayed mean,
over sampled decode buckets, of the largest number of distinct experts any
shard streams in a layer. A bucket is the union of randomly drawn recorded
decode steps whose rows add up to the bucket's rows, as in
qwen35-agentx-p2 E1's expert_count_sim. The primaries are placed by a
frequency snake and improved by a swap local search on that objective; the
optional copy slots are placed greedily by the replayed gain of serving an
expert's hits from a lighter shard (a first-order proxy of the megakernel's
two-pass Po2C; the exact Po2C replay is `expert_count_sim.py`).

Usage:
  solve_shards.py --routing moe_profile.npz --shards 16 --copies 0 \
      --bucket-rows 128,512 --samples 64 --out expert_placement_ep16_agentx_corr.json

The recording is an eplb moe_profile npz with dec_ring [L, S, E] (uint8: the
logical experts any row of a decode step hit) and dec_ring_rows [L, S].
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def load_recording(path: str):
    d = np.load(path, allow_pickle=True)
    ring = d["dec_ring"]
    rows = d["dec_ring_rows"]
    freq = d["histogram_decode"] if "histogram_decode" in d.files else None
    return ring, rows, freq


def sample_buckets(
    ring_l: np.ndarray,
    rows_l: np.ndarray,
    bucket_rows: list[int],
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Union-of-steps bucket samples for one layer: bool [samples, E]."""
    valid = np.flatnonzero(rows_l > 0)
    if valid.size == 0:
        raise ValueError("layer has no recorded decode steps")
    out = np.zeros((samples, ring_l.shape[1]), dtype=bool)
    for s in range(samples):
        target = int(bucket_rows[s % len(bucket_rows)])
        got = 0
        hit = np.zeros(ring_l.shape[1], dtype=bool)
        while got < target:
            i = int(valid[rng.integers(valid.size)])
            hit |= ring_l[i].astype(bool)
            got += int(rows_l[i])
        out[s] = hit
    return out


def snake_primaries(freq: np.ndarray, shards: int) -> np.ndarray:
    """Experts by descending frequency dealt to shards in snake order: an
    exactly balanced count per shard with the hot experts spread."""
    e = freq.size
    per = e // shards
    order = np.argsort(-freq, kind="stable")
    prim = np.empty(e, dtype=np.int64)
    counts = np.zeros(shards, dtype=np.int64)
    pos = 0
    direction = 1
    for g in order:
        # next shard in snake order that still has room
        for _ in range(shards):
            if counts[pos] < per:
                break
            pos += direction
            if pos == shards or pos < 0:
                direction = -direction
                pos += direction
        prim[g] = pos
        counts[pos] += 1
        pos += direction
        if pos == shards or pos < 0:
            direction = -direction
            pos += direction
    assert (counts == per).all()
    return prim


def mean_max(load: np.ndarray) -> float:
    return float(load.max(axis=1).mean())


def improve_primaries(
    hits: np.ndarray,
    prim: np.ndarray,
    shards: int,
    iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Swap local search on the replayed mean max-shard load. Returns the
    primaries, the objective before and after."""
    hits_f = hits.astype(np.int32)
    onehot = np.zeros((prim.size, shards), dtype=np.int32)
    onehot[np.arange(prim.size), prim] = 1
    load = hits_f @ onehot  # [S, R]
    before = mean_max(load)
    best = before
    members = [list(np.flatnonzero(prim == r)) for r in range(shards)]
    argmax_count = np.bincount(load.argmax(axis=1), minlength=shards)
    for it in range(iters):
        # a from a shard that is often the maximum, b from a light shard
        if rng.random() < 0.7:
            ra = int(rng.choice(shards, p=(argmax_count + 1) / (argmax_count + 1).sum()))
        else:
            ra = int(rng.integers(shards))
        mean_load = load.mean(axis=0)
        light = np.argsort(mean_load)
        rb = int(light[rng.integers(min(4, shards))])
        if ra == rb:
            continue
        a = int(rng.choice(members[ra]))
        b = int(rng.choice(members[rb]))
        da = hits_f[:, b] - hits_f[:, a]
        new_a = load[:, ra] + da
        new_b = load[:, rb] - da
        other = load.copy()
        other[:, ra] = new_a
        other[:, rb] = new_b
        cand = mean_max(other)
        if cand < best - 1e-12:
            best = cand
            load = other
            prim[a], prim[b] = rb, ra
            members[ra].remove(a)
            members[ra].append(b)
            members[rb].remove(b)
            members[rb].append(a)
            argmax_count = np.bincount(load.argmax(axis=1), minlength=shards)
    return prim, before, best


def place_copies(
    hits: np.ndarray,
    prim: np.ndarray,
    shards: int,
    extra_per_shard: int,
    freq: np.ndarray,
) -> tuple[list[list[int]], float]:
    """Greedy copy slots: each shard gets extra_per_shard copies of experts it
    does not own, chosen by the replayed gain of serving the expert's hits
    there whenever that shard is lighter than the owner at that step
    (first-order proxy of the megakernel's Po2C)."""
    hits_f = hits.astype(np.int32)
    onehot = np.zeros((prim.size, shards), dtype=np.int32)
    onehot[np.arange(prim.size), prim] = 1
    load = hits_f @ onehot
    copies: list[list[int]] = [[] for _ in range(shards)]
    copied: set[int] = set()
    hot = list(np.argsort(-freq)[: max(8 * shards * extra_per_shard, 64)])
    for _ in range(extra_per_shard * shards):
        best_gain, best_pick = 0.0, None
        base = mean_max(load)
        for g in hot:
            if g in copied:
                continue
            r0 = int(prim[g])
            hit_g = hits_f[:, g] == 1
            if not hit_g.any():
                continue
            for r1 in range(shards):
                if r1 == r0 or len(copies[r1]) >= extra_per_shard:
                    continue
                move = hit_g & (load[:, r0] > load[:, r1] + 1)
                if not move.any():
                    continue
                trial = load.copy()
                trial[move, r0] -= 1
                trial[move, r1] += 1
                gain = base - mean_max(trial)
                if gain > best_gain + 1e-12:
                    best_gain, best_pick = gain, (int(g), r1, trial)
        if best_pick is None:
            break
        g, r1, trial = best_pick
        copies[r1].append(g)
        copied.add(g)
        load = trial
    # fill any shard short of slots with cold unowned experts (no gain, valid table)
    cold = list(np.argsort(freq))
    for r1 in range(shards):
        for g in cold:
            if len(copies[r1]) >= extra_per_shard:
                break
            if int(prim[g]) != r1 and g not in copied:
                copies[r1].append(int(g))
                copied.add(int(g))
    return copies, mean_max(load)


def solve(
    routing: str,
    shards: int,
    copies: int,
    bucket_rows: list[int],
    samples: int,
    iters: int,
    seed: int,
    name: str,
    layers: int | None = None,
) -> tuple[dict, dict]:
    ring, rows, freq_all = load_recording(routing)
    n_layers, _, experts = ring.shape
    if experts % shards:
        raise ValueError(f"{experts} experts do not split over {shards} shards")
    if copies % shards:
        raise ValueError(f"{copies} copies must be a multiple of {shards} shards")
    owned = experts // shards
    extra = copies // shards
    rng = np.random.default_rng(seed)
    table_layers: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    t0 = time.time()
    for l in range(n_layers if layers is None else min(layers, n_layers)):
        hits = sample_buckets(ring[l], rows[l], bucket_rows, samples, rng)
        freq = (
            freq_all[l].astype(np.float64)
            if freq_all is not None
            else hits.mean(axis=0)
        )
        prim = snake_primaries(freq, shards)
        prim, before, after = improve_primaries(hits, prim, shards, iters, rng)
        cores = [list(map(int, np.flatnonzero(prim == r))) for r in range(shards)]
        layer_entry: dict = {}
        after_copies = after
        if extra:
            cps, after_copies = place_copies(hits, prim, shards, extra, freq)
            cores = [c + cps[r] for r, c in enumerate(cores)]
            layer_entry["copies"] = {
                str(g): r for r, c in enumerate(cores) for g in c[owned:]
            }
        layer_entry = {"cores": cores, **layer_entry}
        table_layers[str(l)] = layer_entry
        uniform = float(
            np.mean([np.bincount(rng.integers(shards, size=int(h.sum())), minlength=shards).max() for h in hits])
        )
        stats[str(l)] = {
            "mean_max_snake": round(before, 3),
            "mean_max_primaries": round(after, 3),
            "mean_max_with_copies": round(after_copies, 3),
            "mean_distinct_hit": round(float(hits.sum(axis=1).mean()), 2),
            "ideal_even_split": round(float(hits.sum(axis=1).mean()) / shards, 3),
            "random_assignment_reference": round(uniform, 3),
        }
        print(
            f"layer {l}: distinct {stats[str(l)]['mean_distinct_hit']}, snake {before:.2f}, "
            f"primaries {after:.2f}, copies {after_copies:.2f}, even {stats[str(l)]['ideal_even_split']:.2f} "
            f"({time.time() - t0:.0f}s)",
            file=sys.stderr,
        )
    table = {
        "shards": shards,
        "experts": experts,
        "owned_per_core": owned,
        "slots_per_core": owned + extra,
        "copies": bool(extra),
        "extra_slots": extra,
        "model": "Qwen/Qwen3.5-397B-A17B-FP8",
        "parallelism": name,
        "recording": routing,
        "solver": "tools/expert_placement/solve_shards.py",
        "algorithm": (
            "frequency snake primaries + swap local search on replayed mean "
            "max-shard distinct experts over union-of-steps decode buckets "
            f"(rows {bucket_rows}, {samples} samples, {iters} swaps); greedy copy "
            "slots by replayed gain (first-order Po2C proxy)"
        ),
        "layers": table_layers,
    }
    return table, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--copies", type=int, default=0, help="total copy slots (multiple of shards)")
    ap.add_argument("--bucket-rows", default="128,512", help="comma list of verify rows per bucket sample")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3000, help="swap proposals per layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None, help="solve only the first N layers (tests)")
    ap.add_argument("--name", default="", help="parallelism label stored in the table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None, help="per-layer objective JSON")
    a = ap.parse_args(argv)
    bucket_rows = [int(x) for x in a.bucket_rows.split(",") if x.strip()]
    name = a.name or f"EP{a.shards} expert parallel, {a.shards} shards"
    table, stats = solve(
        a.routing, a.shards, a.copies, bucket_rows, a.samples, a.iters, a.seed, name, a.layers
    )
    with open(a.out, "w") as f:
        json.dump(table, f)
    if a.stats:
        with open(a.stats, "w") as f:
            json.dump(stats, f, indent=1)
    print(f"wrote {a.out} ({len(table['layers'])} layers)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
