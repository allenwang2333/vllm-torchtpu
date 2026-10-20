#!/usr/bin/env python3
"""Write a synthetic routing table (npz, topk_ids [L, T, topk]) for
bench_moe_attr.py --routing-npz: distinct experts per row, drawn with the
experts of device 0 weighted so that device 0 receives `--hot` times the mean
routed rows (1.0 = uniform)."""

import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--hot", type=float, default=1.0)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--tokens", type=int, default=2304)
ap.add_argument("--experts", type=int, default=384)
ap.add_argument("--ep", type=int, default=8)
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
g = a.experts // a.ep
weight = (a.ep - 1) * a.hot / (a.ep - a.hot)  # device-0 expert weight vs 1
p = np.ones(a.experts)
p[:g] = weight
p /= p.sum()
rng = np.random.default_rng(a.seed)
# Gumbel top-k == sampling without replacement proportional to p.
keys = np.log(p)[None, None, :] - np.log(
    -np.log(rng.random((a.layers, a.tokens, a.experts)))
)
ids = np.argsort(-keys, axis=-1)[..., : a.topk].astype(np.int32)
rows0 = (ids < g).sum() / a.layers
print(f"device-0 rows {rows0:.0f} vs mean {a.tokens * a.topk / a.ep:.0f}")
np.savez(a.out, topk_ids=ids)
