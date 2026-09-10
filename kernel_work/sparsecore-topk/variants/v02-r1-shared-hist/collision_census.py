"""How much intra-vector address conflict did probes/08's bit-identical run
actually cover?

v2's whole correctness argument is that `plsc.addupdate_scatter` resolves
address conflicts *within* one 16-lane vector, so sixteen lanes may accumulate
into one shared histogram. The incumbent never relies on that: its
`lane_iota * NUM_BUCKETS` stride gives every lane its own histogram, so two
lanes can never target the same word. `references/08-sparsecore.md:646` states
the opposite of v2's premise ("a scatter-add of ones into a counter array
silently drops duplicates that land in the same 16-lane vector -- the lanes
race") while the primitive table at :541 says `addupdate_scatter` is atomic.

probes/08 measured R=1 bit-identical to R=16 on `random` at b=64 and b=1024.
That settles the contradiction *only if* that input actually produced
conflicts. This script counts them on the host, from the kernel's own key
transform, for every distribution the frozen test and the bench use.

Host NumPy only. No TPU, no timing, no kernel.
"""

import numpy as np

LANES = 16
NUM_BUCKETS = 256
L0_SHIFT = 24
MONO_MASK = 0x7FFFFFFF
CSA_TOPK = 2048
SEED = 20260904  # sctopk_2stage_bench.py:118 and the frozen test file:67


def monotone_key(bits: np.ndarray) -> np.ndarray:
    m = np.right_shift(bits, 31) & np.int32(MONO_MASK)
    return np.bitwise_xor(bits, m).astype(np.int32)


def l0_bucket(scores: np.ndarray) -> np.ndarray:
    key = monotone_key(scores.astype(np.float32).view(np.int32))
    return (np.right_shift(key, L0_SHIFT) + NUM_BUCKETS // 2).astype(np.int32)


def census(name: str, scores: np.ndarray, eff_len: int):
    """Per-row vector-level conflict census over the level-0 histogram pass."""
    b, n = scores.shape
    buckets = l0_bucket(scores)[:, :eff_len]
    nv = eff_len // LANES
    v = buckets[:, :nv * LANES].reshape(b, nv, LANES)
    v_sorted = np.sort(v, axis=2)
    dups = (v_sorted[:, :, 1:] == v_sorted[:, :, :-1]).sum(axis=2)
    vectors_with_conflict = int((dups > 0).sum())
    colliding_lanes = int(dups.sum())
    distinct = np.array([len(np.unique(buckets[i])) for i in range(min(b, 8))])
    print(f"  {name:<12} rows={b:<5} n_eff={eff_len:<6} vectors={b * nv:<9} "
          f"conflicting_vectors={vectors_with_conflict:<9} "
          f"({100.0 * vectors_with_conflict / (b * nv):5.1f}%)  "
          f"colliding_lane_pairs={colliding_lanes:<10} "
          f"distinct_L0_buckets(first rows)={distinct.tolist()}")
    return vectors_with_conflict, colliding_lanes


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    n = 10240
    print("L0 histogram intra-vector conflict census, n=10240 (--ctx 9216)")
    for b in (64, 1024):
        print(f"-- b={b}")
        census("random", rng.standard_normal((b, n), dtype=np.float32), n)
        census("heavy_ties",
               rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(
                   np.float32), n)
        census("strict_desc",
               np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0),
               n)
