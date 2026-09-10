"""Probe 05 -- how many of the 3 refinement levels actually run, per row and per subcore?

Runs on the HOST, not on a TPU.  `done` in the kernel is a deterministic function of the
score data alone, so the loop-trip count is derivable exactly by replaying the kernel's own
integer arithmetic in numpy.  Nothing here is a timing measurement.

The number matters because report Sec.5d's cost model multiplies it by the price of a
refinement pass (probe 03), and the refinement term is ~51% of the shipping prefill kernel.
A 2x error in it reorders the whole step-2 candidate list.

The per-SUBCORE number is the one that costs time.  `sparsecore_topk.py` has no
`subcore_barrier` anywhere (lines 19 and 528 both advertise this), so a wave costs what its
SLOWEST subcore costs, not what its average one does -- subcore `flat` owns rows
`flat, flat+32, flat+64, ...` per line 157.

    python probe.py
"""

import numpy as np

# Mirrors sparsecore_topk.py:41-51.
LANES = 16
NUM_BUCKETS = 256
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
BITS_NEG_INF = np.float32(-np.inf).view(np.int32)
KEY_NEG_INF = int(BITS_NEG_INF ^ np.int32(MONO_MASK))

NUM_SUBCORES_TOTAL = 32  # num_sc_cores(2) * num_subcores(16), verified in-job.


def monotone_key(bits):
    """sparsecore_topk.py:116-118.  An involution; order-preserving on fp32."""
    m = np.right_shift(bits, 31) & np.int32(MONO_MASK)
    return np.bitwise_xor(bits, m).astype(np.int32)


def find_bucket(cum, thresh):
    """sparsecore_topk.py:148-154 -- how many buckets have cumulative count <= thresh."""
    return int(np.count_nonzero(cum <= thresh))


def levels_for_row(scores, k):
    """Replay sparsecore_topk.py:183-265 for one row.  Returns the refinement trip count."""
    keys = monotone_key(scores.view(np.int32))
    valid = keys > np.int32(KEY_NEG_INF)
    keys = keys[valid]

    # hist0 (:187-197) then the inclusive scan (:208).
    hist = np.bincount((np.right_shift(keys, L0_SHIFT) + NUM_BUCKETS // 2),
                       minlength=NUM_BUCKETS)
    cum = np.cumsum(hist)
    total = int(cum[-1])

    keep_all = total <= k
    b_star = find_bucket(cum, total - k)
    cum_at = int(cum[b_star])
    cum_lo = int(cum[b_star - 1]) if b_star > 0 else 0
    cnt = cum_at - cum_lo
    quota = k - (total - cum_at)

    prefix = b_star - NUM_BUCKETS // 2
    shift = L0_SHIFT
    done = keep_all or (quota == cnt)

    ran = 0
    for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
        if done:
            # :224-245 -- zero_priv(0), active_walk=0, fold_bound=0.  No pass over L.
            continue
        ran += 1
        match = np.right_shift(keys, shift) == prefix
        sub = np.bincount(np.right_shift(keys[match], lvl_shift) & lvl_mask,
                          minlength=NUM_BUCKETS)
        cum2 = np.cumsum(sub)
        sub_total = int(cum2[-1])

        b2 = find_bucket(cum2, sub_total - quota)
        cum_at2 = int(cum2[b2])
        cum_lo2 = int(cum2[b2 - 1]) if b2 > 0 else 0
        cnt2 = cum_at2 - cum_lo2
        quota2 = quota - (sub_total - cum_at2)

        prefix = (prefix << (shift - lvl_shift)) | b2
        shift = lvl_shift
        quota, cnt = quota2, cnt2
        done = (quota2 == cnt2) or (quota2 == 0)
    return ran


def critical_path(levels):
    """Cost is max over subcores of the sum of ITS OWN rows -- there is no barrier."""
    b = len(levels)
    per_sc = [sum(levels[f::NUM_SUBCORES_TOTAL]) for f in range(NUM_SUBCORES_TOTAL)]
    rows_per_sc = [len(levels[f::NUM_SUBCORES_TOTAL]) for f in range(NUM_SUBCORES_TOTAL)]
    waves = max(rows_per_sc)
    # Per-wave critical subcore, expressed per row so it composes with the cost model.
    return sum(levels) / b, max(per_sc) / waves, waves


def make(dist, b, n, rng):
    if dist == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if dist == "uniform":
        return rng.random((b, n), dtype=np.float32)
    if dist == "heavy_ties":
        # 17 distinct values -- forces every level, the worst case the tests cover.
        vals = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
        return vals[rng.integers(0, 17, size=(b, n))]
    raise ValueError(dist)


def main():
    k, n = 2048, 10240
    print(f"{'dist':<12} {'b':>5} {'mean/row':>9} {'crit/row':>9} "
          f"{'imbal':>7} {'waves':>6} {'passes/row':>11}")
    for dist in ("random", "uniform", "heavy_ties"):
        for b in (64, 1024):
            rng = np.random.default_rng(20260904)
            scores = make(dist, b, n, rng)
            levels = [levels_for_row(scores[r], k) for r in range(b)]
            mean, crit, waves = critical_path(levels)
            # cvt + hist0 + refinement + emit, all passes over the full L.
            print(f"{dist:<12} {b:>5} {mean:>9.2f} {crit:>9.2f} "
                  f"{crit / mean:>7.2f}x {waves:>6} {3 + crit:>11.2f}")


if __name__ == "__main__":
    main()
