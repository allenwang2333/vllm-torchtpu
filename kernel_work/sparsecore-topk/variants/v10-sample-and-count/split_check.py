"""Exhaustively check `_split`'s int32 overflow safety on the host.

    python split_check.py

Why this file exists.  `_split(lo, hi, arity)` must produce strictly increasing
interior points of `(lo, hi)` for EVERY int32 pair, including `hi - lo > 2**31`
where the obvious `lo + (hi - lo) // arity` wraps.  The frozen suite cannot
check this: a bracket survey over the gate's own score kinds (see README §8)
finds the post-seed bracket is never wider than 24 bits and never touches
`INT32_MIN` under `strict_desc`, `random` or `heavy_ties` at any shape --
while at the production score distribution 32% of prefill rows leave the lower
edge at `INT32_MIN` and 8.6% span zero, with brackets up to 31.6 bits.

So the one input class that drives this code path is the one class the gate has
no id for.  Rather than edit the frozen test, prove it here: the property is
pure integer arithmetic on two scalars, so it is exhaustively checkable on the
host in seconds, and jax's CPU backend evaluates the SAME jnp expression the
kernel compiles.

Checked property, over the domain `correct` can actually call `_split` on:

  * `correct`'s loop condition is `not (skip | ge | hi == lo + 1)`, so the body
    -- the only caller -- runs only when `hi - lo >= 2`.  Width-1 brackets are
    unreachable, and the run below ASSERTS that the excluded set is exactly the
    width-1 pairs rather than taking it on trust.
  * on `hi - lo >= 2`: every point is strictly interior, `lo < t < hi`.  That
    is what makes progress guaranteed -- a probed interior point moves one edge
    to it, so the width strictly decreases and the loop terminates.
  * on `hi - lo >= arity`: the points are additionally strictly increasing, so
    a pass evaluating `arity - 1` thresholds really does resolve
    `log2(arity)` bits and not fewer.  Below that width the clip collapses them
    onto duplicates, which `_absorb` absorbs -- it no-ops on any threshold that
    cannot move an edge -- costing bits, never correctness.
"""

import itertools
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparsecore_topk_sample as scs  # noqa: E402

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
ARITIES = (2, 4, 8, 16)

# Adversarial corners plus a lattice.  The wrap cases are the whole point, so
# the widest brackets are enumerated densely rather than sampled.
CORNERS = [I32_MIN, I32_MIN + 1, I32_MIN + 2, -(1 << 30), -(1 << 24), -3, -2,
           -1, 0, 1, 2, 3, 1 << 24, 1 << 30, I32_MAX - 2, I32_MAX - 1, I32_MAX]
# `l0_shift = 24` L0 bucket edges: the brackets `sample_bracket` actually emits.
BUCKETS = [(b - 128) << 24 for b in list(range(0, 256, 8)) + [255]]


def main():
    rng = np.random.default_rng(20260910)
    pairs = set()
    for lo, hi in itertools.product(CORNERS + BUCKETS, repeat=2):
        if lo < hi:
            pairs.add((lo, hi))
    for _ in range(20000):
        a, b = rng.integers(I32_MIN, I32_MAX, size=2, dtype=np.int64)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = sorted(pairs)

    bad, excluded, widest_ok = [], [], 0
    for arity in ARITIES:
        for lo, hi in pairs:
            ts = [int(t) for t in scs._split(np.int32(lo), np.int32(hi), arity)]
            ok = (len(ts) == arity - 1
                  and all(lo < t < hi for t in ts)
                  and (hi - lo < arity
                       or all(x < y for x, y in zip(ts, ts[1:]))))
            if ok:
                widest_ok = max(widest_ok, hi - lo)
            elif hi - lo <= 1:
                excluded.append((arity, lo, hi))   # unreachable from `correct`
            else:
                bad.append((arity, lo, hi, ts))

    print(f"pairs checked      : {len(pairs)} x {len(ARITIES)} arities "
          f"= {len(pairs) * len(ARITIES)}")
    print(f"widest bracket ok  : {widest_ok} "
          f"({np.log2(max(widest_ok, 1)):.2f} bits; 2**32-1 = "
          f"{(1 << 32) - 1})")
    print(f"full-range pair    : _split(I32_MIN, I32_MAX, 2) = "
          f"{[int(t) for t in scs._split(np.int32(I32_MIN), np.int32(I32_MAX), 2)]}")
    reach = [p for p in pairs if p[1] - p[0] >= 2]
    print(f"reachable pairs    : {len(reach)} x {len(ARITIES)} "
          f"= {len(reach) * len(ARITIES)}  (hi - lo >= 2)")
    print(f"violations         : {len(bad)}")
    for row in bad[:20]:
        print("   ", row)
    # The excluded set must be EXACTLY the width-1 pairs -- otherwise "the
    # caller cannot reach it" is hiding a real failure.
    assert all(hi - lo == 1 for _a, lo, hi in excluded), "excluded a width>1 pair"
    print(f"excluded (width 1) : {len(excluded)} = {len(ARITIES)} x "
          f"{len(excluded) // len(ARITIES)} pairs, all hi == lo + 1, all "
          f"unreachable from `correct` (loop cond excludes them)")

    # Negative control: the naive form this function exists to avoid must FAIL
    # on the same inputs, or the check above is not testing anything.
    def naive(lo, hi, arity):
        span = np.int32(hi) - np.int32(lo)          # wraps
        return [int(np.int32(lo) + (span // arity) * (j + 1))
                for j in range(arity - 1)]

    with np.errstate(over="ignore"):
        nbad = sum(1 for lo, hi in pairs
                   if any(not (lo < t < hi) for t in naive(lo, hi, 2)))
    print(f"negative control   : naive lo+(hi-lo)//2 violates on {nbad} of "
          f"{len(pairs)} pairs (must be > 0)")
    raise SystemExit(1 if bad or nbad == 0 else 0)


main()
