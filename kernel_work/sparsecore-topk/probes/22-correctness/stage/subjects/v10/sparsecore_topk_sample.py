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
"""v10: locate the k-th key by SAMPLING then COUNTING, not by radix partition.

`design.md` §9 Plan E.  The incumbent finds the k-th largest key with a dense
256-bucket histogram of the top 8 bits (`plsc.addupdate_scatter` into
tilespmem, one scatter per element) followed by up to three 8-bit refinement
levels of the same shape.  This module replaces both with:

1. one L0 histogram of **one element in sixteen**, giving an *estimate* of the
   bucket the k-th key lives in; and
2. a correction loop of **count passes** -- a compare and a lane
   population-count per vector, no memory scatter at all -- that walks the
   estimate until it brackets the k-th key exactly.

`probes/15-mandatory-v3-v10` Probe A measured a count pass at 20.681 us against
the dense scatter's 108.217 us, a ratio of 0.191x, and placed it at the
read-only walk floor (`cnt - cvt = 0.354 us`, below the 1.282 us resolution
floor).  So a count pass is 5.2x cheaper.  It also carries 8x less information:
a histogram pass resolves 8 bits, a single-threshold count pass resolves 1.
Whether the trade pays is a property of the score distribution and is settled
by `pass_count_sim.py`, not by this file.  Read that file's output before
believing any prediction attached to this one.

Exactness
---------
The emit pass may only emit `key > thr` plus a quota drawn from `key == thr`.
Filling a quota out of a *band* of distinct values returns the wrong set, so the
correction loop must converge on a `thr` with `count(> thr) <= k <= count(>=
thr)` -- the exact k-th order statistic.  There is therefore **no iteration
budget in this module that can be lowered to buy speed**: `_split` shrinks the
bracket by at least one key per pass and `lax.while_loop` runs until `done`,
which is set only by a proof of the bracket condition.  That is Plan E risk row
(c), and it is why the loop is a `while_loop` and not a truncated `fori_loop`.

Rival arms
----------
`COUNT_ARITY` is the number of intervals each count pass splits the live bracket
into; `COUNT_ARITY - 1` thresholds are evaluated in one walk.  `2` is the plan
as written.  `4`, `8`, `16` trade an unmeasured marginal cost per extra
threshold against `log2(COUNT_ARITY)` bits per pass -- see README §5 and §9;
the marginal is the one term that decides this node and nothing has measured it.
"""

# Dimensions.  Same letters as `sparsecore_topk.py`, which this file extends;
# it defines no dimension of its own.
#   B        rows in the batch                    (1024 prefill / 64 decode)
#   N        padded per-row score capacity        (10240 at the shipping shape)
#   K        elements selected per row            (2048)
#   L        live length of one row, `i_hi`       (<= N; E[L]/N = rho)
#   LANES    16-lane SparseCore vector, from `sparsecore_topk.LANES`
#   nw       grid extent over L in vectors, `walk_chunks` = ceil(L / LANES)
#   ns       grid extent of the SAMPLE walk, ceil(nw / SAMPLE_STRIDE)
#   NB       radix width, `NUM_BUCKETS` = 256
#   A        `COUNT_ARITY`; one count pass evaluates A - 1 thresholds

import math

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu_sc as plsc

# One element in sixteen, taken as every SAMPLE_STRIDE-th *vector*.  A
# systematic cluster sample spread over the whole row, not a contiguous 1/16
# prefix: these are attention scores over KV positions and carry positional
# structure, so a prefix would be biased and the bracket would miss more often.
SAMPLE_STRIDE = 16

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def sample_target_rank(k: int) -> int:
    """Rank within the sample that estimates rank `k` in the row.

    The sample is every SAMPLE_STRIDE-th vector of the live prefix, so its size
    is `eff_len / SAMPLE_STRIDE` to within one vector and the scaling is a
    static shift -- no integer divide on the scalar unit.  A scaling error only
    costs correction passes; the bracket is verified by count passes before it
    is used, so it can never cost correctness.
    """
    return -(-k // SAMPLE_STRIDE)


def _split(lo, hi, arity):
    """`arity - 1` interior split points of `(lo, hi)`, strictly increasing.

    Overflow-safe: `hi - lo` reaches 2**32 - 1 on the first pass after a failed
    bracket, which does not fit in int32.  `(hi >> 1) - (lo >> 1)` is exactly
    representable for every int32 pair, and the split points are then built by
    repeated addition so no partial sum leaves `[lo, hi]`.
    """
    half = jnp.right_shift(hi, 1) - jnp.right_shift(lo, 1)
    bits = int(math.log2(arity))
    step = jnp.maximum(jnp.right_shift(half, bits - 1), jnp.int32(1))
    out, t = [], lo
    for _ in range(arity - 1):
        t = t + step
        out.append(jnp.clip(t, lo + 1, hi - 1))
    return out


def make_count_pass(load_keys, lane_reduce, unroll):
    """`thrs -> (count(key >= t) for t in thrs)`, one walk of the row.

    No store and no scatter: the counts land in scalar carries.  That is the
    whole reason this node exists.
    """

    def count_pass(thrs, bound):
        zero = tuple(jnp.int32(0) for _ in thrs)

        def body(i, carry):
            key, valid, _ = load_keys(i)
            return tuple(c + lane_reduce(valid & (key >= t))
                         for c, t in zip(carry, thrs))

        got = plsc.parallel_loop(0, bound, unroll=unroll, carry=zero)(body)
        return tuple(got) if isinstance(got, (tuple, list)) else (got, )

    return count_pass


def sample_bracket(
    *,
    # Inputs.  All closures and refs are the caller's; this function owns none.
    load_keys,        # (i) -> (key i32[LANES], valid bool[LANES], j i32[LANES])
    walk_chunks,      # nw, i32 scalar
    active,           # bool scalar; False on an empty or keep-all row
    num_buckets,      # NB, python int
    l0_shift,         # python int; 24, the top-8-bits radix
    lane_iota,        # i32[LANES], 0..LANES-1
    # Scratch, owned and mutated by the caller's sweep closures.
    priv_vmem,        # i32[LANES * NB + LANES]  lane-private histograms
    zero_priv,        # (bound) -> ()      clears priv_vmem
    fold_priv,        # (bound) -> ()      folds LANES copies into glob_vmem
    scan_glob,        # (bound) -> i32     exclusive scan, returns the total
    find_bucket,      # (bound, thresh) -> i32   first bucket above `thresh`
    ks: int,          # target rank WITHIN the sample, = ceil(K / SAMPLE_STRIDE)
):
    """L0 histogram of the 1/16 sample -> the estimated bucket `[lo, hi)`.

    Returns int32 `(lo, hi)` key bounds.  **Unverified** -- the correction loop
    probes both edges before it trusts either, and simply keeps the wider bound
    if one fails.
    """
    samp_chunks = jnp.right_shift(walk_chunks + (SAMPLE_STRIDE - 1),
                                  int(math.log2(SAMPLE_STRIDE)))
    samp_chunks = jnp.where(active, samp_chunks, 0)
    sweep = jnp.where(active, num_buckets, 0)

    zero_priv(sweep)

    def hist_body(i):
        key, valid, _ = load_keys(i * SAMPLE_STRIDE)
        bucket = jnp.right_shift(key, l0_shift) + jnp.int32(num_buckets // 2)
        addr = lane_iota * num_buckets + bucket
        plsc.addupdate_scatter(priv_vmem, (addr, ),
                               jnp.ones_like(lane_iota),
                               mask=valid)

    plsc.parallel_loop(0, samp_chunks, unroll=4)(hist_body)
    fold_bound = jnp.where(active, num_buckets // 16, 0)
    fold_priv(fold_bound)
    total_s = scan_glob(fold_bound)
    b_star = find_bucket(fold_bound, total_s - jnp.int32(ks))

    lo = jnp.left_shift(b_star - jnp.int32(num_buckets // 2),
                        jnp.int32(l0_shift))
    hi = jnp.where(b_star >= num_buckets - 1, jnp.int32(INT32_MAX),
                   lo + jnp.int32(1 << l0_shift))
    empty = (total_s <= 0) | jnp.logical_not(active)
    return (jnp.where(empty, jnp.int32(INT32_MIN), lo),
            jnp.where(empty, jnp.int32(INT32_MAX), hi))


def _absorb(state, t, c, k):
    """Fold one probed threshold into the bracket.

    Invariant maintained: `count(>= LO) >= k > count(>= HI)`.
    `ge` records that a probe hit `count(>= t) == k` exactly, in which case the
    answer set is `key >= t` and no quota is needed.
    """
    lo, hi, c_hi, ge = state
    fresh = jnp.logical_not(ge)
    hit = fresh & (c == k)
    raise_lo = fresh & (c > k) & (t > lo)
    lower_hi = fresh & (c < k) & (t < hi)
    lo = jnp.where(hit | raise_lo, t, lo)
    hi = jnp.where(lower_hi, t, hi)
    c_hi = jnp.where(lower_hi, c, c_hi)
    return lo, hi, c_hi, ge | hit


def correct(*, count_pass, walk_chunks, lo0, hi0, k, arity, skip):
    """Walk the estimate until it brackets the k-th key exactly.

    Returns `(thr, c_hi, ge)`: emit `key > thr` plus `k - c_hi` drawn from
    `key == thr`, or `key >= thr` outright when `ge`.

    `skip` short-circuits the whole search for rows that keep everything
    (`total <= k`) and for empty rows; those pay no count pass at all.
    """
    lo = jnp.int32(INT32_MIN)
    hi = jnp.int32(INT32_MAX)
    c_hi = jnp.int32(0)
    ge = jnp.zeros((), jnp.bool_)
    state = (lo, hi, c_hi, ge)

    def done_of(st):
        lo, hi, _, ge = st
        return ge | (hi == lo + 1)

    # --- seed pass(es): probe the sample's bracket edges.  At arity 2 that is
    # two passes; from arity 4 up both edges fit in one pass and the spare
    # threshold slots are spent inside the estimated bucket, which is free --
    # the walk is happening either way.
    if arity == 2:
        seed_rounds = ([lo0], [hi0])
    else:
        extra = arity - 3
        inner = []
        if extra > 0:
            w = jnp.right_shift(hi0, 1) - jnp.right_shift(lo0, 1)
            st = jnp.maximum(jnp.right_shift(w, int(math.log2(arity)) - 1),
                             jnp.int32(1))
            t = lo0
            for _ in range(extra):
                t = t + st
                inner.append(t)
        seed_rounds = ([lo0] + inner + [hi0], )

    for thrs in seed_rounds:
        bound = jnp.where(skip | done_of(state), 0, walk_chunks)
        counts = count_pass(thrs, bound)
        for t, c in zip(thrs, counts):
            state = _absorb(state, t, c, k)

    # --- bisection / arity-ary search on the surviving bracket.
    def cond(st):
        return jnp.logical_not(skip | done_of(st))

    def body(st):
        lo, hi, _, _ = st
        thrs = _split(lo, hi, arity)
        counts = count_pass(thrs, walk_chunks)
        for t, c in zip(thrs, counts):
            st = _absorb(st, t, c, k)
        return st

    lo, hi, c_hi, ge = lax.while_loop(cond, body, state)
    return lo, c_hi, ge
