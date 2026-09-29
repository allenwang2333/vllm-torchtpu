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
"""Threshold-gated stage 1 for the two-stage SparseCore top-k.

`sparsecore_topk_l1` is a drop-in for `sparsecore_topk` with the same
arguments, output and contract. It is enabled at the StreamIndex call site by
`TPU_STREAMIDX_SCTOPK_L1`.

The incumbent's stage 1 finds each slice's exact local top-`k_p` with a
histogram pass plus up to three radix refinement passes over the whole slice,
although stage 2 only needs a superset of each row's global top-k. Here:

1. **T on the TensorCore.** Per row, `T_r = min_g max{key[r, c] : c mod k == g,
   c < len_r}` as a monotone i32 key, from one Pallas pass over the live score
   columns. The k group maxima are k distinct elements `>= T_r`, so
   `T_r <= v_k` (the true k-th largest). An empty group gives `T_r = -inf`,
   i.e. no filtering.
2. **Stage 1' on SparseCore.** One pass `store_compressed(key >= T_r)` of each
   slice's (column, score bits) into a slice-sized VMEM scratch. A slice whose
   survivor count fits in `k_p` copies its survivors out as its candidate list.
   A slice that overflows runs the incumbent radix over its *compacted
   survivors* (its local top-`k_p` is a subset of them), which keeps the
   selection exact. The slice-to-subcore map is rotated per wave, so the hot
   recent slice of a prefill chunk (same slice index on every row) spreads over
   all 32 subcores instead of serialising on one.
3. **Stage 2 and the remap** are the incumbent's, unchanged.

Every element above `v_k` is `>= T_r`, so it reaches stage 2 either directly or
through its slice's exact local top-`k_p`. As with the incumbent, equal scores
at the k-th boundary may be broken differently; the selected score multiset is
exact.

Shapes this path does not cover fall back to `sparsecore_topk` for the whole
call (see `l1_supported`): the single-stage case (P == 1), a stage 2 wider than
one slice (P * k_p > MAX_SLICE_WORDS, the >=1M-context recursion), slices too
wide for the extra survivor scratch, `k` not a multiple of 128, and batches too
tall for the T kernel's VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    BITS_NEG_INF,
    KEY_NEG_INF,
    L0_SHIFT,
    LANES,
    MAX_SLICE_WORDS,
    MONO_MASK,
    NUM_BUCKETS,
    REFINEMENT_LEVELS,
    SCAN_UNROLL,
    _align_to,
    _cdiv,
    _cdiv_dyn,
    _pick_partition,
    _sc_topk_direct,
    sparsecore_topk,
)

# Widest stage-1 slice this path runs, in 32-bit words. On top of the
# incumbent's double-buffered slice (2 x slice) plus its outputs, stage 1' keeps
# two slice-sized survivor scratches (compacted columns and score bits), so the
# per-subcore footprint is ~4 x slice + 6 x k_p words out of 128K. 24576 keeps
# that under ~110K; wider slices take the incumbent.
L1_MAX_SLICE_WORDS = 24 * 1024
# The T kernel streams (rows, J * k) column blocks, double-buffered, next to a
# (rows, k) running-max accumulator.
T_BLOCK_VMEM_BYTES = 32 * 1024 * 1024
T_MAX_J = 8


def _t_block_factor(b: int, n: int, k: int) -> int:
    """Column blocks of J * k for the T kernel; 0 if none fit in VMEM."""
    bp = _align_to(b, 8)
    j = T_MAX_J
    while j > 1 and (2 * bp * j * k * 4 > T_BLOCK_VMEM_BYTES or j * k > n):
        j //= 2
    if 2 * bp * j * k * 4 > T_BLOCK_VMEM_BYTES or j * k > n:
        return 0
    return j


def l1_supported(b: int, n: int, k: int) -> bool:
    """Whether `sparsecore_topk_l1` runs the threshold path for this shape.

    Anything else runs `sparsecore_topk` unchanged.
    """
    if n % LANES != 0 or k % 128 != 0 or not 0 < k <= 4096:
        return False
    p = _pick_partition(b, n, k)
    if p == 1 or n % p != 0:
        return False
    n_p = n // p
    k_p = min(k, n_p)
    if p * k_p > MAX_SLICE_WORDS:
        return False
    if _align_to(n_p, LANES) > L1_MAX_SLICE_WORDS:
        return False
    return _t_block_factor(b, n, k) > 0


def epilogue_t_supported(b: int, n: int, k: int, group_widths) -> bool:
    """Whether the StreamIdxTC scoring epilogue can supply T for this call.

    The epilogue's groups are columns equal mod the scoring kernel's KV block
    width `g` (one running max per block lane), so T = min over g >= k groups
    of the group max is still <= v_k. Every pass must use such a block.
    """
    return l1_supported(b, n, k) and all(
        g >= k and g % 128 == 0 and n % g == 0 for g in group_widths
    )


def _monotone_key(bits):
    """fp32 bits -> i32 key with the same total order; an involution."""
    m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
    return jnp.bitwise_xor(bits, m)


# --------------------------------------------------------------------------- #
# TensorCore: per-row threshold key T.
# --------------------------------------------------------------------------- #


def _row_threshold_kernel(sp_ref, x_ref, len_ref, o_ref, acc_ref, *, k, J, nrt):
    """One read of the scores: per-row group max, then min over the groups.

    Grid axis 0 walks column blocks of width J * k. `acc_ref[r, g]` is the
    running max key of row r's group g (columns c with c mod k == g). Blocks
    wholly past every row's length are neither fetched (the index map clamps to
    the last live block, so the pipeline sees a repeated block index and skips
    the DMA) nor computed; blocks wholly inside every row's length skip the
    mask. `sp_ref` (SMEM) is [number of live blocks, min row length].
    """
    c = pl.program_id(0)
    nc = pl.num_programs(0)
    cb = J * k
    n_live = sp_ref[0]
    min_len = sp_ref[1]
    base = c * cb

    @pl.when(c == 0)
    def _():
        acc_ref[...] = jnp.full(acc_ref.shape, KEY_NEG_INF, jnp.int32)

    def accumulate(masked):
        def body(r, carry):
            r0 = pl.multiple_of(r * 8, 8)
            a = acc_ref[pl.ds(r0, 8), :]
            if masked:
                lim = len_ref[pl.ds(r0, 8), :][:, 0:1]  # (8, 1) row lengths
            for s in range(J):
                key = _monotone_key(x_ref[pl.ds(r0, 8), pl.ds(s * k, k)])
                if masked:
                    col = (base + s * k) + lax.broadcasted_iota(jnp.int32, (8, k), 1)
                    key = jnp.where(col < lim, key, jnp.int32(KEY_NEG_INF))
                a = jnp.maximum(a, key)
            acc_ref[pl.ds(r0, 8), :] = a
            return carry

        lax.fori_loop(0, nrt, body, 0)

    live = c < n_live
    full = base + cb <= min_len
    pl.when(live & full)(lambda: accumulate(False))
    pl.when(live & jnp.logical_not(full))(lambda: accumulate(True))

    @pl.when(c == nc - 1)
    def _():
        def fin(r, carry):
            r0 = pl.multiple_of(r * 8, 8)
            t = jnp.min(acc_ref[pl.ds(r0, 8), :], axis=1, keepdims=True)
            o_ref[pl.ds(r0, 8), :] = jnp.broadcast_to(t, (8, 128))
            return carry

        lax.fori_loop(0, nrt, fin, 0)


def row_threshold_keys(
    scores: jax.Array,  # i32[b, n] of f32 bits, or f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    scheduling_group_id: int | None = None,
) -> jax.Array:  # i32[b]
    """`T_r = min_g max{key[r, c] : c mod k == g, c < len_r}` as an i32 key."""
    return row_threshold_rows(
        scores, k, row_lengths, scheduling_group_id=scheduling_group_id
    )[:, 0]


def row_threshold_rows(
    scores: jax.Array,  # i32[b, n] of f32 bits, or f32[b, n]
    k: int,
    row_lengths: jax.Array,  # i32[b]
    *,
    scheduling_group_id: int | None = None,
) -> jax.Array:  # i32[b, 128]
    """`row_threshold_keys` broadcast over 128 lanes, as stage 1' reads it."""
    b, n = scores.shape
    J = _t_block_factor(b, n, k)
    if J == 0:
        raise ValueError(f"no T block fits VMEM for {b=} {n=} {k=}")
    bits = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    lens = row_lengths.astype(jnp.int32)
    bp = _align_to(b, 8)
    if bp != b:
        bits = jnp.pad(bits, ((0, bp - b), (0, 0)))
        lens = jnp.pad(lens, (0, bp - b))
    lens = jnp.clip(lens, 0, n)
    cb = J * k
    nc = _cdiv(n, cb)
    n_live = (jnp.max(lens) + cb - 1) // cb
    sp = jnp.stack([n_live, jnp.min(lens)]).astype(jnp.int32)
    len2d = jnp.broadcast_to(lens[:, None], (bp, 128))

    def x_map(c, sp_ref):
        return (0, jnp.minimum(c, jnp.maximum(sp_ref[0] - 1, 0)))

    vmem = 2 * bp * cb * 4 + bp * k * 4 + 4 * bp * 128 * 4 + (8 << 20)
    out = pl.pallas_call(
        functools.partial(_row_threshold_kernel, k=k, J=J, nrt=bp // 8),
        out_shape=jax.ShapeDtypeStruct((bp, 128), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(nc,),
            in_specs=[
                pl.BlockSpec((bp, cb), x_map),
                pl.BlockSpec((bp, 128), lambda c, sp_ref: (0, 0)),
            ],
            out_specs=pl.BlockSpec((bp, 128), lambda c, sp_ref: (0, 0)),
            scratch_shapes=[pltpu.VMEM((bp, k), jnp.int32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=int(vmem)
        ),
        name=f"tc_rowT_b{bp}_n{n}_k{k}_J{J}",
    )(sp, bits, len2d)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    return out[:b]


# --------------------------------------------------------------------------- #
# SparseCore: stage 1'.
# --------------------------------------------------------------------------- #


def _topk_body_l1(
    scores_hbm,  # i32[padded_b, slice_len] (fp32 score bits)
    lengths_hbm,  # i32[padded_b, LANES]  lane 0 = length
    t_hbm,  # i32[t_rows, 128]  lane 0 = T key of slices [r * p, (r + 1) * p)
    out_hbm,  # i32[padded_b, k]
    keys_hbm,  # i32[padded_b, k]
    glob_vmem,  # i32[NUM_BUCKETS + LANES]
    rowbuf_vmem,  # i32[k + LANES]
    keybuf_vmem,  # i32[k + LANES]
    cidx_vmem,  # i32[slice_len + LANES]   compacted survivor columns
    ckey_vmem,  # i32[slice_len + LANES]   compacted survivor score bits
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    fallback: bool,
    p: int,
    t_rows: int,
    counts_hbm=None,  # i32[padded_b, LANES]  lane 0 = slots emitted (seg tail)
    global_cols: bool = False,
):
    # `fallback=False` exists only for the test's positive control: it copies the
    # first k survivors out of an overflowing slice, which is wrong.
    num_subcores = 16
    num_sc_cores = 2
    cores_per_chip = num_sc_cores * num_subcores
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def slot_of(w):
        # At wave w subcore `flat` takes slot (flat + w) mod 32 of the wave's 32
        # slices, so a hot slice index p (the same p on every row of a prefill
        # chunk) lands on a different subcore each wave.
        return jnp.bitwise_and(flat + w, cores_per_chip - 1)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def key_window(prefix, shift):
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound):
        def body(i):
            glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket, mask):
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def scan_glob(bound):
        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):
        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    def survivor_radix(out_vmem, keys_vmem, active, walk_chunks, i_hi):
        """The incumbent stage-1 radix, run over the compacted survivors.

        `ckey_vmem` holds the survivors' raw bits in increasing column order and
        element j's slice-local column is `cidx_vmem[j]`. Compaction keeps
        column order, so boundary-tie ranking matches the full slice.
        """
        scores_vmem = ckey_vmem
        zero_hist(NUM_BUCKETS // LANES)

        @pl.when(i_hi > 0)
        def _():
            def hist0_body(i):
                key = _monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
                scores_vmem[pl.ds(i * LANES, LANES)] = key
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
                hist_add(bucket, key > neg_inf_key)

            plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

        total = scan_glob(NUM_BUCKETS // LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            zero_hist(hist_bound)
            active_walk = jnp.where(done, 0, walk_chunks)
            lo_excl, hi_incl = key_window(prefix, shift)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                lo_excl=lo_excl,
                hi_incl=hi_incl,
            ):
                key = scores_vmem[pl.ds(i * LANES, LANES)]
                match = (key > lo_excl) & (key <= hi_incl)
                bucket = jnp.bitwise_and(
                    jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                )
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

            sub_total = scan_glob(hist_bound)
            b2 = find_bucket(hist_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
            c_above2 = sub_total - cum_at2
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - c_above2

            width = shift - lvl_shift
            new_prefix = jnp.left_shift(prefix, width) | b2
            new_done = done | (quota2 == cnt2) | (quota2 == 0)

            prefix = jnp.where(done, prefix, new_prefix)
            shift = jnp.where(done, shift, jnp.int32(lvl_shift))
            c_hi = jnp.where(done, c_hi, c_hi + c_above2)
            quota = jnp.where(done, quota, quota2)
            cnt = jnp.where(done, cnt, cnt2)
            done = new_done

        my_take = jnp.where(keep_all, 0, quota)

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
            keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                (LANES,), BITS_NEG_INF, jnp.int32
            )

        fill_bound = jnp.where(total < k, k // LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        lo_excl, hi_incl = key_window(prefix, shift)
        take_all_ties = keep_all | (my_take >= cnt)
        gate = jnp.where(take_all_ties, lo_excl, hi_incl)
        whole_bucket = take_all_ties | (my_take <= 0)
        emit_chunks = jnp.where(active, walk_chunks, 0)

        def emit(i, woff, key, sel):
            col = cidx_vmem[pl.ds(i * LANES, LANES)]
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)], col, mask=sel)
            # monotone_key is an involution, so this recovers the fp32 bits.
            plsc.store_compressed(
                keybuf_vmem.at[pl.ds(woff, LANES)], _monotone_key(key), mask=sel
            )
            return woff + plsc.all_reduce_population_count(sel)[0]

        def emit_whole(i, woff):
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            return emit(i, woff, key, key > gate)

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, emit_chunks, 0),
            unroll=SCAN_UNROLL,
            carry=jnp.int32(0),
        )(emit_whole)  # pytype: disable=bad-argument-type

        def emit_ranked(i, carry):
            woff, rank = carry
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            in_set = key > lo_excl
            strict = key > hi_incl
            tie_rank = (
                plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
            )
            sel = strict | (in_set & (tie_rank <= my_take))
            return emit(i, woff, key, sel), tie_rank[LANES - 1]

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, 0, emit_chunks),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_ranked)

        def copy_out(i):
            out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]
            keys_vmem[pl.ds(i * LANES, LANES)] = keybuf_vmem[pl.ds(i * LANES, LANES)]

        plsc.parallel_loop(0, k // LANES, unroll=4)(copy_out)

    def pipeline_step(
        scores_vmem, len_slice_vmem, t_vmem, out_vmem, keys_vmem, counts_vmem=None
    ):
        w = pl.program_id(0)
        row = w * cores_per_chip + slot_of(w)
        active = row < b
        lvec = len_slice_vmem[pl.ds(0, LANES)]
        eff_len = jnp.minimum(lvec[0], n)
        eff_len = jnp.where(active, eff_len, 0)
        t_key = t_vmem[pl.ds(0, LANES)][0]
        # key >= T  <=>  key > T - 1, floored so -inf (and the poisoned tail) is
        # never selected even when T is the -inf key.
        gate_excl = jnp.maximum(t_key - 1, neg_inf_key)

        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi)

        @pl.when(i_hi > 0)
        def _():
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

        # With `global_cols` a slot holds the row's column (slice s of a row
        # starts at s * n), which the incumbent adds on the TensorCore.
        # (Default path untouched: no `+ 0` enters its kernel.)
        col_base = jnp.bitwise_and(row, p - 1) * n if global_cols else None

        def thresh_body(i, woff):
            bits = scores_vmem[pl.ds(i * LANES, LANES)]
            sel = _monotone_key(bits) > gate_excl
            col = i * LANES + lane_iota
            if col_base is not None:
                col = col + col_base
            plsc.store_compressed(cidx_vmem.at[pl.ds(woff, LANES)], col, mask=sel)
            plsc.store_compressed(ckey_vmem.at[pl.ds(woff, LANES)], bits, mask=sel)
            return woff + plsc.all_reduce_population_count(sel)[0]

        count = plsc.parallel_loop(
            0, walk_chunks, unroll=SCAN_UNROLL, carry=jnp.int32(0)
        )(thresh_body)  # pytype: disable=bad-argument-type

        overflow = count > k
        if counts_vmem is not None:
            # Slots this slice emits: its survivors, or exactly k after the
            # radix. Everything past them in the slice's k slots is -inf.
            counts_vmem[pl.ds(0, LANES)] = jnp.zeros(
                (LANES,), jnp.int32
            ) + jnp.minimum(count, k)

        def threshold_copy_out():
            def copy_c(i):
                valid = i * LANES + lane_iota < count
                out_vmem[pl.ds(i * LANES, LANES)] = jnp.where(
                    valid, cidx_vmem[pl.ds(i * LANES, LANES)], jnp.int32(-1)
                )
                keys_vmem[pl.ds(i * LANES, LANES)] = jnp.where(
                    valid, ckey_vmem[pl.ds(i * LANES, LANES)], jnp.int32(BITS_NEG_INF)
                )

            plsc.parallel_loop(0, k // LANES, unroll=4)(copy_c)

        if not fallback:
            threshold_copy_out()
            return

        pl.when(jnp.logical_not(overflow))(threshold_copy_out)

        # The slice's local top-k is a subset of its `count > k` survivors, so
        # the radix walks the compacted survivors rather than the whole slice.
        @pl.when(overflow)
        def _():
            walk_s = _cdiv_dyn(count)
            tail_s = (walk_s - 1) * LANES
            # store_compressed leaves stale words past `count`: poison them.
            ckey_vmem[pl.ds(tail_s, LANES)] = jnp.where(
                tail_s + lane_iota < count,
                ckey_vmem[pl.ds(tail_s, LANES)],
                jnp.int32(BITS_NEG_INF),
            )
            survivor_radix(out_vmem, keys_vmem, active, walk_s, count)

    buf_count = 1 if num_waves == 1 else 2

    def imap(w):
        return (w * cores_per_chip + slot_of(w),)

    def tmap(w):
        # Slice s reads T of its row s // p; one row is 128 words = 128 / LANES
        # blocks. Padded slices (s >= b) clamp to the last row; they are empty.
        r = jnp.minimum((w * cores_per_chip + slot_of(w)) // p, t_rows - 1)
        return (r * (128 // LANES),)

    in_specs = (
        pl.BlockSpec((slice_len,), imap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((LANES,), imap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((LANES,), tmap, pipeline_mode=pl.Buffered(buf_count)),
    )
    out_specs = (
        pl.BlockSpec((k,), imap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((k,), imap, pipeline_mode=pl.Buffered(buf_count)),
    )
    outs = (out_hbm, keys_hbm)
    if counts_hbm is not None:
        out_specs += (
            pl.BlockSpec((LANES,), imap, pipeline_mode=pl.Buffered(buf_count)),
        )
        outs += (counts_hbm,)
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_waves,),
        in_specs=in_specs,
        out_specs=out_specs,
    )(scores_hbm, lengths_hbm, t_hbm, *outs)


def _topk_body_l1_counts(
    scores_hbm, lengths_hbm, t_hbm, out_hbm, keys_hbm, counts_hbm, *scratch, **kw
):
    _topk_body_l1(
        scores_hbm, lengths_hbm, t_hbm, out_hbm, keys_hbm, *scratch,
        counts_hbm=counts_hbm, **kw,
    )


def _topk_body_seg_pos(
    scores_hbm, counts_hbm, lengths_hbm, out_hbm, keys_hbm,
    glob_vmem, rowbuf_vmem, keybuf_vmem, pos_vmem, **kw
):
    _topk_body_seg(
        scores_hbm, counts_hbm, lengths_hbm, None, out_hbm, keys_hbm,
        glob_vmem, rowbuf_vmem, keybuf_vmem, pos_vmem, **kw,
    )


def _topk_body_seg_idx(
    scores_hbm, counts_hbm, lengths_hbm, idx_hbm, out_hbm, keys_hbm,
    glob_vmem, rowbuf_vmem, keybuf_vmem, **kw
):
    _topk_body_seg(
        scores_hbm, counts_hbm, lengths_hbm, idx_hbm, out_hbm, keys_hbm,
        glob_vmem, rowbuf_vmem, keybuf_vmem, None, **kw,
    )


def _topk_body_seg(
    scores_hbm,  # i32[padded_b, p * k_p]  stage-1 candidate score bits
    counts_hbm,  # i32[padded_b, p * LANES] lane 0 of block s = live slots of s
    lengths_hbm,  # i32[padded_b, LANES]   lane 0 = candidate row length
    idx_hbm,  # i32[padded_b, p * k_p] stage-1 columns, or None
    out_hbm,  # i32[padded_b, k]
    keys_hbm,  # i32[padded_b, k]
    glob_vmem,  # i32[NUM_BUCKETS + LANES]
    rowbuf_vmem,  # i32[k + LANES]
    keybuf_vmem,  # i32[k + LANES]
    pos_vmem,  # i32[p * k_p + LANES]  candidate slot of each word, or None
    *,
    b: int,
    k: int,
    p: int,
    k_p: int,
    num_waves: int,
    seg_loop: str,
    fault: str,
    idx_buf: int = 1,
):
    """Stage 2 over only the live prefix of each stage-1 segment.

    Segment s of a candidate row holds `cnt_s` candidates then `-inf` pad
    (`_topk_body_l1` compacts). One pass copies those prefixes, in segment
    order, to the front of the row in place and records each word's original
    slot in `pos_vmem`; the incumbent `_topk_body` selection then runs over
    `sum_s cnt_s` words instead of `p * k_p`, and emits `pos` where the
    incumbent emits `i * LANES + lane`. The skipped words are all `-inf`,
    which no pass of the incumbent counts or selects, and order is preserved,
    so slots and keys are bit-identical to `_sc_topk_direct`'s.

    With `idx_hbm` (stage 1's columns, row-global) the pass compacts the
    columns alongside the scores, in place, and the kernel emits the column of
    each winner: the incumbent's `take_along_axis` remap folded in. Same
    values in the same slots as the incumbent's remapped output.

    `fault` is test-only: "floor" walks `cnt_s // LANES` chunks per segment,
    "droplast" one chunk fewer than needed; both lose candidates.
    """
    num_subcores = 16
    num_sc_cores = 2
    cores_per_chip = num_sc_cores * num_subcores
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub
    slice_len = p * k_p

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def key_window(prefix, shift):
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def zero_hist(bound):
        def body(i):
            glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=8)(body)

    def hist_add(bucket, mask):
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)

    def scan_glob(bound):
        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):
        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(
            body
        )  # pytype: disable=bad-argument-type

    with_idx = idx_hbm is not None

    def pipeline_step(scores_vmem, cnt_vmem, len_slice_vmem, *refs):
        if with_idx:
            idx_vmem, out_vmem, keys_vmem = refs
            val_vmem = idx_vmem
        else:
            out_vmem, keys_vmem = refs
            val_vmem = pos_vmem
        w = pl.program_id(0)
        row = w * cores_per_chip + flat
        active = row < b
        eff_len = jnp.minimum(vec_at(len_slice_vmem, 0), slice_len)
        live_row = active & (eff_len > 0)

        def segment(s, woff, s_static=None):
            if s_static is None:
                cnt = cnt_vmem[pl.ds(pl.multiple_of(s * LANES, LANES), LANES)][0]
            else:
                cnt = cnt_vmem[pl.ds(s_static * LANES, LANES)][0]
            cnt = jnp.where(live_row, jnp.minimum(cnt, k_p), 0)
            if fault == "floor":
                chunks = jnp.right_shift(cnt, 4)
            elif fault == "droplast":
                chunks = jnp.maximum(_cdiv_dyn(cnt) - 1, 0)
            else:
                chunks = _cdiv_dyn(cnt)
            base = s * k_p

            # In place: the write window [woff, woff + popcount) never reaches
            # past the chunk just read (woff <= base + c * LANES), so no later
            # chunk's words are overwritten before they are read.
            def comp(c, woff):
                src = base + c * LANES
                bits = scores_vmem[pl.ds(pl.multiple_of(src, LANES), LANES)]
                sel = c * LANES + lane_iota < cnt
                plsc.store_compressed(
                    scores_vmem.at[pl.ds(woff, LANES)], bits, mask=sel
                )
                if with_idx:
                    col = idx_vmem[pl.ds(pl.multiple_of(src, LANES), LANES)]
                else:
                    col = src + lane_iota
                plsc.store_compressed(val_vmem.at[pl.ds(woff, LANES)], col, mask=sel)
                return woff + plsc.all_reduce_population_count(sel)[0]

            return plsc.parallel_loop(0, chunks, unroll=4, carry=woff)(
                comp
            )  # pytype: disable=bad-argument-type

        if seg_loop == "py":
            i_hi = jnp.int32(0)
            for s in range(p):
                i_hi = segment(jnp.int32(s), i_hi, s_static=s)
        else:
            i_hi = lax.fori_loop(0, p, segment, jnp.int32(0))
        walk_chunks = _cdiv_dyn(i_hi)

        zero_hist(NUM_BUCKETS // LANES)

        @pl.when(i_hi > 0)
        def _():
            # Stale words past the compacted prefix: poison them once.
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            def hist0_body(i):
                key = monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
                scores_vmem[pl.ds(i * LANES, LANES)] = key
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
                hist_add(bucket, key > neg_inf_key)

            plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

        total = scan_glob(NUM_BUCKETS // LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0, vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
            zero_hist(hist_bound)
            active_walk = jnp.where(done, 0, walk_chunks)
            lo_excl, hi_incl = key_window(prefix, shift)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                lo_excl=lo_excl,
                hi_incl=hi_incl,
            ):
                key = scores_vmem[pl.ds(i * LANES, LANES)]
                match = (key > lo_excl) & (key <= hi_incl)
                bucket = jnp.bitwise_and(
                    jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
                )
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

            sub_total = scan_glob(hist_bound)
            b2 = find_bucket(hist_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0, vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
            c_above2 = sub_total - cum_at2
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - c_above2

            width = shift - lvl_shift
            new_prefix = jnp.left_shift(prefix, width) | b2
            new_done = done | (quota2 == cnt2) | (quota2 == 0)

            prefix = jnp.where(done, prefix, new_prefix)
            shift = jnp.where(done, shift, jnp.int32(lvl_shift))
            c_hi = jnp.where(done, c_hi, c_hi + c_above2)
            quota = jnp.where(done, quota, quota2)
            cnt = jnp.where(done, cnt, cnt2)
            done = new_done

        my_take = jnp.where(keep_all, 0, quota)

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
            keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                (LANES,), BITS_NEG_INF, jnp.int32
            )

        fill_bound = jnp.where(total < k, k // LANES, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        lo_excl, hi_incl = key_window(prefix, shift)
        take_all_ties = keep_all | (my_take >= cnt)
        gate = jnp.where(take_all_ties, lo_excl, hi_incl)
        whole_bucket = take_all_ties | (my_take <= 0)
        emit_chunks = jnp.where(active, walk_chunks, 0)

        def emit(i, woff, key, sel):
            plsc.store_compressed(
                rowbuf_vmem.at[pl.ds(woff, LANES)],
                val_vmem[pl.ds(i * LANES, LANES)],
                mask=sel,
            )
            plsc.store_compressed(
                keybuf_vmem.at[pl.ds(woff, LANES)],
                monotone_key(key),
                mask=sel,
            )
            return woff + plsc.all_reduce_population_count(sel)[0]

        def emit_whole(i, woff):
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            return emit(i, woff, key, key > gate)

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, emit_chunks, 0),
            unroll=SCAN_UNROLL,
            carry=jnp.int32(0),
        )(emit_whole)  # pytype: disable=bad-argument-type

        def emit_ranked(i, carry):
            woff, rank = carry
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            in_set = key > lo_excl
            strict = key > hi_incl
            tie_rank = (
                plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
            )
            sel = strict | (in_set & (tie_rank <= my_take))
            return emit(i, woff, key, sel), tie_rank[LANES - 1]

        plsc.parallel_loop(
            0,
            jnp.where(whole_bucket, 0, emit_chunks),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_ranked)

        def copy_out(i):
            out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]
            keys_vmem[pl.ds(i * LANES, LANES)] = keybuf_vmem[pl.ds(i * LANES, LANES)]

        plsc.parallel_loop(0, k // LANES, unroll=4)(copy_out)

    buf_count = 1 if num_waves == 1 else 2

    def rmap(w):
        return (w * cores_per_chip + flat,)

    in_specs = (
        pl.BlockSpec((slice_len,), rmap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((p * LANES,), rmap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((LANES,), rmap, pipeline_mode=pl.Buffered(buf_count)),
    )
    ins = (scores_hbm, counts_hbm, lengths_hbm)
    if with_idx:
        # Single-buffered: two double-buffered 32K-word rows would not leave
        # room for the rest in 128K words of subcore VMEM.
        in_specs += (
            pl.BlockSpec((slice_len,), rmap, pipeline_mode=pl.Buffered(idx_buf)),
        )
        ins += (idx_hbm,)
    out_specs = (
        pl.BlockSpec((k,), rmap, pipeline_mode=pl.Buffered(buf_count)),
        pl.BlockSpec((k,), rmap, pipeline_mode=pl.Buffered(buf_count)),
    )
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_waves,),
        in_specs=in_specs,
        out_specs=out_specs,
    )(*ins, out_hbm, keys_hbm)


def _sc_topk_stage2_seg(
    cand_scores: jax.Array,  # i32/f32 [b, p * k_p]
    counts: jax.Array,  # i32[>= b * p * LANES] from stage 1 (emit_counts)
    k: int,
    cand_lengths: jax.Array,  # i32[b]
    p: int,
    k_p: int,
    *,
    seg_loop: str = "fori",
    fault: str = "",
    scheduling_group_id: int | None = None,
    cand_idx: jax.Array | None = None,  # i32[b, p * k_p] row-global columns
    idx_buf: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """`_sc_topk_direct` over stage-1 candidates, walking only live prefixes.

    Returns (candidate slots, score bits); with `cand_idx`, (columns, score
    bits), i.e. the remap is done in the kernel.
    """
    b, n = cand_scores.shape
    if n != p * k_p or k_p % LANES:
        raise ValueError(f"{cand_scores.shape=} is not {p} x {k_p}")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    words = (
        cand_scores
        if cand_scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(cand_scores, jnp.int32)
    )
    num_waves = _cdiv(b, 32)
    padded_b = num_waves * 32
    pad_b = padded_b - b
    if pad_b > 0:
        words = jnp.pad(words, ((0, pad_b), (0, 0)))
    counts = counts[: b * p * LANES]
    if pad_b > 0:
        counts = jnp.pad(counts, (0, pad_b * p * LANES))
    lengths = jnp.pad(cand_lengths.astype(jnp.int32), (0, pad_b))
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1)))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
    )
    ins = (words.reshape(-1), counts, lengths_2d.reshape(-1))
    if cand_idx is None:
        scratch_types += (pltpu.VMEM((n + LANES,), jnp.int32),)
        body, extra, tag = _topk_body_seg_pos, {}, ""
    else:
        idx = cand_idx.astype(jnp.int32)
        if pad_b > 0:
            idx = jnp.pad(idx, ((0, pad_b), (0, 0)), constant_values=-1)
        ins += (idx.reshape(-1),)
        body, extra, tag = _topk_body_seg_idx, dict(idx_buf=idx_buf), "_idx"
    out = pl.kernel(
        functools.partial(
            body,
            **extra,
            b=b,
            k=k,
            p=p,
            k_p=k_p,
            num_waves=num_waves,
            seg_loop=seg_loop,
            fault=fault,
        ),
        out_type=(slot_type, slot_type),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"sc_topk_seg{tag}_b{b}_n{n}_k{k}{('_' + fault) if fault else ''}",
    )(*ins)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    slots, keys = out
    return slots.reshape(padded_b, k)[:b], keys.reshape(padded_b, k)[:b]


def _sc_topk_stage1_l1(
    scores: jax.Array,  # f32/i32 [b, n] (slices)
    k: int,
    row_lengths: jax.Array,  # i32[b]
    t_rows: jax.Array,  # i32[b // p, 128]  lane 0 = monotone-key lower bound
    p: int,  # slices per T row
    *,
    fallback: bool = True,
    scheduling_group_id: int | None = None,
    emit_counts: bool = False,
    global_cols: bool = False,
):
    """Stage 1': threshold compaction with per-slice radix fallback.

    `global_cols` writes each slot's row column (`(s % p) * n + local`)
    instead of the slice-local column; `-1` pad is unchanged.

    `emit_counts` adds a third output, i32[b]: the number of leading slots of
    each slice that hold a candidate (`min(survivors, k)`); every later slot is
    `-1` / `-inf`. The segmented stage 2 walks only those.

    Slice s is gated by `t_rows[s // p, 0]`, read straight from the array the
    scoring epilogue (or the T kernel) wrote, so no XLA op sits between the
    T producer and this kernel.

    Same output format as `_sc_topk_direct`: slots i32[b, k], `-1` suffix
    padded, and each slot's fp32 bits as i32, `-inf` bits in the pad. A slice's
    slots are a superset of its winners, not necessarily its local top-k.
    """
    b, n = scores.shape
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    if sc.num_cores * sc.num_subcores != 32:
        raise NotImplementedError(
            f"stage 1' assumes 32 subcores, got {sc.num_cores}x{sc.num_subcores}"
        )
    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)
    padded_b = num_waves * 32
    pad_b = padded_b - b
    pad_n = slice_len - n
    if pad_b > 0 or pad_n > 0:
        words = jnp.pad(words, ((0, pad_b), (0, pad_n)))
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, pad_b))
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1)))
    if t_rows.shape != (b // p, 128) or b % p:
        raise ValueError(f"t_rows {t_rows.shape} does not cover {b} slices / {p}")
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    out_type = (slot_type, slot_type)
    if emit_counts:
        out_type += (jax.ShapeDtypeStruct((padded_b * LANES,), jnp.int32),)
    scratch_types = (
        pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((k + LANES,), jnp.int32),
        pltpu.VMEM((slice_len + LANES,), jnp.int32),
        pltpu.VMEM((slice_len + LANES,), jnp.int32),
    )
    body = _topk_body_l1 if not emit_counts else _topk_body_l1_counts
    out = pl.kernel(
        functools.partial(
            body,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            fallback=fallback,
            p=p,
            t_rows=b // p,
            **(dict(global_cols=True) if global_cols else {}),
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_topk_l1_b{b}_n{n}_k{k}{'' if fallback else '_nofb'}"
            f"{'_cnt' if emit_counts else ''}{'_gcol' if global_cols else ''}"
        ),
    )(words.reshape(-1), lengths_2d.reshape(-1), t_rows.astype(jnp.int32).reshape(-1))
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    if emit_counts:
        slots, keys, counts = out
        return (
            slots.reshape(padded_b, k)[:b],
            keys.reshape(padded_b, k)[:b],
            counts,  # i32[padded_b * LANES], lane 0 of each LANES block
        )
    slots, keys = out
    return slots.reshape(padded_b, k)[:b], keys.reshape(padded_b, k)[:b]


def _sparsecore_topk_l1_impl(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    write_empty_rows: bool,
    scheduling_group_id: int | None,
    stage2_scheduling_group_id: int | None,
    threshold_scheduling_group_id: int | None,
    t_rows: jax.Array | None = None,
    t_keys: jax.Array | None = None,
    fallback: bool = True,
    topk_tail: str = "",
    seg_loop: str = "fori",
    seg_fault: str = "",
) -> tuple[jax.Array, jax.Array]:
    """The threshold path.

    `t_rows` (i32[b, 128], lane 0 = T) is T from the scoring epilogue; None
    runs the TensorCore T kernel here. `t_keys` (i32[b]) / `fallback` are
    test-only overrides. `topk_tail="seg"` has stage 1 emit per-slice
    candidate counts and runs stage 2 over only those (`_sc_topk_stage2_seg`);
    "" is the incumbent stage 2. `seg_loop` / `seg_fault` are test-only.
    """
    if topk_tail not in ("", "seg", "segc"):
        raise ValueError(f"unknown {topk_tail=}")
    b, n = scores.shape
    p = _pick_partition(b, n, k)
    n_p = n // p
    k_p = min(k, n_p)
    if t_keys is not None:
        t_rows = jnp.broadcast_to(t_keys.astype(jnp.int32)[:, None], (b, 128))
    elif t_rows is None:
        t_rows = row_threshold_rows(
            scores,
            k,
            row_lengths,
            scheduling_group_id=threshold_scheduling_group_id,
        )
        if b % 64 != 0:
            # Without this barrier XLA's memory-space assignment fails to
            # compile ("Too many buffers are colored in the alternate memory")
            # for i32 scores at b=32/40, n=264192 (b=8..128 otherwise measured
            # fine). It serialises T against the scores producer, which costs
            # ~0.7 ms per chunked prefill call at b=128, so the 128-row chunk
            # shape skips it. T from the scoring epilogue needs neither.
            scores, t_rows = jax.lax.optimization_barrier((scores, t_rows))
    scores_p = scores.reshape(b * p, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0, n_p).reshape(b * p)

    seg = topk_tail in ("seg", "segc")
    fold = topk_tail == "segc"
    if fold and p & (p - 1):
        raise ValueError(f"segc needs a power-of-two partition, got {p=}")
    stage1 = _sc_topk_stage1_l1(
        scores_p,
        k_p,
        lengths_p,
        t_rows,
        p,
        fallback=fallback,
        scheduling_group_id=scheduling_group_id,
        emit_counts=seg,
        global_cols=fold,
    )
    if seg:
        local_slots, cand_scores, counts = stage1
    else:
        local_slots, cand_scores = stage1
    if fold:
        # Stage 1 wrote row columns; stage 2 compacts and emits them.
        return _sc_topk_stage2_seg(
            cand_scores.reshape(b, p * k_p),
            counts,
            k,
            jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0)),
            p,
            k_p,
            seg_loop=seg_loop,
            fault=seg_fault,
            scheduling_group_id=stage2_scheduling_group_id,
            cand_idx=local_slots.reshape(b, p * k_p),
        )
    local_indices = local_slots.reshape(b, p, k_p)
    cand_scores = cand_scores.reshape(b, p * k_p)
    global_cand_indices = jnp.where(
        local_indices < 0,
        -1,
        offsets[None, :, None] + local_indices,
    ).reshape(b, p * k_p)

    # Stage 2 and the remap: the incumbent's, unchanged.
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))
    if seg:
        final_cand_slots, final_cand_scores = _sc_topk_stage2_seg(
            cand_scores,
            counts,
            k,
            cand_lengths,
            p,
            k_p,
            seg_loop=seg_loop,
            fault=seg_fault,
            scheduling_group_id=stage2_scheduling_group_id,
        )
    else:
        final_cand_slots, final_cand_scores = _sc_topk_direct(
            cand_scores,
            k,
            cand_lengths,
            write_empty=write_empty_rows,
            scheduling_group_id=stage2_scheduling_group_id,
        )
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots, axis=1)
    final_indices = jnp.where(final_cand_slots >= 0, final_indices, -1)
    return final_indices, final_cand_scores


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "write_empty_rows",
        "scheduling_group_id",
        "stage2_scheduling_group_id",
        "threshold_scheduling_group_id",
        "return_scores",
        "topk_tail",
    ),
)
def sparsecore_topk_l1(
    scores: jax.Array,  # f32[b, n] or i32[b, n] of f32 bits
    k: int,
    row_lengths: jax.Array | None = None,  # i32[b], defaults to n
    *,
    write_empty_rows: bool = True,
    scheduling_group_id: int | None = None,
    stage2_scheduling_group_id: int | None = None,
    threshold_scheduling_group_id: int | None = None,
    return_scores: bool = False,
    t_rows: jax.Array | None = None,  # i32[b, 128], lane 0 = T key
    topk_tail: str = "",
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """`sparsecore_topk` with the threshold-gated stage 1.

    Same arguments, outputs and contract as `sparsecore_topk`, plus
    `threshold_scheduling_group_id`, the XLA scheduling group of the
    TensorCore T kernel, and `t_rows`, a precomputed T per row (the
    StreamIdxTC scoring epilogue's) that replaces the T kernel. Any
    `t_rows[r, 0] <= v_k(r)` keeps the result exact. Shapes outside
    `l1_supported` run `sparsecore_topk` and ignore `t_rows`.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    b, n = scores.shape
    if not l1_supported(b, n, k):
        return sparsecore_topk(
            scores,
            k,
            row_lengths,
            write_empty_rows=write_empty_rows,
            scheduling_group_id=scheduling_group_id,
            stage2_scheduling_group_id=stage2_scheduling_group_id,
            return_scores=return_scores,
        )
    if scores.dtype not in (jnp.float32, jnp.int32):
        raise ValueError(f"scores must be f32 or i32, got {scores.dtype}")
    if row_lengths is None:
        row_lengths = jnp.full((b,), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    idx, cand_scores = _sparsecore_topk_l1_impl(
        scores,
        k,
        row_lengths,
        write_empty_rows=write_empty_rows,
        scheduling_group_id=scheduling_group_id,
        stage2_scheduling_group_id=stage2_scheduling_group_id,
        threshold_scheduling_group_id=threshold_scheduling_group_id,
        t_rows=t_rows,
        topk_tail=topk_tail,
    )
    if return_scores:
        return idx, cand_scores
    return idx
