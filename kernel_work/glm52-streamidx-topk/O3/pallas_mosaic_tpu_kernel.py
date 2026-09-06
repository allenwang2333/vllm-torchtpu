"""O3: the SparseCore top-k kernel, run at a `p` chosen from the batch's live extent."""

# Dimensions.
#   B          rows of the score matrix = padded token count
#   N          padded score extent per row; also the HBM stride between rows
#   K          top-k requested
#   tn         slice width in words -- the tile extent of N one subcore holds
#              resident. Fixed at N // P_MAX and the SAME on every ladder rung.
#   p          N-tiles covered this call. code-style spells a grid extent `nn`;
#              the exercise calls it `p` everywhere, so it stays `p`.
#   n_sl       slices in flight = B * p
#   nw         waves = cdiv(n_sl, slices_per_wave)
#   kp         per-slice candidate count in stage 1 = min(K, tn)
#
# WHAT O3 CHANGES, in one line: the incumbent partitions every row into P_MAX
# slices because static N says it must (sparsecore_topk.py:360-377), and at
# 256k live in a 1M-wide buffer twelve of every sixteen slices are dead. O3
# covers only `p * tn >= batch-max live length` columns, bucketed UPWARD over a
# precompiled ladder, so `n_sl` falls, `nw` falls, and M2's
# `X_crit = sum_waves max_row_walked` falls with them. Rows are NOT shortened:
# P-WALK level B (j-69ee326d) measured that shortening rows at fixed p is 20.6%
# SLOWER at b=16. The saving here is a packing saving, not a pruning one.
#
# code-style.md 4 asks for jaxtyping on the public wrapper. jaxtyping is not
# installed in the target image, so shapes live in trailing comments and are
# asserted at the boundary. Declared, not silently dropped.

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

import pallas_mosaic_tpu as policy
import tiling

# Algorithm constants, not hardware ones: the radix is 8 bits x 4 levels over a
# monotone i32 key. These stay literal; the chip does not have an opinion.
NUM_BUCKETS = 256
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
KEY_NEG_INF = int(
    np.int32(np.array(-np.inf, np.float32).view(np.int32))
    ^ np.int32(0x7FFFFFFF))


def _topk_kernel(
    # Inputs.
    scores_hbm_ref,   # [B * N] i32          full array, fp32 score bits
    lengths_hbm_ref,  # [n_sl_pad] i32       full array, per-SLICE live length
    # Outputs.
    out_hbm_ref,      # [n_sl * K] i32       full array
    # Scratch.
    resident_vmem,    # [tn] i32             one slice, DMA'd from scores_hbm_ref
    priv_vmem,        # [lanes * 256 + lanes] i32   lane-private histograms
    glob_vmem,        # [256 + lanes] i32    folded histogram, then its prefix sum
    rowbuf_vmem,      # [K + lanes] i32      winner indices for one slice
    len_vmem,         # [n_sl_pad] i32       lengths, resident
    sems,             # DMA[4]
    *,
    n_sl: int,
    tn: int,
    k: int,
    p: int,
    row_stride: int,
    nw: int,
    lanes: int,
    slices_per_wave: int,
    num_subcores: int,
    write_empty: bool,
):
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(lanes, dtype=jnp.int32)
    n_sl_pad = len_vmem.shape[0]

    # p is a static power of two, so row/tile decomposition is a shift and a
    # mask rather than a division.
    log2p = int(p).bit_length() - 1
    part_mask = jnp.int32(p - 1)

    cp = pltpu.make_async_copy(lengths_hbm_ref.at[pl.ds(0, n_sl_pad)], len_vmem,
                               sems.at[0])
    cp.start()
    cp.wait()

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, lanes)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def cdiv_dyn(x):
        return jnp.right_shift(x + (lanes - 1), int(lanes).bit_length() - 1)

    def zero_priv(bound):

        def body(i):
            priv_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes, ), jnp.int32)

        plsc.parallel_loop(0, bound, unroll=4)(body)

    def fold_priv(bound):

        def body(i):
            off = i * lanes
            acc = jnp.zeros((lanes, ), jnp.int32)
            for m in range(lanes):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, lanes)]
            glob_vmem[pl.ds(off, lanes)] = acc

        plsc.parallel_loop(0, bound, unroll=2)(body)

    def scan_glob(bound):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * lanes, lanes)]) + carry
            glob_vmem[pl.ds(i * lanes, lanes)] = c
            return c[lanes - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * lanes, lanes)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def wave_body(w, _):
        slot = w * slices_per_wave + flat
        active = slot < n_sl
        slot_c = jnp.minimum(slot, n_sl - 1)
        eff_len = jnp.minimum(vec_at(len_vmem, slot_c), tn)
        eff_len = jnp.where(active, eff_len, 0)

        # THE O3 LINE. The incumbent reads slice `slot_c` at `slot_c * tn`,
        # which is only right when the slices tile a row exactly -- i.e. when
        # p == P_MAX. Here p <= P_MAX, so a slice is tile (slot_c % p) of row
        # (slot_c // p) and the row stride N has to be applied explicitly.
        # At p == P_MAX this is arithmetically identical to `slot_c * tn`,
        # which is why the p == P_MAX rung must reproduce the incumbent bit for
        # bit.
        row_id = jnp.right_shift(slot_c, log2p)
        part_id = jnp.bitwise_and(slot_c, part_mask)
        s = row_id * row_stride + part_id * tn

        i_hi = jnp.where(eff_len > 0, jnp.minimum(tn, eff_len), 0)
        walk_chunks = cdiv_dyn(i_hi)

        @pl.when(i_hi > 0)
        def _():
            dma(scores_hbm_ref.at[pl.ds(s, tn)], resident_vmem, 1)

            def cvt(i):
                bits = resident_vmem[pl.ds(i * lanes, lanes)]
                resident_vmem[pl.ds(i * lanes, lanes)] = monotone_key(bits)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)

        def load_keys(i):
            key = resident_vmem[pl.ds(i * lanes, lanes)]
            j = i * lanes + lane_iota
            valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))
            return key, valid, j

        @pl.when(i_hi > 0)
        def _():
            zero_priv(NUM_BUCKETS)

            def hist0_body(i):
                key, valid, _ = load_keys(i)
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                    NUM_BUCKETS // 2)
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((lanes, ), jnp.int32),
                                       mask=valid)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            fold_priv(NUM_BUCKETS // lanes)

        @pl.when(i_hi == 0)
        def _():

            def zg(i):
                glob_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes, ),
                                                               jnp.int32)

            plsc.parallel_loop(0, NUM_BUCKETS // lanes, unroll=4)(zg)

        total = scan_glob(NUM_BUCKETS // lanes)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // lanes, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0,
                           vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            zero_priv(jnp.where(done, 0, NUM_BUCKETS))
            active_walk = jnp.where(done, 0, walk_chunks)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                prefix=prefix,
                shift=shift,
            ):
                key, valid, _ = load_keys(i)
                match = valid & (jnp.right_shift(key, shift) == prefix)
                bucket = jnp.bitwise_and(jnp.right_shift(key, lvl_shift),
                                         jnp.int32(lvl_mask))
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((lanes, ), jnp.int32),
                                       mask=match)

            plsc.parallel_loop(0, active_walk, unroll=4)(ref_body)
            fold_bound = jnp.where(done, 0, NUM_BUCKETS // lanes)
            fold_priv(fold_bound)

            sub_total = scan_glob(fold_bound)
            b2 = find_bucket(fold_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0,
                                vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
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

        out_gate = active if write_empty else (active & (i_hi > 0))

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * lanes, lanes)] = jnp.full((lanes, ), -1,
                                                            jnp.int32)

        fill_bound = jnp.where(out_gate & (total < k), k // lanes, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        def emit_body(i, carry):
            woff, rank = carry
            key, valid, j = load_keys(i)
            hi = jnp.right_shift(key, shift)
            strict = valid & (hi > prefix)
            tie = valid & (hi == prefix)
            tie_rank = plsc.cumsum(tie.astype(jnp.int32)) + rank
            take = tie & (tie_rank <= my_take)
            sel = jnp.where(keep_all, valid, strict | take)
            pos = j
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)],
                                  pos,
                                  mask=sel)
            woff = woff + plsc.all_reduce_population_count(sel)[0]
            rank = tie_rank[lanes - 1]
            return woff, rank

        n_win, _ = plsc.parallel_loop(
            0,
            jnp.where(active, walk_chunks, 0),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_body)

        @pl.when(out_gate)
        def _():
            dma(
                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm_ref.at[pl.ds(slot_c * k, k)],
                2,
            )

        return 0

    lax.fori_loop(0, nw, wave_body, 0)


def _sc_topk_slices(
    words_flat: jax.Array,  # [B * N] i32, the f32 scores bitcast and flattened
    B: int,
    N: int,
    k: int,
    lengths: jax.Array,  # [B * p] i32   per-SLICE, already clipped to tn
    *,
    p: int,
    tn: int,
    write_empty: bool,
) -> jax.Array:  # [B * p, k] i32
    """Run one ladder rung: B*p slices of `tn` words each, over the subcore mesh.

    `p == 1` and `tn == N` is the incumbent's `_sc_topk_direct`; every other
    (p, tn) covers only the first `p * tn` columns of each row, which is legal
    exactly when the caller bucketed upward.

    Takes the ALREADY flat int32 view rather than the [B, N] f32 array. The
    bitcast and the reshape are free on their own, but performed inside a
    `lax.switch` branch they are not: iteration 2 (j-fad1a0f8) measured the
    conditional materialising a whole copy of the operand at 1.30 TB/s
    effective -- 51 us at 64 MiB and 3310 us at 4 GiB, the same rate at both
    sizes. Hoisting them above the switch is the fix, and it only works if
    nothing downstream needs the [B, N] f32 view.
    """
    lanes, slices_per_wave, _ = policy.hardware()
    sc = pltpu.get_tpu_info().sparse_core

    if tn % lanes:
        raise ValueError(f"{tn=} must be a multiple of {lanes}")
    if p * tn > N:
        raise ValueError(f"p*tn={p * tn} overruns the row extent {N}")
    if words_flat.dtype != jnp.int32:
        raise ValueError(f"words_flat must be i32, got {words_flat.dtype}")

    n_sl = B * p
    n_sl_pad = tiling.align_to(n_sl, lanes)
    lengths = jnp.pad(lengths.astype(jnp.int32), (0, n_sl_pad - n_sl))
    nw = tiling.num_waves(n_sl, slices_per_wave)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = pl.kernel(
        functools.partial(
            _topk_kernel,
            n_sl=n_sl,
            tn=tn,
            k=k,
            p=p,
            row_stride=N,
            nw=nw,
            lanes=lanes,
            slices_per_wave=slices_per_wave,
            num_subcores=sc.num_subcores,
            write_empty=write_empty,
        ),
        out_type=jax.ShapeDtypeStruct((n_sl * k, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((tn, ), jnp.int32),
            pltpu.VMEM((lanes * NUM_BUCKETS + lanes, ), jnp.int32),
            pltpu.VMEM((NUM_BUCKETS + lanes, ), jnp.int32),
            pltpu.VMEM((k + lanes, ), jnp.int32),
            pltpu.VMEM((n_sl_pad, ), jnp.int32),
            pltpu.SemaphoreType.DMA((4, )),
        ),
        mesh=mesh,
        name=f"o3_sc_topk_b{B}_p{p}_tn{tn}_k{k}",
    )(words_flat, lengths)
    return out.reshape(n_sl, k)


def _select_at_p(
    words_flat: jax.Array,  # [B * N] i32
    B: int,
    N: int,
    k: int,
    row_lengths: jax.Array,  # [B] i32
    *,
    p: int,
    tn: int,
    write_empty_rows: bool,
) -> jax.Array:  # [B, k] i32
    """One precompiled `lax.switch` branch: the whole selection at a fixed `p`.

    Everything here consumes the flat int32 view. Nothing in the branch may
    reshape or bitcast `words_flat` -- doing so is what iteration 2 priced at
    a full copy of the array, see `_sc_topk_slices`.
    """
    if p == 1:
        # No stage 2 at all -- and this is the largest single step on the
        # ladder, not a special case. Dropping to p == 1 deletes a pl.kernel
        # launch and both gathers, on top of the wave saving.
        return _sc_topk_slices(words_flat, B, N, k,
                               jnp.minimum(row_lengths, tn), p=1, tn=tn,
                               write_empty=write_empty_rows)

    kp = min(k, tn)
    offsets = jnp.arange(p, dtype=jnp.int32) * tn
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         tn).reshape(B * p)

    local_indices = _sc_topk_slices(words_flat, B, N, kp, lengths_p, p=p,
                                    tn=tn,
                                    write_empty=True).reshape(B, p, kp)

    # Local slot -> global column.
    global_cand_indices = jnp.where(
        local_indices < 0,
        -1,
        offsets[None, :, None] + local_indices,
    ).reshape(B, p * kp)

    # Gather the candidate scores out of the FLAT array with flat indices, so
    # the [B, N] f32 view is never rematerialised inside the branch.
    safe_cand_indices = jnp.maximum(global_cand_indices, 0)
    row_base = (jnp.arange(B, dtype=jnp.int32) * jnp.int32(N))[:, None]
    cand_words = jnp.take(words_flat,
                          (row_base + safe_cand_indices).reshape(-1),
                          mode="clip").reshape(B, p * kp)
    cand_scores = jax.lax.bitcast_convert_type(cand_words, jnp.float32)
    cand_scores = jnp.where(global_cand_indices >= 0, cand_scores, -jnp.inf)

    # Stage 2 over p*kp candidates -- and with p chosen from the live extent
    # there are no dead slices in that set. P-WALK level B (j-69ee326d)
    # measured the alternative: at p=16 with 256k live, stage 2 receives 32768
    # candidates most of which are -1/-inf padding, and the radix select runs
    # against a distribution with a huge atom. That arm was 20.6% SLOWER than
    # full-length rows at the same p.
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * kp), jnp.int32(0))
    # This bitcast+reshape IS inside the branch, and that is fine: the array
    # is [B, p*kp], 128x smaller than the scores it came from.
    cand_words_masked = jax.lax.bitcast_convert_type(cand_scores, jnp.int32)
    final_cand_slots = _sc_topk_slices(cand_words_masked.reshape(-1), B,
                                       p * kp, k, cand_lengths, p=1,
                                       tn=p * kp,
                                       write_empty=write_empty_rows)

    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


@functools.partial(jax.jit,
                   static_argnames=("k", "write_empty_rows", "hoist"))
def sparsecore_topk(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32, defaults to N
    *,
    write_empty_rows: bool = True,
    hoist: bool = True,
) -> jax.Array:  # [B, k] i32
    """Exact top-k indices per row, unsorted, -1 suffix-padded. O3's entry point.

    Same contract as the incumbent, and deliberately the same selected set: the
    partition decides HOW the exact top-k is found, not WHAT it is. Ties at the
    cut may break differently, which the instrument's churn tolerance covers
    and the incumbent's own docstring already permits.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    B, N = scores.shape
    policy.supported_on(B, N, k)
    cfg = policy._get_heuristics_config(B, N, k)

    if row_lengths is None:
        row_lengths = jnp.full((B, ), N, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    if B * N >= 2**31:
        raise NotImplementedError(
            f"{B}*{N} flat elements overruns the int32 gather index used for "
            f"the candidate scores")
    words_flat = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)

    ladder = cfg.ladder
    if len(ladder) == 1:
        # p_max == 1: there is no choice to make and no switch to pay for.
        # This is regime (A), and it is why O3 is worth exactly zero there.
        return _select_at_p(words_flat, B, N, k, row_lengths, p=1, tn=cfg.tn,
                            write_empty_rows=write_empty_rows)

    # Bucket UPWARD over the ladder. `coverage` is what each rung below the top
    # can reach; the index is the count of rungs that fall short, so the rung
    # selected is the smallest one that covers the batch's longest live row.
    # Downward would truncate a live candidate set silently -- see
    # tiling.ladder_index for why that direction is the dangerous one.
    live_max = jnp.max(row_lengths)
    coverage = jnp.asarray([p * cfg.tn for p in ladder[:-1]], jnp.int32)
    rung = jnp.sum((coverage < live_max).astype(jnp.int32))

    # k is static and stays closed over: passing it as a switch operand would
    # trace it and `min(k, tn)` would stop being a Python int.
    if hoist:
        # The shipping form. `words_flat` crosses the conditional boundary
        # already in the shape `pl.kernel` wants, so no branch reshapes it.
        def branch(p):

            def go(words_, row_lengths_):
                return _select_at_p(words_, B, N, k, row_lengths_, p=p,
                                    tn=cfg.tn,
                                    write_empty_rows=write_empty_rows)

            return go

        return lax.switch(rung, tuple(branch(p) for p in ladder), words_flat,
                          row_lengths)

    # hoist=False reproduces iteration 1 exactly: the bitcast and the reshape
    # happen inside each branch. Kept ONLY so the two can be timed as adjacent
    # arms in one job; it is not a shipping option.
    def branch_nohoist(p):

        def go(scores_, row_lengths_):
            w_ = jax.lax.bitcast_convert_type(scores_, jnp.int32).reshape(-1)
            return _select_at_p(w_, B, N, k, row_lengths_, p=p, tn=cfg.tn,
                                write_empty_rows=write_empty_rows)

        return go

    return lax.switch(rung, tuple(branch_nohoist(p) for p in ladder), scores,
                      row_lengths)
