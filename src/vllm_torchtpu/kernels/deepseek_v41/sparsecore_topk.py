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
"""SparseCore top-k and two-level candidate top-k kernels for DeepSeek-V4.1."""

import functools
import math

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
    MONO_MASK,
    NUM_BUCKETS,
    REFINEMENT_LEVELS,
    SCAN_UNROLL,
    _align_to,
    _cdiv,
    _cdiv_dyn,
    _pick_partition,
)


def _get_sc_subcores() -> int:
    sc = pltpu.get_tpu_info().sparse_core
    return 32 if sc is None else sc.num_cores * sc.num_subcores


def _as_i32_bits(scores: jax.Array) -> jax.Array:
    return (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )


def _vec_at(ref, idx):
    return ref[pl.ds(idx, LANES)][0]


def _monotone_key(bits):
    m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
    return jnp.bitwise_xor(bits, m)


def _key_window(prefix, shift):
    neg_inf_key = jnp.int32(KEY_NEG_INF)
    lo = jnp.left_shift(prefix, shift)
    hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
    return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)


def _zero_hist(glob_vmem, bound):
    def body(i):
        glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

    plsc.parallel_loop(0, bound, unroll=16)(body)


def _hist_add(glob_vmem, bucket, mask):
    dup, last = plsc.scan_count(bucket, mask=mask)
    plsc.addupdate_scatter(glob_vmem, (bucket,), dup, mask=last)


def _scan_glob(glob_vmem, bound):
    def body(i, carry):
        c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
        glob_vmem[pl.ds(i * LANES, LANES)] = c
        return c[LANES - 1]

    return plsc.parallel_loop(0, bound, unroll=16, carry=jnp.int32(0))(body)


def _find_bucket(glob_vmem, bound, thresh):
    def body(i, cnt):
        below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
        return cnt + plsc.all_reduce_population_count(below)[0]

    return plsc.parallel_loop(0, bound, unroll=16, carry=jnp.int32(0))(body)


def _radix_select_and_emit(
    scores_vmem,
    glob_vmem,
    walk_chunks,
    target_k: int,
    *,
    fill_fn,
    fill_unroll: int,
    emit_fn,
):
    """4-level 8-bit radix select over VMEM followed by compressed emission."""
    neg_inf_key = jnp.int32(KEY_NEG_INF)

    def hist0_body(i):
        key = _monotone_key(scores_vmem[pl.ds(i * LANES, LANES)])
        scores_vmem[pl.ds(i * LANES, LANES)] = key
        bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(NUM_BUCKETS // 2)
        _hist_add(glob_vmem, bucket, key > neg_inf_key)

    plsc.parallel_loop(0, walk_chunks, unroll=SCAN_UNROLL)(hist0_body)

    total = _scan_glob(glob_vmem, NUM_BUCKETS // LANES)
    keep_all = total <= target_k

    b_star = _find_bucket(glob_vmem, NUM_BUCKETS // LANES, total - target_k)
    cum_at = _vec_at(glob_vmem, b_star)
    cum_lo = jnp.where(b_star > 0, _vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
    c_hi = total - cum_at
    cnt = cum_at - cum_lo
    quota = target_k - c_hi

    prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
    shift = jnp.int32(L0_SHIFT)
    done = keep_all | (quota == cnt)

    for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
        hist_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
        _zero_hist(glob_vmem, hist_bound)
        active_walk = jnp.where(done, 0, walk_chunks)
        lo_excl, hi_incl = _key_window(prefix, shift)

        def ref_body(
            i, lvl_shift=lvl_shift, lvl_mask=lvl_mask, lo_excl=lo_excl, hi_incl=hi_incl
        ):
            key = scores_vmem[pl.ds(i * LANES, LANES)]
            match = (key > lo_excl) & (key <= hi_incl)
            bucket = jnp.bitwise_and(
                jnp.right_shift(key, lvl_shift), jnp.int32(lvl_mask)
            )
            _hist_add(glob_vmem, bucket, match)

        plsc.parallel_loop(0, active_walk, unroll=SCAN_UNROLL)(ref_body)

        sub_total = _scan_glob(glob_vmem, hist_bound)
        b2 = _find_bucket(glob_vmem, hist_bound, sub_total - quota)
        cum_at2 = _vec_at(glob_vmem, b2)
        cum_lo2 = jnp.where(b2 > 0, _vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
        c_above2 = sub_total - cum_at2
        cnt2 = cum_at2 - cum_lo2
        quota2 = quota - c_above2

        new_prefix = jnp.left_shift(prefix, shift - lvl_shift) | b2
        new_done = done | (quota2 == cnt2) | (quota2 == 0)

        prefix = jnp.where(done, prefix, new_prefix)
        shift = jnp.where(done, shift, jnp.int32(lvl_shift))
        quota = jnp.where(done, quota, quota2)
        cnt = jnp.where(done, cnt, cnt2)
        done = new_done

    my_take = jnp.where(keep_all, 0, quota)
    fill_bound = jnp.where(total < target_k, target_k // LANES, 0)
    plsc.parallel_loop(0, fill_bound, unroll=fill_unroll)(fill_fn)

    lo_excl, hi_incl = _key_window(prefix, shift)
    take_all_ties = keep_all | (my_take >= cnt)
    gate = jnp.where(take_all_ties, lo_excl, hi_incl)
    whole_bucket = take_all_ties | (my_take <= 0)

    def emit_whole(i, woff):
        key = scores_vmem[pl.ds(i * LANES, LANES)]
        return emit_fn(i, woff, key, key > gate)

    plsc.parallel_loop(
        0,
        jnp.where(whole_bucket, walk_chunks, 0),
        unroll=SCAN_UNROLL,
        carry=jnp.int32(0),
    )(emit_whole)

    def emit_ranked(i, carry):
        woff, rank = carry
        key = scores_vmem[pl.ds(i * LANES, LANES)]
        in_set = key > lo_excl
        strict = key > hi_incl
        tie_rank = (
            plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32)) + rank
        )
        sel = strict | (in_set & (tie_rank <= my_take))
        return emit_fn(i, woff, key, sel), tie_rank[LANES - 1]

    plsc.parallel_loop(
        0,
        jnp.where(whole_bucket, 0, walk_chunks),
        unroll=4,
        carry=(jnp.int32(0), jnp.int32(0)),
    )(emit_ranked)


def _topk_body_no_keys(
    aw_hbm,
    scores_hbm,
    lengths_hbm,
    out_hbm,
    glob_vmem,
    rowbuf_vmem,
    *,
    b: int,
    p: int = 1,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    use_dyn_waves: bool = False,
):
    cores_per_chip = _get_sc_subcores()
    sc = pltpu.get_tpu_info().sparse_core
    sub_stride = 16 if sc is None else sc.num_subcores
    flat = lax.axis_index("core") * sub_stride + lax.axis_index("subcore")
    b_tokens = b // p
    g_slices = cores_per_chip // math.gcd(b_tokens, cores_per_chip)

    def remap_idx(w):
        idx = w * cores_per_chip + flat
        if use_dyn_waves:
            group_items = b_tokens * g_slices
            g_id, in_g = idx // group_items, idx % group_items
            r, s_local = in_g // g_slices, in_g % g_slices
            return (jnp.where(idx < b, r * p + g_id * g_slices + s_local, idx),)
        return (idx,)

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)

    def pipeline_step(scores_vmem, len_slice_vmem, out_vmem):
        row = remap_idx(pl.program_id(0))[0]
        eff_len = jnp.where(row < b, jnp.minimum(_vec_at(len_slice_vmem, 0), n), 0)
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi)

        @pl.when(i_hi == 0)
        def _empty():
            def fill_empty(i):
                out_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

            plsc.parallel_loop(0, k // LANES, unroll=8)(fill_empty)

        @pl.when(i_hi > 0)
        def _nonempty():
            _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            def fill_body(i):
                rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

            def emit(i, woff, _key, sel):
                plsc.store_compressed(
                    rowbuf_vmem.at[pl.ds(woff, LANES)], i * LANES + lane_iota, mask=sel
                )
                return woff + plsc.all_reduce_population_count(sel)[0]

            _radix_select_and_emit(
                scores_vmem,
                glob_vmem,
                walk_chunks,
                k,
                fill_fn=fill_body,
                fill_unroll=8,
                emit_fn=emit,
            )

            def copy_out(i):
                out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]

            plsc.parallel_loop(0, k // LANES, unroll=8)(copy_out)

    mode = pl.Buffered(1 if num_waves == 1 else 2)

    def spec(size):
        return pl.BlockSpec((size,), remap_idx, pipeline_mode=mode)

    def run_waves(num_steps):
        pltpu.emit_pipeline(
            pipeline_step,
            grid=(num_steps,),
            in_specs=(spec(slice_len), spec(LANES)),
            out_specs=(spec(k),),
        )(scores_hbm, lengths_hbm, out_hbm)

    if use_dyn_waves:
        pltpu.emit_pipeline(
            lambda aw_vmem: run_waves(aw_vmem[pl.ds(0, LANES)][0]),
            grid=(1,),
            in_specs=(pl.BlockSpec((LANES,), lambda _: (0,)),),
        )(aw_hbm)
    else:
        run_waves(num_waves)


def _topk_body(
    aw_hbm,
    scores_hbm,
    lengths_hbm,
    out_hbm,
    keys_hbm,
    glob_vmem,
    rowbuf_vmem,
    keybuf_vmem,
    *,
    b: int,
    p: int = 1,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    use_dyn_waves: bool = False,
):
    cores_per_chip = _get_sc_subcores()
    sc = pltpu.get_tpu_info().sparse_core
    sub_stride = 16 if sc is None else sc.num_subcores
    flat = lax.axis_index("core") * sub_stride + lax.axis_index("subcore")
    b_tokens = b // p
    g_slices = cores_per_chip // math.gcd(b_tokens, cores_per_chip)

    def remap_idx(w):
        idx = w * cores_per_chip + flat
        if use_dyn_waves:
            group_items = b_tokens * g_slices
            g_id, in_g = idx // group_items, idx % group_items
            r, s_local = in_g // g_slices, in_g % g_slices
            return (jnp.where(idx < b, r * p + g_id * g_slices + s_local, idx),)
        return (idx,)

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)

    def pipeline_step(scores_vmem, len_slice_vmem, out_vmem, keys_vmem):
        row = remap_idx(pl.program_id(0))[0]
        eff_len = jnp.where(row < b, jnp.minimum(_vec_at(len_slice_vmem, 0), n), 0)
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi)

        @pl.when(i_hi == 0)
        def _empty():
            def fill_empty(i):
                out_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
                keys_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES,), BITS_NEG_INF, jnp.int32
                )

            plsc.parallel_loop(0, k // LANES, unroll=8)(fill_empty)

        @pl.when(i_hi > 0)
        def _nonempty():
            _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
            tail = (walk_chunks - 1) * LANES
            scores_vmem[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < i_hi,
                scores_vmem[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

            def fill_body(i):
                rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)
                keybuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES,), BITS_NEG_INF, jnp.int32
                )

            def emit(i, woff, key, sel):
                plsc.store_compressed(
                    rowbuf_vmem.at[pl.ds(woff, LANES)], i * LANES + lane_iota, mask=sel
                )
                plsc.store_compressed(
                    keybuf_vmem.at[pl.ds(woff, LANES)], _monotone_key(key), mask=sel
                )
                return woff + plsc.all_reduce_population_count(sel)[0]

            _radix_select_and_emit(
                scores_vmem,
                glob_vmem,
                walk_chunks,
                k,
                fill_fn=fill_body,
                fill_unroll=8,
                emit_fn=emit,
            )

            def copy_out(i):
                out_vmem[pl.ds(i * LANES, LANES)] = rowbuf_vmem[pl.ds(i * LANES, LANES)]
                keys_vmem[pl.ds(i * LANES, LANES)] = keybuf_vmem[
                    pl.ds(i * LANES, LANES)
                ]

            plsc.parallel_loop(0, k // LANES, unroll=8)(copy_out)

    mode = pl.Buffered(1 if num_waves == 1 else 2)

    def spec(size):
        return pl.BlockSpec((size,), remap_idx, pipeline_mode=mode)

    def run_waves(num_steps):
        pltpu.emit_pipeline(
            pipeline_step,
            grid=(num_steps,),
            in_specs=(spec(slice_len), spec(LANES)),
            out_specs=(spec(k), spec(k)),
        )(scores_hbm, lengths_hbm, out_hbm, keys_hbm)

    if use_dyn_waves:
        pltpu.emit_pipeline(
            lambda aw_vmem: run_waves(aw_vmem[pl.ds(0, LANES)][0]),
            grid=(1,),
            in_specs=(pl.BlockSpec((LANES,), lambda _: (0,)),),
        )(aw_hbm)
    else:
        run_waves(num_waves)


def _run_sc_kernel(
    body_fn, out_type, scratch_types, name: str, scheduling_group_id: int | None, *args
):
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = pl.kernel(
        body_fn,
        out_type=out_type,
        scratch_types=scratch_types,
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        name=name,
    )(*args)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    return out


def _sc_topk_direct(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    p: int = 1,
    active_waves: jax.Array | None = None,
    scheduling_group_id: int | None = None,
    need_keys: bool = True,
) -> tuple[jax.Array, jax.Array | None]:
    """Single-stage lock-free SparseCore top-k."""
    scores = _as_i32_bits(scores)
    b, n = scores.shape
    slice_len = _align_to(n, LANES)
    num_subcores = _get_sc_subcores()
    num_waves = _cdiv(b, num_subcores)
    padded_b = num_waves * num_subcores
    if padded_b > b or slice_len > n:
        scores = jnp.pad(scores, ((0, padded_b - b), (0, slice_len - n)))

    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b))
    lengths_2d = jnp.pad(lengths[:, None], ((0, 0), (0, LANES - 1)))
    slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
    use_dyn = active_waves is not None
    aw_vec = jnp.full((LANES,), active_waves if use_dyn else num_waves, dtype=jnp.int32)

    if not need_keys:
        slots = _run_sc_kernel(
            functools.partial(
                _topk_body_no_keys,
                b=b,
                p=p,
                n=n,
                k=k,
                slice_len=slice_len,
                num_waves=num_waves,
                use_dyn_waves=use_dyn,
            ),
            slot_type,
            (
                pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
                pltpu.VMEM((k + LANES,), jnp.int32),
            ),
            f"sc_topk_nokeys_b{b}_n{n}_k{k}_dyn{int(use_dyn)}",
            scheduling_group_id,
            aw_vec,
            scores.reshape(-1),
            lengths_2d.reshape(-1),
        )
        return slots.reshape(padded_b, k)[:b], None

    slots, keys = _run_sc_kernel(
        functools.partial(
            _topk_body,
            b=b,
            p=p,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            use_dyn_waves=use_dyn,
        ),
        (slot_type, slot_type),
        (
            pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
        ),
        f"sc_topk_direct_b{b}_n{n}_k{k}_dyn{int(use_dyn)}",
        scheduling_group_id,
        aw_vec,
        scores.reshape(-1),
        lengths_2d.reshape(-1),
    )
    return slots.reshape(padded_b, k)[:b], keys.reshape(padded_b, k)[:b]


@functools.partial(
    jax.jit, static_argnames=("k", "scheduling_group_id", "stage2_scheduling_group_id")
)
def sparsecore_topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    scheduling_group_id: int | None = None,
    stage2_scheduling_group_id: int | None = None,
) -> jax.Array:
    """Exact top-k indices per row, unsorted, -1 suffix-padded."""
    scores = _as_i32_bits(scores)
    b, n = scores.shape
    p = _pick_partition(b, n, k)
    if p == 1:
        slots, _ = _sc_topk_direct(
            scores,
            k,
            row_lengths,
            scheduling_group_id=scheduling_group_id,
            need_keys=False,
        )
        return slots

    n_p = n // p
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0, n_p).reshape(b * p)
    active_slices = jnp.clip((row_lengths + n_p - 1) // n_p, 0, p)

    num_subcores = _get_sc_subcores()
    active_waves = None
    g_slices = num_subcores // math.gcd(b, num_subcores)
    if p % g_slices == 0:
        max_active_s = jnp.maximum(1, jnp.max(active_slices))
        active_waves = _cdiv(max_active_s, g_slices) * (b * g_slices // num_subcores)

    local_slots, cand_scores = _sc_topk_direct(
        scores.reshape(b * p, n_p),
        k,
        lengths_p,
        p=p,
        active_waves=active_waves,
        scheduling_group_id=scheduling_group_id,
        need_keys=True,
    )
    local_indices = local_slots.reshape(b, p, k)
    global_cand_indices = jnp.where(
        local_indices < 0, -1, offsets[None, :, None] + local_indices
    ).reshape(b, p * k)

    final_cand_slots, _ = _sc_topk_direct(
        cand_scores.reshape(b, p * k),
        k,
        active_slices * k,
        scheduling_group_id=stage2_scheduling_group_id,
        need_keys=False,
    )
    final_indices = jnp.take_along_axis(
        global_cand_indices, jnp.maximum(final_cand_slots, 0), axis=1
    )
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


def _sc_cand_gather_topk_body(
    scores_flat_hbm,
    cands_hbm,
    row_lens_hbm,
    out_hbm,
    cands_0,
    cands_1,
    idx_0,
    idx_1,
    scores_0,
    scores_1,
    glob_vmem,
    rowbuf_0,
    rowbuf_1,
    state_vmem,
    sem_ref,
    *,
    b: int,
    n: int,
    cb: int,
    k: int,
    num_waves: int,
    flat=None,
    select_cands=None,
):
    """1D ping-pong software-pipelined candidate gather + 256-bucket radix select."""
    num_subcores = _get_sc_subcores()
    sc = pltpu.get_tpu_info().sparse_core
    sub_stride = 16 if sc is None else sc.num_subcores
    if flat is None:
        flat = lax.axis_index("core") * sub_stride + lax.axis_index("subcore")
    padded_b = num_waves * num_subcores
    vecs_per_cand = (cb * 8) // LANES
    neg_inf_vec = jnp.full((LANES,), BITS_NEG_INF, dtype=jnp.int32)

    rl_dma = pltpu.make_async_copy(
        src_ref=row_lens_hbm.at[pl.ds(0, padded_b)],
        dst_ref=state_vmem.at[pl.ds(0, padded_b)],
        sem=sem_ref.at[0],
    )
    rl_dma.start()
    rl_dma.wait()

    def expand_slot(cands_s, idx_s, scores_s, row, row_len):
        row_base = row * n

        def expand_idx(j):
            blk = cands_s[pl.ds(j * LANES, LANES)]
            blk_ok = blk >= 0
            tok0 = jnp.left_shift(blk, 3)
            base0 = row_base + tok0
            for s in range(8):
                off = (j * 8 + s) * LANES
                idx_s[pl.ds(off, LANES)] = jnp.where(
                    blk_ok & (tok0 + s < row_len), base0 + s, -1
                )
                scores_s[pl.ds(off, LANES)] = neg_inf_vec

        plsc.parallel_loop(0, cb // LANES, unroll=4)(expand_idx)

    if select_cands is None and num_waves >= 2:
        num_pairs = _cdiv(num_waves, 2)

        def prefetch_gather(row, cands_s, idx_s, scores_s, c_sem, g_sem):
            row_safe = jnp.minimum(row, b - 1)
            row_len = jnp.where(row < b, jnp.minimum(_vec_at(state_vmem, row_safe), n), 0)
            c_dma = pltpu.make_async_copy(
                src_ref=cands_hbm.at[pl.ds(row_safe * cb, cb)],
                dst_ref=cands_s,
                sem=sem_ref.at[c_sem],
            )
            c_dma.start()
            c_dma.wait()
            expand_slot(cands_s, idx_s, scores_s, row_safe, row_len)
            pltpu.make_async_copy(
                src_ref=scores_flat_hbm.at[plsc.Indices(idx_s, ignored_value=-1)],
                dst_ref=scores_s,
                sem=sem_ref.at[g_sem],
            ).start()

        def finish_and_select(w, idx_s, scores_s, rowbuf_s, g_sem, o_sem):
            row = w * num_subcores + flat
            active = (w < num_waves) & (row < b)
            row_len = jnp.where(active, jnp.minimum(_vec_at(state_vmem, jnp.minimum(row, padded_b - 1)), n), 0)
            pltpu.make_async_copy(
                src_ref=scores_flat_hbm.at[plsc.Indices(idx_s, ignored_value=-1)],
                dst_ref=scores_s,
                sem=sem_ref.at[g_sem],
            ).wait()

            def fill_s(i):
                rowbuf_s[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

            @pl.when(row_len == 0)
            def _empty():
                @pl.when(active)
                def _w_empty():
                    plsc.parallel_loop(0, k // LANES, unroll=8)(fill_s)
                    d = pltpu.make_async_copy(
                        src_ref=rowbuf_s.at[pl.ds(0, k)],
                        dst_ref=out_hbm.at[pl.ds(row * k, k)],
                        sem=sem_ref.at[o_sem],
                    )
                    d.start()
                    d.wait()

            @pl.when(row_len > 0)
            def _nonempty():
                _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
                row_base = row * n

                def emit(i, woff, _key, sel):
                    plsc.store_compressed(
                        rowbuf_s.at[pl.ds(woff, LANES)],
                        idx_s[pl.ds(i * LANES, LANES)] - row_base,
                        mask=sel,
                    )
                    return woff + plsc.all_reduce_population_count(sel)[0]

                _radix_select_and_emit(
                    scores_s,
                    glob_vmem,
                    vecs_per_cand,
                    k,
                    fill_fn=fill_s,
                    fill_unroll=8,
                    emit_fn=emit,
                )
                d = pltpu.make_async_copy(
                    src_ref=rowbuf_s.at[pl.ds(0, k)],
                    dst_ref=out_hbm.at[pl.ds(row * k, k)],
                    sem=sem_ref.at[o_sem],
                )
                d.start()
                d.wait()

        prefetch_gather(flat, cands_0, idx_0, scores_0, 0, 2)

        @pl.loop(0, num_pairs)
        def _pair(p_idx):
            w0 = p_idx * 2
            w1 = w0 + 1
            @pl.when(w1 < num_waves)
            def _pf1():
                prefetch_gather(w1 * num_subcores + flat, cands_1, idx_1, scores_1, 1, 3)

            finish_and_select(w0, idx_0, scores_0, rowbuf_0, 2, 4)

            @pl.when(w1 < num_waves)
            def _do1():
                @pl.when(w0 + 2 < num_waves)
                def _pf0():
                    prefetch_gather((w0 + 2) * num_subcores + flat, cands_0, idx_0, scores_0, 0, 2)
                finish_and_select(w1, idx_1, scores_1, rowbuf_1, 3, 5)
        return

    def fill_body(i):
        rowbuf_0[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

    def write_row(row):
        o_dma = pltpu.make_async_copy(
            src_ref=rowbuf_0.at[pl.ds(0, k)],
            dst_ref=out_hbm.at[pl.ds(row * k, k)],
            sem=sem_ref.at[0],
        )
        o_dma.start()
        o_dma.wait()

    @pl.loop(0, num_waves)
    def _wave(w):
        row = w * num_subcores + flat
        active = row < b
        row_len = jnp.where(active, jnp.minimum(_vec_at(state_vmem, row), n), 0)

        @pl.when(row_len == 0)
        def _empty():
            @pl.when(active)
            def _write_empty():
                plsc.parallel_loop(0, k // LANES, unroll=8)(fill_body)
                write_row(row)

        @pl.when(row_len > 0)
        def _nonempty():
            c_out_dma = None
            if select_cands is None:
                c_dma = pltpu.make_async_copy(
                    src_ref=cands_hbm.at[pl.ds(row * cb, cb)],
                    dst_ref=cands_0,
                    sem=sem_ref.at[0],
                )
                c_dma.start()
                c_dma.wait()
            else:
                c_out_dma = select_cands(row, row_len, cands_0)
            expand_slot(cands_0, idx_0, scores_0, row, row_len)

            g_dma = pltpu.make_async_copy(
                src_ref=scores_flat_hbm.at[plsc.Indices(idx_0, ignored_value=-1)],
                dst_ref=scores_0,
                sem=sem_ref.at[1],
            )
            g_dma.start()
            _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
            g_dma.wait()
            row_base = row * n

            def emit(i, woff, _key, sel):
                plsc.store_compressed(
                    rowbuf_0.at[pl.ds(woff, LANES)],
                    idx_0[pl.ds(i * LANES, LANES)] - row_base,
                    mask=sel,
                )
                return woff + plsc.all_reduce_population_count(sel)[0]

            _radix_select_and_emit(
                scores_0,
                glob_vmem,
                vecs_per_cand,
                k,
                fill_fn=fill_body,
                fill_unroll=8,
                emit_fn=emit,
            )
            write_row(row)
            if c_out_dma is not None:
                c_out_dma.wait()


def sc_cand_gather_topk(
    scores: jax.Array,
    candidates: jax.Array,
    row_lengths: jax.Array,
    *,
    k: int = 512,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """1D SparseCore indirect gather DMA + in-VMEM top-k over candidate blocks."""
    scores = _as_i32_bits(scores)
    b, n = scores.shape
    cb = candidates.shape[1]
    cand_len = cb * 8
    num_subcores = _get_sc_subcores()
    num_waves = _cdiv(b, num_subcores)
    padded_b = num_waves * num_subcores
    if padded_b > b:
        candidates = jnp.pad(
            candidates, ((0, padded_b - b), (0, 0)), constant_values=-1
        )
    rl_pad = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b + LANES))

    out = _run_sc_kernel(
        functools.partial(
            _sc_cand_gather_topk_body, b=b, n=n, cb=cb, k=k, num_waves=num_waves
        ),
        jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
        (
            pltpu.VMEM((cb,), jnp.int32),
            pltpu.VMEM((cb,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((padded_b + LANES,), jnp.int32),
            pltpu.SemaphoreType.DMA((6,)),
        ),
        f"sc_cand_gather_topk_b{b}_n{n}_cb{cb}_k{k}",
        scheduling_group_id,
        scores.reshape(-1),
        candidates.reshape(-1),
        rl_pad,
    )
    return out.reshape(padded_b, k)[:b]


def sc_cand_pregathered_topk(
    cand_scores: jax.Array,
    candidates: jax.Array,
    row_lengths: jax.Array,
    *,
    k: int = 512,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """SparseCore top-k over pre-scored [b, cb * 8] contiguous candidate slots."""
    cand_scores = _as_i32_bits(cand_scores)
    b, cand_len = cand_scores.shape
    cb = candidates.shape[1]
    vecs_per_cand = cand_len // LANES
    num_subcores = _get_sc_subcores()
    num_waves = _cdiv(b, num_subcores)
    padded_b = num_waves * num_subcores
    if padded_b > b:
        cand_scores = jnp.pad(
            cand_scores, ((0, padded_b - b), (0, 0)), constant_values=BITS_NEG_INF
        )
        candidates = jnp.pad(
            candidates, ((0, padded_b - b), (0, 0)), constant_values=-1
        )
    rl_pad = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b + LANES))

    def _pregathered_body(
        cand_scores_flat_hbm,
        cands_hbm,
        row_lens_hbm,
        out_hbm,
        cands_vmem,
        idx_vmem,
        scores_vmem,
        glob_vmem,
        rowbuf_vmem,
        state_vmem,
        sem_ref,
    ):
        sc = pltpu.get_tpu_info().sparse_core
        sub_stride = 16 if sc is None else sc.num_subcores
        flat = lax.axis_index("core") * sub_stride + lax.axis_index("subcore")
        rl_dma = pltpu.make_async_copy(
            src_ref=row_lens_hbm.at[pl.ds(0, padded_b + LANES)],
            dst_ref=state_vmem,
            sem=sem_ref.at[0],
        )
        rl_dma.start()
        rl_dma.wait()

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

        def write_row(row):
            o_dma = pltpu.make_async_copy(
                src_ref=rowbuf_vmem.at[pl.ds(0, k)],
                dst_ref=out_hbm.at[pl.ds(row * k, k)],
                sem=sem_ref.at[0],
            )
            o_dma.start()
            o_dma.wait()

        @pl.loop(0, num_waves)
        def _wave(w):
            row = w * num_subcores + flat
            active = row < b
            row_len = jnp.where(active, _vec_at(state_vmem, row), 0)

            @pl.when(row_len == 0)
            def _empty():
                @pl.when(active)
                def _w():
                    plsc.parallel_loop(0, k // LANES, unroll=8)(fill_body)
                    write_row(row)

            @pl.when(row_len > 0)
            def _nonempty():
                c_dma = pltpu.make_async_copy(
                    src_ref=cands_hbm.at[pl.ds(row * cb, cb)],
                    dst_ref=cands_vmem,
                    sem=sem_ref.at[0],
                )
                s_dma = pltpu.make_async_copy(
                    src_ref=cand_scores_flat_hbm.at[pl.ds(row * cand_len, cand_len)],
                    dst_ref=scores_vmem,
                    sem=sem_ref.at[1],
                )
                c_dma.start()
                s_dma.start()
                _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
                c_dma.wait()

                def expand_idx(j):
                    blk = cands_vmem[pl.ds(j * LANES, LANES)]
                    blk_ok = blk >= 0
                    tok0 = jnp.left_shift(blk, 3)
                    for s in range(8):
                        off = (j * 8 + s) * LANES
                        idx_vmem[pl.ds(off, LANES)] = jnp.where(
                            blk_ok & (tok0 + s < row_len), tok0 + s, -1
                        )

                plsc.parallel_loop(0, cb // LANES, unroll=4)(expand_idx)
                s_dma.wait()

                def emit(i, woff, _key, sel):
                    plsc.store_compressed(
                        rowbuf_vmem.at[pl.ds(woff, LANES)],
                        idx_vmem[pl.ds(i * LANES, LANES)],
                        mask=sel,
                    )
                    return woff + plsc.all_reduce_population_count(sel)[0]

                _radix_select_and_emit(
                    scores_vmem,
                    glob_vmem,
                    vecs_per_cand,
                    k,
                    fill_fn=fill_body,
                    fill_unroll=8,
                    emit_fn=emit,
                )
                write_row(row)

    out = _run_sc_kernel(
        _pregathered_body,
        jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
        (
            pltpu.VMEM((cb,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((padded_b + LANES,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        f"sc_cand_pregathered_topk_b{b}_cb{cb}_k{k}",
        scheduling_group_id,
        cand_scores.reshape(-1),
        candidates.reshape(-1),
        rl_pad,
    )
    return out.reshape(padded_b, k)[:b]


def _sc_write_cand_and_topk_body(
    blk_scores_flat_hbm,
    scores_flat_hbm,
    row_lens_hbm,
    out_hbm,
    cands_hbm,
    blk_vmem,
    cands_0,
    cands_1,
    idx_0,
    idx_1,
    scores_0,
    scores_1,
    glob_vmem,
    rowbuf_0,
    rowbuf_1,
    state_vmem,
    sem_ref,
    *,
    b: int,
    n_blk: int,
    n: int,
    cb: int,
    k: int,
    num_blk_sub_bkv: int,
    num_waves: int,
):
    sc = pltpu.get_tpu_info().sparse_core
    sub_stride = 16 if sc is None else sc.num_subcores
    flat = lax.axis_index("core") * sub_stride + lax.axis_index("subcore")
    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    w_bkv = num_blk_sub_bkv * 128
    edges = (0, *(min(s, n_blk) for s in (8192, 14336, 26624, 38912)), n_blk)

    def select_cands(row, row_len, cands_vmem):
        blk_rl = jnp.right_shift(row_len + 7, 3)
        blk_rl_aligned = jnp.minimum(((blk_rl + w_bkv - 1) // w_bkv) * w_bkv, n_blk)
        blk_row_base = row * n_blk

        def _dma_blk(sz):
            d = pltpu.make_async_copy(
                src_ref=blk_scores_flat_hbm.at[pl.ds(blk_row_base, sz)],
                dst_ref=blk_vmem.at[pl.ds(0, sz)],
                sem=sem_ref.at[0],
            )
            d.start()
            _zero_hist(glob_vmem, NUM_BUCKETS // LANES)
            d.wait()

        for j in range(5):
            lo, hi = edges[j], edges[j + 1]
            if j == 0:
                cond = blk_rl_aligned <= hi
            elif j == 4:
                cond = blk_rl_aligned > lo
            else:
                cond = (blk_rl_aligned > lo) & (blk_rl_aligned <= hi)
            pl.when(cond)(functools.partial(_dma_blk, hi))

        walk_blk = _cdiv_dyn(blk_rl_aligned)

        def fill_cands(i):
            cands_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

        def emit_blk(i, woff, _key, sel):
            col = i * LANES + lane_iota
            bkv_base = jnp.bitwise_and(col, ~(w_bkv - 1))
            sub_idx = jnp.right_shift(jnp.bitwise_and(col, w_bkv - 1), 7)
            b_lane = jnp.bitwise_and(jnp.right_shift(col, 3), 15)
            s_idx = jnp.bitwise_and(col, 7)
            blk_id = bkv_base | ((s_idx * num_blk_sub_bkv + sub_idx) << 4 | b_lane)
            plsc.store_compressed(cands_vmem.at[pl.ds(woff, LANES)], blk_id, mask=sel)
            return woff + plsc.all_reduce_population_count(sel)[0]

        _radix_select_and_emit(
            blk_vmem,
            glob_vmem,
            walk_blk,
            cb,
            fill_fn=fill_cands,
            fill_unroll=8,
            emit_fn=emit_blk,
        )
        c_out_dma = pltpu.make_async_copy(
            src_ref=cands_vmem.at[pl.ds(0, cb)],
            dst_ref=cands_hbm.at[pl.ds(row * cb, cb)],
            sem=sem_ref.at[2],
        )
        c_out_dma.start()
        return c_out_dma

    _sc_cand_gather_topk_body(
        scores_flat_hbm,
        cands_hbm,
        row_lens_hbm,
        out_hbm,
        cands_0,
        cands_1,
        idx_0,
        idx_1,
        scores_0,
        scores_1,
        glob_vmem,
        rowbuf_0,
        rowbuf_1,
        state_vmem,
        sem_ref,
        b=b,
        n=n,
        cb=cb,
        k=k,
        num_waves=num_waves,
        flat=flat,
        select_cands=select_cands,
    )


def sc_write_cand_and_topk(
    blk_scores: jax.Array,
    scores: jax.Array,
    row_lengths: jax.Array,
    *,
    cb: int = 2048,
    k: int = 512,
    num_blk_sub_bkv: int = 8,
    scheduling_group_id: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore block-top-cb + 1D indirect gather token-top-k."""
    blk_scores = _as_i32_bits(blk_scores)
    scores = _as_i32_bits(scores)
    b, n_blk = blk_scores.shape
    _, n = scores.shape
    cand_len = cb * 8
    num_subcores = _get_sc_subcores()
    num_waves = _cdiv(b, num_subcores)
    padded_b = num_waves * num_subcores
    rl_pad = jnp.pad(row_lengths.astype(jnp.int32), (0, padded_b - b + LANES))

    topk_flat, cands_flat = _run_sc_kernel(
        functools.partial(
            _sc_write_cand_and_topk_body,
            b=b,
            n_blk=n_blk,
            n=n,
            cb=cb,
            k=k,
            num_blk_sub_bkv=num_blk_sub_bkv,
            num_waves=num_waves,
        ),
        (
            jax.ShapeDtypeStruct((padded_b * k,), jnp.int32),
            jax.ShapeDtypeStruct((padded_b * cb,), jnp.int32),
        ),
        (
            pltpu.VMEM((n_blk,), jnp.int32),
            pltpu.VMEM((cb + LANES,), jnp.int32),
            pltpu.VMEM((cb + LANES,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((cand_len,), jnp.int32),
            pltpu.VMEM((NUM_BUCKETS + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((k + LANES,), jnp.int32),
            pltpu.VMEM((padded_b + LANES,), jnp.int32),
            pltpu.SemaphoreType.DMA((6,)),
        ),
        f"sc_write_cand_and_topk_b{b}_nb{n_blk}_n{n}_cb{cb}_k{k}",
        scheduling_group_id,
        blk_scores.reshape(-1),
        scores.reshape(-1),
        rl_pad,
    )
    return topk_flat.reshape(padded_b, k)[:b], cands_flat.reshape(padded_b, cb)[:b]
