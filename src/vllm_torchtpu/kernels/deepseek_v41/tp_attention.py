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
"""Token-parallel attention core for DeepSeek-V4.1 under TP (VLLM_DSV41_TP_ATTN).

Head-parallel TP attention makes every rank read the same SWA window pages and
CSA top-k rows for all T tokens, for its H/tp heads. Here every rank attends
with all H heads for its contiguous token slice [rank * T/tp, (rank+1) * T/tp)
instead, so those cache reads are split across ranks, and the attention output
reduce-scatter disappears (the caller projects with replicated wq_b/wo_a/wo_b).

A request's slice on rank r reads earlier tokens of the same chunk that belong
to ranks < r from the cache, so every rank first writes all T new SWA rows to
its cache replica (`swa_store_all_xla`). The attention kernel's own write of the
local rows then rewrites identical bytes. CED decoder layers are not supported
(their q_len > 1 window start depends on the whole chunk).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    prepare_kv_inputs,
)


def rebase_token_slice(
    cu_q_lens: jax.Array,  # i32[S + 1]
    kv_lens: jax.Array | None,  # i32[S]
    page_indices: jax.Array | None,  # i32[S * P] (flat) or i32[S, P]
    distribution: jax.Array,  # i32[3]
    token_start: jax.Array | int,
    *,
    local_tokens: int,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array]:
    """Per-request metadata for the token slice [token_start, +local_tokens).

    Requests overlapping the slice are packed to local indices 0..n_local-1
    (`_run_swa` trims to the first align(num_tokens, 8) requests), cu_q_lens is
    clipped to the slice, kv_lens drops a request's tokens past the slice end
    (so every local token keeps its absolute position), page-table rows move
    with their request, and distribution shifts by the requests that end before
    the slice. Static shapes are unchanged.
    """
    num_seq_slots = cu_q_lens.shape[0] - 1
    ts = jnp.asarray(token_start, jnp.int32)
    te = ts + local_tokens
    num_seqs = distribution[2]

    seq = jnp.arange(num_seq_slots, dtype=jnp.int32)
    real = seq < num_seqs
    starts, ends = cu_q_lens[:-1], cu_q_lens[1:]
    s0 = jnp.sum((real & (ends <= ts)).astype(jnp.int32))
    n_local = jnp.sum((real & (starts < te) & (ends > ts)).astype(jnp.int32))

    src = jnp.minimum(s0 + seq, num_seq_slots - 1)
    lvalid = seq < n_local
    g_end = ends[src]
    total_local = jnp.clip(cu_q_lens[num_seqs], ts, te) - ts
    cu_local = jnp.concatenate(
        [
            jnp.zeros((1,), jnp.int32),
            jnp.where(lvalid, jnp.clip(g_end, ts, te) - ts, total_local),
        ]
    ).astype(cu_q_lens.dtype)

    kv_local = None
    if kv_lens is not None:
        dropped = g_end - jnp.minimum(g_end, te)
        kv_local = jnp.where(lvalid, kv_lens[src] - dropped, 0).astype(kv_lens.dtype)

    table_local = None
    if page_indices is not None:
        table = page_indices.reshape(num_seq_slots, -1)
        table_local = jnp.where(lvalid[:, None], table[src], 0)
        table_local = table_local.astype(page_indices.dtype).reshape(page_indices.shape)

    dist_local = jnp.stack(
        [
            jnp.clip(distribution[0] - s0, 0, n_local),
            jnp.clip(distribution[1] - s0, 0, n_local),
            n_local,
        ]
    ).astype(distribution.dtype)
    return cu_local, kv_local, table_local, dist_local


def _swa_token_rows_for_indices(
    t: jax.Array,  # i32[K]
    kv_lens: jax.Array,  # i32[S]
    table: jax.Array,  # i32[S, P], raw (un-ringed) page ids
    cu_q_lens: jax.Array,  # i32[S + 1]
    n_valid_tokens: jax.Array,  # i32[]
    *,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_seq_slots = kv_lens.shape[0]
    req = jnp.searchsorted(cu_q_lens[1:], t, side="right", method="compare_all")
    req = jnp.minimum(req, num_seq_slots - 1).astype(jnp.int32)
    pos = kv_lens[req] - (cu_q_lens[req + 1] - t)
    col = jnp.clip(pos // logical_page_size, 0, table.shape[1] - 1)
    page = table[req, col]
    if num_cache_pages > 1:
        page = jnp.where(page > 0, ((page - 1) % (num_cache_pages - 1)) + 1, 0)
    row = page * phys_tokens_per_page + pos % logical_page_size
    valid = t < n_valid_tokens
    row = jnp.where(valid, row, num_cache_pages * phys_tokens_per_page)
    return req, pos, row


def swa_token_rows(
    kv_lens: jax.Array,  # i32[S]
    page_indices: jax.Array,  # i32[S * P] or i32[S, P], raw (un-ringed) page ids
    cu_q_lens: jax.Array,  # i32[S + 1]
    distribution: jax.Array,  # i32[3]
    *,
    num_tokens: int,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
) -> jax.Array:
    """Cache token row of every new token, as the SWA kernel computes it.

    Row = ring(page_indices[req, pos // P]) * phys_tokens_per_page + pos % P,
    with ring(p) = ((p - 1) % (N - 1)) + 1 for p > 0 (`_prepare_swa_metadata`).
    Padding tokens get the out-of-range row N * phys_tokens_per_page.
    """
    num_seq_slots = kv_lens.shape[0]
    table = page_indices.reshape(num_seq_slots, -1)
    t = jnp.arange(num_tokens, dtype=jnp.int32)
    _, _, row = _swa_token_rows_for_indices(
        t,
        kv_lens,
        table,
        cu_q_lens,
        cu_q_lens[distribution[2]],
        num_cache_pages=num_cache_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    return row


def _pack_kv_words(
    new_kv: jax.Array,
    sw_cache_dtype: jnp.dtype,
    slots_per_token: int,
    kv_packing: int,
    lanes: int,
) -> jax.Array:
    T = new_kv.shape[0]
    if csa_cache_layout.is_word_array(sw_cache_dtype) and kv_packing == 4:
        u16 = lax.bitcast_convert_type(
            new_kv.reshape(T * slots_per_token, 2 * lanes), jnp.uint16
        )
        lo = u16[:, :lanes].astype(jnp.uint32)
        hi = u16[:, lanes:].astype(jnp.uint32)
        return lax.bitcast_convert_type(lo | (hi << 16), sw_cache_dtype).reshape(
            T, slots_per_token, lanes
        )
    tok = prepare_kv_inputs(new_kv).reshape(T, slots_per_token, kv_packing, lanes)
    if csa_cache_layout.is_word_array(sw_cache_dtype):
        # Byte b of word w is slab sub-row b (`csa_cache_layout.slabs_to_words`).
        tok = lax.bitcast_convert_type(jnp.moveaxis(tok, 2, -1), sw_cache_dtype)
    return tok


def swa_store_all_xla(
    sw_cache: jax.Array,  # int32 [N, rows, 128] or uint8 [N, rows, packing, 128]
    new_kv: jax.Array,  # bf16 [T, head_dim]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    logical_page_size: int,
) -> jax.Array:
    """Writes every new token's SWA row with one XLA row scatter."""
    num_pages, rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    slots_per_token = new_kv.shape[1] * 2 // (kv_packing * lanes)
    phys_tokens_per_page = rows // slots_per_token
    dst = swa_token_rows(
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_tokens=new_kv.shape[0],
        num_cache_pages=num_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    tok = _pack_kv_words(new_kv, sw_cache.dtype, slots_per_token, kv_packing, lanes)
    flat = sw_cache.reshape(num_pages * phys_tokens_per_page, *tok.shape[1:])
    return flat.at[dst].set(tok, mode="drop").reshape(sw_cache.shape)


def _swa_store_pallas_coalesced_kernel(
    dst_ref,        # SMEM: i32[T] (packed: (run_len << 24) | row_start for has_rem != 0)
    blk_meta_ref,   # SMEM: i32[T // blk_sz, 5] -> (len0, row0, len1, row1, has_rem)
    sw_cache_ref,   # HBM: [total_rows, ...]
    tok_ref,        # HBM: [T, ...]
    out_cache_ref,  # HBM aliased to sw_cache_ref
    sem,
):
    del sw_cache_ref
    T = tok_ref.shape[0]
    total_rows = out_cache_ref.shape[0]
    blk_sz = 64 if T % 64 == 0 else (16 if T % 16 == 0 else 1)
    num_blks = T // blk_sz

    def blk_body(b, total_rows_copied):
        len0 = blk_meta_ref[b, 0]
        row0 = blk_meta_ref[b, 1]
        len1 = blk_meta_ref[b, 2]
        row1 = blk_meta_ref[b, 3]
        has_rem = blk_meta_ref[b, 4] != 0
        t0 = b * blk_sz

        @pl.when(len0 > 0)
        def _cp0():
            pltpu.make_async_copy(
                src_ref=tok_ref.at[pl.ds(t0, len0)],
                dst_ref=out_cache_ref.at[pl.ds(row0, len0)],
                sem=sem.at[0],
            ).start()

        @pl.when(len1 > 0)
        def _cp1():
            pltpu.make_async_copy(
                src_ref=tok_ref.at[pl.ds(t0 + len0, len1)],
                dst_ref=out_cache_ref.at[pl.ds(row1, len1)],
                sem=sem.at[0],
            ).start()

        copied = total_rows_copied + len0 + len1

        def do_rem(c):
            def run_cond(state):
                i, _ = state
                return i < blk_sz

            def run_body(state):
                i, cnt = state
                t = t0 + i
                packed = dst_ref[t]
                row_start = packed & 0xFFFFFF
                run_len = packed >> 24
                valid = row_start < total_rows

                def copy_run():
                    pltpu.make_async_copy(
                        src_ref=tok_ref.at[pl.ds(t, run_len)],
                        dst_ref=out_cache_ref.at[pl.ds(row_start, run_len)],
                        sem=sem.at[0],
                    ).start()
                    return (i + run_len, cnt + run_len)

                return lax.cond(valid, copy_run, lambda: (i + run_len, cnt))

            _, final_cnt = lax.while_loop(run_cond, run_body, (jnp.int32(0), c))
            return final_cnt

        return lax.cond(has_rem, do_rem, lambda c: c, copied)

    n_copied = lax.fori_loop(0, num_blks, blk_body, jnp.int32(0))

    @pl.when(n_copied > 0)
    def _wait_all():
        pltpu.make_async_copy(
            src_ref=out_cache_ref.at[pl.ds(0, n_copied)],
            dst_ref=out_cache_ref.at[pl.ds(0, n_copied)],
            sem=sem.at[0],
        ).wait()


def _pack_blk_run_len_fast(rows_batch: jax.Array, total_rows: int) -> jax.Array:
    """Packs (run_len << 24) | row for each [B_BATCH, blk_sz] row using log2 shift-min."""
    b_batch, blk_sz = rows_batch.shape
    if blk_sz == 1:
        return (jnp.int32(1) << 24) | (rows_batch & 0xFFFFFF)
    r0 = rows_batch[:, :-1]
    r1 = rows_batch[:, 1:]
    v0 = r0 < total_rows
    v1 = r1 < total_rows
    same_run = (v0 & v1 & (r1 == r0 + 1)) | (~v0 & ~v1)
    is_break = jnp.concatenate(
        [~same_run, jnp.ones((b_batch, 1), dtype=jnp.bool_)], axis=1
    )
    idx = jnp.arange(blk_sz, dtype=jnp.int32)[None, :]
    run_end = jnp.where(is_break, idx + 1, blk_sz)
    d = 1
    while d < blk_sz:
        run_end = jnp.minimum(
            run_end,
            jnp.pad(run_end[:, d:], ((0, 0), (0, d)), constant_values=blk_sz),
        )
        d *= 2
    run_len = run_end - idx
    return (run_len << 24) | (rows_batch & 0xFFFFFF)


def swa_store_all_pallas(
    sw_cache: jax.Array,  # int32 [N, rows, 128] or uint8 [N, rows, packing, 128]
    new_kv: jax.Array,  # bf16 [T, head_dim]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    logical_page_size: int,
) -> jax.Array:
    """Writes every new token's SWA row in-place via page-coalesced Pallas HBM->HBM DMA."""
    num_pages, rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    slots_per_token = new_kv.shape[1] * 2 // (kv_packing * lanes)
    phys_tokens_per_page = rows // slots_per_token
    total_rows = num_pages * phys_tokens_per_page
    if total_rows == 0:
        return sw_cache
    T = new_kv.shape[0]
    num_seq_slots = kv_lens.shape[0]
    table = page_indices.reshape(num_seq_slots, -1)

    tok = _pack_kv_words(new_kv, sw_cache.dtype, slots_per_token, kv_packing, lanes)
    flat = sw_cache.reshape(total_rows, *tok.shape[1:])

    blk_sz = 64 if T % 64 == 0 else (16 if T % 16 == 0 else 1)
    num_blks = T // blk_sz
    b_idx = jnp.arange(num_blks, dtype=jnp.int32)
    n_valid_tokens = cu_q_lens[distribution[2]]
    n_blk = jnp.clip(n_valid_tokens - b_idx * blk_sz, 0, blk_sz)
    t_start = b_idx * blk_sz
    t_end = t_start + jnp.maximum(n_blk - 1, 0)
    t_ends = jnp.stack([t_start, t_end], axis=1)  # [num_blks, 2]
    t_flat = t_ends.reshape(-1)  # [2 * num_blks]

    req_flat, pos_flat, row_flat = _swa_token_rows_for_indices(
        t_flat,
        kv_lens,
        table,
        cu_q_lens,
        n_valid_tokens,
        num_cache_pages=num_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    req_ends = req_flat.reshape(num_blks, 2)
    pos_ends = pos_flat.reshape(num_blks, 2)
    row_ends = row_flat.reshape(num_blks, 2)

    can_split2 = (
        (n_blk > 0)
        & (req_ends[:, 0] == req_ends[:, 1])
        & (pos_ends[:, 0] >= 0)
        & (n_blk <= logical_page_size)
    )
    pos0 = pos_ends[:, 0]
    rem_in_page = logical_page_size - (pos0 % logical_page_size)
    len0 = jnp.where(can_split2, jnp.minimum(n_blk, rem_in_page), 0)
    row0 = row_ends[:, 0]
    len1 = jnp.where(can_split2, n_blk - len0, 0)
    row1 = row_ends[:, 1] - len1 + 1
    has_rem = (~can_split2 & (n_blk > 0)).astype(jnp.int32)
    blk_meta = jnp.stack([len0, row0, len1, row1, has_rem], axis=1)

    n_rem = jnp.sum(has_rem)
    B_BATCH = min(16, num_blks)
    rem_pos = jnp.cumsum(has_rem) - 1
    pad_len = ((num_blks + B_BATCH - 1) // B_BATCH) * B_BATCH
    rem_blk_ids = (
        jnp.zeros((pad_len,), dtype=jnp.int32)
        .at[jnp.where(has_rem != 0, rem_pos, pad_len)]
        .set(b_idx, mode="drop")
    )

    init_dst = jnp.zeros((num_blks, blk_sz), dtype=jnp.int32)
    iota_blk = jnp.arange(blk_sz, dtype=jnp.int32)[None, :]
    iota_batch = jnp.arange(B_BATCH, dtype=jnp.int32)

    def rem_body(state):
        k, acc = state
        blks = rem_blk_ids[k + iota_batch]
        t_batch = (blks[:, None] * blk_sz + iota_blk).reshape(-1)
        _, _, rows_batch = _swa_token_rows_for_indices(
            t_batch,
            kv_lens,
            table,
            cu_q_lens,
            n_valid_tokens,
            num_cache_pages=num_pages,
            phys_tokens_per_page=phys_tokens_per_page,
            logical_page_size=logical_page_size,
        )
        rows_batch = rows_batch.reshape(B_BATCH, blk_sz)
        packed_batch = _pack_blk_run_len_fast(rows_batch, total_rows)
        valid_k = (k + iota_batch) < n_rem
        target_blks = jnp.where(valid_k, blks, num_blks)
        acc = acc.at[target_blks].set(packed_batch, mode="drop")
        return (k + B_BATCH, acc)

    _, dst_2d = lax.while_loop(
        lambda s: s[0] < n_rem,
        rem_body,
        (jnp.int32(0), init_dst),
    )
    dst = dst_2d.reshape(T)

    out_flat = pl.pallas_call(
        _swa_store_pallas_coalesced_kernel,
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        ),
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="swa_store_all_pallas",
    )(dst, blk_meta, flat, tok)
    return out_flat.reshape(sw_cache.shape)


def attention_tp_local(
    q: jax.Array,  # [T / tp, H, D]: all heads of this rank's token slice
    new_kv: jax.Array,  # [T, D]: every token's normed, roped KV
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # [T, ...]: CSA top-k rows (positions + 1 for SWA, unused)
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,  # [H]
    rank: jax.Array,  # i32[1]
    main_cache_rope: jax.Array | None = None,  # CSA layers only
    *,
    base_fn: Callable,
    logical_page_size: int,
    store_fn: Callable = swa_store_all_pallas,
) -> tuple[jax.Array, jax.Array]:
    """Stores all T SWA rows, then runs `base_fn` on this rank's token slice.

    `base_fn` is `_attention_swa` / `_attention_csa` with static kwargs bound.
    The CSA kernel reads only the main cu_q_lens / distribution (its top-k rows
    are absolute cache rows), so main kv_lens and page indices pass through.
    """
    local_tokens = q.shape[0]
    token_start = rank[0] * local_tokens
    sw_cache = store_fn(
        sw_cache,
        new_kv,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        logical_page_size=logical_page_size,
    )
    swa_cu, swa_kv, swa_pages, swa_dist = rebase_token_slice(
        swa_cu_q_lens,
        swa_kv_lens,
        swa_page_indices,
        swa_distribution,
        token_start,
        local_tokens=local_tokens,
    )
    is_csa = main_cache_rope is not None
    if is_csa:
        main_cu, _, _, main_dist = rebase_token_slice(
            main_cu_q_lens,
            None,
            None,
            main_distribution,
            token_start,
            local_tokens=local_tokens,
        )
    else:
        main_cu, main_dist = main_cu_q_lens, main_distribution

    def local(x: jax.Array) -> jax.Array:
        if x.shape[0] == local_tokens:
            return x
        return lax.dynamic_slice_in_dim(x, token_start, local_tokens, axis=0)

    return base_fn(
        q,
        local(new_kv),
        sw_cache,
        swa_kv,
        swa_pages,
        swa_cu,
        swa_dist,
        main_cache_kv,
        main_kv_lens,
        local(extra),
        main_page_indices,
        main_cu,
        main_dist,
        attention_sinks,
        *((main_cache_rope,) if is_csa else ()),
    )
