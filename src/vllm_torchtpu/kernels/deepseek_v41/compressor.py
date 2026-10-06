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
"""DeepSeek-V4.1 compressor and indexer JAX/Pallas kernels."""

import functools
import inspect

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compute
from vllm_torchtpu.kernels.deepseek_v4.rope import rope, rope_quant
from vllm_torchtpu.kernels.deepseek_v41.sparse_mla import (
    align_to,
    cdiv,
    encode_csa_gather_indices,
)
from vllm_torchtpu.kernels.deepseek_v41.streamindex_topk import (
    streamindex_topk,
)


def _rope_block_start(
    x: jax.Array, pos: jax.Array, cos_sin_cache: jax.Array, compress_ratio: int
) -> jax.Array:
    """GPT-J RoPE on the trailing 64 channels at each token's block-start position."""
    x = x.astype(jnp.float32)
    start = (pos // compress_ratio) * compress_ratio
    return rope(x, jnp.clip(start, 0, cos_sin_cache.shape[0] - 1), cos_sin_cache)


def _compressed_kv_positions(
    pos: jax.Array,
    kv_block_table: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    page_size: int,
    compress_ratio: int,
    block_stride: int = 1,
):
    num_tokens = pos.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    req = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        jnp.diff(query_start_loc),
        total_repeat_length=num_tokens,
    )
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = valid & (((pos + 1) % compress_ratio) == 0)
    kv_idx = jnp.maximum(pos // compress_ratio, 0)
    kv_page_idx = jnp.clip((kv_idx // page_size) * block_stride, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]
    return req, token_idx, is_boundary, kv_page, kv_page_off


def compress_r2_pool_norm_jax(
    wkv_wgate: jax.Array,  # [num_tokens, 1024] f32
    norm_weight: jax.Array,  # [512] bf16/f32
    positions: jax.Array,  # [num_tokens] i32
    state_block_table: jax.Array,  # [max_num_seqs * max_state_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    state_cache: jax.Array,  # [num_pages, page_size, 128] i32
    *,
    state_block_size: int,
    rms_eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Pool pairs of tokens (even, odd) for compress_ratio=2 and apply RMSNorm."""
    head_dim = norm_weight.shape[0]
    num_tokens = wkv_wgate.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_state_blocks = state_block_table.shape[0] // max_num_seqs

    num_pages, page_size = state_cache.shape[:2]
    states_per_page = page_size // 8
    total_state_slots = num_pages * states_per_page
    state_slots = state_cache.reshape((total_state_slots, 8, *state_cache.shape[2:]))

    req = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        jnp.diff(query_start_loc),
        total_repeat_length=num_tokens,
    )
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    pos = positions.astype(jnp.int32)

    def _state_slot_for(r: jax.Array, p: jax.Array) -> jax.Array:
        page_idx = jnp.clip(p // state_block_size, 0, max_state_blocks - 1)
        page = state_block_table[r * max_state_blocks + page_idx]
        if num_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_pages - 1)) + 1, 0)
        return jnp.clip(
            page * states_per_page + (p % state_block_size), 0, total_state_slots - 1
        )

    in_batch_prev = token_idx > query_start_loc[req]
    batch_prev = wkv_wgate[jnp.clip(token_idx - 1, 0, num_tokens - 1)]
    raw_prev = state_slots[_state_slot_for(req, jnp.maximum(pos - 1, 0))]
    cache_prev = jax.lax.bitcast_convert_type(raw_prev, jnp.float32).reshape(
        num_tokens, 1024
    )
    prev_kv_score = jnp.where(in_batch_prev[:, None], batch_prev, cache_prev)

    seq_end_idx = query_start_loc[req + 1] - 1
    write_mask = valid & (token_idx == seq_end_idx) & ((pos & 1) == 0)
    max_write = min(num_tokens, max(8, align_to(max_num_seqs, 8)))
    write_idx = jnp.nonzero(write_mask, size=max_write, fill_value=-1)[0].astype(
        jnp.int32
    )
    safe_write_idx = jnp.maximum(write_idx, 0)
    write_slots = jnp.where(
        write_idx >= 0,
        _state_slot_for(req[safe_write_idx], pos[safe_write_idx]),
        total_state_slots,
    )
    wkv_words = jax.lax.bitcast_convert_type(
        wkv_wgate[safe_write_idx].reshape(max_write, 8, 128), jnp.int32
    )
    state_slots = state_slots.at[write_slots].set(wkv_words, mode="drop")

    kv0, score0 = prev_kv_score[:, :head_dim], prev_kv_score[:, head_dim:]
    kv1, score1 = wkv_wgate[:, :head_dim], wkv_wgate[:, head_dim:]
    max_s = jnp.maximum(score0, score1)
    e0, e1 = jnp.exp(score0 - max_s), jnp.exp(score1 - max_s)
    pf32 = ((e0 * kv0 + e1 * kv1) / (e0 + e1)).astype(jnp.bfloat16).astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(pf32 * pf32, axis=-1, keepdims=True) + rms_eps)
    latent = (pf32 * rstd).astype(jnp.bfloat16) * norm_weight.astype(jnp.bfloat16)
    return latent, state_slots.reshape(state_cache.shape)


def compressor_insert_cache_jax(
    latent: jax.Array,  # [num_tokens, 512] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    nope_cache: jax.Array,  # [num_pages, page_size, 128] i32 (donated)
    rope_cache: jax.Array,  # [num_pages, page_size // 4, 128] i32 (donated)
    *,
    compress_ratio: int,
) -> tuple[jax.Array, jax.Array]:
    """Rotate latent, quantize NoPE (FP8) + pack RoPE (BF16), and store."""
    M = latent.shape[0]
    num_pages, page_size = nope_cache.shape[:2]
    total_kv_slots = num_pages * page_size

    pos = positions.astype(jnp.int32)
    _, _, is_boundary, kv_page, kv_page_off = _compressed_kv_positions(
        pos, kv_block_table, query_start_loc, distribution, page_size, compress_ratio
    )

    roped_f32 = _rope_block_start(latent, pos, cos_sin_cache, compress_ratio)
    tiled = roped_f32.reshape(M, 4, 128)
    q, scale = compute.quantize_fp8_lane_periodic(tiled, 64, 448)
    nope_slabs = compute.pack_nope_lane_periodic(q, scale, 448, 512, 128)
    nope_records = csa_cache_layout.slabs_to_words(nope_slabs)
    rope_records = csa_cache_layout.rope_words(tiled[:, 3, 64:].astype(jnp.bfloat16))

    kv_slot = jnp.clip(kv_page * page_size + kv_page_off, 0, total_kv_slots - 1)
    write_slot = jnp.where(is_boundary, kv_slot, total_kv_slots)
    nope_cache = (
        nope_cache.reshape((total_kv_slots, *nope_cache.shape[2:]))
        .at[write_slot]
        .set(nope_records, mode="drop")
        .reshape(nope_cache.shape)
    )

    rope_cache = _scatter_packed_4_jax(
        rope_cache.reshape(num_pages * (page_size // 4), csa_cache_layout.ROW_WORDS),
        rope_records,
        kv_slot,
        is_boundary,
        compress_ratio,
    ).reshape(rope_cache.shape)
    return nope_cache, rope_cache


def _scatter_packed_4_jax(
    cache_2d: jax.Array,  # [total_rows, 128] i32
    records: jax.Array,  # [num_tokens, 32] i32
    kv_slot: jax.Array,  # [num_tokens] i32
    is_boundary: jax.Array,  # [num_tokens] bool
    step: int,
) -> jax.Array:
    """Scatter [num_tokens, 32] RoPE word records into [total_rows, 128]."""
    num_tokens, total_rows = records.shape[0], cache_2d.shape[0]
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    row_id = kv_slot // 4
    prev_idx = jnp.maximum(token_idx - step, 0)
    same_as_prev = (
        (token_idx >= step)
        & is_boundary[prev_idx]
        & ((kv_slot[prev_idx] // 4) == row_id)
    )
    write_row = jnp.where(is_boundary & (~same_as_prev), row_id, total_rows)

    records_128 = jnp.tile(records, (1, 4))
    lane_q = (
        jnp.arange(csa_cache_layout.ROW_WORDS, dtype=jnp.int32)
        // csa_cache_layout.ROPE_WORDS
    )[None, :]
    tiles = cache_2d[jnp.clip(row_id, 0, total_rows - 1)]
    for k in range(4):
        idx = jnp.minimum(token_idx + k * step, num_tokens - 1)
        valid_k = is_boundary[idx] & ((kv_slot[idx] // 4) == row_id)
        mask_k = valid_k[:, None] & ((kv_slot[idx] % 4)[:, None] == lane_q)
        tiles = jnp.where(mask_k, records_128[idx], tiles)
    return cache_2d.at[write_row].set(tiles, mode="drop")


def indexer_produce_k_jax(
    k_normed: jax.Array,  # [num_tokens, 128] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    k_cache: jax.Array,  # [num_pages, 33, 4, page_size] u8 (donated)
    *,
    compress_ratio: int,
) -> jax.Array:
    """Rotate k_normed, quantize to 128 FP8 + E8M0 scale, and store."""
    M = k_normed.shape[0]
    num_tokens = positions.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    num_pages, sublane_groups, pack, page_size = k_cache.shape
    tiles_per_page = page_size // 128
    block_stride = max(1, (page_size * compress_ratio) // 1024)

    pos = positions.astype(jnp.int32)
    req, token_idx, is_boundary, kv_page, kv_page_off = _compressed_kv_positions(
        pos,
        kv_block_table,
        query_start_loc,
        distribution,
        page_size,
        compress_ratio,
        block_stride=block_stride,
    )
    pos_work = pos
    if compress_ratio > 1:
        # Compact to boundary tokens so the tile loop only visits written rows.
        M = min(
            num_tokens,
            max(128, align_to(cdiv(num_tokens, compress_ratio) + max_num_seqs, 128)),
        )
        owned_idx = jnp.nonzero(is_boundary, size=M, fill_value=-1)[0].astype(jnp.int32)
        safe_idx = jnp.maximum(owned_idx, 0)
        k_normed = k_normed[safe_idx]
        pos_work = pos[safe_idx]
        req = req[safe_idx]
        token_idx = jnp.arange(M, dtype=jnp.int32)
        is_boundary = (owned_idx >= 0) & is_boundary[safe_idx]
        kv_page = kv_page[safe_idx]
        kv_page_off = kv_page_off[safe_idx]

    roped_f32 = _rope_block_start(k_normed, pos_work, cos_sin_cache, compress_ratio)
    q, scale = compute.quantize_fp8_tiled(roped_f32[:, None, :], 128)
    q_u8 = jax.lax.bitcast_convert_type(q, jnp.uint8).reshape(M, 128)
    scale_u8 = jax.lax.bitcast_convert_type(scale, jnp.uint8).reshape(M, 1)
    pad_u8 = jnp.zeros((M, 3), dtype=jnp.uint8)
    k_records = jnp.concatenate([q_u8, scale_u8, pad_u8], axis=1).reshape(
        M, sublane_groups, pack
    )

    kv_page = jnp.clip(kv_page, 0, num_pages - 1)
    tile_id = kv_page * tiles_per_page + (kv_page_off // 128)
    lane_off = kv_page_off % 128
    valid_tile_id = jnp.where(is_boundary, tile_id, -1)

    prev_idx = jnp.maximum(token_idx - 1, 0)
    same_as_prev = (
        (token_idx >= 1)
        & is_boundary[prev_idx]
        & (req[prev_idx] == req)
        & (valid_tile_id[prev_idx] == valid_tile_id)
    )
    is_first = is_boundary & (~same_as_prev)

    max_seqs_with_1 = min(max_num_seqs, num_tokens)
    max_seqs_with_2 = min(max_num_seqs, num_tokens // (compress_ratio + 1))
    max_tiles = min(
        M,
        max_seqs_with_1 + max_seqs_with_2 + cdiv(num_tokens, 128 * compress_ratio),
    )
    first_tok_idxs = jnp.nonzero(is_first, size=max_tiles, fill_value=-1)[0].astype(
        jnp.int32
    )
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    lane0 = lane_off[safe_t0]
    req0 = req[safe_t0]

    padded_groups = align_to(sublane_groups + 3, 8)
    pad_groups = padded_groups - (sublane_groups + 3)
    k_records_i32 = jax.lax.bitcast_convert_type(k_records, jnp.int32)
    tok_words = jnp.concatenate(
        [
            k_records_i32,
            valid_tile_id[:, None],
            req[:, None],
            lane_off[:, None],
            jnp.zeros((M, pad_groups), dtype=jnp.int32),
        ],
        axis=1,
    )
    num_m_tiles = cdiv(M, 128)
    pad_m = (num_m_tiles + 1) * 128 - M
    pad_row = jnp.asarray(
        [0] * sublane_groups + [-1, -1, -1] + [0] * pad_groups, dtype=jnp.int32
    )[None, :]
    tok_words_padded = jnp.concatenate(
        [tok_words, jnp.broadcast_to(pad_row, (pad_m, padded_groups))], axis=0
    )
    tok_tiles = tok_words_padded.reshape(num_m_tiles + 1, 128, padded_groups).transpose(
        0, 2, 1
    )

    # SEQ_ALONG_LANE puts tokens on lanes, so each touched 128-token tile is
    # read, merged with the in-VMEM rolled token records, and written back.
    def _pallas_kernel(
        tile_ids_ref,
        safe_t0_ref,
        lane0_ref,
        req0_ref,
        n_tiles_ref,
        tok_tiles_vmem_ref,
        cache_in_hbm_ref,
        cache_out_hbm_ref,
        vmem_cache_ref,
        sem_ref,
    ):
        lanes_full = jax.lax.broadcasted_iota(jnp.int32, (padded_groups, 128), 1)
        lane_1d = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)

        @pl.loop(0, n_tiles_ref[0])
        def _step(i):
            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = lane0_ref[i]
            r0 = req0_ref[i]
            safe_tid = jnp.maximum(tid, 0)
            p, t_off = safe_tid // tiles_per_page, (safe_tid % tiles_per_page) * 128
            cr = pltpu.make_async_copy(
                cache_in_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                vmem_cache_ref,
                sem_ref.at[0],
            )
            cr.start()

            m_tile = t0 // 128
            m_off = t0 % 128
            tile0 = tok_tiles_vmem_ref[m_tile]
            tile1 = tok_tiles_vmem_ref[m_tile + 1]
            combined = jnp.where(lanes_full >= m_off, tile0, tile1)
            shift = jnp.bitwise_and(l0 - m_off, 127)
            rolled = pltpu.roll(combined, shift, axis=1)

            tok_tid = rolled[sublane_groups : sublane_groups + 1, :]
            tok_req = rolled[sublane_groups + 1 : sublane_groups + 2, :]
            tok_loff = rolled[sublane_groups + 2 : sublane_groups + 3, :]
            valid_mask = (
                (tid >= 0)
                & (lane_1d >= l0)
                & (tok_tid == tid)
                & (tok_req == r0)
                & (tok_loff == lane_1d)
            )
            upd_u8 = pltpu.bitcast(rolled[:sublane_groups, :], jnp.uint8).reshape(
                sublane_groups, pack, 128
            )
            cr.wait()
            vmem_cache_ref[...] = jnp.where(
                valid_mask[:, None, :], upd_u8, vmem_cache_ref[...]
            )
            cw = pltpu.make_async_copy(
                vmem_cache_ref,
                cache_out_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                sem_ref.at[1],
            )
            cw.start()
            cw.wait()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    vmem_tile = pltpu.VMEM((sublane_groups, pack, 128), jnp.uint8)
    return pl.pallas_call(
        _pallas_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[vmem_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=[vmem_tile, pltpu.SemaphoreType.DMA((2,))],
        ),
        out_shape=jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
        input_output_aliases={6: 0},
    )(
        tile_ids,
        safe_t0,
        lane0,
        req0,
        jnp.asarray([n_active_tiles], dtype=jnp.int32),
        tok_tiles,
        k_cache,
    )


def indexer_v41_jax(
    q: jax.Array,  # [num_tokens, n_head, 128] bf16
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,
    precomputed_eff_row_lengths: jax.Array | None = None,
    precomputed_idx_page_indices: jax.Array | None = None,
    precomputed_pass_num_steps: jax.Array | None = None,
    precomputed_pass_schedules: jax.Array | None = None,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
    has_precomputed_meta: bool = False,
) -> jax.Array:
    """Indexer top-k over the full compressed KV, encoded as CSA gather slots."""
    if has_precomputed_meta and cand_mode != "read":
        # Without a candidates input the 4 metadata arrays shift left by one slot.
        (
            precomputed_eff_row_lengths,
            precomputed_idx_page_indices,
            precomputed_pass_num_steps,
            precomputed_pass_schedules,
        ) = (
            candidates,
            precomputed_eff_row_lengths,
            precomputed_idx_page_indices,
            precomputed_pass_num_steps,
        )
        candidates = None

    q_quant, q_scales = rope_quant(
        q, positions, cos_sin_cache, quant_dtype=jnp.float8_e4m3fn
    )
    weights = (
        indexer_weights.astype(jnp.float32) * softmax_scale * (n_head**-0.5) * q_scales
    )
    num_pages, _, _, page_size = cache_kv.shape
    block_stride = max(1, (page_size * compression_ratio) // 1024)
    if precomputed_idx_page_indices is not None:
        idx_page_indices = precomputed_idx_page_indices
    else:
        idx_page_indices = (
            page_indices.reshape(seq_lens.shape[0], -1)[:, ::block_stride].reshape(-1)
            if block_stride > 1
            else page_indices.flatten()
        )
    # DSpark decode/verify batches (<= 512 tokens) run in MIXED; bq=8 avoids 16x
    # query padding and fits 2x larger KV blocks. 96/384 buckets use multiples of 48.
    is_decode_or_verify = q.shape[0] <= 512
    bq_mixed = 8 if is_decode_or_verify else 128
    bkv_pages = (16 if is_decode_or_verify else 8) * max(1, compression_ratio)
    chunk_tokens = (
        256
        if q.shape[0] >= 1024 and q.shape[0] % 256 == 0
        else {
            512: 128,
            384: 96,
            256: 64,
            192: 96,
            128: 64,
            96: 48,
            64: 32,
            16: 8,
        }.get(q.shape[0], 512)
    )
    dense = functools.partial(
        streamindex_topk,
        q=q_quant,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=idx_page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        precomputed_eff_row_lengths=precomputed_eff_row_lengths,
        precomputed_pass_num_steps=precomputed_pass_num_steps,
        precomputed_pass_schedules=precomputed_pass_schedules,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=bkv_pages,
        num_queries_per_block=bq_mixed,
        chunk_tokens=chunk_tokens,
    )
    cands = None
    if cand_mode is None:
        topk_indices = dense()
    else:
        live = jnp.arange(seq_lens.shape[0]) < distribution[2]
        max_visible = jnp.max(jnp.where(live, seq_lens, 0)) // compression_ratio
        use_candidates = max_visible > cand_blocks * cand_block_size
        if cand_mode == "write":
            topk_indices, cands = jax.lax.cond(
                use_candidates,
                lambda: dense(candidate_topk_blocks=cand_blocks),
                lambda: (
                    dense(),
                    jnp.full((q.shape[0], cand_blocks), -1, jnp.int32),
                ),
            )
        else:
            assert candidates is not None
            topk_indices = jax.lax.cond(
                use_candidates,
                lambda: dense(candidates=candidates),
                dense,
            )
    encoded = encode_csa_gather_indices(
        topk_indices,
        main_page_indices.flatten(),
        cu_q_lens,
        num_pages,
        page_size // block_stride,
    )
    return (encoded, cands) if cands is not None else encoded  # type: ignore[return-value]


def indexer_v41_write_jax(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return indexer_v41_jax(*args, **kwargs)  # type: ignore[return-value]


indexer_v41_write_jax.__signature__ = inspect.signature(indexer_v41_jax).replace(  # type: ignore[attr-defined]
    return_annotation=tuple[jax.Array, jax.Array]
)
