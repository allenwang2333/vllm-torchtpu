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
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.deepseek_v41.sparse_mla import (
    align_to,
    cdiv,
    encode_csa_gather_indices,
)
from vllm_torchtpu.kernels.deepseek_v41.streamindex_topk import (
    streamindex_topk,
)


def _seq_ids_from_locs(
    query_start_loc: jax.Array, num_tokens: int, eff_max_seqs: int
) -> jax.Array:
    """Map each token index in `[0, num_tokens)` to its sequence ID in `[0, eff_max_seqs)`."""
    q_ends = query_start_loc[1:eff_max_seqs]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    return jnp.sum((tok >= q_ends[None, :]).astype(jnp.int32), axis=1)


def _compact_indices(mask: jax.Array, size: int) -> jax.Array:
    """Compact True indices of 1D boolean `mask` into a `size`-length int32 array (-1 padded)."""
    n = mask.shape[0]
    vals = jnp.where(mask, jnp.arange(n, dtype=jnp.int32), n)
    if size > n:
        vals = jnp.pad(vals, (0, size - n), constant_values=n)
    s = jnp.sort(vals)[:size]
    return jnp.where(s < n, s, -1)


def _rope_block_start(
    x: jax.Array, pos: jax.Array, cos_sin_cache: jax.Array, compress_ratio: int
) -> jax.Array:
    """GPT-J RoPE on the trailing 64 channels at each token's block-start position."""
    x = x.astype(jnp.float32)
    start = jnp.clip(
        (pos // compress_ratio) * compress_ratio, 0, cos_sin_cache.shape[0] - 1
    )
    cs = cos_sin_cache[start]
    cos = jnp.repeat(cs[:, :32], 2, axis=-1)
    sin = jnp.repeat(cs[:, 32:], 2, axis=-1)
    x_nope, x_rope = x[:, :64], x[:, 64:]
    x_rot = jnp.stack([-x_rope[:, 1::2], x_rope[:, 0::2]], axis=-1).reshape(
        x.shape[0], 64
    )
    return jnp.concatenate([x_nope, x_rope * cos + x_rot * sin], axis=-1)


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
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
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
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))

    num_pages, page_size = state_cache.shape[:2]
    states_per_page = page_size // 8
    total_state_slots = num_pages * states_per_page
    state_slots = state_cache.reshape((total_state_slots, 8, *state_cache.shape[2:]))

    pos = positions.astype(jnp.int32)
    seq_ids = jnp.arange(eff_max_seqs, dtype=jnp.int32)
    q_start = query_start_loc[:eff_max_seqs]
    q_end = query_start_loc[1 : eff_max_seqs + 1]
    seq_valid = (seq_ids < distribution[2]) & (q_end > q_start)

    def _state_slot_for(r: jax.Array, p: jax.Array) -> jax.Array:
        page_idx = jnp.clip(p // state_block_size, 0, max_state_blocks - 1)
        page = state_block_table[r * max_state_blocks + page_idx]
        if num_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_pages - 1)) + 1, 0)
        return jnp.clip(
            page * states_per_page + (p % state_block_size), 0, total_state_slots - 1
        )

    first_tok_idx = jnp.clip(q_start, 0, num_tokens - 1)
    first_pos = pos[first_tok_idx]
    seq_read_slots = _state_slot_for(seq_ids, jnp.maximum(first_pos - 1, 0))
    seq_cache_prev = jax.lax.bitcast_convert_type(
        state_slots[seq_read_slots], jnp.float32
    ).reshape(eff_max_seqs, 1024)

    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    batch_prev = wkv_wgate[jnp.clip(token_idx - 1, 0, num_tokens - 1)]
    if num_tokens >= 2048:
        first_write_idx = jnp.where(seq_valid, first_tok_idx, num_tokens)
        prev_kv_score = batch_prev.at[first_write_idx].set(seq_cache_prev, mode="drop")
    else:
        req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
        in_batch_prev = token_idx > query_start_loc[req]
        prev_kv_score = jnp.where(
            in_batch_prev[:, None], batch_prev, seq_cache_prev[req]
        )

    last_tok_idx = jnp.clip(q_end - 1, 0, num_tokens - 1)
    last_pos = pos[last_tok_idx]
    should_write = seq_valid & ((last_pos & 1) == 0)
    write_slots = jnp.where(
        should_write,
        _state_slot_for(seq_ids, last_pos),
        total_state_slots,
    )
    wkv_words = jax.lax.bitcast_convert_type(
        wkv_wgate[last_tok_idx].reshape(eff_max_seqs, 8, 128), jnp.int32
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
    num_tokens = latent.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    num_pages, page_size = nope_cache.shape[:2]
    total_kv_slots = num_pages * page_size

    pos = positions.astype(jnp.int32)
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = valid & (((pos + 1) % compress_ratio) == 0)

    use_pallas = num_tokens > 512 and page_size % 32 == 0
    M = num_tokens
    latent_work = latent
    pos_work = pos
    if use_pallas and compress_ratio > 1:
        M = min(
            num_tokens,
            max(32, align_to((num_tokens + eff_max_seqs) // compress_ratio, 32)),
        )
        owned_idx = _compact_indices(is_boundary, M)
        safe_idx = jnp.maximum(owned_idx, 0)
        latent_work = latent[safe_idx]
        pos_work = pos[safe_idx]
        req = req[safe_idx]
        token_idx = jnp.arange(M, dtype=jnp.int32)
        is_boundary = owned_idx >= 0

    kv_idx = jnp.maximum(pos_work // compress_ratio, 0)
    kv_page_idx = jnp.clip(kv_idx // page_size, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]
    kv_slot = jnp.clip(kv_page * page_size + kv_page_off, 0, total_kv_slots - 1)

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    f32_work = latent_work.astype(jnp.float32)
    r0 = f32_work[:, :128]
    r1 = f32_work[:, 128:256]
    r2 = f32_work[:, 256:384]
    r3_nope = f32_work[:, 384:448]
    x_rope = f32_work[:, 448:512]
    m012 = jnp.maximum(jnp.maximum(jnp.abs(r0), jnp.abs(r1)), jnp.abs(r2))
    folded = jnp.maximum(jnp.maximum(m012[:, :64], m012[:, 64:]), jnp.abs(r3_nope))
    amax = jnp.clip(folded, 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    inv_s = 1.0 / scale
    inv_s128 = jnp.concatenate([inv_s, inv_s], axis=-1)
    b0 = jax.lax.bitcast_convert_type(
        (r0 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b1 = jax.lax.bitcast_convert_type(
        (r1 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b2 = jax.lax.bitcast_convert_type(
        (r2 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_lo = jax.lax.bitcast_convert_type(
        (r3_nope * inv_s).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_hi = (jax.lax.bitcast_convert_type(scale, jnp.uint32) >> 23) & 0xFF
    b3 = jnp.concatenate([b3_lo, b3_hi], axis=-1)
    nope_records = jax.lax.bitcast_convert_type(
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), jnp.int32
    )

    start = jnp.clip(
        (pos_work // compress_ratio) * compress_ratio, 0, cos_sin_cache.shape[0] - 1
    )
    cs = cos_sin_cache[start]
    cos = jnp.repeat(cs[:, :32], 2, axis=-1)
    sin = jnp.repeat(cs[:, 32:], 2, axis=-1)
    x_rot = jnp.stack([-x_rope[:, 1::2], x_rope[:, 0::2]], axis=-1).reshape(M, 64)
    y_rope = (x_rope * cos + x_rot * sin).astype(jnp.bfloat16)
    rope_records = csa_cache_layout.rope_words(y_rope)

    if not use_pallas:
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

    tile_k = 256 if (num_tokens >= 1024 and page_size % 256 == 0) else 32
    rope_tile = tile_k // 4
    tiles_per_page = page_size // tile_k
    tile_id = kv_slot // tile_k
    slot_off = kv_slot % tile_k
    valid_tile_id = jnp.where(is_boundary, tile_id, -1)

    prev_idx = jnp.maximum(token_idx - 1, 0)
    same_as_prev = (
        (token_idx >= 1)
        & is_boundary[prev_idx]
        & (req[prev_idx] == req)
        & (valid_tile_id[prev_idx] == valid_tile_id)
        & (slot_off == slot_off[prev_idx] + 1)
    )
    is_first = is_boundary & (~same_as_prev)

    max_seqs_with_1 = min(eff_max_seqs, M)
    max_seqs_with_2 = min(eff_max_seqs, M // 2)
    max_tiles = min(M, max_seqs_with_1 + max_seqs_with_2 + cdiv(M, tile_k))

    first_tok_idxs = _compact_indices(is_first, max_tiles)
    num_valid_b = jnp.sum(is_boundary.astype(jnp.int32))
    next_first = jnp.concatenate(
        [first_tok_idxs[1:], jnp.asarray([-1], dtype=jnp.int32)]
    )
    last_tok_idxs = jnp.where(
        first_tok_idxs >= 0,
        jnp.where(next_first >= 0, next_first - 1, num_valid_b - 1),
        -1,
    )
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    safe_t1 = jnp.maximum(last_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    slot0 = slot_off[safe_t0]
    slot_end = slot_off[safe_t1] + 1
    is_full = jnp.where(
        (first_tok_idxs >= 0) & (slot0 == 0) & (slot_end == tile_k), 1, 0
    ).astype(jnp.int32)

    num_m_tiles = cdiv(M, tile_k)
    pad_m = (num_m_tiles + 1) * tile_k - M
    nope_padded = jnp.pad(nope_records, ((0, pad_m), (0, 0)))
    nope_tiles = nope_padded.reshape(num_m_tiles + 1, tile_k, 128)
    rope_padded = jnp.pad(rope_records, ((0, pad_m), (0, 0)))
    rope_tiles = rope_padded.reshape(num_m_tiles + 1, rope_tile, 128)

    def _pallas_kernel(
        tile_ids_ref,
        safe_t0_ref,
        slot0_ref,
        slot_end_ref,
        is_full_ref,
        n_tiles_ref,
        nope_tiles_vmem_ref,
        rope_tiles_vmem_ref,
        nope_in_hbm_ref,
        rope_in_hbm_ref,
        nope_out_hbm_ref,
        rope_out_hbm_ref,
        vmem_nope_ref,
        vmem_rope_ref,
        sem_ref,
    ):
        sublanes_k = jax.lax.broadcasted_iota(jnp.int32, (tile_k, 128), 0)
        row_r = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 0)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 1) // 32
        slot_rope = row_r * 4 + q_iota
        n_tiles = n_tiles_ref[0]

        def _tile_hbm_slice(idx):
            tid = tile_ids_ref[idx]
            safe_tid = jnp.maximum(tid, 0)
            p = safe_tid // tiles_per_page
            nope_off = (safe_tid % tiles_per_page) * tile_k
            rope_off = (safe_tid % tiles_per_page) * rope_tile
            return p, nope_off, rope_off

        def _start_read(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                vmem_nope_ref.at[buf],
                sem_ref.at[0, buf],
            ).start()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                vmem_rope_ref.at[buf],
                sem_ref.at[1, buf],
            ).start()

        def _wait_read(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                vmem_nope_ref.at[buf],
                sem_ref.at[0, buf],
            ).wait()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                vmem_rope_ref.at[buf],
                sem_ref.at[1, buf],
            ).wait()

        def _start_write(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_nope_ref.at[buf],
                nope_out_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                sem_ref.at[2, buf],
            ).start()
            pltpu.make_async_copy(
                vmem_rope_ref.at[buf],
                rope_out_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                sem_ref.at[3, buf],
            ).start()

        def _wait_write(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_nope_ref.at[buf],
                nope_out_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                sem_ref.at[2, buf],
            ).wait()
            pltpu.make_async_copy(
                vmem_rope_ref.at[buf],
                rope_out_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                sem_ref.at[3, buf],
            ).wait()

        @pl.when(n_tiles > 0)
        def _prologue():
            @pl.when(is_full_ref[0] == 0)
            def _read0():
                _start_read(0, 0)

        @pl.loop(0, n_tiles)
        def _step(i):
            cur_buf = i & 1
            nxt_buf = 1 - cur_buf

            @pl.when(i + 1 < n_tiles)
            def _prep_next():
                @pl.when(i >= 1)
                def _wait_prev_write_on_nxt():
                    _wait_write(i - 1, nxt_buf)

                @pl.when(is_full_ref[i + 1] == 0)
                def _prefetch_next():
                    _start_read(i + 1, nxt_buf)

            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = slot0_ref[i]
            l_end = slot_end_ref[i]
            full = is_full_ref[i]

            m_tile = t0 // tile_k
            m_off = t0 % tile_k
            shift = jnp.bitwise_and(l0 - m_off, tile_k - 1)

            n0 = nope_tiles_vmem_ref[m_tile]
            n1 = nope_tiles_vmem_ref[m_tile + 1]
            comb_nope = jnp.where(sublanes_k >= m_off, n0, n1)
            upd_nope = pltpu.roll(comb_nope, shift, axis=0)

            r0 = rope_tiles_vmem_ref[m_tile]
            r1 = rope_tiles_vmem_ref[m_tile + 1]
            comb_rope = jnp.where(slot_rope >= m_off, r0, r1)
            d_r = shift >> 2
            d_q = jnp.bitwise_and(shift, 3)
            lane_sh = pltpu.roll(comb_rope, d_q * 32, axis=1)
            sh0 = pltpu.roll(lane_sh, d_r, axis=0)
            sh1 = pltpu.roll(lane_sh, jnp.bitwise_and(d_r + 1, rope_tile - 1), axis=0)
            upd_rope = jnp.where(q_iota < d_q, sh1, sh0)

            @pl.when(full == 1)
            def _store_full():
                vmem_nope_ref.at[cur_buf][...] = upd_nope
                vmem_rope_ref.at[cur_buf][...] = upd_rope

            @pl.when(full == 0)
            def _store_partial():
                valid_nope = (tid >= 0) & (sublanes_k >= l0) & (sublanes_k < l_end)
                valid_rope = (tid >= 0) & (slot_rope >= l0) & (slot_rope < l_end)
                _wait_read(i, cur_buf)
                vmem_nope_ref.at[cur_buf][...] = jnp.where(
                    valid_nope, upd_nope, vmem_nope_ref.at[cur_buf][...]
                )
                vmem_rope_ref.at[cur_buf][...] = jnp.where(
                    valid_rope, upd_rope, vmem_rope_ref.at[cur_buf][...]
                )

            _start_write(i, cur_buf)

        @pl.when(n_tiles >= 2)
        def _epilogue_prev():
            _wait_write(n_tiles - 2, (n_tiles - 2) & 1)

        @pl.when(n_tiles >= 1)
        def _epilogue_last():
            _wait_write(n_tiles - 1, (n_tiles - 1) & 1)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _pallas_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1,),
            in_specs=[vmem_spec, vmem_spec, hbm_spec, hbm_spec],
            out_specs=[hbm_spec, hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((2, tile_k, 128), jnp.int32),
                pltpu.VMEM((2, rope_tile, 128), jnp.int32),
                pltpu.SemaphoreType.DMA((4, 2)),
            ],
        ),
        out_shape=(
            jax.ShapeDtypeStruct(nope_cache.shape, nope_cache.dtype),
            jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
        ),
        input_output_aliases={8: 0, 9: 1},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"compressor_insert_cache_r{compress_ratio}",
    )(
        tile_ids,
        safe_t0,
        slot0,
        slot_end,
        is_full,
        jnp.asarray([n_active_tiles], dtype=jnp.int32),
        nope_tiles,
        rope_tiles,
        nope_cache,
        rope_cache,
    )


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
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    num_pages, sublane_groups, pack, page_size = k_cache.shape
    tiles_per_page = page_size // 128
    block_stride = max(1, (page_size * compress_ratio) // 1024)

    pos = positions.astype(jnp.int32)
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = valid & (((pos + 1) % compress_ratio) == 0)

    pos_work = pos
    if compress_ratio > 1:
        # Compact to boundary tokens so the tile loop only visits written rows.
        M = min(
            num_tokens,
            max(128, align_to((num_tokens + eff_max_seqs) // compress_ratio, 128)),
        )
        owned_idx = _compact_indices(is_boundary, M)
        safe_idx = jnp.maximum(owned_idx, 0)
        k_normed = k_normed[safe_idx]
        pos_work = pos[safe_idx]
        req = req[safe_idx]
        token_idx = jnp.arange(M, dtype=jnp.int32)
        is_boundary = owned_idx >= 0

    kv_idx = jnp.maximum(pos_work // compress_ratio, 0)
    kv_page_idx = jnp.clip((kv_idx // page_size) * block_stride, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]

    roped_f32 = _rope_block_start(k_normed, pos_work, cos_sin_cache, compress_ratio)
    q, scale = compute.quantize_fp8_tiled(roped_f32[:, None, :], 128)
    q_i32 = jax.lax.bitcast_convert_type(q.reshape(M, 32, 4), jnp.int32)
    scale_i32 = (
        jax.lax.bitcast_convert_type(scale, jnp.uint8).astype(jnp.int32).reshape(M, 1)
    )
    k_records_i32 = jnp.concatenate([q_i32, scale_i32], axis=1)

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

    max_seqs_with_1 = min(eff_max_seqs, num_tokens)
    max_seqs_with_2 = min(eff_max_seqs, num_tokens // (compress_ratio + 1))
    max_tiles = min(
        M,
        max_seqs_with_1 + max_seqs_with_2 + cdiv(num_tokens, 128 * compress_ratio),
    )
    first_tok_idxs = _compact_indices(is_first, max_tiles)
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    lane0 = lane_off[safe_t0]
    req0 = req[safe_t0]

    padded_groups = align_to(sublane_groups + 3, 8)
    pad_groups = padded_groups - (sublane_groups + 3)
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
        n_tiles = n_tiles_ref[0]

        def _tile_hbm_slice(idx):
            tid = tile_ids_ref[idx]
            safe_tid = jnp.maximum(tid, 0)
            p = safe_tid // tiles_per_page
            t_off = (safe_tid % tiles_per_page) * 128
            return p, t_off

        def _start_read(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                cache_in_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                vmem_cache_ref.at[buf],
                sem_ref.at[0, buf],
            ).start()

        def _wait_read(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                cache_in_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                vmem_cache_ref.at[buf],
                sem_ref.at[0, buf],
            ).wait()

        def _start_write(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_cache_ref.at[buf],
                cache_out_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                sem_ref.at[1, buf],
            ).start()

        def _wait_write(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_cache_ref.at[buf],
                cache_out_hbm_ref.at[p, :, :, pl.ds(t_off, 128)],
                sem_ref.at[1, buf],
            ).wait()

        @pl.when(n_tiles > 0)
        def _prologue():
            _start_read(0, 0)

        @pl.loop(0, n_tiles)
        def _step(i):
            cur_buf = i & 1
            nxt_buf = 1 - cur_buf

            @pl.when(i + 1 < n_tiles)
            def _prefetch_next():
                @pl.when(i >= 1)
                def _wait_prev_write_on_nxt():
                    _wait_write(i - 1, nxt_buf)

                _start_read(i + 1, nxt_buf)

            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = lane0_ref[i]
            r0 = req0_ref[i]

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
            _wait_read(i, cur_buf)
            vmem_cache_ref.at[cur_buf][...] = jnp.where(
                valid_mask[:, None, :], upd_u8, vmem_cache_ref.at[cur_buf][...]
            )
            _start_write(i, cur_buf)

        @pl.when(n_tiles >= 2)
        def _epilogue_prev():
            _wait_write(n_tiles - 2, (n_tiles - 2) & 1)

        @pl.when(n_tiles >= 1)
        def _epilogue_last():
            _wait_write(n_tiles - 1, (n_tiles - 1) & 1)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    vmem_tile = pltpu.VMEM((2, sublane_groups, pack, 128), jnp.uint8)
    return pl.pallas_call(
        _pallas_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[vmem_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=[vmem_tile, pltpu.SemaphoreType.DMA((2, 2))],
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


def _indexer_v41_impl(
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
    cfg_num_tokens: int | None = None,
    use_candidates_override: jax.Array | None = None,
    token_start: jax.Array | int = 0,
) -> jax.Array:
    """Indexer top-k over the full compressed KV, encoded as CSA gather slots.

    cfg_num_tokens: token count that selects the query/KV block sizes (defaults
    to q.shape[0]); the token-parallel path passes the full batch size so a
    prefill slice keeps the prefill blocks. Chunking follows q.shape[0].
    use_candidates_override: batch-wide candidate predicate (token-parallel path).
    """
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

    w_scale = (
        indexer_weights.astype(jnp.float32) * softmax_scale * (n_head**-0.5)
    )
    q_quant, q_scales = rope_quant(
        q, positions, cos_sin_cache, w_scale=w_scale, quant_dtype=jnp.float8_e4m3fn
    )
    weights = q_scales
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
    n_cfg = q.shape[0] if cfg_num_tokens is None else cfg_num_tokens
    # DSpark decode/verify batches (<= 512 tokens) run in MIXED; bq=8 avoids 16x
    # query padding and fits 2x larger KV blocks. 96/384 buckets use multiples of 48.
    is_decode_or_verify = n_cfg <= 512
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
        use_candidates = (
            max_visible > cand_blocks * cand_block_size
            if use_candidates_override is None
            else use_candidates_override
        )
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
        token_start=token_start,
    )
    return (encoded, cands) if cands is not None else encoded  # type: ignore[return-value]


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
    """torch.library.custom_op-safe wrapper around _indexer_v41_impl."""
    return _indexer_v41_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        main_page_indices,
        candidates,
        precomputed_eff_row_lengths,
        precomputed_idx_page_indices,
        precomputed_pass_num_steps,
        precomputed_pass_schedules,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        cand_mode=cand_mode,
        cand_blocks=cand_blocks,
        cand_block_size=cand_block_size,
        has_precomputed_meta=has_precomputed_meta,
    )


def indexer_v41_write_jax(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return indexer_v41_jax(*args, **kwargs)  # type: ignore[return-value]


indexer_v41_write_jax.__signature__ = inspect.signature(indexer_v41_jax).replace(  # type: ignore[attr-defined]
    return_annotation=tuple[jax.Array, jax.Array]
)


def tp_local_indexer_batch(
    rank: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq] (or flat)
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    main_page_indices: jax.Array,  # i32[max_num_seqs, main_pages_per_seq] (or flat)
    *,
    num_local_tokens: int,
    num_local_rows: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Tokens [rank * T_l, (rank + 1) * T_l) of an indexer batch as a standalone batch.

    Every live request has >= 1 token, so at most T_l requests overlap the slice;
    they are contiguous in request order and keep it. A request cut at the slice
    end gets a shorter seq_len, so that token_pos = seq_len - q_len + offset (the
    kernel's position rule) is unchanged for every local token.

    num_local_rows (>= T_l, default T_l) sizes the local batch; the extra request
    slots are empty (seq_len 0, q_len 0), like the tail of a padded decode bucket.
    """
    t_l = num_local_tokens
    lo = rank[0] * t_l
    hi = lo + t_l
    max_num_seqs = seq_lens.shape[0]
    # A slice of t_l tokens spans at most min(t_l, max_num_seqs) requests.
    t_w = min(t_l if num_local_rows is None else num_local_rows, max_num_seqs)
    live = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
    q_start, q_end = cu_q_lens[:-1], cu_q_lens[1:]
    req_lo = jnp.sum(live & (q_end <= lo)).astype(jnp.int32)
    req_hi = jnp.sum(live & (q_start < hi)).astype(jnp.int32)

    def window(x: jax.Array, size: int, mode: str = "constant") -> jax.Array:
        pad = [(0, t_w)] + [(0, 0)] * (x.ndim - 1)
        return jax.lax.dynamic_slice_in_dim(jnp.pad(x, pad, mode=mode), req_lo, size)

    sl = window(seq_lens, t_w)
    cq = window(cu_q_lens, t_w + 1, mode="edge")
    pi = window(page_indices.reshape(max_num_seqs, -1), t_w)
    mpi = window(main_page_indices.reshape(max_num_seqs, -1), t_w)
    q_end_l = cq[1:]
    active = jnp.arange(t_w, dtype=jnp.int32) < req_hi - req_lo
    sl_local = jnp.where(active, sl - (q_end_l - jnp.minimum(q_end_l, hi)), 0)
    cu_local = jnp.clip(cq - lo, 0, t_l)
    dist_local = jnp.clip(distribution, req_lo, req_hi) - req_lo
    return (
        sl_local.astype(seq_lens.dtype),
        pi.reshape(-1),
        cu_local.astype(cu_q_lens.dtype),
        dist_local.astype(distribution.dtype),
        mpi.reshape(-1),
    )


def indexer_v41_tp_jax(
    rank: jax.Array,  # i32[1], this TP rank
    q: jax.Array,  # [T / tp, n_head, 128] bf16, this rank's tokens
    positions: jax.Array,  # [T / tp]
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,  # [T / tp, n_head]
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,  # [T, cand_blocks], all tokens
    *,
    tp_size: int,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
) -> jax.Array:
    """Token-parallel indexer: top-k for this rank's T/tp tokens only.

    The caller all-gathers the [T/tp, k] result. The K cache is replicated, so
    each token's top-k is the same as in the replicated path.
    """
    t_l = q.shape[0]
    # Decode configs run 8-query blocks (bq_mixed). A rank with fewer tokens
    # (T/tp = 2 or 4) runs as an 8-row local batch whose tail rows belong to no
    # request, like a padded decode bucket, and returns only its T/tp rows.
    t_rows = max(8, -(-t_l // 8) * 8)
    sl, pi, cq, dist, mpi = tp_local_indexer_batch(
        rank,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        main_page_indices,
        num_local_tokens=t_l,
        num_local_rows=t_rows,
    )
    use_candidates = None
    if cand_mode is not None:
        # Same predicate as the replicated path, over the whole batch.
        live = jnp.arange(seq_lens.shape[0]) < distribution[2]
        max_visible = jnp.max(jnp.where(live, seq_lens, 0)) // compression_ratio
        use_candidates = max_visible > cand_blocks * cand_block_size
        if cand_mode == "read":
            assert candidates is not None
            if candidates.shape[0] != t_l:
                candidates = jax.lax.dynamic_slice_in_dim(candidates, rank[0] * t_l, t_l)
    if t_rows != t_l:

        def pad_rows(x: jax.Array, **kw) -> jax.Array:
            return jnp.pad(x, [(0, t_rows - t_l)] + [(0, 0)] * (x.ndim - 1), **kw)

        q = pad_rows(q, mode="edge")
        positions = pad_rows(positions, mode="edge")
        indexer_weights = pad_rows(indexer_weights, mode="edge")
        if candidates is not None:
            candidates = pad_rows(candidates, constant_values=-1)
    out = _indexer_v41_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        sl,
        pi,
        cq,
        dist,
        mpi,
        candidates,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        cand_mode=cand_mode,
        cand_blocks=cand_blocks,
        cand_block_size=cand_block_size,
        cfg_num_tokens=t_l * tp_size,
        use_candidates_override=use_candidates,
        token_start=rank[0] * t_l,
    )
    if t_rows == t_l:
        return out
    if isinstance(out, tuple):
        return tuple(x[:t_l] for x in out)  # type: ignore[return-value]
    return out[:t_l]


def indexer_v41_tp_write_jax(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return indexer_v41_tp_jax(*args, **kwargs)  # type: ignore[return-value]


indexer_v41_tp_write_jax.__signature__ = inspect.signature(  # type: ignore[attr-defined]
    indexer_v41_tp_jax
).replace(return_annotation=tuple[jax.Array, jax.Array])
