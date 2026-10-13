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

"""100% coverage unit tests for DeepSeek V4.1 compress_store_ref."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compress_store_ref,
    project_and_save_state_ref,
)


def _baseline_ref_compress_norm_rope_store(
    cache: jax.Array,
    rope_cache: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    block_table: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    cos_sin_cache: jax.Array,
    block_table_stride: int,
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    is_quantized: bool = True,
    has_rope: bool = True,
    has_rope_cache: bool | None = None,
    state_cache: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    cache_is_words = csa_cache_layout.is_word_array(cache.dtype)
    if cache_is_words:
        cache = csa_cache_layout.words_to_slabs(cache)
    if state_cache is not None and csa_cache_layout.is_word_array(state_cache.dtype):
        state_cache = csa_cache_layout.words_to_slabs(state_cache)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width

    if has_rope_cache is None:
        has_rope_cache = is_quantized and has_rope

    is_indexer_mode = (cache.shape[-1] == 256 and cache.shape[-2] == 4) or (
        cache.shape[-1] == 128 and cache.shape[-2] == 8
    )
    if is_indexer_mode:
        pack_factor = cache.shape[-2]
        last_dim = cache.shape[-1]
        unpacked_page_size = cache.shape[1] * pack_factor
        cache_for_unpack = cache.reshape(
            cache.shape[0], unpacked_page_size, 1, last_dim
        )
        num_pages, page_size, dummy, d2 = cache_for_unpack.shape
        assert dummy == 1
        slots_per_part_hbm = 0
        slot_bytes = d2
    else:
        cache_for_unpack = cache
        num_pages, page_size, slots_per_part_hbm, d2 = cache.shape
        slot_bytes = slots_per_part_hbm * d2

    state_source = cache if state_cache is None else state_cache
    phys_pages, phys_page_size, hbm_pack, last_dim = state_source.shape
    state_row_bytes = hbm_pack * last_dim
    state_rows = state_dim * 4 // state_row_bytes
    tokens_per_page = phys_page_size // state_rows
    state_bytes_t = state_source.reshape(
        phys_pages, tokens_per_page, state_rows, hbm_pack, last_dim
    ).transpose(0, 1, 2, 4, 3)
    state_view = jax.lax.bitcast_convert_type(state_bytes_t, jnp.float32).reshape(
        phys_pages, tokens_per_page, state_dim
    )

    window = coff * compress_ratio
    start = positions - window + 1
    w_idx = jnp.arange(window)
    pos = start[:, None] + w_idx[None, :]
    valid_mask = pos >= 0
    safe_pos = jnp.maximum(pos, 0)
    req = token_to_req_indices[:, None]
    block_numbers = block_table[req * block_table_stride + safe_pos // state_block_size]
    block_offsets = safe_pos % state_block_size
    head_offset = (w_idx >= compress_ratio).astype(jnp.int32) * head_dim
    col = head_offset[None, :, None] + jnp.arange(head_dim)[None, None, :]
    bn = block_numbers[:, :, None]
    bo = block_offsets[:, :, None]
    kv_window = state_view[bn, bo, col]
    score_window = state_view[bn, bo, state_width + col]

    compressed_pos = (positions // compress_ratio) * compress_ratio
    compressed = compress_store_ref.compress_norm_rope(
        kv_window=kv_window,
        score_window=score_window,
        valid_mask=valid_mask,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        compressed_pos=compressed_pos,
        rms_eps=rms_eps,
        rope_head_dim=rope_head_dim,
    )

    nope_dim = head_dim - rope_head_dim
    nope_store_dim = head_dim - rope_head_dim if has_rope_cache else head_dim
    if is_quantized:
        if is_indexer_mode:
            q_full, scale_full = compress_store_ref.quantize_fp8_ue8m0(
                compressed, quant_block
            )
        else:
            q_full, scale_full = compress_store_ref.quantize_fp8_lane_periodic(
                compressed, quant_block, nope_store_dim
            )
        q_bytes = jax.lax.bitcast_convert_type(q_full, jnp.uint8)
        scale_bytes = jax.lax.bitcast_convert_type(scale_full, jnp.uint8)
        q_bytes_flat = q_bytes.reshape(q_bytes.shape[0], -1)
        scale_bytes_flat = scale_bytes.reshape(scale_bytes.shape[0], -1)
        q_nope = q_bytes_flat[..., :nope_store_dim]
        if is_indexer_mode:
            nope_blocks = (nope_store_dim + quant_block - 1) // quant_block
            scale_nope = scale_bytes_flat[..., :nope_blocks]
        else:
            scale_nope = scale_bytes_flat
        nope_record = jnp.concatenate([q_nope, scale_nope], axis=-1)
    else:
        compressed_bf16 = compressed.astype(jnp.bfloat16)
        slots_per_part = head_dim // 128
        compressed_bf16_tiled = compressed_bf16.reshape(
            compressed_bf16.shape[0], slots_per_part, 128
        )
        compressed_bytes = jax.lax.bitcast_convert_type(
            compressed_bf16_tiled, jnp.uint8
        )
        compressed_bytes_split = compressed_bytes.reshape(
            compressed_bytes.shape[0], slots_per_part, 128, 2
        )
        compressed_bytes_t = compressed_bytes_split.transpose(0, 1, 3, 2)
        nope_record = compressed_bytes_t.reshape(compressed_bytes.shape[0], -1)

    total_bytes_out = (
        (256 if head_dim == 128 else 512) if is_quantized else head_dim * 2
    )
    slots_per_part_out = (
        total_bytes_out // slot_bytes
        if is_indexer_mode
        else total_bytes_out // (slots_per_part_hbm * 128)
    )
    nope_width = slots_per_part_out * slot_bytes
    nope_pad = nope_width - nope_record.shape[-1]
    if nope_pad > 0:
        nope_record = jnp.pad(nope_record, ((0, 0), (0, nope_pad)))
    nope_record_split = nope_record.reshape(
        nope_record.shape[0], slots_per_part_out, slot_bytes
    )

    page_size_physical = cache_for_unpack.shape[1]
    nope_num_slots_physical = num_pages * page_size_physical
    dest_nope_physical = compress_store_ref._boundary_dest(
        positions,
        slot_mapping,
        kv_slot_mapping,
        compress_ratio,
        nope_num_slots_physical,
    )
    flat_nope = cache_for_unpack.reshape(nope_num_slots_physical, slot_bytes)
    flat_nope_padded = jnp.concatenate(
        [flat_nope, jnp.zeros((slots_per_part_out, slot_bytes), dtype=jnp.uint8)],
        axis=0,
    )
    for o in range(slots_per_part_out):
        dest_o = jnp.where(
            dest_nope_physical < nope_num_slots_physical,
            dest_nope_physical + o,
            nope_num_slots_physical + o,
        )
        flat_nope_padded = flat_nope_padded.at[dest_o].set(nope_record_split[:, o])
    new_cache = (
        flat_nope_padded[:-slots_per_part_out]
        .reshape(cache_for_unpack.shape)
        .reshape(cache.shape)
    )
    if cache_is_words:
        new_cache = csa_cache_layout.slabs_to_words(new_cache)

    if not has_rope_cache:
        return new_cache, rope_cache
    rope_is_words = csa_cache_layout.is_word_array(rope_cache.dtype)
    rope_rows = (
        rope_cache if rope_is_words else csa_cache_layout.slabs_to_words(rope_cache)
    )
    rope = compressed[..., nope_dim:].astype(jnp.bfloat16)
    words = csa_cache_layout.rope_words(rope)
    flat_rope = rope_rows.reshape(-1, csa_cache_layout.ROPE_WORDS)
    rope_num_slots = flat_rope.shape[0]
    dest_rope = compress_store_ref._boundary_dest(
        positions, slot_mapping, kv_slot_mapping, compress_ratio, rope_num_slots
    )
    flat_rope = jnp.concatenate([flat_rope, jnp.zeros_like(flat_rope[:1])], axis=0)
    flat_rope = flat_rope.at[dest_rope].set(words)
    new_rope_rows = flat_rope[:-1].reshape(rope_rows.shape)
    if rope_is_words:
        return new_cache, new_rope_rows
    return new_cache, csa_cache_layout.words_to_slabs(new_rope_rows)


def test_interleaved_rope_and_validation_errors():
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (4, 128), dtype=jnp.float32)
    cos_sin = jax.random.normal(k2, (4, 32), dtype=jnp.float32)
    out = compress_store_ref.interleaved_rope(x, cos_sin, 32)
    assert out.shape == (4, 128)
    assert bool(jnp.array_equal(out[:, :96], x[:, :96]))

    with pytest.raises(ValueError, match="head_dim must be even"):
        compress_store_ref.interleaved_rope(jnp.zeros((2, 127)), cos_sin, 32)

    with pytest.raises(ValueError, match="rope_head_dim must be even"):
        compress_store_ref.interleaved_rope(x, jnp.zeros((4, 31)), 31)

    with pytest.raises(ValueError, match="rope_head_dim must be even"):
        compress_store_ref.interleaved_rope(x, jnp.zeros((4, 130)), 130)


def test_csa_words_and_slabs_modes_bit_exact():
    num_pages = 4
    page_size = 32
    head_dim = 512
    rope_head_dim = 64
    compress_ratio = 4
    overlap = True
    state_block_size = 2
    num_tokens = 8
    hidden_size = 256
    state_width = 2 * head_dim
    state_dim = 2 * state_width

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(42), 5)
    hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.float32)
    wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim), dtype=jnp.float32)
    ape = jax.random.normal(k3, (compress_ratio, state_width), dtype=jnp.float32)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    slot_mapping = positions * 16

    init_cache_u8 = jnp.zeros((num_pages, page_size, 4, 128), dtype=jnp.uint8)
    cache_u8 = project_and_save_state_ref.ref_wkv_proj_and_save_state(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        init_cache_u8,
        state_block_size,
        head_dim,
        compress_ratio,
        overlap,
    )
    cache_words = csa_cache_layout.slabs_to_words(cache_u8)

    rope_cache_u8 = jnp.zeros((num_pages, page_size // 4, 4, 128), dtype=jnp.uint8)
    rope_cache_words = jnp.zeros((num_pages, page_size // 4, 128), dtype=jnp.int32)

    block_table = jnp.arange(num_pages, dtype=jnp.int32)
    token_to_req_indices = jnp.zeros((num_tokens,), dtype=jnp.int32)
    kv_slot_mapping = jnp.array([-1, -1, -1, 100, -1, -1, -1, 101], dtype=jnp.int32)
    rms_weight = jax.random.normal(k4, (head_dim,), dtype=jnp.float32)
    cos_sin_cache = jax.random.normal(k5, (32, rope_head_dim), dtype=jnp.float32)

    # 1. Words mode (cache=int32, rope_cache=int32, has_rope_cache=None)
    cand_c_w, cand_r_w = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_words,
        rope_cache=rope_cache_words,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=None,
        state_cache=None,
    )
    ref_c_w, ref_r_w = _baseline_ref_compress_norm_rope_store(
        cache=cache_words,
        rope_cache=rope_cache_words,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=None,
        state_cache=None,
    )
    assert bool(jnp.array_equal(cand_c_w, ref_c_w))
    assert bool(jnp.array_equal(cand_r_w, ref_r_w))

    # 2. Slabs mode (cache=uint8, rope_cache=uint8, rms_weight=bf16)
    cand_c_u8, cand_r_u8 = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_u8,
        rope_cache=rope_cache_u8,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight.astype(jnp.bfloat16),
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=True,
        state_cache=None,
    )
    ref_c_u8, ref_r_u8 = _baseline_ref_compress_norm_rope_store(
        cache=cache_u8,
        rope_cache=rope_cache_u8,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight.astype(jnp.bfloat16),
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=True,
        state_cache=None,
    )
    assert bool(jnp.array_equal(cand_c_u8, ref_c_u8))
    assert bool(jnp.array_equal(cand_r_u8, ref_r_u8))


def test_hca_and_indexer_modes_and_error_branches():
    num_pages = 4
    head_dim = 512
    rope_head_dim = 64
    compress_ratio = 4
    state_block_size = 4
    num_tokens = 4
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)

    # HCA mode: is_quantized=False, overlap=False, separate state_cache, nope_pad == 0
    cache_hca = jnp.zeros((num_pages, 16, 4, 128), dtype=jnp.uint8)
    state_cache_hca = jax.random.randint(
        k1, (num_pages, 32, 4, 128), 0, 100, dtype=jnp.uint8
    )
    rope_cache_dummy = jnp.zeros((num_pages, 1, 1, 128), dtype=jnp.uint8)
    positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    slot_mapping = jnp.array([0, 8, 16, 24], dtype=jnp.int32)
    block_table = jnp.arange(num_pages, dtype=jnp.int32)
    token_to_req_indices = jnp.zeros((num_tokens,), dtype=jnp.int32)
    kv_slot_mapping = jnp.array([-1, -1, -1, 8], dtype=jnp.int32)
    rms_weight = jax.random.normal(k2, (head_dim,), dtype=jnp.float32)
    cos_sin_cache = jax.random.normal(k3, (16, rope_head_dim), dtype=jnp.float32)

    cand_hca, cand_rope_hca = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_hca,
        rope_cache=rope_cache_dummy,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=False,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=False,
        has_rope=True,
        has_rope_cache=False,
        state_cache=state_cache_hca,
    )
    ref_hca, ref_rope_hca = _baseline_ref_compress_norm_rope_store(
        cache=cache_hca,
        rope_cache=rope_cache_dummy,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=False,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=False,
        has_rope=True,
        has_rope_cache=False,
        state_cache=state_cache_hca,
    )
    assert bool(jnp.array_equal(cand_hca, ref_hca))
    assert bool(jnp.array_equal(cand_rope_hca, ref_rope_hca))

    # Indexer mode: head_dim=128, cache shape (num_pages, page_size, 4, 256)
    cache_idx = jax.random.randint(k1, (num_pages, 16, 4, 256), 0, 100, dtype=jnp.uint8)
    rms_weight_idx = jax.random.normal(k2, (128,), dtype=jnp.float32)
    cand_idx, _ = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_idx,
        rope_cache=rope_cache_dummy,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight_idx,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=128,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=True,
        rms_eps=1e-6,
        quant_block=128,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=False,
        state_cache=None,
    )
    ref_idx, _ = _baseline_ref_compress_norm_rope_store(
        cache=cache_idx,
        rope_cache=rope_cache_dummy,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight_idx,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=num_pages,
        state_block_size=state_block_size,
        head_dim=128,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=True,
        rms_eps=1e-6,
        quant_block=128,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=False,
        state_cache=None,
    )
    assert bool(jnp.array_equal(cand_idx, ref_idx))

    # Error 1: nope_pad < 0 (quant_block=128 -> 576B > 512B)
    with pytest.raises(ValueError, match="exceeds cache width"):
        compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_hca,
            rope_cache=jnp.zeros((num_pages, 4, 4, 128), dtype=jnp.uint8),
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=cos_sin_cache,
            block_table_stride=num_pages,
            state_block_size=1,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=compress_ratio,
            overlap=True,
            rms_eps=1e-6,
            quant_block=128,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=True,
            state_cache=None,
        )

    # Error 2: rope.shape[-1] != ROPE_DIM (64) when has_rope_cache=True
    with pytest.raises(ValueError, match="CSA RoPE is"):
        compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_hca,
            rope_cache=jnp.zeros((num_pages, 4, 4, 128), dtype=jnp.uint8),
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=jnp.zeros((16, 32), dtype=jnp.float32),
            block_table_stride=num_pages,
            state_block_size=1,
            head_dim=512,
            rope_head_dim=32,
            compress_ratio=compress_ratio,
            overlap=True,
            rms_eps=1e-6,
            quant_block=16,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=True,
            state_cache=None,
        )
