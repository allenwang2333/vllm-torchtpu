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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

_ROPE_GATHER_DN = jax.lax.GatherDimensionNumbers(
    offset_dims=(),
    collapsed_slice_dims=(1,),
    start_index_map=(1,),
    operand_batching_dims=(0,),
    start_indices_batching_dims=(0,),
)
_FP8_MAX: float = float(jnp.finfo(jnp.float8_e4m3fn).max)
_INV_FP8_MAX: float = 1.0 / _FP8_MAX


def interleaved_rope_vector(x, cos_val_32, sin_val_32):
    """Applies interleaved Rotary Position Embedding (RoPE) to a vector."""
    # x: (tile_n, 1, 128)
    # cos_val_32: (tile_n, 1, 32)
    # sin_val_32: (tile_n, 1, 32)
    tile_n = x.shape[0]
    x_dtype = x.dtype

    # We work with 2D tensors for gather to keep dimensions simple.
    x_2d = jax.lax.squeeze(x, (1,))  # (tile_n, 128)

    # swap adjacent pairs: [1, 0, 3, 2, ...]
    iota = jax.lax.broadcasted_iota(jnp.int32, (tile_n, 128, 1), 1)
    # (128,)
    swap_coords = jax.lax.bitwise_xor(iota, 1)  # (tile_n, 128, 1)

    x_swapped_2d = jax.lax.gather(
        x_2d,
        swap_coords,
        dimension_numbers=_ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    # cos_val_32/sin_val_32 are (tile_n, 1, 32). Squeeze to (tile_n, 32)
    cos_32 = jax.lax.squeeze(cos_val_32, (1,)).astype(x_dtype)
    sin_32 = jax.lax.squeeze(sin_val_32, (1,)).astype(x_dtype)

    ones_32 = jax.lax.full((tile_n, 32), 1, x_dtype)
    zeros_32 = jax.lax.full((tile_n, 32), 0, x_dtype)

    # pad to pairs: (tile_n, 64)
    cos_pairs = jax.lax.concatenate((ones_32, cos_32), 1)
    sin_pairs = jax.lax.concatenate((zeros_32, sin_32), 1)

    # duplicate indices: [0, 0, 1, 1, 2, 2, ..., 63, 63]
    dup_coords = jax.lax.shift_right_arithmetic(iota, 1)

    cos_dup_2d = jax.lax.gather(
        cos_pairs,
        dup_coords,
        dimension_numbers=_ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    sin_dup_2d = jax.lax.gather(
        sin_pairs,
        dup_coords,
        dimension_numbers=_ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    # alternate sin: [-1, 1, -1, 1, ...]
    iota_2d = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    alt_mask = (jax.lax.shift_left(jax.lax.bitwise_and(iota_2d, 1), 1) - 1).astype(
        x_dtype
    )  # (1, 128)
    sin_alt_2d = sin_dup_2d * alt_mask

    out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d

    return jax.lax.expand_dims(out_2d, (1,))


def quantize_fp8_tiled(x, block_size):
    # x: (tile_n, S, 128) f32
    tile_n, slots, width = x.shape  # (tile_n, S, 128) f32
    num_blocks = width // block_size

    if num_blocks == 1:
        amax = jax.lax.max(
            jnp.max(jax.lax.abs(x), axis=-1, keepdims=True), 1e-4
        )  # (tile_n, S, 1) f32
        log2_val = jax.lax.log2(amax * _INV_FP8_MAX)  # f32
        scale = jax.lax.exp2(jax.lax.ceil(log2_val))  # (tile_n, S, 1) f32
        q = (x * (1.0 / scale)).astype(jnp.float8_e4m3fn)  # (tile_n, S, 128) fp8
    else:
        qs = []
        scales = []
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            x_block = x[:, :, start:end]  # (tile_n, S, block_size) f32

            amax = jax.lax.max(
                jnp.max(jax.lax.abs(x_block), axis=-1, keepdims=True), 1e-4
            )  # (tile_n, S, 1) f32

            log2_val = jax.lax.log2(amax * _INV_FP8_MAX)  # f32
            scale = jax.lax.exp2(jax.lax.ceil(log2_val))  # (tile_n, S, 1) f32

            q_block = (x_block * (1.0 / scale)).astype(
                jnp.float8_e4m3fn
            )  # (tile_n, S, block_size) fp8

            qs.append(q_block)
            scales.append(scale)

        q = jax.lax.concatenate(qs, 2)  # (tile_n, S, 128) fp8
        scale = jax.lax.concatenate(scales, 2)  # (tile_n, S, num_blocks) f32

    # Bitcast workaround directly on f32 to extract exponent
    # f32 exponent is at bits 23-30. Shift right by 23 to align it.
    scale_u32 = pltpu.bitcast(scale, jnp.uint32)
    scale_exp = jax.lax.shift_right_logical(scale_u32, jnp.uint32(23))
    scale_u8 = scale_exp.astype(jnp.uint8)
    scale_f8 = pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)

    return q, scale_f8


def quantize_fp8_lane_periodic(x, period, nope_dim):
    """Quantize with one e8m0 scale per lane of a `period`-lane cycle.

    Group j is ``{d < nope_dim : d % period == j}``, so the scale a stored
    value needs is a function of its lane alone. The attention kernel then
    reads the scales as a tile it can rotate into place instead of expanding
    them with a one-hot matmul (see sparse_mla._dequant_dsv4_fp8).

    x: (tile_n, S, 128) f32, of which the first `nope_dim` flattened dims are
    the ones the record keeps.
    Returns (tile_n, S, 128) fp8 values and (tile_n, 1, period) e8m0 scales.
    """
    _, _, width = x.shape
    assert width == 2 * period, (width, period)

    dim = jax.lax.broadcasted_iota(
        jnp.int32, x.shape, 1
    ) * width + jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
    # Dims past the record's end are dropped by the packer; keeping them out
    # of the amax stops them widening a group they are not stored in.
    a = jax.lax.select(
        dim < nope_dim, jax.lax.abs(x), jax.lax.full(x.shape, 0.0, x.dtype)
    )
    folded = jax.lax.max(a[:, :, :period], a[:, :, period:])
    amax = jax.lax.max(jnp.max(folded, axis=1, keepdims=True), 1e-4)
    scale = jax.lax.exp2(
        jax.lax.ceil(jax.lax.log2(amax * _INV_FP8_MAX))
    )  # (tile_n, 1, period)

    q = (x * (1.0 / jax.lax.concatenate((scale, scale), 2))).astype(jnp.float8_e4m3fn)
    scale_u8 = jax.lax.shift_right_logical(
        pltpu.bitcast(scale, jnp.uint32), jnp.uint32(23)
    ).astype(jnp.uint8)
    return q, pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)


def pack_nope_lane_periodic(q, scale, nope_dim, nope_width_bytes, last_dim_size):
    """Pack `nope_dim` fp8 values followed by the e8m0 scales."""
    tile_n = q.shape[0]
    q_nope = pltpu.bitcast(q, jnp.uint8).reshape(tile_n, -1)[:, :nope_dim]
    scale_bytes = pltpu.bitcast(scale, jnp.uint8).reshape(tile_n, -1)
    record = jax.lax.concatenate((q_nope, scale_bytes), 1)
    assert record.shape[1] == nope_width_bytes, record.shape
    return record.reshape(tile_n, -1, last_dim_size)


def pack_nope_tiled(
    q, scale, nope_dim, block_size, nope_width_bytes=512, last_dim_size=128
):
    # q: (tile_n, S, 128) fp8
    # scale: (tile_n, S, num_blocks) e8m0 (uint8 bitcasted)
    tile_n, _, _ = q.shape

    # Bitcast to uint8
    q_bytes = pltpu.bitcast(q, jnp.uint8)
    scale_bytes = pltpu.bitcast(scale, jnp.uint8)

    # Flat representations
    q_flat = q_bytes.reshape(tile_n, -1)  # (tile_n, S * 128)
    scale_flat = scale_bytes.reshape(tile_n, -1)  # (tile_n, S * num_blocks)

    # Select NOPE parts
    q_nope = q_flat[:, :nope_dim] if nope_dim < q_flat.shape[1] else q_flat

    nope_blocks = (nope_dim + block_size - 1) // block_size
    if nope_blocks < scale_flat.shape[1]:
        scale_nope = scale_flat[:, :nope_blocks]
    else:
        scale_nope = scale_flat

    # Pad with zeros
    pad_size = nope_width_bytes - (nope_dim + nope_blocks)
    if pad_size > 0:
        zeros = jax.lax.full((tile_n, pad_size), 0, jnp.uint8)
        nope_record_padded = jax.lax.concatenate(
            (q_nope, scale_nope, zeros), 1
        )  # (tile_n, 512)
    else:
        nope_record_padded = jax.lax.concatenate((q_nope, scale_nope), 1)
    return nope_record_padded.reshape(tile_n, -1, last_dim_size)


def rope_words_tiled(rope_slot_ropped, rope_head_dim):
    """RoPE channels as cache words: low half bf16 ``k``, high half ``k + 32``.

    rope_slot_ropped: (tile_n, 1, 128) f32 holding the channels in its last
    ``rope_head_dim`` lanes. Returns (tile_n, 1, 128) int32 with word ``k`` at
    lane ``128 - rope_head_dim + k``; the lanes around it are junk.
    """
    bits = pltpu.bitcast(rope_slot_ropped.astype(jnp.bfloat16), jnp.uint16)
    bits = bits.astype(jnp.int32)
    half = rope_head_dim >> 1
    # Lane c picks up channel c + half as its high half.
    return jax.lax.bitwise_or(
        bits, jax.lax.shift_left(pltpu.roll(bits, bits.shape[-1] - half, 2), 16)
    )


def merge_rope_row(
    row,  # (1, 128) int32, the RoPE row as read from HBM
    kv_slots,  # (tile_n,) int, token index of every token in the tile
    words,  # rope_words_tiled(...)
    rope_lane,  # int, lane of RoPE word 0 in `words`
    n,  # int, the token whose row this is
):
    """Writes every tile token whose RoPE lands in token n's row into it.

    Token ``j`` of a row owns words ``[32 j, 32 j + 32)`` (see
    `csa_cache_layout`), so one rotation puts a token's words in its window.
    """
    per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
    width = csa_cache_layout.ROPE_WORDS
    lanes = row.shape[-1]
    window = jax.lax.div(jax.lax.broadcasted_iota(jnp.int32, row.shape, 1), width)
    window_masks = tuple(window == j for j in range(per_row))
    shifts = tuple((j * width - rope_lane) % lanes for j in range(per_row))
    curr_slot = kv_slots[n]
    curr_row = curr_slot // per_row
    curr_valid = curr_slot >= 0
    curr_pos = curr_slot % per_row

    for i in range(kv_slots.shape[0]):
        if i == n:
            hit = curr_valid
            pos_i = curr_pos
        else:
            slot_i = kv_slots[i]
            hit = (slot_i >= 0) & curr_valid & (slot_i // per_row == curr_row)
            pos_i = slot_i % per_row
        words_i = words[i]
        for j in range(per_row):
            placed = pltpu.roll(words_i, shifts[j], 1)
            row = jax.lax.select((hit & (pos_i == j)) & window_masks[j], placed, row)
    return row


def merge_slot_updates(
    slots_val,  # (pack_factor, physical_slot_size) uint8 (the row)
    kv_slots,  # (tile_n,) int (all slots in tile)
    val_padded,  # (tile_n, record_subslots, 128) uint8 (all values in tile)
    n,  # int (current token index)
):
    """multiple KV slots are packed into a single physical 512-byte row in HBM.

    Thus, different slots may map to the same tile.

    We only send the DMA from the first tile that updates a physical row,
    defined by `is_first_mask`.

    In this function, we:
    1. Read the current value of `slot_val`
    2. Scan all other tokens in the current tile
    3. Consolidate all other updates into the single tile.
    """
    tile_n = kv_slots.shape[0]
    curr_slot = kv_slots[n]
    pack_factor = slots_val.shape[0]

    curr_row = curr_slot // pack_factor
    curr_valid = curr_slot >= 0
    curr_sub_idx = curr_slot % pack_factor

    slots_list = [slots_val[i] for i in range(pack_factor)]

    for i in range(tile_n):
        if i == n:
            update_cond = curr_valid
            sub_idx = curr_sub_idx
        else:
            slot_i = kv_slots[i]
            valid_i = slot_i >= 0
            row_i = slot_i // pack_factor
            sub_idx = slot_i % pack_factor

            update_cond = valid_i & (row_i == curr_row) & curr_valid

        val = val_padded[i].reshape(-1)

        for target_s in range(pack_factor):
            slots_list[target_s] = jax.lax.select(
                update_cond & (sub_idx == target_s),
                val,
                slots_list[target_s],
            )

    return jnp.stack(slots_list)


def gather_from_page_buffer(
    page_buffer,
    positions_ref,
    kv_window_u8,
    score_window_u8,
    *,
    global_idx: int,
    num_tokens: int,
    window: int,
    block_size: int,
    pages_to_buffer_per_token: int,
    field_rows: int,
    state_rows_per_token: int,
    overlap: bool,
    is_indexer: bool = False,
):
    """Extracts kv_window, score_window from page buffer."""
    tile_n = page_buffer.shape[0]
    half_window = window // 2
    last_token_idx = num_tokens - 1
    page_offset = pages_to_buffer_per_token - 1
    if is_indexer:
        # CSA_INDEXER (layout 32x4x256)
        # page_buffer shape: (tile_n, pages_to_buffer, 32, 4, 256)
        for n in range(tile_n):
            safe_idx = jnp.minimum(global_idx + n, last_token_idx)
            pos = positions_ref[safe_idx]

            block_idx_curr = pos // block_size
            pos_start = pos - window + 1

            for w in range(window):
                pos_w = pos_start + w
                block_idx_w = pos_w // block_size
                p = block_idx_w - block_idx_curr + page_offset

                offset_in_block = pos_w % block_size
                token_row_start = offset_in_block * 2

                kv_row = token_row_start + 0
                score_row = token_row_start + 1

                valid_w = pos_w >= 0

                @pl.when(valid_w)
                def _():
                    row_kv = page_buffer[n, p, kv_row][...]  # shape (4, 256)
                    row_score = page_buffer[n, p, score_row][...]  # shape (4, 256)

                    # `proj_and_save_state` stores the f32 state byte-transposed
                    # (u8[s, l] == byte s of f32 l), so one (4, 256) row holds
                    # 256 floats across its LANES. The prev/curr halves of the
                    # overlapping state are therefore lanes 0:128 and 128:256.
                    if overlap:
                        if w < half_window:
                            val_kv = row_kv[:, 0:128]
                            val_score = row_score[:, 0:128]
                        else:
                            val_kv = row_kv[:, 128:256]
                            val_score = row_score[:, 128:256]
                    else:
                        val_kv = row_kv[:, 0:128]
                        val_score = row_score[:, 0:128]

                    kv_window_u8[n, w, 0, :, :] = val_kv
                    score_window_u8[n, w, 0, :, :] = val_score

    else:
        # Standard HCA/CSA (last dim 128)
        slots_per_part_head = kv_window_u8.shape[2]
        slots_per_part_row = field_rows
        slots_per_token_row = state_rows_per_token
        score_prev_start = 2 * slots_per_part_row
        score_curr_start = 3 * slots_per_part_row

        for n in range(tile_n):
            safe_idx = jnp.minimum(global_idx + n, last_token_idx)
            pos = positions_ref[safe_idx]

            block_idx_curr = pos // block_size
            pos_start = pos - window + 1

            @pl.loop(0, window)
            def body_w(w):
                pos_w = pos_start + w
                block_idx_w = pos_w // block_size
                p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

                if overlap:
                    is_prev = w < half_window
                    kv_slot_start_row = jax.lax.select(is_prev, 0, slots_per_part_row)
                    score_slot_start_row = jax.lax.select(
                        is_prev, score_prev_start, score_curr_start
                    )
                else:
                    kv_slot_start_row = 0
                    score_slot_start_row = slots_per_part_row

                offset_in_block = pos_w % block_size
                base_token_row = offset_in_block * slots_per_token_row
                kv_base_row = base_token_row + kv_slot_start_row
                score_base_row = base_token_row + score_slot_start_row

                @pl.loop(0, slots_per_part_head, unroll=True)
                def gather_loop(d_idx):
                    kv_src_row = kv_base_row + d_idx
                    score_src_row = score_base_row + d_idx

                    kv_window_u8[n, w, d_idx, :, :] = page_buffer[
                        n, p, kv_src_row, :, :
                    ]
                    score_window_u8[n, w, d_idx, :, :] = page_buffer[
                        n, p, score_src_row, :, :
                    ]
