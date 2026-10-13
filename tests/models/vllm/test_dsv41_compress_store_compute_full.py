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
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compute


def _ref_interleaved_rope_vector(
    x: jax.Array, cos_val_32: jax.Array, sin_val_32: jax.Array
) -> jax.Array:
    tile_n = x.shape[0]
    x_2d = jnp.squeeze(x, axis=1)
    iota = jnp.arange(128)
    swap_indices = jnp.bitwise_xor(iota, 1)
    swap_coords = jnp.broadcast_to(swap_indices, (tile_n, 128))[:, :, None]
    x_swapped_2d = jax.lax.gather(
        x_2d,
        swap_coords,
        dimension_numbers=compute._ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    cos_32 = jnp.squeeze(cos_val_32, axis=1).astype(x.dtype)
    sin_32 = jnp.squeeze(sin_val_32, axis=1).astype(x.dtype)
    ones_32 = jnp.ones((tile_n, 32), dtype=x.dtype)
    zeros_32 = jnp.zeros((tile_n, 32), dtype=x.dtype)
    cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
    sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)
    dup_indices = iota // 2
    dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]
    cos_dup_2d = jax.lax.gather(
        cos_pairs,
        dup_coords,
        dimension_numbers=compute._ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    sin_dup_2d = jax.lax.gather(
        sin_pairs,
        dup_coords,
        dimension_numbers=compute._ROPE_GATHER_DN,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]
    sin_alt_2d = sin_dup_2d * alt_mask
    out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d
    return out_2d[:, None, :]


def _install_cpu_pltpu_shims(monkeypatch):
    monkeypatch.setattr(compute.pltpu, "bitcast", jax.lax.bitcast_convert_type)
    monkeypatch.setattr(
        compute.pltpu,
        "roll",
        lambda x, shift, axis, **_kw: jnp.roll(x, shift, axis=axis),
    )


def test_interleaved_rope_and_words(monkeypatch):
    _install_cpu_pltpu_shims(monkeypatch)
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.standard_normal((4, 1, 128), dtype=np.float32))
    cos_val = jnp.asarray(rng.standard_normal((4, 1, 32), dtype=np.float32))
    sin_val = jnp.asarray(rng.standard_normal((4, 1, 32), dtype=np.float32))

    out = compute.interleaved_rope_vector(x, cos_val, sin_val)
    ref = _ref_interleaved_rope_vector(x, cos_val, sin_val)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))

    words = compute.rope_words_tiled(out, 64)
    assert words.shape == (4, 1, 128)
    assert words.dtype == jnp.int32


def test_quantize_and_pack_functions(monkeypatch):
    _install_cpu_pltpu_shims(monkeypatch)
    rng = np.random.default_rng(7)
    x_csa = jnp.asarray(rng.standard_normal((4, 5, 128), dtype=np.float32))
    q_lp, s_lp = compute.quantize_fp8_lane_periodic(x_csa, 64, 448)
    assert q_lp.shape == (4, 5, 128)
    assert s_lp.shape == (4, 1, 64)

    packed_lp = compute.pack_nope_lane_periodic(q_lp, s_lp, 448, 512, 128)
    assert packed_lp.shape == (4, 4, 128)

    # quantize_fp8_tiled: num_blocks == 1 and num_blocks == 2
    x_idx = jnp.asarray(rng.standard_normal((4, 1, 128), dtype=np.float32))
    q1, s1 = compute.quantize_fp8_tiled(x_idx, 128)
    assert q1.shape == (4, 1, 128)
    assert s1.shape == (4, 1, 1)

    q2, s2 = compute.quantize_fp8_tiled(x_csa, 64)
    assert q2.shape == (4, 5, 128)
    assert s2.shape == (4, 5, 2)

    # pack_nope_tiled:
    # 1) nope_dim < q_flat.shape[1], nope_blocks < scale_flat.shape[1], pad_size > 0
    p1 = compute.pack_nope_tiled(
        q2, s2, nope_dim=512, block_size=64, nope_width_bytes=528, last_dim_size=16
    )
    assert p1.shape == (4, 33, 16)

    # 2) nope_dim == q_flat.shape[1], nope_blocks == scale_flat.shape[1], pad_size == 0
    p2 = compute.pack_nope_tiled(
        q1, s1, nope_dim=128, block_size=128, nope_width_bytes=129, last_dim_size=129
    )
    assert p2.shape == (4, 1, 129)

    # 3) pad_size > 0 with full q_flat and scale_flat (CSA_INDEXER: 128 + 1 -> 256)
    p3 = compute.pack_nope_tiled(
        q1, s1, nope_dim=128, block_size=128, nope_width_bytes=256, last_dim_size=256
    )
    assert p3.shape == (4, 1, 256)


def test_merge_rope_row_and_slot_updates(monkeypatch):
    _install_cpu_pltpu_shims(monkeypatch)
    per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
    width = csa_cache_layout.ROPE_WORDS
    lanes = 128
    row = jnp.zeros((1, lanes), dtype=jnp.int32)
    kv_slots = jnp.array([4, 5, 12, -1], dtype=jnp.int32)
    words = jnp.arange(4 * 1 * 128, dtype=jnp.int32).reshape(4, 1, 128)
    rope_lane = 64

    merged_row = compute.merge_rope_row(row, kv_slots, words, rope_lane, 0)
    # Reference calculation
    window = jax.lax.broadcasted_iota(jnp.int32, row.shape, 1) // width
    curr_slot = kv_slots[0]
    curr_row = curr_slot // per_row
    curr_valid = curr_slot >= 0
    ref_row = row
    for i in range(kv_slots.shape[0]):
        slot_i = kv_slots[i]
        hit = (slot_i >= 0) & curr_valid & (slot_i // per_row == curr_row)
        pos_i = slot_i % per_row
        words_i = words[i]
        for j in range(per_row):
            placed = pltpu.roll(words_i, (j * width - rope_lane) % lanes, 1)
            ref_row = jnp.where(hit & (pos_i == j) & (window == j), placed, ref_row)
    np.testing.assert_array_equal(np.asarray(merged_row), np.asarray(ref_row))

    # merge_slot_updates
    slots_val = jnp.zeros((4, 256), dtype=jnp.uint8)
    val_padded = jnp.arange(4 * 2 * 128, dtype=jnp.uint8).reshape(4, 2, 128)
    merged_slots = compute.merge_slot_updates(slots_val, kv_slots, val_padded, 0)
    assert merged_slots.shape == (4, 256)
    np.testing.assert_array_equal(
        np.asarray(merged_slots[0]), np.asarray(val_padded[0].reshape(-1))
    )
    np.testing.assert_array_equal(
        np.asarray(merged_slots[1]), np.asarray(val_padded[1].reshape(-1))
    )


def test_gather_from_page_buffer_all_branches():
    tile_n = 2
    window = 4
    block_size = 8
    pages_to_buffer = 2

    # 1. is_indexer=True, overlap=True and overlap=False
    page_buf_idx = jnp.arange(
        tile_n * pages_to_buffer * 32 * 4 * 256, dtype=jnp.uint8
    ).reshape(tile_n, pages_to_buffer, 32, 4, 256)
    positions = jnp.array([2, 5], dtype=jnp.int32)

    for overlap in (True, False):

        def _kern_idx(page_ref, pos_ref, kv_out_ref, score_out_ref, ov=overlap):
            compute.gather_from_page_buffer(
                page_ref,
                pos_ref,
                kv_out_ref,
                score_out_ref,
                global_idx=0,
                num_tokens=2,
                window=window,
                block_size=block_size,
                pages_to_buffer_per_token=pages_to_buffer,
                field_rows=1,
                state_rows_per_token=2,
                overlap=ov,
                is_indexer=True,
            )

        out_shape = jax.ShapeDtypeStruct((tile_n, window, 1, 4, 128), jnp.uint8)
        kv_out, score_out = pl.pallas_call(
            _kern_idx,
            out_shape=(out_shape, out_shape),
            interpret=True,
        )(page_buf_idx, positions)
        assert kv_out.shape == (tile_n, window, 1, 4, 128)
        assert score_out.shape == (tile_n, window, 1, 4, 128)

    # 2. is_indexer=False, overlap=True and overlap=False
    page_buf_std = jnp.arange(
        tile_n * pages_to_buffer * 64 * 4 * 128, dtype=jnp.uint8
    ).reshape(tile_n, pages_to_buffer, 64, 4, 128)

    for overlap in (True, False):
        state_rows = 8 if overlap else 4

        def _kern_std(
            page_ref,
            pos_ref,
            kv_out_ref,
            score_out_ref,
            ov=overlap,
            sr=state_rows,
        ):
            compute.gather_from_page_buffer(
                page_ref,
                pos_ref,
                kv_out_ref,
                score_out_ref,
                global_idx=0,
                num_tokens=2,
                window=window,
                block_size=block_size,
                pages_to_buffer_per_token=pages_to_buffer,
                field_rows=2,
                state_rows_per_token=sr,
                overlap=ov,
                is_indexer=False,
            )

        out_shape = jax.ShapeDtypeStruct((tile_n, window, 2, 4, 128), jnp.uint8)
        kv_out, score_out = pl.pallas_call(
            _kern_std,
            out_shape=(out_shape, out_shape),
            interpret=True,
        )(page_buf_std, positions)
        assert kv_out.shape == (tile_n, window, 2, 4, 128)
        assert score_out.shape == (tile_n, window, 2, 4, 128)
