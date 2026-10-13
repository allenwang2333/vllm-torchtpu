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

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1


def _ref_derive_metadata(
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    compress_ratio: int,
    state_block_size: int,
    head_dim: int,
    overlap: bool,
    cos_sin_cache: jax.Array | None,
    state_cache: jax.Array | None = None,
):
    num_tokens = positions.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0
    state_source = cache if state_cache is None else state_cache

    cfgs = compressor_v1.config.Configs.make(
        compressor_v1.config.select_mode(head_dim, overlap),
        size_n=num_tokens,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_source.shape[1],
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
    )
    page_size = cfgs.state_physical_page_size
    state_rows_per_token = cfgs.state_rows_per_token
    kv_block_size = cfgs.kv_block_size
    kv_stride = cfgs.kv_stride
    kv_page_stride = kv_block_size * kv_stride

    batch_size = query_start_loc.shape[0] - 1
    token_to_req_indices = jnp.minimum(
        jnp.searchsorted(
            query_start_loc[1:],
            jnp.arange(num_tokens, dtype=query_start_loc.dtype),
            side="right",
        ),
        batch_size - 1,
    ).astype(jnp.int32)
    req = token_to_req_indices

    block_table_stride = block_table.shape[0] // batch_size
    kv_block_table_stride = kv_block_table.shape[0] // batch_size
    req_state_base = req * block_table_stride
    req_kv_base = (
        req_state_base
        if kv_block_table_stride == block_table_stride
        else req * kv_block_table_stride
    )

    state_page_idx = positions // state_block_size
    state_page_offset = positions % state_block_size
    state_page_numbers = block_table[req_state_base + state_page_idx]
    slot_mapping_slots = (
        state_page_numbers * page_size + state_page_offset * state_rows_per_token
    )

    kv_idx = positions // compress_ratio
    kv_page_idx = kv_idx // kv_block_size
    kv_page_offset = kv_idx % kv_block_size
    kv_page_number = kv_block_table[req_kv_base + kv_page_idx]
    kv_slot_mapping = kv_page_number * kv_page_stride + kv_page_offset * kv_stride
    is_boundary = ((positions + 1) % compress_ratio) == 0
    kv_slot_mapping = jnp.where(is_boundary, kv_slot_mapping, -1)
    return token_to_req_indices, slot_mapping_slots, kv_slot_mapping


def _ref_prepare_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    is_boundary: jax.Array,
    is_decode_token: jax.Array,
    compress_ratio: int,
    num_reqs: int,
    rope_pack: int = 4,
):
    num_tokens = positions.shape[0]
    max_prefill_boundary = num_tokens // compress_ratio + 1 + num_reqs
    packed_size = num_tokens + max_prefill_boundary

    order = jnp.nonzero(is_boundary, size=packed_size, fill_value=-1)[0]
    valid = order >= 0
    safe = jnp.where(valid, order, 0)

    pos = positions[safe]
    req = token_to_req_indices[safe]
    kv = jnp.where(valid, kv_slot_mapping[safe], -1)
    is_decode = valid & is_decode_token[safe]

    valid_token = kv >= 0
    is_prefill = valid_token & ~is_decode

    row = jnp.where(is_prefill, kv // rope_pack, -1)
    starts_row = is_prefill & jnp.pad(row[1:] != row[:-1], (1, 0), constant_values=True)
    tile_index = jnp.cumsum(starts_row.astype(jnp.int32)) - 1
    index = jnp.arange(packed_size)
    run_start = jax.lax.cummax(jnp.where(starts_row, index, 0))
    pos_within_tile = index - run_start

    k_val = jnp.sum(is_decode)
    k_aligned = ((k_val + 3) // 4) * 4
    prefill_size = rope_pack * max_prefill_boundary
    output_size = num_tokens + prefill_size

    dest_prefill = k_aligned + tile_index * rope_pack + pos_within_tile
    dest = jnp.where(is_decode, index, dest_prefill)
    dest = jnp.where(valid_token, dest, output_size)

    def scatter(default_val, src):
        return (
            jnp.full((output_size,), default_val, dtype=src.dtype)
            .at[dest]
            .set(src, mode="drop")
        )

    return scatter(0, pos), scatter(0, req), scatter(-1, kv)


def test_derive_metadata_and_packing_helpers():
    bt = jnp.arange(8, dtype=jnp.int32)
    assert compressor_v1.block_table_row_stride(bt, 2) == 4

    positions = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=jnp.int32)
    qsl = jnp.array([0, 8, 12], dtype=jnp.int32)
    kv_bt_same = jnp.arange(8, dtype=jnp.int32)
    kv_bt_diff = jnp.arange(12, dtype=jnp.int32)
    cache = jnp.zeros((16, 256, 4, 128), dtype=jnp.uint8)
    state_cache = jnp.zeros((16, 512, 4, 128), dtype=jnp.uint8)
    cos_sin = jnp.ones((64, 64), dtype=jnp.float32)

    # 1) Same stride, cos_sin provided, state_cache=None
    req1, slots1, kv1 = compressor_v1.derive_metadata(
        positions, bt, qsl, kv_bt_same, cache, 4, 16, 512, True, cos_sin, None
    )
    r_req1, r_slots1, r_kv1 = _ref_derive_metadata(
        positions, bt, qsl, kv_bt_same, cache, 4, 16, 512, True, cos_sin, None
    )
    np.testing.assert_array_equal(np.asarray(req1), np.asarray(r_req1))
    np.testing.assert_array_equal(np.asarray(slots1), np.asarray(r_slots1))
    np.testing.assert_array_equal(np.asarray(kv1), np.asarray(r_kv1))

    # 2) Different stride, cos_sin=None, state_cache provided
    req2, slots2, kv2 = compressor_v1.derive_metadata(
        positions, bt, qsl, kv_bt_diff, cache, 4, 16, 512, False, None, state_cache
    )
    r_req2, r_slots2, r_kv2 = _ref_derive_metadata(
        positions, bt, qsl, kv_bt_diff, cache, 4, 16, 512, False, None, state_cache
    )
    np.testing.assert_array_equal(np.asarray(req2), np.asarray(r_req2))
    np.testing.assert_array_equal(np.asarray(slots2), np.asarray(r_slots2))
    np.testing.assert_array_equal(np.asarray(kv2), np.asarray(r_kv2))

    # 3) Single-request fast path (batch_size == 1)
    pos_single = jnp.arange(8, dtype=jnp.int32)
    qsl_single = jnp.array([0, 8], dtype=jnp.int32)
    bt_single = jnp.arange(4, dtype=jnp.int32)
    req3, slots3, kv3 = compressor_v1.derive_metadata(
        pos_single, bt_single, qsl_single, bt_single, cache, 4, 16, 512, True, cos_sin
    )
    r_req3, r_slots3, r_kv3 = _ref_derive_metadata(
        pos_single, bt_single, qsl_single, bt_single, cache, 4, 16, 512, True, cos_sin
    )
    np.testing.assert_array_equal(np.asarray(req3), np.asarray(r_req3))
    np.testing.assert_array_equal(np.asarray(slots3), np.asarray(r_slots3))
    np.testing.assert_array_equal(np.asarray(kv3), np.asarray(r_kv3))

    # Direct helper check for _pack_array
    keep = jnp.array([True, False, True, False])
    safe, valid = compressor_v1._get_packing_indices(keep, 4)
    packed = compressor_v1._pack_array(jnp.array([10, 20, 30, 40]), safe, valid, -1)
    np.testing.assert_array_equal(np.asarray(packed), np.array([10, 30, -1, -1]))


def test_prepare_boundary_batch_and_compressor_forward(monkeypatch):
    positions = jnp.array([3, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
    qsl = jnp.array([0, 2, 10], dtype=jnp.int32)
    bt = jnp.arange(8, dtype=jnp.int32)
    kv_bt = jnp.arange(8, dtype=jnp.int32)
    cache = jnp.zeros((16, 256, 4, 128), dtype=jnp.uint8)
    state_cache = jnp.zeros((16, 512, 4, 128), dtype=jnp.uint8)
    rope_cache = jnp.zeros((16, 64, 4, 128), dtype=jnp.uint8)
    cos_sin = jnp.ones((64, 64), dtype=jnp.float32)

    req, _, kv_slots = compressor_v1.derive_metadata(
        positions, bt, qsl, kv_bt, cache, 4, 16, 512, True, cos_sin, state_cache
    )
    is_boundary = kv_slots >= 0
    is_decode = jnp.arange(positions.shape[0]) < 2

    b_pos, b_req, b_kv = compressor_v1.prepare_boundary_batch(
        positions, req, kv_slots, is_boundary, is_decode, 4, 2
    )
    r_pos, r_req, r_kv = _ref_prepare_boundary_batch(
        positions, req, kv_slots, is_boundary, is_decode, 4, 2
    )
    np.testing.assert_array_equal(np.asarray(b_pos), np.asarray(r_pos))
    np.testing.assert_array_equal(np.asarray(b_req), np.asarray(r_req))
    np.testing.assert_array_equal(np.asarray(b_kv), np.asarray(r_kv))

    # Test compressor_forward for both shares_buffer=True (state_cache=None)
    # and shares_buffer=False (state_cache provided)
    monkeypatch.setattr(
        compressor_v1.proj_kernel,
        "proj_and_save_state",
        lambda **kwargs: kwargs["cache"] + jnp.uint8(1),
    )
    monkeypatch.setattr(
        compressor_v1.compress_kernel,
        "compress_norm_rope_store",
        lambda c, *_args, **kwargs: (c + jnp.uint8(2), kwargs["rope_cache"]),
    )

    hidden = jnp.ones((10, 128), dtype=jnp.float32)
    wkv = jnp.ones((1024, 128), dtype=jnp.float32)
    ape = jnp.ones((4, 512), dtype=jnp.float32)
    norm_w = jnp.ones((512,), dtype=jnp.float32)
    dist = jnp.array([1, 1, 2], dtype=jnp.int32)

    out_c1, out_r1, out_s1 = compressor_v1.compressor_forward(
        hidden,
        wkv,
        ape,
        norm_w,
        cos_sin,
        positions,
        bt,
        qsl,
        kv_bt,
        cache,
        rope_cache,
        dist,
        state_block_size=16,
        head_dim=512,
        compress_ratio=4,
        overlap=True,
        rms_eps=1e-6,
        quant_block=64,
        state_cache=None,
    )
    assert int(out_c1[0, 0, 0, 0]) == 3
    assert out_s1 is out_c1
    assert out_r1 is rope_cache

    out_c2, out_r2, out_s2 = compressor_v1.compressor_forward(
        hidden,
        wkv,
        ape,
        norm_w,
        cos_sin,
        positions,
        bt,
        qsl,
        kv_bt,
        cache,
        rope_cache,
        dist,
        state_block_size=16,
        head_dim=512,
        compress_ratio=4,
        overlap=True,
        rms_eps=1e-6,
        quant_block=64,
        state_cache=state_cache,
    )
    assert int(out_c2[0, 0, 0, 0]) == 2
    assert int(out_s2[0, 0, 0, 0]) == 1
    assert out_r2 is rope_cache
