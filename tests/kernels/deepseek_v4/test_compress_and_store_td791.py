# SPDX-License-Identifier: Apache-2.0
"""Comprehensive CPU unit tests for TD7.91 compress_and_store modules."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compress_store_ref,
    compressor_v1,
    proj_and_save_state,
    project_and_save_state_ref,
)


def test_project_and_save_state_ref_all_branches():
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    num_tokens = 4
    hidden_size = 128
    head_dim = 128
    compress_ratio = 4
    overlap = False
    state_width = head_dim
    state_dim = 2 * state_width

    hidden_states = jax.random.normal(k1, (num_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    # Mix valid and invalid (-1) slot_mapping
    slot_mapping = jnp.array([0, 2, -1, 4], dtype=jnp.int32)

    num_pages = 2
    page_size = 8
    cache = jnp.zeros((num_pages, page_size, 4, 128), dtype=jnp.uint8)

    out_cache = project_and_save_state_ref.ref_wkv_proj_and_save_state(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping,
        cache=cache,
        state_block_size=4,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    assert out_cache.shape == cache.shape
    assert out_cache.dtype == jnp.uint8

    # Assert state_block_size > tokens_per_page fails
    with pytest.raises(AssertionError, match="state_block_size"):
        project_and_save_state_ref.ref_wkv_proj_and_save_state(
            hidden_states=hidden_states,
            wkv_wgate=wkv_wgate,
            ape=ape,
            positions=positions,
            slot_mapping=slot_mapping,
            cache=cache,
            state_block_size=999,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )


def test_compress_store_ref_helpers_and_errors():
    # 1. quantize_fp8_ue8m0 & quantize_fp8_lane_periodic
    x = jnp.arange(256, dtype=jnp.float32).reshape(2, 128) * 0.1
    q_ue, s_ue = compress_store_ref.quantize_fp8_ue8m0(x, 64)
    assert q_ue.shape == (2, 128)
    assert s_ue.shape == (2, 2)

    q_lp, s_lp = compress_store_ref.quantize_fp8_lane_periodic(x, 64, 64)
    assert q_lp.shape == (2, 128)
    assert s_lp.shape == (2, 64)

    # 2. interleaved_rope validation errors and normal execution
    with pytest.raises(ValueError, match="head_dim must be even"):
        compress_store_ref.interleaved_rope(jnp.zeros((2, 127)), jnp.zeros((2, 64)), 64)
    with pytest.raises(ValueError, match="rope_head_dim must be even"):
        compress_store_ref.interleaved_rope(jnp.zeros((2, 128)), jnp.zeros((2, 63)), 63)
    with pytest.raises(ValueError, match="rope_head_dim must be even and <="):
        compress_store_ref.interleaved_rope(
            jnp.zeros((2, 128)), jnp.zeros((2, 130)), 130
        )

    out_rope = compress_store_ref.interleaved_rope(
        jnp.ones((2, 128)), jnp.ones((2, 64)), 64
    )
    assert out_rope.shape == (2, 128)


def test_ref_compress_norm_rope_store_all_modes():
    key = jax.random.key(7)
    k1, k2 = jax.random.split(key, 2)

    # Mode A: CSA (is_quantized=True, has_rope=True, has_rope_cache=None -> True)
    # Test both word-array (int32) and slab-array (uint8) inputs.
    num_pages = 2
    head_dim = 512
    rope_head_dim = 64
    compress_ratio = 4
    state_block_size = 4
    phys_page_size = 64  # state_rows = 16 -> tokens_per_page = 4

    cache_u8 = jnp.zeros((num_pages, phys_page_size, 4, 128), dtype=jnp.uint8)
    cache_words = csa_cache_layout.slabs_to_words(cache_u8)
    state_words = csa_cache_layout.slabs_to_words(cache_u8)
    rope_u8 = jnp.zeros((num_pages, phys_page_size // 4, 4, 128), dtype=jnp.uint8)
    rope_words = csa_cache_layout.slabs_to_words(rope_u8)

    positions = jnp.array([3, 7, 1, 3], dtype=jnp.int32)
    slot_mapping = jnp.array([0, 16, 4, 32], dtype=jnp.int32)
    block_table = jnp.array([0, 1], dtype=jnp.int32)
    token_to_req_indices = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    kv_slot_mapping = jnp.array([0, 1, -1, 2], dtype=jnp.int32)
    rms_weight = jax.random.normal(k1, (head_dim,))
    cos_sin_cache = jax.random.normal(k2, (16, rope_head_dim))

    # A1: cache and state_cache and rope_cache as int32 words (has_rope_cache=None)
    new_c_w, new_r_w = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_words,
        rope_cache=rope_words,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=2,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=True,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=None,
        state_cache=state_words,
    )
    assert new_c_w.dtype == jnp.int32
    assert new_r_w.dtype == jnp.int32

    # A2: cache and rope_cache as uint8 slabs, state_cache=None
    new_c_u8, new_r_u8 = compress_store_ref.ref_compress_norm_rope_store(
        cache=cache_u8,
        rope_cache=rope_u8,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=2,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=True,
        rms_eps=1e-6,
        quant_block=64,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=True,
        state_cache=None,
    )
    assert new_c_u8.dtype == jnp.uint8
    assert new_r_u8.dtype == jnp.uint8

    # A3: Invalid rope_head_dim != 64 when has_rope_cache=True -> ValueError
    with pytest.raises(ValueError, match="CSA RoPE is"):
        compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_u8,
            rope_cache=rope_u8,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=jnp.zeros((16, 128)),
            block_table_stride=2,
            state_block_size=state_block_size,
            head_dim=head_dim,
            rope_head_dim=128,
            compress_ratio=compress_ratio,
            overlap=True,
            rms_eps=1e-6,
            quant_block=64,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=True,
            state_cache=None,
        )

    # Mode B: CSA_INDEXER (head_dim=128, cache (..., 4, 256), has_rope_cache=False)
    idx_cache = jnp.zeros((num_pages, 16, 4, 256), dtype=jnp.uint8)
    idx_c_out, _ = compress_store_ref.ref_compress_norm_rope_store(
        cache=idx_cache,
        rope_cache=rope_u8,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=jnp.ones((128,)),
        cos_sin_cache=cos_sin_cache,
        block_table_stride=2,
        state_block_size=state_block_size,
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=compress_ratio,
        overlap=True,
        rms_eps=1e-6,
        quant_block=128,
        is_quantized=True,
        has_rope=True,
        has_rope_cache=False,
        state_cache=None,
    )
    assert idx_c_out.shape == idx_cache.shape

    # Trigger nope_pad < 0 ValueError by passing rope_head_dim=32 on CSA
    with pytest.raises(ValueError, match="exceeds cache width"):
        compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_u8,
            rope_cache=rope_u8,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=jnp.zeros((16, 32)),
            block_table_stride=2,
            state_block_size=state_block_size,
            head_dim=head_dim,
            rope_head_dim=32,
            compress_ratio=compress_ratio,
            overlap=True,
            rms_eps=1e-6,
            quant_block=64,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=True,
            state_cache=None,
        )

    # Mode C: HCA (is_quantized=False, has_rope_cache=False, nope_pad == 0)
    hca_cache = jnp.zeros((num_pages, 16, 4, 128), dtype=jnp.uint8)
    hca_state = jnp.zeros((num_pages, 64, 4, 128), dtype=jnp.uint8)
    hca_c_out, _ = compress_store_ref.ref_compress_norm_rope_store(
        cache=hca_cache,
        rope_cache=rope_u8,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=2,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=4,
        overlap=False,
        rms_eps=1e-6,
        quant_block=0,
        is_quantized=False,
        has_rope=True,
        has_rope_cache=False,
        state_cache=hca_state,
    )
    assert hca_c_out.shape == hca_cache.shape


def test_compressor_v1_all_functions_and_branches(monkeypatch):
    # 1. block_table_row_stride & _get_packing_indices & _pack_array
    bt = jnp.arange(6, dtype=jnp.int32)
    assert compressor_v1.block_table_row_stride(bt, 2) == 3

    safe, valid = compressor_v1._get_packing_indices(jnp.array([True, False, True]), 4)
    packed = compressor_v1._pack_array(jnp.array([10, 20, 30]), safe, valid, -1)
    np.testing.assert_array_equal(np.asarray(packed), np.array([10, 30, -1, -1]))

    # 2. derive_metadata with same and different block_table / kv_block_table strides,
    # and cos_sin_cache None vs not None, state_cache None vs not None
    positions = jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 4, 8], dtype=jnp.int32)
    block_table_same = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # stride 2
    kv_block_table_same = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # stride 2
    kv_block_table_diff = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)  # stride 3
    cache = jnp.zeros((4, 64, 4, 128), dtype=jnp.uint8)
    cos_sin = jnp.zeros((16, 64), dtype=jnp.float32)

    req_idx, slot_map, kv_slot_map = compressor_v1.derive_metadata(
        positions=positions,
        block_table=block_table_same,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table_same,
        cache=cache,
        compress_ratio=4,
        state_block_size=4,
        head_dim=512,
        overlap=True,
        cos_sin_cache=cos_sin,
        state_cache=None,
    )
    assert req_idx.shape == (8,)
    assert slot_map.shape == (8,)
    assert kv_slot_map.shape == (8,)

    # Different kv_block_table stride and cos_sin_cache=None and separate state_cache
    state_cache = jnp.zeros((4, 64, 4, 128), dtype=jnp.uint8)
    req_idx2, _, _ = compressor_v1.derive_metadata(
        positions=positions,
        block_table=block_table_same,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table_diff,
        cache=cache,
        compress_ratio=4,
        state_block_size=4,
        head_dim=512,
        overlap=False,
        cos_sin_cache=None,
        state_cache=state_cache,
    )
    assert req_idx2.shape == (8,)

    # 3. prepare_boundary_batch
    is_boundary = jnp.array([False, False, False, True, False, False, False, True])
    is_decode = jnp.array([True, True, True, True, False, False, False, False])
    b_pos, b_req, b_kv = compressor_v1.prepare_boundary_batch(
        positions=positions,
        token_to_req_indices=req_idx,
        kv_slot_mapping=kv_slot_map,
        is_boundary=is_boundary,
        is_decode_token=is_decode,
        compress_ratio=4,
        num_reqs=2,
    )
    assert b_pos.shape == b_req.shape == b_kv.shape

    # 4. compressor_forward: shared buffer (state_cache=None) vs separate state_cache
    monkeypatch.setattr(
        compressor_v1.proj_kernel,
        "proj_and_save_state",
        lambda **kw: kw["cache"] + 1,
    )
    monkeypatch.setattr(
        compressor_v1.compress_kernel,
        "compress_norm_rope_store",
        lambda c, *a, **kw: (c + 1, kw["rope_cache"] + 1),
    )

    hidden_states = jnp.zeros((8, 128), dtype=jnp.float32)
    wkv_wgate = jnp.zeros((2048, 128), dtype=jnp.float32)
    ape = jnp.zeros((4, 1024), dtype=jnp.float32)
    norm_weight = jnp.ones((512,), dtype=jnp.float32)
    rope_cache = jnp.zeros((4, 16, 128), dtype=jnp.int32)
    distribution = jnp.array([1, 1, 2], dtype=jnp.int32)

    c_out1, r_out1, s_out1 = compressor_v1.compressor_forward(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin,
        positions=positions,
        block_table=block_table_same,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table_same,
        cache=cache,
        rope_cache=rope_cache,
        distribution=distribution,
        state_block_size=4,
        head_dim=512,
        compress_ratio=4,
        overlap=True,
        rms_eps=1e-6,
        quant_block=64,
        state_cache=None,
    )
    assert c_out1 is s_out1

    c_out2, r_out2, s_out2 = compressor_v1.compressor_forward(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin,
        positions=positions,
        block_table=block_table_same,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table_same,
        cache=cache,
        rope_cache=rope_cache,
        distribution=distribution,
        state_block_size=4,
        head_dim=512,
        compress_ratio=4,
        overlap=False,
        rms_eps=1e-6,
        quant_block=0,
        state_cache=state_cache,
    )
    assert c_out2.shape == cache.shape
    assert s_out2.shape == state_cache.shape


def test_proj_and_save_state_configs_helpers_and_kernel(monkeypatch):
    # 1. Configs properties
    ts = proj_and_save_state.TileSizes(tile_m=256, tile_k=128, tile_n=8)
    dims = proj_and_save_state.Dimensions(
        size_m=512, size_k=256, size_n=16, compress_ratio=4
    )
    cfgs = proj_and_save_state.Configs(tile_sizes=ts, dims=dims, token_page_size=16)
    assert cfgs.num_m == 2
    assert cfgs.num_k == 2
    assert cfgs.num_n == 2

    # 2. _select_tile_k
    proj_and_save_state._select_tile_k.cache_clear()
    assert proj_and_save_state._select_tile_k(7168) == 3584
    assert proj_and_save_state._select_tile_k(4096) == 2048
    with pytest.raises(AssertionError, match="Cannot find a tile_k"):
        proj_and_save_state._select_tile_k(100)

    # 3. Execute load_from_acc_ref, store_to_acc_ref, kernel, and proj_and_save_state
    monkeypatch.setattr(
        proj_and_save_state.pltpu, "bitcast", lambda v, dt: v.astype(dt)
    )

    class _FakeRef:
        def __init__(self, arr):
            self.arr = np.asarray(arr)
            self.shape = self.arr.shape
            self.dtype = jnp.dtype(self.arr.dtype)

        def reshape(self, *shape):
            return _FakeRef(self.arr.reshape(*shape))

        def bitcast(self, dt):
            return _FakeRef(self.arr.astype(np.uint8))

        @property
        def at(self):
            parent = self

            class _At:
                def __getitem__(self, idx):
                    if (
                        isinstance(idx, tuple)
                        and idx
                        and isinstance(idx[0], int)
                        and len(parent.shape) == 5
                    ):
                        return _FakeRef(parent.arr[idx[0]])
                    if isinstance(idx, int) and len(parent.shape) == 5:
                        return _FakeRef(parent.arr[idx])
                    return _FakeRef(parent.arr)

            return _At()

        def _norm_idx(self, idx):
            if hasattr(idx, "start") and hasattr(idx, "size"):
                stride = getattr(idx, "stride", 1) or 1
                start = int(idx.start)
                size = int(idx.size)
                return slice(start, start + size * stride, stride)
            return idx

        def __getitem__(self, idx):
            if idx is Ellipsis:
                return jnp.asarray(self.arr)
            if isinstance(idx, int) and self.arr.ndim == 1:
                return _FakeRef(self.arr[idx])
            norm = self._norm_idx(idx)
            return jnp.asarray(self.arr[norm])

        def __setitem__(self, idx, val):
            if isinstance(idx, int) and self.arr.ndim == 1:
                self.arr[idx] = int(np.asarray(val))
                return
            norm = self._norm_idx(idx)
            self.arr[norm] = np.asarray(val)

    ref_1chunk = _FakeRef(np.zeros((4, 1, 1, 128), dtype=np.float32))
    val_1chunk = jnp.ones((4, 128), dtype=jnp.float32)
    proj_and_save_state.store_to_acc_ref(ref_1chunk, val_1chunk)
    loaded_1_nodt = proj_and_save_state.load_from_acc_ref(ref_1chunk, dtype=None)
    loaded_1_dt = proj_and_save_state.load_from_acc_ref(ref_1chunk, dtype=jnp.float32)
    assert loaded_1_nodt.shape == (4, 128)
    assert loaded_1_dt.shape == (4, 128)

    ref_2chunk = _FakeRef(np.zeros((4, 2, 1, 128), dtype=np.float32))
    val_2chunk = jnp.ones((4, 256), dtype=jnp.float32)
    proj_and_save_state.store_to_acc_ref(ref_2chunk, val_2chunk)
    loaded_2 = proj_and_save_state.load_from_acc_ref(ref_2chunk, dtype=jnp.float32)
    assert loaded_2.shape == (4, 256)

    monkeypatch.setattr(
        proj_and_save_state.pl,
        "when",
        lambda cond: (lambda fn: (fn(), fn)[1]),
    )
    monkeypatch.setattr(proj_and_save_state.pl, "semaphore_wait", lambda *a, **kw: None)
    monkeypatch.setattr(
        proj_and_save_state.pltpu,
        "make_async_copy",
        lambda src, dst, sem: SimpleNamespace(start=lambda: None),
    )

    prog_ids = [0, 0, 0]
    monkeypatch.setattr(
        proj_and_save_state.pl, "program_id", lambda axis: prog_ids[axis]
    )

    def _fake_emit_pipeline(inner_fn, *, grid, in_specs):
        for spec in in_specs:
            spec.index_map(0, 0, 0)

        def _run(wkv_wgate, hidden_states, positions):
            w_shape = in_specs[0].block_shape
            h_shape = in_specs[1].block_shape
            p_shape = in_specs[2].block_shape
            w_ref = _FakeRef(np.ones(w_shape, dtype=np.float32))
            h_ref = _FakeRef(np.ones(h_shape, dtype=np.float32))
            p_ref = _FakeRef(np.arange(p_shape[0], dtype=np.int32))
            for n in range(grid[0]):
                for m in range(grid[1]):
                    for k in range(grid[2]):
                        prog_ids[0], prog_ids[1], prog_ids[2] = n, m, k
                        inner_fn(w_ref, h_ref, p_ref)

        return _run

    monkeypatch.setattr(proj_and_save_state.pltpu, "emit_pipeline", _fake_emit_pipeline)

    def _fake_pallas_call(kernel_fn, **kwargs):
        def _invoke(slot_mapping, positions, hidden_states, wkv_wgate, ape, cache):
            cfgs_call = kernel_fn.keywords["cfgs"]
            tile_n_c = cfgs_call.tile_sizes.tile_n
            slots_per_m = cfgs_call.tile_sizes.tile_m // cache.shape[-1]
            acc_ref = _FakeRef(
                np.zeros(
                    (2, tile_n_c, slots_per_m, 1, cache.shape[-1]),
                    dtype=np.float32,
                )
            )
            sem_ref = _FakeRef(np.zeros((2,), dtype=np.int32))
            write_counts_ref = _FakeRef(np.zeros((2,), dtype=np.int32))
            kernel_fn(
                slot_mapping,
                positions,
                hidden_states,
                wkv_wgate,
                _FakeRef(np.asarray(ape)),
                _FakeRef(np.asarray(cache)),
                None,
                acc_ref,
                sem_ref,
                write_counts_ref,
            )
            return cache

        return _invoke

    monkeypatch.setattr(proj_and_save_state.pl, "pallas_call", _fake_pallas_call)

    # Call proj_and_save_state.__wrapped__ with unpadded (8) and padded (10) tokens
    hs_8 = jnp.ones((8, 256), dtype=jnp.float32)
    wkv = jnp.ones((256, 512), dtype=jnp.float32)
    ape = jnp.ones((4, 256), dtype=jnp.float32)
    pos_8 = jnp.arange(8, dtype=jnp.int32)
    slot_8 = jnp.arange(8, dtype=jnp.int32)
    cache = jnp.zeros((2, 16, 4, 128), dtype=jnp.uint8)

    out_8 = proj_and_save_state.proj_and_save_state.__wrapped__(
        hs_8, wkv, ape, pos_8, slot_8, cache, compress_ratio=4
    )
    assert out_8.shape == cache.shape

    # Padded case: num_tokens=10 -> tile_n=16 -> padded_num_tokens=16 != 10
    hs_10 = jnp.ones((10, 256), dtype=jnp.float32)
    pos_10 = jnp.arange(10, dtype=jnp.int32)
    slot_10 = jnp.arange(10, dtype=jnp.int32)
    out_10 = proj_and_save_state.proj_and_save_state.__wrapped__(
        hs_10, wkv, ape, pos_10, slot_10, cache, compress_ratio=4
    )
    assert out_10.shape == cache.shape
