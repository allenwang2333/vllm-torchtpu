# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.megablox import common, tuned_block_sizes


@pytest.mark.parametrize(
    "kind, generation",
    [("TPU v4", 4), ("TPU v5p", 5), ("TPU v5 lite", 5), ("TPU v6e", 6), ("TPU7x", 7)],
)
def test_tpu_identification(monkeypatch, kind, generation):
    monkeypatch.setattr(
        common.jax, "devices", lambda: [SimpleNamespace(device_kind=kind)]
    )
    assert common.is_tpu()
    assert common.tpu_kind() == kind
    assert common.tpu_generation() == generation


@pytest.mark.parametrize("kind", ["cpu", "NVIDIA A100", "TPU unknown"])
def test_unsupported_generation(monkeypatch, kind):
    monkeypatch.setattr(
        common.jax, "devices", lambda: [SimpleNamespace(device_kind=kind)]
    )
    assert common.is_tpu() == kind.startswith("TPU")
    with pytest.raises(NotImplementedError, match="only TPU"):
        common.tpu_generation()


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.bfloat16,
        jnp.float32,
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
        jnp.int8,
        jnp.int4,
        jnp.float4_e2m1fn,
        jnp.uint4,
    ],
)
def test_supported_dtype(dtype):
    common.assert_is_supported_dtype(dtype)
    common.assert_is_supported_dtype(jnp.dtype(dtype))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.int32, jnp.bool_])
def test_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="No support"):
        common.assert_is_supported_dtype(dtype)
    with pytest.raises(ValueError, match="No support"):
        common.assert_is_supported_dtype(jnp.dtype(dtype))


@pytest.mark.parametrize(
    "x, limit, expected",
    [
        (1, 512, 128),
        (128, 512, 128),
        (129, 512, 256),
        (511, 512, 512),
        (2048, 2048, 2048),
        (3072, 2048, 1536),
        (2051, 2048, 2048),
    ],
)
def test_rounding_and_divisor_selection(x, limit, expected):
    assert (
        tuned_block_sizes.round_up_to_multiple_of_128_within_limit(x, limit) == expected
    )


@pytest.mark.parametrize("limit", [0, 127, 129])
def test_invalid_tile_limit(limit):
    with pytest.raises(AssertionError):
        tuned_block_sizes.round_up_to_multiple_of_128_within_limit(256, limit)


@pytest.mark.parametrize(
    "m, k, n, groups",
    [
        (320, 3072, 1000, 8),
        (64, 128, 256, 8),
        (4096, 4096, 6144, 8),
    ],
)
def test_default_tiles(m, k, n, groups):
    tm, tk, tn = tuned_block_sizes.get_default_gmm_block_sizes(m, k, n, groups)
    assert 0 < tm <= min(m, 512)
    assert m % tm == 0
    for tile in (tk, tn):
        assert 0 < tile <= 2048
        assert tile % 128 == 0


@pytest.mark.parametrize("tiles", [(128, 512, 1024), (64, 1024, 512)])
def test_tuned_lookup(monkeypatch, tiles):
    key = (128, 320, 6144, 160, 160, "bfloat16", "float8_e4m3fn", 320)
    monkeypatch.setattr(tuned_block_sizes, "TUNED_BLOCK_SIZES", {key: tiles})
    result = tuned_block_sizes.get_tuned_block_sizes(*key)
    assert result == tiles
    assert result != tuned_block_sizes.get_default_gmm_block_sizes(128, 320, 6144, 160)


def test_lookup_falls_back_for_untuned_shape(monkeypatch):
    monkeypatch.setattr(tuned_block_sizes, "TUNED_BLOCK_SIZES", {})
    assert tuned_block_sizes.get_tuned_block_sizes(
        320, 3072, 1000, 16, 8, "float32", "float32", 0
    ) == tuned_block_sizes.get_default_gmm_block_sizes(320, 3072, 1000, 8)


def test_masked_dense_prefill_cost_model():
    from vllm_torchtpu.kernels.mla.masked_dense import cost_model

    assert cost_model.KV_BLOCK_SIZE == 1024
    assert cost_model.QUERY_BLOCK_SIZE == 32
    m = cost_model.GLM52_TPU7X_PREFILL_COST_MODEL
    assert isinstance(m, cost_model.MaskedDensePrefillCostModel)
    assert m.analytic_fixed_ns == 215_000
    assert m.analytic_token_ns == 245
    assert m.bitmap_fixed_ns == 227_000
    assert m.bitmap_token_ns == 451
    assert m.q_kv_block_ns == 8_700
    assert m.sparse_fixed_ns == 259_000
    assert m.sparse_token_ns == 2_156


def test_mla_scatter_dispatch_and_validation(monkeypatch):
    from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter, scatter_utils

    spec_tc = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.TENSORCORE,
        num_pages=4,
        page_size=16,
        head_dim=512,
        kv_packing=4,
    )
    spec_sc = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages=4,
        page_size=16,
        head_dim=512,
        kv_packing=4,
    )
    values = jnp.zeros((2, 512), dtype=jnp.float8_e4m3fn)

    with pytest.raises(AssertionError, match="does not match spec"):
        scatter.scatter(
            jnp.zeros(spec_sc.shape, dtype=jnp.uint8),
            values,
            spec_sc,
            dst_rows=jnp.zeros((2,), dtype=jnp.int32),
        )

    cache_tc = jnp.zeros(spec_tc.shape, dtype=spec_tc.jax_dtype)
    with pytest.raises(AssertionError, match="pass dst_rows"):
        scatter.scatter(cache_tc, values, spec_tc)

    captured = []

    def fake_scatter_rows(cache, src, dst_rows, *, name):
        captured.append((cache.shape, src.shape, dst_rows.shape, name))
        return cache

    monkeypatch.setattr(scatter_utils, "scatter_rows", fake_scatter_rows)

    seq_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)

    out_tc = scatter.scatter(
        cache_tc,
        values,
        spec_tc,
        seq_lens=seq_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
    )
    assert out_tc is cache_tc
    assert captured[-1][3] == "scatter_nope_tensorcore_cache"

    cache_sc = jnp.zeros(spec_sc.shape, dtype=spec_sc.jax_dtype)
    dst_rows = jnp.array([0, 1], dtype=jnp.int32)
    out_sc = scatter.scatter(
        cache_sc,
        values,
        spec_sc,
        dst_rows=dst_rows,
    )
    assert out_sc is cache_sc
    assert captured[-1][3] == "scatter_nope_sparsecore_cache"


def test_rpa_v3_util(monkeypatch):
    from vllm_torchtpu.kernels.ragged_paged_attention.v3 import util as rpa_util

    assert rpa_util.cdiv(10, 3) == 4
    with pytest.raises(AssertionError):
        rpa_util.cdiv(10, 0)
    assert rpa_util.align_to(10, 4) == 12
    assert rpa_util.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert rpa_util.get_dtype_packing(jnp.bfloat16) == 2

    with pytest.raises(AssertionError):
        rpa_util.next_power_of_2(0)
    assert rpa_util.next_power_of_2(1) == 1
    assert rpa_util.next_power_of_2(5) == 8

    for kind, expected in [
        ("cpu", -1),
        ("TPU v4", 4),
        ("TPU v5 lite", 5),
        ("TPU v5p", 5),
        ("TPU v6e", 6),
        ("TPU7x", 7),
    ]:
        monkeypatch.setattr(
            rpa_util.jax, "devices", lambda k=kind: [SimpleNamespace(device_kind=k)]
        )
        assert rpa_util.get_tpu_version() == expected

    monkeypatch.setattr(
        rpa_util.jax, "devices", lambda: [SimpleNamespace(device_kind="TPU badX")]
    )
    with pytest.raises(AssertionError):
        rpa_util.get_tpu_version()


def test_sparse_core_map_helper(monkeypatch):
    from vllm_torchtpu.kernels.sparse_core import core_map_helper

    class FakeRef:
        def __init__(self, val, memory_space=None):
            self.val = val
            self.memory_space = memory_space

        def __getitem__(self, idx):
            return ("loaded", self.val, self.memory_space)

    monkeypatch.setattr(
        core_map_helper.pl_core,
        "_convert_out_shape_to_aval",
        lambda out_type: out_type,
    )
    monkeypatch.setattr(
        core_map_helper.lax,
        "empty",
        lambda shape, dtype, out_sharding=None: ("empty", shape, dtype, out_sharding),
    )
    monkeypatch.setattr(
        core_map_helper.jax_core,
        "new_ref",
        lambda val, memory_space=None: FakeRef(val, memory_space=memory_space),
    )

    aval_default_mem = SimpleNamespace(
        shape=(4,),
        dtype=jnp.float32,
        sharding=None,
        memory_space=core_map_helper.jax_core.MemorySpace.Device,
    )
    ref_default = core_map_helper._empty_out_ref(aval_default_mem)
    assert ref_default.memory_space is None

    aval_custom_mem = SimpleNamespace(
        shape=(4,),
        dtype=jnp.float32,
        sharding=None,
        memory_space="vmem",
    )
    ref_custom = core_map_helper._empty_out_ref(aval_custom_mem)
    assert ref_custom.memory_space == "vmem"

    monkeypatch.setattr(core_map_helper.api, "jit", lambda fn: fn)

    def fake_core_map(mesh, **kwargs):
        def decorator(fn):
            fn("scratch_pos", kw="scratch_kw")
            return fn

        return decorator

    monkeypatch.setattr(core_map_helper.pl_core, "core_map", fake_core_map)

    seen_calls = []

    def body(*args, **kwargs):
        seen_calls.append((args, kwargs))

    single_fn = core_map_helper.kernel(body, out_type=aval_custom_mem, mesh="mesh")
    single_out = single_fn("op0")
    assert single_out == ("loaded", ("empty", (4,), jnp.float32, None), "vmem")
    assert len(seen_calls) == 1

    multi_fn = core_map_helper.kernel(
        body, out_type=(aval_default_mem, aval_custom_mem), mesh="mesh"
    )
    multi_out = multi_fn("op0")
    assert isinstance(multi_out, tuple) and len(multi_out) == 2


def test_typed_ldst():
    from vllm_torchtpu.kernels import typed_ldst

    class MockBlockRef:
        def __init__(self, arr):
            self.arr = arr
            self.dtype = arr.dtype
            self.shape = arr.shape
            self.bitcast_calls = 0

        def bitcast(self, target_dtype):
            self.bitcast_calls += 1
            new_arr = self.arr.view(target_dtype)
            res = MockBlockRef(new_arr)
            res.bitcast_calls = self.bitcast_calls
            return res

        def reshape(self, *new_shape):
            res = MockBlockRef(self.arr.reshape(*new_shape))
            res.bitcast_calls = self.bitcast_calls
            return res

        def __getitem__(self, idx):
            return self.arr[idx]

        def __setitem__(self, idx, val):
            self.arr = val

    raw_same = MockBlockRef(jnp.arange(16, dtype=jnp.float32).reshape(2, 8))
    loaded_same = typed_ldst.load_typed(raw_same, view_dtype=jnp.float32, lane_split=2)
    assert loaded_same.shape == (4, 4)

    raw_diff = MockBlockRef(jnp.arange(16, dtype=jnp.uint8).reshape(2, 8))
    loaded_diff = typed_ldst.load_typed(raw_diff, view_dtype=jnp.uint16, lane_split=2)
    assert loaded_diff.shape == (4, 2)

    target_ref = MockBlockRef(jnp.zeros((2, 8), dtype=jnp.float32))
    values = jnp.ones((4, 4), dtype=jnp.float32)
    typed_ldst.store_typed(target_ref, values, lane_split=2)


def test_csa_cache_layout():
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

    assert csa_cache_layout.is_word_array(jnp.int32)
    assert not csa_cache_layout.is_word_array(jnp.uint8)

    assert csa_cache_layout.u8_slab_shape((2, 4, 4, 128), jnp.uint8) == (2, 4, 4, 128)
    assert csa_cache_layout.u8_slab_shape((2, 4, 128), jnp.int32) == (2, 4, 4, 128)

    class MockSlabRef:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

        def bitcast(self, dt):
            return MockSlabRef(self.shape, dt)

        def reshape(self, *shape):
            return MockSlabRef(shape, self.dtype)

    u8_ref = MockSlabRef((2, 4, 4, 128), jnp.uint8)
    assert csa_cache_layout.as_u8_slabs(u8_ref) is u8_ref

    i32_ref = MockSlabRef((2, 4, 128), jnp.int32)
    slab_ref = csa_cache_layout.as_u8_slabs(i32_ref)
    assert slab_ref.shape == (2, 4, 4, 128)
    assert slab_ref.dtype == jnp.uint8

    slabs = jnp.arange(2 * 4 * 8, dtype=jnp.uint8).reshape(2, 4, 8)
    words = csa_cache_layout.slabs_to_words(slabs)
    assert words.shape == (2, 8) and words.dtype == jnp.int32
    roundtrip_slabs = csa_cache_layout.words_to_slabs(words)
    assert jnp.array_equal(roundtrip_slabs, slabs)

    rope = jnp.ones((4, 64), dtype=jnp.bfloat16)
    rw = csa_cache_layout.rope_words(rope)
    assert rw.shape == (4, 32) and rw.dtype == jnp.int32
    encoded = csa_cache_layout.encode_rope(rope)
    assert encoded.shape == (1, 128) and encoded.dtype == jnp.int32
