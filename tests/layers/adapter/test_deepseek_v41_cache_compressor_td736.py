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
"""Unit and 100% coverage tests for TD7.36 deepseek_v41_cache and compressor."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_cache as cache_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_compressor as comp_mod,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR


class _BaseCacheOwner:
    def __init__(self) -> None:
        self._bound_cache = None

    def bind_kv_cache(self, cache: torch.Tensor) -> None:
        self.kv_cache = cache

    def get_kv_cache_spec(self, vllm_config):
        return ("base_spec", vllm_config)


class _DummyCacheBinding(cache_mod.NativeCacheBinding, _BaseCacheOwner):
    pass


class _FakeTpuCache:
    """Tensor-like proxy with ``device.type == 'tpu'`` for TPU branch tests."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._t = tensor
        self.device = SimpleNamespace(type="tpu")
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.copied_from = None

    def new_empty(self, shape, dtype=None):
        return self._t.new_empty(shape, dtype=dtype)

    def copy_(self, other: torch.Tensor) -> None:
        self.copied_from = other
        self._t.copy_(other)


def test_index_page_geometry_and_byte_layout_and_record_layout() -> None:
    with pytest.raises(ValueError, match="complete native word rows"):
        cache_mod.IndexPageGeometry(nope_bytes=0, scale_bytes=64)
    with pytest.raises(ValueError, match="complete native word rows"):
        cache_mod.IndexPageGeometry(nope_bytes=128, scale_bytes=64)
    with pytest.raises(ValueError, match="exceed the companion array budget"):
        cache_mod.IndexPageGeometry(nope_bytes=cache_mod.ROW_BYTES, scale_bytes=4)

    geom = cache_mod.IndexPageGeometry(nope_bytes=cache_mod.ROW_BYTES, scale_bytes=64)
    assert geom.keys_per_page == cache_mod.ROW_BYTES // cache_mod.INDEX_HEAD_DIM

    raw_cpu = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
    cache_cpu = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
    bl_auto = cache_mod.cache_byte_layout(raw_cpu, cache_cpu, None)
    assert isinstance(bl_auto, cache_mod.ByteLayout)
    assert cache_mod.cache_byte_layout(raw_cpu, cache_cpu, bl_auto) is bl_auto
    bl_tuple = cache_mod.cache_byte_layout(
        raw_cpu, cache_cpu, ((4, 2, 128), torch.bfloat16, 512, 0)
    )
    assert bl_tuple == bl_auto

    raw_tpu = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    with pytest.raises(ValueError, match="explicit allocator layout"):
        cache_mod.cache_byte_layout(raw_tpu, cache_cpu, None)

    # cache_record_layout error branches
    with pytest.raises(ValueError, match="requires DSV4 int32"):
        cache_mod.cache_record_layout(torch.zeros((16, 64), dtype=torch.int32), bl_auto)
    with pytest.raises(ValueError, match="must be BF16 or FP32"):
        cache_mod.cache_record_layout(
            raw_cpu,
            cache_mod.ByteLayout((4, 2, 128), torch.int32, 1024, 0),
        )
    with pytest.raises(ValueError, match="divide or be a multiple of 512"):
        cache_mod.cache_record_layout(
            raw_cpu,
            cache_mod.ByteLayout((4, 2, 120), torch.bfloat16, 480, 0),
        )
    with pytest.raises(ValueError, match="must be record-aligned"):
        cache_mod.cache_record_layout(
            raw_cpu,
            cache_mod.ByteLayout((4, 2, 128), torch.bfloat16, 300, 0),
        )
    huge_raw = SimpleNamespace(
        dtype=torch.int32,
        ndim=2,
        shape=(1 << 31, ROW_WORDS),
        nbytes=(1 << 31) * 512,
    )
    with pytest.raises(ValueError, match="must fit int32"):
        cache_mod.cache_record_layout(huge_raw, bl_auto)

    rec_layout = cache_mod.cache_record_layout(raw_cpu, bl_auto)
    assert rec_layout.geometry == (2, 2, 0)

    # MaterializedNativeKVCache & native_cache_view
    mat = cache_mod.MaterializedNativeKVCache(kv_caches={}, raw_tensors=[])
    assert mat.native_raw_cache == {}
    raw_bytes = torch.arange(2048, dtype=torch.uint8)
    view = cache_mod.native_cache_view(
        raw_bytes, (2, 2, 128), torch.bfloat16, 1024, 512
    )
    assert view.shape == (2, 2, 128)
    assert view.dtype == torch.bfloat16


def test_native_cache_binding_and_cached_viewed_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_torchtpu.runner.kv_cache_dsv4 as kv_dsv4

    binding_obj = _DummyCacheBinding()
    with pytest.raises(RuntimeError, match="requires native cache binding"):
        binding_obj.native()
    assert not binding_obj.has_native_cache

    with pytest.raises(TypeError, match="requires a single Tensor"):
        binding_obj.bind_kv_cache(["not_a_tensor"])  # type: ignore[arg-type]

    cache_tensor = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
    binding_obj.bind_kv_cache(cache_tensor)
    assert binding_obj.bound_cache is cache_tensor
    assert binding_obj.kv_cache is cache_tensor

    binding_obj.bind_native(None, None)
    assert not binding_obj.has_native_cache

    raw_pool = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
    with pytest.raises(ValueError, match="validated record geometry"):
        binding_obj.bind_native(raw_pool, None)

    binding_obj.bind_kv_cache(cache_tensor, raw_cache=raw_pool)
    assert binding_obj.has_native_cache
    assert binding_obj.native().pool is raw_pool
    assert binding_obj.kv_cache is raw_pool

    # Bind with kernel_shape and verify viewed_pool is cached once
    rec_layout = binding_obj.native().layout
    binding_obj.bind_native(raw_pool, rec_layout, kernel_shape=(4, 4, ROW_WORDS))
    kv1 = binding_obj.kv_cache
    kv2 = binding_obj.kv_cache
    assert kv1 is kv2
    assert kv1.shape == (4, 4, ROW_WORDS)

    monkeypatch.setattr(
        kv_dsv4, "deepseek_v41_cache_spec", lambda spec, cfg: ("wrapped", spec, cfg)
    )
    assert binding_obj.get_kv_cache_spec("cfg") == (
        "wrapped",
        ("base_spec", "cfg"),
        "cfg",
    )


def test_row_indices_read_write_and_gather_scatter_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    # Unit-stride fast path: page_size == page_stride
    assert cache_mod.row_indices(slots, 2, 2, 0) is slots
    assert torch.equal(cache_mod.row_indices(slots, 2, 2, 5), slots + 5)
    # Non-unit stride path: page_size != page_stride
    assert torch.equal(
        cache_mod.row_indices(slots, 2, 4, 1),
        torch.tensor([1, 2, 5, 6], dtype=torch.int32),
    )

    # masked_row_indices
    j_slots = jnp.array([0, -1, 2], dtype=jnp.int32)
    assert np.array_equal(
        np.asarray(cache_mod.masked_row_indices(j_slots, 2, 4, 1)),
        np.array([1, -1, 5], dtype=np.int32),
    )

    # read_cache & write_cache with layout=None (CPU 2D and 3D)
    cache_2d = torch.zeros((4, 8), dtype=torch.float32)
    vals_2d = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    valid = torch.tensor([True, False])
    cache_mod.write_cache(
        cache_2d, torch.tensor([2, 3], dtype=torch.int32), vals_2d, valid
    )
    assert torch.equal(cache_2d[2], vals_2d[0])

    cache_3d = torch.zeros((2, 2, 8), dtype=torch.float32)
    cache_mod.write_cache(
        cache_3d, torch.tensor([1, 3], dtype=torch.int32), vals_2d, valid
    )
    read_unstructured = cache_mod.read_cache(
        cache_3d, torch.tensor([1, 0], dtype=torch.int64)
    )
    assert torch.equal(read_unstructured[0], vals_2d[0])

    # read_cache & write_cache with layout!=None on CPU
    pool_cpu = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    layout_bf16 = cache_mod.RecordLayout(
        page_size=2,
        page_stride=2,
        row_offset=0,
        elements=128,
        dtype=torch.bfloat16,
    )
    vals_bf16 = torch.ones((2, 128), dtype=torch.bfloat16) * 2.0
    # Pass float32 values to exercise values.dtype != dtype conversion
    cache_mod.write_cache(
        pool_cpu,
        torch.tensor([1, 2], dtype=torch.int32),
        vals_bf16.float(),
        torch.tensor([True, True]),
        layout=layout_bf16,
    )
    # Also pass matching dtype values
    cache_mod.write_cache(
        pool_cpu,
        torch.tensor([1, 2], dtype=torch.int32),
        vals_bf16,
        torch.tensor([True, False]),
        layout=layout_bf16,
    )
    read_back = cache_mod.read_cache(
        pool_cpu, torch.tensor([1, 2], dtype=torch.int32), layout=layout_bf16
    )
    assert torch.equal(read_back[0], vals_bf16[0])

    # initialize_cache_ops + TPU branches of read_cache and write_cache
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as ctx_mod

    monkeypatch.setattr(cache_mod, "_read_values_op", None)
    monkeypatch.setattr(cache_mod, "_write_op", None)
    monkeypatch.setattr(cache_mod, "_write_values_op", None)

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    monkeypatch.setattr(
        ctx_mod,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )

    registered_ops: dict[str, SimpleNamespace] = {}

    def _fake_jax_op(name, fn, **kwargs):
        def _call(*args):
            if name.endswith("read_values"):
                return torch.zeros((args[1].numel(), args[2]), dtype=torch.bfloat16)
            return torch.ones_like(args[0]._t)

        op = SimpleNamespace(
            fn=fn,
            fake=None,
            calls=[],
            register_fake=lambda f: setattr(op, "fake", f),
        )

        def _wrapped(*args):
            op.calls.append(args)
            return _call(*args)

        _wrapped.register_fake = op.register_fake  # type: ignore[attr-defined]
        _wrapped.op = op  # type: ignore[attr-defined]
        registered_ops[name] = op
        return _wrapped

    monkeypatch.setattr(pallas, "jax_op", _fake_jax_op)
    cache_mod.initialize_cache_ops()
    # Second call hits early return
    cache_mod.initialize_cache_ops()

    # Exercise register_fake callbacks
    dummy_t = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
    assert registered_ops["pallas::deepseek_v41_cache_read_values"].fake(
        dummy_t, torch.zeros(2, dtype=torch.int32), 128, False, 2, 2, 0
    ).shape == (2, 128)
    assert (
        registered_ops["pallas::deepseek_v41_cache_write"]
        .fake(dummy_t, slots, vals_2d)
        .shape
        == dummy_t.shape
    )
    assert (
        registered_ops["pallas::deepseek_v41_cache_write_values"]
        .fake(dummy_t, slots, vals_bf16, 2, 2, 0)
        .shape
        == dummy_t.shape
    )

    # Exercise inner `write_values` function registered in initialize_cache_ops
    j_pool = jnp.zeros((4, ROW_WORDS), dtype=jnp.int32)
    j_vals = jnp.ones((2, 128), dtype=jnp.bfloat16)
    wv_out = registered_ops["pallas::deepseek_v41_cache_write_values"].fn(
        j_pool, jnp.array([0, -1], dtype=jnp.int32), j_vals, 2, 2, 0
    )
    assert wv_out.shape == (4, ROW_WORDS)

    # TPU branches of read_cache and write_cache (both int32 and int64 slots)
    tpu_pool = _FakeTpuCache(pool_cpu.clone())
    tpu_read = cache_mod.read_cache(
        tpu_pool, torch.tensor([0, 1], dtype=torch.int32), layout=layout_bf16
    )
    assert tpu_read.shape == (2, 128)

    cache_mod.write_cache(
        tpu_pool,
        torch.tensor([0, 1], dtype=torch.int32),
        vals_bf16,
        torch.tensor([True, True]),
        layout=layout_bf16,
    )
    cache_mod.write_cache(
        tpu_pool,
        torch.tensor([0, 1], dtype=torch.int64),
        vals_bf16,
        torch.tensor([True, True]),
        layout=layout_bf16,
    )
    cache_mod.write_cache(
        tpu_pool,
        torch.tensor([0, 1], dtype=torch.int32),
        vals_bf16,
        torch.tensor([True, True]),
        layout=None,
    )
    cache_mod.write_cache(
        tpu_pool,
        torch.tensor([0, 1], dtype=torch.int64),
        vals_bf16,
        torch.tensor([True, True]),
        layout=None,
    )


def test_jax_cache_gather_scatter_and_pallas_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    j_pool = jnp.zeros((4, ROW_WORDS), dtype=jnp.int32)
    # _scatter_rows
    scattered_rows = cache_mod._scatter_rows(
        j_pool,
        jnp.array([0, -1], dtype=jnp.int32),
        jnp.ones((2, ROW_WORDS), dtype=jnp.int32),
    )
    assert int(scattered_rows[0, 0]) == 1

    # read_cache_values & gather_cache_values:
    # 1. fp32=True with record_words >= ROW_WORDS (elements = ROW_WORDS)
    out_fp32_full = cache_mod.read_cache_values(
        scattered_rows,
        jnp.array([0, 1], dtype=jnp.int32),
        ROW_WORDS,
        True,
        2,
        2,
        0,
    )
    assert out_fp32_full.shape == (2, ROW_WORDS)
    assert out_fp32_full.dtype == jnp.float32

    # 2. fp32=False with record_words < ROW_WORDS (elements = 128)
    out_bf16_sub = cache_mod.read_cache_values(
        scattered_rows,
        jnp.array([0, 1], dtype=jnp.int32),
        128,
        False,
        2,
        2,
        0,
    )
    assert out_bf16_sub.shape == (2, 128)
    assert out_bf16_sub.dtype == jnp.bfloat16

    # scatter_cache_values & scatter_cache_words empty fast paths
    assert (
        cache_mod.scatter_cache_values(
            j_pool,
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, 128), dtype=jnp.bfloat16),
        )
        is j_pool
    )
    assert (
        cache_mod.scatter_cache_words(
            j_pool,
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, ROW_WORDS), dtype=jnp.int32),
        )
        is j_pool
    )

    # scatter_cache_values with float32 full row (width == ROW_WORDS) on CPU
    updated_full = cache_mod.scatter_cache_values(
        j_pool,
        jnp.array([1], dtype=jnp.int32),
        jnp.ones((1, ROW_WORDS), dtype=jnp.float32),
    )
    assert updated_full.shape == j_pool.shape

    # scatter_cache_words on TPU when width < ROW_WORDS -> calls write_cache_words
    from vllm_torchtpu.kernels.deepseek_v41 import cache_write as cw_mod

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(cw_mod, "write_cache_words", lambda c, r, w: c + 7)
    tpu_sub = cache_mod.scatter_cache_words(
        j_pool,
        jnp.array([0], dtype=jnp.int32),
        jnp.ones((1, 64), dtype=jnp.int32),
    )
    assert int(tpu_sub[0, 0]) == 7

    # scatter_cache_words on TPU when width >= ROW_WORDS -> Pallas kernel path
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    copy_events: list[str] = []

    class _FakeCopy:
        def start(self) -> None:
            copy_events.append("start")

        def wait(self) -> None:
            copy_events.append("wait")

    monkeypatch.setattr(pltpu, "make_async_copy", lambda src, dst, sem: _FakeCopy())

    class _RefWithAt:
        def __init__(self, arr):
            self.arr = arr
            self.at = self

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.arr[idx]
            return self

    monkeypatch.setattr(pl, "program_id", lambda axis: 0)

    def _fake_pallas_call(kernel_fn, **kwargs):
        def _runner(records, flat_words, cache):
            rec_ref = _RefWithAt(records)
            val_ref = _RefWithAt(flat_words)
            out_ref = _RefWithAt(cache)
            row_ref = _RefWithAt(jnp.zeros((ROW_WORDS,), dtype=jnp.int32))
            kernel_fn(rec_ref, val_ref, out_ref, out_ref, row_ref, "sem")
            return cache

        return _runner

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)
    tpu_full = cache_mod.scatter_cache_words(
        j_pool,
        jnp.array([0], dtype=jnp.int32),
        jnp.ones((1, ROW_WORDS), dtype=jnp.int32),
    )
    assert tpu_full.shape == j_pool.shape
    assert copy_events == ["start", "wait", "start", "wait"]


def test_cache_slots_and_row_slots() -> None:
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    empty_meta = SimpleNamespace(seq_lens=torch.zeros((0,), dtype=torch.int32))
    assert torch.equal(
        cache_mod.cache_slots(empty_meta, positions, rows_per_page=4),
        torch.full_like(positions, -1),
    )

    meta = SimpleNamespace(
        seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        block_tables=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
    )
    reqs = cache_mod.token_requests(meta, positions)
    s_auto = cache_mod.cache_slots(meta, positions, rows_per_page=2, ring=False)
    s_with_reqs = cache_mod.cache_slots(
        meta, positions, rows_per_page=2, ring=False, requests=reqs
    )
    assert torch.equal(s_auto, s_with_reqs)
    assert int(s_auto[3]) == -1

    s_ring = cache_mod.cache_slots(
        meta, positions, rows_per_page=2, compress_ratio=2, ring=True, requests=reqs
    )
    assert int(s_ring[0]) == 20

    r_slots = cache_mod.row_slots(
        meta.block_tables,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        rows_per_page=2,
    )
    assert r_slots.shape == (2, 2)


def test_compressor_save_compress_norm_and_csa_packing_and_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dim = 4
    norm_weight = torch.ones((dim,), dtype=torch.float32)
    kv_score = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    state_cache = torch.zeros((2, 4, 5), dtype=torch.float32)
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    latent = torch.empty((4, dim), dtype=torch.bfloat16)

    # Cover false branch of `if slot_mapping is not None:`
    with pytest.raises(TypeError):
        comp_mod.save_compress_norm(
            kv_score,
            positions,
            state_cache,
            None,
            query_start_loc,
            token_to_req,
            norm_weight,
            1e-6,
            1,
            latent,
        )

    # Invalid ratio raises ValueError
    with pytest.raises(ValueError, match="ratio must be 1 or 2"):
        comp_mod.save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slot_mapping,
            query_start_loc,
            token_to_req,
            norm_weight,
            1e-6,
            3,
            latent,
        )

    # Ratio == 1 with token_to_req_indices=None and float32 kv_score
    comp_mod.save_compress_norm(
        kv_score.float(),
        positions,
        None,
        slot_mapping,
        None,
        None,
        norm_weight,
        1e-6,
        1,
        latent,
    )
    assert not torch.isnan(latent).any()

    # Ratio == 2 with bfloat16 kv_score
    comp_mod.save_compress_norm(
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req,
        norm_weight,
        1e-6,
        2,
        latent,
    )
    assert torch.all(latent[0] == 0)
    assert not torch.all(latent[1] == 0)

    # pack_csa_values and write_csa_values on CPU
    vals_512 = jnp.ones((2, 512), dtype=jnp.bfloat16) * 0.5
    nope_cache = jnp.zeros((4, 4, ROW_WORDS), dtype=jnp.int32)
    rope_cache = jnp.zeros((4, 1, ROW_WORDS), dtype=jnp.int32)
    up_nope, up_rope = comp_mod.write_csa_values(
        nope_cache,
        rope_cache,
        jnp.array([0, 1], dtype=jnp.int32),
        vals_512,
    )
    assert up_nope.shape == nope_cache.shape
    assert up_rope.shape == rope_cache.shape

    # initialize_compressor_ops & write_compressed_cache
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as ctx_mod

    monkeypatch.setattr(comp_mod, "_write_op", None)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    monkeypatch.setattr(
        ctx_mod,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )

    comp_reg: dict = {}

    def _fake_comp_jax_op(name, fn, **kwargs):
        def _call(nope, rope, records, values):
            comp_reg["called"] = (nope, rope, records, values)
            return nope + 1, rope + 2

        _call.register_fake = lambda f: comp_reg.setdefault("fake", f)  # type: ignore[attr-defined]
        comp_reg["fn"] = fn
        return _call

    monkeypatch.setattr(pallas, "jax_op", _fake_comp_jax_op)
    comp_mod.initialize_compressor_ops()
    comp_mod.initialize_compressor_ops()

    fake_n, fake_r = comp_reg["fake"](
        torch.zeros((2, 2)), torch.zeros((2, 1)), "r", "v"
    )
    assert fake_n.shape == (2, 2)
    assert fake_r.shape == (2, 1)

    nope_t = torch.zeros((2, 2, ROW_WORDS), dtype=torch.int32)
    rope_t = torch.zeros((2, 1, ROW_WORDS), dtype=torch.int32)
    comp_mod.write_compressed_cache(
        (nope_t, rope_t),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros((2, 512), dtype=torch.bfloat16),
        torch.tensor([True, False]),
        2,
    )
    assert int(nope_t[0, 0, 0]) == 1
    assert int(rope_t[0, 0, 0]) == 2

    # CompressorStateCache & DeepseekCompressor
    state_cache_inst = comp_mod.CompressorStateCache.__new__(
        comp_mod.CompressorStateCache
    )
    assert state_cache_inst.get_attn_backend() is comp_mod.AttentionBackend

    monkeypatch.setattr(
        comp_mod.compressor_upstream.DeepseekCompressor,
        "__init__",
        lambda self, *a, **kw: None,
    )
    comp = comp_mod.DeepseekCompressor()
    comp.fused_wkv_wgate = SimpleNamespace(
        weight=torch.ones((5, 4), dtype=torch.bfloat16)
    )
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    p_unflipped = comp.project(hidden)
    assert p_unflipped.shape == (3, 5)

    setattr(comp.fused_wkv_wgate, WEIGHT_FLIPPED_ATTR, True)
    comp.fused_wkv_wgate.weight = torch.ones((4, 5), dtype=torch.bfloat16)
    p_flipped = comp.project(hidden)
    assert p_flipped.shape == (3, 5)

    # DeepseekCompressor.forward:
    # 1. metadata is not a dict -> returns None
    monkeypatch.setattr(
        comp_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    assert comp.forward(p_flipped, positions[:3]) is None

    # 2. self.state_cache is None (ratio = 1)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
    )
    monkeypatch.setattr(
        comp_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"k_pref": meta, "s_pref": meta}),
    )
    comp.state_cache = None
    comp.k_cache_prefix = "k_pref"
    comp._static_forward_context = {"k_pref": SimpleNamespace(cache_num_states=4)}
    comp.head_dim = 4
    comp.norm = SimpleNamespace(weight=norm_weight)
    comp.rms_norm_eps = 1e-6
    comp.compress_ratio = 1
    out_r1 = comp.forward(p_flipped, positions[:3])
    assert out_r1 is not None and out_r1.shape == (3, 4)

    # 3. self.state_cache is not None (ratio = 2, without native cache)
    comp.compress_ratio = 2
    sc_obj = _DummyCacheBinding()
    sc_obj.prefix = "s_pref"
    sc_obj.bind_kv_cache(torch.zeros((2, 4, 5), dtype=torch.float32))
    comp.state_cache = sc_obj
    out_r2 = comp.forward(p_flipped, positions[:3])
    assert out_r2 is not None and out_r2.shape == (3, 4)
