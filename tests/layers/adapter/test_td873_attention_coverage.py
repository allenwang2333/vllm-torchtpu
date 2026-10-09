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

"""Comprehensive CPU unit tests and optimization verification for TD8.73:
`src/vllm_torchtpu/layers/adapter/attention.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

import vllm_torchtpu.layers.adapter.attention as attn_mod
import vllm_torchtpu.layers.core.attention_interface as core_attn_if
from vllm_torchtpu import envs
from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl,
    PallasMLAttentionBackend,
    PallasMLAttentionBackendImpl,
    VllmTPUDeepseekV32IndexerBackend,
    _extract_scalar_scale,
    _get_kernel_entry_param_names,
    _is_fp8_kv_cache_dtype,
    _pallas_rpa_kernel_batched,
    _pallas_rpa_kernel_default,
    _pallas_rpa_kernel_default_bundled,
    _pallas_rpa_kernel_impl,
    _pallas_rpa_kernel_local,
    _reshape_packed_kv_cache,
    _resolve_kv_cache_dtype,
    _weight_by_lse,
    all_gather_heads,
    all_gather_q_nope_pe_heads,
    get_dtype_packing,
    get_tpu_min_page_size,
    merge_lse_partials_scatter_heads,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind


def _make_fake_vllm_config(
    *,
    layout: VllmKVCacheLayout = VllmKVCacheLayout.LBNHC,
    pcp_size: int = 1,
    dcp_size: int = 1,
    interleave_size: int = 4,
    max_model_len: int = 1024,
    max_num_seqs: int = 16,
    speculative_tokens: int | None = None,
):
    spec = (
        SimpleNamespace(num_speculative_tokens=speculative_tokens)
        if speculative_tokens is not None
        else None
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
    )


@contextmanager
def _patched_vllm_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vllm_config=None,
    mesh=None,
    kv_cache_bundle=None,
    bm_enabled: bool = False,
):
    if vllm_config is None:
        vllm_config = _make_fake_vllm_config()
    if mesh is None:
        mesh = SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})
    ctx = SimpleNamespace(
        vllm_config=vllm_config,
        mesh=mesh,
        kv_cache_bundle=kv_cache_bundle,
    )
    monkeypatch.setattr(attn_mod, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        attn_mod, "get_current_vllm_config_or_none", lambda: vllm_config
    )
    monkeypatch.setattr(attn_mod, "get_vllm_model_wrapper_context", lambda: ctx)
    monkeypatch.setattr(
        attn_mod, "unified_block_major_enabled", lambda _cfg: bm_enabled
    )
    yield ctx


class _FakeJaxOp:
    def __init__(self, name, fn, **kwargs):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self.fake_fn = None

    def register_fake(self, fake_fn):
        self.fake_fn = fake_fn

    def __call__(self, *args, **kwargs):
        if self.fake_fn is not None:
            return self.fake_fn(*args, **kwargs)
        return self.fn(*args, **kwargs)


def test_dtype_and_scale_helpers(monkeypatch: pytest.MonkeyPatch):
    assert _is_fp8_kv_cache_dtype("fp8")
    assert _is_fp8_kv_cache_dtype(" FP8_E4M3 ")
    assert _is_fp8_kv_cache_dtype("fp8_e5m2")
    assert not _is_fp8_kv_cache_dtype("bfloat16")
    assert not _is_fp8_kv_cache_dtype("fp8_ds_mla")

    assert _resolve_kv_cache_dtype(torch.bfloat16) == torch.bfloat16
    assert _resolve_kv_cache_dtype(" bfloat16 ") == torch.bfloat16
    with pytest.raises(ValueError, match="cache_dtype='auto'"):
        _resolve_kv_cache_dtype("auto")
    with pytest.raises(ValueError, match="Unsupported KV cache dtype string"):
        _resolve_kv_cache_dtype("unsupported_dtype")

    get_dtype_packing.cache_clear()
    assert get_dtype_packing(torch.bfloat16, 32) == 2
    assert get_dtype_packing(torch.bfloat16, 32) == 2
    assert get_dtype_packing.cache_info().hits >= 1
    assert get_dtype_packing(torch.uint8, 32) == 4
    with pytest.raises(ValueError, match="The bit width must divide"):
        get_dtype_packing(torch.bfloat16, 24)

    _get_kernel_entry_param_names.cache_clear()
    names = _get_kernel_entry_param_names(_pallas_rpa_kernel_batched)
    assert "kv_layout" in names and "decode_query_size" in names
    assert _get_kernel_entry_param_names(_pallas_rpa_kernel_batched) is names
    assert _get_kernel_entry_param_names.cache_info().hits >= 1

    # _extract_scalar_scale branches
    layer_float = SimpleNamespace(_q_scale_float=1.25)
    assert _extract_scalar_scale(layer_float, "_q_scale_float", "_q_scale") == 1.25

    layer_none = SimpleNamespace()
    assert _extract_scalar_scale(layer_none, "_q_scale_float", "_q_scale") is None

    layer_0d = SimpleNamespace(_q_scale_float=None, _q_scale=torch.tensor(2.5))
    assert _extract_scalar_scale(layer_0d, "_q_scale_float", "_q_scale") == 2.5

    layer_1d = SimpleNamespace(_q_scale_float=None, _q_scale=torch.tensor([3.5]))
    assert _extract_scalar_scale(layer_1d, "_q_scale_float", "_q_scale") == [3.5]

    layer_scalar = SimpleNamespace(_q_scale_float=None, _q_scale=4.5)
    assert _extract_scalar_scale(layer_scalar, "_q_scale_float", "_q_scale") == 4.5

    # DCP MLA helper functions in attention.py
    q = torch.ones((2, 3, 4), dtype=torch.bfloat16)
    q_pe = torch.ones((2, 3, 2), dtype=torch.bfloat16)
    lse = torch.zeros((2, 6), dtype=torch.float32)
    out = torch.ones((2, 6, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(attn_mod, "_get_dcp_group", lambda: None)
    assert all_gather_heads(q) is q
    assert all_gather_q_nope_pe_heads(q, q_pe, 4) == (q, q_pe)
    assert merge_lse_partials_scatter_heads(out, lse) is out

    class _DummyDcpGroup:
        world_size = 2

        def all_gather(self, x, dim=0):
            return torch.cat([x, x], dim=dim)

        def reduce_scatter(self, x, dim=0):
            return x[: x.shape[0] // 2]

    monkeypatch.setattr(attn_mod, "_get_dcp_group", _DummyDcpGroup)
    assert all_gather_heads(q).shape == (2, 6, 4)
    g_nope, g_pe = all_gather_q_nope_pe_heads(q, q_pe, 4)
    assert g_nope.shape == (2, 6, 4) and g_pe.shape == (2, 6, 2)
    merged = merge_lse_partials_scatter_heads(out, lse.to(torch.bfloat16))
    assert merged.shape == (2, 3, 4)

    class _ListAllGatherGroup:
        def all_gather(self, x, dim=0):
            return [x, x]

    w_hm = _weight_by_lse(out, lse, _ListAllGatherGroup())
    assert w_hm.shape == (6, 2, 4)


class _BadShapeJaxOp:
    def register_fake(self, _fn):
        pass

    def __call__(self, _first, query_or_idx, *args, **kwargs):
        out = args[0] if args else query_or_idx
        return torch.zeros((1,)), out


def test_min_page_size_and_reshape_packed_kv_cache():
    cfg_pcp = _make_fake_vllm_config(pcp_size=2)
    assert get_tpu_min_page_size(cfg_pcp) == 1

    cfg_dcp = _make_fake_vllm_config(
        pcp_size=1, dcp_size=2, max_model_len=131072, max_num_seqs=16
    )
    assert get_tpu_min_page_size(cfg_dcp) == 8

    arr = jnp.zeros((2, 4, 2, 2, 128), dtype=jnp.bfloat16)
    same_tail = _reshape_packed_kv_cache(arr, (4, 2, 2, 2, 128))
    assert same_tail.shape == (4, 2, 2, 2, 128)

    diff_tail = _reshape_packed_kv_cache(arr, (4, 4, 2, 2, 64))
    assert diff_tail.shape == (4, 4, 2, 2, 64)


def test_pallas_rpa_kernel_entries(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_attention(kv_cache, query, key, value, metadata, mesh, **kwargs):
        captured["kv_cache"] = kv_cache
        captured["metadata"] = metadata
        captured["kwargs"] = kwargs
        return kv_cache, query

    monkeypatch.setattr(attn_mod, "attention", fake_attention)
    monkeypatch.setattr(attn_mod, "get_chip_version", lambda: 7)

    mesh = SimpleNamespace()
    q = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((4, 1, 128), dtype=jnp.bfloat16)
    v = jnp.zeros((4, 1, 128), dtype=jnp.bfloat16)
    seq_lens = jnp.ones((1,), dtype=jnp.int32)
    block_tables = jnp.ones((1, 2), dtype=jnp.int32)
    qsl = jnp.array([0, 4], dtype=jnp.int32)
    req_dist = jnp.array([0, 0, 1], dtype=jnp.int32)

    # 1. 5D standard via _pallas_rpa_kernel_default
    kv_5d = jnp.zeros((2, 16, 1, 2, 128), dtype=jnp.bfloat16)
    new_kv, out = _pallas_rpa_kernel_default(
        kv_5d,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
    )
    assert new_kv.shape == kv_5d.shape and out.shape == q.shape

    # 2. _pallas_rpa_kernel_local
    new_kv, out = _pallas_rpa_kernel_local(
        kv_5d,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
    )
    assert captured["kwargs"]["shard"] is False
    assert captured["kwargs"]["kv_block_cap"] == attn_mod._DRAFT_KV_BLOCK_CAP

    # 3. _pallas_rpa_kernel_batched with decode_query_size=1 and 2, block_major=True
    kv_6d = jnp.zeros((2, 2, 16, 1, 2, 128), dtype=jnp.bfloat16)
    new_kv, out = _pallas_rpa_kernel_batched(
        kv_6d,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_query_size=1,
        block_major=True,
    )
    assert new_kv.shape == kv_6d.shape

    new_kv, out = _pallas_rpa_kernel_batched(
        kv_5d,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_query_size=2,
        block_major=False,
    )
    assert isinstance(captured["kwargs"]["rpa_func"], attn_mod.functools.partial)

    # 4. 4D kv_cache with HEAD_ALONG_SUBLANE and SEQ_ALONG_LANE
    kv_4d_nhd = jnp.zeros((2, 16, 2, 128), dtype=jnp.bfloat16)
    new_kv, _ = _pallas_rpa_kernel_impl(
        kv_4d_nhd,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
        rpa_func=attn_mod.ragged_paged_attention,
        kv_layout=batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    assert new_kv.shape == kv_4d_nhd.shape

    monkeypatch.setattr(
        attn_mod.rpa_batched_wrapper,
        "get_kv_cache_shape",
        lambda **kw: (1, 16, 1, 2, 128),
    )
    new_kv, _ = _pallas_rpa_kernel_impl(
        kv_4d_nhd,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
        rpa_func=attn_mod.ragged_paged_attention_batched,
        kv_layout=batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
    )
    assert new_kv.shape == kv_4d_nhd.shape

    # 5. _pallas_rpa_kernel_default_bundled
    monkeypatch.setattr(
        attn_mod,
        "attention_bundled",
        lambda bundle, layer_idx, query, *a, **kw: (bundle, query),
    )
    layer_idx = jnp.array(0, dtype=jnp.int32)
    with pytest.raises(NotImplementedError, match="attention sinks are not supported"):
        _pallas_rpa_kernel_default_bundled(
            kv_5d,
            layer_idx,
            q,
            k,
            v,
            seq_lens,
            block_tables,
            qsl,
            req_dist,
            jnp.ones((2,)),
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=None,
        )
    b_out, q_out = _pallas_rpa_kernel_default_bundled(
        kv_5d,
        layer_idx,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        qsl,
        req_dist,
        None,
        None,
        None,
        None,
        mesh=mesh,
        sliding_window=None,
    )
    assert b_out.shape == kv_5d.shape and q_out.shape == q.shape


def test_pallas_attention_backends(monkeypatch: pytest.MonkeyPatch):
    assert PallasAttentionBackend.get_name() == "FLASH_ATTN"
    assert PallasAttentionBackend.get_impl_cls() is PallasAttentionBackendImpl
    assert PallasAttentionBackend.get_kv_cache_shape(2, 16, 4, 64, "auto") == (
        2,
        16,
        4,
        1,
        128,
    )
    assert PallasAttentionBackend.get_kv_cache_shape(2, 16, 4, 128, "auto") == (
        2,
        16,
        8,
        1,
        128,
    )
    assert PallasAttentionBackend.get_kv_cache_shape(2, 16, 4, 64, "bfloat16") == (
        2,
        16,
        2,
        2,
        128,
    )
    assert PallasAttentionBackend.get_kv_cache_shape(2, 16, 4, 128, "bfloat16") == (
        2,
        16,
        4,
        2,
        128,
    )
    assert (
        PallasAttentionBackend.get_kv_cache_page_size_bytes(16, 4, 128, "bfloat16")
        == 16 * 4 * 2 * 128 * 2
    )
    with pytest.raises(RuntimeError, match="swap_blocks is not used"):
        PallasAttentionBackend.swap_blocks(
            torch.empty(0), torch.empty(0), torch.empty(0)
        )
    assert (
        PallasAttentionBackend.get_min_page_size(_make_fake_vllm_config(pcp_size=2))
        == 1
    )

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
    assert PallasAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.LBNHC,
    )
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)
    assert PallasAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.BLNHC,
    )
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )
    customized = PallasAttentionBackend.customize_spec(spec)
    assert customized.num_head_slots == 1

    assert PallasAttentionBackend.get_kv_cache_stride_order(False) == (0, 1, 2, 3, 4)
    assert PallasAttentionBackend.get_kv_cache_stride_order(True) == (1, 0, 2, 3, 4, 5)
    assert PallasAttentionBackend.get_max_num_seqs(1024, 16) > 0

    assert (
        PallasAttentionBackend.get_page_size(
            _make_fake_vllm_config(max_model_len=16384)
        )
        == 16
    )
    assert (
        PallasAttentionBackend.get_page_size(_make_fake_vllm_config(max_model_len=128))
        == 16
    )
    assert (
        PallasAttentionBackend.get_page_size(_make_fake_vllm_config(max_model_len=8192))
        == 256
    )
    assert (
        PallasAttentionBackend.get_page_size(_make_fake_vllm_config(max_model_len=1024))
        == 64
    )

    # PallasBatchedRPAAttentionBackend
    assert PallasBatchedRPAAttentionBackend.get_name() == "CUSTOM"
    assert (
        PallasBatchedRPAAttentionBackend.get_impl_cls()
        is PallasBatchedRPAAttentionBackendImpl
    )
    assert PallasBatchedRPAAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.LBNHC,
        VllmKVCacheLayout.LBHNC,
    )
    with _patched_vllm_env(
        monkeypatch, vllm_config=_make_fake_vllm_config(layout=VllmKVCacheLayout.LBNHC)
    ):
        assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            2, 16, 4, 128, "bfloat16"
        ) == (2, 16, 4, 2, 128)
        assert PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes() == [
            128,
            256,
        ]

    monkeypatch.setattr(attn_mod, "get_chip_version", lambda: 7)
    monkeypatch.setattr(
        attn_mod.rpa_batched_wrapper,
        "get_kv_cache_shape",
        lambda **kw: (2, 4, 64, 2, 16),
    )
    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(layout=VllmKVCacheLayout.LBHNC, pcp_size=2),
    ):
        assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            2, 16, 4, 128, "bfloat16"
        ) == (2, 4, 64, 2, 16)
        assert PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes() == [
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]


def test_pallas_attention_impl_init_and_build_kernels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(attn_mod.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(
        attn_mod,
        "pcp_streaming_jax_op",
        lambda name, fn, **kw: _FakeJaxOp(name, fn, **kw),
    )
    monkeypatch.setattr(attn_mod, "get_pcp_streaming_mesh", lambda: SimpleNamespace())
    monkeypatch.setattr(
        attn_mod, "make_pcp_streaming_rpa_kernel", lambda **kw: (lambda *a, **k: a[:2])
    )
    PallasAttentionBackendImpl._kernel_registry.clear()
    PallasAttentionBackendImpl._bundled_kernel_registry.clear()

    with _patched_vllm_env(monkeypatch):
        with pytest.raises(NotImplementedError, match="Alibi slopes"):
            PallasAttentionBackendImpl(2, 128, 1.0, 1, [0.5], None, "bfloat16")
        with pytest.raises(NotImplementedError, match="Encoder self-attention"):
            PallasAttentionBackendImpl(
                2, 128, 1.0, 1, None, None, "bfloat16", attn_type="encoder"
            )

        sinks = torch.ones(2, dtype=torch.bfloat16)
        impl = PallasAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "fp8", sinks=sinks
        )
        assert impl.kv_cache_quantized_dtype == torch.float8_e4m3fn
        assert not impl.runs_batched_rpa_schedule()
        impl.process_weights_after_loading(torch.bfloat16)
        assert impl.sinks is not None and impl.sinks.dtype == torch.float32

        impl_no_sinks = PallasAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "bfloat16"
        )
        impl_no_sinks.process_weights_after_loading(torch.bfloat16)

        # Kernel config cache & registry reuse
        k1 = impl_no_sinks._build_rpa_kernel(None, None, None)
        k2 = impl_no_sinks._build_rpa_kernel(None, None, None)
        assert k1 is k2
        impl_no_sinks._kernel_config_cache.clear()
        k3 = impl_no_sinks._build_rpa_kernel(None, None, None)
        assert k1 is k3

        # Execute rpa_kernel_impl (both compatible and incompatible kv_cache shapes)
        kv = torch.zeros((2, 16, 1, 2, 128))
        q = torch.ones((4, 2, 128))
        out = k1(kv, q)
        assert out.shape == q.shape

        # Trigger shape mismatch RuntimeError in rpa_kernel_impl
        impl_bad = PallasAttentionBackendImpl(2, 128, 0.5, 1, None, 64, "bfloat16")
        monkeypatch.setattr(
            attn_mod.pallas,
            "jax_op",
            lambda name, fn, **kw: _BadShapeJaxOp(),
        )
        k_bad = impl_bad._build_rpa_kernel(None, None, None)
        with pytest.raises(RuntimeError, match="incompatible KV cache shape"):
            k_bad(kv, q)

    # PCP streaming + block_major + TPU_KERNEL_ITER_MODE
    monkeypatch.setattr(attn_mod.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(envs, "TPU_KERNEL_ITER_MODE", True)
    registered_builders = {}
    import vllm_torchtpu.compilation.kernel_reload as kernel_reload
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
        vllm_adapter as pcp_adapter,
    )

    monkeypatch.setattr(
        kernel_reload,
        "register_builder",
        lambda name, fn: registered_builders.setdefault(name, fn),
    )
    monkeypatch.setattr(
        pcp_adapter,
        "make_pcp_streaming_rpa_kernel",
        lambda **kw: (lambda *a, **k: a[:2]),
    )
    monkeypatch.setattr(
        pcp_adapter,
        "build_pcp_streaming_callable",
        lambda *a, **kw: "rebuilt_pcp_callable",
    )
    monkeypatch.setattr(
        attn_mod,
        "invoke_pcp_streaming_op",
        lambda op, kv_c, args, kwargs: (kv_c.clone(), args[0]),
    )

    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(pcp_size=2),
        bm_enabled=True,
    ):
        impl_pcp = PallasAttentionBackendImpl(2, 128, 1.0, 1, None, None, "bfloat16")
        layout = BlockMajorPoolLayout(
            num_blocks=2, num_pools=2, split=1, pool_index_by_layer={}
        )
        impl_pcp.set_block_major_pool(0, layout)
        k_pcp = impl_pcp._build_rpa_kernel(
            None, None, None, use_pcp_streaming=True, block_major=True
        )
        out_pcp = k_pcp(torch.zeros((2, 2, 16, 1, 2, 128)), torch.ones((4, 2, 128)))
        assert out_pcp.shape == (4, 2, 128)
        assert len(registered_builders) == 1
        builder_fn = next(iter(registered_builders.values()))
        assert builder_fn() == "rebuilt_pcp_callable"

        # block_major with entry that lacks block_major param
        impl_pcp._kernel_entry = _pallas_rpa_kernel_local
        with pytest.raises(
            NotImplementedError, match="cannot address the block-major pool"
        ):
            impl_pcp._build_rpa_kernel(
                None, None, None, use_pcp_streaming=False, block_major=True
            )
    monkeypatch.setattr(envs, "TPU_KERNEL_ITER_MODE", False)

    # Batched impl with speculative config and SEQ_ALONG_LANE draft error
    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(
            layout=VllmKVCacheLayout.LBHNC, speculative_tokens=3
        ),
    ):
        batched_impl = PallasBatchedRPAAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "fp8"
        )
        assert batched_impl.decode_query_size == 4
        k_batched = batched_impl._build_rpa_kernel(
            None, 1.0, 1.0, use_pcp_streaming=False, block_major=True
        )
        assert callable(k_batched)

        # Rebind to _pallas_rpa_kernel_local under SEQ_ALONG_LANE -> NotImplementedError
        batched_impl._kernel_entry = _pallas_rpa_kernel_local
        batched_impl._kernel_config_cache.clear()
        with pytest.raises(NotImplementedError, match="allocated SEQ_ALONG_LANE pages"):
            batched_impl._build_rpa_kernel(None, 1.0, 1.0, use_pcp_streaming=False)


def test_pallas_attention_bundled_and_initialize_and_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(attn_mod.pallas, "jax_op", _FakeJaxOp)
    PallasAttentionBackendImpl._kernel_registry.clear()
    PallasAttentionBackendImpl._bundled_kernel_registry.clear()
    monkeypatch.setattr(attn_mod, "synchronize_tensors", lambda *a, **kw: None)

    with _patched_vllm_env(monkeypatch):
        batched = PallasBatchedRPAAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "bfloat16"
        )
        with pytest.raises(NotImplementedError, match="has no bundled variant"):
            batched._build_bundled_kernel(None, None, None)

        impl = PallasAttentionBackendImpl(2, 128, 1.0, 1, None, None, "fp8")
        layer = SimpleNamespace(_k_scale_float=0.0, _v_scale_float=0.0)
        impl.setup_bundled(0, torch.device("cpu"), layer)
        layer_nonzero = SimpleNamespace(_k_scale_float=1.5, _v_scale_float=2.5)
        impl.setup_bundled(1, torch.device("cpu"), layer_nonzero)
        # Hit _bundled_kernel_registry cache
        impl.setup_bundled(2, torch.device("cpu"), layer_nonzero)

        bundle = torch.zeros((2, 4, 16, 1, 2, 128))
        q = torch.ones((4, 2, 128))
        assert impl.rpa_kernel_bundled is not None
        out = impl.rpa_kernel_bundled(bundle, impl._bundle_layer_idx_tensor, q)
        assert out.shape == q.shape

        # Bundled shape mismatch RuntimeError
        monkeypatch.setattr(
            attn_mod.pallas,
            "jax_op",
            lambda name, fn, **kw: _BadShapeJaxOp(),
        )
        PallasAttentionBackendImpl._bundled_kernel_registry.clear()
        impl_bad_bundle = PallasAttentionBackendImpl(
            2, 128, 0.25, 1, None, None, "bfloat16"
        )
        impl_bad_bundle.setup_bundled(0, torch.device("cpu"), layer)
        assert impl_bad_bundle.rpa_kernel_bundled is not None
        with pytest.raises(RuntimeError, match="incompatible bundle shape"):
            impl_bad_bundle.rpa_kernel_bundled(
                bundle, impl_bad_bundle._bundle_layer_idx_tensor, q
            )

    monkeypatch.setattr(attn_mod.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(
        attn_mod,
        "pcp_streaming_jax_op",
        lambda name, fn, **kw: _FakeJaxOp(name, fn, **kw),
    )
    monkeypatch.setattr(attn_mod, "get_pcp_streaming_mesh", lambda: SimpleNamespace())
    monkeypatch.setattr(
        attn_mod, "make_pcp_streaming_rpa_kernel", lambda **kw: (lambda *a, **k: a[:2])
    )

    # _validate_pcp_streaming_support errors and valid HND+CUSTOM+FP8
    with _patched_vllm_env(
        monkeypatch, vllm_config=_make_fake_vllm_config(layout=VllmKVCacheLayout.LBHNC)
    ):
        impl_hnd_non_custom = PallasAttentionBackendImpl(
            2,
            128,
            1.0,
            1,
            None,
            None,
            "bfloat16",
            logits_soft_cap=30.0,
            sinks=torch.ones(2),
        )
        impl_hnd_non_custom.kv_layout = batched_rpa_configs.KVLayout.SEQ_ALONG_LANE
        with pytest.raises(NotImplementedError, match="PCP streaming attention"):
            impl_hnd_non_custom._validate_pcp_streaming_support(skip_kv_update=True)

        impl_hnd_custom_fp8 = PallasBatchedRPAAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "fp8"
        )
        impl_hnd_custom_fp8._validate_pcp_streaming_support(skip_kv_update=False)

    # initialize_kernel: PCP, DCP (errors + group None + group present), standard
    with _patched_vllm_env(monkeypatch, vllm_config=_make_fake_vllm_config(pcp_size=2)):
        impl_pcp = PallasAttentionBackendImpl(2, 128, 1.0, 1, None, None, "fp8")
        impl_pcp.initialize_kernel(
            SimpleNamespace(_k_scale_float=0.0, _v_scale_float=0.0)
        )
        impl_pcp.initialize_kernel(
            SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
        )
        assert impl_pcp._pcp_streaming is True

    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(layout=VllmKVCacheLayout.LBHNC, dcp_size=2),
    ):
        impl_dcp_bad = PallasBatchedRPAAttentionBackendImpl(
            2, 64, 1.0, 1, None, None, "bfloat16"
        )
        with pytest.raises(
            NotImplementedError, match="DCP with HND requires the CUSTOM"
        ):
            impl_dcp_bad.initialize_kernel(SimpleNamespace())

        impl_dcp_ok = PallasBatchedRPAAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "bfloat16"
        )
        monkeypatch.setattr(attn_mod, "_get_dcp_group", lambda: None)
        impl_dcp_ok.initialize_kernel(SimpleNamespace())
        assert impl_dcp_ok.dcp_world_size == 1

        monkeypatch.setattr(
            attn_mod,
            "_get_dcp_group",
            lambda: SimpleNamespace(world_size=2, rank_in_group=1),
        )
        monkeypatch.setattr(
            attn_mod, "_build_dcp_kernels", lambda **kw: ("cache_k", "new_k")
        )
        impl_dcp_ok.initialize_kernel(SimpleNamespace())
        assert impl_dcp_ok.dcp_world_size == 2 and impl_dcp_ok.dcp_rank == 1

    with _patched_vllm_env(monkeypatch):
        impl_std = PallasAttentionBackendImpl(2, 128, 1.0, 1, None, None, "bfloat16")
        impl_std.initialize_kernel(SimpleNamespace())
        assert impl_std.rpa_kernel is not None

        # forward branches:
        layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
        q3 = torch.ones((4, 2, 128))
        k3 = torch.ones((4, 1, 128))
        v3 = torch.ones((4, 1, 128))
        meta = AttentionMetadata(
            input_positions=torch.zeros(4, dtype=torch.int32),
            block_tables=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )
        with pytest.raises(
            NotImplementedError, match="fused output quantization is not yet supported"
        ):
            impl_std.forward(
                layer, q3, k3, v3, torch.empty(0), meta, output_scale=torch.ones(1)
            )

        # Empty kv_cache probe: output=None and output provided
        out_probe = impl_std.forward(layer, q3, k3, v3, torch.empty(0), meta)
        assert out_probe.shape == q3.shape
        given_out = torch.zeros_like(q3)
        assert (
            impl_std.forward(layer, q3, k3, v3, torch.empty(0), meta, output=given_out)
            is given_out
        )

        # 4D kv_cache + 2D query/key + output provided
        kv_4d = torch.zeros((2, 16, 2, 128))
        out_buf_2d = torch.zeros((4, 256))
        res_2d = impl_std.forward(
            layer,
            q3.view(4, 256),
            k3.view(4, 128),
            v3.view(4, 128),
            kv_4d,
            meta,
            output=out_buf_2d,
        )
        assert res_2d.shape == (4, 256) and torch.equal(out_buf_2d, res_2d)

        # Wider pool padding + torch.compiler.is_compiling()=True
        impl_narrow = PallasAttentionBackendImpl(2, 64, 1.0, 1, None, None, "fp8")
        kv_wide = torch.zeros((2, 16, 1, 2, 128))
        with pytest.raises(
            ValueError, match="k_scale_float and v_scale_float must be non-zero"
        ):
            impl_narrow.forward(
                SimpleNamespace(_k_scale_float=0.0, _v_scale_float=1.0),
                torch.ones((4, 2, 64)),
                torch.ones((4, 1, 64)),
                torch.ones((4, 1, 64)),
                kv_wide,
                meta,
            )
        monkeypatch.setattr(attn_mod, "quantize_kv", lambda dtype, k, v, ks, vs: (k, v))
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        res_narrow = impl_narrow.forward(
            layer,
            torch.ones((4, 2, 64)),
            torch.ones((4, 1, 64)),
            torch.ones((4, 1, 64)),
            kv_wide,
            meta,
        )
        assert res_narrow.shape == (4, 2, 64)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

        # SEQ_ALONG_LANE pool_head_dim resolution
        impl_seq = PallasBatchedRPAAttentionBackendImpl(
            2, 64, 1.0, 1, None, None, "bfloat16"
        )
        impl_seq._pool_is_seq_along_lane = True
        kv_seq = torch.zeros((2, 2, 64, 2, 16))
        res_seq = impl_seq.forward(
            layer,
            torch.ones((4, 2, 64)),
            torch.ones((4, 1, 64)),
            torch.ones((4, 1, 64)),
            kv_seq,
            meta,
        )
        assert res_seq.shape == (4, 2, 64)

        # DCP forward branches (error + 2D padded + 3D unpadded)
        impl_dcp = PallasAttentionBackendImpl(2, 64, 1.0, 1, None, None, "bfloat16")
        impl_dcp.dcp_world_size = 2
        impl_dcp.sinks = torch.ones(2)
        with pytest.raises(
            NotImplementedError, match="DCP attention does not support attention sinks"
        ):
            impl_dcp.forward(
                layer,
                torch.ones((4, 2, 64)),
                torch.ones((4, 1, 64)),
                torch.ones((4, 1, 64)),
                kv_wide,
                meta,
            )
        impl_dcp.sinks = None
        monkeypatch.setattr(attn_mod, "_forward_with_dcp", lambda **kw: kw["query"])
        dcp_out_2d = torch.zeros((4, 128))
        res_dcp_2d = impl_dcp.forward(
            layer,
            torch.ones((4, 128)),
            torch.ones((4, 64)),
            torch.ones((4, 64)),
            kv_wide,
            meta,
            output=dcp_out_2d,
        )
        assert res_dcp_2d.shape == (4, 128)
        res_dcp_3d = impl_dcp.forward(
            layer,
            torch.ones((4, 2, 64)),
            torch.ones((4, 1, 64)),
            torch.ones((4, 1, 64)),
            torch.zeros((2, 16, 1, 2, 64)),
            meta,
        )
        assert res_dcp_3d.shape == (4, 2, 64)

    # PCP configured mismatch, valid PCP streaming, bundled, and bm_kernel forward
    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(pcp_size=2, interleave_size=8),
    ) as ctx:
        impl_pcp_fwd = PallasAttentionBackendImpl(
            2, 128, 1.0, 1, None, None, "bfloat16"
        )
        with pytest.raises(RuntimeError, match="PCP is configured, but attention"):
            impl_pcp_fwd.forward(
                layer, q3, k3, v3, torch.zeros((2, 16, 1, 2, 128)), meta
            )

        meta_pcp = AttentionMetadata(
            input_positions=torch.zeros(4, dtype=torch.int32),
            block_tables=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
            sequence_layout_kind=SequenceLayoutKind.PARTIAL,
        )
        monkeypatch.setattr(
            attn_mod,
            "is_pcp_streaming_attention_metadata",
            lambda m: m is meta_pcp,
        )
        monkeypatch.setattr(
            attn_mod,
            "invoke_pcp_streaming_op",
            lambda op, kv_c, args, kwargs: (kv_c.clone(), args[0]),
        )
        res_pcp = impl_pcp_fwd.forward(
            layer, q3, k3, v3, torch.zeros((2, 16, 1, 2, 128)), meta_pcp
        )
        assert res_pcp.shape == (4, 2, 128)

        # Bundled + PCP streaming -> NotImplementedError
        ctx.kv_cache_bundle = torch.zeros((2, 4, 16, 1, 2, 128))
        with pytest.raises(
            NotImplementedError, match="does not support PCP streaming or KV sharing"
        ):
            impl_pcp_fwd.forward(
                layer, q3, k3, v3, torch.zeros((2, 16, 1, 2, 128)), meta_pcp
            )

    with _patched_vllm_env(
        monkeypatch,
        kv_cache_bundle=torch.zeros((2, 4, 16, 1, 2, 128)),
        bm_enabled=True,
    ) as ctx:
        impl_bm = PallasAttentionBackendImpl(2, 128, 1.0, 1, None, None, "bfloat16")
        impl_bm.setup_bundled(0, torch.device("cpu"), layer)
        res_bundled = impl_bm.forward(
            layer, q3, k3, v3, torch.zeros((2, 16, 1, 2, 128)), meta
        )
        assert res_bundled.shape == (4, 2, 128)

        # Non-bundled bm_kernel path (flat_kernel_ids)
        ctx.kv_cache_bundle = None
        impl_bm.set_block_major_pool(
            0,
            BlockMajorPoolLayout(
                num_blocks=2, num_pools=2, split=1, pool_index_by_layer={}
            ),
        )
        res_bm = impl_bm.forward(
            layer, q3, k3, v3, torch.zeros((2, 2, 16, 1, 2, 128)), meta
        )
        assert res_bm.shape == (4, 2, 128)


def test_mla_backend_and_indexer_and_impl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(attn_mod.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(
        attn_mod,
        "pcp_streaming_jax_op",
        lambda name, fn, **kw: _FakeJaxOp(name, fn, **kw),
    )

    assert PallasMLAttentionBackend.get_name() == "FLASH_ATTN_MLA"
    assert PallasMLAttentionBackend.is_mla() is True
    assert PallasMLAttentionBackend.get_impl_cls() is PallasMLAttentionBackendImpl
    assert PallasMLAttentionBackend._is_ds_mla_packed_cache("fp8_ds_mla")
    assert PallasMLAttentionBackend._is_ds_mla_packed_cache(torch.uint8)
    assert not PallasMLAttentionBackend._is_ds_mla_packed_cache("bfloat16")

    assert PallasMLAttentionBackend.get_kv_cache_shape(2, 16, 1, 576, "auto") == (
        2,
        16,
        1,
        640,
    )
    s_packed_false = PallasMLAttentionBackend.get_kv_cache_shape(
        2, 16, 1, 576, "fp8_ds_mla", head_size_is_packed_width=False
    )
    s_packed_true = PallasMLAttentionBackend.get_kv_cache_shape(
        2, 16, 1, 648, "fp8_ds_mla", head_size_is_packed_width=True
    )
    assert s_packed_false == s_packed_true
    assert (
        len(PallasMLAttentionBackend.get_kv_cache_shape(2, 16, 1, 576, "bfloat16")) == 4
    )

    nope_spec, rope_spec = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        2, 16, 576, "fp8"
    )
    assert nope_spec.head_dim == 512 and rope_spec.head_dim == 128

    assert (
        PallasMLAttentionBackend.get_kv_cache_page_size_bytes(16, 1, 576, "bfloat16")
        > 0
    )
    with pytest.raises(RuntimeError, match="swap_blocks is not used"):
        PallasMLAttentionBackend.swap_blocks(
            torch.empty(0), torch.empty(0), torch.empty(0)
        )
    vcfg = _make_fake_vllm_config(pcp_size=2)
    assert PallasMLAttentionBackend.get_page_size(vcfg) == 1024
    assert PallasMLAttentionBackend.get_min_page_size(vcfg) == 1

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
    assert PallasMLAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.LBNHC,
    )
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)
    assert PallasMLAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.BLNHC,
    )
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    assert PallasMLAttentionBackend.customize_spec(spec).num_head_slots == 1
    assert PallasMLAttentionBackend.get_kv_cache_stride_order(False) == (0, 1, 2, 3)
    assert PallasMLAttentionBackend.get_kv_cache_stride_order(True) == (1, 0, 2, 3, 4)

    # VllmTPUDeepseekV32IndexerBackend
    assert VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(2, 16, 1, 128) == (
        2,
        16,
        128,
    )
    assert VllmTPUDeepseekV32IndexerBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.LBNHC,
    )
    assert VllmTPUDeepseekV32IndexerBackend.get_name() == "TPU_STREAMINDEX_INDEXER"
    assert VllmTPUDeepseekV32IndexerBackend.get_page_size(vcfg) == 1024
    assert VllmTPUDeepseekV32IndexerBackend.get_min_page_size(vcfg) == 1

    # PallasMLAttentionBackendImpl __init__ branches
    with (
        _patched_vllm_env(
            monkeypatch, vllm_config=_make_fake_vllm_config(dcp_size=2, pcp_size=2)
        ),
        pytest.raises(NotImplementedError, match="sparse MLA has one KV-position"),
    ):
        PallasMLAttentionBackendImpl(
            4, 576, 1.0, 1, None, None, "fp8", None, "decoder", None
        )

    with (
        _patched_vllm_env(
            monkeypatch,
            vllm_config=_make_fake_vllm_config(dcp_size=2, interleave_size=3),
        ),
        pytest.raises(ValueError, match="multiple of"),
    ):
        PallasMLAttentionBackendImpl(
            4, 576, 1.0, 1, None, None, "fp8", None, "decoder", None
        )

    monkeypatch.setattr(attn_mod, "get_or_create_dcp_mesh", lambda: SimpleNamespace())
    monkeypatch.setattr(envs, "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", False)
    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(dcp_size=2, interleave_size=4),
    ):
        impl_dcp_nofuse = PallasMLAttentionBackendImpl(
            4, 576, 1.0, 1, None, None, "fp8", None, "decoder", None
        )
        assert impl_dcp_nofuse.dcp_size == 2

    monkeypatch.setattr(envs, "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", True)
    with _patched_vllm_env(
        monkeypatch,
        vllm_config=_make_fake_vllm_config(dcp_size=2, interleave_size=4),
    ):
        impl_dcp = PallasMLAttentionBackendImpl(
            4, 576, 1.0, 1, None, None, "fp8", None, "decoder", None
        )
        assert impl_dcp._dcp_mesh is not None

        # _get_kv_scales with explicit v_scale vs fallback to k_scale
        q_s, k_s, v_s = impl_dcp._get_kv_scales(
            SimpleNamespace(_q_scale_float=1.0, _k_scale_float=2.0, _v_scale_float=3.0)
        )
        assert (q_s, k_s, v_s) == (1.0, 2.0, 3.0)
        q_s2, k_s2, v_s2 = impl_dcp._get_kv_scales(
            SimpleNamespace(_q_scale_float=1.0, _k_scale_float=2.0)
        )
        assert (q_s2, k_s2, v_s2) == (1.0, 2.0, 2.0)

        # Build and invoke _build_mla_op, _build_sparse_mla_op, _build_sparse_mla_dcp_op
        # including inner core TPU functions and fake functions!
        captured_ops = {}

        def capturing_jax_op(name, fn, **kw):
            op = _FakeJaxOp(name, fn, **kw)
            captured_ops[name] = op
            return op

        monkeypatch.setattr(attn_mod.pallas, "jax_op", capturing_jax_op)
        monkeypatch.setattr(attn_mod, "pcp_streaming_jax_op", capturing_jax_op)
        monkeypatch.setattr(attn_mod, "mla_attention", lambda *a, **kw: (a[4], a[0]))
        monkeypatch.setattr(
            core_attn_if,
            "sparse_mla_attention",
            lambda *a, **kw: (a[4], a[5], a[0]),
        )
        monkeypatch.setattr(
            core_attn_if,
            "sparse_mla_attention_dcp",
            lambda *a, **kw: (
                a[4],
                a[5],
                a[0],
                jnp.zeros(a[0].shape[:2], dtype=jnp.float32),
            ),
        )

        layer_mla = SimpleNamespace(
            layer_name="model.layers.0.self_attn",
            num_heads=4,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            scale=1.0,
            kv_cache_quantized_dtype=torch.float8_e4m3fn,
            _k_scale_float=1.0,
            mla_kv_spec=(nope_spec, rope_spec),
            allow_masked_dense=True,
            W_UK_T=torch.ones((4, 128, 512), dtype=torch.float32),
            W_UK_T_scale=torch.tensor(1.0),
            W_UK_T_dcp_qrep=torch.ones((8, 128, 512), dtype=torch.bfloat16),
            W_UK_T_dcp_qrep_scale=None,
            W_UV=torch.ones((4, 512, 128), dtype=torch.float32),
            W_UV_scale=torch.tensor(1.0),
        )

        mla_fn = impl_dcp._build_mla_op(layer_mla, 1.0, 1.0, 1.0)
        sparse_fn = impl_dcp._build_sparse_mla_op(layer_mla)
        sparse_dcp_fn = impl_dcp._build_sparse_mla_dcp_op(layer_mla, 2, 4)

        # Test _build_sparse_mla_op/_dcp_op with kv_cache_quantized_dtype=None
        layer_unquant = SimpleNamespace(
            layer_name="model.layers.1.self_attn",
            num_heads=4,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            scale=1.0,
            kv_cache_quantized_dtype=None,
            mla_kv_spec=(nope_spec, rope_spec),
            W_UK_T=torch.ones((4, 128, 512), dtype=torch.bfloat16),
            W_UV=torch.ones((4, 512, 128), dtype=torch.bfloat16),
        )
        impl_dcp._build_sparse_mla_op(layer_unquant)
        impl_dcp._build_sparse_mla_dcp_op(layer_unquant, 2, 4)

        # Execute inner core_tpu functions directly to cover all MLA op closures
        q_nope_t = torch.ones((2, 4, 512), dtype=torch.bfloat16)
        q_pe_t = torch.ones((2, 4, 64), dtype=torch.bfloat16)
        kv_c_t = torch.ones((2, 512), dtype=torch.bfloat16)
        k_pe_t = torch.ones((2, 64), dtype=torch.bfloat16)
        topk_t = torch.zeros((2, 8), dtype=torch.int32)
        sl_t = torch.ones((1,), dtype=torch.int32)
        bt_t = torch.zeros((1, 2), dtype=torch.int32)
        qsl_t = torch.tensor([0, 2], dtype=torch.int32)
        rd_t = torch.tensor([0, 0, 1], dtype=torch.int32)
        kv_dense = torch.zeros((2, 16, 1, 2, 640), dtype=torch.bfloat16)
        kv_nope = torch.zeros((2, 16, 1, 4, 512), dtype=torch.uint8)
        kv_rope = torch.zeros((2, 16, 1, 2, 64), dtype=torch.bfloat16)

        assert mla_fn(
            kv_dense, q_nope_t, q_pe_t, kv_c_t, k_pe_t, sl_t, bt_t, qsl_t, rd_t
        ).shape == (2, 4, 512)
        assert sparse_fn(
            (kv_nope, kv_rope),
            q_nope_t,
            q_pe_t,
            kv_c_t,
            k_pe_t,
            topk_t,
            sl_t,
            bt_t,
            qsl_t,
            rd_t,
        ).shape == (2, 4, 512)
        p_out, p_lse = sparse_dcp_fn(
            (kv_nope, kv_rope),
            q_nope_t,
            q_pe_t,
            kv_c_t,
            k_pe_t,
            topk_t,
            sl_t,
            bt_t,
            qsl_t,
            rd_t,
        )
        assert p_out.shape == (2, 4, 512) and p_lse.shape == (2, 4)

        for op_name, op_obj in captured_ops.items():
            if (
                "sparse_mla_attention_dcp" in op_name
                or "sparse_mla_attention" in op_name
            ):
                op_obj.fn(
                    kv_nope,
                    kv_rope,
                    q_nope_t,
                    q_pe_t,
                    kv_c_t,
                    k_pe_t,
                    topk_t,
                    sl_t,
                    bt_t,
                    qsl_t,
                    rd_t,
                )
            elif "mla_attention" in op_name:
                op_obj.fn(
                    kv_dense, q_nope_t, q_pe_t, kv_c_t, k_pe_t, sl_t, bt_t, qsl_t, rd_t
                )

        # PallasMLAttentionBackendImpl.forward branches
        meta = SimpleNamespace(
            seq_lens=sl_t,
            block_tables=bt_t,
            query_start_loc=qsl_t,
            request_distribution=rd_t,
        )
        q_in = (
            torch.ones((2, 4, 128), dtype=torch.bfloat16),
            torch.ones((2, 4, 64), dtype=torch.bfloat16),
        )
        # Empty kv_cache probe: output=None and output provided
        probe_out = impl_dcp.forward(
            layer_mla, q_in, kv_c_t, k_pe_t, torch.empty(0), meta
        )
        assert probe_out.shape == (2, 512)
        given_probe = torch.zeros((2, 512), dtype=torch.bfloat16)
        assert (
            impl_dcp.forward(
                layer_mla,
                q_in,
                kv_c_t,
                k_pe_t,
                torch.empty(0),
                meta,
                output=given_probe,
            )
            is given_probe
        )

        # Dense forward (topk_indices=None) with quantize_kv & output copy
        monkeypatch.setattr(
            attn_mod, "quantize_kv", lambda dtype, t, value=None, k_scale=1.0: (t, None)
        )
        out_buf = torch.zeros((2, 512), dtype=torch.bfloat16)
        res_dense = impl_dcp.forward(
            layer_mla, q_in, kv_c_t, k_pe_t, kv_dense, meta, output=out_buf
        )
        assert res_dense.shape == (2, 512)
        # Call again when layer_mla.mla_op is already populated
        impl_dcp.forward(layer_mla, q_in, kv_c_t, k_pe_t, kv_dense, meta)

        # Sparse non-DCP forward (dcp_size == 1) with unquantized layer
        with _patched_vllm_env(
            monkeypatch, vllm_config=_make_fake_vllm_config(dcp_size=1)
        ):
            impl_nodcp = PallasMLAttentionBackendImpl(
                4, 576, 1.0, 1, None, None, "bfloat16", None, "decoder", None
            )
        assert impl_nodcp.dcp_size == 1
        res_sparse = impl_nodcp.forward(
            layer_unquant,
            q_in,
            kv_c_t,
            k_pe_t,
            (kv_nope, kv_rope),
            meta,
            topk_indices=topk_t,
        )
        assert res_sparse.shape == (2, 512)
        # Call again when layer_unquant.sparse_mla_op is already populated
        impl_nodcp.forward(
            layer_unquant,
            q_in,
            kv_c_t,
            k_pe_t,
            (kv_nope, kv_rope),
            meta,
            topk_indices=topk_t,
        )

        # Sparse DCP forward: 1) use_qrep=True, 2) fused gather, 3) unfused gather
        layer_mla.sparse_mla_dcp_op = lambda kv_c, qn, qp, *a: (
            torch.ones((2, 8, 512), dtype=torch.bfloat16),
            torch.zeros((2, 8), dtype=torch.float32),
        )
        monkeypatch.setattr(
            attn_mod,
            "merge_lse_partials_scatter_heads",
            lambda partial, lse: partial[:, :4, :],
        )
        monkeypatch.setattr(
            attn_mod,
            "all_gather_q_nope_pe_heads",
            lambda qn, qp, rank: (qn.repeat(1, 2, 1), qp.repeat(1, 2, 1)),
        )
        monkeypatch.setattr(
            attn_mod,
            "all_gather_heads",
            lambda t: t.repeat(1, 2, 1),
        )

        q_rep = (
            torch.ones((2, 8, 128), dtype=torch.bfloat16),
            torch.ones((2, 8, 64), dtype=torch.bfloat16),
        )
        res_qrep = impl_dcp.forward(
            layer_mla,
            q_in,
            kv_c_t,
            k_pe_t,
            (kv_nope, kv_rope),
            meta,
            topk_indices=topk_t,
            q_dcp_replicated=q_rep,
        )
        assert res_qrep.shape == (2, 512)

        monkeypatch.setattr(envs, "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", True)
        res_fused = impl_dcp.forward(
            layer_mla,
            q_in,
            kv_c_t,
            k_pe_t,
            (kv_nope, kv_rope),
            meta,
            topk_indices=topk_t,
        )
        assert res_fused.shape == (2, 512)

        monkeypatch.setattr(envs, "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", False)
        res_unfused = impl_dcp.forward(
            layer_mla,
            q_in,
            kv_c_t,
            k_pe_t,
            (kv_nope, kv_rope),
            meta,
            topk_indices=topk_t,
        )
        assert res_unfused.shape == (2, 512)

        # Structural no-op methods
        assert (
            impl_dcp.forward_mha(
                q_nope_t, kv_c_t, k_pe_t, kv_dense, meta, torch.ones(1), out_buf
            )
            is None
        )
        assert impl_dcp.forward_mqa(q_in, kv_dense, meta, layer_mla) is None
        assert (
            impl_dcp.do_kv_cache_update(
                kv_c_t, k_pe_t, kv_dense, torch.zeros(2), "fp8", torch.ones(1)
            )
            is None
        )
