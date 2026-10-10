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
"""Full-module CPU unit tests for DeepSeek-V4 attention and TPUWorker (TD7.84)."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_torchtpu.models.vllm.deepseek_v4.attention as attn_mod
import vllm_torchtpu.worker.tpu_worker as worker_mod
from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
    DeepseekTPUAttentionMixin,
    DeepseekV4TPUAttentionBackend,
    VllmDeepseekV4MLAAttention,
    _extract_attn_meta_fields,
    _fake_o_proj,
    _fake_rope,
    _live_cache,
    _name_float,
    _o_proj_jax,
    _pallas_op_cache,
    _qnorm_rope_jax,
    _rope_jax,
)
from vllm_torchtpu.worker.tpu_worker import (
    TPUWorker,
    _configure_tpu_process_env,
    _get_kv_connector_handshake_metadata_key,
)


class _DummyAttention(DeepseekTPUAttentionMixin):
    def __init__(self, kind: str = "csa") -> None:
        self._kind = kind
        self.prefix = "model.layers.0.self_attn"
        self.custom_prefix = self.prefix
        self.compress_ratio = 4 if kind == "csa" else (1 if kind == "swa" else 2)
        self.window_size = 128
        self.head_dim = 128
        self.n_local_heads = 16
        self.n_local_groups = 2
        self.o_lora_rank = 32
        self.attn_sink = torch.zeros(16, dtype=torch.float32)
        self.swa_cache_layer = SimpleNamespace(
            block_size=64,
            prefix=f"{self.prefix}.swa_cache",
            kv_cache=torch.empty(0, dtype=torch.uint8),
        )
        self.kv_cache: object = torch.empty(0, dtype=torch.uint8)
        self.cache_num_states = 16
        self.wo_a = SimpleNamespace(
            weight=torch.zeros((64, 1024), dtype=torch.float8_e4m3fn),
            weight_scale=torch.ones(64, dtype=torch.float32),
        )
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros((16, 64), dtype=torch.float32)
        )

    @property
    def attention_kind(self) -> str:
        return self._kind


@pytest.mark.cpu_test
def test_attention_module_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DeepseekV4TPUAttentionBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )

    called: dict[str, object] = {}

    def _mock_qnorm_rope(x, pos, cache, *, eps):
        called["qnorm_rope"] = (x, pos, cache, eps)
        return "qnorm_out"

    def _mock_rope(x, pos, cache):
        called["rope"] = (x, pos, cache)
        return "rope_out"

    def _mock_o_proj(
        x,
        pos,
        cache,
        wo_a,
        wo_a_scale,
        *,
        head_dim,
        inverse,
        quantize_activations,
    ):
        called["o_proj"] = (
            x,
            pos,
            cache,
            wo_a,
            wo_a_scale.shape,
            head_dim,
            inverse,
            quantize_activations,
        )
        return "o_proj_out"

    monkeypatch.setattr(attn_mod.rope_kernel, "qnorm_rope", _mock_qnorm_rope)
    monkeypatch.setattr(attn_mod.rope_kernel, "rope", _mock_rope)
    monkeypatch.setattr(attn_mod, "fused_reverse_rope_wo_a_projection", _mock_o_proj)

    assert _qnorm_rope_jax("x", "p", "c", eps=1e-5) == "qnorm_out"
    assert _rope_jax("x", "p", "c") == "rope_out"
    scale = torch.ones((2, 4), dtype=torch.float32)
    assert _o_proj_jax("x", "p", "c", "w", scale, head_dim=128) == "o_proj_out"
    assert called["o_proj"][4] == (8,)

    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16)
    wo_a = torch.zeros((16, 32), dtype=torch.float8_e4m3fn)
    fake_r = _fake_rope(x, None, None)
    assert fake_r.shape == x.shape and fake_r.dtype == x.dtype
    fake_o = _fake_o_proj(x, None, None, wo_a, None)
    assert fake_o.shape == (4, 32) and fake_o.dtype == x.dtype

    assert _name_float(1e-6) == "1em06"
    assert _name_float(0.125) == "0p125"

    assert _live_cache(None) is None
    assert _live_cache("not_a_tensor") is None
    assert _live_cache(torch.empty(0, dtype=torch.uint8)) is None
    live_t = torch.ones(4, dtype=torch.uint8)
    assert _live_cache(live_t) is live_t

    assert _extract_attn_meta_fields(None) == (None, None, None, None)
    dict_meta = {
        "seq_lens": "s",
        "block_tables": "b",
        "query_start_loc": "q",
        "request_distribution": "r",
    }
    assert _extract_attn_meta_fields(dict_meta) == ("s", "b", "q", "r")
    obj_meta = SimpleNamespace(
        seq_lens="s2",
        block_tables="b2",
        query_start_loc="q2",
        request_distribution="r2",
    )
    assert _extract_attn_meta_fields(obj_meta) == ("s2", "b2", "q2", "r2")


@pytest.mark.cpu_test
def test_deepseek_tpu_attention_mixin_build_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pallas_op_cache.clear()
    base_mixin = DeepseekTPUAttentionMixin()
    with pytest.raises(NotImplementedError):
        _ = base_mixin.attention_kind
    assert base_mixin.main_cache_layer is base_mixin

    registered_fakes: list[object] = []

    class _FakeJaxOp:
        def __init__(self, name: str, fn: object) -> None:
            self.name = name
            self.fn = fn
            self.fake_fn = None

        def register_fake(self, fake_fn: object) -> None:
            self.fake_fn = fake_fn
            registered_fakes.append(fake_fn)

        def __call__(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
            q = args[0]
            sw_cache = args[2]
            assert isinstance(q, torch.Tensor)
            assert isinstance(sw_cache, torch.Tensor)
            return (
                torch.ones((q.shape[0], 16, 128), dtype=q.dtype),
                torch.ones_like(sw_cache),
            )

    monkeypatch.setattr(
        attn_mod.pallas,
        "jax_op",
        lambda name, fn, **kwargs: _FakeJaxOp(name, fn),
    )

    ctx = SimpleNamespace(
        mesh="fake_mesh",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(head_dim=128)),
            cache_config=SimpleNamespace(block_size=64),
        ),
    )
    monkeypatch.setattr(attn_mod, "get_vllm_model_wrapper_context", lambda: ctx)

    attn = _DummyAttention("csa")
    assert attn.swa_page_size == 64

    # 1. Profiling build (swa_tensor is None) and cached _attn_op_prof_instance
    op_prof = attn.attn_op
    assert attn.__dict__["_attn_op_prof_instance"] is op_prof
    assert attn.attn_op is op_prof
    # Hit _pallas_op_cache hit path inside _build_attn_op
    del attn.__dict__["_attn_op_prof_instance"]
    assert attn.attn_op is op_prof

    # Exercise _fake_attn registered callback
    assert len(registered_fakes) == 1
    fake_q = torch.zeros((3, 16, 128), dtype=torch.bfloat16)
    fake_sw = torch.zeros((2, 64, 1, 128), dtype=torch.uint8)
    f_out, f_sw = registered_fakes[0](fake_q, None, fake_sw)
    assert f_out.shape == (3, 16, 128) and f_sw.shape == fake_sw.shape

    # 2. Real cache build with aliased buffer and non_causal=True
    shared_buf = torch.ones((2, 64, 1, 128), dtype=torch.uint8)
    attn.swa_cache_layer.kv_cache = shared_buf
    attn.kv_cache = (shared_buf, torch.ones((2, 16, 1, 64), dtype=torch.uint8))
    attn.non_causal_multi_token_decode = True
    op_real = attn.attn_op
    assert attn.__dict__["_attn_op_instance"] is op_real
    assert attn.attn_op is op_real

    # 3. SWA and HCA variants with explicit softmax_scale
    attn_swa = _DummyAttention("swa")
    attn_swa.softmax_scale = 0.25
    _ = attn_swa.attn_op

    attn_hca = _DummyAttention("hca")
    attn_hca.scale = 0.5
    _ = attn_hca.attn_op

    # 4. Error branches in _build_attn_op and _get_active_sw_cache
    attn_no_swa = _DummyAttention("csa")
    attn_no_swa.swa_cache_layer = None
    with pytest.raises(RuntimeError, match="swa_cache_layer unavailable"):
        attn_no_swa._build_attn_op()
    with pytest.raises(RuntimeError, match="swa_cache_layer unavailable"):
        attn_no_swa._get_active_sw_cache()

    attn_bad_page = _DummyAttention("csa")
    attn_bad_page.swa_cache_layer.block_size = 0
    with pytest.raises(RuntimeError, match="SWA logical page size is 0"):
        attn_bad_page._build_attn_op()

    ctx_no_head = SimpleNamespace(
        mesh="fake_mesh",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace())
        ),
    )
    monkeypatch.setattr(attn_mod, "get_vllm_model_wrapper_context", lambda: ctx_no_head)
    attn_no_scale = _DummyAttention("csa")
    with pytest.raises(RuntimeError, match="attention scale unavailable"):
        attn_no_scale._build_attn_op()

    monkeypatch.setattr(attn_mod, "get_vllm_model_wrapper_context", lambda: ctx)

    # 5. kv_rope_op and o_proj_op caching (instance dict + _pallas_op_cache)
    kv_op1 = attn.kv_rope_op
    assert attn.kv_rope_op is kv_op1
    del attn.__dict__["_kv_rope_op_instance"]
    assert attn.kv_rope_op is kv_op1

    # process_weights_after_loading: transpose + idempotent reload + invalid shape
    attn.process_weights_after_loading(torch.bfloat16)
    assert tuple(attn.wo_a.weight.shape) == (1024, 64)
    attn.process_weights_after_loading(torch.bfloat16)
    assert tuple(attn.wo_a.weight.shape) == (1024, 64)

    o_op1 = attn.o_proj_op
    assert attn.o_proj_op is o_op1
    del attn.__dict__["_o_proj_op_instance"]
    assert attn.o_proj_op is o_op1

    attn_bad_w = _DummyAttention("csa")
    attn_bad_w.wo_a.weight = torch.zeros((10, 10), dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="expected the linear to hold"):
        attn_bad_w.process_weights_after_loading(torch.bfloat16)


@pytest.mark.cpu_test
def test_deepseek_tpu_attention_mixin_run_mqa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _DummyAttention("csa")
    captured_operands: list[tuple[object, ...]] = []

    def _fake_attn_op(*operands: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured_operands.append(operands)
        q = operands[0]
        sw_cache = operands[2]
        assert isinstance(q, torch.Tensor)
        assert isinstance(sw_cache, torch.Tensor)
        return (
            torch.zeros((q.shape[0], 16, 128), dtype=q.dtype),
            torch.full_like(sw_cache, 7),
        )

    object.__setattr__(attn, "_attn_op_instance", _fake_attn_op)

    q = torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    kv = torch.zeros((4, 1, 128), dtype=torch.bfloat16)
    positions = torch.arange(4, dtype=torch.int32)
    topk = torch.zeros((4, 8), dtype=torch.int32)

    # 1. Dict attn_metadata missing swa_layer_name -> RuntimeError
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}),
    )
    with pytest.raises(RuntimeError, match="no attention metadata for SWA layer"):
        attn._run_mqa(q, kv, positions, None, topk_indices=topk)

    # 2. Dict attn_metadata profiling pass (empty caches, None main_cache_kv/rope)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                attn.swa_cache_layer.prefix: meta,
                attn.prefix: SimpleNamespace(
                    seq_lens=None,
                    block_tables=None,
                    query_start_loc=None,
                    request_distribution=None,
                ),
            }
        ),
    )
    out = attn._run_mqa(q, kv, positions, None, topk_indices=topk)
    assert out.shape == (4, 16, 128)

    # 3. CSA live NoPE cache without companion RoPE cache -> RuntimeError
    attn.swa_cache_layer.kv_cache = torch.ones((2, 64, 1, 64), dtype=torch.bfloat16)
    attn.kv_cache = torch.ones((2, 16, 1, 128), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="CSA layer has a live compressed-KV array"):
        attn._run_mqa(q, kv, positions, None, topk_indices=topk)

    # 4. CSA live (nope, rope) pair + non-dict attn_metadata + in-place sw_cache copy
    attn.kv_cache = (
        torch.ones((2, 16, 1, 128), dtype=torch.uint8),
        torch.ones((2, 16, 1, 64), dtype=torch.uint8),
    )
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=meta),
    )
    out = attn._run_mqa(q, kv, positions, None, topk_indices=topk)
    assert out.shape == (4, 16, 128)
    assert torch.all(attn.swa_cache_layer.kv_cache.view(torch.uint8) == 7)

    # 5. SWA-only mode with both populated and None metadata fields
    attn_swa = _DummyAttention("swa")
    object.__setattr__(attn_swa, "_attn_op_instance", _fake_attn_op)
    out_swa = attn_swa._run_mqa(q, kv, positions, None, topk_indices=None)
    assert out_swa.shape == (4, 16, 128)

    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    out_swa_none = attn_swa._run_mqa(q, kv, positions, None, topk_indices=None)
    assert out_swa_none.shape == (4, 16, 128)


@pytest.mark.cpu_test
def test_vllm_deepseek_v4_mla_attention_full_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pallas_op_cache.clear()

    class _SubModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.get_kv_cache_spec = lambda cfg: "orig"

    _SubModule.__name__ = "DeepseekV4Attention"

    def _fake_super_init(
        self,
        vllm_config,
        prefix="",
        topk_indices_buffer=None,
        aux_stream_list=None,
    ):
        nn.Module.__init__(self)
        # Verify torch.cuda.Event stub works inside super().__init__
        assert torch.cuda.Event() is None
        self.prefix = prefix
        self.compress_ratio = getattr(vllm_config, "_test_compress_ratio", 4)
        self.window_size = 128
        self.head_dim = 128
        self.n_local_heads = 16
        self.n_local_groups = 2
        self.o_lora_rank = 32
        self.q_lora_rank = 64
        self.eps = 1e-6
        self.swa_cache_layer = SimpleNamespace(
            prefix=f"{prefix}.swa", block_size=64, kv_cache=None
        )
        if getattr(vllm_config, "_with_submodules", True):
            self.sub_attn = _SubModule()
            self.compressor = SimpleNamespace(
                state_cache=SimpleNamespace(prefix=f"{prefix}.comp_state"),
                called=False,
            )
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(prefix=f"{prefix}.idx_k"),
                weights_proj=lambda h: (torch.ones((h.shape[0], 4)), None),
            )
            self.attn_sink = nn.Parameter(
                torch.zeros(4, dtype=torch.float32), requires_grad=False
            )
        else:
            self.compressor = None
            self.indexer = None
            self.attn_sink = None

        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros((16, 64), dtype=torch.float32)
        )
        self.wo_a = SimpleNamespace(
            weight=torch.zeros((1024, 64), dtype=torch.float8_e4m3fn),
            weight_scale=torch.ones(64, dtype=torch.float32),
        )

    monkeypatch.setattr(
        attn_mod.dsv4_attention.DeepseekV4Attention, "__init__", _fake_super_init
    )
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_rank", lambda: 1)

    def _make_cfg(
        compress_ratio: int = 4,
        with_submodules: bool = True,
        with_comp_cfg: bool = True,
    ):
        hf_cfg = SimpleNamespace(
            num_attention_heads=16,
            head_dim=128,
            num_hidden_layers=2,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        comp_cfg = SimpleNamespace(static_forward_context={}) if with_comp_cfg else None
        return SimpleNamespace(
            _test_compress_ratio=compress_ratio,
            _with_submodules=with_submodules,
            model_config=SimpleNamespace(hf_config=hf_cfg),
            cache_config=SimpleNamespace(block_size=64),
            compilation_config=comp_cfg,
        )

    # 1. CSA layer with submodules and compilation_config
    cfg_csa = _make_cfg(compress_ratio=4, with_submodules=True, with_comp_cfg=True)
    layer_csa = VllmDeepseekV4MLAAttention(cfg_csa, prefix="model.layers.0.attn")
    assert layer_csa.compressor.k_cache is layer_csa
    assert "model.layers.0.attn" in cfg_csa.compilation_config.static_forward_context
    assert layer_csa.sub_attn.get_kv_cache_spec == layer_csa.get_kv_cache_spec

    # Test attn_sink.weight_loader
    loaded_sink = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    layer_csa.attn_sink.weight_loader(layer_csa.attn_sink, loaded_sink)
    assert torch.equal(layer_csa.attn_sink, torch.tensor([5.0, 6.0, 7.0, 8.0]))

    # 2. SWA layer (compress_ratio=1) and HCA layer (compress_ratio=2)
    cfg_swa = _make_cfg(compress_ratio=1, with_submodules=True, with_comp_cfg=False)
    layer_swa = VllmDeepseekV4MLAAttention(cfg_swa, prefix="model.layers.1.attn")
    assert layer_swa.compressor.k_cache is layer_swa.swa_cache_layer
    assert layer_swa.attention_kind == "swa"
    assert layer_swa.get_kv_cache_spec(cfg_swa) is None

    cfg_hca = _make_cfg(compress_ratio=2, with_submodules=False, with_comp_cfg=True)
    delattr(cfg_hca, "_with_submodules")
    cfg_hca._with_submodules = False
    layer_hca = VllmDeepseekV4MLAAttention(cfg_hca, prefix="model.layers.2.attn")

    # Also test compilation_config when swa_cache_layer is absent
    def _fake_super_no_swa(self, vllm_config, prefix="", **kwargs):
        _fake_super_init(self, vllm_config, prefix=prefix, **kwargs)
        delattr(self, "swa_cache_layer")

    monkeypatch.setattr(
        attn_mod.dsv4_attention.DeepseekV4Attention,
        "__init__",
        _fake_super_no_swa,
    )
    _ = VllmDeepseekV4MLAAttention(
        _make_cfg(compress_ratio=2, with_submodules=False, with_comp_cfg=True),
        prefix="model.layers.3.attn",
    )

    assert layer_csa.attention_kind == "csa"
    assert layer_hca.attention_kind == "hca"
    assert VllmDeepseekV4MLAAttention.get_padded_num_q_heads(16) == 16
    assert layer_csa.get_attn_backend() is DeepseekV4TPUAttentionBackend

    spec_csa = layer_csa.get_kv_cache_spec(cfg_csa)
    assert spec_csa is not None and spec_csa.tokens_per_state == 4
    spec_hca = layer_hca.get_kv_cache_spec(cfg_hca)
    assert spec_hca is not None and spec_hca.head_size == 1024

    # Test swa_page_size, qnorm_rope_op, qnorm_rope, kv_rope, and forward
    ctx = SimpleNamespace(mesh="fake_mesh", vllm_config=cfg_csa)
    monkeypatch.setattr(attn_mod, "get_vllm_model_wrapper_context", lambda: ctx)
    assert layer_csa.swa_page_size == 8

    class _IdentityOp:
        def register_fake(self, fn: object) -> None:
            pass

        def __call__(self, x: torch.Tensor, *args: object) -> torch.Tensor:
            return x

    monkeypatch.setattr(
        attn_mod.pallas, "jax_op", lambda name, fn, **kwargs: _IdentityOp()
    )
    qop1 = layer_csa.qnorm_rope_op
    assert layer_csa.qnorm_rope_op is qop1
    del layer_csa.__dict__["_qnorm_rope_op_instance"]
    assert layer_csa.qnorm_rope_op is qop1

    q_t = torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    pos_t = torch.arange(4, dtype=torch.int32)
    assert layer_csa.qnorm_rope(q_t, pos_t) is q_t
    assert layer_csa.kv_rope(q_t, pos_t) is q_t

    # Wire up forward/attention_impl/attn_gemm/_o_proj mocks
    layer_csa.fused_wqa_wkv = lambda h: (
        torch.zeros((h.shape[0], 64 + 128), dtype=torch.bfloat16),
        None,
    )
    layer_csa.q_norm = lambda qr: qr
    layer_csa.kv_norm = lambda kv: kv
    layer_csa.wq_b = lambda qr: torch.zeros(
        (qr.shape[0], 16 * 128), dtype=torch.bfloat16
    )
    layer_csa.indexer = MagicMock(return_value=torch.zeros((4, 8), dtype=torch.int32))
    layer_csa.indexer.weights_proj = lambda h: (
        torch.ones((h.shape[0], 4), dtype=torch.bfloat16),
        None,
    )
    layer_csa.compressor = MagicMock()
    layer_csa._run_mqa = MagicMock(
        return_value=torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    )
    object.__setattr__(
        layer_csa,
        "_o_proj_op_instance",
        lambda o, pos, cache, w, s: torch.zeros((o.shape[0], 64), dtype=o.dtype),
    )
    # Test wo_b returning tuple vs tensor
    layer_csa.wo_b = lambda z: (
        torch.ones((z.shape[0], 256), dtype=z.dtype),
        None,
    )

    hidden = torch.zeros((4, 256), dtype=torch.bfloat16)
    # 2D positions + indexer_rotary_emb=None fallback
    out_2d = layer_csa.forward(pos_t.unsqueeze(0), hidden)
    assert out_2d.shape == (4, 256)
    layer_csa.compressor.assert_called_once()

    # 1D positions + indexer_rotary_emb + split_sizes fallback
    layer_csa.indexer_rotary_emb = "custom_idx_rotary"
    del layer_csa.__dict__["_qr_kv_split_sizes"]
    layer_csa.wo_b = lambda z: torch.ones((z.shape[0], 256), dtype=z.dtype)
    out_1d = layer_csa.forward(pos_t, hidden)
    assert out_1d.shape == (4, 256)

    # Layer without indexer or compressor
    layer_hca.fused_wqa_wkv = layer_csa.fused_wqa_wkv
    layer_hca.q_norm = layer_csa.q_norm
    layer_hca.kv_norm = layer_csa.kv_norm
    layer_hca.wq_b = layer_csa.wq_b
    layer_hca._run_mqa = layer_csa._run_mqa
    object.__setattr__(layer_hca, "_o_proj_op_instance", layer_csa._o_proj_op_instance)
    layer_hca.wo_b = layer_csa.wo_b
    out_hca = layer_hca.forward(pos_t, hidden)
    assert out_hca.shape == (4, 256)


# ==============================================================================
# Part 2: TPUWorker 100% Statement & Branch Coverage Tests
# ==============================================================================


def _make_worker_vllm_config(
    *,
    cache_dtype: str = "auto",
    model_dtype: object = "bfloat16",
    profiler_dir: str | None = None,
    kv_cache_memory_bytes: int | None = None,
    enable_ep: bool = False,
    dp_rank_local: int | None = None,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=model_dtype, model="dummy-model"),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            gpu_memory_utilization=0.9,
            num_gpu_blocks=0,
            num_cpu_blocks=0,
            num_gpu_blocks_override=42,
        ),
        compilation_config=SimpleNamespace(
            compile_ranges_endpoints=[(1, 8)], compilation_time=0.0
        ),
        parallel_config=SimpleNamespace(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            enable_expert_parallel=enable_ep,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=dp_rank_local,
            data_parallel_master_ip="127.0.0.1",
        ),
        profiler_config=SimpleNamespace(torch_profiler_dir=profiler_dir),
    )


def _patch_worker_base_init(
    monkeypatch: pytest.MonkeyPatch, *, set_device_attr: bool = True
) -> None:
    def _fake_wb_init(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker=False,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.parallel_config = vllm_config.parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if set_device_attr:
            self.device = None

    monkeypatch.setattr(worker_mod.WorkerBase, "__init__", _fake_wb_init)
    monkeypatch.setattr(
        "vllm_torchtpu.patch_registry.apply", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.pp_push.widen_message_rings",
        lambda cfg: None,
    )
    monkeypatch.setattr(worker_mod, "ensure_runai_aux_files", lambda cfg: None)


@pytest.mark.cpu_test
def test_tpu_worker_helpers_and_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker_mod,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=3, is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 5)

    assert _get_kv_connector_handshake_metadata_key(None) == (3, 5)
    assert _get_kv_connector_handshake_metadata_key(
        SimpleNamespace(transfer_rank=7)
    ) == (3, 7)

    for env_key in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
        monkeypatch.delenv(env_key, raising=False)
    _configure_tpu_process_env(1, 0, 4, 2)
    assert os.environ["RANK"] == "1"
    assert os.environ["WORLD_SIZE"] == "4"
    _configure_tpu_process_env(0, 0, 1, 1)
    assert "WORLD_SIZE" not in os.environ

    # Test TPUWorker.__init__ variants
    _patch_worker_base_init(monkeypatch, set_device_attr=True)
    cfg1 = _make_worker_vllm_config(
        cache_dtype="auto", model_dtype="bfloat16", profiler_dir="/tmp/prof"
    )
    w1 = TPUWorker(
        cfg1,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:12345",
        devices=[SimpleNamespace(id=0), SimpleNamespace(id=1)],
    )
    assert not hasattr(w1, "device")
    assert w1.device_ranks == {0, 1}
    assert w1.kv_cache_dtype == torch.bfloat16
    assert w1.profile_dir == "/tmp/prof"

    _patch_worker_base_init(monkeypatch, set_device_attr=False)
    cfg2 = _make_worker_vllm_config(
        cache_dtype="auto", model_dtype=torch.float32, profiler_dir=None
    )
    w2 = TPUWorker(
        cfg2,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:12345",
        devices=None,
    )
    assert w2.devices == []
    assert w2.kv_cache_dtype == torch.float32
    assert w2.profile_dir is None

    cfg3 = _make_worker_vllm_config(cache_dtype="fp8", model_dtype="bfloat16")
    w3 = TPUWorker(
        cfg3,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:12345",
    )
    assert w3.kv_cache_dtype == torch.float8_e4m3fn

    # set_kv_cache_layout & initialize_cache
    monkeypatch.setattr(
        worker_mod.WorkerBase,
        "get_supported_kv_cache_layouts",
        lambda self: ["BLHNC"],
    )
    monkeypatch.setattr(
        worker_mod.WorkerBase,
        "set_kv_cache_layout",
        lambda self, layout: setattr(self, "_layout_set", layout),
    )
    with pytest.raises(ValueError, match="incompatible with the loaded"):
        w1.set_kv_cache_layout("NHD")
    w1.set_kv_cache_layout("BLHNC")
    assert getattr(w1, "_layout_set", None) == "BLHNC"

    w1.initialize_cache(100, 50)
    assert w1.cache_config.num_gpu_blocks == 100
    assert w1.cache_config.num_cpu_blocks == 50


@pytest.mark.cpu_test
def test_tpu_worker_init_device_all_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_key in (
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "TPU_VISIBLE_CHIPS",
        "TORCH_TPU_TOPOLOGY",
        "VLLM_CACHE_ROOT",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    ):
        monkeypatch.delenv(env_key, raising=False)
    _patch_worker_base_init(monkeypatch, set_device_attr=True)

    fake_binding = SimpleNamespace(
        rank=1,
        world_size=4,
        dp_rank=0,
        local_rank=1,
        local_world_size=2,
        init_rank=1,
        init_world_size=2,
        init_local_rank=1,
        native_local_rank=1,
        as_env=lambda: {"RANK": "1", "WORLD_SIZE": "2"},
    )
    monkeypatch.setattr(
        worker_mod, "get_tpu_worker_binding", lambda *a, **kw: fake_binding
    )
    monkeypatch.setattr(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._get_tpu_topology",
        classmethod(lambda cls, ws: f"1x{ws}"),
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_worker_distributed_backend=lambda ws: "gloo"),
    )
    monkeypatch.setattr(
        worker_mod, "set_current_vllm_config", lambda cfg: nullcontext()
    )
    monkeypatch.setattr(worker_mod, "init_distributed_environment", lambda **kw: None)
    monkeypatch.setattr(
        worker_mod, "pcp_topology_order", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(
        worker_mod, "ensure_model_parallel_initialized", lambda **kw: None
    )
    monkeypatch.setattr(
        worker_mod,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(worker_mod, "TPUModelRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(worker_mod.dist_utils, "get_node_id", lambda: 0)
    monkeypatch.setattr(worker_mod.vllm_utils, "report_usage_stats", lambda cfg: None)
    monkeypatch.setattr(torch, "device", lambda name: SimpleNamespace(type=name, id=0))

    # 1. dp_size > 1 with enable_expert_parallel=True
    monkeypatch.setattr(worker_mod.utils, "get_dp_size", lambda pc: 2)
    monkeypatch.setenv("TORCH_TPU_DP_MASTER_PORT", "29500")
    cfg_ep1 = _make_worker_vllm_config(enable_ep=True, dp_rank_local=None)
    w_ep1 = TPUWorker(cfg_ep1, 0, 0, "tcp://127.0.0.1:12345", devices=None)
    w_ep1.init_device()
    assert cfg_ep1.parallel_config.data_parallel_rank_local == 0

    cfg_ep2 = _make_worker_vllm_config(enable_ep=True, dp_rank_local=1)
    w_ep2 = TPUWorker(
        cfg_ep2,
        0,
        0,
        "tcp://127.0.0.1:12345",
        devices=[SimpleNamespace(id=0)],
    )
    w_ep2.init_device()
    assert cfg_ep2.parallel_config.data_parallel_rank_local == 1

    # 2. dp_size > 1 with enable_expert_parallel=False
    monkeypatch.setenv(
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "10.0.0.1:1,10.0.0.1:2,10.0.0.1:3,10.0.0.1:4",
    )
    cfg_dense_dp = _make_worker_vllm_config(enable_ep=False)
    w_dense_dp = TPUWorker(cfg_dense_dp, 0, 0, "tcp://127.0.0.1:12345")
    w_dense_dp.init_device()
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == "10.0.0.1:1,10.0.0.1:2"

    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
    w_dense_dp.init_device()

    # 3. dp_size == 1: tcp://, file://, invalid scheme, pcp_size > 1, offsets
    monkeypatch.setattr(worker_mod.utils, "get_dp_size", lambda pc: 1)

    cfg_pcp = _make_worker_vllm_config(pcp_size=2, dcp_size=2)
    w_pcp = TPUWorker(cfg_pcp, 0, 0, "tcp://127.0.0.1:12345")
    monkeypatch.setitem(vars(worker_mod.envs), "TPU_LOCAL_RANK_OFFSET", 2)
    w_pcp.init_device()

    monkeypatch.setitem(vars(worker_mod.envs), "TPU_LOCAL_RANK_OFFSET", 0)
    monkeypatch.setitem(vars(worker_mod.envs), "DEBUG_TPU_LOCAL_RANK_OFFSET", 4)
    cfg_file = _make_worker_vllm_config(pcp_size=1)
    w_file = TPUWorker(cfg_file, 0, 0, "file:///tmp/vllm_store")
    w_file.init_device()

    monkeypatch.setitem(vars(worker_mod.envs), "DEBUG_TPU_LOCAL_RANK_OFFSET", 0)
    w_file.init_device()

    w_bad = TPUWorker(cfg_file, 0, 0, "invalid://method")
    with pytest.raises(ValueError, match="Expected tcp://<host>:<port>"):
        w_bad.init_device()


@pytest.mark.cpu_test
def test_tpu_worker_memory_execution_profiling_and_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_worker_base_init(monkeypatch, set_device_attr=True)

    # 1. _reserve_deepseek_v41_rope_companion & determine_available_memory
    cfg = _make_worker_vllm_config(
        kv_cache_memory_bytes=1024, profiler_dir=str(tmp_path)
    )
    worker = TPUWorker(cfg, 0, 0, "tcp://127.0.0.1:12345")
    worker.model_runner = MagicMock()
    worker.model_runner.max_num_tokens = 128

    monkeypatch.setitem(vars(worker_mod.envs), "TPU_DEEPSEEK_V41_IMPL", "v1")
    assert worker.determine_available_memory() == 1024

    monkeypatch.setitem(vars(worker_mod.envs), "TPU_DEEPSEEK_V41_IMPL", "v2")
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.reserve_rope_companion",
        lambda vcfg, avail: avail - 256,
    )
    assert worker.determine_available_memory() == 768
    monkeypatch.setitem(vars(worker_mod.envs), "TPU_DEEPSEEK_V41_IMPL", "v1")

    worker.cache_config.kv_cache_memory_bytes = None
    monkeypatch.setattr(
        worker_mod.utils,
        "compute_hbm_budget",
        lambda devs, util: SimpleNamespace(
            available=2000, total_limit=4000, cap=3600, total_used=1600
        ),
    )
    monkeypatch.setattr(
        worker_mod.utils, "estimate_kv_connector_hbm_reserve", lambda vcfg: 500
    )
    assert worker.determine_available_memory() == 1500

    # Negative available memory raises ValueError
    monkeypatch.setattr(
        worker_mod.utils, "estimate_kv_connector_hbm_reserve", lambda vcfg: 2500
    )
    with pytest.raises(ValueError, match="exceeds"):
        worker.determine_available_memory()

    # 2. Execution & delegation methods
    worker.execute_model("sched_out")
    worker.model_runner.execute_model.assert_called_once_with("sched_out")
    worker.sample_tokens("gram_out")
    worker.model_runner.sample_tokens.assert_called_once_with("gram_out")
    worker.pp_push()
    worker.model_runner.pp_push.assert_called_once()
    worker.pp_settle()
    worker.model_runner.pp_settle.assert_called_once()
    worker.profile_pipeline_chunks()
    worker.model_runner.profile_pipeline_chunks.assert_called_once()
    worker.take_draft_token_ids()
    worker.model_runner.take_draft_token_ids.assert_called_once()

    # execute_dummy_batch: bucket is None vs bucket is not None
    worker.model_runner._dp_coordinated_step.return_value = (None, 0)
    worker.model_runner.num_tokens_paddings = [16]
    worker.model_runner.num_reqs_max_model_len = 4
    worker.model_runner.max_num_blocks_per_req = 8
    worker.execute_dummy_batch()
    worker.model_runner._dummy_run.assert_called_once_with(
        16, 4, 8, use_max_model_len=True
    )

    worker.model_runner._dp_coordinated_step.return_value = (32, 2)
    worker.execute_dummy_batch()
    worker.model_runner._run_dp_idle_pairing.assert_called_once_with(32, 2)

    # 3. Phased & standard profiling
    monkeypatch.setitem(vars(worker_mod.envs), "USE_PHASED_PROFILER", True)
    worker.profile(is_start=True, profile_prefix="p1")
    worker.model_runner.start_phased_profiling.assert_called_once_with(
        "p1", profiler_kwargs=None
    )
    worker.profile(is_start=False)
    worker.model_runner.stop_phased_profiling.assert_called_once()

    monkeypatch.setitem(vars(worker_mod.envs), "USE_PHASED_PROFILER", False)
    worker.profile_dir = None
    worker.profile(is_start=True)  # logs warning & returns

    worker.profile_dir = str(tmp_path)
    worker.profile(is_start=False)  # profile_context is None -> logs warning

    monkeypatch.setattr(
        "vllm_torchtpu.tracing.options.resolve_profile_dir_and_opts",
        lambda pdir, prefix, profiler_kwargs=None: (
            str(tmp_path / "run"),
            {
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 0,
            },
            {},
        ),
    )
    monkeypatch.setattr(
        worker_mod.profiler_trace,
        "resolve_canonical_dst_ts",
        lambda pdir, rank, session_key: "2026_10_09",
    )
    monkeypatch.setattr(
        worker_mod.profiler_trace,
        "rank_capture_dir",
        lambda pdir, rank: str(tmp_path / f"rank_{rank}"),
    )
    merged_calls: list[int] = []
    cleared_calls: list[str | None] = []
    monkeypatch.setattr(
        worker_mod.profiler_trace,
        "merge_rank_capture",
        lambda cdir, pdir, ts, rank, world_size: merged_calls.append(rank),
    )
    monkeypatch.setattr(
        worker_mod.profiler_trace,
        "clear_canonical_ts_marker",
        lambda pdir, skey: cleared_calls.append(skey),
    )
    monkeypatch.setattr(
        torch.profiler, "tensorboard_trace_handler", lambda **kw: "handler"
    )
    monkeypatch.setattr(worker_mod, "TpuProfilerConfig", lambda **kw: "cfg")

    fake_prof_ctx = MagicMock()
    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: fake_prof_ctx)

    # Start & duplicate start warning & stop on rank 0
    worker.profile_rank = 0
    worker.profile(is_start=True)
    assert worker.profile_context is fake_prof_ctx
    worker.profile(is_start=True)  # already running warning
    worker.profile(is_start=False)
    assert worker.profile_context is None
    assert merged_calls == [0]
    assert len(cleared_calls) == 1

    # Start & stop on rank 1 (does not call clear_canonical_ts_marker)
    worker.profile_rank = 1
    worker.profile(is_start=True)
    worker.profile(is_start=False)
    assert merged_calls == [0, 1]
    assert len(cleared_calls) == 1

    # Stop when profile_capture_dir is None
    worker.profile(is_start=True)
    worker.profile_capture_dir = None
    worker.profile(is_start=False)

    # Exception during profile_context.__enter__()
    bad_prof_ctx = MagicMock()
    bad_prof_ctx.__enter__.side_effect = RuntimeError("profiler init failed")
    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: bad_prof_ctx)
    with pytest.raises(RuntimeError, match="profiler init failed"):
        worker.profile(is_start=True)
    assert worker.profile_context is None

    # 4. load_model, compile_or_warm_up_model, reload_kernels, KV & weight methods
    worker.load_model()
    worker.model_runner.load_model.assert_called_once()

    monkeypatch.setattr(worker_mod, "set_torch_threads_for_runtime", lambda: None)
    ct = worker.compile_or_warm_up_model()
    assert ct.encoder == 0.0
    worker.model_runner.capture_model.assert_called_once()

    monkeypatch.setitem(vars(worker_mod.envs), "TPU_KERNEL_ITER_MODE", False)
    with pytest.raises(
        RuntimeError, match="reload_kernels requires TPU_KERNEL_ITER_MODE=1"
    ):
        worker.reload_kernels()

    monkeypatch.setitem(vars(worker_mod.envs), "TPU_KERNEL_ITER_MODE", True)
    monkeypatch.setattr(
        "vllm_torchtpu.compilation.kernel_reload.reload_kernels",
        lambda mods: {"reloaded": mods or ["all"]},
    )
    stats = worker.reload_kernels(["mod_a"])
    assert stats["rank"] == 0 and stats["reloaded"] == ["mod_a"]

    worker.get_model()
    worker.get_supported_tasks()
    worker.get_kv_cache_spec()
    assert worker.get_num_gpu_blocks_override() == 42
    worker.get_kv_prewarm_shapes()
    worker.prewarm_kv_offload_shape(16)

    monkeypatch.setattr(
        worker_mod, "ensure_kv_transfer_initialized", lambda vcfg, kcfg: None
    )
    kv_cfg = SimpleNamespace(num_blocks=256)
    worker.initialize_from_config(kv_cfg)
    assert worker.cache_config.num_gpu_blocks == 256
    worker.model_runner.initialize_kv_cache.assert_called_once_with(kv_cfg)

    # get_node_kv_ip_port: n_cfg <= 0 and n_cfg > 0
    monkeypatch.setattr(worker_mod.dist_utils, "get_node_id", lambda: 2)
    monkeypatch.setattr(worker_mod.dist_utils, "get_host_ip", lambda: "10.0.0.5")
    monkeypatch.setattr(worker_mod.dist_utils, "get_kv_transfer_port", lambda: "15000")
    monkeypatch.setattr(worker_mod.dist_utils, "get_transfer_channel_number", lambda: 0)
    assert worker.get_node_kv_ip_port() == (2, "10.0.0.5", 15004)
    monkeypatch.setattr(worker_mod.dist_utils, "get_transfer_channel_number", lambda: 1)
    assert worker.get_node_kv_ip_port() == (2, "10.0.0.5", 15002)

    worker.sync_weights("w", {}, {}, reshard_fn=None)
    worker.model_runner._sync_weights.assert_called_once()

    # get_kv_connector_handshake_metadata: 3 branches
    monkeypatch.setattr(worker_mod, "has_kv_transfer_group", lambda: False)
    assert worker.get_kv_connector_handshake_metadata() is None

    monkeypatch.setattr(worker_mod, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        worker_mod,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(get_handshake_metadata=lambda: None),
    )
    assert worker.get_kv_connector_handshake_metadata() is None

    meta_obj = SimpleNamespace(transfer_rank=2)
    monkeypatch.setattr(
        worker_mod,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(get_handshake_metadata=lambda: meta_obj),
    )
    monkeypatch.setattr(
        worker_mod, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    assert worker.get_kv_connector_handshake_metadata() == {(1, 2): meta_obj}
