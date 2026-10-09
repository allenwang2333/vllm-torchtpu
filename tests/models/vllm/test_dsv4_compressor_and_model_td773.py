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

"""100% statement and branch coverage tests for TD7.73 modules:
- vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor
- vllm_torchtpu.models.vllm.deepseek_v4.model
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import pytest
import torch
import torch.nn as nn
from vllm.config import CompilationMode
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_compressor as comp_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v4 import model as model_mod


class TestDeepseekV4Compressor:
    def test_fake_and_jax_compressor_entrypoints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_t = torch.zeros((2, 4), dtype=torch.uint8)
        second_t = torch.zeros((2, 2), dtype=torch.uint8)
        f_idx = comp_mod._fake_compressor_indexer(
            None, None, None, None, None, None, None, None, None, None, cache_t
        )
        assert f_idx.shape == cache_t.shape
        f_two = comp_mod._fake_compressor_two_cache(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            cache_t,
            second_t,
        )
        assert f_two[0].shape == cache_t.shape
        assert f_two[1].shape == second_t.shape

        recorded: dict = {}

        def fake_compressor_forward(**kwargs):
            recorded.update(kwargs)
            c = kwargs["cache"] + 1
            rc = kwargs["rope_cache"] + 2 if kwargs["rope_cache"] is not None else None
            sc = (
                kwargs["state_cache"] + 3 if kwargs["state_cache"] is not None else None
            )
            return c, rc, sc

        monkeypatch.setattr(comp_mod, "compressor_forward", fake_compressor_forward)

        arr = jnp.ones((2, 4), dtype=jnp.uint8)
        empty = jnp.zeros((0, 4), dtype=jnp.uint8)
        common_args = (arr,) * 10
        static_kw = dict(
            state_block_size=16,
            head_dim=512,
            compress_ratio=4,
            overlap=True,
            rms_eps=1e-6,
            quant_block=64,
        )

        # CSA: empty vs non-empty
        c0, r0 = comp_mod._compressor_jax_csa(*common_args, empty, arr, **static_kw)
        assert c0.shape == (0, 4)
        assert jnp.array_equal(r0, arr)

        c1, r1 = comp_mod._compressor_jax_csa(*common_args, arr, arr, **static_kw)
        assert jnp.array_equal(c1, arr + 1)
        assert jnp.array_equal(r1, arr + 2)
        assert recorded["state_block_size"] == 16

        # HCA: empty cache or empty state_cache vs non-empty
        hc0, hs0 = comp_mod._compressor_jax_hca(*common_args, empty, arr, **static_kw)
        assert hc0.shape == (0, 4)
        hc0b, hs0b = comp_mod._compressor_jax_hca(*common_args, arr, empty, **static_kw)
        assert hs0b.shape == (0, 4)

        hc1, hs1 = comp_mod._compressor_jax_hca(*common_args, arr, arr, **static_kw)
        assert jnp.array_equal(hc1, arr + 1)
        assert jnp.array_equal(hs1, arr + 3)

        # Indexer: empty vs non-empty
        ic0 = comp_mod._compressor_jax_indexer(*common_args, empty, **static_kw)
        assert ic0.shape == (0, 4)
        ic1 = comp_mod._compressor_jax_indexer(*common_args, arr, **static_kw)
        assert jnp.array_equal(ic1, arr + 1)

    def test_state_cache_init_derive_and_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_super_init(self, state_dim, dtype, compress_ratio, prefix):
            self.state_dim = state_dim
            self.dtype = dtype
            self.compress_ratio = compress_ratio
            self.prefix = prefix
            self.sliding_window = (1 + (compress_ratio == 4)) * compress_ratio
            self.block_size = 0

        monkeypatch.setattr(comp_mod.CompressorStateCache, "__init__", fake_super_init)
        sc = comp_mod.VllmCompressorStateCache(
            state_dim=2048,
            dtype=torch.float32,
            compress_ratio=4,
            prefix="layers.0.attn.compressor.state_cache",
        )
        assert sc.head_dim == 512
        assert sc.compress_ratio == 4

        # Error branch 1: kv_cache_block_size // compress_ratio <= 0
        with pytest.raises(ValueError, match="cannot be paged"):
            sc._derive_block_size(2)

        # Error branch 2: state_block_size <= 0
        monkeypatch.setattr(
            comp_mod.compressor_config, "state_block_size", lambda *a, **kw: 0
        )
        with pytest.raises(ValueError, match="state rows do not fit"):
            sc._derive_block_size(1024)

        monkeypatch.undo()
        monkeypatch.setattr(comp_mod.CompressorStateCache, "__init__", fake_super_init)
        sc_ok = comp_mod.VllmCompressorStateCache(
            state_dim=2048,
            dtype=torch.float32,
            compress_ratio=4,
            prefix="layers.0.attn.compressor.state_cache",
        )
        spec = sc_ok.get_kv_cache_spec(
            SimpleNamespace(cache_config=SimpleNamespace(block_size=1024))
        )
        assert isinstance(spec, SlidingWindowMLASpec)
        assert spec.block_size == 16
        assert spec.dtype == torch.uint8

    def test_compressor_init_transpose_build_and_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_base_init(self, *args, **kwargs):
            nn.Module.__init__(self)
            self.use_fp4_cache = kwargs.get("use_fp4_cache", False)
            self.head_dim = kwargs.get("head_dim", 512)
            self.compress_ratio = kwargs.get("compress_ratio", 4)
            self.overlap = self.compress_ratio == 4
            self.rms_norm_eps = 1e-6
            self._quant_block = 64 if self.head_dim == 512 else 128
            self.state_cache = SimpleNamespace(
                prefix="l0.state", block_size=16, kv_cache=None
            )
            self.k_cache_prefix = "l0.k"
            self.fused_wkv_wgate = nn.Linear(8, 16, bias=False)
            self.ape = nn.Parameter(torch.ones((4, 8), dtype=torch.float32))
            self.norm = SimpleNamespace(weight=torch.ones((8,), dtype=torch.float32))

        monkeypatch.setattr(comp_mod.DeepseekCompressor, "__init__", fake_base_init)
        monkeypatch.setattr(
            comp_mod,
            "get_current_vllm_config",
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(num_hidden_layers=61)
                )
            ),
        )

        # 1. use_fp4_cache=True -> NotImplementedError
        with pytest.raises(NotImplementedError, match="use_fp4_indexer_cache"):
            comp_mod.VllmDeepseekCompressor(use_fp4_cache=True)

        # 2. Normal init + transpose_wkv_wgate idempotency
        comp = comp_mod.VllmDeepseekCompressor(head_dim=512, compress_ratio=4)
        assert comp.num_layers == 61
        orig_shape = comp.fused_wkv_wgate.weight.shape
        comp.transpose_wkv_wgate()
        assert comp.fused_wkv_wgate.weight.shape == (orig_shape[1], orig_shape[0])
        comp.transpose_wkv_wgate()  # second call hits early return
        assert comp.fused_wkv_wgate.weight.shape == (orig_shape[1], orig_shape[0])

        # 3. _build_compressor_op for CSA, HCA, Indexer + cache hit
        monkeypatch.setattr(comp_mod, "_compressor_op_cache", {})
        monkeypatch.setattr(
            comp_mod,
            "get_vllm_model_wrapper_context",
            lambda: SimpleNamespace(mesh="fake_mesh"),
        )

        def fake_jax_op(name, fn, **kwargs):
            op = MagicMock(name=name)
            op.name = name
            op.kwargs = kwargs
            return op

        monkeypatch.setattr(comp_mod.pallas, "jax_op", fake_jax_op)

        # CSA (head_dim=512, overlap=True)
        op_csa = comp.compressor_op
        assert "csa" in op_csa.name
        assert comp.compressor_op is op_csa
        # Cache hit on fresh instance with same config
        comp_csa2 = comp_mod.VllmDeepseekCompressor(head_dim=512, compress_ratio=4)
        assert comp_csa2.compressor_op is op_csa

        # HCA (head_dim=512, overlap=False)
        comp_hca = comp_mod.VllmDeepseekCompressor(head_dim=512, compress_ratio=128)
        op_hca = comp_hca.compressor_op
        assert "hca" in op_hca.name

        # Indexer (head_dim=128, overlap=True)
        comp_idx = comp_mod.VllmDeepseekCompressor(head_dim=128, compress_ratio=4)
        op_idx = comp_idx.compressor_op
        assert "indexer" in op_idx.name

        # 4. _as_kernel_cache_view branches
        assert comp._as_kernel_cache_view(None) is None
        assert comp._as_kernel_cache_view(torch.empty(0, dtype=torch.uint8)) is None
        u8 = torch.ones((2, 4), dtype=torch.uint8)
        i32 = torch.ones((2, 4), dtype=torch.int32)
        f8 = torch.zeros((2, 4), dtype=torch.int8)
        assert comp._as_kernel_cache_view(u8) is u8
        assert comp._as_kernel_cache_view(i32) is i32
        viewed = comp._as_kernel_cache_view(f8)
        assert viewed is not None and viewed.dtype is torch.uint8

        # 5. forward branches
        hidden = torch.ones((2, 8), dtype=torch.bfloat16)
        pos = torch.tensor([0, 1], dtype=torch.int32)
        rotary = SimpleNamespace(cos_sin_cache=torch.ones((4, 4)))

        # 5a. attn_ctx not a dict -> return
        monkeypatch.setattr(
            comp_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
        comp.forward(hidden, pos, rotary)

        # 5b. Missing state_prefix or k_cache_prefix -> KeyError
        monkeypatch.setattr(
            comp_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={}),
        )
        with pytest.raises(KeyError, match="state_cache"):
            comp.forward(hidden, pos, rotary)

        monkeypatch.setattr(
            comp_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"l0.state": None}),
        )
        with pytest.raises(KeyError, match="k_cache"):
            comp.forward(hidden, pos, rotary)

        # 5c. Metadata is None -> return
        monkeypatch.setattr(
            comp_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"l0.state": None, "l0.k": None}),
        )
        comp.forward(hidden, pos, rotary)

        meta = SimpleNamespace(
            block_tables=torch.zeros((1, 2), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )
        monkeypatch.setattr(
            comp_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"l0.state": meta, "l0.k": meta}),
        )

        # 5d. k_cache has no live kv_cache -> return
        comp.k_cache = SimpleNamespace(prefix="l0.k", kv_cache=torch.empty(0))
        comp.forward(hidden, pos, rotary)

        # 5e. CSA missing companion rope_cache -> RuntimeError
        main_buf = torch.zeros((2, 4), dtype=torch.uint8)
        comp.k_cache = SimpleNamespace(prefix="l0.k", kv_cache=(main_buf, None))
        with pytest.raises(RuntimeError, match="companion RoPE KV array"):
            comp.forward(hidden, pos, rotary)

        # 5f. CSA success path
        rope_buf = torch.zeros((2, 4), dtype=torch.uint8)
        comp.k_cache = SimpleNamespace(prefix="l0.k", kv_cache=(main_buf, rope_buf))
        object.__setattr__(
            comp,
            "_compressor_op_instance",
            lambda *args: (
                torch.full_like(main_buf, 7),
                torch.full_like(rope_buf, 9),
            ),
        )
        comp.forward(hidden, pos, rotary)
        assert torch.all(main_buf == 7)
        assert torch.all(rope_buf == 9)

        # 5g. HCA missing state_cache -> RuntimeError, then success path
        comp_hca.transpose_wkv_wgate()
        hca_main = torch.zeros((2, 4), dtype=torch.uint8)
        comp_hca.k_cache = SimpleNamespace(custom_prefix="l0.k", kv_cache=hca_main)
        comp_hca.state_cache.kv_cache = None
        with pytest.raises(RuntimeError, match="no state array"):
            comp_hca.forward(hidden, pos, rotary)

        hca_state = torch.zeros((2, 4), dtype=torch.uint8)
        comp_hca.state_cache.kv_cache = hca_state
        object.__setattr__(
            comp_hca,
            "_compressor_op_instance",
            lambda *args: (
                torch.full_like(hca_main, 3),
                torch.full_like(hca_state, 5),
            ),
        )
        comp_hca.forward(hidden, pos, rotary)
        assert torch.all(hca_main == 3)
        assert torch.all(hca_state == 5)

        # 5h. Indexer success path
        comp_idx.transpose_wkv_wgate()
        idx_main = torch.zeros((2, 4), dtype=torch.uint8)
        comp_idx.k_cache = SimpleNamespace(kv_cache=idx_main)
        object.__setattr__(
            comp_idx,
            "_compressor_op_instance",
            lambda *args: torch.full_like(idx_main, 11),
        )
        comp_idx.forward(hidden, pos, rotary)
        assert torch.all(idx_main == 11)


class TestDeepseekV4ModelFull:
    def _make_vllm_cfg(self, *, expert_dtype: str = "fp4"):
        hf_config = SimpleNamespace(
            hidden_size=8,
            hc_mult=2,
            hc_sinkhorn_iters=4,
            hc_eps=1e-5,
            hc_post_alpha=2.0,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
            num_hash_layers=1,
            index_topk=4,
            vocab_size=16,
            num_attention_heads=4,
            n_routed_experts=2,
            expert_dtype=expert_dtype,
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf_config, dtype=torch.bfloat16, head_dtype=None
            ),
            quant_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            compilation_config=SimpleNamespace(
                static_forward_context={}, mode=CompilationMode.NONE
            ),
        )

    def test_decoder_layer_and_host_staged_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            model_mod,
            "VllmDeepseekV4MLAAttention",
            lambda cfg, **kw: (lambda pos, x: x + 1.0),
        )
        monkeypatch.setattr(
            model_mod,
            "DeepseekV4MoE",
            lambda cfg, **kw: (lambda x, ids=None: x * 2.0),
        )
        monkeypatch.setattr(model_mod, "RMSNorm", lambda h, eps: (lambda x: x))
        fake_ops = SimpleNamespace(
            pre=lambda x, fn, sc, b: (
                torch.ones_like(x),
                torch.zeros_like(x),
                x.mean(dim=1),
            ),
            fused=lambda x, res, pm, rm, fn, sc, b: (
                res + 1.0,
                pm,
                rm,
                res.mean(dim=1),
            ),
        )
        monkeypatch.setattr(model_mod, "get_mhc_ops", lambda *a: fake_ops)

        cfg = self._make_vllm_cfg()
        layer0 = model_mod.DeepseekV4DecoderLayer(cfg, prefix="model.layers.0")
        assert layer0.layer_idx == 0
        layer_neg = model_mod.DeepseekV4DecoderLayer(cfg, prefix="custom")
        assert layer_neg.layer_idx == -1

        assert layer0.mhc_ops is fake_ops
        assert layer0.mhc_ops is fake_ops  # cached hit

        x_in = torch.ones((3, 2, 8))
        pos = torch.arange(3)
        x_out, res, pm, rm = layer0(x_in, pos, input_ids=pos, residual=None)
        assert x_out.shape == (3, 8)
        x_out2, res2, pm2, rm2 = layer0(
            x_out, pos, input_ids=pos, post_mix=pm, res_mix=rm, residual=res
        )
        assert x_out2.shape == (3, 8)

        # _HostStagedParams: stage twice, flush_all
        staged = model_mod._HostStagedParams()
        p = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
        p.custom_tag = "ok"
        h1 = staged.stage("p", p)
        h2 = staged.stage("p", p)
        assert h1 is h2
        assert getattr(h1, "custom_tag", None) == "ok"
        h1.data.fill_(42.0)
        staged.flush_all()
        assert torch.all(p == 42.0)

    def test_model_pp_dp_forward_and_mhc_post_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(model_mod, "set_offloader", lambda x: None)
        monkeypatch.setattr(
            model_mod,
            "VocabParallelEmbedding",
            lambda v, h, **kw: nn.Embedding(v, h),
        )
        monkeypatch.setattr(model_mod, "RMSNorm", lambda h, eps: (lambda x: x))
        monkeypatch.setattr(
            model_mod,
            "mhc_collapse_head",
            lambda hs, *a: hs.mean(dim=1),
        )

        class _FakeLayer(nn.Module):
            def __init__(self, routes_dp: bool = True):
                super().__init__()
                self.ffn = SimpleNamespace(routes_dp_gathered_tokens=routes_dp)

            def forward(
                self,
                x,
                positions,
                input_ids=None,
                post_mix=None,
                res_mix=None,
                residual=None,
            ):
                res = x if residual is None else residual
                pm = torch.ones_like(res)
                rm = torch.zeros_like(res)
                return res.mean(dim=1), res, pm, rm

        monkeypatch.setattr(
            model_mod,
            "make_layers",
            lambda n, fn, prefix="": (
                0,
                n,
                nn.ModuleList(
                    [
                        fn(f"{prefix}.0") if i == -1 else _FakeLayer(True)
                        for i in range(n)
                    ]
                ),
            ),
        )
        # Ensure the lambda inside make_layers is also exercised once
        called_lambda = []

        def fake_make_layers(n, fn, prefix=""):
            monkeypatch.setattr(
                model_mod,
                "DeepseekV4DecoderLayer",
                lambda **kw: (_FakeLayer(True), called_lambda.append(kw))[0],
            )
            layers = nn.ModuleList([fn(f"{prefix}.{i}") for i in range(n)])
            return 0, n, layers

        monkeypatch.setattr(model_mod, "make_layers", fake_make_layers)

        def post_fn(h, r, pm, rm):
            return r + pm - rm

        monkeypatch.setattr(model_mod, "get_mhc_post_op", lambda: post_fn)

        # 1. First + Last PP rank
        monkeypatch.setattr(
            model_mod,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        monkeypatch.setattr(
            model_mod,
            "get_dp_group",
            lambda: SimpleNamespace(
                world_size=1, all_gather=lambda x, dim=0: torch.cat([x, x], dim=dim)
            ),
        )

        cfg = self._make_vllm_cfg()
        m = model_mod.DeepseekV4Model(vllm_config=cfg, prefix="model")
        assert len(called_lambda) == 2
        assert m.mhc_post_op is post_fn
        assert m.mhc_post_op is post_fn  # cached hit

        inter = m.make_empty_intermediate_tensors(3, torch.float32, torch.device("cpu"))
        assert "hidden_states" in inter.tensors

        # Forward with input_ids (embed_input_ids path) and no aux layers
        ids = torch.tensor([1, 2, 3])
        pos = torch.tensor([0, 1, 2])
        out = m(ids, pos)
        assert isinstance(out, torch.Tensor) and out.shape == (3, 8)

        # Forward with aux_hidden_state_layers=(1,) (non-final layer tagged)
        m.aux_hidden_state_layers = (1,)
        out_aux1, aux_list1 = m(ids, pos, inputs_embeds=torch.ones((3, 8)))
        assert len(aux_list1) == 1 and out_aux1.shape == (3, 8)

        # Forward with aux_hidden_state_layers=(2,) (final layer tagged)
        m.aux_hidden_state_layers = (2,)
        out_aux2, aux_list2 = m(ids, pos, inputs_embeds=torch.ones((3, 8)))
        assert len(aux_list2) == 1 and out_aux2.shape == (3, 8)

        # Forward with 0 active layers (start_layer == end_layer -> post_mix is None)
        m.start_layer = 2
        m.end_layer = 2
        m.aux_hidden_state_layers = ()
        out_empty_layers = m(ids, pos, inputs_embeds=torch.ones((3, 8)))
        assert out_empty_layers.shape == (3, 8)
        m.start_layer = 0
        m.end_layer = 2

        # _dp_gather_hash_moe_input_ids with dp_group.world_size > 1
        monkeypatch.setattr(
            model_mod,
            "get_dp_group",
            lambda: SimpleNamespace(
                world_size=2, all_gather=lambda x, dim=0: torch.cat([x, x], dim=dim)
            ),
        )
        gathered = m._dp_gather_hash_moe_input_ids(ids)
        assert gathered.shape == (6,)
        m.layers[0].ffn.routes_dp_gathered_tokens = False
        not_gathered = m._dp_gather_hash_moe_input_ids(ids)
        assert not_gathered is ids

        # 2. Non-first, non-last PP rank
        monkeypatch.setattr(
            model_mod,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
        )
        m_mid = model_mod.DeepseekV4Model(vllm_config=cfg, prefix="model")
        assert isinstance(m_mid.embed_tokens, model_mod.PPMissingLayer)
        assert isinstance(m_mid.norm, model_mod.PPMissingLayer)
        assert m_mid._mtp_hidden_buffer is None

        mid_out = m_mid(
            ids,
            pos,
            intermediate_tensors=IntermediateTensors(
                {"hidden_states": torch.ones((3, 2, 8))}
            ),
        )
        assert isinstance(mid_out, IntermediateTensors)

    def test_model_load_weights_and_causal_lm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(model_mod, "set_offloader", lambda x: None)
        monkeypatch.setattr(
            model_mod, "get_tensor_model_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(model_mod, "get_tensor_model_parallel_rank", lambda: 1)

        m = model_mod.DeepseekV4Model.__new__(model_mod.DeepseekV4Model)
        nn.Module.__init__(m)
        m.config = SimpleNamespace(num_attention_heads=4, n_routed_experts=2)

        # Parameters for stacked, expert, attn_sink, and plain branches
        p_stacked = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
        p_stacked.weight_loader = lambda param, w, shard_id: param.data.copy_(
            w + shard_id
        )
        p_expert = nn.Parameter(
            torch.zeros((4,), dtype=torch.uint8), requires_grad=False
        )

        def expert_loader(
            param, w, name_mapped, shard_id, expert_id, return_success=True
        ):
            if expert_id == 0:
                return False
            param.data.copy_(w)
            return True

        p_expert.weight_loader = expert_loader
        p_sink = nn.Parameter(torch.zeros((2,), dtype=torch.float32))
        p_plain = nn.Parameter(torch.zeros((4,), dtype=torch.float32))

        m.l0 = nn.Module()
        m.l0.experts = nn.Module()
        m.l0.register_parameter("gate_up_proj", p_stacked)
        m.l0.experts.register_parameter("w13_weight_scale", p_expert)
        m.l0.register_parameter("attn_sink", p_sink)
        m.l0.register_parameter("plain", p_plain)

        monkeypatch.setattr(
            model_mod,
            "fused_moe_make_expert_params_mapping",
            lambda *a, **kw: [
                ("experts.w13_", "experts.0.w1.", 0, "w1"),
                ("experts.w13_", "experts.1.w1.", 1, "w1"),
            ],
        )
        assert len(m.get_expert_mapping()) == 2

        monkeypatch.setattr(
            model_mod,
            "is_pp_missing_parameter",
            lambda name, mod: "pp_missing" in name,
        )

        e8m0_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        plain_u8 = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
        f32_4 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        weights = [
            # Stacked: pp_missing, not in params_dict, and valid
            ("pp_missing.w1", f32_4),
            ("unknown.w1", f32_4),
            ("l0.w1", f32_4),
            # Experts: weight_name not in name, pp_missing, not in params_dict,
            # expert_id=0 returns False (no break), expert_id=1 with e8m0
            # returns True (break), and non-weight_scale expert
            ("l0.experts.other", plain_u8),
            ("pp_missing.experts.1.w1.weight_scale", e8m0_tensor),
            ("unknown.experts.1.w1.weight_scale", e8m0_tensor),
            ("l0.experts.0.w1.weight_scale", plain_u8),
            ("l0.experts.1.w1.weight_scale", e8m0_tensor),
            # attn_sink: pp_missing, not in params_dict, and valid
            ("pp_missing.attn_sink", f32_4),
            ("unknown.attn_sink", f32_4),
            ("l0.attn_sink", f32_4),
            # Plain: pp_missing, not in params_dict, and valid
            ("pp_missing.plain", f32_4),
            ("unknown.plain", f32_4),
            ("l0.plain", f32_4),
        ]
        loaded = m.load_weights(weights)
        assert loaded == {
            "l0.gate_up_proj",
            "l0.experts.w13_weight_scale",
            "l0.attn_sink",
            "l0.plain",
        }
        assert torch.equal(p_sink, torch.tensor([3.0, 4.0]))

        # _make_deepseek_v4_weights_mapper fp4 vs non-fp4
        mapper_fp4 = model_mod._make_deepseek_v4_weights_mapper("fp4")
        mapper_fp8 = model_mod._make_deepseek_v4_weights_mapper("fp8")
        assert mapper_fp4.apply_list(["layers.0.ffn.experts.2.w1.scale"]) == [
            "model.layers.0.ffn.experts.2.w1.weight_scale"
        ]
        assert mapper_fp8.apply_list(["layers.0.ffn.experts.2.w1.scale"]) == [
            "model.layers.0.ffn.experts.2.w1.weight_scale_inv"
        ]

        # DeepseekV4ForCausalLM __init__ (last rank True/False, expert_dtype fp8),
        # embed_input_ids, compute_logits, get_mtp_target_hidden_states,
        # get_expert_mapping, and load_weights with VllmDeepseekCompressor child
        class _StubModel(nn.Module):
            def __init__(self, *, vllm_config, prefix=""):
                super().__init__()
                self.layers = nn.ModuleList([nn.Identity()])
                self.aux_hidden_state_layers = ()
                self._mtp_hidden_buffer = torch.zeros((2, 4))

            def make_empty_intermediate_tensors(self, *a):
                return "empty_inter"

            def embed_input_ids(self, x):
                return x + 1

            def get_expert_mapping(self):
                return [("a", "b", 0, "w1")]

            def forward(
                self,
                input_ids,
                positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            ):
                return input_ids

        monkeypatch.setattr(
            model_mod, "ParallelLMHead", lambda v, h, prefix="": nn.Linear(h, v)
        )
        monkeypatch.setattr(
            model_mod, "LogitsProcessor", lambda v: (lambda head, hs: head(hs))
        )

        # PP last rank = True, expert_dtype = "fp8"
        monkeypatch.setattr(
            model_mod,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        monkeypatch.setattr(model_mod.DeepseekV4ForCausalLM, "model_cls", _StubModel)
        cfg_fp8 = self._make_vllm_cfg(expert_dtype="fp8")
        lm = model_mod.DeepseekV4ForCausalLM(vllm_config=cfg_fp8, prefix="")
        assert torch.equal(lm.embed_input_ids(torch.tensor([1])), torch.tensor([2]))
        assert lm.compute_logits(torch.zeros((2, 8))).shape == (2, 16)
        assert torch.equal(lm(torch.tensor([5]), torch.tensor([0])), torch.tensor([5]))
        assert lm.get_mtp_target_hidden_states() is lm.model._mtp_hidden_buffer
        assert lm.get_expert_mapping() == [("a", "b", 0, "w1")]
        lm.set_aux_hidden_state_layers((1,))
        assert lm.model.aux_hidden_state_layers == (1,)
        with pytest.raises(ValueError, match="Invalid DeepseekV4 aux"):
            lm.set_aux_hidden_state_layers((0,))

        # PP last rank = False, expert_dtype = "fp4"
        monkeypatch.setattr(
            model_mod,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
        )
        cfg_fp4 = self._make_vllm_cfg(expert_dtype="fp4")
        lm_mid = model_mod.DeepseekV4ForCausalLM(vllm_config=cfg_fp4, prefix="p")
        assert isinstance(lm_mid.lm_head, model_mod.PPMissingLayer)

        # load_weights with VllmDeepseekCompressor child
        comp_child = comp_mod.VllmDeepseekCompressor.__new__(
            comp_mod.VllmDeepseekCompressor
        )
        nn.Module.__init__(comp_child)
        comp_child._wkv_wgate_transposed = False
        comp_child.fused_wkv_wgate = nn.Linear(4, 6, bias=False)
        lm.model.add_module("comp", comp_child)

        monkeypatch.setattr(
            model_mod,
            "AutoWeightsLoader",
            lambda mod: SimpleNamespace(
                load_weights=lambda w, mapper=None: {"loaded_ok"}
            ),
        )
        res_loaded = lm.load_weights([])
        assert res_loaded == {"loaded_ok"}
        assert comp_child._wkv_wgate_transposed is True
