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

"""100% statement and branch coverage tests for TD7.68 modules:
- vllm_torchtpu.models.vllm.deepseek_v4.layers
- vllm_torchtpu.models.vllm.deepseek_v4.moe
- vllm_torchtpu.models.vllm.deepseek_v4.dspark
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models.utils import WeightsMapper

from vllm_torchtpu.models.vllm.deepseek_v4 import dspark as dspark_mod
from vllm_torchtpu.models.vllm.deepseek_v4 import layers as layers_mod
from vllm_torchtpu.models.vllm.deepseek_v4 import moe as moe_mod


class TestDeepseekV4Layers:
    def test_mhc_collapse_head_and_mlp(self, monkeypatch: pytest.MonkeyPatch):
        # 1. mhc_collapse_head
        hidden = torch.ones((3, 2, 4), dtype=torch.bfloat16)
        hc_fn = torch.ones((2, 8), dtype=torch.float32) * 0.1
        hc_scale = torch.tensor([1.0], dtype=torch.float32)
        hc_base = torch.zeros((2,), dtype=torch.float32)
        collapsed = layers_mod.mhc_collapse_head(
            hidden, hc_fn, hc_scale, hc_base, 1e-6, 1e-5
        )
        assert collapsed.shape == (3, 4)
        assert collapsed.dtype == torch.bfloat16

        # 2. DeepseekV4MLP with mocked linear layers
        monkeypatch.setattr(
            layers_mod,
            "MergedColumnParallelLinear",
            lambda h, inter_list, **kw: (
                lambda x: (torch.ones((x.shape[0], sum(inter_list))), None)
            ),
        )
        monkeypatch.setattr(
            layers_mod,
            "RowParallelLinear",
            lambda inter, h, **kw: lambda x: (torch.ones((x.shape[0], h)), None),
        )
        monkeypatch.setattr(
            layers_mod,
            "SiluAndMul",
            lambda: lambda g: g[:, : g.shape[1] // 2],
        )
        monkeypatch.setattr(
            layers_mod,
            "SiluAndMulWithClamp",
            lambda lim: lambda g: g[:, : g.shape[1] // 2] * lim,
        )

        # Invalid hidden_act -> ValueError
        with pytest.raises(ValueError, match="Unsupported activation"):
            layers_mod.DeepseekV4MLP(8, 16, "gelu")

        # swiglu_limit is None -> SiluAndMul
        mlp_plain = layers_mod.DeepseekV4MLP(8, 16, "silu", swiglu_limit=None)
        out_plain = mlp_plain(torch.zeros((2, 8)))
        assert out_plain.shape == (2, 8)

        # swiglu_limit is not None -> SiluAndMulWithClamp
        mlp_clamped = layers_mod.DeepseekV4MLP(8, 16, "silu", swiglu_limit=7.0)
        out_clamped = mlp_clamped(torch.zeros((2, 8)))
        assert out_clamped.shape == (2, 8)


class TestDeepseekV4MoE:
    def test_deepseek_v4_moe_branches(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(moe_mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(moe_mod, "get_tensor_model_parallel_rank", lambda: 0)

        class FakeGateLinear(nn.Module):
            def __init__(self, **kw):
                super().__init__()
                self.kw = kw

        class FakeFusedMoE(nn.Module):
            def __init__(self, **kw):
                super().__init__()
                self.kw = kw
                self.routed_experts = SimpleNamespace()
                self.do_naive_dispatch_combine = True

            def forward(self, hidden_states, router_logits, input_ids=None):
                return hidden_states * 2.0

        monkeypatch.setattr(moe_mod, "GateLinear", FakeGateLinear)
        monkeypatch.setattr(moe_mod, "FusedMoEFactory", FakeFusedMoE)
        monkeypatch.setattr(
            moe_mod,
            "DeepseekV4MLP",
            lambda **kw: SimpleNamespace(**kw),
        )

        def make_vllm_cfg(
            num_hash_layers: int,
            topk_method: str | None,
            n_shared_experts: int | None,
        ):
            hf_config = SimpleNamespace(
                routed_scaling_factor=1.5,
                hidden_size=8,
                n_routed_experts=4,
                swiglu_limit=None,
                norm_topk_prob=True,
                scoring_func="sqrtsoftplus",
                num_hash_layers=num_hash_layers,
                vocab_size=16,
                num_experts_per_tok=2,
                topk_method=topk_method,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=16,
                hidden_act="silu",
            )
            return SimpleNamespace(
                model_config=SimpleNamespace(hf_config=hf_config),
                quant_config=None,
            )

        # 1. Hash MoE layer (layer 0 < num_hash_layers=1) with shared_experts
        moe_hash = moe_mod.DeepseekV4MoE(
            make_vllm_cfg(1, None, 2), prefix="model.layers.0.ffn"
        )
        assert moe_hash.gate.tid2eid is not None
        assert (
            moe_hash.experts.routed_experts.hash_indices_table is moe_hash.gate.tid2eid
        )
        assert moe_hash.shared_experts is not None
        assert moe_hash.routes_dp_gathered_tokens is True

        # Hash MoE forward without input_ids raises ValueError
        x = torch.ones((3, 8))
        with pytest.raises(ValueError, match="requires input_ids"):
            moe_hash(x, input_ids=None)

        # Hash MoE forward with input_ids succeeds
        out_hash = moe_hash(x, input_ids=torch.tensor([1, 2, 3], dtype=torch.int32))
        assert torch.equal(out_hash, x * 2.0)

        # 2. Non-hash MoE with topk_method == "noaux_tc" and n_shared_experts is None
        moe_noaux = moe_mod.DeepseekV4MoE(
            make_vllm_cfg(0, "noaux_tc", None), prefix="model.layers.1.ffn"
        )
        assert moe_noaux.gate.tid2eid is None
        assert moe_noaux.gate.e_score_correction_bias is not None
        assert moe_noaux.shared_experts is None
        out_noaux = moe_noaux(x, input_ids=None)
        assert torch.equal(out_noaux, x * 2.0)

        # 3. Non-hash MoE with topk_method != "noaux_tc"
        moe_plain = moe_mod.DeepseekV4MoE(
            make_vllm_cfg(0, "greedy", None), prefix="model.layers.2.ffn"
        )
        assert moe_plain.gate.tid2eid is None
        assert moe_plain.gate.e_score_correction_bias is None


class TestDSparkModelAndHelpers:
    def test_remap_and_rewrite_dspark_name(self):
        # 1. Non-mtp name -> None
        assert dspark_mod._remap_dspark_name("embed.weight") is None
        # 2. Starts with "mtp." but regex does not match -> None
        assert dspark_mod._remap_dspark_name("mtp.invalid_stage") is None
        # 3. confidence_head -> None
        assert (
            dspark_mod._remap_dspark_name("mtp.2.confidence_head.proj.weight") is None
        )
        # 4. Head/main prefix -> model.<rest>
        assert (
            dspark_mod._remap_dspark_name("mtp.0.main_proj.weight")
            == "model.main_proj.weight"
        )
        assert dspark_mod._remap_dspark_name("mtp.2.norm.weight") == "model.norm.weight"
        # 5. Layer weight -> model.layers.<stage>.<rest>
        assert (
            dspark_mod._remap_dspark_name("mtp.1.attn.wq_a.weight")
            == "model.layers.1.attn.wq_a.weight"
        )

        # _rewrite_dspark_name with mapper returning None vs valid string
        mapper_skip = WeightsMapper(orig_to_new_prefix={"model.layers.1.": None})
        assert dspark_mod._rewrite_dspark_name("embed.weight", mapper_skip) is None
        assert (
            dspark_mod._rewrite_dspark_name("mtp.1.attn.wq_a.weight", mapper_skip)
            is None
        )
        assert (
            dspark_mod._rewrite_dspark_name("mtp.0.main_proj.weight", mapper_skip)
            == "main_proj.weight"
        )

    def test_insert_context_kv_branches(self):
        kv = torch.ones((3, 8), dtype=torch.bfloat16)
        positions = torch.tensor([0, 1, 2], dtype=torch.int32)
        meta = SimpleNamespace(
            seq_lens=torch.tensor([3], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )

        # 1. orig_sw_cache is None -> early return
        attn_none = SimpleNamespace(_get_active_sw_cache=lambda: None)
        dspark_mod._insert_context_kv(attn_none, kv, positions, meta)

        # 2. orig_sw_cache present; attn_op returns non-empty vs empty new_sw_cache
        orig_sw = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
        updated_sw = torch.full((2, 4, 8), 5.0, dtype=torch.bfloat16)

        attn_active = SimpleNamespace(
            _get_active_sw_cache=lambda: orig_sw,
            compress_ratio=1,
            custom_prefix="model.layers.6.attn",
            kv_rope=lambda k, p: k,
            mla_attn=SimpleNamespace(n_local_heads=2, head_dim=4),
            attn_sink=torch.zeros((2,), dtype=torch.float32),
            attn_op=lambda *ops: (None, updated_sw),
        )
        dspark_mod._insert_context_kv(attn_active, kv, positions, meta)
        assert torch.all(orig_sw == 5.0)

        # Empty new_sw_cache -> skips copy_
        orig_sw.zero_()
        attn_active.attn_op = lambda *ops: (None, torch.empty(0))
        dspark_mod._insert_context_kv(attn_active, kv, positions, meta)
        assert torch.all(orig_sw == 0.0)

    def test_dspark_model_and_causal_lm_methods(self, monkeypatch: pytest.MonkeyPatch):
        class FakeLinear(nn.Module):
            def __init__(self, in_f, out_f, **kw):
                super().__init__()
                self.weight = nn.Parameter(torch.ones((out_f, in_f)))

            def forward(self, x):
                return x[..., : self.weight.shape[0]]

        class FakeRMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                return x * 1.0

        class FakeMarkovHead(nn.Module):
            def __init__(self, vocab_size, draft_vocab_size, rank, prefix=""):
                super().__init__()
                self.markov_w1 = nn.Linear(rank, vocab_size, bias=False)

            def embed(self, token_ids):
                return torch.ones((token_ids.shape[0], 4))

            def bias(self, markov_embed, logits_proc):
                return torch.ones((markov_embed.shape[0], 16))

        class FakeDecoderLayer(nn.Module):
            window_size = 64

            def __init__(self, vllm_cfg, prefix="", topk_indices_buffer=None):
                super().__init__()
                self.prefix = prefix
                self.w = nn.Parameter(torch.ones(4))
                self.attn = SimpleNamespace(
                    window_size=type(self).window_size,
                    swa_cache_layer=SimpleNamespace(prefix=f"{prefix}.attn.swa"),
                    fused_wqa_wkv=lambda x: (torch.ones((x.shape[0], 12)), None),
                    mla_attn=SimpleNamespace(
                        q_lora_rank=4, n_local_heads=2, head_dim=4
                    ),
                    kv_norm=lambda kv: kv,
                    _get_active_sw_cache=lambda: None,
                )

            def forward(self, h, pos, ids, pm, rm, res):
                return h + 1.0, h, torch.ones_like(h), torch.zeros_like(h)

        monkeypatch.setattr(dspark_mod, "ReplicatedLinear", FakeLinear)
        monkeypatch.setattr(dspark_mod, "RMSNorm", FakeRMSNorm)
        monkeypatch.setattr(dspark_mod, "DSparkMarkovHead", FakeMarkovHead)
        monkeypatch.setattr(dspark_mod, "DeepseekV4DecoderLayer", FakeDecoderLayer)
        monkeypatch.setattr(
            dspark_mod,
            "LogitsProcessor",
            lambda vocab_size: lambda lm_head, h: lm_head(h),
        )

        hf_config = SimpleNamespace(
            hidden_size=8,
            hc_mult=2,
            hc_eps=1e-5,
            rms_norm_eps=1e-6,
            num_hidden_layers=4,
            dspark_target_layer_ids=[2, 3],
            n_mtp_layers=2,
            index_topk=4,
            vocab_size=16,
            draft_vocab_size=None,
            dspark_markov_rank=4,
            expert_dtype="fp4",
        )
        vllm_cfg = SimpleNamespace(
            quant_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=3,
                draft_model_config=SimpleNamespace(hf_config=hf_config),
            ),
        )
        monkeypatch.setattr(dspark_mod, "get_current_vllm_config", lambda: vllm_cfg)

        # 1. Window size error branch
        FakeDecoderLayer.window_size = 2
        with pytest.raises(ValueError, match="exceeds the DSV4 sliding window"):
            dspark_mod.DSparkDeepseekV4ForCausalLM(vllm_config=vllm_cfg)

        # 2. Valid construction
        FakeDecoderLayer.window_size = 64
        causal_lm = dspark_mod.DSparkDeepseekV4ForCausalLM(vllm_config=vllm_cfg)

        causal_lm.model.embed_tokens = lambda ids: torch.ones((ids.shape[0], 8))
        causal_lm.lm_head = lambda h: torch.ones((h.shape[0], 16))
        object.__setattr__(
            causal_lm.model,
            "_mhc_post_op_instance",
            lambda h, res, pm, rm: h,
        )

        ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        pos = torch.tensor([0, 1, 2], dtype=torch.int32)

        assert causal_lm.embed_input_ids(ids).shape == (3, 8)
        assert causal_lm.combine_hidden_states(torch.ones((3, 16))).shape == (3, 8)
        assert causal_lm.get_draft_attn_causal() == [False, False]
        assert causal_lm.get_draft_kv_cache_layer_names() == [
            "model.layers.4.attn.swa",
            "model.layers.5.attn.swa",
        ]

        # precompute_and_store_context_kv
        meta_tuple = (SimpleNamespace(), SimpleNamespace())
        main_x = causal_lm.precompute_and_store_context_kv(
            torch.ones((3, 16)), pos, meta_tuple
        )
        assert main_x.shape == (3, 8)

        # forward with inputs_embeds=None and inputs_embeds!=None
        out_from_ids = causal_lm(ids, pos, inputs_embeds=None)
        assert out_from_ids.shape == (3, 8)
        out_from_emb = causal_lm(ids, pos, inputs_embeds=torch.ones((3, 8)))
        assert out_from_emb.shape == (3, 8)

        # compute_logits, compute_draft_logits, map_draft_to_target, markov_*
        assert causal_lm.compute_logits(out_from_emb).shape == (3, 16)
        assert causal_lm.compute_draft_logits(out_from_emb).shape == (3, 16)
        assert torch.equal(causal_lm.map_draft_to_target(ids), ids)
        m_emb = causal_lm.markov_embed(ids)
        assert m_emb.shape == (3, 4)
        assert causal_lm.markov_bias(m_emb).shape == (3, 16)

        # load_weights: exercises mapped is None, layers.*, and direct param load
        seen_layer_weights = []
        object.__setattr__(
            causal_lm.model,
            "load_weights",
            lambda stream: {
                name for name, _w in (seen_layer_weights.append(x) or x for x in stream)
            },
        )
        weights = [
            ("embed.weight", torch.ones(4)),
            ("mtp.0.norm.weight", torch.ones_like(causal_lm.model.norm.weight)),
            ("mtp.1.attn.wq_a.weight", torch.ones(4)),
        ]
        loaded = causal_lm.load_weights(weights)
        assert loaded == {"model.norm.weight", "model.layers.1.attn.wq_a.weight"}
        assert len(seen_layer_weights) == 1
        assert seen_layer_weights[0][0] == "layers.1.attn.wq_a.weight"
