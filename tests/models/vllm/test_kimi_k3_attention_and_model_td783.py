# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-module unit tests for Kimi-K3 attention.py and model.py (TD7.83)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import vllm_torchtpu.models.vllm.kimi_k3.attention as attn_mod
import vllm_torchtpu.models.vllm.kimi_k3.model as model_mod
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import FusedPrefillCollectives


class _FakeLinear(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        self.input_size = input_size
        self.output_size = (
            sum(output_size) if isinstance(output_size, list) else output_size
        )
        self.weight = nn.Parameter(
            torch.zeros((self.output_size, input_size), dtype=torch.bfloat16)
        )
        self.bias = None

    def forward(self, x: torch.Tensor):
        return (
            torch.zeros((x.shape[0], self.output_size), dtype=x.dtype, device=x.device),
            None,
        )


def test_attention_helpers_and_mixed_parallel_linear(monkeypatch: pytest.MonkeyPatch):
    # 1. kda_state_dtype & _load_a_log (1D and 4D)
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    assert attn_mod.kda_state_dtype(vllm_cfg) == (torch.float32, torch.float32)

    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_rank", lambda: 0)
    param = torch.empty(2)
    attn_mod._load_a_log(param, torch.tensor([10.0, 20.0, 30.0, 40.0]))
    torch.testing.assert_close(param, torch.tensor([10.0, 20.0]))
    attn_mod._load_a_log(param, torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4, 1))
    torch.testing.assert_close(param, torch.tensor([1.0, 2.0]))

    # 2. MixedParallelMergedLinear validation & _slice_loaded_weight branches
    with pytest.raises(ValueError, match="sharding mode"):
        attn_mod.MixedParallelMergedLinear(
            4, [4, 4], [True], bias=False, quant_config=None, prefix="m"
        )
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(ValueError, match="must divide TP"):
        attn_mod.MixedParallelMergedLinear(
            4, [3, 4], [False, True], bias=False, quant_config=None, prefix="m"
        )

    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    mpl = attn_mod.MixedParallelMergedLinear(
        2, [4, 2], [False, True], bias=False, quant_config=None, prefix="m"
    )
    w_raw = torch.ones((4, 2))
    # loaded_shard_id is None
    assert mpl._slice_loaded_weight(mpl.weight, w_raw, None) is w_raw
    # replicate_outputs[1] is True
    assert mpl._slice_loaded_weight(mpl.weight, w_raw, 1) is w_raw
    # output_dim is None
    p_no_dim = nn.Parameter(torch.ones((4, 2)))
    assert mpl._slice_loaded_weight(p_no_dim, w_raw, 0) is w_raw
    # is_sharded_weight is True
    p_sharded = nn.Parameter(torch.ones((4, 2)))
    set_weight_attrs(p_sharded, {"output_dim": 0, "is_sharded_weight": True})
    assert mpl._slice_loaded_weight(p_sharded, w_raw, 0) is w_raw
    # loaded_size not divisible by mixed_tp_size
    with pytest.raises(ValueError, match="does not divide TP"):
        mpl._slice_loaded_weight(mpl.weight, torch.ones((3, 2)), 0)
    # weight_loader and weight_loader_v2
    mpl.weight_loader(mpl.weight, torch.ones((4, 2)), 0)
    mpl.weight_loader_v2(mpl.weight, torch.ones((2, 2)), 1)

    # 3. CausalDepthwiseConv1d
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 3)
    with pytest.raises(ValueError, match="divisible by TP size"):
        attn_mod.CausalDepthwiseConv1d(4, 3)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_rank", lambda: 1)
    conv = attn_mod.CausalDepthwiseConv1d(4, 3)
    conv.weight_loader(conv.weight, torch.ones((4, 3)))
    assert conv.weight.shape == (2, 1, 3)
    conv.weight_loader(conv.weight, torch.full((4, 1, 3), 2.0))
    assert torch.all(conv.weight == 2.0)


def test_mla_attention_full_coverage(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "MixedParallelMergedLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(attn_mod, name, _FakeLinear)
    monkeypatch.setattr(
        attn_mod,
        "RMSNorm",
        lambda dim, eps: SimpleNamespace(weight=torch.ones(dim), variance_epsilon=eps),
    )

    class _FakeMLAWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kimi_preprocess = None
            self.kimi_gate_weight = None

        def forward(self, positions, hidden_states, prefill_group=None):
            if prefill_group is not None:
                return hidden_states + 10.0
            return hidden_states + 1.0

    monkeypatch.setattr(attn_mod, "MultiHeadLatentAttentionWrapper", _FakeMLAWrapper)
    monkeypatch.setattr(attn_mod, "get_rope", lambda *a, **kw: ("rope", kw))
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)

    base_vllm_cfg = SimpleNamespace(
        quant_config=None,
        cache_config=SimpleNamespace(),
    )

    # Rejections in __init__
    cfg_no_nope = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=False,
    )
    with pytest.raises(ValueError, match="NoPE MLA only"):
        attn_mod.MultiHeadLatentAttention(cfg_no_nope, base_vllm_cfg, prefix="p")

    cfg_incomplete = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        kv_lora_rank=None,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
    )
    with pytest.raises(ValueError, match="Incomplete Kimi K3 MLA config"):
        attn_mod.MultiHeadLatentAttention(cfg_incomplete, base_vllm_cfg, prefix="p")

    cfg_bad_tp = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=3,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
    )
    with pytest.raises(ValueError, match="divisible by TP size"):
        attn_mod.MultiHeadLatentAttention(cfg_bad_tp, base_vllm_cfg, prefix="p")

    # 1. q_lora_rank=None with mla_use_output_gate=True and use_rope=True (default rope)
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    cfg_no_qlora = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=False,
        mla_use_output_gate=True,
    )
    cfg_no_qlora.rope_parameters = None
    mla_no_qlora = attn_mod.MultiHeadLatentAttention(
        cfg_no_qlora, base_vllm_cfg, prefix="p", use_rope=True
    )
    assert mla_no_qlora.q_proj is not None
    assert mla_no_qlora.g_proj is not None

    # 2. q_lora_rank!=None with mla_use_output_gate=False and yarn/llama rope
    cfg_qlora_no_gate = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=False,
    )
    cfg_qlora_no_gate.rope_parameters = {
        "rope_type": "yarn",
        "apply_yarn_scaling": True,
    }
    mla_yarn = attn_mod.MultiHeadLatentAttention(
        cfg_qlora_no_gate, base_vllm_cfg, prefix="p", use_rope=True
    )
    assert mla_yarn.fused_qkv_a_proj is not None
    assert not mla_yarn.mla_gate_is_fused

    cfg_qlora_no_gate.rope_parameters = {
        "rope_type": "yarn",
        "apply_yarn_scaling": False,
    }
    attn_mod.MultiHeadLatentAttention(
        cfg_qlora_no_gate, base_vllm_cfg, prefix="p", use_rope=True
    )

    # 3. q_lora_rank!=None with mla_use_output_gate=True + SP prefill
    def _init_fpc_mla(self, prefix):
        self.mla_ops = lambda eps: (lambda w: ("packed", w), "preprocess_fn")
        self.all_gather = lambda x, dim=0: x * 2.0
        self.reduce_scatter = lambda x, dim=0: x * 3.0

    monkeypatch.setenv("TPU_K3_SP_PREFILL", "1")
    monkeypatch.setattr(FusedPrefillCollectives, "__init__", _init_fpc_mla)

    cfg_qlora_gate = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=True,
    )
    mla_sp = attn_mod.MultiHeadLatentAttention(
        cfg_qlora_gate, base_vllm_cfg, prefix="p"
    )
    # Early return in prepare_sp_mla_gate (weight shape != (7168, 2496))
    mla_sp.prepare_sp_mla_gate()
    assert mla_sp.mla_attn.kimi_preprocess is None

    # Matching weight shape + flipped, mismatched eps -> ValueError, then matching eps
    mla_sp.fused_qkv_a_proj.weight = nn.Parameter(
        torch.zeros((7168, 2496), dtype=torch.bfloat16)
    )
    setattr(mla_sp.fused_qkv_a_proj, WEIGHT_FLIPPED_ATTR, True)
    mla_sp.kv_a_layernorm.variance_epsilon = 1e-4
    with pytest.raises(ValueError, match="matching norm epsilons"):
        mla_sp.prepare_sp_mla_gate()
    mla_sp.kv_a_layernorm.variance_epsilon = mla_sp.q_a_layernorm.variance_epsilon
    mla_sp.prepare_sp_mla_gate()
    assert mla_sp.mla_attn.kimi_preprocess == "preprocess_fn"

    # forward branches:
    pos = torch.arange(2)
    x = torch.ones((2, 16))
    # (a) sequence_parallel=True with FusedPrefillCollectives & kimi_preprocess!=None
    assert torch.all(mla_sp(pos, x, sequence_parallel=True) == 11.0)
    # (b) sequence_parallel=True with kimi_preprocess=None and non-FPC group
    mla_sp.mla_attn.kimi_preprocess = None
    mla_sp.prefill_group = SimpleNamespace(
        all_gather=lambda t, dim=0: t * 2.0,
        reduce_scatter=lambda t, dim=0: t * 4.0,
    )
    assert torch.all(mla_sp(pos, x, sequence_parallel=True) == 12.0)
    # (c) sequence_parallel=False
    monkeypatch.setattr(
        attn_mod,
        "get_tp_group",
        lambda: SimpleNamespace(all_reduce=lambda t: t * 5.0),
    )
    assert torch.all(mla_sp(pos, x, sequence_parallel=False) == 10.0)


def test_kda_attention_full_coverage(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(attn_mod, name, _FakeLinear)
    monkeypatch.setattr(
        attn_mod, "RMSNorm", lambda dim, eps: SimpleNamespace(weight=torch.ones(dim))
    )
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        attn_mod,
        "build_kimi_dispatched_kda_op",
        lambda *a, **kw: (lambda qkv, *rest: qkv[:, :384].view(-1, 3, 128)),
    )
    monkeypatch.setattr(
        attn_mod,
        "build_kimi_pooled_kda_op",
        lambda *a, **kw: (lambda qkv, *rest: qkv[:, :384].view(-1, 3, 128)),
    )

    # Rejections in __init__
    cfg_none = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        linear_attn_config=None,
    )
    with pytest.raises(ValueError, match="requires linear_attn_config"):
        attn_mod.KimiDeltaAttention(
            cfg_none, SimpleNamespace(speculative_config=None), prefix="k0"
        )

    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 5)
    cfg_kda = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 3,
            "short_conv_kernel_size": 3,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )
    with pytest.raises(ValueError, match="divisible by TP size"):
        attn_mod.KimiDeltaAttention(
            cfg_kda, SimpleNamespace(speculative_config=None), prefix="k0"
        )

    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    def _init_fpc_kda(self, prefix):
        self.gather_project = lambda h, w: h @ w
        self.project_reduce_scatter = lambda out, proj: out + 5.0

    monkeypatch.setenv("TPU_K3_SP_PREFILL", "1")
    monkeypatch.setattr(FusedPrefillCollectives, "__init__", _init_fpc_kda)

    ctx: dict[str, object] = {}
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        quant_config=None,
        compilation_config=SimpleNamespace(static_forward_context=ctx),
    )
    kda_full = attn_mod.KimiDeltaAttention(cfg_kda, vllm_cfg, prefix="k_full")
    with pytest.raises(ValueError, match="Duplicate layer name"):
        attn_mod.KimiDeltaAttention(cfg_kda, vllm_cfg, prefix="k_full")

    assert kda_full.mamba_type == attn_mod.MambaAttentionBackendEnum.LINEAR
    assert kda_full.get_state_dtype() == (torch.float32, torch.float32)
    assert kda_full.get_state_shape() == ((2, 1, 1152), (3, 128, 128))

    # Also build low-rank gate variant (use_full_rank_gate=False)
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    cfg_kda_low = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 3,
            "short_conv_kernel_size": 3,
            "gate_lower_bound": None,
            "use_full_rank_gate": False,
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )
    kda_low = attn_mod.KimiDeltaAttention(cfg_kda_low, vllm_cfg, prefix="k_low")

    # _metadata branches
    monkeypatch.setattr(attn_mod, "is_forward_context_available", lambda: False)
    assert kda_full._metadata() is None
    monkeypatch.setattr(attn_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata="not_dict"),
    )
    assert kda_full._metadata() is None
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"k_full": "bad"}),
    )
    with pytest.raises(TypeError, match="incompatible attention metadata"):
        kda_full._metadata()

    real_md = AttentionMetadata.__new__(AttentionMetadata)
    real_md.query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    real_md.seq_lens = torch.tensor([2], dtype=torch.int32)
    real_md.request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    real_md.mamba_state_indices = torch.tensor([0], dtype=torch.int32)
    real_md.mamba_request_distribution = None
    real_md.mamba_slot_read_offsets = None
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"k_full": real_md, "k_low": real_md}),
    )
    assert kda_full._metadata() is real_md

    # process_weights_after_loading & _pack_prefill_projection
    kda_low.process_weights_after_loading(torch.bfloat16)
    assert kda_low.packed_prefill_weight is None
    kda_full.process_weights_after_loading(torch.bfloat16)
    assert kda_full.packed_prefill_weight is None

    for proj, width in (
        (kda_full.fused_qkvb_proj, 1155),
        (kda_full.g_proj, 384),
        (kda_full.f_a_proj, 128),
    ):
        proj.weight = nn.Parameter(torch.zeros((7168, width), dtype=torch.bfloat16))
        setattr(proj, WEIGHT_FLIPPED_ATTR, True)
    kda_full._pack_prefill_projection()
    assert kda_full.packed_prefill_weight is not None

    # forward branches on kda_full (with packed_prefill_weight != None):
    tp_group = SimpleNamespace(
        all_gather=lambda t, dim=0: t,
        reduce_scatter=lambda t, dim=0: t,
        all_reduce=lambda t: t,
    )
    monkeypatch.setattr(attn_mod, "get_tp_group", lambda: tp_group)
    x7168 = torch.zeros((2, 7168), dtype=torch.bfloat16)
    pos = torch.arange(2)

    # (1) kv_cache is None -> zeros path + sequence_parallel=True with FPC
    assert kda_full(pos, x7168, sequence_parallel=True).shape == (2, 384)

    # (2) packed_prefill_weight != None with non-FusedPrefillCollectives prefill_group
    #     and sequence_parallel=True vs False
    kda_full.prefill_group = tp_group
    assert kda_full(pos, x7168, sequence_parallel=True).shape == (2, 16)
    assert kda_full(pos, x7168, sequence_parallel=False).shape == (2, 16)

    # (3) Singleton kv_cache -> _core_attention_pooled error and success branches
    kda_full.packed_prefill_weight = None
    x16 = torch.zeros((2, 16), dtype=torch.bfloat16)
    kda_full.kv_cache = [torch.ones((2, 4, 2, 8), dtype=torch.bfloat16)]

    monkeypatch.setattr(
        attn_mod, "is_pcp_streaming_attention_metadata", lambda md: True
    )
    with pytest.raises(NotImplementedError, match="PCP streaming prefill"):
        kda_full(pos, x16, sequence_parallel=True)

    monkeypatch.setattr(
        attn_mod, "is_pcp_streaming_attention_metadata", lambda md: False
    )
    real_md.mamba_slot_read_offsets = torch.zeros((1,), dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="Speculative decoding"):
        kda_full(pos, x16, sequence_parallel=False)

    real_md.mamba_slot_read_offsets = None
    real_md.mamba_state_indices = None
    with pytest.raises(RuntimeError, match="requires mamba_state_indices"):
        kda_full(pos, x16, sequence_parallel=False)

    real_md.mamba_state_indices = torch.tensor([0], dtype=torch.int32)
    assert kda_full(pos, x16, sequence_parallel=False).shape == (2, 16)

    # (4) 2-tensor kv_cache on kda_low: cover window_distribution != None and
    #     slot_read_offsets != None vs None (with mamba_state_indices None and not None)
    kda_low.kv_cache = (
        torch.ones((1, 8, 3), dtype=torch.float32),
        torch.ones((1, 3, 128, 128), dtype=torch.float32),
    )
    real_md.mamba_request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    real_md.mamba_slot_read_offsets = torch.tensor([0], dtype=torch.int32)
    assert kda_low(pos, x16, sequence_parallel=False).shape == (2, 16)

    real_md.mamba_request_distribution = None
    real_md.mamba_slot_read_offsets = None
    real_md.mamba_state_indices = None
    assert kda_low(pos, x16, sequence_parallel=False).shape == (2, 16)

    real_md.mamba_state_indices = torch.tensor([0], dtype=torch.int32)
    assert kda_low(pos, x16, sequence_parallel=False).shape == (2, 16)


def test_kimi_decoder_layer_model_and_causal_lm_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        model_mod,
        "KimiDeltaAttention",
        lambda cfg, vcfg, prefix: (lambda pos, h, sequence_parallel=False: h + 1.0),
    )
    monkeypatch.setattr(
        model_mod,
        "MultiHeadLatentAttention",
        lambda cfg, vcfg, prefix: (lambda pos, h, sequence_parallel=False: h + 2.0),
    )
    monkeypatch.setattr(
        model_mod,
        "KimiMoE",
        lambda cfg, qcfg, prefix: (lambda h, sequence_parallel=False: h + 3.0),
    )
    monkeypatch.setattr(
        model_mod,
        "KimiMLP",
        lambda *a, **kw: (lambda h, sequence_parallel=False: h + 4.0),
    )
    monkeypatch.setattr(model_mod, "RMSNorm", lambda dim, eps: nn.Identity())
    monkeypatch.setattr(
        model_mod,
        "AttentionResidual",
        lambda dim, eps, prefix: (lambda p, b: p + 0.5),
    )
    monkeypatch.setattr(
        model_mod,
        "VocabParallelEmbedding",
        lambda vocab, dim, **kw: nn.Embedding(vocab, dim),
    )
    monkeypatch.setattr(
        model_mod,
        "ParallelLMHead",
        lambda vocab, dim, **kw: nn.Linear(dim, vocab, bias=False),
    )
    monkeypatch.setattr(
        model_mod,
        "LogitsProcessor",
        lambda vocab: (lambda head, h: head(h)),
    )
    monkeypatch.setattr(
        model_mod,
        "TokenShardCollectives",
        lambda group: SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda t, dim=0: t.repeat(2, 1),
        ),
    )
    monkeypatch.setattr(model_mod, "get_tp_group", lambda: SimpleNamespace())

    cfg = KimiLinearConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        attn_res_block_size=1,
        linear_attn_config={
            "head_dim": 2,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "kda_layers": [1],
            "full_attn_layers": [2],
        },
    )

    # 1. SP prefill parallel_config validation error vs success
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "1")
    bad_vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=cfg, dtype=torch.float32),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=False,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=0),
    )
    with pytest.raises(ValueError, match="requires TP32/EP32 and PP1/DP1"):
        model_mod.KimiModel(vllm_config=bad_vllm_cfg)

    good_vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=cfg, dtype=torch.float32),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=0),
    )
    causal_lm = model_mod.KimiLinearForCausalLM(vllm_config=good_vllm_cfg)

    # KimiDecoderLayer with _starts_attn_res_block=False (layer 1 when block_size=2)
    cfg_bs2 = KimiLinearConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=None,
        attn_res_block_size=2,
    )
    layer_non_start = model_mod.KimiDecoderLayer(
        cfg_bs2, good_vllm_cfg, prefix="model.layers.1"
    )
    h_out, b_out = layer_non_start(
        torch.arange(2),
        torch.ones((2, 4)),
        torch.ones((2, 1, 4)),
        sequence_parallel=False,
    )
    assert h_out.shape == (2, 4) and b_out.shape == (2, 1, 4)

    # KimiDecoderLayer with attn_res_block_size=None (both is_moe=False and is_moe=True)
    # and KimiModel with sp_prefill=False and attn_res_block_size=None
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    cfg_no_res = KimiLinearConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        attn_res_block_size=None,
    )
    vllm_no_res = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=cfg_no_res, dtype=torch.float32),
        parallel_config=good_vllm_cfg.parallel_config,
        cache_config=good_vllm_cfg.cache_config,
        quant_config=None,
        compilation_config=SimpleNamespace(mode=0),
    )
    model_no_res = model_mod.KimiModel(vllm_config=vllm_no_res)
    # Forward with inputs_embeds != None and block_residuals is None:
    # cover both empty aux_layers and non-empty aux_layers where 0 is not in aux_layers
    embeds_in = torch.ones((2, 4))
    assert model_no_res(None, torch.arange(2), inputs_embeds=embeds_in).shape == (2, 4)
    model_no_res.aux_hidden_state_layers = (1,)
    out_nr, aux_nr = model_no_res(None, torch.arange(2), inputs_embeds=embeds_in)
    assert out_nr.shape == (2, 4) and len(aux_nr) == 1

    # KimiLinearForCausalLM methods
    ids = torch.tensor([1, 2, 3, 4])
    pos = torch.arange(4)
    assert causal_lm.embed_input_ids(ids).shape == (4, 4)

    with pytest.raises(ValueError, match="Invalid Kimi aux hidden-state layers"):
        causal_lm.set_aux_hidden_state_layers((-1,))

    # Forward without aux layers (sequence_parallel=True via input_ids)
    out_plain = causal_lm(ids, pos)
    assert out_plain.shape == (4, 4)

    # Forward with aux layers (0, 1) and sequence_parallel=True
    causal_lm.set_aux_hidden_state_layers((0, 1))
    out_aux, aux_list = causal_lm(ids, pos)
    assert out_aux.shape == (4, 4) and len(aux_list) == 2

    with pytest.raises(ValueError, match="does not support PP"):
        causal_lm(ids, pos, intermediate_tensors=object())

    assert causal_lm.compute_logits(out_plain).shape == (4, 16)

    # process_weights_after_loading: attach real-typed child modules
    class _AR(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def process_weights_after_loading(self):
            self.called = True

    class _MOE(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def initialize_dummy_router(self):
            self.called = True

    class _MLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def prepare_sp_mla_gate(self):
            self.called = True

    monkeypatch.setattr(model_mod, "AttentionResidual", _AR)
    monkeypatch.setattr(model_mod, "KimiMoE", _MOE)
    monkeypatch.setattr(model_mod, "MultiHeadLatentAttention", _MLA)
    ar_m, moe_m, mla_m = _AR(), _MOE(), _MLA()
    causal_lm.extra_children = nn.ModuleList([ar_m, moe_m, mla_m])
    causal_lm.process_weights_after_loading()
    assert ar_m.called and moe_m.called and mla_m.called

    # Classmethods for mamba state
    assert model_mod.KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
        good_vllm_cfg
    ) == (torch.float32, torch.float32)
    assert len(model_mod.KimiLinearForCausalLM.get_mamba_state_copy_func()) == 2

    # load_weights: cover q_lora_rank!=None vs None, vision_tower skip,
    # fused_attention_params_mapping, experts_use_weight (.weight_packed -> .weight),
    # and expert_params_mapping (both missing and matching parameter with weight_loader)
    causal_lm.config.q_lora_rank = 8
    causal_lm.config.tie_word_embeddings = False
    p_fused = nn.Parameter(torch.zeros((2, 2)))
    p_expert = nn.Parameter(torch.zeros((2, 2)))
    expert_loaded_calls = []
    p_expert.weight_loader = (
        lambda param, loaded_weight, weight_name, shard_id, expert_id: (
            expert_loaded_calls.append((weight_name, shard_id, expert_id))
        )
    )
    monkeypatch.setattr(
        causal_lm,
        "named_parameters",
        lambda: [
            ("model.layers.0.self_attn.fused_qkvb_proj.weight", p_fused),
            (
                "model.layers.1.block_sparse_moe.experts.routed_experts.w13_weight",
                p_expert,
            ),
            ("model.embed_tokens.weight", causal_lm.model.embed_tokens.weight),
        ],
    )
    monkeypatch.setattr(
        model_mod,
        "AutoWeightsLoader",
        lambda self_: SimpleNamespace(
            load_weights=lambda gen, mapper=None: {name for name, _ in gen}
        ),
    )
    w_dummy = torch.zeros((2, 2))
    loaded_names = causal_lm.load_weights(
        [
            ("vision_tower.patch_embed.weight", w_dummy),
            ("mm_projector.proj.0.weight", w_dummy),
            ("model.layers.0.self_attn.q_proj.weight", w_dummy),
            ("model.layers.9.self_attn.q_proj.weight", w_dummy),
            ("model.layers.0.block_sparse_moe.experts.999.w1.weight", w_dummy),
            ("model.layers.1.block_sparse_moe.experts.0.w1.weight_packed", w_dummy),
            ("model.embed_tokens.weight", w_dummy),
        ]
    )
    assert "model.layers.0.self_attn.fused_qkvb_proj.weight" in loaded_names
    assert "model.layers.9.self_attn.q_proj.weight" in loaded_names
    assert "model.layers.0.block_sparse_moe.experts.999.w1.weight" in loaded_names
    assert (
        "model.layers.1.block_sparse_moe.experts.routed_experts.w13_weight"
        in loaded_names
    )
    assert len(expert_loaded_calls) == 1 and expert_loaded_calls[0][1:] == ("w1", 0)

    # Also cover q_lora_rank=None and tie_word_embeddings=True branch in load_weights
    causal_lm.config.q_lora_rank = None
    causal_lm.config.tie_word_embeddings = True
    assert "model.embed_tokens.weight" in causal_lm.load_weights(
        [("model.embed_tokens.weight", w_dummy)]
    )


def test_kimi_k3_for_conditional_generation_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
):
    # 1. _tpu_tp_all_gather
    sync_calls = []
    monkeypatch.setattr(
        model_mod, "synchronize_tensors", lambda t: sync_calls.append(t.shape)
    )
    monkeypatch.setattr(
        model_mod, "tensor_model_parallel_all_gather", lambda t, dim=-1: t * 2.0
    )
    gathered = model_mod._tpu_tp_all_gather(torch.ones((2, 3)), dim=0)
    assert torch.all(gathered == 2.0) and len(sync_calls) == 2

    # 1b. get_placeholder_str classmethod ("image" and unsupported ValueError)
    assert (
        model_mod.KimiK3ForConditionalGeneration.get_placeholder_str("image", 0)
        == "<|kimi_image_placeholder|>"
    )
    with pytest.raises(ValueError, match="Unsupported modality"):
        model_mod.KimiK3ForConditionalGeneration.get_placeholder_str("audio", 0)

    # 2. KimiK3ForConditionalGeneration.__init__ branches
    with pytest.raises(ValueError, match="requires multimodal config"):
        model_mod.KimiK3ForConditionalGeneration(
            SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(), multimodal_config=None
                )
            )
        )

    class _FakeVisionTower(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.p = nn.Parameter(torch.ones((1,), dtype=torch.float32))
            self.merge_kernel_size = (2, 2)
            self.patch_embed = SimpleNamespace(
                patch_size=(2, 2),
                proj=SimpleNamespace(in_channels=3),
                pos_emb=lambda h, w: torch.zeros((1, h * w, 4), dtype=torch.float32),
                __call__=lambda pv, _, pos_embeds=None: pv,
            )
            self.encoder = SimpleNamespace(
                rope_2d=SimpleNamespace(max_height=16, max_width=16),
                prepare_encoder_metadata=lambda grid, device=None: {"meta": 1},
                __call__=lambda h, _, encoder_metadata=None: h,
            )

        def prepare_encoder_cudagraph_metadata(
            self, grid, *, max_batch_size, device, max_seqlen_override=None
        ):
            return {"cu_seqlens": torch.tensor([0, 4]), "none_val": None}

        def forward(self, pv, grid, encoder_metadata=None):
            return [torch.ones((2, 4))]

    class _FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_layers = ()
            self.processed = False

        def set_aux_hidden_state_layers(self, layers):
            self.aux_layers = layers

        def embed_input_ids(self, ids):
            return torch.zeros((ids.shape[0], 4), dtype=torch.float32)

        def forward(self, **kw):
            return torch.ones((2, 4))

        def compute_logits(self, h):
            return torch.ones((h.shape[0], 8))

        def process_weights_after_loading(self):
            self.processed = True

    monkeypatch.setattr(model_mod, "KimiK3MoonViT3dPretrainedModel", _FakeVisionTower)
    monkeypatch.setattr(
        model_mod, "KimiK25MultiModalProjector", lambda **kw: nn.Identity()
    )
    monkeypatch.setattr(
        model_mod, "init_vllm_registered_model", lambda **kw: _FakeLanguageModel()
    )
    monkeypatch.setattr(model_mod, "is_vit_use_data_parallel", lambda heads: True)
    monkeypatch.setattr(
        model_mod.current_platform, "current_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        model_mod.KimiK3ForConditionalGeneration,
        "_mark_tower_model",
        lambda self, vcfg, mod: nullcontext(),
    )
    monkeypatch.setattr(
        model_mod.KimiK3ForConditionalGeneration,
        "_mark_language_model",
        lambda self, vcfg: nullcontext(),
    )

    hf_cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4,
            linear_attn_config={
                "num_heads": 4,
                "head_dim": 2,
                "short_conv_kernel_size": 3,
            },
        ),
        vision_config=SimpleNamespace(num_attention_heads=4),
        media_placeholder_token_id=99,
    )
    # Construct with CompressedTensorsConfig (ignored -> vision_quant_config is None)
    ct_quant = compressed_tensors.CompressedTensorsConfig.__new__(
        compressed_tensors.CompressedTensorsConfig
    )
    vcfg_ct = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_cfg,
            multimodal_config=SimpleNamespace(),
            dtype=torch.float32,
        ),
        quant_config=ct_quant,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    mm_model = model_mod.KimiK3ForConditionalGeneration(vcfg_ct)

    # Construct with non-CompressedTensorsConfig (vision_quant_config is not None)
    other_quant = SimpleNamespace()
    vcfg_other = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_cfg,
            multimodal_config=SimpleNamespace(),
            dtype=torch.float32,
        ),
        quant_config=other_quant,
    )
    model_mod.KimiK3ForConditionalGeneration(vcfg_other)

    # set_aux_hidden_state_layers (valid + missing attr error)
    mm_model.set_aux_hidden_state_layers((1,))
    assert mm_model.language_model.aux_layers == (1,)
    saved_lm = mm_model.language_model
    mm_model.language_model = nn.Module()
    with pytest.raises(RuntimeError, match="does not expose aux hidden states"):
        mm_model.set_aux_hidden_state_layers((1,))
    mm_model.language_model = saved_lm

    # _parse_and_validate_media_input branches
    assert mm_model._parse_and_validate_media_input() is None
    with pytest.raises(TypeError, match="pixel_values must be a tensor"):
        mm_model._parse_and_validate_media_input(pixel_values="bad")
    with pytest.raises(TypeError, match="grid_thws must be a tensor"):
        mm_model._parse_and_validate_media_input(
            pixel_values=[torch.zeros((1, 2, 3, 2, 2))], grid_thws="bad"
        )
    with pytest.raises(ValueError, match="unexpected grid_thws shape"):
        mm_model._parse_and_validate_media_input(
            pixel_values=torch.zeros((2, 3, 2, 2)),
            grid_thws=torch.zeros((1, 2), dtype=torch.int64),
        )
    parsed = mm_model._parse_and_validate_media_input(
        pixel_values=[torch.zeros((1, 2, 3, 2, 2))],
        grid_thws=torch.tensor([[1, 2, 2]]),
    )
    assert parsed["pixel_values"].shape == (2, 3, 2, 2)

    # embed_multimodal (None, use_data_parallel=True, use_data_parallel=False)
    assert mm_model.embed_multimodal() is None
    monkeypatch.setattr(
        model_mod,
        "vision_tower_forward",
        lambda vt, pv, grid, mm_projector, use_data_parallel: ("vt", use_data_parallel),
    )
    assert mm_model.embed_multimodal(
        pixel_values=torch.zeros((4, 3, 2, 2)),
        grid_thws=torch.tensor([[1, 2, 2]]),
    ) == ("vt", True)
    mm_model.use_data_parallel = False
    assert mm_model.embed_multimodal(
        pixel_values=torch.zeros((4, 3, 2, 2)),
        grid_thws=torch.tensor([[1, 2, 2]]),
    ) == ("vt", False)

    # SupportsEncoderCudaGraph methods
    assert mm_model.get_encoder_cudagraph_config().out_hidden_size == 4
    assert mm_model.get_input_modality({}) == "image"
    assert mm_model.get_max_frames_per_video() == 1
    assert mm_model.get_encoder_cudagraph_budget_range(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
            model_config=SimpleNamespace(max_model_len=512),
        )
    ) == (64, 256)
    assert mm_model._grid_thw_list(torch.tensor([[1, 2, 2]])) == [[1, 2, 2]]
    assert mm_model._grid_thw_list([[1, 2, 2]]) == [[1, 2, 2]]
    specs = mm_model.get_encoder_cudagraph_item_specs(
        {"grid_thws": torch.tensor([[1, 2, 2], [2, 2, 2]])}
    )
    assert (
        len(specs) == 2 and specs[0].output_tokens == 1 and specs[1].output_tokens == 2
    )

    # select_encoder_cudagraph_items: empty indices vs non-empty indices
    mm_kw = {
        "pixel_values": torch.zeros((8, 3, 2, 2)),
        "grid_thws": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    empty_sel = mm_model.select_encoder_cudagraph_items(mm_kw, [])
    assert empty_sel["pixel_values"].shape[0] == 0
    non_empty_sel = mm_model.select_encoder_cudagraph_items(mm_kw, [1])
    assert (
        non_empty_sel["pixel_values"].shape[0] == 4
        and non_empty_sel["grid_thws"].shape[0] == 1
    )

    # prepare_encoder_cudagraph_capture_inputs: normal and ValueError
    cap = mm_model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=2,
        max_batch_size=1,
        max_frames_per_batch=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert "pixel_values" in cap.values
    mm_model.vision_tower.encoder.rope_2d.max_height = 1
    with pytest.raises(ValueError, match="Encoder cudagraph budget"):
        mm_model.prepare_encoder_cudagraph_capture_inputs(
            token_budget=4,
            max_batch_size=1,
            max_frames_per_batch=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    mm_model.vision_tower.encoder.rope_2d.max_height = 16

    replay = mm_model.prepare_encoder_cudagraph_replay_buffers(
        {"pixel_values": torch.zeros((4, 3, 2, 2)), "grid_thws": [[1, 2, 2]]},
        max_batch_size=1,
        max_frames_per_batch=1,
    )
    assert "pixel_values" in replay.values

    mm_model.vision_tower.patch_embed = lambda pv, _, pos_embeds=None: pv
    mm_model.vision_tower.encoder = lambda h, _, encoder_metadata=None: h
    monkeypatch.setattr(model_mod, "tpool_patch_merger_packed", lambda h, idx: h)
    cg_out = mm_model.encoder_cudagraph_forward(
        {
            "pixel_values": torch.ones((2, 4)),
            "merge_gather_idx": torch.zeros((2,), dtype=torch.int64),
            "pos_embeds": torch.zeros((2, 4)),
        }
    )
    assert cg_out.shape == (2, 4)

    mm_model.vision_tower.encoder = SimpleNamespace(
        prepare_encoder_metadata=lambda grid, device=None: {"meta": 1}
    )
    monkeypatch.setattr(model_mod, "mm_projector_forward", lambda proj, outs: outs)
    eager_out = mm_model.encoder_eager_forward(
        {"pixel_values": torch.zeros((4, 3, 2, 2)), "grid_thws": [[1, 2, 2]]}
    )
    assert eager_out.shape == (2, 4)

    # embed_input_ids branches
    monkeypatch.setattr(
        mm_model,
        "_embed_text_input_ids",
        lambda ids, fn, is_multimodal=None: fn(ids),
    )
    ids2 = torch.tensor([1, 2])
    assert mm_model.embed_input_ids(ids2, None).shape == (2, 4)
    assert mm_model.embed_input_ids(ids2, []).shape == (2, 4)
    with pytest.raises(ValueError, match="is_multimodal is required"):
        mm_model.embed_input_ids(ids2, [torch.ones((1, 4))], is_multimodal=None)
    # actual == 0 branch
    assert mm_model.embed_input_ids(
        ids2,
        [torch.zeros((0, 4))],
        is_multimodal=torch.tensor([False, False]),
    ).shape == (2, 4)
    # actual != expected ValueError branch
    with pytest.raises(ValueError, match="Attempted to assign"):
        mm_model.embed_input_ids(
            ids2,
            [torch.ones((2, 4))],
            is_multimodal=torch.tensor([True, False]),
        )
    # CPU non-TPU return branch (merged.device.type != "tpu")
    cpu_merged = mm_model.embed_input_ids(
        ids2,
        [torch.ones((1, 4))],
        is_multimodal=torch.tensor([True, False]),
    )
    assert cpu_merged.shape == (2, 4)

    # TPU device materialization branch in embed_input_ids
    fake_merged = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    monkeypatch.setattr(torch, "where", lambda m, a, b: fake_merged)
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda t: SimpleNamespace(copy_=lambda src: torch.ones((2, 4))),
    )
    tpu_merged = mm_model.embed_input_ids(
        ids2,
        [torch.ones((1, 4))],
        is_multimodal=torch.tensor([True, False]),
    )
    assert torch.all(tpu_merged == 1.0)

    # forward, compute_logits, process_weights, mamba methods, load_weights
    assert mm_model(ids2, torch.arange(2)).shape == (2, 4)
    assert mm_model.compute_logits(torch.ones((2, 4))).shape == (2, 8)
    mm_model.process_weights_after_loading()
    assert mm_model.language_model.processed

    vcfg_with_hf = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_cfg, dtype=torch.bfloat16),
        with_hf_config=lambda tc: SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=tc, dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
        ),
    )
    assert model_mod.KimiK3ForConditionalGeneration.get_mamba_state_dtype_from_config(
        vcfg_with_hf
    ) == (torch.float32, torch.float32)
    assert model_mod.KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(
        vcfg_with_hf
    ) == ((2, 1, 12), (2, 2, 2))
    assert (
        len(model_mod.KimiK3ForConditionalGeneration.get_mamba_state_copy_func()) == 2
    )

    monkeypatch.setattr(
        model_mod,
        "AutoWeightsLoader",
        lambda self_: SimpleNamespace(
            load_weights=lambda w, mapper=None: {"loaded_mm"}
        ),
    )
    assert mm_model.load_weights([]) == {"loaded_mm"}
