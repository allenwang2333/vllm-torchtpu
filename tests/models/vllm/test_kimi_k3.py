# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the TPU Kimi model."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import vllm_torchtpu.models.vllm.kimi_k3 as kimi
import vllm_torchtpu.models.vllm.kimi_k3.attention as kimi_attention
import vllm_torchtpu.models.vllm.kimi_k3.moe as kimi_moe
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.vllm.custom_ops import \
    kda_attention_op as kimi_custom_ops
from vllm_torchtpu.models.vllm.kimi_k3 import (KimiDeltaAttention,
                                               KimiLinearForCausalLM,
                                               KimiModel,
                                               MultiHeadLatentAttention)
from vllm_torchtpu.models.vllm.kimi_k3.layers import SituAndMul


def test_kimi_architectures_are_registered() -> None:
    architectures = (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    )
    previous = {
        architecture: ModelRegistry.models.get(architecture)
        for architecture in architectures
    }
    try:
        register_layers()
        for architecture in architectures:
            registered = ModelRegistry.models[architecture]
            assert registered.load_model_cls() is KimiLinearForCausalLM
    finally:
        for architecture, registered in previous.items():
            if registered is None:
                ModelRegistry.models.pop(architecture, None)
            else:
                ModelRegistry.models[architecture] = registered


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta: float | None) -> None:
    activation = SituAndMul(beta=4.0, linear_beta=linear_beta)
    inputs = torch.tensor([[-3.0, 0.5, 7.0, -30.0, 2.0, 40.0]])
    gate, up = inputs.chunk(2, dim=-1)
    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = (up if linear_beta is None else linear_beta *
                   torch.tanh(up / linear_beta))

    torch.testing.assert_close(activation(inputs), expected_gate * expected_up)


def test_checkpoint_name_mapping_is_conditional_for_mla() -> None:
    common_mapper = KimiLinearForCausalLM.hf_to_vllm_mapper
    assert common_mapper.apply_list([
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ]) == [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ]

    fused_mapper = common_mapper | KimiLinearForCausalLM.fused_mla_mapper
    weight = torch.empty(1)
    [(name, weight)] = list(
        fused_mapper.apply([
            ("language_model.layers.1.self_attn.q_a_proj.weight", weight),
        ]))
    assert name == "model.layers.1.self_attn.fused_qkv_a_proj.weight"
    assert weight.shard_id == 0


def test_expert_weight_loader_uses_current_fused_moe_names() -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=1,
        q_lora_rank=None,
        tie_word_embeddings=False,
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    moe = model.model.layers[0].block_sparse_moe = nn.Module()
    moe.experts = nn.Module()
    moe.experts.routed_experts = nn.Module()
    parameter = nn.Parameter(torch.empty(1))
    moe.experts.routed_experts.w13_weight = parameter

    calls = []

    def weight_loader(param, loaded_weight, weight_name, **kwargs):
        calls.append((param, loaded_weight, weight_name, kwargs))

    set_weight_attrs(parameter, {"weight_loader": weight_loader})
    loaded_weight = torch.ones(1)
    loaded = model.load_weights([
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", loaded_weight),
    ])

    parameter_name = (
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight")
    assert loaded == {parameter_name}
    assert calls == [(
        parameter,
        loaded_weight,
        parameter_name,
        {
            "expert_id": 0,
            "shard_id": "w1"
        },
    )]


def test_mxfp4_expert_weight_loader_rebinds_packed_checkpoint_name() -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=1,
        q_lora_rank=None,
        tie_word_embeddings=False,
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    moe = model.model.layers[0].block_sparse_moe = nn.Module()
    moe.experts = nn.Module()
    moe.experts.routed_experts = nn.Module()
    parameter = nn.Parameter(torch.empty(1))
    moe.experts.routed_experts.w13_weight = parameter

    calls = []

    def weight_loader(param, loaded_weight, weight_name, **kwargs):
        calls.append((param, loaded_weight, weight_name, kwargs))

    set_weight_attrs(parameter, {"weight_loader": weight_loader})
    loaded_weight = torch.ones(1, dtype=torch.uint8)
    loaded = model.load_weights([
        ("language_model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
         loaded_weight),
    ])

    parameter_name = (
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight")
    assert loaded == {parameter_name}
    assert calls == [(
        parameter,
        loaded_weight,
        parameter_name,
        {
            "expert_id": 0,
            "shard_id": "w1"
        },
    )]


@pytest.mark.parametrize(
    ("min_per_partition", "expected_intermediate", "expected_unpadded"),
    [(None, 1024, None), (256, 2048, 128)],
)
def test_small_experts_use_current_padded_weight_location(
        monkeypatch: pytest.MonkeyPatch, min_per_partition: int | None,
        expected_intermediate: int, expected_unpadded: int | None) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeMLP(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.routed_experts.w13_weight = nn.Parameter(torch.ones(1))
            self.routed_experts.w2_weight = nn.Parameter(torch.ones(1))
            self.moe_config = SimpleNamespace(
                intermediate_size_per_partition_unpadded=None)

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 8)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "KimiMLP", FakeMLP)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
    )
    if min_per_partition is not None:
        config.min_moe_intermediate_per_partition = min_per_partition
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["intermediate_size"] == expected_intermediate
    assert fused_moe_args["activation_situ_beta"] is None
    assert fused_moe_args["activation_situ_linear_beta"] is None
    assert (layer.experts.moe_config.intermediate_size_per_partition_unpadded
            == expected_unpadded)
    expect_zero = expected_unpadded is not None
    assert bool(layer.experts.routed_experts.w13_weight.any()) != expect_zero
    assert bool(layer.experts.routed_experts.w2_weight.any()) != expect_zero


def test_expert_parallelism_keeps_native_intermediate_size(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.moe_config = SimpleNamespace()

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True,
                                        data_parallel_size=1,
                                        prefill_context_parallel_size=1))
    monkeypatch.setattr(kimi_moe, "get_current_vllm_config_or_none",
                        lambda: vllm_config)
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 8)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=1024,
    )
    kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["intermediate_size"] == 1024


def test_situ_moe_uses_tpu_activation_descriptor(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.moe_config = SimpleNamespace()

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=128,
    )
    kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["activation"] == "situ"
    assert fused_moe_args["activation_situ_beta"] == 4.0
    assert fused_moe_args["activation_situ_linear_beta"] == 25.0


def test_latent_moe_does_not_reduce_before_norm() -> None:

    class SquareNorm(nn.Module):

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states.square()

    class IdentityLinear(nn.Module):

        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    transform = kimi_moe.KimiRoutedOutputTransform(SquareNorm(),
                                                   IdentityLinear())
    hidden_states = torch.tensor([[2.0, 3.0]])

    torch.testing.assert_close(transform(hidden_states),
                               hidden_states.square())


def test_model_has_no_public_cache_or_pp_api() -> None:
    parameters = inspect.signature(KimiLinearForCausalLM.forward).parameters
    assert "cache" not in parameters
    assert "use_cache" not in parameters
    assert not supports_pp(KimiLinearForCausalLM)
    assert not hasattr(kimi, "KimiK3Cache")
    assert not hasattr(kimi, "MLACache")


def test_model_is_registered_for_torch_compile() -> None:
    assert TorchCompileWithNoGuardsWrapper in KimiModel.__bases__


def test_kimi_linear_mla_keeps_its_checkpoint_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (sum(output_size) if isinstance(
                output_size, list) else output_size)
            self.weight = nn.Parameter(
                torch.empty(self.output_size, input_size))

        def forward(self, inputs):
            return inputs.new_zeros(inputs.shape[0], self.output_size), None

    class FakeMLAWrapper(nn.Module):

        def __init__(self, *args) -> None:
            super().__init__()
            self.mla_modules = args[8]

        def forward(self, positions, hidden_states):
            return hidden_states

    monkeypatch.setattr(kimi_attention, "MultiHeadLatentAttentionWrapper",
                        FakeMLAWrapper)
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MergedColumnParallelLinear",
                        FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    config = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
    )
    # This unit test does not have a model_config, so skip the platform-level
    # normalization that requires one. The layer still gets a complete
    # VllmConfig, including the compilation config used by RMSNorm.
    monkeypatch.setattr(current_platform, "check_and_update_config",
                        lambda _: None)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix="model.layers.0.self_attn",
        )

    assert layer.q_proj is not None
    assert layer.kv_a_proj_with_mqa is not None
    assert layer.fused_qkv_a_proj is None
    hidden_states = torch.randn(2, config.hidden_size)
    torch.testing.assert_close(
        layer(torch.arange(2), hidden_states),
        hidden_states,
    )


def test_mla_fuses_output_gate_with_lora_a_projections(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (sum(output_size) if isinstance(
                output_size, list) else output_size)
            self.weight = nn.Parameter(
                torch.empty(self.output_size, input_size))

    class FakeMLAWrapper(nn.Module):

        def __init__(self, *args) -> None:
            super().__init__()
            self.mla_modules = args[8]

    monkeypatch.setattr(kimi_attention, "MultiHeadLatentAttentionWrapper",
                        FakeMLAWrapper)
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MixedParallelMergedLinear",
                        FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    config = KimiLinearConfig(
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
    monkeypatch.setattr(current_platform, "check_and_update_config",
                        lambda _: None)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix="model.layers.0.self_attn",
        )

    assert layer.fused_qkv_a_proj.output_size == 8 + (8 + 2) + 4 * 4
    assert layer.g_proj is None
    assert layer.mla_attn.mla_modules.gate_is_fused is True


def test_mixed_parallel_merged_linear_loads_replicated_and_tp_slices(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 2)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_rank",
                        lambda: 1)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank",
                        lambda: 0)
    monkeypatch.setattr(parameter_module,
                        "get_tensor_model_parallel_world_size", lambda: 1)
    layer = kimi_attention.MixedParallelMergedLinear(
        2,
        [4, 3],
        [False, True],
        bias=False,
        quant_config=None,
        prefix="mixed",
    )
    column_weight = torch.arange(8, dtype=layer.weight.dtype).view(4, 2)
    replicated_weight = torch.arange(6, dtype=layer.weight.dtype).view(3,
                                                                       2) + 100

    layer.weight_loader(layer.weight, column_weight, 0)
    layer.weight_loader(layer.weight, replicated_weight, 1)

    expected = torch.cat((column_weight[2:], replicated_weight), dim=0)
    torch.testing.assert_close(layer.weight, expected)


class _Projection(nn.Module):

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        values = torch.arange(
            inputs.shape[0] * self.output_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return values.view(inputs.shape[0], self.output_size), None


class _IdentityProjection(nn.Module):

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return inputs, None


class _ConvWeight(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 1, 3))


def test_kda_forward_dispatches_to_both_custom_ops() -> None:
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.num_heads = 2
    layer.head_dim = 2
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.use_naive_kda = False

    sconv_cache = torch.zeros(1, 8, 3)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata

    calls: list[str] = []

    def dispatched_kda_op(mixed_qkv, *args):
        calls.append("dispatched")
        # It owns both caches: the convolution is fused into the decode kernel,
        # so there is no separate convolution op to hand the conv cache to.
        assert args[3] is sconv_cache
        assert args[4] is recurrent_cache
        return mixed_qkv[:, :4].view(-1, 2, 2)

    def unexpected(name):

        def op(*args, **kwargs):
            raise AssertionError(f"{name} must not run for this batch")

        return op

    layer.dispatched_kda_op = dispatched_kda_op
    # Neither of these is reachable without VLLM_TPU_USE_NAIVE_KDA, and a
    # separate `sconv_op` in front of the dispatched op would convolve twice.
    layer.sconv_op = unexpected("sconv_op")
    layer.chunk_kda_op = unexpected("chunk_kda_op")
    layer.kda_op = unexpected("kda_op")

    output = layer(torch.arange(2), torch.ones(2, 4))
    assert calls == ["dispatched"]
    torch.testing.assert_close(
        output,
        torch.tensor([[0, 1, 2, 3], [14, 15, 16, 17]], dtype=torch.float32),
    )


def test_kda_custom_ops_compile_as_one_full_graph(
        monkeypatch: pytest.MonkeyPatch) -> None:

    def jax_op(name, function, donate_argnums=()):
        del function, donate_argnums
        if "dispatched" in name:

            def implementation(
                mixed_qkv: torch.Tensor,
                raw_gate: torch.Tensor,
                beta: torch.Tensor,
                output_gate: torch.Tensor,
                conv_state: torch.Tensor,
                recurrent_state: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                a_log: torch.Tensor,
                dt_bias: torch.Tensor,
                norm_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
                distribution: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                del raw_gate, beta, output_gate, a_log, dt_bias, norm_weight
                del q_weight, k_weight, v_weight, distribution
                del query_start_loc, state_indices, seq_lens
                num_heads, head_dim = recurrent_state.shape[1:3]
                output = mixed_qkv[:, :num_heads * head_dim]
                return (output.view(-1, num_heads, head_dim).clone(),
                        conv_state + 1, recurrent_state + 1)

        elif "sconv" in name:

            def implementation(
                mixed_qkv: torch.Tensor,
                conv_state: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del q_weight, k_weight, v_weight
                del query_start_loc, state_indices, seq_lens
                return conv_state + 1, mixed_qkv.clone()

        else:

            def implementation(
                mixed_qkv: torch.Tensor,
                raw_gate: torch.Tensor,
                beta: torch.Tensor,
                output_gate: torch.Tensor,
                recurrent_state: torch.Tensor,
                a_log: torch.Tensor,
                dt_bias: torch.Tensor,
                norm_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del raw_gate, beta, output_gate, a_log, dt_bias, norm_weight
                del query_start_loc, state_indices, seq_lens
                num_heads, head_dim = recurrent_state.shape[1:3]
                output = mixed_qkv[:, :num_heads * head_dim]
                return (output.view(-1, num_heads,
                                    head_dim).clone(), recurrent_state + 1)

        return torch.library.custom_op(name, mutates_args=())(implementation)

    monkeypatch.setattr(kimi_custom_ops.pallas, "jax_op", jax_op)
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.num_heads = 2
    layer.head_dim = 2
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.use_naive_kda = False
    layer.sconv_op = kimi_custom_ops.build_kimi_sconv_op(
        "test_compile",
        kernel_size=3,
        state_dim_first=True,
    )
    layer.kda_op = kimi_custom_ops.build_kimi_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
    )
    layer.chunk_kda_op = kimi_custom_ops.build_kimi_chunk_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
    )
    layer.dispatched_kda_op = kimi_custom_ops.build_kimi_dispatched_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
        state_dim_first=True,
    )

    sconv_cache = torch.zeros(1, 12, 2)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata
    hidden_states = torch.ones(2, 4)
    positions = torch.arange(2)

    expected = layer(positions, hidden_states)
    sconv_cache.zero_()
    recurrent_cache.zero_()
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    actual = compiled(positions, hidden_states)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sconv_cache, torch.ones_like(sconv_cache))
    torch.testing.assert_close(recurrent_cache,
                               torch.ones_like(recurrent_cache))


def _latent_moe_config(**overrides) -> KimiLinearConfig:
    kwargs = dict(
        hidden_size=16,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=True,
    )
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


class _FakeMoEGate(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.output_size = kwargs.get("output_size", 1)

    def forward(self, hidden_states: torch.Tensor):
        return torch.zeros(hidden_states.shape[0], self.output_size), None


class _FakeSharedMLP(nn.Module):
    """Records what it was called with so tests can pin that KimiMoE.forward
    hands shared_experts the original layer input, not some other tensor."""

    def __init__(self, hidden_size, *args, **kwargs) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_inputs.append(hidden_states)
        return torch.full((hidden_states.shape[0], self.hidden_size), 3.0)


class _FakeMoERunner(nn.Module):
    """Records what it was called with; forward returns a value distinct
    from _FakeSharedMLP's so tests can pin combine() argument order."""

    def __init__(self) -> None:
        super().__init__()
        self.routed_experts = nn.Module()
        self.moe_config = SimpleNamespace(skip_final_all_reduce=True)
        self.forward_inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        del router_logits
        self.forward_inputs.append(hidden_states)
        return torch.full((hidden_states.shape[0], 8), 2.0)


class _FakeReplicatedLinear(nn.Module):
    """Stand-in for vLLM's ReplicatedLinear.

    Real ReplicatedLinear reaches into the live TP process group (via
    get_tensor_model_parallel_rank) even when TP is never actually used, which
    is not initialized in this CPU-only unit test. Constructing KimiMoE's
    latent path needs *a* module here, not the genuine parallel linear.
    """

    def __init__(self, input_size, output_size, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

    def forward(self, inputs: torch.Tensor):
        return inputs.new_zeros(inputs.shape[0], self.output_size), None


class _FakeRMSNorm(nn.Module):
    """Stand-in for vLLM's RMSNorm, matching its (hidden_size, eps) ctor and
    single-tensor return so the identity checks below still hold."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        del eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def _build_kimi_moe(
    monkeypatch: pytest.MonkeyPatch,
    config: KimiLinearConfig,
    tp_size: int = 8,
    configure_vllm_config: Callable[[VllmConfig], None] | None = None,
) -> tuple[nn.Module, dict]:
    fused_moe_args: dict = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return _FakeMoERunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: tp_size)
    monkeypatch.setattr(kimi_moe, "GateLinear", _FakeMoEGate)
    monkeypatch.setattr(kimi_moe, "KimiMLP", _FakeSharedMLP)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
    # Real ReplicatedLinear/RMSNorm construction reaches into the live TP
    # process group, which this CPU-only unit test never initializes (see
    # _FakeReplicatedLinear docstring). Fake both; the fused_combine identity
    # assertions below hold for fakes just as they do for the real classes.
    monkeypatch.setattr(kimi_moe, "ReplicatedLinear", _FakeReplicatedLinear)
    monkeypatch.setattr(kimi_moe, "RMSNorm", _FakeRMSNorm)
    monkeypatch.setattr(current_platform, "check_and_update_config",
                        lambda _: None)
    vllm_config = VllmConfig()
    if configure_vllm_config is not None:
        configure_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")
    return layer, fused_moe_args


def test_fused_combine_single_all_reduce_matches_two_reduce_math(
        monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    ranks, hidden, latent_w = 4, 6, 4
    shared_partials = [torch.randn(2, hidden) for _ in range(ranks)]
    latent_partials = [torch.randn(2, latent_w) for _ in range(ranks)]

    class LocalRmsNorm(nn.Module):

        def __init__(self, size: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.rand(size) + 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            return (x.float() * torch.rsqrt(variance + 1e-6) * self.weight).to(
                x.dtype)

    up_weight = torch.randn(hidden, latent_w)

    class UpProj(nn.Module):

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return x @ up_weight.t(), None

    calls: list[torch.Size] = []

    def fake_all_reduce(t: torch.Tensor) -> torch.Tensor:
        calls.append(t.shape)
        # Pin the concat order: shared columns first, latent columns last.
        # With hidden == latent_w this width check alone can't tell the two
        # apart, so an inverted torch.cat([latent, shared]) in
        # KimiFusedMoECombine.forward would slip through unnoticed without
        # this content check.
        assert torch.equal(
            t, torch.cat([shared_partials[0], latent_partials[0]], dim=-1))
        return sum(
            torch.cat([s, latent], dim=-1)
            for s, latent in zip(shared_partials, latent_partials))

    monkeypatch.setattr(kimi_moe, "tensor_model_parallel_all_reduce",
                        fake_all_reduce)
    norm = LocalRmsNorm(latent_w)
    combine = kimi_moe.KimiFusedMoECombine(norm, UpProj(), hidden, latent_w)

    actual = combine(shared_partials[0], latent_partials[0])

    # Ground truth = today's two-collective math: sum shared partials, sum
    # latent partials, then norm + up_proj on the summed latent.
    expected = (sum(shared_partials) + UpProj()(norm(sum(latent_partials)))[0])
    torch.testing.assert_close(actual, expected)
    assert calls == [torch.Size([2, hidden + latent_w])]


def test_fused_combine_construction_toggles_fused_moe_args(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TPU_K3_FUSED_MOE_COMBINE", "1")
    layer, fused_moe_args = _build_kimi_moe(monkeypatch, _latent_moe_config())

    assert layer.fused_combine is not None
    assert fused_moe_args["shared_experts"] is None
    assert fused_moe_args["reduce_results"] is False
    assert fused_moe_args["routed_output_transform"] is None
    # The combine reuses the exact module instances KimiMoE owns.
    assert layer.fused_combine.norm is layer.routed_expert_norm
    assert layer.fused_combine.up_proj is layer.routed_expert_up_proj


def test_fused_combine_off_by_default_keeps_two_reduce_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TPU_K3_FUSED_MOE_COMBINE", raising=False)
    layer, fused_moe_args = _build_kimi_moe(monkeypatch, _latent_moe_config())

    assert layer.fused_combine is None
    assert fused_moe_args["shared_experts"] is layer.shared_experts
    assert fused_moe_args["reduce_results"] is True
    assert isinstance(fused_moe_args["routed_output_transform"],
                      kimi_moe.KimiRoutedOutputTransform)


def test_sharded_latent_up_uses_row_parallel_and_custom_runner(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRowParallel(_FakeReplicatedLinear):

        def __init__(self, input_size, output_size, **kwargs) -> None:
            self.init_kwargs = kwargs.copy()
            super().__init__(input_size, output_size, **kwargs)

    monkeypatch.setenv("TPU_K3_SHARD_LATENT_UP_PROJ", "1")
    monkeypatch.setattr(kimi_moe, "RowParallelLinear", FakeRowParallel)
    monkeypatch.setattr(kimi_moe, "hierarchical_split_or_none",
                        lambda: (16, 0, 2, 0))
    layer, fused_moe_args = _build_kimi_moe(monkeypatch, _latent_moe_config())

    assert isinstance(layer.routed_expert_up_proj, FakeRowParallel)
    assert layer.routed_expert_up_proj.init_kwargs["input_is_parallel"] is False
    assert layer.routed_expert_up_proj.init_kwargs["reduce_results"] is False
    assert fused_moe_args["runner_cls"] is kimi_moe.KimiShardedUpMoERunner
    assert fused_moe_args["shared_experts"] is layer.shared_experts
    assert fused_moe_args["reduce_results"] is True
    assert fused_moe_args["routed_output_transform"].output_scale == 2 / 8


def test_sharded_latent_up_accepts_independent_tp_width(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRowParallel(_FakeReplicatedLinear):

        def __init__(self, input_size, output_size, **kwargs) -> None:
            self.init_kwargs = kwargs.copy()
            super().__init__(input_size, output_size, **kwargs)

    calls: list[tuple[int, int, int]] = []

    def fake_split(world_size: int, rank: int,
                   tp_size: int) -> tuple[int, int, int, int]:
        calls.append((world_size, rank, tp_size))
        return 2, 0, tp_size, 0

    monkeypatch.setenv("TPU_K3_SHARD_LATENT_UP_PROJ", "1")
    monkeypatch.setenv("TPU_K3_LATENT_UP_TP_SIZE", "4")
    monkeypatch.setattr(kimi_moe, "RowParallelLinear", FakeRowParallel)
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kimi_moe, "hierarchical_moe_split", fake_split)
    layer, fused_moe_args = _build_kimi_moe(monkeypatch,
                                             _latent_moe_config())

    assert calls == [(8, 0, 4)]
    assert isinstance(layer.routed_expert_up_proj, FakeRowParallel)
    assert fused_moe_args["routed_output_transform"].output_scale == 4 / 8


def test_sharded_up_runner_keeps_shared_partial_then_reduces_sum(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(kimi_moe.KimiShardedUpMoERunner)
    shared = torch.randn(2, 6)
    assert runner._maybe_reduce_shared_expert_output(shared, True) is shared

    calls: list[torch.Tensor] = []

    def fake_all_reduce(t: torch.Tensor) -> torch.Tensor:
        calls.append(t)
        return t + 7

    monkeypatch.setattr(kimi_moe, "tensor_model_parallel_all_reduce",
                        fake_all_reduce)
    partial_sum = torch.randn(2, 8)
    actual = runner._maybe_reduce_final_output(partial_sum, 6, True)
    torch.testing.assert_close(actual, (partial_sum + 7)[..., :6])
    assert calls == [partial_sum]


def test_sharded_latent_down_uses_hierarchical_row_parallel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRowParallel(_FakeReplicatedLinear):

        def __init__(self, input_size, output_size, **kwargs) -> None:
            self.init_kwargs = kwargs.copy()
            super().__init__(input_size, output_size, **kwargs)

    class FakeGroup:
        world_size = 2
        rank_in_group = 0

        def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    group = FakeGroup()
    monkeypatch.setenv("TPU_K3_SHARD_LATENT_DOWN_PROJ", "1")
    monkeypatch.setenv("TPU_MOE_HIERARCHICAL_EP", "1")
    monkeypatch.setattr(kimi_moe, "RowParallelLinear", FakeRowParallel)
    monkeypatch.setattr(kimi_moe, "hierarchical_split_or_none",
                        lambda: (16, 0, 2, 0))
    monkeypatch.setattr(kimi_moe, "_get_k3_hierarchical_tp_group",
                        lambda: group)
    monkeypatch.setattr(kimi_moe, "hierarchical_moe_parallel_config",
                        nullcontext)
    def enable_ep(vllm_config: VllmConfig) -> None:
        vllm_config.parallel_config.enable_expert_parallel = True

    layer, fused_moe_args = _build_kimi_moe(
        monkeypatch, _latent_moe_config(), configure_vllm_config=enable_ep)

    assert isinstance(layer.routed_expert_down_proj, FakeRowParallel)
    assert layer.routed_expert_down_proj.init_kwargs["input_is_parallel"] is False
    assert layer.routed_expert_down_proj.init_kwargs["reduce_results"] is False
    transform = fused_moe_args["routed_input_transform"]
    assert isinstance(transform, kimi_moe.KimiRoutedInputTransform)
    assert transform.down_proj is layer.routed_expert_down_proj
    assert transform.tp_group is group


def test_sharded_latent_down_transform_reduces_projection() -> None:
    class DownProj(nn.Module):

        def forward(self, tensor: torch.Tensor):
            return tensor + 3, None

    class Group:

        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
            self.inputs.append(tensor)
            return tensor * 2

    group = Group()
    transform = kimi_moe.KimiRoutedInputTransform(DownProj(), group)
    inputs = torch.randn(2, 4)
    actual = transform(inputs)
    torch.testing.assert_close(actual, (inputs + 3) * 2)
    assert len(group.inputs) == 1
    torch.testing.assert_close(group.inputs[0], inputs + 3)


def test_sharded_router_gate_uses_independent_tp_group(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeShardedGate(_FakeReplicatedLinear):

        def __init__(self, input_size, output_size, **kwargs) -> None:
            self.init_kwargs = kwargs.copy()
            super().__init__(input_size, output_size, **kwargs)

    class FakeGroup:
        pass

    group = FakeGroup()
    split_calls: list[tuple[int, int, int]] = []

    def fake_split(world_size: int, rank: int,
                   tp_size: int) -> tuple[int, int, int, int]:
        split_calls.append((world_size, rank, tp_size))
        return 2, 0, tp_size, 1

    monkeypatch.setenv("TPU_K3_SHARD_ROUTER_GATE", "1")
    monkeypatch.setenv("TPU_K3_ROUTER_GATE_TP_SIZE", "4")
    monkeypatch.setattr(kimi_moe, "KimiShardedRouterGate", FakeShardedGate)
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(kimi_moe, "hierarchical_moe_split", fake_split)
    monkeypatch.setattr(kimi_moe, "_get_k3_projection_tp_group",
                        lambda tp_size: group)

    layer, _ = _build_kimi_moe(monkeypatch, _latent_moe_config())

    assert split_calls == [(8, 5, 4)]
    assert isinstance(layer.gate, FakeShardedGate)
    assert layer.gate.init_kwargs["tp_group"] is group
    assert layer.gate.init_kwargs["out_dtype"] is torch.float32
    assert layer.gate.weight.shape == (4, 16)
    assert layer.gate.e_score_correction_bias.shape == (4, )


def test_sharded_router_gate_reduces_then_casts_logits(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGroup:

        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
            self.inputs.append(tensor)
            return tensor * 2

    def fake_row_parallel_forward(self, hidden_states: torch.Tensor):
        del self
        return hidden_states + 1, None

    monkeypatch.setattr(kimi_moe.RowParallelLinear, "forward",
                        fake_row_parallel_forward)
    gate = object.__new__(kimi_moe.KimiShardedRouterGate)
    nn.Module.__init__(gate)
    gate.tp_group = FakeGroup()
    gate.out_dtype = torch.float32
    inputs = torch.arange(4, dtype=torch.bfloat16).view(1, 4)

    actual, bias = gate(inputs)

    assert bias is None
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, ((inputs + 1) * 2).float())
    assert len(gate.tp_group.inputs) == 1
    torch.testing.assert_close(gate.tp_group.inputs[0], inputs + 1)


def _set_pcp2(vllm_config: VllmConfig) -> None:
    vllm_config.parallel_config.prefill_context_parallel_size = 2


@pytest.mark.parametrize(
    ("config_overrides", "build_kwargs"),
    [
        ({
            "routed_scaling_factor": 2.0
        }, {}),
        ({}, {
            "tp_size": 1
        }),
        ({
            "num_shared_experts": 0
        }, {}),
        ({}, {
            "configure_vllm_config": _set_pcp2
        }),
    ],
    ids=["routed-scaling", "tp1", "no-shared-experts", "pcp"],
)
def test_fused_combine_disabled_when_preconditions_fail(
        monkeypatch: pytest.MonkeyPatch, config_overrides: dict,
        build_kwargs: dict) -> None:
    monkeypatch.setenv("TPU_K3_FUSED_MOE_COMBINE", "1")
    layer, fused_moe_args = _build_kimi_moe(
        monkeypatch, _latent_moe_config(**config_overrides), **build_kwargs)

    assert layer.fused_combine is None
    assert fused_moe_args["reduce_results"] is True


def test_fused_combine_forward_concatenates_shared_then_latent(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Drives KimiMoE.forward() through the fused_combine branch end to end,
    pinning: (a) concat order at the forward level, (b) shared_experts
    receiving the original layer input, and (c) fused_combine being invoked
    as combine(shared, latent) rather than the reverse."""
    monkeypatch.setenv("TPU_K3_FUSED_MOE_COMBINE", "1")
    layer, _ = _build_kimi_moe(monkeypatch, _latent_moe_config())
    assert layer.fused_combine is not None

    hidden_size, expert_hidden_size = 16, 8
    shared_value, latent_value = 3.0, 2.0  # matches the fakes' forward()

    all_reduce_inputs: list[torch.Tensor] = []

    def fake_all_reduce(t: torch.Tensor) -> torch.Tensor:
        all_reduce_inputs.append(t)
        # (a) Concat order: shared columns first, latent columns last. An
        # inverted torch.cat([latent, shared]) in
        # KimiFusedMoECombine.forward, or a swapped combine(latent, shared)
        # call in KimiMoE.forward, would put the wrong value in one of these
        # column ranges.
        assert t.shape == (2, hidden_size + expert_hidden_size)
        assert torch.equal(t[:, :hidden_size],
                           torch.full((2, hidden_size), shared_value))
        assert torch.equal(t[:, hidden_size:],
                           torch.full((2, expert_hidden_size), latent_value))
        return t

    monkeypatch.setattr(kimi_moe, "tensor_model_parallel_all_reduce",
                        fake_all_reduce)

    hidden_states = torch.randn(2, hidden_size)
    output = layer(hidden_states)

    # (b) shared_experts and the routed runner both see the original layer
    # input, not each other's output or some other intermediate.
    assert len(layer.shared_experts.forward_inputs) == 1
    assert torch.equal(layer.shared_experts.forward_inputs[0], hidden_states)
    assert len(layer.experts.forward_inputs) == 1
    assert torch.equal(layer.experts.forward_inputs[0], hidden_states)
    assert len(all_reduce_inputs) == 1

    # fused_combine.norm is the identity _FakeRMSNorm and up_proj
    # (_FakeReplicatedLinear) returns zeros, so shared_sum + routed reduces
    # to exactly the shared partial -- confirming the combine's output, not
    # just its input, is wired end to end.
    torch.testing.assert_close(output,
                               torch.full((2, hidden_size), shared_value))
