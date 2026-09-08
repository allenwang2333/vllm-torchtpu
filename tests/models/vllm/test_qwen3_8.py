# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic bring-up and verification tests for Qwen 3.8 Dense & MoE models."""

import contextlib
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.qwen3_8 import (
    Qwen3_8DecoderLayer,
    Qwen3_8ForCausalLM,
    Qwen3_8Model,
    Qwen3_8MoeForCausalLM,
)
from vllm.transformers_utils.configs.qwen3_8 import Qwen3_8TextConfig
from vllm.transformers_utils.configs.qwen3_8_moe import Qwen3_8MoeTextConfig

from vllm_torchtpu.gdn_pool_layout import (
    POOLED_GDN_ARCHITECTURES,
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True, scope="module")
def init_dist():
    with contextlib.suppress(Exception):
        torch.utils.rename_privateuse1_backend("tpu")
    if not hasattr(torch, "tpu"):
        import types

        tpu_mod = types.ModuleType("torch.tpu")
        tpu_mod.current_device = lambda: None
        torch.tpu = tpu_mod
    elif not hasattr(torch.tpu, "current_device"):
        torch.tpu.current_device = lambda: None
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        if not torch.distributed.is_initialized():
            port = _get_open_port()
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0,
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1)


def _create_mock_vllm_config(
    hf_config,
    max_model_len: int = 256,
    mamba_cache_mode: str = "align",
):
    text_config = getattr(hf_config, "text_config", hf_config)
    arch = (
        "Qwen3_8MoeForCausalLM"
        if "moe" in getattr(text_config, "model_type", "")
        else "Qwen3_8ForCausalLM"
    )
    model_config = SimpleNamespace(
        model="dummy-qwen3-8",
        runner_type="generate",
        trust_remote_code=True,
        dtype=torch.float32,
        seed=42,
        hf_config=hf_config,
        hf_text_config=text_config,
        multimodal_config=None,
        max_model_len=max_model_len,
        architecture=arch,
        get_head_size=lambda: getattr(text_config, "head_dim", 256),
        get_num_kv_heads=lambda parallel_config: getattr(
            text_config, "num_key_value_heads", 4
        ),
        get_sliding_window=lambda: getattr(text_config, "sliding_window", None),
        is_multimodal_model=False,
        is_mm_prefix_lm=False,
        is_hybrid=True,
        disable_sliding_window=False,
        use_mla=False,
        get_num_layers=lambda parallel_config: getattr(
            text_config, "num_hidden_layers", 4
        ),
        get_hidden_size=lambda: getattr(text_config, "hidden_size", 128),
        get_total_num_kv_heads=lambda: getattr(text_config, "num_key_value_heads", 4),
        head_dtype=None,
        rswa_window=None,
    )
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        CompilationConfig,
        ParallelConfig,
        SchedulerConfig,
    )

    cache_config = CacheConfig(
        block_size=16,
        mamba_cache_mode=mamba_cache_mode,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    parallel_config.prefill_context_parallel_size = 1
    attention_config = AttentionConfig()
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )
    from vllm.config import CompilationMode, KernelConfig, ObservabilityConfig

    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["all"]
    )
    observability_config = ObservabilityConfig()
    device_config = SimpleNamespace(device=torch.device("cpu"))
    kernel_config = KernelConfig()
    lora_config = None
    quant_config = None
    speculative_config = None
    kv_transfer_config = None
    load_config = None
    decoding_config = None
    prompt_adapter_config = None
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        attention_config=attention_config,
        device_config=device_config,
        kernel_config=kernel_config,
        lora_config=lora_config,
        quant_config=quant_config,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        kv_transfer_config=kv_transfer_config,
        observability_config=observability_config,
        load_config=load_config,
        decoding_config=decoding_config,
        prompt_adapter_config=prompt_adapter_config,
        additional_config={},
    )
    # Attention layers need the layout the engine core resolves at startup.
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    resolve_kv_cache_layout(vllm_config, [["LBNHC", "LBHNC"]])
    return vllm_config


def test_qwen3_8_dense_config_defaults():
    """Verify default configurations for Qwen 3.8 Dense."""
    config = Qwen3_8TextConfig(
        num_hidden_layers=8,
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
    )
    assert config.model_type == "qwen3_8_text"
    assert config.partial_rotary_factor == 0.25
    assert len(config.layer_types) == 8
    # 3:1 schedule: 3 linear, 1 full, 3 linear, 1 full
    expected_schedule = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert config.layer_types == expected_schedule


def test_qwen3_8_moe_config_defaults():
    """Verify default configurations for Qwen 3.8 MoE."""
    config = Qwen3_8MoeTextConfig(
        num_hidden_layers=4,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=256,
    )
    assert config.model_type == "qwen3_8_moe_text"
    assert config.num_experts == 16
    assert config.num_experts_per_tok == 4
    assert config.shared_expert_intermediate_size == 256
    assert len(config.layer_types) == 4
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_qwen3_8_layer_types_fallback():
    """Verify layer selection fallback when layer_types is None in config."""
    config = Qwen3_8TextConfig(
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=None,
    )
    vllm_config = _create_mock_vllm_config(config)

    with (
        set_vllm_model_wrapper_context(mesh=MagicMock(), vllm_config=vllm_config),
        set_current_vllm_config(vllm_config),
    ):
        # Layer 0 (index 0): (0+1)%4 != 0 -> linear_attention
        layer0 = Qwen3_8DecoderLayer(vllm_config, prefix="model.layers.0")
        assert layer0.layer_type == "linear_attention"
        assert hasattr(layer0, "linear_attn")

        # Layer 3 (index 3): (3+1)%4 == 0 -> full_attention
        layer3 = Qwen3_8DecoderLayer(vllm_config, prefix="model.layers.3")
        assert layer3.layer_type == "full_attention"
        assert hasattr(layer3, "self_attn")


def test_qwen3_8_model_registry():
    """Verify ModelRegistry and MODELS_CONFIG_MAP registration."""
    assert "Qwen3_8ForCausalLM" in ModelRegistry.get_supported_archs()
    assert "Qwen3_8MoeForCausalLM" in ModelRegistry.get_supported_archs()

    dense_reg = ModelRegistry.models["Qwen3_8ForCausalLM"]
    moe_reg = ModelRegistry.models["Qwen3_8MoeForCausalLM"]
    assert dense_reg.load_model_cls() is Qwen3_8ForCausalLM
    assert moe_reg.load_model_cls() is Qwen3_8MoeForCausalLM

    assert "Qwen3_8ForCausalLM" in MODELS_CONFIG_MAP
    assert "Qwen3_8MoeForCausalLM" in MODELS_CONFIG_MAP


def test_qwen3_8_tpu_unified_pool_registration():
    """Verify Qwen3_8 architectures are registered for TPU unified KV pool."""
    assert "Qwen3_8ForCausalLM" in POOLED_GDN_ARCHITECTURES
    assert "Qwen3_8MoeForCausalLM" in POOLED_GDN_ARCHITECTURES
    assert unified_kv_layout_enabled_for_architecture("Qwen3_8ForCausalLM")
    assert unified_kv_layout_enabled_for_architecture("Qwen3_8MoeForCausalLM")


def test_qwen3_8_weights_mapping():
    """Verify WeightsMapper maps HuggingFace weights to vLLM fused format."""
    mapper = Qwen3_8Model.hf_to_vllm_mapper
    assert mapper is not None

    orig_keys = [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    mapped_keys = mapper.apply_list(orig_keys)
    assert "model.layers.0.linear_attn.in_proj_qkvz.weight" in mapped_keys
    assert "model.layers.0.linear_attn.in_proj_ba.weight" in mapped_keys
    assert "model.layers.3.self_attn.qkv_proj.weight" in mapped_keys
    assert "model.layers.0.mlp.gate_up_proj.weight" in mapped_keys


def test_qwen3_8_mamba_state_specs():
    """Verify Mamba state shape and dtype calculators for IsHybrid interface."""
    config = Qwen3_8TextConfig(
        num_hidden_layers=4,
        hidden_size=512,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
    )
    vllm_config = _create_mock_vllm_config(config)

    shapes = Qwen3_8ForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    dtypes = Qwen3_8ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    assert len(shapes) == 2
    assert len(dtypes) == 2
    # Conv state shape: (kernel_size - 1, conv_dim)
    conv_dim = (
        config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    assert shapes[0] == (config.linear_conv_kernel_dim - 1, conv_dim)
    # Temporal SSM state shape: (num_v_heads, head_k_dim, head_v_dim)
    assert shapes[1] == (
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )


def test_qwen3_8_dense_synthetic_forward():
    """Execute synthetic 4-layer Qwen3.8 Dense model forward pass on CPU."""
    torch.manual_seed(42)
    config = Qwen3_8TextConfig(
        vocab_size=1000,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    vllm_config = _create_mock_vllm_config(config)

    with (
        set_vllm_model_wrapper_context(mesh=MagicMock(), vllm_config=vllm_config),
        set_current_vllm_config(vllm_config),
    ):
        device = torch.device("cpu")
        model = Qwen3_8ForCausalLM(vllm_config=vllm_config).to(device)

        # Standard vLLM dummy weight initialization (Host CPU safe copy)
        generator = torch.Generator(device="cpu").manual_seed(42)
        for p in model.parameters():
            if p.numel() > 0 and p.is_floating_point():
                val = torch.empty(p.shape, dtype=p.dtype, device="cpu").uniform_(
                    -1e-3, 1e-3, generator=generator
                )
                p.data.copy_(val)
        model.eval()

        num_tokens = 16
        input_ids = torch.randint(0, config.vocab_size, (num_tokens,), device=device)
        positions = torch.arange(num_tokens, device=device)

        with (
            set_forward_context(None, vllm_config=vllm_config, num_tokens=num_tokens),
            torch.no_grad(),
        ):
            hidden_states = model(input_ids, positions)
            logits = model.compute_logits(hidden_states)

        assert hidden_states.shape == (num_tokens, config.hidden_size)
        assert logits.shape == (num_tokens, config.vocab_size)
        assert not torch.isnan(logits).any(), "NaN detected in Dense logits"
        assert not torch.isinf(logits).any(), "Inf detected in Dense logits"


def test_qwen3_8_moe_synthetic_forward(monkeypatch):
    """Execute synthetic 4-layer Qwen3.8 MoE model forward pass on CPU."""
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    def _cpu_moe_apply_monolithic(self, layer, x, router_logits, input_ids=None):
        probs = torch.softmax(router_logits.float(), dim=-1).to(x.dtype)
        topk_weights, topk_ids = torch.topk(probs, k=layer.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)
        num_tokens, _ = x.shape
        for t in range(num_tokens):
            for k in range(layer.top_k):
                exp_id = topk_ids[t, k].item()
                weight = topk_weights[t, k]
                gate_up = torch.nn.functional.linear(
                    x[t : t + 1], layer.w13_weight[exp_id]
                )
                gate, up = gate_up.chunk(2, dim=-1)
                act = torch.nn.functional.silu(gate) * up
                down = torch.nn.functional.linear(act, layer.w2_weight[exp_id])
                output[t] += weight * down.squeeze(0)
        return output

    monkeypatch.setattr(
        UnquantizedFusedMoEMethod, "is_monolithic", property(lambda self: True)
    )
    monkeypatch.setattr(
        UnquantizedFusedMoEMethod, "apply_monolithic", _cpu_moe_apply_monolithic
    )

    torch.manual_seed(42)
    config = Qwen3_8MoeTextConfig(
        vocab_size=1000,
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
    )
    vllm_config = _create_mock_vllm_config(config)

    with (
        set_vllm_model_wrapper_context(mesh=MagicMock(), vllm_config=vllm_config),
        set_current_vllm_config(vllm_config),
    ):
        device = torch.device("cpu")
        model = Qwen3_8MoeForCausalLM(vllm_config=vllm_config).to(device)

        # Standard vLLM dummy weight initialization (Host CPU safe copy)
        generator = torch.Generator(device="cpu").manual_seed(42)
        for p in model.parameters():
            if p.numel() > 0 and p.is_floating_point():
                val = torch.empty(p.shape, dtype=p.dtype, device="cpu").uniform_(
                    -1e-3, 1e-3, generator=generator
                )
                p.data.copy_(val)
        model.eval()

        num_tokens = 16
        input_ids = torch.randint(0, config.vocab_size, (num_tokens,), device=device)
        positions = torch.arange(num_tokens, device=device)

        with (
            set_forward_context(None, vllm_config=vllm_config, num_tokens=num_tokens),
            torch.no_grad(),
        ):
            hidden_states = model(input_ids, positions)
            logits = model.compute_logits(hidden_states)

        assert hidden_states.shape == (num_tokens, config.hidden_size)
        assert logits.shape == (num_tokens, config.vocab_size)
        assert not torch.isnan(logits).any(), "NaN detected in MoE logits"
        assert not torch.isinf(logits).any(), "Inf detected in MoE logits"
