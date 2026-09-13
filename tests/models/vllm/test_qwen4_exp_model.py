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
"""Unit and POC tests for Qwen3.8-Flash-Next (Qwen4Exp) on TPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models import ModelRegistry

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.layers.vllm.custom_ops.qwen4_exp_ops import (
    _load_upstream_config_module,
    register_qwen4_exp_ops,
)

register_qwen4_exp_ops()
_cfg_mod = _load_upstream_config_module()
Qwen4ExpConfig = _cfg_mod.Qwen4ExpConfig
Qwen4ExpTextConfig = _cfg_mod.Qwen4ExpTextConfig
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
from vllm.models.qwen4_exp.nvidia.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)


@pytest.fixture(autouse=True)
def _mock_tp_group(monkeypatch):
    import vllm.distributed.parallel_state as ps

    fake_tp = SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)
    monkeypatch.setattr(ps, "_TP", fake_tp)
    monkeypatch.setattr(ps, "get_tp_group", lambda: fake_tp)
    monkeypatch.setattr(ps, "_DP", fake_tp)
    monkeypatch.setattr(ps, "get_dp_group", lambda: fake_tp)
    monkeypatch.setattr(ps, "_PP", fake_tp)
    monkeypatch.setattr(ps, "get_pp_group", lambda: fake_tp)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)


def test_qwen4_exp_model_registration():
    """Verify Qwen4Exp models resolve from the vLLM ModelRegistry."""
    register_models()
    for name, expected_cls in [
        ("Qwen4ExpForCausalLM", Qwen4ExpForCausalLM),
        ("Qwen4ExpForConditionalGeneration", Qwen4ExpForConditionalGeneration),
    ]:
        entry = ModelRegistry.models.get(name)
        assert entry is not None, f"{name} is not registered"
        model_cls = entry.load_model_cls()
        assert model_cls is expected_cls


def test_qwen4_exp_config_initialization():
    """Verify Qwen4ExpTextConfig and Qwen4ExpConfig validate Qwen3.8-Flash-Next params."""
    text_cfg = Qwen4ExpTextConfig(
        hidden_size=256,
        num_hidden_layers=4,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1024,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    assert text_cfg.hc_count == 4
    assert text_cfg.short_conv_layer_ids == [1]
    assert text_cfg.short_conv_state_shape == (256 * 4, 3 * 3)
    assert text_cfg.ngram_context_len == 2

    cfg = Qwen4ExpConfig(text_config=text_cfg)
    assert cfg.model_type == "qwen4_exp"
    assert cfg.text_config.hc_count == 4


def test_qwen4_exp_hyperconnection_forward():
    """Verify GatedResidual mix, combine, and combine_and_mix on multi-stream hidden states."""
    hc_cfg = HyperConnectionConfig(
        hc_count=4,
        hidden_size=64,
        params_dtype=torch.bfloat16,
        hc_lowrank=16,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    hc = GatedResidual(hc_cfg, use_combine=True)
    num_tokens = 5
    hyper_hidden = torch.randn(
        num_tokens,
        hc_cfg.hc_count * hc_cfg.hidden_size,
        dtype=torch.bfloat16,
    )

    hidden_states, block_input, injection = hc.mix(hyper_hidden)
    assert block_input.shape == (num_tokens, hc_cfg.hidden_size)
    assert injection is not None and injection.shape == (
        num_tokens,
        hc_cfg.hc_count,
    )

    block_out = torch.randn(
        num_tokens,
        hc_cfg.hidden_size,
        dtype=torch.bfloat16,
    )
    combined, next_input, next_injection = hc.combine_and_mix(
        hidden_states, block_out, injection
    )
    assert combined.shape == (num_tokens, hc_cfg.hc_count * hc_cfg.hidden_size)
    assert next_input.shape == (num_tokens, hc_cfg.hidden_size)
    assert next_injection is not None and next_injection.shape == (
        num_tokens,
        hc_cfg.hc_count,
    )


def test_qwen4_exp_ple_layer_forward():
    """Verify Qwen4ExpPLELayer N-gram lookup, dilated causal conv, and gated injection."""
    text_cfg = Qwen4ExpTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=512,
        eos_token_id=151645,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    from vllm.config import set_current_vllm_config
    from vllm.config.compilation import CompilationConfig

    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=text_cfg),
        cache_config=None,
        quant_config=None,
        engram_config=None,
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=16),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        num_speculative_tokens=0,
    )
    with set_current_vllm_config(vllm_cfg):
        ple = Qwen4ExpPLELayer(
            text_cfg,
            vllm_config=vllm_cfg,
            layer_idx=1,
            prefix="model.layers.1.ple",
        ).to(torch.bfloat16)
        for param in ple.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.02)

    num_tokens = 6
    hidden_states = torch.randn(
        num_tokens,
        text_cfg.hc_count * text_cfg.hidden_size,
        dtype=torch.bfloat16,
    )
    input_ids = torch.randint(
        0,
        text_cfg.ngram_vocab_size_base,
        (num_tokens,),
        dtype=torch.long,
    )
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    ngram_context = torch.zeros(
        1,
        text_cfg.ngram_context_len,
        dtype=torch.int64,
    )

    ple_out = ple(
        hidden_states=hidden_states,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
    )
    assert ple_out.shape == hidden_states.shape
    assert ple_out.dtype == torch.bfloat16
