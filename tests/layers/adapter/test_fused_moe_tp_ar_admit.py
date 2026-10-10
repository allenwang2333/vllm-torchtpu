# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Which MoE layers USE_MOE_FUSED_TP_AR admits, and with which shared-expert
and routed-scale convention: Qwen3.5's gated shared expert with no routed
scale, and GLM-5.2's ungated DeepseekV2MLP shared expert with
routed_scaling_factor 2.5 (applied by vLLM's runner to the routed output only).
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_torchtpu.layers.adapter import fused_moe_tp_ar as ftp
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

N, H, INTER, INTER_S, E = 16, 6144, 2048, 128, 16
F8 = torch.float8_e4m3fn


def _linear(k, n):
    return SimpleNamespace(
        weight=torch.zeros((k, n), dtype=F8), weight_scale=torch.ones(n)
    )


class DeepseekV2MLP(torch.nn.Module):
    """Same class name and layout as vLLM's (the admission keys on both)."""

    def __init__(self, act=None):
        super().__init__()
        self.gate_up_proj = _linear(H, 2 * INTER_S)
        self.down_proj = _linear(INTER_S, H)
        self.act_fn = act if act is not None else SiluAndMul.__new__(SiluAndMul)


class PlainMLP(torch.nn.Module):
    """Ungated, but not a class the admission can vouch for."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = _linear(H, 2 * INTER_S)
        self.down_proj = _linear(INTER_S, H)


class GatedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = _linear(H, 2 * INTER_S)
        self.down_proj = _linear(INTER_S, H)
        self.expert_gate = SimpleNamespace(
            weight=torch.zeros((1, H), dtype=torch.bfloat16), bias=None
        )


def _runner(mlp, *, scale=1.0, dtype=torch.bfloat16, shared_expert_gate=None):
    routed = SimpleNamespace(
        w13_weight=torch.zeros((E, H, 2 * INTER), dtype=F8),
        w2_weight=torch.zeros((E, INTER, H), dtype=F8),
        w13_weight_scale_inv=torch.ones((E, 1, 1, 2 * INTER)),
        w2_weight_scale_inv=torch.ones((E, 1, 1, H)),
    )
    par = SimpleNamespace(use_ep=True, ep_size=N, dp_size=1, tp_size=N)
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=par,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            in_dtype=dtype,
        ),
        _quant_method=VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU),
        routed_experts=routed,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=scale,
        shared_expert_gate=shared_expert_gate,
        _shared_experts=SimpleNamespace(_layer=mlp),
    )


def test_glm_ungated_shared_expert_with_routed_scale_is_admitted():
    admitted = ftp._admit(_runner(DeepseekV2MLP(), scale=2.5), N)
    assert admitted is not None
    shared, expert_parallel, routed_scale = admitted
    assert shared[4] is None  # no gate: coefficient 1
    assert expert_parallel and routed_scale == 2.5


def test_qwen_gated_shared_expert_is_unchanged():
    admitted = ftp._admit(_runner(GatedMLP()), N)
    assert admitted is not None
    shared, _, routed_scale = admitted
    assert shared[4] is not None and routed_scale == 1.0


@pytest.mark.parametrize(
    "runner",
    [
        # An ungated MLP the admission cannot vouch for (not DeepseekV2MLP).
        _runner(PlainMLP()),
        # DeepseekV2MLP with a different activation.
        _runner(DeepseekV2MLP(act=torch.nn.GELU())),
        # FP16: vLLM divides the shared output instead; not mirrored.
        _runner(DeepseekV2MLP(), scale=2.5, dtype=torch.float16),
        # A runner-level shared expert gate is applied outside the MLP.
        _runner(DeepseekV2MLP(), shared_expert_gate=object()),
    ],
)
def test_declines_what_it_does_not_mirror(runner):
    assert ftp._admit(runner, N) is None
