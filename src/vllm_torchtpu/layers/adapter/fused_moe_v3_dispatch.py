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
"""Non-invasive dispatch adapter for Pallas MoE v3 and few-token kernels.

Provides zero-drift layer forward routing that preserves the upstream standard
1615-tensor graph signature while leveraging hardware-accelerated Mosaic/Pallas
kernels when requested via environment configuration.
"""

from typing import Any

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def prebuild_fused_moe_v3(
    layer: Any,
    topk: int,
    renormalize: bool,
    activation: str,
) -> None:
    """Prebuild the Pallas MoE v3 and low-token kernel ops at weight load time."""
    if envs.USE_MOE_FUSED_EP_KERNEL:
        fused_moe_ep_v3.prebuild_fused_moe_ep(
            layer,
            topk=topk,
            renormalize=renormalize,
            activation=activation,
        )


def fused_moe_v3_supported(layer_or_method: Any) -> bool:
    """Check if the fused MoE v3 kernel is enabled and armed for this layer."""
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return False
    return fused_moe_ep_v3.fused_moe_ep_supported(layer_or_method)


def dispatch_fused_moe_v3(
    layer: Any,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Dispatch forward execution to the fused MoE v3 Pallas kernel."""
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    s13 = layer.w13_weight_scale_inv
    s2 = layer.w2_weight_scale_inv

    return fused_moe_ep_v3.fused_moe_ep(
        hidden_states=x,
        w1=w13,
        w2=w2,
        w1_scale=s13,
        w2_scale=s2,
        router_logits=router_logits,
    )
