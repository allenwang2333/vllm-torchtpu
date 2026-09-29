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
"""The FP8 MoE method's dispatch to the fused EP kernel v3.

`VllmFp8MoEMethodTPU` calls into this module at three seams, as it does for
the adaptive kernel: `prebuild` at weight load, `supported` in
`supports_internal_mk`, and `apply` in `apply_monolithic`. A layer v3 declines
keeps the method's grouped-matmul path and vLLM's own dispatch and combine.
"""

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.fused_moe_ep_v3 import (
    FUSED_MOE_EP_V3_OP_ATTR,
    fused_moe_ep_v3,
    fused_moe_ep_v3_supported,
    prebuild_fused_moe_ep_v3,
)


def enabled() -> bool:
    """Whether the operator selected v3 (MOE_FUSED_EP_KERNEL_IMPL=v3)."""
    return envs.MOE_FUSED_EP_KERNEL_IMPL == "v3"


def prebuild(method, layer, activation: str) -> None:
    """Build the v3 op for `layer` and record it on its quant method."""
    op = prebuild_fused_moe_ep_v3(
        layer,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        activation=activation,
    )
    setattr(method, FUSED_MOE_EP_V3_OP_ATTR, op)


def supported(method) -> bool:
    """Whether v3 serves this method's layer, and so owns its exchange."""
    return fused_moe_ep_v3_supported(method)


def apply(method, layer, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """The layer's forward through v3: this rank's tokens in and out."""
    return fused_moe_ep_v3(
        method,
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        router_logits,
    )
