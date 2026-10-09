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
"""DeepSeek-V4 feed-forward and Manifold-Harmonized Coupling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    quantized_matmul_prequant,
)


# Pure PyTorch head projection for MHC manifold collapse;
# upstream vLLM only provides Triton/TileLang GPU kernels for hc_head.
def mhc_collapse_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse multi-stream residual manifold into a single vector before LM head."""
    residual_flat = hidden_states.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(), dim=-2).to(
        hidden_states.dtype
    )


class DeepseekV4MLP(nn.Module):
    """DeepSeek-V4 feed-forward layer for dense/shared expert projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._prequant: list[torch.Tensor | None] = [None, None]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def can_accept_prequant_hidden(self, *, token_sharded: bool = False) -> bool:
        if not envs.VLLM_DSV41_FUSED_SEAM_QUANT:
            return False
        w_s = self.gate_up_proj._parameters.get("weight_scale", None)
        ok = (
            w_s is not None
            and w_s.ndim == 1
            and self.gate_up_proj._parameters.get("bias", None) is None
        )
        return ok and not token_sharded

    def set_prequant_hidden(
        self,
        hidden_states_fp8: torch.Tensor | None,
        hidden_states_scale: torch.Tensor | None,
    ) -> None:
        self._prequant[0] = hidden_states_fp8
        self._prequant[1] = hidden_states_scale

    def _apply_gate_up_proj(
        self,
        x: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None,
        hidden_states_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states_fp8 is not None and hidden_states_scale is not None:
            weight = self.gate_up_proj.weight
            if getattr(self.gate_up_proj, KEEP_VLLM_LAYOUT_ATTR, False):
                weight = weight.transpose(0, 1)
            return quantized_matmul_prequant(
                hidden_states_fp8,
                hidden_states_scale,
                weight,
                self.gate_up_proj.weight_scale,
            )
        return self.gate_up_proj(x)[0]

    def forward(
        self,
        x: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states_fp8 is None:
            hidden_states_fp8 = self._prequant[0]
            hidden_states_scale = self._prequant[1]
        self._prequant[0] = None
        self._prequant[1] = None
        gate_up = self._apply_gate_up_proj(
            x, hidden_states_fp8, hidden_states_scale
        )
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
