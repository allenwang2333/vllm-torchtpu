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

import types
from collections.abc import Callable

import torch
import torch.nn as nn
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

_TPU_OP_CACHE: dict[tuple, Callable] = {}


def _get_tpu_op(key: tuple, fn: Callable) -> Callable:
    op = _TPU_OP_CACHE.get(key)
    if op is not None:
        return op
    name = f"dsv4_layer_op_{len(_TPU_OP_CACHE)}"
    specialized = types.FunctionType(
        fn.__code__.replace(co_name=name),
        fn.__globals__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    op = torch.compile(specialized, backend="tpu", dynamic=False)
    _TPU_OP_CACHE[key] = op
    return op


def _mhc_norm_scale_impl(
    residual_flat: torch.Tensor,
    mean_sq: torch.Tensor,
    rms_norm_eps: float,
) -> torch.Tensor:
    return residual_flat * torch.rsqrt(mean_sq + rms_norm_eps)


def _mhc_gate_weight_impl(
    pre_mix: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hs_f32: torch.Tensor,
    hc_eps: float,
) -> torch.Tensor:
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return pre_mix.unsqueeze(-1) * hs_f32


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
    orig_dtype = hidden_states.dtype
    hs_f32 = hidden_states if orig_dtype == torch.float32 else hidden_states.float()
    residual_flat = hs_f32.flatten(-2)
    mean_sq = residual_flat.square().mean(dim=-1, keepdim=True)
    if hidden_states.device.type == "tpu" and not torch.compiler.is_compiling():
        norm_op = _get_tpu_op(
            (
                "mhc_norm_scale",
                residual_flat.shape,
                mean_sq.shape,
                rms_norm_eps,
            ),
            lambda rf, ms: _mhc_norm_scale_impl(rf, ms, rms_norm_eps),
        )
        residual_norm = norm_op(residual_flat, mean_sq)
        pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
        gate_w_op = _get_tpu_op(
            (
                "mhc_gate_weight",
                pre_mix.shape,
                pre_mix.dtype,
                hc_scale.shape,
                hc_scale.dtype,
                hc_base.shape,
                hc_base.dtype,
                hs_f32.shape,
                hc_eps,
            ),
            lambda pm, sc, bs, hs: _mhc_gate_weight_impl(pm, sc, bs, hs, hc_eps),
        )
        weighted = gate_w_op(pre_mix, hc_scale, hc_base, hs_f32)
    else:
        residual_norm = _mhc_norm_scale_impl(residual_flat, mean_sq, rms_norm_eps)
        pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
        weighted = _mhc_gate_weight_impl(pre_mix, hc_scale, hc_base, hs_f32, hc_eps)
    out = torch.sum(weighted, dim=-2)
    return out if orig_dtype == torch.float32 else out.to(orig_dtype)


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
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
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
        self.swiglu_limit = swiglu_limit
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if gate_up.device.type == "tpu" and not torch.compiler.is_compiling():
            act_fn = self.act_fn
            act_op = _get_tpu_op(
                (
                    "mlp_act",
                    type(act_fn),
                    self.swiglu_limit,
                    gate_up.shape,
                    gate_up.dtype,
                ),
                lambda gu: act_fn(gu),
            )
            x = act_op(gate_up)
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
