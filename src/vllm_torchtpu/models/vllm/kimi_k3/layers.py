# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small Kimi layers that are not attention or MoE."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

from vllm_torchtpu.layers.vllm.custom_ops.k3_attn_res_op import \
    build_k3_attn_res_mix_op

_MAX_KERNEL_STACK = 8


def use_attn_res_kernel() -> bool:
    """Whether AttentionResidual dispatches to the fused Pallas mix kernel.

    The platform registers out-of-tree even on chip-less hosts, so gate on
    the physical chip count, not the platform type alone.
    """
    return (current_platform.device_type == "tpu"
            and current_platform.device_count() > 0)


class SituAndMul(nn.Module):
    """Kimi's SITU gated activation.

    Keep this implementation local because released vLLM versions do not
    expose ``SituAndMul`` from their generic activation module.
    """

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = (self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate))
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return gate * up


class KimiMLP(nn.Module):
    """The standard vLLM merged-projection gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=situ_beta or 1.0,
                linear_beta=situ_linear_beta,
            )
        else:
            raise ValueError(f"Unsupported Kimi activation {hidden_act!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class AttentionResidual(nn.Module):
    """K3's attention-residual weighted sum in ordinary PyTorch."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.proj = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=prefix,
        )
        self.eps = eps
        self._use_mix_kernel = False
        self._mix_ops: dict[int, Callable[..., torch.Tensor]] = {}
        if use_attn_res_kernel():
            self.register_buffer("w_score",
                                 torch.empty(1,
                                             hidden_size,
                                             dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("w_norm_row",
                                 torch.empty(1, hidden_size),
                                 persistent=False)
            self.register_buffer("w_proj_row",
                                 torch.empty(1, hidden_size),
                                 persistent=False)

    def process_weights_after_loading(self) -> None:
        """Contract the RMSNorm -> rank-1 scorer into one weighted dot.

        score(x) = <x, w_norm * w_proj> * rsqrt(mean(x^2) + eps), so the
        kernel never materializes the normed stack. Fresh copy_-filled
        buffers break the lazy lineage (same reason as conv_weight_fused in
        attention.py). Ops are built eagerly here because the Pallas op
        builder runs inspect.signature(), which Dynamo cannot trace
        inside a compiled forward.
        """
        if not use_attn_res_kernel():
            return
        w_norm = self.norm.weight.data.reshape(1, -1)
        w_proj = self.proj.weight.data.reshape(1, -1)
        self.w_score.copy_(w_norm.float() * w_proj.float())
        self.w_norm_row.copy_(w_norm)
        self.w_proj_row.copy_(w_proj)
        self._mix_ops = {
            n: build_k3_attn_res_mix_op(n, self.eps, exact_emulation=False)
            for n in range(1, _MAX_KERNEL_STACK + 1)
        }
        self._use_mix_kernel = True

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residuals: torch.Tensor,
    ) -> torch.Tensor:
        n = block_residuals.shape[-2]
        # n is a trace-time constant per layer, so this branch freezes
        # correctly under TorchCompileWithNoGuardsWrapper.
        if (self._use_mix_kernel and 1 <= n <= _MAX_KERNEL_STACK
                and prefix_sum.dtype == torch.bfloat16):
            return self._mix_ops[n](prefix_sum, block_residuals, self.w_score,
                                    self.w_norm_row, self.w_proj_row)
        values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
        scores, _ = self.proj(self.norm(values))
        probabilities = scores.float().softmax(dim=-2)
        return (probabilities * values.float()).sum(dim=-2).to(values.dtype)
