# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small Kimi layers that are not attention or MoE."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import nn
import vllm.model_executor.layers.linear as linear_module
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@contextmanager
def _linear_tp_override(tp_rank: int, tp_size: int):
    """Construct standard vLLM linears with a subgroup's TP geometry.

    The linear classes store the resolved rank and size on the layer and its
    parameters, so the temporary override only affects construction and weight
    loading continues to select the intended replicated shard.
    """
    old_rank = linear_module.get_tensor_model_parallel_rank
    old_size = linear_module.get_tensor_model_parallel_world_size
    linear_module.get_tensor_model_parallel_rank = lambda: tp_rank
    linear_module.get_tensor_model_parallel_world_size = lambda: tp_size
    try:
        yield
    finally:
        linear_module.get_tensor_model_parallel_rank = old_rank
        linear_module.get_tensor_model_parallel_world_size = old_size


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
        tp_group: GroupCoordinator | None = None,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        tp_context = (_linear_tp_override(tp_group.rank_in_group,
                                          tp_group.world_size)
                      if tp_group is not None else _linear_tp_override(
                          linear_module.get_tensor_model_parallel_rank(),
                          linear_module.get_tensor_model_parallel_world_size()))
        with tp_context:
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
                reduce_results=(reduce_results and tp_group is None),
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
        if self.tp_group is not None:
            hidden_states = self.tp_group.all_reduce(hidden_states)
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

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residuals: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
        scores, _ = self.proj(self.norm(values))
        probabilities = scores.float().softmax(dim=-2)
        return (probabilities * values.float()).sum(dim=-2).to(values.dtype)
