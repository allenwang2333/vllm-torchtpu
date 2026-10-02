# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.

Sequence-parallel MoE (``is_sequence_parallel=True``) dispatches over the EP
group (``DP x TP``) instead, because each TP rank enters the MoE with a
``[T / TP, H]`` token slice (the decoder layer reduce-scatters the attention
output and keeps the residual chunked). When the model computes
``router_logits`` itself, the ``hidden_states`` and ``router_logits`` gathers
are packed into a single ``all_gather`` along the feature dim when their dtypes
match, halving the number of latency-bound collective launches per layer. When
the runner owns the gate, ``VllmMoERunner._forward_impl`` gathers only
``hidden_states`` and runs the gate afterwards, so this path is not reached.
"""

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


class TpuDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)

    def _get_dispatch_group(self, is_sequence_parallel: bool):
        return get_ep_group() if is_sequence_parallel else get_dp_group()

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    @staticmethod
    def _can_pack_gather(hidden_states: torch.Tensor, router_logits: torch.Tensor):
        return (
            hidden_states.ndim == 2
            and router_logits.ndim == 2
            and hidden_states.dtype == router_logits.dtype
            and hidden_states.shape[0] == router_logits.shape[0]
        )

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._get_dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states, router_logits

        if is_sequence_parallel and self._can_pack_gather(hidden_states, router_logits):
            # One packed [T/TP, H + E] gather instead of two launches.
            hidden_dim = hidden_states.shape[-1]
            packed = torch.cat([hidden_states, router_logits], dim=-1)
            packed = group.all_gather(packed, dim=0)
            return packed[:, :hidden_dim], packed[:, hidden_dim:]

        hidden_states = group.all_gather(hidden_states, dim=0)
        router_logits = group.all_gather(router_logits, dim=0)
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._get_dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = group.all_gather(hidden_states, dim=0)
        topk_weights = group.all_gather(topk_weights, dim=0)
        topk_ids = group.all_gather(topk_ids, dim=0)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._get_dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states
        return group.reduce_scatter(hidden_states, dim=0)
