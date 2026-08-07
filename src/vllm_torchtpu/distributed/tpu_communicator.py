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
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed._functional_collectives as funcol
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group


@dataclass
class TpuAsyncCollective:
    """Functional collective result with explicit lifetime and wait."""

    tensor: torch.Tensor
    retained_tensors: tuple[torch.Tensor, ...]
    operation: str
    microbatch_index: int
    _wait: Callable[[torch.Tensor], torch.Tensor]
    _completed: torch.Tensor | None = None

    def wait(self) -> torch.Tensor:
        if self._completed is None:
            self._completed = self._wait(self.tensor)
            # Inputs must remain live until the explicit wait has been emitted.
            self.retained_tensors = ()
        return self._completed


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


def _raw_all_gather_tensor(x: torch.Tensor, group_size: int,
                           group_name: str) -> torch.Tensor:
    """Launch all-gather without wrapping its result in an async tensor."""
    return torch.ops._c10d_functional.all_gather_into_tensor(
        x, group_size, group_name)


def _raw_reduce_scatter_tensor(x: torch.Tensor, group_size: int,
                               group_name: str) -> torch.Tensor:
    """Launch sum/reduce-scatter without an implicit wrapper wait."""
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        x, "sum", group_size, group_name)


class TpuDeviceCommunicator(DeviceCommunicatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)

    def all_gather_start(
            self, x: torch.Tensor,
            microbatch_index: int) -> TpuAsyncCollective:
        """Launch a functional DP all-gather without inserting its wait."""
        dp = get_dp_group()
        if dp.world_size == 1:
            result = x
            wait = lambda value: value
        else:
            contiguous_x = x.contiguous()
            group_name = funcol._resolve_group_name(dp.device_group)
            result = _raw_all_gather_tensor(contiguous_x, dp.world_size,
                                            group_name)
            x = contiguous_x
            wait = funcol.wait_tensor
        return TpuAsyncCollective(
            tensor=result, retained_tensors=(x,), operation="AG",
            microbatch_index=microbatch_index, _wait=wait)

    def reduce_scatter_start(
            self, x: torch.Tensor,
            microbatch_index: int) -> TpuAsyncCollective:
        """Launch a functional DP sum/reduce-scatter without its wait."""
        dp = get_dp_group()
        if dp.world_size == 1:
            result = x
            wait = lambda value: value
        else:
            group_name = funcol._resolve_group_name(dp.device_group)
            result = _raw_reduce_scatter_tensor(x, dp.world_size, group_name)
            wait = funcol.wait_tensor
        return TpuAsyncCollective(
            tensor=result, retained_tensors=(x,), operation="RS",
            microbatch_index=microbatch_index, _wait=wait)

    @staticmethod
    def wait(handle: TpuAsyncCollective) -> torch.Tensor:
        return handle.wait()

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, router_logits
        hidden_states = self._dp_gather(hidden_states)
        router_logits = self._dp_gather(router_logits)
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
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self._dp_gather(hidden_states)
        topk_weights = self._dp_gather(topk_weights)
        topk_ids = self._dp_gather(topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        return dp.reduce_scatter(hidden_states, dim=0)
