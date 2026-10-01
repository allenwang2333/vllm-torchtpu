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

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase, DeviceCommunicatorBase)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.layers.vllm import sequence_parallel as _sp


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

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    @staticmethod
    def _sequence_parallel_step(is_sequence_parallel: bool) -> bool:
        """Whether this call's rows are one rank's slice of the step.

        Under the row-gated sequence parallelism (layers/vllm/sequence_parallel)
        the MoE config is sequence-parallel for every compiled shape, but only
        the steps above the row threshold carry sliced rows; the smaller steps
        carry replicated rows and are combined with an all-reduce.
        """
        return bool(is_sequence_parallel and _sp.enabled() and _sp.is_active())

    @staticmethod
    def _replicated_sequence_parallel_step(is_sequence_parallel: bool) -> bool:
        return bool(is_sequence_parallel and _sp.enabled()
                    and not _sp.is_active())

    @staticmethod
    def _sp_gather(x: torch.Tensor) -> torch.Tensor:
        # every rank's owned rows in rank order, over the expert-parallel group
        return get_ep_group().all_gather(x, dim=0)

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
        if self._replicated_sequence_parallel_step(is_sequence_parallel):
            return hidden_states, router_logits
        if self._sequence_parallel_step(is_sequence_parallel):
            return self._sp_gather(hidden_states), self._sp_gather(
                router_logits)
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
        if self._replicated_sequence_parallel_step(is_sequence_parallel):
            return hidden_states, topk_weights, topk_ids
        if self._sequence_parallel_step(is_sequence_parallel):
            return (self._sp_gather(hidden_states),
                    self._sp_gather(topk_weights), self._sp_gather(topk_ids))
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
        if self._replicated_sequence_parallel_step(is_sequence_parallel):
            # replicated rows, partial sums over the ranks' experts: the
            # runner skips its final all-reduce under sequence parallelism
            from vllm_torchtpu.layers.vllm import small_all_reduce_op
            return small_all_reduce_op.all_reduce_ep(hidden_states)
        if self._sequence_parallel_step(is_sequence_parallel):
            return get_ep_group().reduce_scatter(hidden_states, dim=0)
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        # the small all-reduce kernel plus this rank's slice when the block
        # and the group allow it (small_all_reduce_op.dp_combine), else
        # vLLM's reduce-scatter
        from vllm_torchtpu.layers.vllm import small_all_reduce_op
        combined = small_all_reduce_op.dp_combine(hidden_states)
        if combined is not None:
            return combined
        return small_all_reduce_op.retain_dp_combine_inputs(
            dp.reduce_scatter(hidden_states, dim=0))
