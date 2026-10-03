# SPDX-License-Identifier: Apache-2.0
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

    def _moe_group(self, is_sequence_parallel: bool):
        """Group that gathers MoE inputs and scatters outputs, or None.

        Sequence-parallel tokens are sharded over the EP group (DP x TP).
        Otherwise they are sharded over DP only.
        """
        if is_sequence_parallel:
            return get_ep_group()
        dp = get_dp_group()
        return dp if dp.world_size > 1 else None

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
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, router_logits
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
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, topk_weights, topk_ids
        hidden_states = group.all_gather(hidden_states, dim=0)
        topk_weights = group.all_gather(topk_weights, dim=0)
        topk_ids = group.all_gather(topk_ids, dim=0)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        return group.reduce_scatter(hidden_states, dim=0)
