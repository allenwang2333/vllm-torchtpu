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
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group

from vllm_torchtpu import envs

# TPU_TP_ALLREDUCE_IMPL=twoshot range. Set from the workload-microbenchmark collective component
# on 2x2x2 (16 TensorCores, 2 hosts): XLA's all-reduce is 12.5 us at 1 KiB and 19.7 us at the
# decode step's bf16[16, 6144]; see kernels/collectives/oneshot.py.
_TWOSHOT_MAX_BYTES = 512 * 1024


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

    def prepare_all_reduce_kernel(self) -> None:
        """Build the two-shot all-reduce op for this group, once, outside any trace (building
        its mesh runs a CPU collective). A no-op unless TPU_TP_ALLREDUCE_IMPL=twoshot."""
        if envs.TPU_TP_ALLREDUCE_IMPL != "twoshot" or self.world_size < 2:
            return
        if getattr(self, "_twoshot_ar", None) is not None:
            return
        from vllm_torchtpu.kernels.collectives import oneshot

        self._twoshot_ar = oneshot.torch_op(
            "all_reduce", self, self.unique_name.replace(":", "_"), variant="twoshot"
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        op = getattr(self, "_twoshot_ar", None)
        # The size cut-off is a Python branch on the token count: only the backbone warmup gives
        # each side of such a branch its own trace (shape_variants). The MTP drafter is traced
        # once for all buckets and would freeze the branch, so it keeps XLA's all-reduce.
        from vllm_torchtpu.compilation import shape_variants

        if op is None or not shape_variants.in_warmup():
            return super().all_reduce(input_)
        nbytes = input_.numel() * input_.element_size()
        if (
            nbytes <= _TWOSHOT_MAX_BYTES
            and input_.numel() % self.world_size == 0
            and input_.dim() >= 1
        ):
            flat = input_.reshape(input_.shape[0], -1)
            return op(flat).view(input_.shape)
        return super().all_reduce(input_)

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
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
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
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self._dp_gather(hidden_states)
        topk_weights = self._dp_gather(topk_weights)
        topk_ids = self._dp_gather(topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        return dp.reduce_scatter(hidden_states, dim=0)
