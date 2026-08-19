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
from vllm.distributed.parallel_state import get_dp_group

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import fp8_wire
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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
        # Read once: one communicator is built per process at init.
        self.dispatch_fp8 = envs.TPU_MOE_DISPATCH_FP8
        self.packed_wire = fp8_wire.packed_wire()
        if self.dispatch_fp8:
            logger.info_once(
                "MoE dispatch all-gather runs in fp8 (TPU_MOE_DISPATCH_FP8=1, "
                f"wire={'uint8 packed' if self.packed_wire else 'fp8 + scale'})"
            )

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    def dispatch_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """DP all-gather of MoE token rows, in fp8 when the flag is set.

        The single owner of the dispatch wire format: anything that gathers
        activations for the MoE -- including a patched upstream
        ``MoERunner._maybe_dispatch`` -- must come through here, or
        ``TPU_MOE_DISPATCH_FP8`` silently stops applying. Safe to call at
        ``dp_size == 1``.

        The rows are rounded to fp8 one step later regardless -- gmm_v2
        quantizes its lhs before the matmul -- so the wire mostly costs a
        rounding the MoE was going to do anyway. Not exactly: the wire scales
        per row where the kernel scales per 512-wide k block, and that second
        grid is what the layer's rel-L2 pays for (5.93e-2 against an fp32
        reference where the kernel's own blocks would give 4.62e-2).

        The rows and their scales cross as two collectives, not one packed
        uint8 buffer: packing saves ~9 us of fixed cost and loses the async
        collective form, which measured -60.8% and -61.5% on device. See
        ``fp8_wire.packed_wire``.
        """
        if get_dp_group().world_size == 1:
            return x
        if not self.dispatch_fp8 or not x.is_floating_point():
            return self._dp_gather(x)
        if self.packed_wire:
            payload = self._dp_gather(fp8_wire.encode(x))
            return fp8_wire.decode(payload, x.shape[-1], x.dtype)
        # Two collectives, both in the async form: the fp8 rows and the fp32
        # scale that reconstructs them. The scale is [T, 1], which XLA lays out
        # as a dense vector -- 4 B/token, not a lane-padded tile.
        q, scale = fp8_wire.quantize_rows(x)
        return fp8_wire.dequantize_rows(self._dp_gather(q),
                                        self._dp_gather(scale), x.dtype)

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
        hidden_states = self.dispatch_hidden_states(hidden_states)
        # Router logits stay bf16: they are 1/8 of the hidden payload and a
        # rounding there moves top-k membership, not just magnitudes.
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
        hidden_states = self.dispatch_hidden_states(hidden_states)
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
