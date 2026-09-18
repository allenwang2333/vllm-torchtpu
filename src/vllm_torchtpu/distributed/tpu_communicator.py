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
import vllm_torchtpu.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase, DeviceCommunicatorBase)
from vllm.distributed.parallel_state import get_dp_group
from vllm.utils.torch_utils import direct_register_custom_op


def _tpu_dp_all_to_all(
    tensor: torch.Tensor, world_size: int, group_name: str
) -> torch.Tensor:
    from vllm.distributed.parallel_state import _groups
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    input_contig = tensor.contiguous()
    output_tensor = torch.empty_like(input_contig)
    torch.distributed.all_to_all_single(
        output_tensor, input_contig, group=group.device_group
    )
    return output_tensor


def _tpu_dp_all_to_all_fake(
    tensor: torch.Tensor, world_size: int, group_name: str
) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="tpu_dp_all_to_all",
    op_func=_tpu_dp_all_to_all,
    fake_impl=_tpu_dp_all_to_all_fake,
)


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
        dp = get_dp_group()
        if dp.world_size == 1:
            return x
        return dp.all_gather(x, dim=0)

    def _dp_all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return x
        if dp.use_custom_op_call:
            return torch.ops.vllm.tpu_dp_all_to_all(
                x, dp.world_size, dp.unique_name
            )
        input_contig = x.contiguous()
        output_tensor = torch.empty_like(input_contig)
        torch.distributed.all_to_all_single(
            output_tensor, input_contig, group=dp.device_group
        )
        return output_tensor

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, router_logits

        fp8_mode = envs.TPU_MOE_FP8_DISPATCH
        if (fp8_mode in (1, 2)
                and isinstance(router_logits, tuple)
                and hidden_states.shape[-1] == 7168
                and router_logits[0].shape[-1] == 8):
            topk_weights, topk_ids = router_logits
            s_local = hidden_states.shape[0]
            if fp8_mode == 1:
                hs_blk = hidden_states.view(s_local, 14, 512)
                amax = hs_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                hs_scale = (amax / 448.0).to(torch.bfloat16)
                hs_fp8_bf16 = (hs_blk / hs_scale).to(
                    torch.float8_e4m3fn).view(s_local, 7168).view(torch.bfloat16)
                scale_flat = hs_scale.view(s_local, 14)
            else:
                amax = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                hs_scale = (amax / 448.0).to(torch.bfloat16)
                hs_fp8_bf16 = (hidden_states / hs_scale).to(
                    torch.float8_e4m3fn).view(torch.bfloat16)
                scale_flat = hs_scale.view(s_local, 1)

            ag_hs_fp8_bf16 = self._dp_gather(hs_fp8_bf16)
            s_global = ag_hs_fp8_bf16.shape[0]

            meta_local = torch.cat(
                [
                    scale_flat,
                    topk_weights.to(torch.bfloat16),
                    topk_ids.to(torch.int32).view(torch.bfloat16),
                ],
                dim=-1,
            )
            ag_meta = self._dp_gather(meta_local)

            world_size = dp.world_size
            if fp8_mode == 1:
                ag_scale_ranks = ag_meta[:, :14].view(world_size, s_local, 14, 1)
                ag_tw = ag_meta[:, 14:22].clone()
                ag_ti = ag_meta[:, 22:38].contiguous().view(torch.int32)
                ag_hs_fp8_ranks = ag_hs_fp8_bf16.view(
                    torch.float8_e4m3fn).view(world_size, s_local, 14, 512)
                ag_hs = torch.cat(
                    [
                        (ag_hs_fp8_ranks[r].to(torch.bfloat16) *
                         ag_scale_ranks[r]).view(s_local, 7168)
                        for r in range(world_size)
                    ],
                    dim=0,
                )
            else:
                ag_scale_ranks = ag_meta[:, :1].view(world_size, s_local, 1)
                ag_tw = ag_meta[:, 1:9].clone()
                ag_ti = ag_meta[:, 9:25].contiguous().view(torch.int32)
                ag_hs_fp8_ranks = ag_hs_fp8_bf16.view(
                    torch.float8_e4m3fn).view(world_size, s_local, 7168)
                ag_hs = torch.cat(
                    [
                        ag_hs_fp8_ranks[r].to(torch.bfloat16) *
                        ag_scale_ranks[r]
                        for r in range(world_size)
                    ],
                    dim=0,
                )
            return ag_hs, (ag_tw, ag_ti)

        if (fp8_mode in (3, 4)
                and isinstance(router_logits, tuple)
                and router_logits[0].shape[-1] == 8):
            topk_weights, topk_ids = router_logits
            ag_hs = self._dp_gather(hidden_states)
            meta_local = torch.cat(
                [
                    topk_weights.to(torch.float32),
                    topk_ids.to(torch.float32),
                ],
                dim=-1,
            )
            if fp8_mode == 4:
                meta_local = torch.nn.functional.pad(meta_local, (0, 112))
            ag_meta = self._dp_gather(meta_local)
            ag_tw = ag_meta[:, :8].to(hidden_states.dtype)
            ag_ti = ag_meta[:, 8:16].to(torch.int32)
            return ag_hs, (ag_tw, ag_ti)

        hidden_states = self._dp_gather(hidden_states)
        if isinstance(router_logits, tuple):
            topk_weights, topk_ids = router_logits
            router_logits = (
                self._dp_gather(topk_weights),
                self._dp_gather(topk_ids),
            )
        else:
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

        fp8_combine = envs.TPU_MOE_FP8_COMBINE
        if fp8_combine in (1, 2) and hidden_states.shape[-1] == 7168:
            world_size = dp.world_size
            s_global = hidden_states.shape[0]
            s_local = s_global // world_size
            if fp8_combine == 1:
                hs_blk = hidden_states.view(world_size, s_local, 14, 512)
                scale = (hs_blk.abs().amax(
                    dim=-1, keepdim=True).clamp(min=1e-12) / 448.0).to(
                        torch.bfloat16)
                q_fp8_bf16 = torch.cat(
                    [
                        (hs_blk[r] / scale[r]).to(torch.float8_e4m3fn).view(
                            1, s_local, 7168).view(torch.bfloat16)
                        for r in range(world_size)
                    ],
                    dim=0,
                )
                recv_fp8_bf16 = self._dp_all_to_all(q_fp8_bf16)
                recv_scale = self._dp_all_to_all(
                    scale.view(world_size, s_local, 14))
                recv_fp8 = recv_fp8_bf16.view(torch.float8_e4m3fn).view(
                    world_size, s_local, 14, 512)
                recv_scale_blk = recv_scale.view(world_size, s_local, 14, 1)
                acc = recv_fp8[0].to(torch.bfloat16) * recv_scale_blk[0]
                for r in range(1, world_size):
                    acc = acc + recv_fp8[r].to(
                        torch.bfloat16) * recv_scale_blk[r]
                return acc.view(s_local, 7168)
            else:
                hs_blk = hidden_states.view(world_size, s_local, 7168)
                scale = (hs_blk.abs().amax(
                    dim=-1, keepdim=True).clamp(min=1e-12) / 448.0).to(
                        torch.bfloat16)
                q_fp8_bf16 = torch.cat(
                    [
                        (hs_blk[r] / scale[r]).to(torch.float8_e4m3fn).view(
                            torch.bfloat16).view(1, s_local, 3584)
                        for r in range(world_size)
                    ],
                    dim=0,
                )
                recv_fp8_bf16 = self._dp_all_to_all(q_fp8_bf16)
                recv_scale = self._dp_all_to_all(scale)
                recv_fp8 = recv_fp8_bf16.view(torch.float8_e4m3fn)
                acc = recv_fp8[0].to(torch.bfloat16) * recv_scale[0]
                for r in range(1, world_size):
                    acc = acc + recv_fp8[r].to(torch.bfloat16) * recv_scale[r]
                return acc

        return dp.reduce_scatter(hidden_states, dim=0)
