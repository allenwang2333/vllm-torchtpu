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
"""Pipelined Fused MoE wrapper with collective communication chunking.

This module provides `pipelined_fused_moe_gmm`, which acts as a wrapper around
`fused_moe_gmm` (or a custom MoE kernel) to manage and overlap DP or PCP
collective communications (AllGather and ReduceScatter) with MoE computation.
"""

from typing import Optional

import torch
from vllm.distributed.parallel_state import get_dp_group

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def enable_pipelined_collective_and_compute() -> bool:
    """Check whether collective communication and compute pipelining is enabled.

    Returns:
        bool: True if `envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0`, False otherwise.

    """
    return envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0


# Alias for alternative naming convention
enable_pipeline_collective_and_compute = enable_pipelined_collective_and_compute


def calculate_moe_chunks(seq_len: int, parallel_size: int,
                         chunk_size: int) -> tuple[int, int]:
    """Calculate the number of chunks and local chunk size for MoE chunk pipelining.

    Uses ceiling division so `chunk_size` acts as a maximum global chunk threshold:
        num_chunks = ceil((seq_len * parallel_size) / chunk_size)
        chunk_size_local = seq_len // num_chunks

    Args:
        seq_len: Number of local tokens on this rank (S).
        parallel_size: World size of the active DP or PCP collective group.
        chunk_size: Maximum post-gather chunk size threshold in tokens (C).

    Returns:
        tuple[int, int]: (num_chunks N_chunk, chunk_size_local S_chunk).

    Raises:
        ValueError: If local sequence length is not divisible by chunk count.
    """
    if chunk_size <= 0 or parallel_size <= 1 or seq_len <= 0:
        return 1, seq_len

    total_tokens = seq_len * parallel_size
    num_chunks = int((total_tokens + chunk_size - 1) // chunk_size)
    if seq_len % num_chunks != 0:
        raise ValueError(
            f"Local sequence length ({seq_len}) is not divisible by chunk count ({num_chunks})."
        )

    chunk_size_local = seq_len // num_chunks
    return num_chunks, chunk_size_local


def _tpu_opt_barrier(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(torch.ops, "tpu") and hasattr(torch.ops.tpu, "optimization_barrier"):
        res = torch.ops.tpu.optimization_barrier([a, b])
        return res[0], res[1]
    return a, b


def _gather_routing_meta(
    collective_group,
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (topk_weights is not None and topk_ids is not None
            and topk_weights.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32):
        meta_local = torch.cat(
            [topk_weights.view(torch.int16).to(torch.int32), topk_ids],
            dim=-1,
        )
        ag_meta = collective_group.all_gather(meta_local, dim=0)
        k = topk_weights.shape[-1]
        ag_tw = ag_meta[:, :k].to(torch.int16).view(torch.bfloat16)
        ag_ti = ag_meta[:, k:]
        return ag_tw, ag_ti
    ag_tw = (collective_group.all_gather(topk_weights, dim=0)
             if topk_weights is not None else None)
    ag_ti = (collective_group.all_gather(topk_ids, dim=0)
             if topk_ids is not None else None)
    return ag_tw, ag_ti


def _quantize_hs_fp8(hs_slice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = hs_slice.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    hs_scale = (amax / 448.0).to(torch.bfloat16)
    hs_fp8_bf16 = (hs_slice / hs_scale).to(torch.float8_e4m3fn).view(torch.bfloat16)
    return hs_fp8_bf16, hs_scale.view(hs_slice.shape[0], 1)


def _gather_hs_and_meta(
    collective_group,
    hs_slice: torch.Tensor,
    tw_slice: Optional[torch.Tensor],
    ti_slice: Optional[torch.Tensor],
    use_fp8: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (use_fp8 and tw_slice is not None and ti_slice is not None
            and hs_slice.dtype == torch.bfloat16
            and hs_slice.shape[-1] == 7168 and tw_slice.shape[-1] == 8):
        hs_fp8_bf16, hs_scale = _quantize_hs_fp8(hs_slice)
        ag_hs_fp8 = collective_group.all_gather(hs_fp8_bf16, dim=0)
        meta_local = torch.cat(
            [
                hs_scale,
                tw_slice.to(torch.bfloat16),
                ti_slice.to(torch.int32).view(torch.bfloat16),
            ],
            dim=-1,
        )
        ag_meta = collective_group.all_gather(meta_local, dim=0)
        ag_scale = ag_meta[:, :1]
        ag_tw = ag_meta[:, 1:9]
        ag_ti = ag_meta[:, 9:25].contiguous().view(torch.int32)
        ag_hs = ag_hs_fp8.view(torch.float8_e4m3fn).to(torch.bfloat16) * ag_scale
        return ag_hs, ag_tw, ag_ti

    ag_hs = collective_group.all_gather(hs_slice, dim=0)
    ag_tw, ag_ti = _gather_routing_meta(collective_group, tw_slice, ti_slice)
    return ag_hs, ag_tw, ag_ti


def _get_moe_collective_group():
    """Select the collective group whose expert shards the kernel combines."""
    pcp_group = get_pcp_group()
    dp_group = get_dp_group()
    pcp_size = int(pcp_group.world_size) if pcp_group is not None else 1
    dp_size = int(dp_group.world_size) if dp_group is not None else 1

    if pcp_size > 1 and dp_size > 1:
        raise NotImplementedError(
            "MoE collective chunking does not yet support simultaneous "
            f"PCP ({pcp_size}) and DP ({dp_size}).")
    if pcp_size > 1:
        return pcp_group, "PCP"
    return dp_group, "DP"


@torch.compiler.assume_constant_result
def _log_active_chunk_pipeline() -> None:
    """Emit trace-time evidence without passing symbolic values to logging."""
    collective_group, collective_kind = _get_moe_collective_group()
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    logger.info_once(
        "MoE multi-chunk pipeline branch active: group=%s, world_size=%d, "
        "threshold=%d", collective_kind, parallel_size,
        envs.TPU_MOE_COLLECTION_CHUNK_SIZE)


def pipelined_fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    experts_start: Optional[torch.Tensor],
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: Optional[bool] = None,
    route_fn=None,
    pre_gate_ag=None,
    shared_experts=None,
):
    """Wrapper around fused_moe_gmm to pipeline DP/PCP collectives and compute."""
    chunk_size = envs.TPU_MOE_COLLECTION_CHUNK_SIZE
    collective_group, _ = _get_moe_collective_group()

    seq_len = hidden_states.shape[0]
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    num_chunks, chunk_size_local = calculate_moe_chunks(
        seq_len, parallel_size, chunk_size)
    use_fp8 = (envs.TPU_MOE_FP8_DISPATCH == 5 and num_chunks > 1)
    use_barrier = (envs.TPU_MOE_PIPELINE_BARRIER >= 1)

    if num_chunks == 1 or parallel_size == 1:
        if parallel_size > 1:
            if pre_gate_ag is not None and pre_gate_ag[1] is not None:
                ag_hidden_states = pre_gate_ag[1]
            else:
                ag_hidden_states = collective_group.all_gather(hidden_states, dim=0)
            if route_fn is not None and (topk_weights is None or topk_ids is None):
                topk_weights, topk_ids = route_fn()
            ag_topk_weights, ag_topk_ids = _gather_routing_meta(
                collective_group, topk_weights, topk_ids)
        else:
            if route_fn is not None and (topk_weights is None or topk_ids is None):
                topk_weights, topk_ids = route_fn()
            ag_hidden_states = hidden_states
            ag_topk_weights = topk_weights
            ag_topk_ids = topk_ids

        shared_act = None
        if shared_experts is not None:
            shared_layer, shared_in = shared_experts
            gate_up, _ = shared_layer.gate_up_proj(shared_in)
            shared_act = shared_layer.act_fn(gate_up)

        out = fused_moe_gmm(
            hidden_states=ag_hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            topk_weights=ag_topk_weights,
            topk_ids=ag_topk_ids,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )

        if parallel_size > 1:
            rs_out = collective_group.reduce_scatter(out, dim=0)
            if shared_act is not None:
                shared_out, _ = shared_experts[0].down_proj(shared_act)
                return rs_out, shared_out
            if shared_experts is not None:
                return rs_out, None
            return rs_out
        if shared_act is not None:
            shared_out, _ = shared_experts[0].down_proj(shared_act)
            return out, shared_out
        return out

    _log_active_chunk_pipeline()

    # Slice local inputs into num_chunks chunks along dim 0
    if pre_gate_ag is not None and pre_gate_ag[0] is not None:
        hs_slices = pre_gate_ag[0]
        pre_ag_hs0 = pre_gate_ag[1]
    else:
        hs_slices = list(torch.chunk(hidden_states, num_chunks, dim=0))
        pre_ag_hs0 = None

    # Stage 0: Issue AllGather for chunk 0 hidden_states BEFORE route_fn() so SparseCore overlaps with TensorCore router
    if use_fp8 and hs_slices[0].dtype == torch.bfloat16 and hs_slices[0].shape[-1] == 7168:
        hs0_fp8_bf16, hs0_scale = _quantize_hs_fp8(hs_slices[0])
        ag_hs0_fp8 = collective_group.all_gather(hs0_fp8_bf16, dim=0)
        if route_fn is not None and (topk_weights is None or topk_ids is None):
            topk_weights, topk_ids = route_fn()
        weights_slices = list(torch.chunk(topk_weights, num_chunks, dim=0))
        ids_slices = list(torch.chunk(topk_ids, num_chunks, dim=0))
        meta0_local = torch.cat(
            [
                hs0_scale,
                weights_slices[0].to(torch.bfloat16),
                ids_slices[0].to(torch.int32).view(torch.bfloat16),
            ],
            dim=-1,
        )
        ag_meta0 = collective_group.all_gather(meta0_local, dim=0)
        ag_scale0 = ag_meta0[:, :1]
        ag_tw_curr = ag_meta0[:, 1:9]
        ag_ti_curr = ag_meta0[:, 9:25].contiguous().view(torch.int32)
        ag_hs_curr = ag_hs0_fp8.view(torch.float8_e4m3fn).to(torch.bfloat16) * ag_scale0
    else:
        ag_hs_curr = (pre_ag_hs0 if pre_ag_hs0 is not None else
                      collective_group.all_gather(hs_slices[0], dim=0))
        if route_fn is not None and (topk_weights is None or topk_ids is None):
            topk_weights, topk_ids = route_fn()
        weights_slices = (list(torch.chunk(topk_weights, num_chunks, dim=0))
                          if topk_weights is not None else [None] * num_chunks)
        ids_slices = (list(torch.chunk(topk_ids, num_chunks, dim=0))
                      if topk_ids is not None else [None] * num_chunks)
        ag_tw_curr, ag_ti_curr = _gather_routing_meta(
            collective_group, weights_slices[0], ids_slices[0])

    shared_act = None
    if shared_experts is not None:
        shared_layer, shared_in = shared_experts
        gate_up, _ = shared_layer.gate_up_proj(shared_in)
        shared_act = shared_layer.act_fn(gate_up)

    rs_outputs = []

    for i in range(num_chunks):
        # Issue asynchronous all-gather for next chunk if available
        if i + 1 < num_chunks:
            hs_next = hs_slices[i + 1]
            if use_barrier:
                ag_hs_curr, hs_next = _tpu_opt_barrier(ag_hs_curr, hs_next)
            ag_hs_next, ag_tw_next, ag_ti_next = _gather_hs_and_meta(
                collective_group,
                hs_next,
                weights_slices[i + 1],
                ids_slices[i + 1],
                use_fp8,
            )

        # Compute MoE kernel for current gathered chunk
        out_i = fused_moe_gmm(
            hidden_states=ag_hs_curr,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            topk_weights=ag_tw_curr,
            topk_ids=ag_ti_curr,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )

        if use_barrier and i + 1 < num_chunks:
            ag_hs_next, out_i = _tpu_opt_barrier(ag_hs_next, out_i)
        if use_barrier and i > 0:
            rs_outputs[-1], out_i = _tpu_opt_barrier(rs_outputs[-1], out_i)

        # Reduce-scatter partial output for current chunk
        rs_i = collective_group.reduce_scatter(out_i, dim=0)
        rs_outputs.append(rs_i)

        if i + 1 < num_chunks:
            ag_hs_curr, ag_tw_curr, ag_ti_curr = ag_hs_next, ag_tw_next, ag_ti_next

    combined_out = torch.cat(rs_outputs, dim=0)
    if shared_act is not None:
        shared_out, _ = shared_experts[0].down_proj(shared_act)
        return combined_out, shared_out
    if shared_experts is not None:
        return combined_out, None
    return combined_out
