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
from vllm_torchtpu import env_override as _env_override
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm
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

# The GMM kernel quantizes its own lhs to fp8 e4m3, per row, over contiguous
# 512-element blocks of the contracting dimension:
# `kernels/megablox/gmm_v2.py` sets the lhs `quant_block_size` to 512 in
# `make_gmm_configs` and the kernel body takes `jnp.max(jnp.abs(block_lhs),
# axis=1)` over `tiled_lhs[:, start_k:end_k]`. That reduction runs along the
# hidden dimension and never crosses a row, so the scale a token gets is a
# function of that token alone -- not of which rank the token came from, and
# not of how many tokens were gathered next to it. Quantizing here, before the
# all-gather, therefore reproduces those scales rather than changing their
# granularity, and the wire carries one byte per element instead of two.
#
# Blocks line up with the kernel's only when its `tile_k` equals the full
# contracting dimension, which is the case for the shapes this path runs
# (the emitted kernel is `...-k_6144-...-tk_6144-...`). `_FP8_WIRE_QUANT_BLOCK`
# must stay equal to `gmm_v2`'s lhs `quant_block_size`.
_FP8_WIRE_QUANT_BLOCK = 512
_FP8_WIRE_E4M3_MAX = 448.0
_FP8_WIRE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def fp8_wire_allgather_applies(x: torch.Tensor, parallel_size: int) -> bool:
    """Whether `x` should ride the dispatch all-gather as fp8 rather than bf16.

    The size test is not a safety margin, it is the whole point. Halving a
    payload only helps a collective whose cost is its payload. Below the
    SparseCore offload threshold the gather runs on the TensorCore as one
    synchronous `distributed_all_gather_into_tensor` and is dominated by its
    fixed cost -- measured on GLM-5.2-FP8 decode, 30.31 us for a 3 MiB gather,
    against a ~36 us/call fixed term at world=16, so the bytes were never the
    constraint. Quantizing there returns nothing and still pays for the second
    collective the scales ride in: 75 extra small all-gathers per core-step at
    ~14.84 us measured = +1.11 ms, which is 62% of the +1.79 ms median TPOT
    regression the ungated version cost in j-86cfd828-8634-40fe-beca. Prefill
    gathers 96 MiB and clears the threshold; decode gathers 3 MiB and does not.
    """
    if not (envs.TPU_MOE_FP8_WIRE_ALLGATHER and x.dim() == 2
            and x.dtype in _FP8_WIRE_DTYPES
            and x.shape[1] % _FP8_WIRE_QUANT_BLOCK == 0):
        return False
    gathered_bytes = (x.shape[0] * parallel_size * x.shape[1] *
                      x.element_size())
    return gathered_bytes >= _env_override.SC_OFFLOAD_THRESHOLD_BYTES


def _all_gather_activations(collective_group, x: torch.Tensor) -> torch.Tensor:
    """All-gather activations along dim 0, optionally quantized on the wire.

    Returns a tensor of `x`'s own dtype either way, so everything downstream --
    routing, the SparseCore ragged gather, the GMM kernel -- sees what it saw
    before. Only the bytes the collective moves change.

    The round trip nearly, but not exactly, preserves what the kernel derives.
    The element that set a block's abs-max maps to exactly +/-448, so the
    kernel re-derives the same fp8 codes -- measured on device, 0 differing
    codes in 50.3M. Its re-derived *scale* moves by exactly one bf16 ULP on
    30.6% of blocks (max relative error 0.75%, never more than 1 ULP), because
    `fl(fl(x_max/448) * 448) / 448` does not land back on `x_max` in XLA's
    rounding the way it does in CPU torch. Reconstructing in float32 does not
    help; it gives the identical count. So this is a small, bounded, real
    numerical change, not a no-op, and it needs an accuracy check rather than
    an argument. Measured in j-3c06b709-ed10-468c-9646.

    The scale ride-along is a second, tiny all-gather (2 bytes per 512
    elements). It stays far below the SparseCore offload threshold, so it
    lands on the TensorCore next to the existing topk_weights/topk_ids
    gathers rather than competing for the SparseCore this change is trying to
    free -- but it is not free itself, which is why the payload size gate in
    `fp8_wire_allgather_applies` exists.
    """
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    if not fp8_wire_allgather_applies(x, parallel_size):
        return collective_group.all_gather(x, dim=0)

    # Everything below stays in `x`'s own dtype, never promoted to float32.
    # The kernel derives its scale from a bf16 tile in bf16 arithmetic, and a
    # float32 scale here is a *different* scale: it differs from the kernel's
    # in the last bits, which moves fp8 codes on elements sitting near a
    # rounding boundary. Matching the dtype holds the codes exactly; promoting
    # moves 3.85% of them, by up to 32 code steps. Both measured on device.
    rows, hidden = x.shape
    blocks = hidden // _FP8_WIRE_QUANT_BLOCK
    blocked = x.reshape(rows, blocks, _FP8_WIRE_QUANT_BLOCK)
    scale = blocked.abs().amax(dim=-1, keepdim=True) / _FP8_WIRE_E4M3_MAX
    # An all-zero block gives scale 0; mirror the kernel's own `where` and send
    # zeros rather than the inf/NaN a bare reciprocal would produce.
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                            scale.reciprocal())
    quantized = (blocked * scale_inv).to(torch.float8_e4m3fn).reshape(
        rows, hidden)

    ag_quantized = collective_group.all_gather(quantized, dim=0)
    ag_scale = collective_group.all_gather(scale.reshape(rows, blocks), dim=0)

    ag_rows = ag_quantized.shape[0]
    ag_blocked = ag_quantized.reshape(ag_rows, blocks,
                                      _FP8_WIRE_QUANT_BLOCK).to(x.dtype)
    return (ag_blocked * ag_scale.reshape(ag_rows, blocks, 1)).reshape(
        ag_rows, hidden)


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
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    if seq_len % num_chunks != 0:
        raise ValueError(
            f"Local sequence length ({seq_len}) is not divisible by chunk count ({num_chunks})."
        )

    chunk_size_local = seq_len // num_chunks
    return num_chunks, chunk_size_local


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
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: Optional[torch.Tensor],
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: Optional[bool] = None,
) -> torch.Tensor:
    """Wrapper around fused_moe_gmm to pipeline DP/PCP collectives and compute.

    This function wraps `fused_moe_gmm` by taking control of the active DP or
    PCP collective communications (AllGather of activations/routing tensors
    and ReduceScatter of outputs) and staging them in chunks to overlap network
    communication with TPU matrix multiplication compute.

    When `enable_pipelined_collective_and_compute()` is True and the collective
    world size is greater than one, the sequence is partitioned into balanced
    chunks and executed via a 3-stage pipelined schedule:
      ag0_start -> ag1_start || moe0(ag0_done) -> rs0_start || moe1(ag1_done) -> rs1_start || rs0_done -> rs1_done -> concat

    Args:
        hidden_states: Local input activations [S, H].
        w1: First expert projection weights [E, 2*I, H] or [E, H, 2*I]
            (concatenated gate W1 and up W3 projections with intermediate dimension
            2*I for gated activations like SwiGLU when activation="silu").
        w2: Second expert projection weights [E, H, I] or [E, I, H]
            (down projection W2 from intermediate dimension I back to hidden dimension H).
        w1_scale: Optional first projection scale tensor.
        w2_scale: Optional second projection scale tensor.
        w1_bias: Optional first projection bias tensor.
        w2_bias: Optional second projection bias tensor.
        topk_weights: Routing top-k weights [S, topk].
        topk_ids: Routing top-k expert IDs [S, topk].
        experts_start: Optional 0-d tensor of starting expert ID for this shard.
        topk: Number of experts selected per token.
        activation: Gated activation function string (e.g., "silu" for SwiGLU).
        rhs_quant_dtype: Optional RHS quantization data type.
        skip_padded_tokens: Optional flag to skip padded tokens.

    Returns:
        torch.Tensor: Reduced local output activations [S, H].
    """
    chunk_size = envs.TPU_MOE_COLLECTION_CHUNK_SIZE
    collective_group, _ = _get_moe_collective_group()

    seq_len = hidden_states.shape[0]
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    num_chunks, chunk_size_local = calculate_moe_chunks(
        seq_len, parallel_size, chunk_size)

    if num_chunks == 1 or parallel_size == 1:
        if parallel_size > 1:
            ag_hidden_states = _all_gather_activations(
                collective_group, hidden_states)
            ag_topk_weights = (collective_group.all_gather(topk_weights, dim=0)
                               if topk_weights is not None else None)
            ag_topk_ids = (collective_group.all_gather(topk_ids, dim=0)
                           if topk_ids is not None else None)
        else:
            ag_hidden_states = hidden_states
            ag_topk_weights = topk_weights
            ag_topk_ids = topk_ids

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
            return collective_group.reduce_scatter(out, dim=0)
        return out

    _log_active_chunk_pipeline()

    # Slice local inputs into num_chunks chunks along dim 0
    hs_slices = [
        hidden_states[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ]
    weights_slices = [
        topk_weights[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ] if topk_weights is not None else [None] * num_chunks
    ids_slices = [
        topk_ids[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ] if topk_ids is not None else [None] * num_chunks

    # Prime pipeline with initial all-gather for chunk 0
    ag_hs_curr = _all_gather_activations(collective_group, hs_slices[0])
    ag_tw_curr = (collective_group.all_gather(weights_slices[0], dim=0)
                  if weights_slices[0] is not None else None)
    ag_ti_curr = (collective_group.all_gather(ids_slices[0], dim=0)
                  if ids_slices[0] is not None else None)

    rs_outputs = []

    for i in range(num_chunks):
        # Issue asynchronous all-gather for next chunk if available
        if i + 1 < num_chunks:
            ag_hs_next = _all_gather_activations(collective_group,
                                                 hs_slices[i + 1])
            ag_tw_next = (collective_group.all_gather(weights_slices[i + 1],
                                                      dim=0)
                          if weights_slices[i + 1] is not None else None)
            ag_ti_next = (collective_group.all_gather(ids_slices[i + 1], dim=0)
                          if ids_slices[i + 1] is not None else None)

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

        # Reduce-scatter partial output for current chunk
        rs_i = collective_group.reduce_scatter(out_i, dim=0)
        rs_outputs.append(rs_i)

        if i + 1 < num_chunks:
            ag_hs_curr, ag_tw_curr, ag_ti_curr = ag_hs_next, ag_tw_next, ag_ti_next

    return torch.cat(rs_outputs, dim=0)
