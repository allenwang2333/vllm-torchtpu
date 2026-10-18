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
"""Decode attention that splits the KV read over the TP ranks sharing a KV head.

Under tensor parallelism with fewer KV heads than ranks, every rank of a
KV-head group holds the same KV copy and streams all of it for its own query
heads. With ``TPU_RPA_KV_SPLIT=1`` a decode step on rank ``g`` of a ``G``-rank
group reads only the kernel pages ``p`` with ``p % G == g`` of every sequence,
for the group's query heads gathered to it, and the group merges the partial
softmax results. The two passes and the merge are the DCP machinery of
``cp_attention``: the cached tokens go through the context-parallel kernel with
``cp_group_size=G`` on a rank-local view of the page table, the step's new
tokens through a ``cp_group_size=1`` ``NEW_TOKENS_ONLY`` pass, which also writes
them into every rank's full copy so offload, transfer and prefix reuse keep
seeing a complete KV cache on each rank.

A bucket counts as a decode step when its token count is at most
``max_num_seqs * decode_query_size``; larger buckets (prefill) keep the stock
kernel, whose single pass does not multiply the query-side work by ``G``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.distributed.kv_head_group import get_kv_head_group
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.layers.adapter import cp_attention
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def enabled() -> bool:
    return envs.TPU_RPA_KV_SPLIT


@dataclasses.dataclass
class KVSplitKernels:
    """The kernel pair and group of one attention layer's split decode path."""

    group: Any
    group_size: int
    rank: int
    # CACHE_ONLY over this rank's pages, for the group's gathered query heads.
    cache_kernel: Any
    # NEW_TOKENS_ONLY with cp_group_size 1: writes the step's tokens to their
    # global pages and attends only to them.
    new_kernel: Any
    # Largest token bucket that is a decode step.
    max_tokens: int


def rank_page_view(
    block_tables: torch.Tensor, num_seqs: int, group_size: int, rank: int
) -> torch.Tensor:
    """The kernel pages rank ``rank`` reads.

    Every ``group_size``-th page of each sequence.

    ``block_tables`` is the kernel's flat ``[num_seqs * pages_per_seq]`` table.
    A ``pages_per_seq`` that is not a multiple of ``group_size`` is padded with
    page 0; the context-parallel kernel reads at most
    ``cp_local_cache_len`` tokens of a sequence, which never reaches the pad.
    """
    table = block_tables.view(num_seqs, -1)
    remainder = table.shape[1] % group_size
    if remainder:
        table = torch.nn.functional.pad(table, (0, group_size - remainder))
    return table[:, rank::group_size].contiguous().view(-1)


def build(
    *,
    tp_size: int,
    total_num_kv_heads: int,
    max_num_seqs: int,
    decode_query_size: int,
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    kv_layout: _batched_rpa_configs.KVLayout,
) -> KVSplitKernels | None:
    """The split path's kernels for one layer.

    None when this layout duplicates no KV head.
    """
    group = get_kv_head_group(tp_size, total_num_kv_heads)
    if group is None or int(group.world_size) == 1:
        logger.warning_once(
            "TPU_RPA_KV_SPLIT=1 but no KV head is duplicated across tensor-parallel "
            "ranks (tp_size=%d, total_num_kv_heads=%d); the KV read stays unsplit.",
            tp_size,
            total_num_kv_heads,
        )
        return None
    if sliding_window is not None:
        raise NotImplementedError(
            "TPU_RPA_KV_SPLIT: the context-parallel kernel has no sliding window."
        )
    group_size = int(group.world_size)
    rank = int(group.rank_in_group)
    cache_kernel, _ = cp_attention.build_dcp_kernels(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=group_size,
        cp_rank=rank,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    _, new_kernel = cp_attention.build_dcp_kernels(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=1,
        cp_rank=0,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    max_tokens = max_num_seqs * decode_query_size
    logger.info_once(
        "[Attention] TPU_RPA_KV_SPLIT: decode buckets of <= %d tokens read 1/%d of "
        "the KV copy per rank (rank %d of its KV-head group) and merge partials "
        "over the group.",
        max_tokens,
        group_size,
        rank,
    )
    return KVSplitKernels(
        group=group,
        group_size=group_size,
        rank=rank,
        cache_kernel=cache_kernel,
        new_kernel=new_kernel,
        max_tokens=max_tokens,
    )


def forward(
    kernels: KVSplitKernels,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
) -> torch.Tensor:
    """Split decode attention for this rank's query heads.

    1. Gather the group's query heads, attend to this rank's pages of the cache
       (``CACHE_ONLY``), merge the partials back to own heads.
    2. Attend to the step's new tokens with the local query heads
       (``NEW_TOKENS_ONLY``), which also writes them into the cache.
    3. Merge the two by log-sum-exp.
    """
    group = kernels.group
    seq_lens = attn_metadata.seq_lens
    page_view = rank_page_view(
        attn_metadata.block_tables, seq_lens.shape[0], kernels.group_size, kernels.rank
    )
    query_across_group = group.all_gather(query.contiguous(), dim=1)
    ctx_output, ctx_lse = kernels.cache_kernel(
        kv_cache,
        query_across_group,
        key,
        value,
        seq_lens,
        page_view,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        q_scale,
        k_scale,
        v_scale,
    )
    ctx_output, ctx_lse = cp_lse_ag_out_rs(ctx_output, ctx_lse, group, return_lse=True)

    new_output, new_lse = kernels.new_kernel(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        q_scale,
        k_scale,
        v_scale,
    )
    output, _ = cp_attention.merge_attn_results(
        ctx_output, ctx_lse, new_output, new_lse
    )
    output = output.to(new_output.dtype)
    if not torch.compiler.is_compiling():
        cp_attention.synchronize_tensors(kv_cache, wait=False)
    return output
