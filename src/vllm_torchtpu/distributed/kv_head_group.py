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
"""Process group over the tensor-parallel ranks that hold the same KV head.

With fewer KV heads than tensor-parallel ranks, vLLM's ``QKVParallelLinear``
gives tensor-parallel rank ``r`` KV head ``r // replicas`` with ``replicas =
tp_size // total_num_kv_heads``, so the ranks that share a KV head are a
contiguous run of ``replicas`` ranks of the TP group. Tensor parallelism is
the fastest-varying axis of vLLM's rank grid, so a TP group is itself a
contiguous run of ``tp_size`` global ranks and the KV-head groups are the
runs of ``replicas`` global ranks aligned to ``replicas``.
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_group: Any | None = None
_initialized = False


def kv_head_group_ranks(
    world_size: int, tp_size: int, total_num_kv_heads: int
) -> list[list[int]] | None:
    """Global ranks grouped by the KV head they hold.

    None when no rank duplicates a KV head.
    """
    if total_num_kv_heads <= 0 or tp_size <= total_num_kv_heads:
        return None
    if tp_size % total_num_kv_heads or world_size % tp_size:
        return None
    replicas = tp_size // total_num_kv_heads
    return [
        list(range(start, start + replicas)) for start in range(0, world_size, replicas)
    ]


def get_kv_head_group(tp_size: int, total_num_kv_heads: int) -> Any | None:
    """The ``GroupCoordinator`` over the ranks holding this rank's KV head, or None.

    Built once, on first call. Every rank must reach this at the same point in
    model construction, since creating a process group is collective; that
    holds because all ranks build the same layers in the same order.
    """
    global _group, _initialized
    if _initialized:
        return _group
    _initialized = True

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    from vllm.distributed.parallel_state import (
        get_tp_group,
        get_world_group,
        init_model_parallel_group,
    )

    world_size = dist.get_world_size()
    groups = kv_head_group_ranks(world_size, tp_size, total_num_kv_heads)
    if groups is None:
        return None
    tp_ranks = tuple(int(r) for r in get_tp_group().ranks)
    if tp_ranks != tuple(range(tp_ranks[0], tp_ranks[0] + tp_size)) or (
        tp_ranks[0] % tp_size
    ):
        logger.warning(
            "TP group %s is not a contiguous run aligned to tp_size=%d; the KV-head "
            "groups cannot be derived from the rank index, leaving the KV read "
            "unsplit.",
            tp_ranks,
            tp_size,
        )
        return None

    world = get_world_group()
    backend = dist.get_backend(world.device_group)
    _group = init_model_parallel_group(
        groups,
        world.local_rank,
        backend,
        group_name="kv_head",
    )
    logger.info(
        "KV-head group ready: %d group(s) of %d rank(s); this rank is %d/%d within "
        "its group (tp_size=%d, total_num_kv_heads=%d).",
        len(groups),
        len(groups[0]),
        _group.rank_in_group,
        _group.world_size,
        tp_size,
        total_num_kv_heads,
    )
    return _group
