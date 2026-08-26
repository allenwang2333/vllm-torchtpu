# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared runtime support for model-level sequence parallelism."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SEQUENCE_PARALLEL_CONFIG_KEY = "sequence_parallel_size"


def _positive_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return value


def _configured_parallel_size(parallel_config: Any) -> int | None:
    try:
        attributes = vars(parallel_config)
    except TypeError:
        attributes = {}
    if SEQUENCE_PARALLEL_CONFIG_KEY not in attributes:
        return None
    return _positive_int(
        attributes[SEQUENCE_PARALLEL_CONFIG_KEY],
        source=f"ParallelConfig.{SEQUENCE_PARALLEL_CONFIG_KEY}",
    )


def resolve_sequence_parallel_size(vllm_config: Any) -> int:
    """Resolve SP without requiring a patched vLLM config schema."""
    parallel_size = _configured_parallel_size(vllm_config.parallel_config)

    additional_size: int | None = None
    additional_config = getattr(vllm_config, "additional_config", None)
    if (isinstance(additional_config, Mapping)
            and SEQUENCE_PARALLEL_CONFIG_KEY in additional_config):
        additional_size = _positive_int(
            additional_config[SEQUENCE_PARALLEL_CONFIG_KEY],
            source=(f"additional_config[{SEQUENCE_PARALLEL_CONFIG_KEY!r}]"),
        )

    if (parallel_size not in (None, 1) and additional_size not in (None, 1)
            and parallel_size != additional_size):
        raise ValueError("Conflicting sequence parallel sizes: "
                         f"ParallelConfig={parallel_size}, additional_config="
                         f"{additional_size}")

    if parallel_size not in (None, 1):
        return parallel_size
    if additional_size is not None:
        return additional_size
    return parallel_size if parallel_size is not None else 1


@dataclass(frozen=True)
class SequenceParallelGroupPlan:
    """Global-rank groups for a logical ``lane x sequence`` TP mesh."""

    tp_ranks: tuple[int, ...]
    sequence_parallel_size: int
    lane_size: int
    # Keeps token shards fixed while reducing weight-shard partials.
    lane_groups: tuple[tuple[int, ...], ...]
    # Keeps weight shards fixed while moving token shards.
    sequence_groups: tuple[tuple[int, ...], ...]


def build_sequence_parallel_group_plan(
    tp_ranks: Sequence[int],
    sequence_parallel_size: int,
) -> SequenceParallelGroupPlan:
    """Factor ordered TP ranks into logical lane and sequence axes."""
    sp_size = _positive_int(
        sequence_parallel_size,
        source="sequence_parallel_size",
    )
    if sp_size == 1:
        raise ValueError("sequence_parallel_size must be greater than 1")

    ordered_ranks = tuple(int(rank) for rank in tp_ranks)
    if not ordered_ranks:
        raise ValueError("TP ranks must not be empty")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError(f"TP ranks must be unique, got {ordered_ranks}")
    if len(ordered_ranks) % sp_size:
        raise ValueError(
            f"TP world size {len(ordered_ranks)} is not divisible by "
            f"sequence_parallel_size={sp_size}")

    lane_size = len(ordered_ranks) // sp_size
    lane_groups = tuple(
        tuple(ordered_ranks[lane * sp_size + sequence_rank]
              for lane in range(lane_size))
        for sequence_rank in range(sp_size))
    sequence_groups = tuple(
        tuple(ordered_ranks[lane * sp_size + sequence_rank]
              for sequence_rank in range(sp_size))
        for lane in range(lane_size))
    return SequenceParallelGroupPlan(
        tp_ranks=ordered_ranks,
        sequence_parallel_size=sp_size,
        lane_size=lane_size,
        lane_groups=lane_groups,
        sequence_groups=sequence_groups,
    )


@dataclass(frozen=True)
class SequenceParallelMesh:
    """TP subgroups and token collectives shared by model adapters."""

    plan: SequenceParallelGroupPlan
    tp_group: Any
    lane_group: Any
    sequence_group: Any
    lane_rank: int
    sequence_rank: int

    @property
    def sequence_parallel_size(self) -> int:
        return self.plan.sequence_parallel_size

    @classmethod
    def from_tensor_parallel_group(
        cls,
        sequence_parallel_size: int,
    ) -> SequenceParallelMesh:
        from vllm.distributed.parallel_state import (get_tp_group,
                                                     get_world_group,
                                                     init_model_parallel_group)

        tp_group = get_tp_group()
        plan = build_sequence_parallel_group_plan(
            tp_group.ranks,
            sequence_parallel_size,
        )
        world_group = get_world_group()
        backend = tp_group.torch_distributed_backend
        lane_group = init_model_parallel_group(
            [list(group) for group in plan.lane_groups],
            world_group.local_rank,
            backend,
            group_name="sequence_parallel_lane",
        )
        sequence_group = init_model_parallel_group(
            [list(group) for group in plan.sequence_groups],
            world_group.local_rank,
            backend,
            group_name="sequence_parallel_axis",
        )

        tp_rank = int(tp_group.rank_in_group)
        lane_rank, sequence_rank = divmod(tp_rank, sequence_parallel_size)
        expected_lane_group = plan.lane_groups[sequence_rank]
        expected_sequence_group = plan.sequence_groups[lane_rank]
        if tuple(lane_group.ranks) != expected_lane_group:
            raise RuntimeError(
                "Sequence-parallel lane group order mismatch: expected "
                f"{expected_lane_group}, got {tuple(lane_group.ranks)}")
        if tuple(sequence_group.ranks) != expected_sequence_group:
            raise RuntimeError(
                "Sequence-parallel group order mismatch: expected "
                f"{expected_sequence_group}, got "
                f"{tuple(sequence_group.ranks)}")

        logger.info(
            "Sequence parallelism enabled | TP=%d SP=%d lanes=%d | "
            "tp_rank=%d lane_rank=%d sequence_rank=%d",
            len(plan.tp_ranks),
            plan.sequence_parallel_size,
            plan.lane_size,
            tp_rank,
            lane_rank,
            sequence_rank,
        )
        return cls(
            plan=plan,
            tp_group=tp_group,
            lane_group=lane_group,
            sequence_group=sequence_group,
            lane_rank=lane_rank,
            sequence_rank=sequence_rank,
        )

    @staticmethod
    def _all_gather_dim0(group: Any,
                         local_tensor: torch.Tensor) -> torch.Tensor:
        # Tensor-derived buffers avoid symbolic shape lists in TPU export.
        output = torch.cat(
            [torch.empty_like(local_tensor) for _ in range(group.world_size)],
            dim=0,
        )
        torch.distributed.all_gather_into_tensor(
            output,
            local_tensor,
            group=group.device_group,
        )
        return output

    @staticmethod
    def _reduce_scatter_dim0(group: Any, tensor: torch.Tensor) -> torch.Tensor:
        contiguous = tensor.contiguous()
        shard = contiguous.unflatten(0, (group.world_size, -1)).select(0, 0)
        output = torch.empty_like(shard)
        torch.distributed.reduce_scatter_tensor(
            output,
            contiguous,
            group=group.device_group,
        )
        return output

    def shard_tokens(
        self,
        tensor: torch.Tensor,
        *,
        full_token_sizes: Sequence[int],
    ) -> torch.Tensor:
        """Return this rank's contiguous token shard."""
        ordered_token_sizes = tuple(sorted(set(full_token_sizes)))
        if not ordered_token_sizes:
            raise ValueError(
                "Sequence-parallel token buckets must not be empty")
        for full_token_count in ordered_token_sizes:
            if full_token_count % self.sequence_parallel_size:
                raise ValueError(
                    f"Token bucket {full_token_count} is not divisible by "
                    "sequence_parallel_size="
                    f"{self.sequence_parallel_size}")

        if (not torch.compiler.is_compiling()
                and tensor.shape[0] not in ordered_token_sizes):
            raise ValueError(
                f"Token count {tensor.shape[0]} is not configured for "
                "sequence parallelism")

        # Bucket guards keep exported shapes concrete without specializing T.
        local_token_count = (ordered_token_sizes[-1] //
                             self.sequence_parallel_size)
        for full_token_count in ordered_token_sizes[:-1]:
            if tensor.shape[0] <= full_token_count:
                local_token_count = (full_token_count //
                                     self.sequence_parallel_size)
                break

        # Negative upper-rank offsets avoid a full-length lower-bound guard.
        midpoint = self.sequence_parallel_size // 2
        shard_start = (self.sequence_rank *
                       local_token_count if self.sequence_rank < midpoint else
                       -(self.sequence_parallel_size - self.sequence_rank) *
                       local_token_count)
        shard = tensor.narrow(0, shard_start, local_token_count)
        materialized = torch.empty_like(shard)
        materialized.copy_(shard)
        return materialized

    def all_gather_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._all_gather_dim0(self.sequence_group, tensor)

    def reduce_scatter_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._reduce_scatter_dim0(self.sequence_group, tensor)

    def reduce_attention_output(self, tensor: torch.Tensor) -> torch.Tensor:
        lane_reduced = self.lane_group.all_reduce(tensor)
        return self.reduce_scatter_tokens(lane_reduced)
