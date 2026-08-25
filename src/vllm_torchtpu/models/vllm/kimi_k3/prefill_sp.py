# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental SP4 outer layout for Kimi-K3 long prefill.

The TP ranks are interpreted as ``rank = lane * sp_size + sp_rank``. For
Phase 1 this factors TP8 into ``SP4 x lane2``; Phase 2 uses the same code to
factor TP32 into ``SP4 x lane8``. Attention output reduction is therefore:

* all-reduce ranks with the same ``sp_rank`` along the lane axis;
* reduce-scatter ranks with the same ``lane`` along the SP axis.

The result is one contiguous ``T / 4`` token shard, replicated in every lane.
This module deliberately does not implement hybrid MoE or A2A. The existing
TP/EP feed-forward path is retained behind an explicit AG4 + local reshard as
the Phase 1/2 correctness reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.annotation import TraceAnnotation

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_PHASE_1_2_SP_SIZE = 4
_PHASE_1_2_TOKEN_SIZES = (1024, 2048, 4096)


def _sp_trace(name: str) -> TraceAnnotation:
    return TraceAnnotation(name, compile_safe=True)


@dataclass(frozen=True)
class K3PrefillSPGroupPlan:
    """Global-rank groups for the two axes of the logical SP mesh."""

    tp_ranks: tuple[int, ...]
    sp_size: int
    lane_size: int
    # Same sp_rank, varying lane: attention partial-output all-reduce.
    lane_groups: tuple[tuple[int, ...], ...]
    # Same lane, varying sp_rank: token reduce-scatter/all-gather.
    sp_groups: tuple[tuple[int, ...], ...]


def build_k3_prefill_sp_group_plan(
    tp_ranks: list[int] | tuple[int, ...],
    sp_size: int,
) -> K3PrefillSPGroupPlan:
    """Factor ordered TP ranks into ``SP x lane`` collective groups."""
    ordered_ranks = tuple(int(rank) for rank in tp_ranks)
    if sp_size <= 1:
        raise ValueError(f"K3 prefill SP requires sp_size > 1, got {sp_size}")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError(f"TP ranks must be unique, got {ordered_ranks}")
    if len(ordered_ranks) % sp_size:
        raise ValueError(
            f"TP world size {len(ordered_ranks)} is not divisible by "
            f"K3 prefill sp_size={sp_size}")

    lane_size = len(ordered_ranks) // sp_size
    lane_groups = tuple(
        tuple(ordered_ranks[lane * sp_size + sp_rank]
              for lane in range(lane_size)) for sp_rank in range(sp_size))
    sp_groups = tuple(
        tuple(ordered_ranks[lane * sp_size + sp_rank]
              for sp_rank in range(sp_size)) for lane in range(lane_size))
    return K3PrefillSPGroupPlan(
        tp_ranks=ordered_ranks,
        sp_size=sp_size,
        lane_size=lane_size,
        lane_groups=lane_groups,
        sp_groups=sp_groups,
    )


class K3PrefillSPLayout:
    """Runtime collectives and shard transforms for Phase 1/2 Kimi SP4."""

    def __init__(
        self,
        *,
        plan: K3PrefillSPGroupPlan | None = None,
        tp_group: Any | None = None,
        lane_group: Any | None = None,
        sp_group: Any | None = None,
        lane_rank: int = 0,
        sp_rank: int = 0,
    ) -> None:
        self.plan = plan
        self.tp_group = tp_group
        self.lane_group = lane_group
        self.sp_group = sp_group
        self.lane_rank = lane_rank
        self.sp_rank = sp_rank

    @property
    def enabled(self) -> bool:
        return self.plan is not None

    @property
    def sp_size(self) -> int:
        return 1 if self.plan is None else self.plan.sp_size

    @property
    def lane_size(self) -> int:
        return 1 if self.plan is None else self.plan.lane_size

    @classmethod
    def disabled(cls) -> K3PrefillSPLayout:
        return cls()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        *,
        sp_size: int | None = None,
    ) -> K3PrefillSPLayout:
        """Create both logical-axis groups, or the no-op default layout."""
        configured_size = (envs.VLLM_TORCHTPU_K3_PREFILL_SP_SIZE
                           if sp_size is None else sp_size)
        if configured_size == 1:
            return cls.disabled()
        if configured_size != _PHASE_1_2_SP_SIZE:
            raise ValueError(
                "Kimi-K3 prefill SP Phase 1/2 only supports sp_size=4; "
                f"got {configured_size}")

        parallel_config = vllm_config.parallel_config
        dp_size = int(parallel_config.data_parallel_size)
        pp_size = int(parallel_config.pipeline_parallel_size)
        if dp_size != 1 or pp_size != 1:
            raise ValueError(
                "Kimi-K3 prefill SP Phase 1/2 requires DP1/PP1, got "
                f"DP{dp_size}/PP{pp_size}")

        from vllm.distributed.parallel_state import (get_tp_group,
                                                     get_world_group,
                                                     init_model_parallel_group)

        tp_group = get_tp_group()
        plan = build_k3_prefill_sp_group_plan(tp_group.ranks, configured_size)
        configured_tp_size = int(parallel_config.tensor_parallel_size)
        if configured_tp_size != len(plan.tp_ranks):
            raise RuntimeError(
                "Kimi-K3 prefill SP saw inconsistent TP sizes: config="
                f"{configured_tp_size}, group={len(plan.tp_ranks)}")

        world_group = get_world_group()
        backend = tp_group.torch_distributed_backend
        lane_group = init_model_parallel_group(
            [list(group) for group in plan.lane_groups],
            world_group.local_rank,
            backend,
            group_name="k3_prefill_sp_lane",
        )
        sp_group = init_model_parallel_group(
            [list(group) for group in plan.sp_groups],
            world_group.local_rank,
            backend,
            group_name="k3_prefill_sp_axis",
        )

        tp_rank = int(tp_group.rank_in_group)
        lane_rank, sp_rank = divmod(tp_rank, configured_size)
        expected_lane_group = plan.lane_groups[sp_rank]
        expected_sp_group = plan.sp_groups[lane_rank]
        if tuple(lane_group.ranks) != expected_lane_group:
            raise RuntimeError(
                "Kimi-K3 lane group order mismatch: expected "
                f"{expected_lane_group}, got {tuple(lane_group.ranks)}")
        if tuple(sp_group.ranks) != expected_sp_group:
            raise RuntimeError(
                "Kimi-K3 SP group order mismatch: expected "
                f"{expected_sp_group}, got {tuple(sp_group.ranks)}")

        # Preserve record_function enter/exit nodes through Dynamo fullgraph
        # capture. This is opt-in with a valid K3 SP mesh so ordinary serving
        # graphs and the default TraceAnnotation behavior remain unchanged.
        torch._dynamo.config.capture_profiler_record_function = True

        logger.info(
            "Kimi-K3 prefill SP enabled | mapping: tp_rank = lane * %d + "
            "sp_rank | TP=%d SP=%d lanes=%d lane_groups=%s sp_groups=%s | "
            "this rank: tp=%d lane=%d sp=%d",
            configured_size,
            len(plan.tp_ranks),
            configured_size,
            plan.lane_size,
            plan.lane_groups,
            plan.sp_groups,
            tp_rank,
            lane_rank,
            sp_rank,
        )
        return cls(
            plan=plan,
            tp_group=tp_group,
            lane_group=lane_group,
            sp_group=sp_group,
            lane_rank=lane_rank,
            sp_rank=sp_rank,
        )

    def should_shard(self, hidden_states: torch.Tensor) -> bool:
        """Whether this physical token bucket uses the experimental layout."""
        return (self.enabled
                and hidden_states.shape[0] in _PHASE_1_2_TOKEN_SIZES)

    def _local_shard(self, full_hidden_states: torch.Tensor) -> torch.Tensor:
        if full_hidden_states.shape[0] % self.sp_size:
            raise ValueError(
                f"Token count {full_hidden_states.shape[0]} is not divisible "
                f"by Kimi-K3 prefill sp_size={self.sp_size}")
        local_tokens = full_hidden_states.shape[0] // self.sp_size
        shard = full_hidden_states.narrow(0, self.sp_rank * local_tokens,
                                          local_tokens)
        # Break the full tensor's view lineage before the shard crosses a TPU
        # custom-op boundary. This also makes the local tensor contract visible
        # in XLA instead of retaining a full-hidden backing allocation.
        materialized = torch.empty_like(shard)
        materialized.copy_(shard)
        return materialized

    def shard_embeddings(self,
                         full_hidden_states: torch.Tensor) -> torch.Tensor:
        with _sp_trace("embedding_sp_reshard"):
            return self._local_shard(full_hidden_states)

    def gather_next_attention(
            self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.sp_group is not None
        with _sp_trace("next_attention_sp_ag"):
            return self.sp_group.all_gather(local_hidden_states, dim=0)

    def reduce_attention(self,
                         partial_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lane_group is not None
        assert self.sp_group is not None
        with _sp_trace("attention_lane_ar"):
            lane_reduced = self.lane_group.all_reduce(partial_hidden_states)
        with _sp_trace("attention_sp_rs"):
            return self.sp_group.reduce_scatter(lane_reduced, dim=0)

    def finish_standard_attention(
            self, partial_hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore the ordinary TP reduction for decode/non-SP buckets."""
        if not self.enabled:
            return partial_hidden_states
        assert self.tp_group is not None
        with _sp_trace("attention_tp_ar_fallback"):
            return self.tp_group.all_reduce(partial_hidden_states)

    def gather_moe_compat(self,
                          local_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.sp_group is not None
        with _sp_trace("moe_compat_ag"):
            return self.sp_group.all_gather(local_hidden_states, dim=0)

    def reshard_moe_compat(self,
                           full_hidden_states: torch.Tensor) -> torch.Tensor:
        with _sp_trace("moe_compat_reshard"):
            return self._local_shard(full_hidden_states)

    def gather_final(self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.sp_group is not None
        with _sp_trace("final_sp_ag"):
            return self.sp_group.all_gather(local_hidden_states, dim=0)
