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

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_PHASE_1_2_SP_SIZE = 4
_PHASE_1_2_TOKEN_SIZES = (1024, 2048, 4096)


def _sp_trace(name: str) -> Any:
    # Keep this as the native context manager. Dynamo recognizes the WITH
    # bytecode for record_function and, with capture enabled below, emits the
    # profiler enter/exit ops into the compiled graph. Calling __enter__ from a
    # wrapper context is unsupported in full-graph mode.
    return torch.profiler.record_function(name)


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

        # Dynamo captures the model with the full token dimension symbolic,
        # but TorchTPU's FX exporter requires the token dimension passed to
        # downstream linear reshapes to be concrete.  Make the three Phase
        # 1/2 buckets structurally explicit here. ``narrow`` with a literal
        # length gives the shard a concrete first dimension. Range guards,
        # rather than equality guards, remain valid for a dimension declared
        # dynamic; the existing shape-variant warmup consumes them and
        # retraces once per incompatible compile bucket before serving starts.
        if full_hidden_states.shape[0] <= 1024:
            local_tokens = 256
        elif full_hidden_states.shape[0] <= 2048:
            local_tokens = 512
        else:
            local_tokens = 1024

        # Take the latter half from the end.  For sp_rank=3, a positive
        # ``3 * local_tokens`` start plus ``local_tokens`` length would add
        # ``T >= 4 * local_tokens`` to the range guard above. Together those
        # constraints specialize a strict-dynamic T to the bucket's exact
        # value on ranks 3/7. Negative offsets preserve the identical ordered
        # shard while requiring only the non-specializing lower bound.
        shard_start = (self.sp_rank * local_tokens if self.sp_rank < 2 else
                       -(self.sp_size - self.sp_rank) * local_tokens)
        shard = full_hidden_states.narrow(0, shard_start, local_tokens)
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

    @staticmethod
    def _all_gather_dim0(group: Any, input_: torch.Tensor) -> torch.Tensor:
        # vLLM's generic communicator passes ``world_size * SymInt`` to
        # torch.empty. TorchTPU's FX exporter requires concrete shape lists,
        # so derive the output from local-shape empty_like tensors instead.
        output = torch.cat(
            [torch.empty_like(input_) for _ in range(group.world_size)],
            dim=0,
        )
        torch.distributed.all_gather_into_tensor(
            output,
            input_,
            group=group.device_group,
        )
        return output

    @staticmethod
    def _reduce_scatter_dim0(group: Any, input_: torch.Tensor) -> torch.Tensor:
        contiguous = input_.contiguous()
        # unflatten uses only constants (world_size, -1), while preserving all
        # trailing dimensions. empty_like then inherits the symbolic shard
        # dimension without constructing a SymIntArrayRef.
        shard_prototype = contiguous.unflatten(0,
                                               (group.world_size, -1)).select(
                                                   0, 0)
        output = torch.empty_like(shard_prototype)
        torch.distributed.reduce_scatter_tensor(
            output,
            contiguous,
            group=group.device_group,
        )
        return output

    def gather_next_attention(
            self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.sp_group is not None
        with _sp_trace("next_attention_sp_ag"):
            return self._all_gather_dim0(self.sp_group, local_hidden_states)

    def reduce_attention(self,
                         partial_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lane_group is not None
        assert self.sp_group is not None
        output_dtype = partial_hidden_states.dtype
        # A single TP all-reduce materializes only its final result. Splitting
        # the same sum into lane-AR followed by SP-RS would otherwise
        # materialize (and round) the intermediate lane sum in BF16 before the
        # second collective. Keep both stages in FP32 so the SP decomposition
        # adds no extra low-precision rounding point, then restore the model's
        # activation dtype after the complete TP sum has been formed.
        accumulator = (partial_hidden_states.float()
                       if output_dtype in (torch.bfloat16, torch.float16) else
                       partial_hidden_states)
        with _sp_trace("attention_lane_ar"):
            lane_reduced = self.lane_group.all_reduce(accumulator)
        with _sp_trace("attention_sp_rs"):
            reduced_shard = self._reduce_scatter_dim0(
                self.sp_group, lane_reduced)
        return reduced_shard.to(output_dtype)

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
            return self._all_gather_dim0(self.sp_group, local_hidden_states)

    def reshard_moe_compat(self,
                           full_hidden_states: torch.Tensor) -> torch.Tensor:
        with _sp_trace("moe_compat_reshard"):
            return self._local_shard(full_hidden_states)

    def gather_final(self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.sp_group is not None
        with _sp_trace("final_sp_ag"):
            return self._all_gather_dim0(self.sp_group, local_hidden_states)
