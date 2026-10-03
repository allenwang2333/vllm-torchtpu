# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical MoE parallelism: experts split EP between groups, TP within one.

Two ways to ask for it:

- ``TPU_MOE_HIERARCHICAL_EP``: EP between chips, TP between the cores of a
  chip. Model code opts in by building its experts inside
  ``hierarchical_moe_parallel_config()`` (Kimi K3).
- ``TPU_MOE_EP_TP_SIZE=N``: EP between groups of N consecutive ranks, TP
  inside a group. Applied to every ``FusedMoE`` the process builds by
  ``install_ep_tp_split_patch`` (a registry patch), so upstream models such as
  DeepSeek-V2/V3 and GLM-5 take it without model code.
"""

from __future__ import annotations

import contextlib
import dataclasses

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.chip_topology import hierarchical_moe_split
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def hierarchical_split_or_none() -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)``, or ``None`` to stay flat.

    ``None`` is the safe answer everywhere: the caller keeps vLLM's own
    parallel config and the model runs exactly as it does today.
    """
    ep_tp = envs.TPU_MOE_EP_TP_SIZE
    if ep_tp > 1 and envs.TPU_MOE_HIERARCHICAL_EP:
        raise ValueError(
            "TPU_MOE_EP_TP_SIZE and TPU_MOE_HIERARCHICAL_EP both pick the MoE "
            "expert split; set one."
        )
    if ep_tp <= 1 and not envs.TPU_MOE_HIERARCHICAL_EP:
        return None

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    if world_size <= 1:
        return None

    if ep_tp > 1:
        return ep_tp_split(world_size, rank, ep_tp)

    split = hierarchical_moe_split(world_size, rank)
    if split is None:
        logger.warning_once(
            "TPU_MOE_HIERARCHICAL_EP is set but the chip topology could not be "
            "used; falling back to flat expert parallelism."
        )
        return None

    return split


def ep_tp_split(
    world_size: int, rank: int, tp_size: int
) -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)`` for EP between groups of
    ``tp_size`` consecutive ranks and TP inside a group, or ``None`` when
    ``tp_size`` does not divide ``world_size`` (stay flat)."""
    if tp_size <= 1 or world_size % tp_size != 0:
        logger.warning_once(
            "TPU_MOE_EP_TP_SIZE=%d does not divide the TP world size %d; "
            "falling back to flat expert parallelism.",
            tp_size,
            world_size,
        )
        return None
    return (world_size // tp_size, rank // tp_size, tp_size, rank % tp_size)


def _split_parallel_config(config, split):
    """Rewrite one ``FusedMoEParallelConfig`` to the split; flat stays flat."""
    if not config.use_ep:
        # Expert parallelism is off entirely (no --enable-expert-parallel);
        # there is no expert split to make hierarchical, and forcing one
        # would silently change what the flat TP config means.
        logger.warning_once(
            "A hierarchical MoE split is set but expert parallelism is off; "
            "leaving the MoE parallel config alone."
        )
        return config
    ep_size, ep_rank, tp_size, tp_rank = split
    return dataclasses.replace(
        config, ep_size=ep_size, ep_rank=ep_rank, tp_size=tp_size, tp_rank=tp_rank
    )


_filter_patched = False


def align_ep_weight_filter() -> None:
    """Teach ``--enable-ep-weight-filter`` the chip-block expert ownership."""
    global _filter_patched
    if _filter_patched:
        return

    from vllm.model_executor.model_loader import default_loader

    original = default_loader.compute_local_expert_ids

    def group_aware(num_experts, ep_size, ep_rank, placement="linear"):
        # Read at load time, not patch time: the registry installs this before
        # the TP group exists. Ignore the caller's rank-derived ownership; the
        # experts this rank needs are its group's block, shared with the other
        # ranks of the group.
        split = hierarchical_split_or_none()
        if split is not None:
            ep_size, ep_rank = split[0], split[1]
        return original(num_experts, ep_size, ep_rank, placement=placement)

    default_loader.compute_local_expert_ids = group_aware
    _filter_patched = True
    logger.info("EP weight filter realigned to the hierarchical MoE expert split.")


@contextlib.contextmanager
def hierarchical_moe_parallel_config():
    """Make ``FusedMoE`` built inside this context manager use the chip-aware split."""
    split = hierarchical_split_or_none()
    if split is None:
        yield None
        return

    ep_size, ep_rank, tp_size, tp_rank = split

    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    original_make = FusedMoEParallelConfig.make

    def patched_make(*args, **kwargs):
        return _split_parallel_config(original_make(*args, **kwargs), split)

    # Safe here: the loader calls _init_ep_weight_filter from load_weights,
    # which runs after the model is constructed, so this lands in time.
    align_ep_weight_filter()

    logger.info_once(
        "Hierarchical MoE parallelism: EP between chips (ep_size=%d, "
        "ep_rank=%d), TP within a chip (tp_size=%d, tp_rank=%d)",
        ep_size,
        ep_rank,
        tp_size,
        tp_rank,
    )
    FusedMoEParallelConfig.make = staticmethod(patched_make)
    try:
        yield split
    finally:
        FusedMoEParallelConfig.make = original_make


def install_ep_tp_split_patch() -> None:
    """Apply ``TPU_MOE_EP_TP_SIZE`` to every ``FusedMoE`` this process builds.

    A registry patch (worker init). Inert unless the env is set. The split is
    computed at each ``make`` call, after the TP group exists; a world size the
    env does not divide stays flat.
    """
    if envs.TPU_MOE_EP_TP_SIZE <= 1:
        return

    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False):
        return
    original_make = FusedMoEParallelConfig.make

    def ep_tp_make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        split = hierarchical_split_or_none()
        if split is None:
            return config
        logger.info_once(
            "MoE experts split EP %d x TP %d (TPU_MOE_EP_TP_SIZE): this rank is "
            "ep_rank=%d, tp_rank=%d.",
            split[0],
            split[2],
            split[1],
            split[3],
        )
        return _split_parallel_config(config, split)

    FusedMoEParallelConfig.make = staticmethod(ep_tp_make)
    FusedMoEParallelConfig._tpu_ep_tp_split_patch = True
    align_ep_weight_filter()
    logger.info("Applied TPU patch: MoE EP x TP expert split (TPU_MOE_EP_TP_SIZE).")
