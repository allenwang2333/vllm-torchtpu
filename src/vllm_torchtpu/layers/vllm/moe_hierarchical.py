# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical MoE parallelism: EP between chips, TP within a chip."""

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
    if not envs.TPU_MOE_HIERARCHICAL_EP:
        return None

    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size,
                                  get_world_group)

    vllm_config = get_current_vllm_config_or_none()
    use_ep = (vllm_config is not None
              and vllm_config.parallel_config.enable_expert_parallel)
    if use_ep:
        # With expert parallelism the expert space is flattened across DP:
        # FusedMoEParallelConfig.make sets ep_size = dp*pcp*tp and
        # ep_rank = dp_rank*pcp*tp + ..., which is exactly the world group.
        # Use it rather than the TP group so the chip split still sees every
        # device under data-parallel attention (e.g. DP=4 x TP=8, where the
        # TP group alone holds only 8 of the 32 devices and the split would
        # refuse to engage).
        world_size = get_world_group().world_size
        rank = get_world_group().rank_in_group
    else:
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
    if world_size <= 1:
        return None

    split = hierarchical_moe_split(world_size, rank)
    if split is None:
        logger.warning_once(
            "TPU_MOE_HIERARCHICAL_EP is set but the chip topology could not be "
            "used; falling back to flat expert parallelism.")
        return None

    return split


_filter_patched = False


def align_ep_weight_filter() -> None:
    """Teach ``--enable-ep-weight-filter`` the chip-block expert ownership."""
    global _filter_patched
    if _filter_patched:
        return

    split = hierarchical_split_or_none()
    if split is None:
        return
    ep_size, ep_rank, _, _ = split

    from vllm.model_executor.model_loader import default_loader

    original = default_loader.compute_local_expert_ids

    def chip_aware(num_experts, _ep_size, _ep_rank, placement="linear"):
        # Ignore the caller's rank-derived ownership; the experts this rank
        # needs are its chip's block, shared with the other chiplet.
        return original(num_experts, ep_size, ep_rank, placement=placement)

    default_loader.compute_local_expert_ids = chip_aware
    _filter_patched = True
    logger.info(
        "EP weight filter realigned to chip-block ownership: ep_size=%d, "
        "ep_rank=%d (was rank-derived).", ep_size, ep_rank)


@contextlib.contextmanager
def hierarchical_moe_parallel_config():
    """Make ``FusedMoE`` built inside this context manager use the chip-aware split."""
    split = hierarchical_split_or_none()
    if split is None:
        yield None
        return

    ep_size, ep_rank, tp_size, tp_rank = split

    from vllm.model_executor.layers.fused_moe.config import \
        FusedMoEParallelConfig

    original_make = FusedMoEParallelConfig.make

    def patched_make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        if not config.use_ep:
            # Expert parallelism is off entirely (no --enable-expert-parallel);
            # there is no expert split to make hierarchical, and forcing one
            # would silently change what the flat TP config means.
            logger.warning_once(
                "TPU_MOE_HIERARCHICAL_EP is set but expert parallelism "
                "is off; leaving the MoE parallel config alone.")
            return config
        return dataclasses.replace(config,
                                   ep_size=ep_size,
                                   ep_rank=ep_rank,
                                   tp_size=tp_size,
                                   tp_rank=tp_rank)

    # Safe here: the loader calls _init_ep_weight_filter from load_weights,
    # which runs after the model is constructed, so this lands in time.
    align_ep_weight_filter()

    logger.info_once(
        "Hierarchical MoE parallelism: EP between chips (ep_size=%d, "
        "ep_rank=%d), TP within a chip (tp_size=%d, tp_rank=%d)", ep_size,
        ep_rank, tp_size, tp_rank)
    FusedMoEParallelConfig.make = staticmethod(patched_make)
    try:
        yield split
    finally:
        FusedMoEParallelConfig.make = original_make
