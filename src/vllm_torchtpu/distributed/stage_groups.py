# SPDX-License-Identifier: Apache-2.0
"""Standalone process groups for multi-device pipeline stages.

By default torch_tpu compiles the collectives of a process subgroup into
programs for every device of the slice, which every rank must launch together
in the same order. Pipeline stages run their forwards at different times, and
some collectives are launched by one stage only (the last stage's vocab
all-gather for the logits), so a stage's collectives would wait for devices of
the other stages and hang at the first uneven launch.

A *standalone* subgroup (google-pytorch/torch_tpu#4387,
`torch_tpu._internal.distributed.spmd_util.make_standalone`) compiles its
collectives for its own devices only, so the stage launches them alone. The
pipeline hand-off (`pp_shift`) stays one collective over the whole slice.

Mirrors sglang-torchtpu#827's `make_stage_groups_standalone`.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DONE_ATTR = "_tpu_stage_groups_standalone"


def _stage_groups() -> list[Any]:
    """The distinct model-parallel groups of more than one device inside this
    rank's tensor-parallel pipeline stage, in the same order on every rank.

    The PCP group is deliberately left alone: PP x PCP stages (#1346) move
    K/V through shard_map meshes and run on stock torch_tpu.
    """
    from vllm.distributed import parallel_state

    candidates: list[Any] = [parallel_state.get_tp_group()]
    for getter in ("get_ep_group", "get_dp_group"):
        fn = getattr(parallel_state, getter, None)
        if fn is None:
            continue
        try:
            candidates.append(fn())
        except (AssertionError, AttributeError):
            continue
    groups: list[Any] = []
    for group in candidates:
        if group is None or group.world_size <= 1:
            continue
        if any(group is seen for seen in groups):
            continue
        # Two coordinators over identical ranks (e.g. TP and EP with DP=1)
        # back the same device set; one call per distinct rank list.
        if any(tuple(group.ranks) == tuple(seen.ranks) for seen in groups):
            continue
        groups.append(group)
    return groups


def _load_make_standalone() -> Callable[[Any], None] | None:
    import os

    mode = os.environ.get("VLLM_PP_STAGE_STANDALONE", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        logger.info("VLLM_PP_STAGE_STANDALONE=%s: skipping make_standalone", mode)
        return None
    try:
        from torch_tpu._internal.distributed.spmd_util import make_standalone
    except ImportError as e:
        if mode in ("1", "true", "yes", "on"):
            raise RuntimeError(
                "Pipeline stages of more than one device (--pipeline-parallel-size "
                "> 1 with --tensor-parallel-size > 1) need a torch_tpu with "
                "standalone process subgroups (google-pytorch/torch_tpu#4387); the "
                "installed torch_tpu has no "
                "torch_tpu._internal.distributed.spmd_util.make_standalone. Without "
                "it the stage's collectives compile slice-wide and the first uneven "
                "launch (the last stage's logits all-gather) hangs."
            ) from e
        logger.info(
            "torch_tpu has no make_standalone; relying on process-group "
            "rank_to_device_id subgroup collectives (google-pytorch/torch_tpu#4178)"
        )
        return None
    return make_standalone


def make_stage_groups_standalone(parallel_config: Any) -> None:
    """Let every tensor-parallel pipeline stage launch its collectives without
    the other stages. Must run after the model-parallel groups exist and
    before their first collective. Nothing to do for one-device stages or a
    single stage. Idempotent per process."""
    pp = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
    tp = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    if pp == 1 or tp == 1:
        return
    from vllm.distributed import parallel_state

    pp_group = parallel_state.get_pp_group()
    if getattr(pp_group, _DONE_ATTR, False):
        return
    make_standalone = _load_make_standalone()
    groups = _stage_groups()
    if make_standalone is not None:
        for group in groups:
            # Returns once every member of the group has called it.
            make_standalone(group.device_group)
    setattr(pp_group, _DONE_ATTR, True)
    logger.info(
        "pipeline stage %d: standalone process groups %s (make_standalone=%s)",
        pp_group.rank_in_group,
        [(getattr(g, "unique_name", "?"), list(g.ranks)) for g in groups],
        make_standalone is not None,
    )
