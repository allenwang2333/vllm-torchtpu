# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured tail heights for fused expert-parallel MoE workloads."""

from typing import NamedTuple

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class HeightKey(NamedTuple):
    generation: int
    ep: int
    local_experts: int
    hidden: int
    intermediate: int
    capacity: int
    ragged_rows: int
    weight_format: str
    scale_block: int
    activation: str
    has_bias: bool


# v7x-8, 16384 global tokens: 1.19133 -> 1.12652 ms, identical outputs.
_HEIGHTS = {
    HeightKey(
        7,
        8,
        48,
        5120,
        2560,
        128,
        98816,
        "fp4",
        512,
        "silu_and_mul_with_clamp",
        False,
    ): (32, 64, 96, 128),
}


def get_height_rungs(key: HeightKey) -> tuple[int, ...] | None:
    rungs = _HEIGHTS.get(key)
    if rungs is None:
        logger.warning_once("fused EP MoE: no tuned prefill heights for %s", key)
    return rungs
