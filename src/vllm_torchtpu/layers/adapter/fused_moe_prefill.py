# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serve a MoE layer's prefill steps with the fused MoE prefill kernel.

With ``FUSED_MOE_PREFILL_KERNEL=1`` the layers the fused MoE decode kernel is
eligible for (TP experts, or EP with TP attention) are armed, and their steps
of at least ``FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS`` tokens run
``fused_moe_prefill`` per rank under shard_map: routed + shared experts, the
combine and the all-reduce. Smaller steps run the decode kernel when
``FUSED_MOE_DECODE_KERNEL=1`` and its cap allows, else the GMM MoE path.

Arming, the exported op and the GMM fallback are shared with the decode kernel
in ``fused_moe_decode.py``: one exported program per layer serves every token
bucket, and the path is picked inside it, where the token count is static.
This module is the prefill path that program calls.
"""

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill as _kernel_body,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def prefill_min_tokens() -> int | None:
    """The smallest step the prefill kernel serves, or None when it is off."""
    if not envs.FUSED_MOE_PREFILL_KERNEL:
        return None
    min_tokens = envs.FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS
    if min_tokens < 1:
        raise ValueError(
            f"FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS={min_tokens}: must be >= 1."
        )
    return min_tokens


def prefill_layer(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    start,
):
    """Per rank, inside shard_map: the all-reduced shared + routed output.
    `start` is the rank's first global expert under EP, None under TP."""
    # Trace-time, once per token bucket: the evidence that the prefill kernel
    # serves this shape.
    logger.info("Fused MoE prefill kernel traced for %d tokens", x.shape[0])
    return _kernel_body(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        axis_name=EP_AXIS_NAME,
        expert_start=start,
    )
