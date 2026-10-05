# SPDX-License-Identifier: Apache-2.0
"""Direct token-window micro-batching instrumentation with XLA optimization barriers."""

from vllm_torchtpu.models.microbatch.fence import (FenceMode, fence_windows,
                                                   microbatch_fence,
                                                   microbatch_fence_pair,
                                                   parse_fence_mode)
from vllm_torchtpu.models.microbatch.metadata import (
    bind_step_inputs_for_trace_alignment, slice_attention_metadata,
    slice_forward_metadata, slice_padding_state)
from vllm_torchtpu.models.microbatch.qwen import (
    MicroBatchConfig, ScheduleMode, can_split_moe_stages, install_microbatch,
    maybe_install_microbatch, parse_schedule_mode, patch_qwen_microbatch,
    uninstall_microbatch, unsupported_config_reasons)

__all__ = [
    "FenceMode",
    "MicroBatchConfig",
    "ScheduleMode",
    "bind_step_inputs_for_trace_alignment",
    "can_split_moe_stages",
    "fence_windows",
    "install_microbatch",
    "maybe_install_microbatch",
    "microbatch_fence",
    "microbatch_fence_pair",
    "parse_fence_mode",
    "parse_schedule_mode",
    "patch_qwen_microbatch",
    "slice_attention_metadata",
    "slice_forward_metadata",
    "slice_padding_state",
    "uninstall_microbatch",
    "unsupported_config_reasons",
]
