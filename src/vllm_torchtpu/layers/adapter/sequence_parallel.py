# Copyright 2025 Google LLC
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
"""Sequence parallelism for the tensor-parallel prefill, gated by row count."""

from vllm_torchtpu import envs

_ACTIVE = False


def min_rows() -> int:
    return int(envs.TPU_SP_MIN_ROWS or 0)


def enabled() -> bool:
    return min_rows() > 0


def fp8_gather() -> bool:
    """Whether the attention-side all-gather carries fp8 rows and scales."""
    return int(envs.TPU_SP_FP8_GATHER or 0) > 0


def step_is_sequence_parallel(full_num_tokens: int) -> bool:
    return enabled() and int(full_num_tokens) >= min_rows()


def set_active(active: bool) -> None:
    """Record the form of the step about to run.

    The runner calls it before the compiled model, never a traced forward:
    a global written inside the trace becomes a graph output.
    """
    global _ACTIVE
    _ACTIVE = bool(active)


def is_active() -> bool:
    return _ACTIVE
