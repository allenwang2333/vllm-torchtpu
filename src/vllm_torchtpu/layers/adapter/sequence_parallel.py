# Copyright 2026 Google LLC
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
"""Sequence parallelism for the tensor-parallel prefill, gated by row count.

Selected with TPU_SEQUENCE_PARALLEL_PREFILL. A step of at least `MIN_ROWS`
rows reduce-scatters each attention output, so every rank runs the residual
add, the norms, the router and the shared expert on its own slice of the rows,
and the MoE dispatch all-gathers them for the experts. Smaller steps keep the
all-reduce form: they are latency-bound, and two collectives per layer beat
four.

The decoder layers (models/vllm/qwen3_next_sp_patch) take the form from the
row count, a shape, so each form is its own trace (compilation/shape_variants).
The communicator cannot see the row count, so the caller of each compiled
model records the form here first (`set_active`); a trace reads it once.
"""

import contextlib
from collections.abc import Iterator, Sequence

from vllm_torchtpu import envs

# Rows from which a step is sequence parallel. PR #989's value, kept pending a
# probe: at TP8 its first sequence-parallel bucket (512) gives each rank 64
# rows, above the fused EP v3 step gate.
MIN_ROWS = 257

_active = False
# Set while a layer that implements the row-gated form is being built.
_building = False
_layers_built = 0


def enabled() -> bool:
    """Whether the operator selected sequence parallelism."""
    return envs.TPU_SEQUENCE_PARALLEL_PREFILL


def use_sequence_parallel_moe(parallel_config) -> bool:
    """`ParallelConfig.use_sequence_parallel_moe` on TPU.

    True only while a layer that implements the row-gated form is built, so
    no other model takes vLLM's always-on form, which the communicator does
    not serve.
    """
    return (
        _building
        and enabled()
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.enable_expert_parallel
        and parallel_config.data_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.pipeline_parallel_size == 1
    )


@contextlib.contextmanager
def building_layer() -> Iterator[None]:
    """Build a decoder layer that implements the row-gated form."""
    global _building
    previous, _building = _building, True
    try:
        yield
    finally:
        _building = previous


def record_layer() -> None:
    global _layers_built
    _layers_built += 1


def check_buckets(compile_sizes: Sequence[int], tp_size: int) -> None:
    """Raise unless every sequence-parallel bucket splits evenly over TP."""
    uneven = [n for n in compile_sizes if n >= MIN_ROWS and n % tp_size]
    if uneven:
        raise NotImplementedError(
            f"TPU_SEQUENCE_PARALLEL_PREFILL splits every token bucket of at "
            f"least {MIN_ROWS} rows over tensor_parallel_size={tp_size}, and "
            f"buckets {uneven} do not divide. Choose --max-num-batched-tokens "
            f"and TPU_TOKEN_BUCKET_EXTRA as multiples of {tp_size}."
        )


def check_installed(tp_size: int) -> None:
    """Raise at boot when the operator selected sequence parallelism and no
    layer of the loaded model implements it."""
    if enabled() and _layers_built == 0:
        raise NotImplementedError(
            "TPU_SEQUENCE_PARALLEL_PREFILL serves Qwen3-Next and Qwen3.5 MoE "
            "decoders under --tensor-parallel-size>1 --enable-expert-parallel "
            "with data, prefill-context and pipeline parallelism 1; this "
            f"model or deployment (tensor_parallel_size={tp_size}) has no such "
            "layer. Unset TPU_SEQUENCE_PARALLEL_PREFILL."
        )


def step_is_sequence_parallel(num_tokens: int) -> bool:
    return enabled() and num_tokens >= MIN_ROWS


def set_active(active: bool) -> None:
    """Record the form of the compiled-model call about to run.

    Called outside the compiled region: a global written inside a traced
    forward becomes a graph output.
    """
    global _active
    _active = bool(active)


def is_active() -> bool:
    return _active


def rows_are_sliced(is_sequence_parallel: bool) -> bool:
    """Whether a MoE layer's rows are this rank's slice of the step."""
    return is_sequence_parallel and _active


def rows_are_replicated(is_sequence_parallel: bool) -> bool:
    """Whether a sequence-parallel MoE layer sees the step's rows replicated
    on every rank (a step below `MIN_ROWS`)."""
    return is_sequence_parallel and not _active
