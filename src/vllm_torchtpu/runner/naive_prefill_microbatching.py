# SPDX-License-Identifier: Apache-2.0
"""Request-safe planning for the fixed Qwen3.5 two-prefill experiment.

The feature splits one prefill step into a fixed number of equal-length
microbatches so their expert-parallel collectives can later be overlapped.
This module owns the part of that decision which needs no device state: which
local work is eligible, what every rank must agree on, and the padding lengths
the staged forwards are required to use.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

PREFILL_MICROBATCH_TOKENS = 1024
NUM_PREFILL_MICROBATCHES = 2


class PrefillStepMode(IntEnum):
    """What graph a step runs, agreed DP-wide before any forward launches.

    Ordered so a MAX reduction over the ranks' local modes yields the
    coordinated mode: an idle rank must not veto an eligible one, and any rank
    that cannot stage forces every rank onto the serial path.
    """
    IDLE = 0
    TWO_MICROBATCH = 1
    SERIAL = 2


@dataclass(frozen=True)
class PrefillMicrobatch:
    request_id: str
    input_start: int
    input_end: int
    output_index: int

    @property
    def num_tokens(self) -> int:
        return self.input_end - self.input_start


@dataclass(frozen=True)
class TwoMicrobatchPlan:
    """One or two real requests placed in a fixed two-slot graph."""
    microbatches: tuple[PrefillMicrobatch, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.microbatches) <= NUM_PREFILL_MICROBATCHES:
            raise ValueError("a plan requires one or two real microbatches")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(microbatch.request_id
                     for microbatch in self.microbatches)

    @property
    def num_real_microbatches(self) -> int:
        return len(self.microbatches)

    def restore_output_order(self, outputs: Sequence[Any]) -> list[Any]:
        """Map slot-ordered outputs back onto the scheduler's request order."""
        if len(outputs) != self.num_real_microbatches:
            raise ValueError("one output is required per real microbatch")
        restored: list[Any] = [None] * self.num_real_microbatches
        for output, microbatch in zip(outputs, self.microbatches):
            restored[microbatch.output_index] = output
        return restored


def build_two_microbatch_plan(
    request_ids: Sequence[str | None],
    num_scheduled_tokens: Mapping[str, int],
    num_computed_tokens: Mapping[str, int] | Sequence[int],
) -> TwoMicrobatchPlan | None:
    """Project one or two fresh, complete 1024-token requests.

    The compiled graph always has ``NUM_PREFILL_MICROBATCHES`` physical slots.
    Missing local requests are supplied by the runner as dummy bundles after
    this scheduler-owned plan is built, so a rank holding one real request
    still runs the same shaped step as a rank holding two.

    Returns ``None`` for anything outside the fixed envelope -- a partially
    computed request, a chunked one, a duplicate id -- which sends the step
    down the serial path instead.
    """
    live_ids = [request_id for request_id in request_ids
                if request_id is not None]
    if not 1 <= len(live_ids) <= NUM_PREFILL_MICROBATCHES:
        return None
    if len(set(live_ids)) != len(live_ids):
        return None
    if isinstance(num_computed_tokens, Mapping):
        computed = [int(num_computed_tokens.get(req_id, -1))
                    for req_id in live_ids]
    else:
        if len(num_computed_tokens) < len(live_ids):
            return None
        computed = [int(num_computed_tokens[i])
                    for i in range(len(live_ids))]
    if any(value != 0 for value in computed):
        return None
    if any(int(num_scheduled_tokens.get(req_id, -1)) !=
           PREFILL_MICROBATCH_TOKENS
           for req_id in live_ids):
        return None
    batches = tuple(PrefillMicrobatch(
        req_id, i * PREFILL_MICROBATCH_TOKENS,
        (i + 1) * PREFILL_MICROBATCH_TOKENS, i)
                    for i, req_id in enumerate(live_ids))
    return TwoMicrobatchPlan(batches)


def staged_token_bucket(mode: PrefillStepMode,
                        step_bucket: int | None) -> int | None:
    """Resolve the padded token length a step's forwards must all use.

    ``step_bucket`` is the DP max-reduction over each rank's *whole-step*
    scheduled token count, which is the right padding target only when a rank's
    step is one forward. ``TWO_MICROBATCH`` decomposes the step into
    ``NUM_PREFILL_MICROBATCHES`` forwards of ``PREFILL_MICROBATCH_TOKENS``, so
    the step total is off by exactly the number of real microbatches the rank
    holds and must not reach a per-microbatch padding decision.

    This is not a preference. The staged graph's EP collectives are in-graph,
    so a rank that pads its microbatches differently from its peers compiles a
    structurally different program, the collective operand sizes disagree, and
    every core blocks forever with no error. One real microbatch per rank hides
    this entirely: 1 x 1024 tokens makes the step total and the microbatch
    length the same number by coincidence.
    """
    if mode is PrefillStepMode.TWO_MICROBATCH:
        return PREFILL_MICROBATCH_TOKENS
    return step_bucket


def validate_staged_microbatch_tokens(num_tokens: int, *, role: str,
                                      index: int, field: str) -> None:
    """Reject a staged bundle whose padded token length is not the fixed one.

    Raised locally at bundle-construction time so a divergence surfaces as one
    rank's traceback instead of a silent DP-wide hang. See
    :func:`staged_token_bucket` for why the lengths must agree.
    """
    if int(num_tokens) != PREFILL_MICROBATCH_TOKENS:
        raise RuntimeError(
            f"staged prefill bundle {index} ({role}) has {field} padded to "
            f"{int(num_tokens)} tokens; every rank must pad every staged "
            f"microbatch to exactly {PREFILL_MICROBATCH_TOKENS}, or the "
            "in-graph EP collectives desync and halt every core")


def validate_staged_microbatch_bundle(bundle: Any, *, role: str,
                                      index: int) -> None:
    """Check every length the staged graph is traced against.

    Both roles build their bundles independently -- the busy rank through the
    real ``_prepare_inputs`` path and the idle rank through ``prepare_dummy``
    -- so nothing structurally forces them to agree. This is the checkpoint
    that does, and it runs before the launch that would otherwise hang.
    """
    validate_staged_microbatch_tokens(bundle.input_ids.shape[0], role=role,
                                      index=index, field="input_ids")
    validate_staged_microbatch_tokens(bundle.position_ids.shape[-1], role=role,
                                      index=index, field="position_ids")
    layout = bundle.layout_plan
    if layout is not None:
        validate_staged_microbatch_tokens(
            layout.global_padded_num_tokens, role=role, index=index,
            field="layout global_padded_num_tokens")
        validate_staged_microbatch_tokens(
            layout.local_padded_num_tokens, role=role, index=index,
            field="layout local_padded_num_tokens")


def local_step_mode(total_num_scheduled_tokens: int,
                    plan: TwoMicrobatchPlan | None) -> PrefillStepMode:
    """Classify local scheduler work before the DP-wide reduction."""
    if total_num_scheduled_tokens == 0:
        return PrefillStepMode.IDLE
    if plan is not None:
        return PrefillStepMode.TWO_MICROBATCH
    return PrefillStepMode.SERIAL


def coordinated_step_mode(modes: Sequence[PrefillStepMode]) -> PrefillStepMode:
    """Choose one graph globally; idle ranks do not veto eligible ranks."""
    busy = [mode for mode in modes if mode != PrefillStepMode.IDLE]
    if not busy:
        return PrefillStepMode.IDLE
    if all(mode == PrefillStepMode.TWO_MICROBATCH for mode in busy):
        return PrefillStepMode.TWO_MICROBATCH
    return PrefillStepMode.SERIAL


def validate_feature_config(vllm_config: Any) -> None:
    """Fail fast when opt-in configuration is outside the fixed envelope.

    The staged path is a fixed-shape experiment, not a general scheduling
    mode: it assumes exactly two 1024-token slots on eight EP ranks. Checking
    up front turns every unsupported combination into one clear startup error
    rather than a shape mismatch thousands of steps later.
    """
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    architectures = getattr(model.hf_config, "architectures", ()) or ()
    errors: list[str] = []
    if not any("Qwen3_5Moe" in architecture for architecture in architectures):
        errors.append("model must be Qwen3.5-MoE")
    if int(parallel.data_parallel_size) != 8 or not parallel.enable_expert_parallel:
        errors.append("expert parallelism must be enabled with EP/DP size 8")
    if int(parallel.tensor_parallel_size) != 1:
        errors.append("tensor parallel size must be 1")
    if int(scheduler.max_num_batched_tokens) != 2048:
        errors.append("max_num_batched_tokens must be 2048")
    if int(scheduler.max_num_seqs) != 2:
        errors.append("max_num_seqs must be 2")
    if int(model.max_model_len) < 2048:
        errors.append("max_model_len must be at least 2048")
    if getattr(vllm_config, "speculative_config", None) is not None:
        errors.append("speculative decoding must be disabled")
    if bool(getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
        errors.append("prefix caching must be disabled")
    if errors:
        raise ValueError("VLLM_TPU_NAIVE_PREFILL_MICROBATCHING: " +
                         "; ".join(errors))
