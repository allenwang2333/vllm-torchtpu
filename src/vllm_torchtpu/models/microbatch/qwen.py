# SPDX-License-Identifier: Apache-2.0
"""Direct token-window micro-batching instrumentation for Qwen3-Next / Qwen3.5.

Instruments ``Qwen3NextSparseMoeBlock``, ``Qwen3NextDecoderLayer``, and
``Qwen3NextModel`` (and ``Qwen3_5DecoderLayer`` / ``Qwen3_5Model``, which
inherit from them) to split large-token batches into ``N`` token windows using
explicit ``jax.lax.optimization_barrier`` fences (`pallas::microbatch_fence`
and `pallas::microbatch_fence_pair`).

Supported schedules (``TPU_MICROBATCH_SCHEDULE`` / ``MicroBatchConfig.schedule``):
- ``"stagewise"`` (default): splits each decoder stage (Attention, MoE) across
  windows; when ``merge_moe=True``, concatenates the routed-expert inputs across
  windows so ``fused_moe_gmm`` runs once per layer at full batch density.
- ``"layerwise"`` (also enabled by ``TPU_MICROBATCH_LAYERWISE=1``): runs each
  window through an entire decoder layer (Attention + MoE) before advancing to
  the next window.
- ``"staggered"``: explicitly interleaves Window ``w-1``'s collective sub-stage
  (Attention TP All-Reduce, MoE DP/EP All-Gather, MoE Combine/All-Reduce) with
  Window ``w``'s compute sub-stage across coupled diamond fences.

Supported fence modes (``TPU_MICROBATCH_FENCE`` / ``MicroBatchConfig.fence``):
- ``"none"``: no ``optimization_barrier`` ops emitted.
- ``"window"``: 1-tensor ``microbatch_fence`` emitted per window at stage ends.
- ``"stage"``: coupled 2-tensor ``microbatch_fence_pair`` chain across windows
  at stage boundaries, blocking cross-window horizontal fusion.
- ``"substage"`` (default): coupled 2-tensor ``microbatch_fence_pair`` diamond
  fences between compute and collective sub-stages, blocking cross-window
  horizontal fusion while leaving Window ``w-1``'s collective and Window ``w``'s
  compute as concurrent sibling branches in the XLA HLO DAG.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm import forward_context as _fc_mod
from vllm.config import VllmConfig
from vllm.distributed import parallel_state as _ps
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.runner import \
    moe_runner as _moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm import moe_routing, token_padding
from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.microbatch.fence import (FenceMode, fence_windows,
                                                   microbatch_fence,
                                                   microbatch_fence_pair,
                                                   parse_fence_mode)
from vllm_torchtpu.models.microbatch.metadata import (
    bind_step_inputs_for_trace_alignment, slice_forward_metadata,
    slice_padding_state)

logger = init_logger(__name__)


class ScheduleMode(StrEnum):
    """Execution schedule across token windows within a decoder layer."""

    STAGEWISE = "stagewise"
    LAYERWISE = "layerwise"
    STAGGERED = "staggered"


def parse_schedule_mode(
    schedule: str | ScheduleMode | None,
    layerwise: bool = False,
) -> ScheduleMode:
    """Resolve ``schedule`` and ``layerwise`` into a ``ScheduleMode``."""
    if isinstance(schedule, ScheduleMode):
        return schedule
    if schedule is None or schedule.strip() == "":
        return ScheduleMode.LAYERWISE if layerwise else ScheduleMode.STAGEWISE
    normalized = schedule.strip().lower()
    try:
        return ScheduleMode(normalized)
    except ValueError as exc:
        valid = [m.value for m in ScheduleMode]
        raise ValueError(
            f"Invalid microbatch schedule {schedule!r}. Valid options: {valid}"
        ) from exc


@dataclass(frozen=True)
class MicroBatchConfig:
    """Configuration for Qwen token-window micro-batching."""

    factor: int = 1
    min_tokens: int = 1024
    merge_moe: bool = True
    split_moe_min_tokens: int = 0
    layerwise: bool = False
    schedule: ScheduleMode | str = ""
    fence: FenceMode | str = FenceMode.SUBSTAGE

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ValueError(
                f"MicroBatchConfig.factor must be >= 1, got {self.factor}")
        if self.min_tokens < 0:
            raise ValueError(
                f"MicroBatchConfig.min_tokens must be >= 0, got {self.min_tokens}"
            )
        # Validate schedule and fence eagerly.
        _ = self.resolved_schedule
        _ = self.resolved_fence

    @property
    def resolved_schedule(self) -> ScheduleMode:
        return parse_schedule_mode(self.schedule, self.layerwise)

    @property
    def resolved_fence(self) -> FenceMode:
        return parse_fence_mode(self.fence)

    @classmethod
    def from_envs(cls) -> "MicroBatchConfig":
        return cls(
            factor=envs.TPU_MICROBATCH_FACTOR,
            min_tokens=envs.TPU_MICROBATCH_MIN_TOKENS,
            merge_moe=envs.TPU_MICROBATCH_MERGE_MOE,
            split_moe_min_tokens=envs.TPU_MICROBATCH_SPLIT_MOE_MIN_TOKENS,
            layerwise=envs.TPU_MICROBATCH_LAYERWISE,
            schedule=envs.TPU_MICROBATCH_SCHEDULE,
            fence=envs.TPU_MICROBATCH_FENCE,
        )


def _split_tokens(x: torch.Tensor, factor: int) -> list[torch.Tensor]:
    """Split ``[num_tokens, ...]`` into ``factor`` contiguous views."""
    return list(x.view(factor, -1, *x.shape[1:]).unbind(0))


def _split_positions(
    positions: torch.Tensor,
    factor: int,
) -> list[torch.Tensor]:
    """Split 1D ``[num_tokens]`` or 2D MRoPE ``[3, num_tokens]`` positions."""
    if positions.ndim == 2:
        return list(positions.view(positions.shape[0], factor, -1).unbind(1))
    return list(positions.view(factor, -1, *positions.shape[1:]).unbind(0))


@contextmanager
def _window_context(
    forward_context: Any,
    window_attn_metadata: Any,
    window_padding_state: token_padding.TokenPaddingState | None,
) -> Iterator[None]:
    """Temporarily bind ``attn_metadata`` and ``TokenPaddingState`` for one window."""
    saved_md = forward_context.attn_metadata if forward_context is not None else None
    saved_pad = token_padding.get_padding_state()
    if forward_context is not None:
        forward_context.attn_metadata = window_attn_metadata
    token_padding.set_padding_state(window_padding_state)
    try:
        yield
    finally:
        if forward_context is not None:
            forward_context.attn_metadata = saved_md
        token_padding.set_padding_state(saved_pad)


# ============================================================================
# Qwen3NextSparseMoeBlock Direct Sub-Stage Instrumentation
# ============================================================================


def _get_moe_runner(experts: Any) -> MoERunner | None:
    """Resolve the ``MoERunner`` instance from ``moe_block.experts``."""
    if isinstance(experts, MoERunner):
        return experts
    runner = getattr(experts, "runner", None)
    if isinstance(runner, MoERunner):
        return runner
    return None


def _maybe_sp_context(runner: MoERunner | None):
    """Enter ``runner._sequence_parallel_context()`` when forward context is active."""
    if (runner is not None and hasattr(runner, "_sequence_parallel_context")
            and _fc_mod.is_forward_context_available()):
        return runner._sequence_parallel_context()
    return nullcontext()


def _apply_layer_scale(
    hidden_states: torch.Tensor,
    scale_param: torch.Tensor,
) -> torch.Tensor:
    """Apply ``Qwen3NextDecoderLayer`` / ``Qwen3_5DecoderLayer`` layer scale."""
    scale = scale_param.to(hidden_states.dtype)
    if scale.ndim == 3:
        if hidden_states.ndim == 2:
            return hidden_states * (scale[0] + 1)
        return hidden_states * (scale + 1)
    return hidden_states * scale


def can_split_moe_stages(moe_block: nn.Module | None) -> bool:
    """Return True if ``moe_block`` supports decomposed pre/kernel/post MoE stages."""
    if moe_block is None:
        return False
    if hasattr(moe_block, "_moe_kernel_fn"):
        return True
    experts = getattr(moe_block, "experts", None)
    if experts is None:
        return False
    runner = _get_moe_runner(experts)
    if runner is None:
        return False
    has_fused_gate_se = (
        getattr(moe_block, "_fused_gate_se_weight", None) is not None
    )
    gate = getattr(runner, "gate", None) or getattr(moe_block, "gate", None)
    if gate is None and not has_fused_gate_se:
        return False
    moe_config = getattr(runner, "moe_config", None)
    if (getattr(moe_block, "is_sequence_parallel", False)
            or getattr(experts, "is_sequence_parallel", False)
            or getattr(moe_config, "is_sequence_parallel", False)):
        return False
    if (getattr(experts, "has_ici_cp_collectives", False)
            or getattr(moe_config, "pcp_size", 1) > 1):
        return False
    routed = getattr(runner, "routed_experts",
                     getattr(experts, "routed_experts", None))
    quant_method = getattr(routed, "quant_method", None)
    if quant_method is None:
        return False
    if getattr(quant_method, "_use_fused_ep", False):
        return False
    from vllm_torchtpu.layers.vllm.fused_moe_ep import \
        fused_moe_ep_supported
    from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import \
        fused_moe_ep_supported as fused_moe_ep_v3_supported
    if fused_moe_ep_supported(quant_method):
        return False
    if envs.USE_MOE_FUSED_EP_KERNEL and (
        fused_moe_ep_v3_supported(routed)
        or fused_moe_ep_v3_supported(quant_method)
    ):
        return False
    has_weights = (hasattr(routed, "w13_weight") and hasattr(
        routed, "w2_weight")) or (hasattr(routed, "w13_weight_packed")
                                  and hasattr(routed, "w2_weight_packed"))
    if not has_weights:
        return False
    return True


def _moe_forward_pre_compute(
    self: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor, torch.Tensor | None,
           torch.Tensor | None]:
    """Sub-stage 2a (TensorCore compute): router gate + shared expert MLP."""
    orig_shape = hidden_states.shape
    hidden_states_2d = hidden_states.view(-1, hidden_states.shape[-1])

    experts = getattr(self, "experts", None)
    runner = _get_moe_runner(experts)
    routed = getattr(runner, "routed_experts",
                     getattr(experts, "routed_experts", None))
    if routed is not None and hasattr(routed, "_ensure_moe_quant_config_init"):
        routed._ensure_moe_quant_config_init()



    shared_experts_input: torch.Tensor | None = hidden_states_2d
    if runner is not None and hasattr(runner, "apply_routed_input_transform"):
        hidden_states_2d, shared_experts_input = (
            runner.apply_routed_input_transform(hidden_states_2d))
    if runner is not None and hasattr(runner, "_maybe_pad_hidden_states"):
        hidden_states_2d, _, _ = runner._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states_2d,
        )

    if runner is not None and getattr(runner, "_fse_fuse_gate", False):
        runner._maybe_fuse_gate_weights()
        router_logits = F.linear(hidden_states_2d,
                                 runner._combined_gate_weight)
    else:
        gate = getattr(runner, "gate", None) or getattr(self, "gate", None)
        router_logits, _ = gate(hidden_states_2d)

    # Enter _sequence_parallel_context() immediately after gate (before
    # shared_experts) to match MoERunner._forward_impl's exact FX graph
    # placeholder order (specifically dp_metadata.num_tokens_across_dp_cpu).
    with _maybe_sp_context(runner):
        inner_shared_output: torch.Tensor | None = None
        outer_shared_output: torch.Tensor | None = None
        shared_experts = getattr(runner, "shared_experts",
                                 getattr(experts, "shared_experts", None))
        if shared_experts is not None:
            sh_in = (shared_experts_input
                     if shared_experts_input is not None else hidden_states_2d)
            inner_layer = getattr(shared_experts, "_layer", None)
            if inner_layer is not None:
                inner_shared_output = inner_layer(sh_in)
            else:
                inner_shared_output = shared_experts(sh_in)
                if hasattr(shared_experts, "_output"):
                    shared_experts._output = None

        if getattr(self, "replicate_shared_expert", False) and getattr(
                self, "shared_expert", None) is not None:
            outer_shared_output = self.shared_expert(hidden_states_2d)
        elif (shared_experts is None
              and getattr(self, "n_shared_experts", 0) > 0
              and getattr(self, "shared_expert", None) is not None):
            outer_shared_output = self.shared_expert(hidden_states_2d)
            if (getattr(self, "shared_expert_gate", None) is not None and
                    getattr(self.shared_expert, "expert_gate", None) is None):
                outer_shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states_2d)) *
                    outer_shared_output)

    return (
        hidden_states_2d,
        orig_shape,
        router_logits,
        inner_shared_output,
        outer_shared_output,
    )


def _moe_forward_pre_dispatch_route(
    self: Any,
    pre_compute_state: tuple[torch.Tensor, tuple[int, ...], torch.Tensor,
                             torch.Tensor | None, torch.Tensor | None],
) -> tuple[tuple[int, ...], torch.Tensor | None, torch.Tensor | None, bool,
           torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sub-stage 2b (DP/EP All-Gather collective + top-k routing)."""
    (
        hidden_states_2d,
        orig_shape,
        router_logits,
        inner_shared_output,
        outer_shared_output,
    ) = pre_compute_state

    experts = getattr(self, "experts", None)
    if experts is None:
        # Synthetic test MoE block without a vLLM FusedMoE ``experts`` submodule.
        topk_weights, topk_ids = self.route(hidden_states_2d, router_logits)
        return (
            orig_shape,
            inner_shared_output,
            outer_shared_output,
            False,
            None,
            hidden_states_2d,
            topk_weights,
            topk_ids,
        )

    runner = _get_moe_runner(experts)
    dispatch_owner = (runner if runner is not None
                      and hasattr(runner, "_maybe_dispatch") else experts)
    with _maybe_sp_context(runner):
        dispatch_res = dispatch_owner._maybe_dispatch(hidden_states_2d,
                                                      router_logits)
        if len(dispatch_res) == 3:
            dispatched_x, dispatched_logits, should_attempt_combine = (
                dispatch_res)
        else:
            dispatched_x, dispatched_logits = dispatch_res
            should_attempt_combine = True

        routed = getattr(runner, "routed_experts",
                         getattr(experts, "routed_experts", None))
        topk_weights, topk_ids = moe_routing.route(
            routed,
            dispatched_x,
            dispatched_logits,
        )

        if (getattr(routed, "zero_expert_num", None) is not None
                and routed.zero_expert_num > 0
                and hasattr(_moe_runner_mod, "zero_expert_oot")):
            topk_ids, topk_weights, zero_expert_result = (
                _moe_runner_mod.zero_expert_oot(
                    routed,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    hidden_states=dispatched_x,
                ))
        else:
            zero_expert_result = None

        quant_method = getattr(routed, "quant_method", None)
        is_mxfp4 = isinstance(quant_method,
                              VllmCompressedTensorsW4ANMxfp4MoEMethod)
        if is_mxfp4:
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights.to(dispatched_x.dtype)

        if envs.TPU_MOE_SKIP_PADDED_TOKENS or is_mxfp4:
            is_local_tensor = (
                dispatched_x.shape[0] == hidden_states_2d.shape[0])
            topk_ids, topk_weights = (
                token_padding.zero_routing_weights_for_padding(
                    topk_ids,
                    topk_weights,
                    is_local_tensor=is_local_tensor,
                ))

    return (
        orig_shape,
        inner_shared_output,
        outer_shared_output,
        should_attempt_combine,
        zero_expert_result,
        dispatched_x,
        topk_weights,
        topk_ids,
    )


def _moe_forward_pre(
    self: Any,
    hidden_states: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor | None, torch.Tensor | None, bool,
           torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined MoE pre-stage (gate + shared expert + dispatch + routing)."""
    return self.forward_moe_pre_dispatch_route(
        self.forward_moe_pre_compute(hidden_states))


def _moe_forward_kernel(
    self: Any,
    dispatched_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c: routed-expert grouped-matmul kernel (``fused_moe_gmm``)."""
    custom_kernel = getattr(self, "_moe_kernel_fn", None)
    if custom_kernel is not None:
        return custom_kernel(dispatched_x, topk_weights, topk_ids)

    experts = self.experts
    runner = _get_moe_runner(experts)
    routed = getattr(runner, "routed_experts",
                     getattr(experts, "routed_experts", None))
    quant_method = routed.quant_method
    activation_str = getattr(quant_method, "_tpu_activation_str", None)
    if activation_str is None:
        from vllm_torchtpu.layers.vllm.fused_moe import \
            get_fused_moe_activation
        activation_str = get_fused_moe_activation(routed.activation,
                                                  routed.moe_config)

    w1 = getattr(routed, "w13_weight",
                 getattr(routed, "w13_weight_packed", None))
    w2 = getattr(routed, "w2_weight", getattr(routed, "w2_weight_packed",
                                              None))
    w1_scale = getattr(
        routed,
        "w13_weight_scale_inv",
        getattr(routed, "w13_weight_scale", None),
    )
    w2_scale = getattr(
        routed,
        "w2_weight_scale_inv",
        getattr(routed, "w2_weight_scale", None),
    )
    kwargs: dict[str, Any] = {
        "hidden_states": dispatched_x,
        "w1": w1,
        "w2": w2,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_bias": getattr(routed, "w13_bias", None),
        "w2_bias": getattr(routed, "w2_bias", None),
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "experts_start": getattr(routed, "_experts_start", None),
        "topk": routed.moe_config.experts_per_token,
        "activation": activation_str,
    }
    if getattr(routed, "_gmm_primary_experts", None) is not None:
        kwargs["num_primary_experts"] = routed._gmm_primary_experts
    if hasattr(routed, "_rhs_quant_dtype"):
        kwargs["rhs_quant_dtype"] = routed._rhs_quant_dtype
    if isinstance(quant_method, VllmCompressedTensorsW4ANMxfp4MoEMethod):
        kwargs["skip_padded_tokens"] = True

    from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import \
        retain_fused_ep_graph_inputs

    out = fused_moe_gmm(**kwargs)
    if (
        getattr(routed, "_lowtok_has_copies", False)
        and getattr(routed, "_lowtok_tables", None) is not None
    ):
        zero = (
            routed._lowtok_tables[:1]
            + routed._lowtok_prim[:1]
            + routed._lowtok_copy[:1]
            + routed._lowtok_hot1h[0, :1].to(torch.int32)
        ) * 0
        out = out + zero.to(out.dtype)
    return retain_fused_ep_graph_inputs(out)


def _moe_forward_post(
    self: Any,
    pre_state: tuple[tuple[int, ...], torch.Tensor | None, torch.Tensor | None,
                     bool, torch.Tensor | None, torch.Tensor, torch.Tensor,
                     torch.Tensor],
    routed_out: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2d: DP/EP Combine + shared-expert sum + TP All-Reduce."""
    (
        orig_shape,
        inner_shared_output,
        outer_shared_output,
        should_attempt_combine,
        zero_expert_result,
        _,
        _,
        _,
    ) = pre_state

    experts = getattr(self, "experts", None)
    if experts is None:
        final_hidden_states = routed_out
        if inner_shared_output is not None:
            final_hidden_states = final_hidden_states + inner_shared_output
        if outer_shared_output is not None:
            final_hidden_states = final_hidden_states + outer_shared_output
        if getattr(self, "reduce_fn", None) is not None:
            final_hidden_states = self.reduce_fn(final_hidden_states)
        return final_hidden_states.view(orig_shape)

    runner = _get_moe_runner(experts)
    assert runner is not None
    if zero_expert_result is not None:
        experts.zero_expert_result = zero_expert_result

    combine_owner = (runner if hasattr(runner, "_maybe_combine") else experts)
    with _maybe_sp_context(runner):
        combined = combine_owner._maybe_combine(inner_shared_output,
                                                routed_out)
    if isinstance(combined, tuple):
        inner_shared_output, routed_out = combined
    else:
        routed_out = combined

    fused_output_is_reduced = bool(
        getattr(runner, "_fused_output_is_reduced", False))

    routed_out, fused_output_is_reduced = (
        runner._maybe_reduce_routed_output_before_transform(
            routed_out,
            fused_output_is_reduced,
        ))

    inner_shared_output = runner._maybe_reduce_shared_expert_output(
        inner_shared_output,
        fused_output_is_reduced,
    )

    inner_shared_output, routed_out = (
        runner._maybe_apply_routed_scale_to_output(
            inner_shared_output,
            routed_out,
        ))

    fused_output = runner.apply_routed_output_transform(routed_out)
    if inner_shared_output is not None:
        output = inner_shared_output + fused_output
    else:
        output = fused_output

    final_hidden_states = runner._maybe_reduce_final_output(
        output,
        None,
        fused_output_is_reduced,
    )

    zero_owner = (runner if hasattr(runner, "_maybe_add_zero_expert_output")
                  else experts)
    if hasattr(zero_owner, "_maybe_add_zero_expert_output"):
        final_hidden_states = zero_owner._maybe_add_zero_expert_output(
            final_hidden_states)

    if outer_shared_output is not None:
        if getattr(self, "_fused_gate_se_weight", None) is not None:
            se_is_replicated = bool(
                getattr(self, "_fused_se_is_replicated", False)
                or getattr(self, "replicate_shared_expert", False)
                or getattr(self, "is_sequence_parallel", False)
            )
            final_hidden_states = final_hidden_states + outer_shared_output
            if not se_is_replicated and getattr(self, "tp_size", 1) > 1:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
        else:
            final_hidden_states = final_hidden_states + outer_shared_output
    return final_hidden_states.view(orig_shape)


# ============================================================================
# Qwen3NextDecoderLayer Direct Stage & Sub-Stage Instrumentation
# ============================================================================


def _get_attn_module_and_out_proj(
        layer: nn.Module) -> tuple[nn.Module | None, Any]:
    """Return ``(attn_module, out_proj)`` for a Qwen3-Next / Qwen3.5 decoder layer."""
    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention" or hasattr(layer, "linear_attn"):
        attn_mod = getattr(layer, "linear_attn", None)
        out_proj = getattr(attn_mod, "out_proj", None)
        return attn_mod, out_proj
    if layer_type == "full_attention" or hasattr(layer, "self_attn"):
        attn_mod = getattr(layer, "self_attn", None)
        out_proj = getattr(attn_mod, "o_proj", None)
        return attn_mod, out_proj
    if hasattr(layer, "attn"):
        attn_mod = getattr(layer, "attn", None)
        out_proj = getattr(attn_mod, "out_proj",
                           getattr(attn_mod, "o_proj", None))
        return attn_mod, out_proj
    return None, None


def _layer_forward_attn_compute(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any, bool]:
    """Sub-stage 1a: input_layernorm + QKV/GDN/RPA + local output projection."""
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    attn_mod, out_proj = _get_attn_module_and_out_proj(self)
    orig_reduce = bool(getattr(out_proj, "reduce_results", False))
    if out_proj is not None and hasattr(out_proj, "reduce_results"):
        out_proj.reduce_results = False
    try:
        layer_type = getattr(self, "layer_type", None)
        if layer_type == "linear_attention" or hasattr(self, "linear_attn"):
            attn_local = self.linear_attn(hidden_states=hidden_states)
        elif layer_type == "full_attention" or hasattr(self, "self_attn"):
            attn_local = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            attn_local = attn_mod(hidden_states, positions)
        if isinstance(attn_local, tuple):
            attn_local = attn_local[0]
    finally:
        if out_proj is not None and hasattr(out_proj, "reduce_results"):
            out_proj.reduce_results = orig_reduce

    return attn_local, residual, out_proj, orig_reduce


def _layer_forward_attn_reduce(
    self: Any,
    attn_compute_state: tuple[torch.Tensor, torch.Tensor, Any, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-stage 1b: Attention TP All-Reduce + optional layer_scale."""
    attn_local, residual, out_proj, orig_reduce = attn_compute_state
    if orig_reduce and out_proj is not None:
        reduce_fn = getattr(out_proj, "reduce_fn", None)
        if reduce_fn is not None:
            hidden_states = reduce_fn(attn_local)
        elif getattr(out_proj, "tp_size", 1) > 1:
            hidden_states = tensor_model_parallel_all_reduce(attn_local)
        else:
            hidden_states = attn_local
    else:
        hidden_states = attn_local

    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states,
                                           self.attn_layer_scale)
    return hidden_states, residual


def _layer_forward_attn(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: full attention stage (compute + TP All-Reduce)."""
    return self.forward_attn_reduce(
        self.forward_attn_compute(hidden_states, residual, positions))


def _layer_forward_moe_pre_compute(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Sub-stage 2a: post_attention_layernorm + gate + shared_expert compute."""
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    moe_pre_compute = self.mlp.forward_moe_pre_compute(hidden_states)
    return moe_pre_compute, residual


def _layer_forward_moe_pre_dispatch_route(
    self: Any,
    layer_pre_compute_state: tuple[Any, torch.Tensor],
) -> tuple[Any, torch.Tensor]:
    """Sub-stage 2b: MoE DP/EP All-Gather dispatch + top-k routing."""
    moe_pre_compute, residual = layer_pre_compute_state
    moe_pre_state = self.mlp.forward_moe_pre_dispatch_route(moe_pre_compute)
    return moe_pre_state, residual


def _layer_forward_moe_pre(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Combined MoE pre-stage on decoder layer."""
    return self.forward_moe_pre_dispatch_route(
        self.forward_moe_pre_compute(hidden_states, residual))


def _layer_forward_moe_kernel(
    self: Any,
    dispatched_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c: routed-expert grouped-matmul kernel."""
    return self.mlp.forward_moe_kernel(dispatched_x, topk_weights, topk_ids)


def _layer_forward_moe_post(
    self: Any,
    layer_pre_state: tuple[Any, torch.Tensor],
    routed_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-stage 2d: MoE combine + TP All-Reduce + optional ffn_layer_scale."""
    moe_pre_state, residual = layer_pre_state
    hidden_states = self.mlp.forward_moe_post(moe_pre_state, routed_out)
    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, self.ffn_layer_scale)
    return hidden_states, residual


def _layer_forward_mlp(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2 fallback for dense MLP or non-splittable MoE blocks."""
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, self.ffn_layer_scale)
    return hidden_states, residual


# ============================================================================
# Qwen3NextModel / Qwen3_5Model Micro-Batched Forward Schedules
# ============================================================================


def _run_merged_or_split_moe_kernels(
    layer: Any,
    layer_pre_states: list[tuple[Any, torch.Tensor]],
    *,
    merge_moe: bool,
    fence: FenceMode,
) -> list[torch.Tensor]:
    """Execute ``layer.forward_moe_kernel`` either merged once or per window."""
    num_windows = len(layer_pre_states)
    dispatched_xs = [state[0][5] for state in layer_pre_states]
    topk_weights_list = [state[0][6] for state in layer_pre_states]
    topk_ids_list = [state[0][7] for state in layer_pre_states]

    if merge_moe:
        if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE) and num_windows > 1:
            dispatched_xs = fence_windows(dispatched_xs, fence)
        elif fence == FenceMode.WINDOW:
            dispatched_xs = [microbatch_fence(x) for x in dispatched_xs]
        cat_x = torch.cat(dispatched_xs, dim=0)
        cat_weights = torch.cat(topk_weights_list, dim=0)
        cat_ids = torch.cat(topk_ids_list, dim=0)
        merged_out = layer.forward_moe_kernel(cat_x, cat_weights, cat_ids)
        if fence != FenceMode.NONE:
            merged_out = microbatch_fence(merged_out)
        return list(
            merged_out.view(num_windows, -1, merged_out.shape[-1]).unbind(0))

    routed_outs: list[torch.Tensor] = []
    for w in range(num_windows):
        dx = dispatched_xs[w]
        if w > 0 and fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            routed_outs[w - 1], dx = microbatch_fence_pair(
                routed_outs[w - 1], dx)
        out_w = layer.forward_moe_kernel(
            dx,
            topk_weights_list[w],
            topk_ids_list[w],
        )
        if fence == FenceMode.WINDOW:
            out_w = microbatch_fence(out_w)
        routed_outs.append(out_w)
    return routed_outs


def _run_layer_substage_staggered(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    merge_moe: bool,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer with diamond-fenced sub-stage compute/collective overlap.

    Interleaves Window ``w-1``'s collective sub-stage with Window ``w``'s
    compute sub-stage across ``microbatch_fence_pair`` barriers:
      1. ``w0`` ``attn_compute``
      2. For ``w = 1..N-1``:
         ``fence_pair(attn_local[w-1], hs[w])`` ->
         (``w-1`` ``attn_reduce`` || ``w`` ``attn_compute``)
      3. If MoE is splittable:
         ``fence_pair(attn_local[N-1], hs[0])`` ->
         (``N-1`` ``attn_reduce`` || ``w0`` ``moe_pre_compute``)
      4. For ``w = 1..N-1``:
         ``fence_pair(router_logits[w-1], hs[w])`` ->
         (``w-1`` ``moe_pre_dispatch_route`` || ``w`` ``moe_pre_compute``)
      5. ``N-1`` ``moe_pre_dispatch_route``
      6. ``moe_kernel`` (merged across windows if ``merge_moe=True``, else per window)
      7. ``moe_post`` across windows, coupling ``hs[0]`` with ``routed_outs[1]``
         so Layer ``l+1``'s ``w0`` ``attn_compute`` can overlap ``w_{1..N-1}``'s
         ``moe_post`` collectives.
    """
    num_windows = len(hs_windows)
    use_stage_pair_fence = fence in (FenceMode.STAGE, FenceMode.SUBSTAGE)
    use_substage_pair_fence = fence == FenceMode.SUBSTAGE

    # --- Sub-stage 1a (w=0): Attention compute ---
    with _window_context(fc, md_windows[0], pad_windows[0]):
        attn_state_prev = layer.forward_attn_compute(
            hs_windows[0],
            res_windows[0],
            pos_windows[0],
        )

    # --- Sub-stage 1b (w-1) || Sub-stage 1a (w) for w=1..N-1 ---
    for w in range(1, num_windows):
        attn_local_prev, res_prev, out_proj_prev, orig_reduce_prev = (
            attn_state_prev)
        if use_substage_pair_fence:
            attn_local_prev, hs_windows[w] = microbatch_fence_pair(
                attn_local_prev,
                hs_windows[w],
            )
        elif fence == FenceMode.WINDOW:
            attn_local_prev = microbatch_fence(attn_local_prev)

        with _window_context(fc, md_windows[w - 1], pad_windows[w - 1]):
            hs_windows[w - 1], res_windows[w - 1] = layer.forward_attn_reduce(
                (attn_local_prev, res_prev, out_proj_prev, orig_reduce_prev))

        with _window_context(fc, md_windows[w], pad_windows[w]):
            attn_state_prev = layer.forward_attn_compute(
                hs_windows[w],
                res_windows[w],
                pos_windows[w],
            )

    mlp = getattr(layer, "mlp", None)
    if not can_split_moe_stages(mlp):
        # Complete Window N-1's attention reduce, then run dense MLP per window.
        attn_local_last, res_last, out_proj_last, orig_reduce_last = (
            attn_state_prev)
        if fence == FenceMode.WINDOW:
            attn_local_last = microbatch_fence(attn_local_last)
        with _window_context(fc, md_windows[num_windows - 1],
                             pad_windows[num_windows - 1]):
            hs_windows[num_windows -
                       1], res_windows[num_windows -
                                       1] = (layer.forward_attn_reduce((
                                           attn_local_last,
                                           res_last,
                                           out_proj_last,
                                           orig_reduce_last,
                                       )))
        hs_windows = fence_windows(hs_windows, fence)
        for w in range(num_windows):
            if w > 0 and use_stage_pair_fence:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_mlp(
                    hs_windows[w],
                    res_windows[w],
                )
        hs_windows = fence_windows(hs_windows, fence)
        return hs_windows, res_windows

    # --- Overlap Window N-1 attn_reduce with Window 0 moe_pre_compute ---
    attn_local_last, res_last, out_proj_last, orig_reduce_last = attn_state_prev
    if use_stage_pair_fence and num_windows > 1:
        attn_local_last, hs_windows[0] = microbatch_fence_pair(
            attn_local_last,
            hs_windows[0],
        )
    elif fence != FenceMode.NONE:
        attn_local_last = microbatch_fence(attn_local_last)

    with _window_context(fc, md_windows[num_windows - 1],
                         pad_windows[num_windows - 1]):
        hs_windows[num_windows -
                   1], res_windows[num_windows -
                                   1] = (layer.forward_attn_reduce((
                                       attn_local_last,
                                       res_last,
                                       out_proj_last,
                                       orig_reduce_last,
                                   )))

    with _window_context(fc, md_windows[0], pad_windows[0]):
        pre_compute_prev = layer.forward_moe_pre_compute(
            hs_windows[0],
            res_windows[0],
        )

    # --- Sub-stage 2b (w-1) || Sub-stage 2a (w) for w=1..N-1 ---
    layer_pre_states: list[tuple[Any, torch.Tensor]] = []
    for w in range(1, num_windows):
        moe_pc_prev, res_pc_prev = pre_compute_prev
        (
            hs_2d_prev,
            orig_shape_prev,
            logits_prev,
            inner_sh_prev,
            outer_sh_prev,
        ) = moe_pc_prev
        if use_substage_pair_fence:
            logits_prev, hs_windows[w] = microbatch_fence_pair(
                logits_prev,
                hs_windows[w],
            )
            if inner_sh_prev is not None:
                inner_sh_prev, hs_windows[w] = microbatch_fence_pair(
                    inner_sh_prev,
                    hs_windows[w],
                )
            if outer_sh_prev is not None:
                outer_sh_prev, hs_windows[w] = microbatch_fence_pair(
                    outer_sh_prev,
                    hs_windows[w],
                )
        elif fence == FenceMode.WINDOW:
            logits_prev = microbatch_fence(logits_prev)
            if inner_sh_prev is not None:
                inner_sh_prev = microbatch_fence(inner_sh_prev)
            if outer_sh_prev is not None:
                outer_sh_prev = microbatch_fence(outer_sh_prev)

        pre_compute_fenced = (
            (
                hs_2d_prev,
                orig_shape_prev,
                logits_prev,
                inner_sh_prev,
                outer_sh_prev,
            ),
            res_pc_prev,
        )
        with _window_context(fc, md_windows[w - 1], pad_windows[w - 1]):
            layer_pre_states.append(
                layer.forward_moe_pre_dispatch_route(pre_compute_fenced))

        with _window_context(fc, md_windows[w], pad_windows[w]):
            pre_compute_prev = layer.forward_moe_pre_compute(
                hs_windows[w],
                res_windows[w],
            )

    # --- Sub-stage 2b (w=N-1): Dispatch + route last window ---
    with _window_context(fc, md_windows[num_windows - 1],
                         pad_windows[num_windows - 1]):
        layer_pre_states.append(
            layer.forward_moe_pre_dispatch_route(pre_compute_prev))

    # --- Sub-stage 2c: Routed expert GMM kernel (merged or split) ---
    routed_outs = _run_merged_or_split_moe_kernels(
        layer,
        layer_pre_states,
        merge_moe=merge_moe,
        fence=fence,
    )

    # --- Sub-stage 2d: MoE post (Combine + TP All-Reduce) ---
    for w in range(num_windows):
        with _window_context(fc, md_windows[w], pad_windows[w]):
            hs_windows[w], res_windows[w] = layer.forward_moe_post(
                layer_pre_states[w],
                routed_outs[w],
            )
        if w + 1 < num_windows and use_stage_pair_fence:
            # Chain fence_pair across w=0..N-2 so each window's moe_post is
            # coupled with the next window's routed_out before its moe_post runs.
            hs_windows[w], routed_outs[w + 1] = microbatch_fence_pair(
                hs_windows[w],
                routed_outs[w + 1],
            )
        elif fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])

    return hs_windows, res_windows


def _run_layer_stagewise(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    merge_moe: bool,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer in stagewise order (when fence != SUBSTAGE)."""
    num_windows = len(hs_windows)

    # --- Stage 1: Attention across windows 0..N-1 ---
    for w in range(num_windows):
        if w > 0 and fence == FenceMode.STAGE:
            hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                hs_windows[w - 1],
                hs_windows[w],
            )
        with _window_context(fc, md_windows[w], pad_windows[w]):
            hs_windows[w], res_windows[w] = layer.forward_attn(
                hs_windows[w],
                res_windows[w],
                pos_windows[w],
            )
        if fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])
    if fence == FenceMode.STAGE:
        hs_windows = fence_windows(hs_windows, fence)

    # --- Stage 2: MoE / MLP across windows 0..N-1 ---
    mlp = getattr(layer, "mlp", None)
    if can_split_moe_stages(mlp):
        layer_pre_states: list[tuple[Any, torch.Tensor]] = []
        for w in range(num_windows):
            if w > 0 and fence == FenceMode.STAGE:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                layer_pre_states.append(
                    layer.forward_moe_pre(hs_windows[w], res_windows[w]))

        routed_outs = _run_merged_or_split_moe_kernels(
            layer,
            layer_pre_states,
            merge_moe=merge_moe,
            fence=fence,
        )

        for w in range(num_windows):
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_moe_post(
                    layer_pre_states[w],
                    routed_outs[w],
                )
            if fence == FenceMode.WINDOW:
                hs_windows[w] = microbatch_fence(hs_windows[w])
        if fence == FenceMode.STAGE:
            hs_windows = fence_windows(hs_windows, fence)
    else:
        for w in range(num_windows):
            if w > 0 and fence == FenceMode.STAGE:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_mlp(
                    hs_windows[w],
                    res_windows[w],
                )
            if fence == FenceMode.WINDOW:
                hs_windows[w] = microbatch_fence(hs_windows[w])
        if fence == FenceMode.STAGE:
            hs_windows = fence_windows(hs_windows, fence)

    return hs_windows, res_windows


def _run_layer_layerwise(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer in full-layer window order (Window 0, then Window 1, ...)."""
    num_windows = len(hs_windows)
    for w in range(num_windows):
        if w > 0 and fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                hs_windows[w - 1],
                hs_windows[w],
            )
            if res_windows[w - 1] is not None and res_windows[w] is not None:
                res_windows[w - 1], res_windows[w] = microbatch_fence_pair(
                    res_windows[w - 1],
                    res_windows[w],
                )
        with _window_context(fc, md_windows[w], pad_windows[w]):
            if hasattr(layer, "forward_attn") and hasattr(
                    layer, "forward_mlp"):
                hs_w, res_w = layer.forward_attn(
                    hs_windows[w],
                    res_windows[w],
                    pos_windows[w],
                )
                if fence == FenceMode.SUBSTAGE:
                    hs_w = microbatch_fence(hs_w)
                mlp = getattr(layer, "mlp", None)
                if can_split_moe_stages(mlp):
                    pre_state = layer.forward_moe_pre(hs_w, res_w)
                    routed_out = layer.forward_moe_kernel(
                        pre_state[0][5],
                        pre_state[0][6],
                        pre_state[0][7],
                    )
                    hs_w, res_w = layer.forward_moe_post(pre_state, routed_out)
                else:
                    hs_w, res_w = layer.forward_mlp(hs_w, res_w)
                hs_windows[w], res_windows[w] = hs_w, res_w
            else:
                hs_windows[w], res_windows[w] = layer(
                    positions=pos_windows[w],
                    hidden_states=hs_windows[w],
                    residual=res_windows[w],
                )
        if fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])
            if res_windows[w] is not None:
                res_windows[w] = microbatch_fence(res_windows[w])

    if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
        hs_windows = fence_windows(hs_windows, fence)
    return hs_windows, res_windows


def _active_layers(model: Any) -> list[nn.Module]:
    layers = getattr(model, "layers", None)
    if layers is None:
        return []
    start_layer = getattr(model, "start_layer", 0)
    end_layer = getattr(model, "end_layer", len(layers))
    return list(layers[start_layer:end_layer])


def _get_tp_size(layer: Any = None) -> int:
    if layer is not None:
        mlp = getattr(layer, "mlp", None)
        tp = getattr(mlp, "tp_size", None)
        if isinstance(tp, int) and tp > 1:
            return tp
    try:
        return _ps.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _sp_chunk(x: torch.Tensor, full_num_tokens: int) -> torch.Tensor:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    chunk_size = full_num_tokens // tp_size
    return x.narrow(0, tp_rank * chunk_size, chunk_size)


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed import tensor_model_parallel_all_gather

    packed = torch.cat([hidden_states, residual], dim=-1)
    packed = tensor_model_parallel_all_gather(packed, 0)
    packed = packed.narrow(0, 0, full_num_tokens)
    return packed.split([hidden_size, hidden_size], dim=-1)


def _all_gather_sp_windows(
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    window_tokens: int,
    hidden_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """All-gather sequence-parallel ``hs_windows`` and ``res_windows`` across TP in one collective."""
    from vllm.distributed import tensor_model_parallel_all_gather

    num_windows = len(hs_windows)
    has_res = all(r is not None for r in res_windows)
    if not has_res:
        combined = torch.cat(hs_windows, dim=-1)
        gathered = tensor_model_parallel_all_gather(combined, 0).narrow(
            0, 0, window_tokens
        )
        new_hs = list(gathered.split([hidden_size] * num_windows, dim=-1))
        return new_hs, [None] * num_windows

    res_tensors = [r for r in res_windows if r is not None]
    combined = torch.cat(hs_windows + res_tensors, dim=-1)
    gathered = tensor_model_parallel_all_gather(combined, 0).narrow(
        0, 0, window_tokens
    )
    splits = list(gathered.split([hidden_size] * (2 * num_windows), dim=-1))
    return splits[:num_windows], list(splits[num_windows:])


def _sp_attn_pre_norm_quant(
    layer: Any,
    hs_w: torch.Tensor,
    res_w: torch.Tensor | None,
    pos_w: torch.Tensor,
    *,
    input_is_sp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Substage 1a-local (SP): input_layernorm + optional FP8 quantize_rows on TensorCore."""
    from vllm_torchtpu.layers.vllm import linear_common
    from vllm_torchtpu.layers.vllm import sequence_parallel as _sp
    from vllm_torchtpu.layers.vllm import small_all_reduce_op as _sar

    if hasattr(layer, "_tpu_sp_rank"):
        pos_w = pos_w + (layer._tpu_sp_rank * 0)[0] + _sar.zero_from_rank()

    if res_w is None:
        res_w = hs_w
        hs_w = layer.input_layernorm(hs_w)
    else:
        hs_w, res_w = layer.input_layernorm(hs_w, res_w)

    prequant_local: tuple[torch.Tensor, torch.Tensor] | None = None
    if input_is_sp and _sp.fp8_gather():
        x_q, x_s = linear_common.quantize_rows(hs_w)
        prequant_local = (x_q, x_s)

    return hs_w, res_w, pos_w, prequant_local


def _sp_attn_all_gather(
    hs_w: torch.Tensor,
    prequant_local: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    input_is_sp: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Substage 1a-collective (SP): TP all_gather of prequantized or bf16 activations."""
    from vllm.distributed import tensor_model_parallel_all_gather

    if prequant_local is not None:
        x_q = tensor_model_parallel_all_gather(prequant_local[0], 0)
        x_s = tensor_model_parallel_all_gather(prequant_local[1], 0)
        return x_q, (x_q, x_s)
    if input_is_sp:
        hs_w = tensor_model_parallel_all_gather(hs_w, 0)
    return hs_w, None


def _sp_attn_pre_gather(
    layer: Any,
    hs_w: torch.Tensor,
    res_w: torch.Tensor | None,
    pos_w: torch.Tensor,
    *,
    input_is_sp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Substage 1a (SP): input_layernorm + optional FP8 quantize_rows + TP all_gather."""
    hs_w, res_w, pos_w, pq_local = _sp_attn_pre_norm_quant(
        layer, hs_w, res_w, pos_w, input_is_sp=input_is_sp
    )
    hs_w, prequant = _sp_attn_all_gather(
        hs_w, pq_local, input_is_sp=input_is_sp
    )
    return hs_w, res_w, pos_w, prequant


def _sp_attn_compute(
    layer: Any,
    hs_w: torch.Tensor,
    pos_w: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Substage 1b (SP): linear_attn or self_attn compute + optional attn_layer_scale."""
    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention" or (
        layer_type is None and hasattr(layer, "linear_attn")
    ):
        attn_out = (
            layer.linear_attn(hidden_states=hs_w)
            if prequant is None
            else layer.linear_attn(hidden_states=hs_w, prequant=prequant)
        )
    elif layer_type == "full_attention" or (
        layer_type is None and hasattr(layer, "self_attn")
    ):
        attn_out = (
            layer.self_attn(hidden_states=hs_w, positions=pos_w)
            if prequant is None
            else layer.self_attn(
                hidden_states=hs_w, positions=pos_w, prequant=prequant
            )
        )
    else:
        raise ValueError(f"Unsupported layer_type={layer_type!r}")

    if getattr(layer, "layer_scale", False):
        attn_out = _apply_layer_scale(attn_out, layer.attn_layer_scale)
    return attn_out


def _sp_attn_reduce_scatter(
    layer: Any,
    attn_out_w: torch.Tensor,
    res_w: torch.Tensor,
    window_tokens: int,
    *,
    input_is_sp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Substage 1c (SP): TP reduce_scatter on attention output + Layer-0 residual slice."""
    from vllm.distributed import tensor_model_parallel_reduce_scatter

    hs_sp_w = tensor_model_parallel_reduce_scatter(attn_out_w, 0)
    if not input_is_sp:
        tp = _get_tp_size(layer)
        if hasattr(layer, "_tpu_sp_rank") and tp > 1:
            res_w = (
                res_w.reshape(tp, -1, res_w.shape[-1])
                .index_select(0, layer._tpu_sp_rank)
                .squeeze(0)
            )
        elif tp > 1:
            res_w = _sp_chunk(res_w, window_tokens)
    return hs_sp_w, res_w


def _compute_shared_expert_only(
    mlp: Any,
    hs_w: torch.Tensor,
) -> torch.Tensor | None:
    """Run ``mlp.shared_expert`` on ``hs_w``, applying ``shared_expert_gate`` only if not already inside ``shared_expert``."""
    shared_expert = getattr(mlp, "shared_expert", None)
    if shared_expert is None:
        return None
    shared_out_w = shared_expert(hs_w)
    if isinstance(shared_out_w, tuple):
        shared_out_w = shared_out_w[0]
    # Qwen2MoeMLP / Qwen3NextMLP in vLLM already applies expert_gate inside forward()
    # when self.expert_gate is not None. Only apply mlp.shared_expert_gate when
    # shared_expert does not own expert_gate (e.g. unit-test stubs).
    if (
        getattr(shared_expert, "expert_gate", None) is None
        and getattr(mlp, "shared_expert_gate", None) is not None
    ):
        gate_val = mlp.shared_expert_gate(hs_w)
        if isinstance(gate_val, tuple):
            gate_val = gate_val[0]
        shared_out_w = F.sigmoid(gate_val) * shared_out_w
    return shared_out_w


def _call_routed_experts_monolithic(
    mlp: Any,
    hs: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Invoke only the routed experts (without re-running gate or shared_expert)."""
    experts = getattr(mlp, "experts", None)
    runner = _get_moe_runner(experts)
    routed = getattr(runner, "routed_experts", getattr(experts, "routed_experts", None))
    if routed is not None and hasattr(routed, "forward_monolithic"):
        if hasattr(routed, "_ensure_moe_quant_config_init"):
            routed._ensure_moe_quant_config_init()
        with _maybe_sp_context(runner):
            return routed.forward_monolithic(x=hs, router_logits=router_logits)
    return experts(hidden_states=hs, router_logits=router_logits)


def _sp_moe_pre_compute(
    layer: Any,
    hs_sp_w: torch.Tensor,
    res_w: torch.Tensor,
    *,
    include_shared: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Substage 2a (SP): post_attention_layernorm + gate (+ optional shared_expert) on SP shard."""
    hs_sp_w, res_w = layer.post_attention_layernorm(hs_sp_w, res_w)
    mlp = layer.mlp
    experts = getattr(mlp, "experts", None)
    runner = _get_moe_runner(experts)
    routed = getattr(runner, "routed_experts", getattr(experts, "routed_experts", None))
    if routed is not None and hasattr(routed, "_ensure_moe_quant_config_init"):
        routed._ensure_moe_quant_config_init()
    gate_out = mlp.gate(hs_sp_w)
    router_logits_w = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    shared_out_w = (
        _compute_shared_expert_only(mlp, hs_sp_w) if include_shared else None
    )
    return hs_sp_w, res_w, router_logits_w, shared_out_w


def _run_layer_sp_staggered(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    orig_padding: token_padding.TokenPaddingState | None,
    fc: Any,
    *,
    input_is_sp: bool,
    merge_moe: bool,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one sequence-parallel decoder layer with 3-substage attention overlap + merged Fused EP v3 MoE."""
    num_windows = len(hs_windows)
    window_tokens = pos_windows[0].shape[-1]

    hs_0_local, res_prev, pos_prev, pq_0_local = _sp_attn_pre_norm_quant(
        layer,
        hs_windows[0],
        res_windows[0],
        pos_windows[0],
        input_is_sp=input_is_sp,
    )
    hs_prev, prequant_prev = _sp_attn_all_gather(
        hs_0_local,
        pq_0_local,
        input_is_sp=input_is_sp,
    )

    hs_sp_list: list[torch.Tensor] = []
    res_sp_list: list[torch.Tensor] = []

    for w in range(1, num_windows):
        # Compute window w's local input_layernorm + FP8 quantize_rows on TensorCore
        # BEFORE Fence A so it overlaps with window w-1's all_gather on SparseCore.
        hs_w_local, res_curr, pos_curr, pq_w_local = _sp_attn_pre_norm_quant(
            layer,
            hs_windows[w],
            res_windows[w],
            pos_windows[w],
            input_is_sp=input_is_sp,
        )

        # Fence A: couple w-1's all-gather outputs with w's pre-gather local tensors
        # so w's all_gather on SparseCore runs concurrently with w-1's attention on TensorCore.
        if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            if prequant_prev is not None and pq_w_local is not None:
                x_q_prev, x_q_w_local = microbatch_fence_pair(
                    prequant_prev[0], pq_w_local[0]
                )
                x_s_prev, x_s_w_local = microbatch_fence_pair(
                    prequant_prev[1], pq_w_local[1]
                )
                prequant_prev = (x_q_prev, x_s_prev)
                hs_prev = x_q_prev
                pq_w_local = (x_q_w_local, x_s_w_local)
            elif prequant_prev is not None:
                x_q_prev, hs_w_local = microbatch_fence_pair(
                    prequant_prev[0], hs_w_local
                )
                x_s_prev, hs_w_local = microbatch_fence_pair(
                    prequant_prev[1], hs_w_local
                )
                prequant_prev = (x_q_prev, x_s_prev)
                hs_prev = x_q_prev
            else:
                hs_prev, hs_w_local = microbatch_fence_pair(
                    hs_prev, hs_w_local
                )

        hs_curr, prequant_curr = _sp_attn_all_gather(
            hs_w_local,
            pq_w_local,
            input_is_sp=input_is_sp,
        )
        with _window_context(fc, md_windows[w - 1], orig_padding):
            attn_out_prev = _sp_attn_compute(
                layer, hs_prev, pos_prev, prequant_prev
            )

        # Fence B: couple w-1's attention output with w's all-gather outputs so
        # w-1's reduce_scatter on SparseCore runs concurrently with w's attention on TensorCore.
        if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            if prequant_curr is not None:
                attn_out_prev, x_q_curr = microbatch_fence_pair(
                    attn_out_prev, prequant_curr[0]
                )
                attn_out_prev, x_s_curr = microbatch_fence_pair(
                    attn_out_prev, prequant_curr[1]
                )
                prequant_curr = (x_q_curr, x_s_curr)
                hs_curr = x_q_curr
            else:
                attn_out_prev, hs_curr = microbatch_fence_pair(
                    attn_out_prev, hs_curr
                )

        hs_sp_prev, res_sp_prev = _sp_attn_reduce_scatter(
            layer,
            attn_out_prev,
            res_prev,
            window_tokens,
            input_is_sp=input_is_sp,
        )
        hs_sp_list.append(hs_sp_prev)
        res_sp_list.append(res_sp_prev)

        hs_prev, res_prev, pos_prev, prequant_prev = (
            hs_curr,
            res_curr,
            pos_curr,
            prequant_curr,
        )

    with _window_context(fc, md_windows[num_windows - 1], orig_padding):
        attn_out_last = _sp_attn_compute(
            layer, hs_prev, pos_prev, prequant_prev
        )

    # Fence C: couple last window's attention output with window 0's reduce-scattered
    # shard so last window's reduce_scatter overlaps with window 0's gate+shared_expert.
    if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE) and hs_sp_list:
        attn_out_last, hs_sp_list[0] = microbatch_fence_pair(
            attn_out_last, hs_sp_list[0]
        )

    hs_sp_last, res_sp_last = _sp_attn_reduce_scatter(
        layer,
        attn_out_last,
        res_prev,
        window_tokens,
        input_is_sp=input_is_sp,
    )
    hs_sp_list.append(hs_sp_last)
    res_sp_list.append(res_sp_last)

    router_logits_list: list[torch.Tensor] = []
    shared_out_list: list[torch.Tensor | None] = []

    for w in range(num_windows - 1):
        hs_sp_w, res_sp_w, rl_w, se_w = _sp_moe_pre_compute(
            layer, hs_sp_list[w], res_sp_list[w]
        )
        hs_sp_list[w] = hs_sp_w
        res_sp_list[w] = res_sp_w
        router_logits_list.append(rl_w)
        shared_out_list.append(se_w)

    # Fence D: close diamond around (last window reduce_scatter || earlier windows gate+SE).
    # Fence hs_sp, router_logits, AND shared_out of window num_windows-2 with hs_sp_last so
    # XLA must execute window 0's shared_expert concurrently with window 1's reduce_scatter.
    if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE) and router_logits_list:
        hs_sp_list[-2], hs_sp_list[-1] = microbatch_fence_pair(
            hs_sp_list[-2], hs_sp_list[-1]
        )
        router_logits_list[-1], hs_sp_list[-1] = microbatch_fence_pair(
            router_logits_list[-1], hs_sp_list[-1]
        )
        if shared_out_list[-1] is not None:
            se_prev, hs_sp_list[-1] = microbatch_fence_pair(
                shared_out_list[-1], hs_sp_list[-1]
            )
            shared_out_list[-1] = se_prev

    hs_sp_w, res_sp_w, rl_w, se_w = _sp_moe_pre_compute(
        layer,
        hs_sp_list[-1],
        res_sp_list[-1],
        include_shared=not merge_moe,
    )
    hs_sp_list[-1] = hs_sp_w
    res_sp_list[-1] = res_sp_w
    router_logits_list.append(rl_w)
    shared_out_list.append(se_w)

    mlp = layer.mlp
    if merge_moe:
        cat_hs_sp = torch.cat(hs_sp_list, dim=0)
        cat_router_logits = torch.cat(router_logits_list, dim=0)
        if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            cat_router_logits, hs_sp_last_se = microbatch_fence_pair(
                cat_router_logits, hs_sp_list[-1]
            )
        else:
            hs_sp_last_se = hs_sp_list[-1]
        cat_routed_out = _call_routed_experts_monolithic(
            mlp,
            cat_hs_sp,
            cat_router_logits,
        )
        shared_out_list[-1] = _compute_shared_expert_only(mlp, hs_sp_last_se)
        routed_outs = list(
            cat_routed_out.view(
                num_windows, -1, cat_routed_out.shape[-1]
            ).unbind(0)
        )
    else:
        routed_outs = [
            _call_routed_experts_monolithic(
                mlp,
                hs_sp_list[w],
                router_logits_list[w],
            )
            for w in range(num_windows)
        ]

    tp_size = _get_tp_size(layer)
    se_is_replicated = bool(
        getattr(mlp, "_fused_se_is_replicated", False)
        or getattr(mlp, "replicate_shared_expert", False)
        or getattr(mlp, "is_sequence_parallel", False)
    )
    routed_experts = getattr(mlp.experts, "routed_experts", mlp.experts)
    from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import (
        fused_moe_ep_supported as fused_moe_ep_v3_supported,
    )

    v3_owns_combine = envs.USE_MOE_FUSED_EP_KERNEL and (
        fused_moe_ep_v3_supported(routed_experts)
        or fused_moe_ep_v3_supported(getattr(routed_experts, "quant_method", None))
    )

    for w in range(num_windows):
        final_w = routed_outs[w]
        shared_w = shared_out_list[w]
        if shared_w is not None:
            if se_is_replicated:
                final_w = final_w + shared_w
            elif tp_size > 1 and v3_owns_combine:
                shared_w = tensor_model_parallel_all_reduce(shared_w)
                final_w = final_w + shared_w
            else:
                final_w = final_w + shared_w
                if tp_size > 1:
                    final_w = tensor_model_parallel_all_reduce(final_w)
        elif not se_is_replicated and not v3_owns_combine and tp_size > 1:
            final_w = tensor_model_parallel_all_reduce(final_w)
        if getattr(layer, "layer_scale", False):
            final_w = _apply_layer_scale(final_w, layer.ffn_layer_scale)
        hs_windows[w] = final_w
        res_windows[w] = res_sp_list[w]

    return hs_windows, res_windows


def _run_layer_dp_fused_ep_overlapped(
    layer: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    *,
    fence: FenceMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one DP+EP decoder layer with unsplit Attention and substage-fenced Fused-EP MoE + Shared-Expert overlap."""
    if hasattr(layer, "_tpu_sp_rank"):
        from vllm_torchtpu.layers.vllm import small_all_reduce_op as _sar

        positions = (
            positions + (layer._tpu_sp_rank * 0)[0] + _sar.zero_from_rank()
        )

    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)

    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention" or (
        layer_type is None and hasattr(layer, "linear_attn")
    ):
        hidden_states = layer.linear_attn(hidden_states=hidden_states)
    elif layer_type == "full_attention" or (
        layer_type is None and hasattr(layer, "self_attn")
    ):
        hidden_states = layer.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
    else:
        raise ValueError(f"Unsupported layer_type={layer_type!r}")

    if getattr(layer, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, layer.attn_layer_scale)

    hidden_states, residual = layer.post_attention_layernorm(
        hidden_states, residual
    )
    mlp = layer.mlp
    experts = getattr(mlp, "experts", None)
    runner = _get_moe_runner(experts)
    routed = getattr(
        runner, "routed_experts", getattr(experts, "routed_experts", None)
    )
    if routed is not None and hasattr(routed, "_ensure_moe_quant_config_init"):
        routed._ensure_moe_quant_config_init()

    gate_out = mlp.gate(hidden_states)
    router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out

    if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
        router_logits, hs_se = microbatch_fence_pair(
            router_logits, hidden_states
        )
    else:
        hs_se = hidden_states

    # Emit routed_experts.forward_monolithic FIRST so select_top_k and
    # fused_ep_transport_start launch ICI remote DMAs immediately, and emit
    # shared_expert SECOND so its TensorCore matmuls overlap with
    # scatter_row_tables / finish_call / gather_rows_prefix / gather_rows_and_scales.
    routed_out = _call_routed_experts_monolithic(
        mlp, hidden_states, router_logits
    )
    shared_out = _compute_shared_expert_only(mlp, hs_se)

    if shared_out is not None:
        hidden_states = routed_out + shared_out
    else:
        hidden_states = routed_out

    if getattr(layer, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, layer.ffn_layer_scale)

    return hidden_states, residual


def _forward_dp_fused_ep_overlapped(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    fence: FenceMode,
) -> tuple[torch.Tensor | IntermediateTensors, tuple[torch.Tensor, ...]]:
    """Forward loop for DP+EP Fused-EP v3 layers with substage-fenced MoE + Shared-Expert overlap."""
    active_layers = _active_layers(self)
    aux_hidden_states = []
    use_aux = getattr(self, "aux_hidden_state_layers", ())
    for layer_idx, layer in enumerate(
        active_layers,
        start=getattr(self, "start_layer", 0),
    ):
        if layer_idx in use_aux:
            assert residual is not None
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = _run_layer_dp_fused_ep_overlapped(
            layer,
            hidden_states,
            residual,
            positions,
            fence=fence,
        )

    pp_group = getattr(_ps, "_PP", None)
    is_last_rank = pp_group.is_last_rank if pp_group is not None else True
    if not is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        }), tuple(aux_hidden_states)

    if getattr(self, "norm", None) is not None:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states, tuple(aux_hidden_states)


def _forward_unsplit(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor | IntermediateTensors, tuple[torch.Tensor, ...]]:
    """Standard unsplit forward loop over ``self.layers`` (including row-gated SP)."""
    from vllm_torchtpu.layers.vllm import sequence_parallel as _sp

    full_num_tokens = positions.shape[-1]
    active_layers = _active_layers(self)
    sp = (
        any(
            getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            for layer in active_layers
        )
        and _sp.enabled()
        and full_num_tokens >= _sp.min_rows()
    )
    state_is_sp = False
    hidden_size = (
        getattr(getattr(self, "config", None), "hidden_size", None)
        or hidden_states.shape[-1]
    )

    aux_hidden_states = []
    use_aux = getattr(self, "aux_hidden_state_layers", ())
    for layer_idx, layer in enumerate(
        active_layers,
        start=getattr(self, "start_layer", 0),
    ):
        if layer_idx in use_aux:
            if state_is_sp:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    hidden_size,
                )
                state_is_sp = False
            assert residual is not None
            aux_hidden_states.append(hidden_states + residual)
        if state_is_sp and not getattr(
            layer, "use_attn_reduce_scatter_for_moe", False
        ):
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                hidden_size,
            )
            state_is_sp = False
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        state_is_sp = sp and bool(
            getattr(layer, "use_attn_reduce_scatter_for_moe", False)
        )

    pp_group = getattr(_ps, "_PP", None)
    is_last_rank = pp_group.is_last_rank if pp_group is not None else True

    if not is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        }), tuple(aux_hidden_states)

    if state_is_sp:
        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            full_num_tokens,
            hidden_size,
        )

    if getattr(self, "norm", None) is not None:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states, tuple(aux_hidden_states)


def _microbatch_model_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor,
                                                list[torch.Tensor]]:
    """Instrumented ``Qwen3NextModel.forward`` supporting token-window microbatching."""
    config: MicroBatchConfig | None = getattr(self, "_microbatch_config", None)
    if config is None or config.factor <= 1:
        orig_fn = getattr(self, "_orig_unpatched_forward", None)
        if orig_fn is not None:
            return orig_fn(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        orig_cls_fn = getattr(type(self), "_orig_class_forward", None)
        if orig_cls_fn is not None:
            return orig_cls_fn(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        return _orig_qwen3_next_model_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    pp_group = getattr(_ps, "_PP", None)
    is_first_rank = pp_group.is_first_rank if pp_group is not None else True

    if is_first_rank or intermediate_tensors is None:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    fc = getattr(_fc_mod, "_forward_context", None)
    orig_md = fc.attn_metadata if fc is not None else None
    orig_padding = token_padding.get_padding_state()

    # Touch step-level inputs before branching on num_tokens so small (< min_tokens)
    # and large (>= min_tokens) traces register placeholders in identical order.
    num_tokens = hidden_states.shape[0]
    positions, restore_inputs = bind_step_inputs_for_trace_alignment(
        positions,
        orig_md,
        orig_padding,
        num_tokens=num_tokens,
    )
    try:
        from vllm_torchtpu.layers.vllm import sequence_parallel as _sp
        from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import (
            fused_moe_ep_supported as fused_moe_ep_v3_supported,
        )

        active_layers = _active_layers(self)
        first_layer = active_layers[0] if active_layers else None
        tp_size = _get_tp_size(first_layer)
        sp_enabled = _sp.enabled() and any(
            getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            for layer in active_layers
        )
        first_mlp = getattr(first_layer, "mlp", None)
        first_experts = getattr(first_mlp, "experts", None)
        first_runner = _get_moe_runner(first_experts)
        first_routed = getattr(
            first_runner,
            "routed_experts",
            getattr(first_experts, "routed_experts", None),
        )
        dp_fused_ep = (
            not sp_enabled
            and tp_size == 1
            and bool(envs.USE_MOE_FUSED_EP_KERNEL)
            and first_routed is not None
            and (
                fused_moe_ep_v3_supported(first_routed)
                or fused_moe_ep_v3_supported(
                    getattr(first_routed, "quant_method", None)
                )
            )
        )
        if dp_fused_ep:
            dp_min_tokens = max(
                int(envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS or 0), 33
            )
            if num_tokens < dp_min_tokens:
                out, aux = _forward_unsplit(
                    self, positions, hidden_states, residual
                )
            else:
                out, aux = _forward_dp_fused_ep_overlapped(
                    self,
                    positions,
                    hidden_states,
                    residual,
                    fence=config.resolved_fence,
                )
            if getattr(self, "aux_hidden_state_layers", ()):
                return out, list(aux)
            return out

        factor = config.factor
        if num_tokens < config.min_tokens or num_tokens % factor != 0:
            out, aux = _forward_unsplit(self, positions, hidden_states,
                                        residual)
            if getattr(self, "aux_hidden_state_layers", ()):
                return out, list(aux)
            return out

        # Force aot_autograd's ShapeEnv to bind s0 = bucket_size before s0 // factor
        # is evaluated so window shapes are concrete Python ints in StaticCompiler.
        hidden_states = hidden_states.view(hidden_states.size(0),
                                           *hidden_states.shape[1:])
        if residual is not None:
            residual = residual.view(residual.size(0), *residual.shape[1:])

        window_tokens = num_tokens // factor
        hs_windows = _split_tokens(hidden_states, factor)
        res_windows: list[torch.Tensor
                          | None] = ([None] * factor if residual is None else
                                     _split_tokens(residual, factor))
        pos_windows = _split_positions(positions, factor)
        md_windows = [
            slice_forward_metadata(orig_md, w, window_tokens)
            for w in range(factor)
        ]

        from vllm_torchtpu.layers.vllm import sequence_parallel as _sp

        active_layers = _active_layers(self)
        first_layer = active_layers[0] if active_layers else None
        tp_size = _get_tp_size(first_layer)
        sp = (
            any(
                getattr(layer, "use_attn_reduce_scatter_for_moe", False)
                for layer in active_layers
            )
            and _sp.enabled()
            and num_tokens >= _sp.min_rows()
            and window_tokens % max(1, tp_size) == 0
        )
        state_is_sp = False
        hidden_size = (
            getattr(getattr(self, "config", None), "hidden_size", None)
            or hidden_states.shape[-1]
        )

        pad_windows = (
            [orig_padding] * factor
            if sp
            else [
                slice_padding_state(orig_padding, w, window_tokens)
                for w in range(factor)
            ]
        )

        schedule = config.resolved_schedule
        fence = config.resolved_fence
        merge_moe = config.merge_moe
        if not merge_moe and config.split_moe_min_tokens > 0:
            merge_moe = int(hidden_states.size(0)) < config.split_moe_min_tokens

        aux_hidden_states: list[torch.Tensor] = []
        use_aux = getattr(self, "aux_hidden_state_layers", ())
        start_layer = getattr(self, "start_layer", 0)

        for layer_idx, layer in enumerate(active_layers,
                                          start=start_layer):
            if layer_idx in use_aux:
                if state_is_sp:
                    hs_windows, res_windows = _all_gather_sp_windows(
                        hs_windows, res_windows, window_tokens, hidden_size
                    )
                    state_is_sp = False
                aux_windows = [
                    h + r if r is not None else h
                    for h, r in zip(hs_windows, res_windows)
                ]
                aux_hidden_states.append(torch.cat(aux_windows, dim=0))

            layer_uses_sp = sp and bool(
                getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            )
            if state_is_sp and not layer_uses_sp:
                hs_windows, res_windows = _all_gather_sp_windows(
                    hs_windows, res_windows, window_tokens, hidden_size
                )
                state_is_sp = False

            if layer_uses_sp:
                hs_windows, res_windows = _run_layer_sp_staggered(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    orig_padding,
                    fc,
                    input_is_sp=state_is_sp,
                    merge_moe=merge_moe,
                    fence=fence,
                )
                state_is_sp = True
            elif schedule == ScheduleMode.LAYERWISE:
                hs_windows, res_windows = _run_layer_layerwise(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    fence=fence,
                )
            elif schedule == ScheduleMode.STAGGERED or fence == FenceMode.SUBSTAGE:
                hs_windows, res_windows = _run_layer_substage_staggered(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    merge_moe=merge_moe,
                    fence=fence,
                )
            else:
                hs_windows, res_windows = _run_layer_stagewise(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    merge_moe=merge_moe,
                    fence=fence,
                )

        is_last_rank = pp_group.is_last_rank if pp_group is not None else True

        if is_last_rank and state_is_sp:
            hs_windows, res_windows = _all_gather_sp_windows(
                hs_windows, res_windows, window_tokens, hidden_size
            )
            state_is_sp = False

        hidden_states = torch.cat(hs_windows, dim=0)
        if any(r is not None for r in res_windows):
            residual = torch.cat([r for r in res_windows if r is not None],
                                 dim=0)
        else:
            residual = None

        if not is_last_rank:
            out: torch.Tensor | IntermediateTensors = IntermediateTensors({
                "hidden_states":
                hidden_states,
                "residual":
                residual,
            })
        elif getattr(self, "norm", None) is not None:
            if residual is None:
                out = self.norm(hidden_states)
            else:
                out, _ = self.norm(hidden_states, residual)
        else:
            out = hidden_states

        if use_aux:
            return out, aux_hidden_states
        return out
    finally:
        restore_inputs()


# ============================================================================
# Class Patching & Model Installation Entry Points
# ============================================================================

_orig_qwen3_next_model_forward: Any = None
_qwen_classes_patched: bool = False


def _instrument_decoder_layer_class(layer_cls: type[nn.Module]) -> None:
    """Attach stage and sub-stage methods onto a decoder layer class if missing."""
    methods = {
        "forward_attn_compute": _layer_forward_attn_compute,
        "forward_attn_reduce": _layer_forward_attn_reduce,
        "forward_attn": _layer_forward_attn,
        "forward_moe_pre_compute": _layer_forward_moe_pre_compute,
        "forward_moe_pre_dispatch_route":
        _layer_forward_moe_pre_dispatch_route,
        "forward_moe_pre": _layer_forward_moe_pre,
        "forward_moe_kernel": _layer_forward_moe_kernel,
        "forward_moe_post": _layer_forward_moe_post,
        "forward_mlp": _layer_forward_mlp,
    }
    for name, fn in methods.items():
        if not hasattr(layer_cls, name):
            setattr(layer_cls, name, fn)


def _instrument_moe_block_class(moe_cls: type[nn.Module]) -> None:
    """Attach decomposed MoE sub-stage methods onto a SparseMoeBlock class."""
    methods = {
        "forward_moe_pre_compute": _moe_forward_pre_compute,
        "forward_moe_pre_dispatch_route": _moe_forward_pre_dispatch_route,
        "forward_moe_pre": _moe_forward_pre,
        "forward_moe_kernel": _moe_forward_kernel,
        "forward_moe_post": _moe_forward_post,
    }
    for name, fn in methods.items():
        if not hasattr(moe_cls, name):
            setattr(moe_cls, name, fn)


def patch_qwen_microbatch() -> None:
    """Instrument ``Qwen3NextSparseMoeBlock``, ``Qwen3NextDecoderLayer``, and ``Qwen3NextModel``.

    Registered in ``patch_registry.py`` at ``_ACTIVE`` stages so that
    ``TorchCompileWithNoGuardsWrapper`` captures the instrumented
    ``Qwen3NextModel.forward`` bytecode when the model is constructed.
    When ``_microbatch_config`` is not installed on the model instance,
    ``Qwen3NextModel.forward`` delegates directly to its original unpatched
    implementation with zero overhead.
    """
    global _orig_qwen3_next_model_forward, _qwen_classes_patched
    if _qwen_classes_patched:
        return

    import vllm_torchtpu
    vllm_torchtpu._patch_qwen3_next_sequence_parallel_by_rows()
    from vllm.model_executor.models.qwen3_next import (Qwen3NextDecoderLayer,
                                                       Qwen3NextModel,
                                                       Qwen3NextSparseMoeBlock)

    _instrument_moe_block_class(Qwen3NextSparseMoeBlock)
    _instrument_decoder_layer_class(Qwen3NextDecoderLayer)

    if _orig_qwen3_next_model_forward is None:
        _orig_qwen3_next_model_forward = Qwen3NextModel.forward
    Qwen3NextModel.forward = _microbatch_model_forward
    _qwen_classes_patched = True


def _find_inner_model(model: nn.Module) -> nn.Module | None:
    """Locate the inner decoder backbone (``Qwen3NextModel`` or module with ``.layers``)."""
    visited: set[int] = set()
    queue: list[nn.Module] = [model]
    while queue:
        curr = queue.pop(0)
        if id(curr) in visited:
            continue
        visited.add(id(curr))
        layers = getattr(curr, "layers", None)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return curr
        for attr in ("model", "language_model"):
            child = getattr(curr, attr, None)
            if isinstance(child, nn.Module):
                queue.append(child)
    return None


def unsupported_config_reasons(
    model: nn.Module,
    vllm_config: VllmConfig | None = None,
) -> list[str]:
    """Return human-readable reasons why ``model`` / ``vllm_config`` cannot use microbatching."""
    reasons: list[str] = []
    if vllm_config is not None:
        p_cfg = getattr(vllm_config, "parallel_config", None)
        if p_cfg is not None:
            if getattr(p_cfg, "prefill_context_parallel_size", 1) > 1:
                reasons.append("prefill_context_parallel_size > 1")
            if getattr(p_cfg, "decode_context_parallel_size", 1) > 1:
                reasons.append("decode_context_parallel_size > 1")

    if envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0:
        reasons.append("TPU_MOE_COLLECTION_CHUNK_SIZE > 0")

    inner = _find_inner_model(model)
    if inner is None:
        if envs.USE_MOE_FUSED_EP_KERNEL:
            reasons.append("USE_MOE_FUSED_EP_KERNEL=1")
        reasons.append("model has no decoder layers list")
        return reasons

    from vllm_torchtpu.layers.vllm import sequence_parallel as _sp
    from vllm_torchtpu.layers.vllm.fused_moe_ep import (
        fused_moe_ep_supported as fused_moe_ep_v2_supported,
    )
    from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import (
        fused_moe_ep_supported as fused_moe_ep_v3_supported,
    )

    active_layers = _active_layers(inner)
    first_layer = active_layers[0] if active_layers else None
    tp_size = _get_tp_size(first_layer)
    sp_supported = _sp.enabled() and any(
        getattr(layer, "use_attn_reduce_scatter_for_moe", False)
        for layer in active_layers
    )
    dp_v3_supported = (
        not sp_supported
        and tp_size == 1
        and bool(envs.USE_MOE_FUSED_EP_KERNEL)
        and any(
            fused_moe_ep_v3_supported(
                getattr(
                    _get_moe_runner(getattr(getattr(layer, "mlp", None), "experts", None)),
                    "routed_experts",
                    None,
                )
            )
            for layer in active_layers
        )
    )
    if envs.USE_MOE_FUSED_EP_KERNEL and not (sp_supported or dp_v3_supported):
        reasons.append("USE_MOE_FUSED_EP_KERNEL=1")

    for layer in active_layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            runner = _get_moe_runner(experts)
            moe_config = getattr(runner, "moe_config", None)
            if not sp_supported and (
                getattr(mlp, "is_sequence_parallel", False)
                or getattr(experts, "is_sequence_parallel", False)
                or getattr(moe_config, "is_sequence_parallel", False)
            ):
                reasons.append("FusedMoE.is_sequence_parallel=True")
                break
            if (getattr(experts, "has_ici_cp_collectives", False)
                    or getattr(moe_config, "pcp_size", 1) > 1):
                reasons.append("FusedMoE.has_ici_cp_collectives=True")
                break
            routed = getattr(runner, "routed_experts",
                             getattr(experts, "routed_experts", None))
            qm = getattr(routed, "quant_method", None)
            if not (sp_supported or dp_v3_supported) and (
                getattr(qm, "_use_fused_ep", False)
                or fused_moe_ep_v2_supported(qm)
            ):
                reasons.append("FusedMoE quant_method._use_fused_ep=True")
                break

    return reasons


def install_microbatch(
    model: nn.Module,
    config: MicroBatchConfig,
) -> bool:
    """Enable token-window micro-batching on ``model`` with ``config``."""
    patch_qwen_microbatch()
    inner = _find_inner_model(model)
    if inner is None:
        return False

    # Ensure custom decoder layers / MoE blocks (e.g. in unit tests) also have
    # the stage/sub-stage methods attached.
    for layer in _active_layers(inner):
        _instrument_decoder_layer_class(type(layer))
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, nn.Module):
            _instrument_moe_block_class(type(mlp))

    inner._microbatch_config = config
    model._microbatch_config = config

    if type(inner).forward is not _microbatch_model_forward:
        if not hasattr(type(inner), "_orig_class_forward"):
            type(inner)._orig_class_forward = type(inner).forward
        type(inner).forward = _microbatch_model_forward
        if hasattr(inner, "vllm_config") and hasattr(inner,
                                                     "_compiled_callable"):
            from vllm.compilation.wrapper import reset_compile_wrapper
            from vllm.config import set_current_vllm_config

            inner.cleanup()
            with set_current_vllm_config(inner.vllm_config):
                reset_compile_wrapper(inner)

    return True


def uninstall_microbatch(model: nn.Module) -> None:
    """Disable token-window micro-batching on ``model`` and restore original forward."""
    inner = _find_inner_model(model)
    for target in (model, inner):
        if target is None:
            continue
        if hasattr(target, "_microbatch_config"):
            delattr(target, "_microbatch_config")
        if hasattr(target, "_orig_unpatched_forward"):
            target.forward = target._orig_unpatched_forward
            delattr(target, "_orig_unpatched_forward")
        if hasattr(type(target), "_orig_class_forward"):
            type(target).forward = type(target)._orig_class_forward
            delattr(type(target), "_orig_class_forward")


def maybe_install_microbatch(
    model: nn.Module,
    vllm_config: VllmConfig | None = None,
    config: MicroBatchConfig | None = None,
) -> bool:
    """Validate config and install micro-batching on ``model`` if supported."""
    if config is None:
        config = MicroBatchConfig.from_envs()
    if config.factor <= 1:
        return False

    reasons = unsupported_config_reasons(model, vllm_config)
    if reasons:
        logger.warning(
            "Skipping TPU microbatching (factor=%d): unsupported configuration (%s).",
            config.factor,
            ", ".join(reasons),
        )
        return False

    installed = install_microbatch(model, config)
    if installed:
        logger.info(
            "Installed TPU Qwen microbatching: factor=%d, min_tokens=%d, "
            "schedule=%s, merge_moe=%s, split_moe_min_tokens=%d, fence=%s",
            config.factor,
            config.min_tokens,
            config.resolved_schedule.value,
            config.merge_moe,
            config.split_moe_min_tokens,
            config.resolved_fence.value,
        )
    return installed
