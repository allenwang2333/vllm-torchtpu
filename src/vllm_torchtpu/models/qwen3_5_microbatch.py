# SPDX-License-Identifier: Apache-2.0
"""TPU-owned staged adapter for Qwen3.5-MoE decoder layers.

The adapter mirrors Qwen3NextDecoderLayer.forward without modifying vLLM's
installed model. Decomposing the layer into explicit stages is what makes two
microbatch states visible at once, so one microbatch's expert compute can be
issued while the other's expert-parallel collectives are still in flight.

The split-phase collectives this relies on live in
``distributed.tpu_communicator``; the interleave itself is
``Qwen3_5TwoMicrobatchPipeline.forward_layers``, whose Python order is
deliberately static so Dynamo/XLA sees both microbatches in one graph.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


class _StaticMetadataSelector:
    index = 0


_MICROBATCH_METADATA_SELECTOR = _StaticMetadataSelector()


class TwoMicrobatchAttentionMetadata(dict[str, Any]):
    """Dict-compatible per-layer metadata selected by physical microbatch."""

    def __init__(self, metadata0: dict[str, Any], metadata1: dict[str, Any]):
        if metadata0.keys() != metadata1.keys():
            raise ValueError("microbatch attention layer sets must match")
        super().__init__(metadata0)
        self._metadata = (metadata0, metadata1)

    def __getitem__(self, layer_name: str) -> Any:
        return self._metadata[_MICROBATCH_METADATA_SELECTOR.index][layer_name]


@contextmanager
def select_microbatch_attention_metadata(index: int):
    if index not in (0, 1):
        raise ValueError("microbatch metadata index must be 0 or 1")
    previous = _MICROBATCH_METADATA_SELECTOR.index
    _MICROBATCH_METADATA_SELECTOR.index = index
    try:
        yield
    finally:
        _MICROBATCH_METADATA_SELECTOR.index = previous




@dataclass
class QwenAttentionStageOutput:
    moe_input: torch.Tensor
    residual: torch.Tensor


@dataclass
class QwenMoeStageOutput:
    hidden_states: torch.Tensor
    residual: torch.Tensor


class Qwen3_5MoeLayerAdapter:
    """Behavior-preserving stage view of one Qwen3.5-MoE decoder layer."""

    @staticmethod
    def validate_layer(layer: Any) -> None:
        if layer.__class__.__name__ != "Qwen3_5DecoderLayer":
            raise TypeError("adapter requires Qwen3_5DecoderLayer")
        if layer.layer_type not in ("full_attention", "linear_attention"):
            raise ValueError(f"unsupported Qwen3.5 layer type {layer.layer_type}")
        if not hasattr(layer.mlp, "experts"):
            raise TypeError("adapter requires the Qwen3.5 sparse-MoE block")

    @staticmethod
    def attention_stage(
        layer: Any,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
        microbatch_index: int | None = None,
    ) -> QwenAttentionStageOutput:
        """Input norm, full attention/GDN, residual, and post-attention norm."""
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)

        def run_attention() -> torch.Tensor:
            if layer.layer_type == "linear_attention":
                return layer.linear_attn(hidden_states=hidden_states)
            if layer.layer_type == "full_attention":
                if positions is None:
                    raise ValueError("full attention requires positions")
                return layer.self_attn(hidden_states=hidden_states,
                                       positions=positions)
            raise ValueError(
                f"unsupported Qwen3.5 layer type {layer.layer_type}")

        if microbatch_index is None:
            attention_output = run_attention()
        else:
            with select_microbatch_attention_metadata(microbatch_index):
                attention_output = run_attention()

        hidden_states = Qwen3_5MoeLayerAdapter._apply_attention_scale(
            layer, attention_output)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        return QwenAttentionStageOutput(hidden_states, residual)

    @staticmethod
    def moe_prepare_stage(
            attention: QwenAttentionStageOutput) -> QwenAttentionStageOutput:
        """Explicit boundary where routing/quantization/AG will be launched."""
        return attention

    @staticmethod
    def routed_expert_stage(layer: Any,
                            prepared: QwenAttentionStageOutput) -> torch.Tensor:
        """Run the existing sparse-MoE block, including both expert GMMs."""
        return layer.mlp(prepared.moe_input)

    @staticmethod
    def moe_finalize_stage(
        layer: Any,
        prepared: QwenAttentionStageOutput,
        expert_output: torch.Tensor,
    ) -> QwenMoeStageOutput:
        """Apply Qwen layer scaling and carry the residual to the next layer."""
        hidden_states = Qwen3_5MoeLayerAdapter._apply_ffn_scale(
            layer, expert_output)
        return QwenMoeStageOutput(hidden_states, prepared.residual)

    @classmethod
    def forward_layer(
        cls,
        layer: Any,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> QwenMoeStageOutput:
        """Serial reference composition of the staged layer interface."""
        cls.validate_layer(layer)
        attention = cls.attention_stage(layer, hidden_states, residual,
                                        positions)
        prepared = cls.moe_prepare_stage(attention)
        experts = cls.routed_expert_stage(layer, prepared)
        return cls.moe_finalize_stage(layer, prepared, experts)

    @staticmethod
    def _apply_attention_scale(layer: Any,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        if not layer.layer_scale:
            return hidden_states
        scale = layer.attn_layer_scale.to(hidden_states.dtype)
        if hidden_states.ndim == 2:
            scale = scale[0]
        return hidden_states * (scale + 1)

    @staticmethod
    def _apply_ffn_scale(layer: Any,
                         hidden_states: torch.Tensor) -> torch.Tensor:
        if not layer.layer_scale:
            return hidden_states
        scale = layer.ffn_layer_scale.to(hidden_states.dtype)
        if hidden_states.ndim == 2:
            scale = scale[0]
        else:
            assert hidden_states.ndim == scale.ndim, (
                f"shape must be the same {hidden_states.ndim}, {scale.ndim}")
        return hidden_states * (scale + 1)


@dataclass
class QwenAsyncMoePrepared:
    runner: Any
    hidden_gather: Any
    router_gather: Any
    shared_experts_input: torch.Tensor | None
    input_ids: torch.Tensor | None
    original_hidden_dim_pre_transform: int | None
    original_hidden_dim_post_transform: int | None


@dataclass
class QwenMoeRouterOutputs:
    """Local, collective-free result of the routed-input transform and gate."""

    runner: Any
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    shared_experts_input: torch.Tensor | None
    original_hidden_dim_pre_transform: int | None
    original_hidden_dim_post_transform: int | None


@dataclass
class QwenAsyncMoeExperts:
    prepared: QwenAsyncMoePrepared
    shared_output: torch.Tensor | None
    fused_output: torch.Tensor


@dataclass
class QwenAsyncMoeReduce:
    experts: QwenAsyncMoeExperts
    reduce_handle: Any


class Qwen3_5AsyncMoeAdapter:
    """Staged naive-DP/EP path around vLLM's existing expert kernel."""

    @staticmethod
    def _validate_runner(runner: Any) -> None:
        if not runner.do_naive_dispatch_combine:
            raise ValueError("async stages require naive DP/EP dispatch/combine")
        if runner.moe_config.is_sequence_parallel:
            raise ValueError("sequence-parallel MoE is not supported")
        if runner.moe_config.pcp_size != 1:
            raise ValueError("PCP MoE is not supported")

    @classmethod
    def route(cls, runner: Any,
              hidden_states: torch.Tensor) -> QwenMoeRouterOutputs:
        """Apply routed-input setup and the router gate. Launches no collective.

        This is the strict local prefix of :meth:`prepare`. It is factored out
        so the diagnostic router-only endpoint runs byte-identical routing work
        to the staged path instead of a re-implementation that could drift.
        """
        cls._validate_runner(runner)
        hidden_states, shared_input = runner.apply_routed_input_transform(
            hidden_states)
        hidden_states, pre_dim, post_dim = runner._maybe_pad_hidden_states(
            shared_input, hidden_states)
        runner.routed_experts._ensure_moe_quant_config_init()
        runner._maybe_sync_shared_experts_stream(shared_input)

        if runner.gate is None:
            raise ValueError("async Qwen3.5 path requires an external gate")
        if runner._fse_fuse_gate:
            runner._maybe_fuse_gate_weights()
            router_logits = torch.nn.functional.linear(
                hidden_states, runner._combined_gate_weight)
        else:
            router_logits, _ = runner.gate(hidden_states)

        return QwenMoeRouterOutputs(
            runner=runner,
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_input,
            original_hidden_dim_pre_transform=pre_dim,
            original_hidden_dim_post_transform=post_dim,
        )

    @classmethod
    def prepare(
        cls,
        runner: Any,
        hidden_states: torch.Tensor,
        microbatch_index: int,
        communicator: Any,
        input_ids: torch.Tensor | None = None,
    ) -> QwenAsyncMoePrepared:
        """Apply routed-input setup/router and launch hidden/router AGs."""
        routed = cls.route(runner, hidden_states)
        return QwenAsyncMoePrepared(
            runner=routed.runner,
            hidden_gather=communicator.all_gather_start(
                routed.hidden_states, microbatch_index),
            router_gather=communicator.all_gather_start(
                routed.router_logits, microbatch_index),
            shared_experts_input=routed.shared_experts_input,
            input_ids=input_ids,
            original_hidden_dim_pre_transform=routed.
            original_hidden_dim_pre_transform,
            original_hidden_dim_post_transform=routed.
            original_hidden_dim_post_transform,
        )

    @staticmethod
    def experts(prepared: QwenAsyncMoePrepared,
                communicator: Any) -> QwenAsyncMoeExperts:
        """Wait for dispatch and run routing plus both existing expert GMMs."""
        hidden_states = communicator.wait(prepared.hidden_gather)
        router_logits = communicator.wait(prepared.router_gather)
        shared_output, fused_output = prepared.runner._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=prepared.shared_experts_input,
            input_ids=prepared.input_ids,
        )
        return QwenAsyncMoeExperts(prepared, shared_output, fused_output)

    @staticmethod
    def reduce_start(experts: QwenAsyncMoeExperts, microbatch_index: int,
                     communicator: Any) -> QwenAsyncMoeReduce:
        """Launch routed-output reduce-scatter without waiting for it."""
        handle = communicator.reduce_scatter_start(
            experts.fused_output, microbatch_index)
        return QwenAsyncMoeReduce(experts, handle)

    @staticmethod
    def finalize(reduction: QwenAsyncMoeReduce,
                 communicator: Any) -> torch.Tensor:
        """Wait for RS and preserve vLLM shared/routed output transforms."""
        experts = reduction.experts
        prepared = experts.prepared
        runner = prepared.runner
        fused_output = communicator.wait(reduction.reduce_handle)
        if prepared.original_hidden_dim_pre_transform is not None:
            fused_output = fused_output[
                ..., :prepared.original_hidden_dim_pre_transform]

        shared_output = runner._maybe_reduce_shared_expert_output(
            experts.shared_output)
        shared_output, fused_output = (
            runner._maybe_apply_routed_scale_to_output(
                shared_output, fused_output))
        fused_output = runner.apply_routed_output_transform(fused_output)
        result = (shared_output + fused_output
                  if shared_output is not None else fused_output)
        result = runner._maybe_reduce_final_output(
            result, prepared.original_hidden_dim_post_transform)
        return runner._maybe_add_zero_expert_output(result)


@dataclass
class QwenMicrobatchState:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None


@dataclass
class QwenTwoMicrobatchOutput:
    microbatch0: QwenMicrobatchState
    microbatch1: QwenMicrobatchState


class Qwen3_5TwoMicrobatchPipeline:
    """Static two-microbatch dependency schedule across Qwen decoder layers."""

    @classmethod
    def forward_layers(
        cls,
        layers: Any,
        states: tuple[QwenMicrobatchState, QwenMicrobatchState],
        positions: tuple[torch.Tensor, torch.Tensor],
        communicator: Any,
        input_ids: tuple[torch.Tensor | None, torch.Tensor | None] = (None,
                                                                     None),
    ) -> QwenTwoMicrobatchOutput:
        """Run all layers while carrying MB1's pending RS across layers.

        The Python order is intentionally static so Dynamo/XLA sees both
        microbatches and can overlap independent operations in one graph.
        """
        state0, state1 = states
        pending1: tuple[Any, QwenAttentionStageOutput,
                        QwenAsyncMoeReduce] | None = None

        for layer in layers:
            Qwen3_5MoeLayerAdapter.validate_layer(layer)

            attention0 = Qwen3_5MoeLayerAdapter.attention_stage(
                layer, state0.hidden_states, state0.residual, positions[0], 0)
            prepared0 = Qwen3_5AsyncMoeAdapter.prepare(
                layer.mlp.experts, attention0.moe_input, 0, communicator,
                input_ids[0])

            # Preserve the previous layer's MB1 RS across next-layer MB0
            # attention and AG launch, then wait only when MB1 is needed.
            if pending1 is not None:
                previous_layer, previous_attention1, previous_reduce1 = pending1
                previous_expert1 = Qwen3_5AsyncMoeAdapter.finalize(
                    previous_reduce1, communicator)
                finalized1 = Qwen3_5MoeLayerAdapter.moe_finalize_stage(
                    previous_layer, previous_attention1, previous_expert1)
                state1 = QwenMicrobatchState(finalized1.hidden_states,
                                             finalized1.residual)

            attention1 = Qwen3_5MoeLayerAdapter.attention_stage(
                layer, state1.hidden_states, state1.residual, positions[1], 1)
            prepared1 = Qwen3_5AsyncMoeAdapter.prepare(
                layer.mlp.experts, attention1.moe_input, 1, communicator,
                input_ids[1])

            experts0 = Qwen3_5AsyncMoeAdapter.experts(prepared0, communicator)
            reduce0 = Qwen3_5AsyncMoeAdapter.reduce_start(
                experts0, 0, communicator)
            experts1 = Qwen3_5AsyncMoeAdapter.experts(prepared1, communicator)
            reduce1 = Qwen3_5AsyncMoeAdapter.reduce_start(
                experts1, 1, communicator)

            expert0 = Qwen3_5AsyncMoeAdapter.finalize(reduce0, communicator)
            finalized0 = Qwen3_5MoeLayerAdapter.moe_finalize_stage(
                layer, attention0, expert0)
            state0 = QwenMicrobatchState(finalized0.hidden_states,
                                         finalized0.residual)
            pending1 = (layer, attention1, reduce1)

        if pending1 is not None:
            last_layer, last_attention1, last_reduce1 = pending1
            last_expert1 = Qwen3_5AsyncMoeAdapter.finalize(
                last_reduce1, communicator)
            finalized1 = Qwen3_5MoeLayerAdapter.moe_finalize_stage(
                last_layer, last_attention1, last_expert1)
            state1 = QwenMicrobatchState(finalized1.hidden_states,
                                         finalized1.residual)

        return QwenTwoMicrobatchOutput(state0, state1)


@dataclass
class QwenTwoMicrobatchHiddenStates:
    hidden_states0: torch.Tensor
    hidden_states1: torch.Tensor


class Qwen3_5TwoMicrobatchModelAdapter:
    """Embedding, decoder pipeline, and final norm for two Qwen prefills."""

    @staticmethod
    def resolve_backbone(model: Any) -> Any:
        language_model = (model.get_language_model()
                          if hasattr(model, "get_language_model") else model)
        backbone = getattr(language_model, "model", None)
        if backbone is None:
            raise TypeError("Qwen3.5 model backbone was not found")
        if backbone.__class__.__name__ != "Qwen3_5Model":
            raise TypeError("adapter requires a Qwen3_5Model backbone")
        if backbone.start_layer != 0 or backbone.end_layer != len(
                backbone.layers):
            raise ValueError("pipeline-parallel Qwen3.5 is not supported")
        if getattr(backbone, "aux_hidden_state_layers", ()):
            raise ValueError("auxiliary hidden-state capture is not supported")
        return backbone

    @classmethod
    def forward(
        cls,
        model: Any,
        input_ids: tuple[torch.Tensor, torch.Tensor],
        positions: tuple[torch.Tensor, torch.Tensor],
        communicator: Any,
        inputs_embeds: tuple[torch.Tensor | None,
                             torch.Tensor | None] = (None, None),
    ) -> QwenTwoMicrobatchHiddenStates:
        """Run one static model invocation containing both microbatches."""
        backbone = cls.resolve_backbone(model)
        hidden0 = (inputs_embeds[0] if inputs_embeds[0] is not None else
                   backbone.embed_input_ids(input_ids[0]))
        hidden1 = (inputs_embeds[1] if inputs_embeds[1] is not None else
                   backbone.embed_input_ids(input_ids[1]))

        output = Qwen3_5TwoMicrobatchPipeline.forward_layers(
            backbone.layers,
            (QwenMicrobatchState(hidden0, None),
             QwenMicrobatchState(hidden1, None)),
            positions,
            communicator,
            input_ids,
        )
        hidden0, _ = backbone.norm(output.microbatch0.hidden_states,
                                   output.microbatch0.residual)
        hidden1, _ = backbone.norm(output.microbatch1.hidden_states,
                                   output.microbatch1.residual)
        return QwenTwoMicrobatchHiddenStates(hidden0, hidden1)
