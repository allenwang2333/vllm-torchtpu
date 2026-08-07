# SPDX-License-Identifier: Apache-2.0
"""Numerical equivalence of the staged two-microbatch path.

``tests/models/test_qwen3_5_microbatch_adapter.py`` proves the staged pipeline
issues its collectives in the intended *order*, by replacing every stage with a
recorder. Nothing there runs a number through the pipeline, so a change that
preserves the ordering while corrupting the arithmetic passes it.

This module closes that gap. It runs real tensors through
``Qwen3_5TwoMicrobatchPipeline.forward_layers`` -- the interleaved schedule --
and compares against a serial reference that drives the identical stage
functions one microbatch at a time. The two must agree exactly: the staged
schedule is a pure-dataflow rearrangement, so every op each microbatch sees is
the same op on the same values, only issued at a different time.

What that comparison is sensitive to, and nothing else in the suite is:

* the deferred finalize. ``forward_layers`` launches microbatch 1's
  reduce-scatter for layer L, carries ``(layer, attention, reduce)`` in
  ``pending1`` across layer L+1's microbatch-0 work, and finalizes it one
  iteration later. Finalizing against the *current* layer instead of the layer
  the reduction came from applies layer L+1's ``ffn_layer_scale`` to layer L's
  expert output. Wrong answer, no crash, no wedge, no throughput change. The
  layers here are built with distinct per-layer scales so that swap is visible,
  and ``test_comparison_detects_a_single_layer_scale_swap`` proves the check
  actually has that power rather than passing vacuously.
* slot crossing -- microbatch 0's residual, positions, attention metadata, or
  expert handles reaching microbatch 1.
* waiting the wrong collective handle. The fake communicator resolves nothing
  eagerly, so a mispaired wait returns another slot's tensor.

The last test is a different kind: it asserts the *schedule* of collectives is
identical for real+real, real+dummy, and dummy+dummy inputs. That is the
invariant whose violation halts all eight cores with no error message, and it
is data-dependence in the Python schedule that would break it.

Everything runs in float64 on CPU, so equality is exact and a failure is a real
disagreement rather than accumulated rounding.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.qwen3_5_microbatch import (
    Qwen3_5AsyncMoeAdapter, Qwen3_5MoeLayerAdapter,
    Qwen3_5TwoMicrobatchModelAdapter, Qwen3_5TwoMicrobatchPipeline,
    QwenMicrobatchState, TwoMicrobatchAttentionMetadata)

DTYPE = torch.float64
HIDDEN = 6
MOE_PAD = 2
NUM_EXPERTS = 4
NUM_TOKENS = 5

# ---------------------------------------------------------------------------
# A communicator that defers, so a mispaired wait is a wrong number
# ---------------------------------------------------------------------------


@dataclass
class _Handle:
    value: torch.Tensor
    op: str
    microbatch: int


class RecordingCommunicator:
    """Single-rank stand-in for the TPU expert-parallel communicator.

    With one rank an all-gather and a reduce-scatter are both the identity, so
    the reference and the staged path can be compared without modelling peers.
    What is modelled is the *asynchrony*: ``*_start`` hands back a handle and
    resolves nothing, so waiting the wrong handle yields the wrong tensor
    instead of quietly working.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int, tuple[int, ...]]] = []

    def all_gather_start(self, value: torch.Tensor, microbatch: int):
        self.events.append(("AG", microbatch, tuple(value.shape)))
        return _Handle(value, "AG", microbatch)

    def reduce_scatter_start(self, value: torch.Tensor, microbatch: int):
        self.events.append(("RS", microbatch, tuple(value.shape)))
        return _Handle(value, "RS", microbatch)

    def wait(self, handle: _Handle) -> torch.Tensor:
        self.events.append(
            (f"wait-{handle.op}", handle.microbatch, tuple(handle.value.shape)))
        return handle.value


# ---------------------------------------------------------------------------
# Layers whose arithmetic is real, distinct per layer, and non-commuting
# ---------------------------------------------------------------------------


def _weights(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=DTYPE)


class FusedNorm(torch.nn.Module):
    """Qwen's fused residual norm: folds the residual in, then normalizes."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(_weights(HIDDEN, seed=seed))

    def _norm(self, hidden: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
        return hidden * scale * self.weight

    def forward(self, hidden: torch.Tensor,
                residual: torch.Tensor | None = None):
        if residual is None:
            return self._norm(hidden)
        residual = residual + hidden
        return self._norm(residual), residual


class MetadataReadingAttention(torch.nn.Module):
    """Attention whose output depends on positions *and* on its metadata.

    The real attention reads its per-layer metadata out of the forward context,
    which ``TwoMicrobatchAttentionMetadata`` redirects by slot. Reading it here
    too is what makes a metadata mix-up a numerical failure rather than an
    invisible one.
    """

    def __init__(self, layer_name: str,
                 metadata: TwoMicrobatchAttentionMetadata,
                 seed: int) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.metadata = metadata
        self.seen: list[str] = []
        self.weight = torch.nn.Parameter(_weights(HIDDEN, HIDDEN, seed=seed))

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        entry = self.metadata[self.layer_name]
        self.seen.append(entry.tag)
        value = torch.tanh(hidden_states @ self.weight) * entry.scale
        if positions is not None:
            # Added outside the tanh and scaled down: inside it, a position of
            # 50 already saturates in float64, so two different position ranges
            # would produce identical output and the crossing test would pass
            # without testing anything.
            value = value + positions.unsqueeze(-1).to(value.dtype) * 0.01
        return value


class MoeRunner(torch.nn.Module):
    """Sparse-MoE runner implementing the contract ``Qwen3_5AsyncMoeAdapter``
    drives: routed-input transform, padding, gate, expert GMM, and the output
    transforms applied after the reduce-scatter.

    Routing is per token, as the real kernel's is, so nothing here can make one
    token's answer depend on another token's presence. That keeps a failure
    attributable to the staged schedule rather than to the fake.
    """

    do_naive_dispatch_combine = True
    _fse_fuse_gate = False

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.moe_config = SimpleNamespace(is_sequence_parallel=False,
                                          pcp_size=1)
        self.routed_experts = SimpleNamespace(
            _ensure_moe_quant_config_init=lambda: None)
        self.routed_weight = _weights(HIDDEN, HIDDEN, seed=seed)
        self.shared_weight = _weights(HIDDEN, HIDDEN, seed=seed + 1)
        self.shared_out_weight = _weights(HIDDEN, HIDDEN, seed=seed + 2)
        self.expert_weight = _weights(NUM_EXPERTS, HIDDEN + MOE_PAD,
                                      HIDDEN + MOE_PAD, seed=seed + 3)
        self.gate_weight = _weights(HIDDEN + MOE_PAD, NUM_EXPERTS,
                                    seed=seed + 4)
        self.output_weight = _weights(HIDDEN, HIDDEN, seed=seed + 5)
        self.gate = lambda hidden: (hidden @ self.gate_weight, None)

    # -- local prefix, before any collective ---------------------------------

    def apply_routed_input_transform(self, hidden: torch.Tensor):
        return hidden @ self.routed_weight, hidden @ self.shared_weight

    def _maybe_pad_hidden_states(self, shared: torch.Tensor,
                                 hidden: torch.Tensor):
        padded = torch.nn.functional.pad(hidden, (0, MOE_PAD))
        # pre_dim drives the truncation back to HIDDEN after the reduce-scatter.
        return padded, HIDDEN, None

    def _maybe_sync_shared_experts_stream(self, shared: torch.Tensor) -> None:
        pass

    # -- expert compute ------------------------------------------------------

    def _apply_quant_method(self, hidden_states: torch.Tensor,
                            router_logits: torch.Tensor,
                            shared_experts_input: torch.Tensor,
                            input_ids: torch.Tensor | None):
        weights = torch.softmax(router_logits, dim=-1)
        chosen = weights.argmax(dim=-1)
        routed = torch.zeros_like(hidden_states)
        for expert in range(NUM_EXPERTS):
            mask = (chosen == expert).unsqueeze(-1)
            routed = torch.where(mask,
                                 torch.tanh(hidden_states
                                            @ self.expert_weight[expert]),
                                 routed)
        routed = routed * weights.max(dim=-1, keepdim=True).values
        shared = torch.tanh(shared_experts_input @ self.shared_out_weight)
        return shared, routed

    # -- transforms applied after the reduce-scatter -------------------------

    def _maybe_reduce_shared_expert_output(self, shared: torch.Tensor):
        return shared

    def _maybe_apply_routed_scale_to_output(self, shared: torch.Tensor,
                                            fused: torch.Tensor):
        return shared, fused * 0.5

    def apply_routed_output_transform(self, fused: torch.Tensor):
        return fused @ self.output_weight

    def _maybe_reduce_final_output(self, result: torch.Tensor,
                                   truncation: int | None):
        return result

    def _maybe_add_zero_expert_output(self, result: torch.Tensor):
        return result


class SparseMoeBlock(torch.nn.Module):
    """The non-staged block. Same primitives, composed synchronously.

    ``Qwen3_5MoeLayerAdapter.routed_expert_stage`` calls this; the staged path
    calls ``Qwen3_5AsyncMoeAdapter`` against ``self.experts`` instead. Writing
    it as the plain composition of the same runner methods is what makes
    ``test_split_phase_moe_matches_synchronous_composition`` a statement about
    the split-phase decomposition rather than about two unrelated formulas.
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.experts = MoeRunner(seed)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        runner = self.experts
        routed, shared_input = runner.apply_routed_input_transform(hidden)
        routed, pre_dim, post_dim = runner._maybe_pad_hidden_states(
            shared_input, routed)
        router_logits, _ = runner.gate(routed)
        shared, fused = runner._apply_quant_method(
            hidden_states=routed, router_logits=router_logits,
            shared_experts_input=shared_input, input_ids=None)
        fused = fused[..., :pre_dim]
        shared = runner._maybe_reduce_shared_expert_output(shared)
        shared, fused = runner._maybe_apply_routed_scale_to_output(
            shared, fused)
        fused = runner.apply_routed_output_transform(fused)
        result = shared + fused
        result = runner._maybe_reduce_final_output(result, post_dim)
        return runner._maybe_add_zero_expert_output(result)


class Qwen3_5DecoderLayer(torch.nn.Module):
    """Named to satisfy ``Qwen3_5MoeLayerAdapter.validate_layer``."""

    def __init__(self, index: int, layer_type: str,
                 metadata: TwoMicrobatchAttentionMetadata) -> None:
        super().__init__()
        seed = 100 * (index + 1)
        self.index = index
        self.layer_type = layer_type
        self.input_layernorm = FusedNorm(seed)
        self.post_attention_layernorm = FusedNorm(seed + 1)
        self.mlp = SparseMoeBlock(seed + 2)
        attention = MetadataReadingAttention(attention_layer_name(index),
                                             metadata, seed + 20)
        if layer_type == "full_attention":
            self.self_attn = attention
        else:
            self.linear_attn = attention
        # Distinct per layer on purpose: a deferred finalize that used the
        # wrong layer would apply the wrong one of these.
        self.layer_scale = True
        self.attn_layer_scale = torch.nn.Parameter(
            torch.full((1, 1, HIDDEN), 0.1 * (index + 1), dtype=DTYPE))
        self.ffn_layer_scale = torch.nn.Parameter(
            torch.full((1, 1, HIDDEN), 0.7 - 0.2 * index, dtype=DTYPE))

    @property
    def attention(self) -> MetadataReadingAttention:
        return (self.self_attn if self.layer_type == "full_attention"
                else self.linear_attn)


def attention_layer_name(index: int) -> str:
    return f"layer{index}.attn"


def make_metadata(num_layers: int) -> TwoMicrobatchAttentionMetadata:
    """Per-slot attention metadata whose value is readable in the output."""
    def side(slot: int) -> dict:
        return {
            attention_layer_name(index): SimpleNamespace(
                tag=f"mb{slot}",
                scale=torch.tensor(1.0 + 0.25 * slot, dtype=DTYPE))
            for index in range(num_layers)
        }

    return TwoMicrobatchAttentionMetadata(side(0), side(1))


def make_layers(num_layers: int, layer_type: str
                ) -> tuple[list[Qwen3_5DecoderLayer],
                           TwoMicrobatchAttentionMetadata]:
    metadata = make_metadata(num_layers)
    layers = [Qwen3_5DecoderLayer(index, layer_type, metadata)
              for index in range(num_layers)]
    return layers, metadata


def make_hidden(seed: int) -> torch.Tensor:
    return _weights(NUM_TOKENS, HIDDEN, seed=seed)


def make_positions(start: int) -> torch.Tensor:
    return torch.arange(start, start + NUM_TOKENS, dtype=torch.long)


# ---------------------------------------------------------------------------
# The serial reference: one microbatch, alone, through the same stages
# ---------------------------------------------------------------------------


def serial_microbatch(layers, hidden: torch.Tensor, positions: torch.Tensor,
                      slot: int, communicator: RecordingCommunicator,
                      input_ids: torch.Tensor | None = None
                      ) -> QwenMicrobatchState:
    """Run every layer for one microbatch, finalizing before moving on.

    This is the staged schedule with the interleave removed and nothing else
    changed, which is exactly the difference the comparison should be probing.
    """
    state = QwenMicrobatchState(hidden, None)
    for layer in layers:
        Qwen3_5MoeLayerAdapter.validate_layer(layer)
        attention = Qwen3_5MoeLayerAdapter.attention_stage(
            layer, state.hidden_states, state.residual, positions, slot)
        prepared = Qwen3_5AsyncMoeAdapter.prepare(
            layer.mlp.experts, attention.moe_input, slot, communicator,
            input_ids)
        experts = Qwen3_5AsyncMoeAdapter.experts(prepared, communicator)
        reduction = Qwen3_5AsyncMoeAdapter.reduce_start(
            experts, slot, communicator)
        expert_output = Qwen3_5AsyncMoeAdapter.finalize(reduction, communicator)
        finalized = Qwen3_5MoeLayerAdapter.moe_finalize_stage(
            layer, attention, expert_output)
        state = QwenMicrobatchState(finalized.hidden_states, finalized.residual)
    return state


def run_staged(layers, hidden0, hidden1, positions0, positions1,
               communicator=None):
    communicator = communicator or RecordingCommunicator()
    output = Qwen3_5TwoMicrobatchPipeline.forward_layers(
        layers,
        (QwenMicrobatchState(hidden0, None),
         QwenMicrobatchState(hidden1, None)),
        (positions0, positions1),
        communicator,
    )
    return output, communicator


def assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# The equivalence itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_layers", [1, 2, 5])
@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_staged_pipeline_matches_serial_reference_per_microbatch(
        num_layers, layer_type):
    """Both slots come out of the interleaved schedule unchanged.

    ``num_layers=1`` exercises only the post-loop finalize; ``2`` and ``5``
    exercise the cross-layer deferral, which is where microbatch 1's
    reduce-scatter is carried past another microbatch's work.
    """
    layers, _ = make_layers(num_layers, layer_type)
    hidden0, hidden1 = make_hidden(1), make_hidden(2)
    positions0, positions1 = make_positions(0), make_positions(50)

    staged, _ = run_staged(layers, hidden0, hidden1, positions0, positions1)
    expected0 = serial_microbatch(layers, hidden0, positions0, 0,
                                  RecordingCommunicator())
    expected1 = serial_microbatch(layers, hidden1, positions1, 1,
                                  RecordingCommunicator())

    assert_exact(staged.microbatch0.hidden_states, expected0.hidden_states)
    assert_exact(staged.microbatch0.residual, expected0.residual)
    assert_exact(staged.microbatch1.hidden_states, expected1.hidden_states)
    assert_exact(staged.microbatch1.residual, expected1.residual)


def test_comparison_detects_a_single_layer_scale_swap():
    """The equivalence check is not vacuous.

    Give layer 1 a different ``ffn_layer_scale`` between the staged run and the
    reference and the outputs must diverge. If they did not, the test above
    would pass no matter what the deferred finalize applied.
    """
    layers, _ = make_layers(3, "full_attention")
    hidden0, hidden1 = make_hidden(1), make_hidden(2)
    positions0, positions1 = make_positions(0), make_positions(50)

    staged, _ = run_staged(layers, hidden0, hidden1, positions0, positions1)

    mutated = copy.deepcopy(layers)
    with torch.no_grad():
        mutated[1].ffn_layer_scale.copy_(
            mutated[2].ffn_layer_scale)  # the exact swap the deferral risks
    perturbed = serial_microbatch(mutated, hidden1, positions1, 1,
                                  RecordingCommunicator())

    assert not torch.equal(staged.microbatch1.hidden_states,
                           perturbed.hidden_states)


@pytest.mark.parametrize("num_layers", [2, 5])
def test_slots_do_not_contaminate_each_other(num_layers):
    """Changing one slot's input leaves the other slot bit-identical."""
    layers, _ = make_layers(num_layers, "full_attention")
    positions0, positions1 = make_positions(0), make_positions(50)
    baseline, _ = run_staged(layers, make_hidden(1), make_hidden(2),
                             positions0, positions1)

    changed_slot1, _ = run_staged(layers, make_hidden(1), make_hidden(99),
                                  positions0, positions1)
    assert_exact(changed_slot1.microbatch0.hidden_states,
                 baseline.microbatch0.hidden_states)
    assert not torch.equal(changed_slot1.microbatch1.hidden_states,
                           baseline.microbatch1.hidden_states)

    changed_slot0, _ = run_staged(layers, make_hidden(99), make_hidden(2),
                                  positions0, positions1)
    assert_exact(changed_slot0.microbatch1.hidden_states,
                 baseline.microbatch1.hidden_states)
    assert not torch.equal(changed_slot0.microbatch0.hidden_states,
                           baseline.microbatch0.hidden_states)


def test_a_dummy_partner_does_not_change_the_real_microbatch():
    """A real request must not read differently for being paired with filler.

    Every staged step runs two physical slots; when only one request is
    eligible the other carries a zero-filled dummy whose output is discarded.
    The real slot's answer has to be the same either way, or the feature would
    change results depending on how the scheduler happened to batch.
    """
    layers, _ = make_layers(3, "full_attention")
    positions0, positions1 = make_positions(0), make_positions(50)
    real = make_hidden(1)

    with_dummy, _ = run_staged(layers, real, torch.zeros_like(real),
                               positions0, torch.zeros_like(positions1))
    with_partner, _ = run_staged(layers, real, make_hidden(2), positions0,
                                 positions1)
    alone = serial_microbatch(layers, real, positions0, 0,
                              RecordingCommunicator())

    assert_exact(with_dummy.microbatch0.hidden_states, alone.hidden_states)
    assert_exact(with_partner.microbatch0.hidden_states, alone.hidden_states)


def test_positions_are_not_crossed_between_slots():
    """Slot 1's positions must not reach slot 0's attention."""
    layers, _ = make_layers(2, "full_attention")
    hidden0, hidden1 = make_hidden(1), make_hidden(2)

    baseline, _ = run_staged(layers, hidden0, hidden1, make_positions(0),
                             make_positions(50))
    moved_slot1, _ = run_staged(layers, hidden0, hidden1, make_positions(0),
                                make_positions(900))

    assert_exact(moved_slot1.microbatch0.hidden_states,
                 baseline.microbatch0.hidden_states)
    assert not torch.equal(moved_slot1.microbatch1.hidden_states,
                           baseline.microbatch1.hidden_states)


def test_each_slot_reads_its_own_attention_metadata():
    """Per-layer attention metadata is selected by slot, in slot order.

    ``TwoMicrobatchAttentionMetadata`` resolves a layer name against whichever
    slot the enclosing ``select_microbatch_attention_metadata`` names. Every
    layer's attention therefore runs twice, once per slot, and must see its own
    side both times.
    """
    layers, _ = make_layers(3, "full_attention")
    run_staged(layers, make_hidden(1), make_hidden(2), make_positions(0),
               make_positions(50))

    for layer in layers:
        assert layer.attention.seen == ["mb0", "mb1"]


def test_split_phase_moe_matches_synchronous_composition():
    """Starting the collectives early does not change the expert result.

    The staged path replaces the MoE block with launch/wait pairs around the
    same primitives. If a transform moved to the wrong side of the
    reduce-scatter wait, or a handle were resolved out of order, this diverges
    from the block the non-staged path calls.
    """
    block = SparseMoeBlock(seed=7)
    communicator = RecordingCommunicator()
    hidden = make_hidden(3)

    prepared = Qwen3_5AsyncMoeAdapter.prepare(block.experts, hidden, 0,
                                              communicator)
    experts = Qwen3_5AsyncMoeAdapter.experts(prepared, communicator)
    reduction = Qwen3_5AsyncMoeAdapter.reduce_start(experts, 0, communicator)
    staged = Qwen3_5AsyncMoeAdapter.finalize(reduction, communicator)

    assert_exact(staged, block(hidden))


# ---------------------------------------------------------------------------
# The invariant that keeps eight cores alive
# ---------------------------------------------------------------------------


def test_collective_schedule_is_identical_for_real_and_dummy_inputs():
    """Every rank issues the same collectives regardless of what it is running.

    The expert-parallel collectives are compiled into the device program, so
    all eight ranks must launch identically shaped programs or the cores block
    forever with no error. A busy rank runs two real microbatches, a
    half-loaded one runs a real and a dummy, an idle one runs two dummies --
    and the sequence of (operation, slot, shape) has to be the same for all
    three. Any data-dependent branch in the Python schedule breaks this before
    it ever reaches hardware.
    """
    layers, _ = make_layers(3, "full_attention")
    positions0, positions1 = make_positions(0), make_positions(50)
    real0, real1 = make_hidden(1), make_hidden(2)
    dummy = torch.zeros_like(real0)
    dummy_positions = torch.zeros_like(positions0)

    schedules = []
    for hidden0, hidden1, pos0, pos1 in [
        (real0, real1, positions0, positions1),          # busy rank, real=2
        (real0, dummy, positions0, dummy_positions),     # real=1, one dummy
        (dummy, dummy, dummy_positions, dummy_positions),  # idle rank
    ]:
        _, communicator = run_staged(layers, hidden0, hidden1, pos0, pos1)
        schedules.append(communicator.events)

    assert schedules[0] == schedules[1] == schedules[2]
    # And the schedule is the intended one, not merely a consistent one: each
    # slot contributes six operations per layer -- gather the hidden states,
    # gather the router logits, wait for both, reduce-scatter, wait for it --
    # so no slot's work is ever collapsed away for being a dummy.
    events = schedules[0]
    assert len(events) == 12 * len(layers)
    for slot in (0, 1):
        assert sum(1 for _, s, _ in events if s == slot) == 6 * len(layers)


def test_microbatch_1_reduce_is_carried_across_the_next_layer():
    """The overlap is present in the emitted order, not just in intent.

    Layer L+1's microbatch-0 all-gathers must be launched before layer L's
    microbatch-1 reduce-scatter is waited on. That gap is the window the
    compiler has to overlap one slot's expert compute with the other slot's
    collectives; without it the staged path is strictly slower than serial,
    which is what commit 1 measured.
    """
    layers, _ = make_layers(2, "full_attention")
    _, communicator = run_staged(layers, make_hidden(1), make_hidden(2),
                                 make_positions(0), make_positions(50))
    events = communicator.events

    # Second layer's slot-0 gathers, versus the first layer's slot-1 RS wait.
    layer1_slot0_gather = [
        index for index, (op, slot, _) in enumerate(events)
        if op == "AG" and slot == 0
    ][1]
    layer0_slot1_wait = [
        index for index, (op, slot, _) in enumerate(events)
        if op == "wait-RS" and slot == 1
    ][0]
    assert layer1_slot0_gather < layer0_slot1_wait


def test_model_adapter_runs_both_slots_through_embedding_and_final_norm():
    """End to end through the adapter the runner actually calls.

    ``_forward_qwen3_5_two_microbatches`` dispatches
    ``Qwen3_5TwoMicrobatchModelAdapter.forward``, which adds token embedding
    and the final norm around the pipeline. Checking it here means the wiring
    between those and the two slots is covered by real numbers rather than by
    a monkeypatched pipeline.
    """
    num_layers = 2
    layers, metadata = make_layers(num_layers, "full_attention")
    vocab = _weights(32, HIDDEN, seed=11)

    backbone_class = type("Qwen3_5Model", (object,), {})
    backbone = backbone_class()
    backbone.layers = layers
    backbone.start_layer = 0
    backbone.end_layer = num_layers
    backbone.aux_hidden_state_layers = ()
    backbone.embed_input_ids = lambda tokens: vocab[tokens]
    backbone.norm = FusedNorm(seed=13)
    model = SimpleNamespace(model=backbone)

    input_ids0 = torch.arange(NUM_TOKENS, dtype=torch.long)
    input_ids1 = torch.arange(NUM_TOKENS, 2 * NUM_TOKENS, dtype=torch.long)
    positions0, positions1 = make_positions(0), make_positions(50)

    result = Qwen3_5TwoMicrobatchModelAdapter.forward(
        model, (input_ids0, input_ids1), (positions0, positions1),
        RecordingCommunicator())

    for slot, (input_ids, positions, actual) in enumerate([
        (input_ids0, positions0, result.hidden_states0),
        (input_ids1, positions1, result.hidden_states1),
    ]):
        state = serial_microbatch(layers, vocab[input_ids], positions, slot,
                                  RecordingCommunicator())
        expected, _ = backbone.norm(state.hidden_states, state.residual)
        assert_exact(actual, expected)
    assert metadata is not None
