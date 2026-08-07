from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.qwen3_5_microbatch import Qwen3_5MoeLayerAdapter


class FakeNorm(torch.nn.Module):
    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden * 2
        residual = residual + hidden
        return residual * 2, residual


class FakeAttention(torch.nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add = add

    def forward(self, hidden_states, positions=None):
        value = hidden_states + self.add
        if positions is not None:
            value = value + positions.unsqueeze(-1)
        return value


class FakeSparseMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = SimpleNamespace()

    def forward(self, hidden):
        return hidden * 3


def make_layer(layer_type, layer_scale):
    layer_class = type("Qwen3_5DecoderLayer", (SimpleNamespace,), {})
    layer = layer_class()
    layer.layer_type = layer_type
    layer.input_layernorm = FakeNorm()
    layer.post_attention_layernorm = FakeNorm()
    layer.mlp = FakeSparseMoe()
    layer.layer_scale = layer_scale
    layer.attn_layer_scale = torch.nn.Parameter(torch.full((1, 1, 2), .25))
    layer.ffn_layer_scale = torch.nn.Parameter(torch.full((1, 1, 2), .5))
    if layer_type == "full_attention":
        layer.self_attn = FakeAttention(1)
    else:
        layer.linear_attn = FakeAttention(2)
    return layer


def serial_forward(layer, hidden, residual, positions):
    if residual is None:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
    else:
        hidden, residual = layer.input_layernorm(hidden, residual)
    if layer.layer_type == "full_attention":
        output = layer.self_attn(hidden_states=hidden, positions=positions)
    else:
        output = layer.linear_attn(hidden_states=hidden)
    if layer.layer_scale:
        output = output * (layer.attn_layer_scale[0] + 1)
    hidden, residual = layer.post_attention_layernorm(output, residual)
    hidden = layer.mlp(hidden)
    if layer.layer_scale:
        hidden = hidden * (layer.ffn_layer_scale[0] + 1)
    return hidden, residual


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("layer_scale", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
def test_staged_layer_matches_serial(layer_type, layer_scale, has_residual):
    layer = make_layer(layer_type, layer_scale)
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    residual = torch.ones_like(hidden) if has_residual else None
    positions = torch.arange(3, dtype=torch.float32)
    expected = serial_forward(layer, hidden, residual, positions)
    actual = Qwen3_5MoeLayerAdapter.forward_layer(
        layer, hidden, residual, positions)
    torch.testing.assert_close(actual.hidden_states, expected[0])
    torch.testing.assert_close(actual.residual, expected[1])


def test_adapter_rejects_dense_or_wrong_model_layer():
    with pytest.raises(TypeError, match="Qwen3_5DecoderLayer"):
        Qwen3_5MoeLayerAdapter.validate_layer(SimpleNamespace())
    layer = make_layer("full_attention", False)
    layer.mlp = torch.nn.Identity()
    with pytest.raises(TypeError, match="sparse-MoE"):
        Qwen3_5MoeLayerAdapter.validate_layer(layer)


def test_async_moe_stages_preserve_existing_runner_contract():
    from vllm_torchtpu.models.qwen3_5_microbatch import (
        Qwen3_5AsyncMoeAdapter)

    events = []

    class Handle:
        def __init__(self, value, kind, microbatch):
            self.value = value
            self.kind = kind
            self.microbatch = microbatch

    class Communicator:
        def all_gather_start(self, value, microbatch):
            events.append(("AG", microbatch))
            return Handle(value + 10, "AG", microbatch)

        def reduce_scatter_start(self, value, microbatch):
            events.append(("RS", microbatch))
            return Handle(value - 5, "RS", microbatch)

        def wait(self, handle):
            events.append(("wait-" + handle.kind, handle.microbatch))
            return handle.value

    class Gate:
        def __call__(self, hidden):
            return hidden + 1, None

    class RoutedExperts:
        def _ensure_moe_quant_config_init(self):
            events.append(("quant-init", 0))

    class Runner:
        do_naive_dispatch_combine = True
        moe_config = SimpleNamespace(is_sequence_parallel=False, pcp_size=1)
        routed_experts = RoutedExperts()
        gate = Gate()
        _fse_fuse_gate = False

        def apply_routed_input_transform(self, hidden):
            return hidden * 2, hidden * 3

        def _maybe_pad_hidden_states(self, shared, hidden):
            return hidden, None, None

        def _maybe_sync_shared_experts_stream(self, shared):
            pass

        def _apply_quant_method(self, hidden_states, router_logits,
                                shared_experts_input, input_ids):
            events.append(("GMM", 0))
            return shared_experts_input, hidden_states + router_logits

        def _maybe_reduce_shared_expert_output(self, shared):
            return shared

        def _maybe_apply_routed_scale_to_output(self, shared, fused):
            return shared, fused * 2

        def apply_routed_output_transform(self, fused):
            return fused + 1

        def _maybe_reduce_final_output(self, result, truncation):
            return result

        def _maybe_add_zero_expert_output(self, result):
            return result

    communicator = Communicator()
    hidden = torch.tensor([[2.0]])
    prepared = Qwen3_5AsyncMoeAdapter.prepare(
        Runner(), hidden, 0, communicator)
    assert events[:3] == [("quant-init", 0), ("AG", 0), ("AG", 0)]
    experts = Qwen3_5AsyncMoeAdapter.experts(prepared, communicator)
    reduction = Qwen3_5AsyncMoeAdapter.reduce_start(
        experts, 0, communicator)
    output = Qwen3_5AsyncMoeAdapter.finalize(reduction, communicator)
    # gathered hidden=14, gathered router=15, fused=(14+15-5)*2+1=49,
    # local shared=6.
    torch.testing.assert_close(output, torch.tensor([[55.0]]))
    assert events == [
        ("quant-init", 0), ("AG", 0), ("AG", 0),
        ("wait-AG", 0), ("wait-AG", 0), ("GMM", 0),
        ("RS", 0), ("wait-RS", 0),
    ]


@pytest.mark.parametrize("sequence_parallel,pcp", [(True, 1), (False, 2)])
def test_async_moe_rejects_unsupported_parallel_modes(sequence_parallel, pcp):
    from vllm_torchtpu.models.qwen3_5_microbatch import (
        Qwen3_5AsyncMoeAdapter)

    runner = SimpleNamespace(
        do_naive_dispatch_combine=True,
        moe_config=SimpleNamespace(is_sequence_parallel=sequence_parallel,
                                   pcp_size=pcp))
    with pytest.raises(ValueError):
        Qwen3_5AsyncMoeAdapter._validate_runner(runner)


def test_two_microbatch_pipeline_carries_mb1_reduce_across_layers(monkeypatch):
    import vllm_torchtpu.models.qwen3_5_microbatch as staged

    events = []

    class Layer:
        def __init__(self, index):
            self.index = index
            self.mlp = SimpleNamespace(experts=self)

    def validate(layer):
        events.append(("validate", layer.index))

    def attention(layer, hidden, residual, positions, microbatch_index):
        assert microbatch_index in (0, 1)
        mb = int(hidden.split("mb")[1][0])
        events.append(("attention", layer.index, mb))
        return SimpleNamespace(moe_input=(layer.index, mb), residual=residual)

    def prepare(runner, moe_input, mb, communicator, input_ids):
        events.append(("AG", runner.index, mb))
        return (runner.index, mb)

    def experts(prepared, communicator):
        events.append(("GMM", *prepared))
        return prepared

    def reduce(experts_value, mb, communicator):
        events.append(("RS", *experts_value))
        return experts_value

    def async_finalize(reduction, communicator):
        events.append(("wait-RS", *reduction))
        layer, mb = reduction
        return f"layer{layer}-mb{mb}"

    def layer_finalize(layer, prepared, expert_output):
        mb = prepared.moe_input[1]
        events.append(("finalize", layer.index, mb))
        return SimpleNamespace(hidden_states=f"next-mb{mb}-{layer.index}",
                               residual=f"residual-{mb}-{layer.index}")

    monkeypatch.setattr(staged.Qwen3_5MoeLayerAdapter, "validate_layer",
                        validate)
    monkeypatch.setattr(staged.Qwen3_5MoeLayerAdapter, "attention_stage",
                        attention)
    monkeypatch.setattr(staged.Qwen3_5MoeLayerAdapter, "moe_finalize_stage",
                        layer_finalize)
    monkeypatch.setattr(staged.Qwen3_5AsyncMoeAdapter, "prepare", prepare)
    monkeypatch.setattr(staged.Qwen3_5AsyncMoeAdapter, "experts", experts)
    monkeypatch.setattr(staged.Qwen3_5AsyncMoeAdapter, "reduce_start", reduce)
    monkeypatch.setattr(staged.Qwen3_5AsyncMoeAdapter, "finalize",
                        async_finalize)

    states = (staged.QwenMicrobatchState("input-mb0", None),
              staged.QwenMicrobatchState("input-mb1", None))
    staged.Qwen3_5TwoMicrobatchPipeline.forward_layers(
        [Layer(0), Layer(1)], states,
        (torch.tensor(0), torch.tensor(0)), SimpleNamespace())

    # Layer-1 MB0 attention+AG launch before waiting for layer-0 MB1 RS;
    # layer-1 MB1 attention begins only after that wait and finalization.
    att0 = events.index(("attention", 1, 0))
    ag0 = events.index(("AG", 1, 0))
    prior_wait1 = events.index(("wait-RS", 0, 1))
    att1 = events.index(("attention", 1, 1))
    assert att0 < ag0 < prior_wait1 < att1

    for layer_index in (0, 1):
        layer_events = [event for event in events
                        if len(event) == 3 and event[1] == layer_index]
        assert ("AG", layer_index, 0) in layer_events
        assert ("AG", layer_index, 1) in layer_events
        assert ("RS", layer_index, 0) in layer_events
        assert ("RS", layer_index, 1) in layer_events


def test_model_adapter_embeds_pipelines_and_normalizes_both(monkeypatch):
    import vllm_torchtpu.models.qwen3_5_microbatch as staged

    calls = []
    backbone_class = type("Qwen3_5Model", (object,), {})
    backbone = backbone_class()
    backbone.layers = [object(), object()]
    backbone.start_layer = 0
    backbone.end_layer = 2
    backbone.aux_hidden_state_layers = ()

    def embed(tokens):
        calls.append(("embed", int(tokens.item())))
        return tokens.float().reshape(1, 1) * 10

    class Norm:
        def __call__(self, hidden, residual):
            calls.append(("norm", int(hidden.item()), int(residual.item())))
            return hidden + residual, None

    backbone.embed_input_ids = embed
    backbone.norm = Norm()
    model = SimpleNamespace(model=backbone)

    def pipeline(layers, states, positions, communicator, input_ids):
        calls.append(("pipeline", len(layers)))
        return staged.QwenTwoMicrobatchOutput(
            staged.QwenMicrobatchState(states[0].hidden_states + 1,
                                       torch.tensor([[2.0]])),
            staged.QwenMicrobatchState(states[1].hidden_states + 1,
                                       torch.tensor([[3.0]])),
        )

    monkeypatch.setattr(staged.Qwen3_5TwoMicrobatchPipeline,
                        "forward_layers", pipeline)
    result = staged.Qwen3_5TwoMicrobatchModelAdapter.forward(
        model,
        (torch.tensor(4), torch.tensor(5)),
        (torch.tensor([0]), torch.tensor([0])),
        SimpleNamespace(),
    )
    torch.testing.assert_close(result.hidden_states0, torch.tensor([[43.0]]))
    torch.testing.assert_close(result.hidden_states1, torch.tensor([[54.0]]))
    assert calls == [("embed", 4), ("embed", 5), ("pipeline", 2),
                     ("norm", 41, 2), ("norm", 51, 3)]


def test_model_adapter_resolves_conditional_wrapper_and_rejects_pp():
    from vllm_torchtpu.models.qwen3_5_microbatch import (
        Qwen3_5TwoMicrobatchModelAdapter)

    backbone_class = type("Qwen3_5Model", (object,), {})
    backbone = backbone_class()
    backbone.layers = [object(), object()]
    backbone.start_layer = 0
    backbone.end_layer = 1
    backbone.aux_hidden_state_layers = ()
    causal = SimpleNamespace(model=backbone)
    wrapper = SimpleNamespace(get_language_model=lambda: causal)
    with pytest.raises(ValueError, match="pipeline-parallel"):
        Qwen3_5TwoMicrobatchModelAdapter.resolve_backbone(wrapper)


def _attention_only_backbone(num_layers=2):
    """Backbone whose only usable stage is layer 0's attention."""
    backbone_class = type("Qwen3_5Model", (object,), {})
    backbone = backbone_class()
    backbone.layers = [make_layer("linear_attention", False)
                       for _ in range(num_layers)]
    backbone.start_layer = 0
    backbone.end_layer = num_layers
    backbone.aux_hidden_state_layers = ()
    backbone.embed_input_ids = lambda tokens: tokens.float().reshape(1, 2)

    def forbidden_norm(*args, **kwargs):
        raise AssertionError("final norm must not run in the diagnostic "
                             "attention-only endpoint")

    backbone.norm = forbidden_norm
    return backbone




def test_attention_metadata_selects_and_restores_microbatch():
    from vllm_torchtpu.models.qwen3_5_microbatch import (
        TwoMicrobatchAttentionMetadata, select_microbatch_attention_metadata)

    metadata = TwoMicrobatchAttentionMetadata(
        {"full": "full-0", "gdn": "gdn-0"},
        {"full": "full-1", "gdn": "gdn-1"})
    assert isinstance(metadata, dict)
    assert metadata["full"] == "full-0"
    with select_microbatch_attention_metadata(1):
        assert metadata["full"] == "full-1"
        assert metadata["gdn"] == "gdn-1"
        with select_microbatch_attention_metadata(0):
            assert metadata["gdn"] == "gdn-0"
        assert metadata["full"] == "full-1"
    assert metadata["full"] == "full-0"


def test_attention_metadata_requires_identical_layer_sets():
    from vllm_torchtpu.models.qwen3_5_microbatch import (
        TwoMicrobatchAttentionMetadata)

    with pytest.raises(ValueError, match="layer sets"):
        TwoMicrobatchAttentionMetadata({"a": object()}, {"b": object()})
