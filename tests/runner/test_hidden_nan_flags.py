# SPDX-License-Identifier: Apache-2.0
"""Snapshot contents, sampling and readback ownership are diagnostic contracts."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.hidden_nan_flags import (
    HiddenNaNDetector,
    ModelHiddenNaNSnapshot,
    install_hidden_nan_detector,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def detector(tmp_path, device="cpu", capacity=8):
    return HiddenNaNDetector(
        tuple(
            ModelHiddenNaNSnapshot(
                role,
                layers,
                torch.tensor([-1, 0, 0, 0], dtype=torch.int32, device=device),
                torch.zeros((capacity, 128), device=device),
                torch.zeros(capacity, dtype=torch.int32, device=device),
                torch.zeros(capacity, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
            )
            for role, layers in (("target", (7, 11, 19)), ("draft", (60,)))
        ),
        tmp_path,
    )


def record_on_cpu(state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot):
    if armed[0] and state[0] < 0 and torch.isnan(hidden).any():
        saved_hidden[: hidden.shape[0]].copy_(hidden)
        saved_tokens[: tokens.shape[0]].copy_(tokens)
        state[:] = torch.tensor([slot, 1, hidden.shape[0], int(rows[0])])


def test_unarmed_warmup_submits_no_readback(tmp_path):
    capture = detector(tmp_path)
    with (
        patch.object(torch.Tensor, "to") as copy,
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent") as event,
    ):
        assert capture.readback() is None
    copy.assert_not_called()
    event.assert_not_called()


def test_deferred_snapshot_survives_later_round_and_logs_path_not_flags(
    tmp_path, caplog
):
    capture = detector(tmp_path)
    capture.arm()
    capture.begin_round(["original-request"])
    model = capture.models[0]
    model.state[:] = torch.tensor([1, 1, 2, 2])
    model.hidden[1, 0] = float("nan")
    model.input_ids[:2] = torch.tensor([27, 39])
    # TPU D2H has a distinct CPU destination; CPU Tensor.to would alias here.
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent") as event,
    ):
        bad = capture.readback()
        assert bad is not None
        event.return_value.record.assert_called_once_with()
        capture.begin_round(["later-request"])
        assert all(model.state[0] == -1 for model in capture.models)
        model.hidden.fill_(42)
        model.input_ids.fill_(99)
        good = capture.readback()
    assert good is not None
    with caplog.at_level(logging.ERROR, logger="vllm_torchtpu.runner.hidden_nan_flags"):
        good.consume()
        bad.consume()
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "role=target round=1 layer=11" in messages[0]
    assert "isnan=" not in messages[0]
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    saved = torch.load(files[0], weights_only=True)
    assert saved["req_ids"] == ["original-request"]
    assert saved["layer_id"] == 11 and saved["round"] == 1
    assert saved["hidden_states"].shape == (2, 128)
    assert torch.isnan(saved["hidden_states"][1, 0])
    assert saved["input_ids"].tolist() == [27, 39]


@pytest.mark.parametrize("phase", [0, 1, 2, 3, 4, 99])
def test_only_first_three_nan_events_in_window_write_dump(tmp_path, phase):
    capture = detector(tmp_path)
    capture.arm()
    capture.begin_round(["r"])
    capture.models[0].state[:] = torch.tensor([0, phase, 1, 1])
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent"),
    ):
        result = capture.readback()
    assert result is not None
    result.consume()
    assert len(list(tmp_path.glob("*.pt"))) == int(1 <= phase <= 3)


@pytest.mark.parametrize("tp,pcp", [(2, 1), (1, 8)])
def test_async_multirank_flags_require_output_aggregation(monkeypatch, tp, pcp):
    runner = object.__new__(TPUModelRunner)
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
    )
    runner.is_pooling_model = False
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=None, parallel_config=runner.parallel_config
    )
    monkeypatch.setattr(envs, "TPU_HIDDEN_NAN_LOG", True, raising=False)
    with pytest.raises(ValueError, match="all-rank output consumption"):
        runner.capture_model()


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize(
    "container,layer_ids,slots",
    [("list", (0, 1), [0, 1]), ("dict", (7, 19), [1, 0])],
)
def test_layer_hooks_check_full_hidden_once_and_preserve_output(
    tmp_path, monkeypatch, multimodal, tuple_output, container, layer_ids, slots
):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

        def forward(self, hidden):
            for operation in self.inner:
                hidden = operation(hidden)
            result = hidden + 1
            return (result, None) if tuple_output else result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = (
                torch.nn.ModuleList([Layer(), Layer()])
                if container == "list"
                else torch.nn.ModuleDict({"19": Layer(), "7": Layer()})
            )

        def forward(self, input_ids):
            hidden = input_ids[:, None].expand(-1, 128).float().clone()
            hidden[1, 0] = float("nan")
            for layer in self.layers.children():
                output = layer(hidden)
                hidden = output[0] if tuple_output else output
            return hidden

    class MultiModalModel(torch.nn.Module):
        supports_multimodal = True

        def __init__(self):
            super().__init__()
            self.vision_layers = torch.nn.ModuleList([Layer()])
            self.vision_embedding = torch.nn.Embedding(256, 128)
            self.language_model = Model()

        def get_language_model(self):
            return self.language_model

        def forward(self, hidden):
            self.vision_embedding(torch.ones((2, 3), dtype=torch.int64))
            return self.language_model(hidden)

    model = MultiModalModel() if multimodal else Model()
    config = SimpleNamespace(get_hidden_size=lambda: 128, dtype=torch.float32)
    capture = install_hidden_nan_detector(
        [("target", model, config, 8)], torch.device("cpu"), dump_dir=tmp_path
    )
    assert capture.models[0].layer_ids == layer_ids
    assert capture.models[0].hidden.shape == (8, 128)
    capture.arm()
    capture.begin_round(["r"])
    seen_slots = []

    def inspect(state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot):
        seen_slots.append(slot)
        record_on_cpu(
            state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot
        )

    monkeypatch.setattr(
        "vllm_torchtpu.runner.hidden_nan_flags.record_layer_nan", inspect
    )
    tokens = torch.tensor([15, 27], dtype=torch.int32)
    result = model(tokens)
    hidden = tokens[:, None].expand(-1, 128).float().clone()
    hidden[1, 0] = float("nan")
    torch.testing.assert_close(result, hidden + 2, equal_nan=True)
    snapshot = capture.models[0]
    assert snapshot.layer_ids[int(snapshot.state[0])] == layer_ids[slots[0]]
    torch.testing.assert_close(snapshot.hidden[:2], hidden + 1, equal_nan=True)
    assert snapshot.input_ids[:2].tolist() == [15, 27]
    assert seen_slots == slots


@pytest.mark.parametrize("replicated_embedding", [False, True])
def test_unbound_mtp_forward_captures_new_embedding_tokens(
    tmp_path, monkeypatch, replicated_embedding
):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    class Embedding(VocabParallelEmbedding):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, input_ids):
            return input_ids[:, None].expand(-1, 128).float().clone()

    class LMHead(ParallelLMHead):
        def __init__(self):
            torch.nn.Module.__init__(self)

    class Model(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embed_tokens = embedding
            self.embed_positions = torch.nn.Embedding(128, 128)
            self.lm_head = LMHead()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            hidden = hidden + self.embed_positions(torch.arange(input_ids.numel()))
            hidden[1, 0] = float("nan")
            return self.layers[0](hidden)

    embedding = torch.nn.Embedding(256, 128) if replicated_embedding else Embedding()
    target, model = Model(embedding), Model(embedding)
    config = SimpleNamespace(get_hidden_size=lambda: 128, dtype=torch.float32)
    capture = install_hidden_nan_detector(
        [("target", target, config, 2), ("draft", model, config, 8)],
        torch.device("cpu"),
        dump_dir=tmp_path,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.hidden_nan_flags.record_layer_nan", record_on_cpu
    )
    capture.arm()
    capture.begin_round(["step0"])
    target(torch.tensor([15, 27], dtype=torch.int32))
    capture.begin_round(["step1"])
    type(model).forward(model, torch.tensor([39, 42, 81, 99], dtype=torch.int32))
    assert capture.models[0].hidden.shape[0] == 2
    assert capture.models[1].hidden.shape[0] == 8
    assert capture.models[1].input_ids[:4].tolist() == [39, 42, 81, 99]


def test_idle_forward_between_rounds_does_not_spend_nan_sample(tmp_path):
    capture = detector(tmp_path)
    capture.arm()
    assert capture.armed_tpu.tolist() == [False]
    capture.begin_round(["real"])
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent"),
    ):
        capture.readback()
    model = capture.models[0]
    record_on_cpu(
        model.state,
        model.hidden,
        model.input_ids,
        capture.armed_tpu,
        torch.full((1, 128), float("nan")),
        torch.tensor([42]),
        torch.tensor([1]),
        0,
    )
    assert model.state.tolist() == [-1, 0, 0, 0]


@pytest.mark.parametrize("capacity", [2, 8])
def test_native_readback_retains_snapshot_after_later_donated_updates(
    tmp_path, device, capacity
):
    """A Tensor clone alone does not protect a subsequently donated PJRT buffer."""
    if device.type != "tpu":
        pytest.skip("Requires native TorchTPU buffer donation and D2H")
    from vllm_torchtpu.layers.adapter.custom_ops.hidden_nan_flags_op import (
        record_layer_nan,
    )

    capture = detector(tmp_path, device, capacity)
    model = capture.models[0]

    def record(hidden, tokens):
        rows = torch.scalar_tensor(tokens.shape[0], dtype=torch.int32, device=device)
        record_layer_nan(
            model.state,
            model.hidden,
            model.input_ids,
            capture.armed_tpu,
            hidden,
            tokens,
            rows.reshape(1),
            1,
        )
        return hidden

    compiled = torch.compile(record, backend="tpu", fullgraph=True, dynamic=False)
    good = torch.ones((2, 128), device=device)
    bad = good.clone()
    bad[1, 0] = float("nan")
    tokens = torch.tensor([15, 27], dtype=torch.int32, device=device)
    compiled(good, tokens)
    capture.arm()
    pending = []
    for i, hidden in enumerate((bad, good, bad * 3)):
        capture.begin_round([f"r{i + 1}"])
        compiled(hidden, tokens + i)
        pending.append(capture.readback())
    for readback in reversed(pending):
        assert readback is not None
        readback.consume()
    snapshots = [torch.load(path, weights_only=True) for path in tmp_path.glob("*.pt")]
    assert len(snapshots) == 2
    for snapshot in snapshots:
        step = snapshot["round"] - 1
        assert snapshot["input_ids"].tolist() == [15 + step, 27 + step]
        assert snapshot["hidden_states"][0, 0] == 1 + step
        assert torch.isnan(snapshot["hidden_states"][1, 0])
