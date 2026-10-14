# SPDX-License-Identifier: Apache-2.0
"""Round-local logs and readback ownership are diagnostic contracts."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.hidden_nan_flags import (
    HiddenNaNDetector,
    ModelLayerNaNFlags,
    install_hidden_nan_detector,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def detector():
    return HiddenNaNDetector(
        (
            ModelLayerNaNFlags("target", (7, 11, 19), torch.zeros(3, dtype=torch.bool)),
            ModelLayerNaNFlags("draft", (60,), torch.zeros(1, dtype=torch.bool)),
        )
    )


def test_unarmed_warmup_submits_no_readback():
    capture = detector()
    with (
        patch.object(torch.Tensor, "to") as copy,
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent") as event,
    ):
        assert capture.readback() is None
    copy.assert_not_called()
    event.assert_not_called()


def test_round_snapshots_log_only_bad_round_with_original_layer_ids(caplog):
    capture = detector()
    capture.arm()
    capture.begin_round()
    capture.models[0].flags[1:] = True
    # TPU D2H has a distinct CPU destination; CPU Tensor.to would alias here.
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent") as event,
    ):
        bad = capture.readback()
        assert bad is not None
        event.return_value.record.assert_called_once_with()
        capture.begin_round()
        assert not any(model.flags.any() for model in capture.models)
        good = capture.readback()
    assert good is not None
    with caplog.at_level(logging.ERROR, logger="vllm_torchtpu.runner.hidden_nan_flags"):
        good.consume()
        bad.consume()
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "role=target round=1" in messages[0]
    assert "layer_ids=[7, 11, 19]" in messages[0]
    assert "isnan=[false, true, true]" in messages[0]


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
    monkeypatch, multimodal, tuple_output, container, layer_ids, slots
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

        def forward(self, hidden):
            for layer in self.layers.children():
                output = layer(hidden)
                hidden = output[0] if tuple_output else output
            return hidden

    class MultiModalModel(torch.nn.Module):
        supports_multimodal = True

        def __init__(self):
            super().__init__()
            self.vision_layers = torch.nn.ModuleList([Layer()])
            self.language_model = Model()

        def get_language_model(self):
            return self.language_model

        def forward(self, hidden):
            return self.language_model(hidden)

    model = MultiModalModel() if multimodal else Model()
    capture = install_hidden_nan_detector([("target", model)], torch.device("cpu"))
    assert capture.models[0].layer_ids == layer_ids
    assert capture.models[0].flags.shape == (2,)
    capture.arm()
    capture.begin_round()
    seen_slots = []

    def record_on_cpu(flags, armed, hidden, slot):
        seen_slots.append(slot)
        flags[slot] |= armed[0] & torch.isnan(hidden).any()

    monkeypatch.setattr(
        "vllm_torchtpu.runner.hidden_nan_flags.record_layer_nan", record_on_cpu
    )
    hidden = torch.ones((2, 128))
    hidden[1, 0] = float("nan")
    result = model(hidden)
    torch.testing.assert_close(result, hidden + 2, equal_nan=True)
    assert capture.models[0].flags.tolist() == [True, True]
    assert seen_slots == slots
