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

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)

from vllm_torchtpu.layers.vllm.quantization import modelopt_mixed
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod


@pytest.fixture
def linear_layer():
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


@pytest.mark.parametrize(
    ("prefix", "expected_method"),
    [
        ("model.layers.0.self_attn.q_proj", "VllmFp8LinearMethodTPU"),
        ("model.layers.0.mlp.experts", "VllmNvfp4LinearMethod"),
    ],
)
def test_modelopt_mixed_dispatch(linear_layer, prefix, expected_method):
    from vllm_torchtpu.layers.vllm.quantization.modelopt_mixed import \
        VllmModelOptMixedPrecisionConfig
    cfg = VllmModelOptMixedPrecisionConfig.from_config({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.self_attn.q_proj": {
                "quant_algo": "FP8",
            },
            "model.layers.0.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        },
    })
    method = cfg.get_quant_method(linear_layer, prefix=prefix)
    assert type(method).__name__ == expected_method


def test_modelopt_mixed_unlisted_linear_is_unquantized(linear_layer):
    from vllm_torchtpu.layers.vllm.quantization.modelopt_mixed import \
        VllmModelOptMixedPrecisionConfig
    cfg = VllmModelOptMixedPrecisionConfig.from_config({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.self_attn.q_proj": {
                "quant_algo": "FP8",
            },
        },
    })
    method = cfg.get_quant_method(linear_layer,
                                  prefix="model.layers.0.self_attn.o_proj")
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.fixture(params=["linear", "moe"])
def dispatch_layer(request, linear_layer, monkeypatch):
    if request.param == "linear":
        return linear_layer
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    # Keep this a dispatch test without initializing distributed MoE state.
    method = object.__new__(VllmUnquantizedFusedMoEMethod)
    monkeypatch.setattr(modelopt_mixed.VllmModelOptMixedPrecisionConfig,
                        "get_moe_config", lambda self, layer: None)
    monkeypatch.setattr(modelopt_mixed, "VllmUnquantizedFusedMoEMethod",
                        lambda config: method)
    return layer


@pytest.mark.parametrize("quant_algo", ["UNSUPPORTED", ""])
def test_modelopt_mixed_rejects_unknown_quant_algo(dispatch_layer, quant_algo):
    prefix = "model.layers.0.proj"
    cfg = modelopt_mixed.VllmModelOptMixedPrecisionConfig.from_config({
        "quant_method":
        "modelopt",
        "quant_algo":
        "MIXED_PRECISION",
        "quantized_layers": {
            prefix: {
                "quant_algo": quant_algo
            }
        },
    })
    with pytest.raises(ValueError, match="Unsupported.*quant_algo") as exc:
        cfg.get_quant_method(dispatch_layer, prefix)
    assert prefix in str(exc.value)
    assert repr(quant_algo) in str(exc.value)


@pytest.mark.parametrize("excluded", [False, True])
def test_modelopt_mixed_preserves_unquantized_dispatch(dispatch_layer,
                                                       excluded):
    prefix = "model.layers.0.proj"
    cfg = modelopt_mixed.VllmModelOptMixedPrecisionConfig.from_config({
        "quant_method":
        "modelopt",
        "quant_algo":
        "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.1.proj": {
                "quant_algo": "FP8"
            },
        },
    })
    if excluded:
        cfg.quantized_layers[prefix] = {"quant_algo": "UNSUPPORTED"}
        cfg.exclude_modules = [prefix]
    method = cfg.get_quant_method(dispatch_layer, prefix)
    expected = (UnquantizedLinearMethod if isinstance(
        dispatch_layer, LinearBase) else VllmUnquantizedFusedMoEMethod)
    assert isinstance(method, expected)


def test_modelopt_mixed_still_rejects_mxfp8(dispatch_layer):
    prefix = "model.layers.0.proj"
    cfg = modelopt_mixed.VllmModelOptMixedPrecisionConfig.from_config({
        "quant_method":
        "modelopt",
        "quant_algo":
        "MIXED_PRECISION",
        "quantized_layers": {
            prefix: {
                "quant_algo": "MXFP8"
            }
        },
    })
    with pytest.raises(NotImplementedError, match="MXFP8"):
        cfg.get_quant_method(dispatch_layer, prefix)
