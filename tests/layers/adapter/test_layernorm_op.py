# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for WeightPreprocessedGemmaRMSNorm custom op."""

import pytest
import torch
import torch.nn as nn
from vllm import ir
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.layers.vllm.custom_ops import \
    force_process_weights_after_loading
from vllm_torchtpu.layers.vllm.custom_ops.layernorm_op import \
    WeightPreprocessedGemmaRMSNorm


@pytest.fixture
def vllm_cpu_config():
    """Provides a minimal VllmConfig configured for CPU."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def test_registration():
    """Test that WeightPreprocessedGemmaRMSNorm subclasses GemmaRMSNorm."""
    assert issubclass(WeightPreprocessedGemmaRMSNorm, GemmaRMSNorm)


@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_process_weights_after_loading(vllm_cpu_config, hidden_size, dtype):
    """Test process_weights_after_loading computes weight.float() + 1.0."""
    norm = WeightPreprocessedGemmaRMSNorm(hidden_size)
    norm.weight.data = torch.randn(hidden_size, dtype=dtype)

    assert not hasattr(norm, "processed_weight")
    norm.process_weights_after_loading(dtype)

    assert hasattr(norm, "processed_weight")
    assert norm.processed_weight.dtype == torch.float32
    expected = norm.weight.data.float() + 1.0
    torch.testing.assert_close(norm.processed_weight, expected)


@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("eps", [1e-6])
def test_forward_without_residual(vllm_cpu_config, hidden_size, eps):
    """Test forward output without residual matches reference implementation."""
    norm = WeightPreprocessedGemmaRMSNorm(hidden_size, eps=eps)
    norm.weight.data = torch.randn(hidden_size)
    norm.process_weights_after_loading(torch.float32)

    x = torch.randn(4, 16, hidden_size)
    out = norm(x)

    # Reference implementation: ir.ops.rms_norm(x, weight.float() + 1.0, eps)
    expected_weight = norm.weight.data.float() + 1.0
    expected_out = ir.ops.rms_norm(x, expected_weight, eps)

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected_out)


@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("eps", [1e-6])
def test_forward_with_residual(vllm_cpu_config, hidden_size, eps):
    """Test forward output with residual matches reference implementation."""
    norm = WeightPreprocessedGemmaRMSNorm(hidden_size, eps=eps)
    norm.weight.data = torch.randn(hidden_size)
    norm.process_weights_after_loading(torch.float32)

    x = torch.randn(4, 16, hidden_size)
    residual = torch.randn(4, 16, hidden_size)
    out, out_residual = norm(x, residual)

    # Reference implementation: ir.ops.fused_add_rms_norm(x, residual, weight.float() + 1.0, eps)
    expected_weight = norm.weight.data.float() + 1.0
    expected_out, expected_residual = ir.ops.fused_add_rms_norm(
        x, residual, expected_weight, eps)

    assert out.shape == x.shape
    assert out_residual.shape == residual.shape
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(out_residual, expected_residual)


def test_forward_before_process_weights_raises_attribute_error(
        vllm_cpu_config):
    """Test calling forward before process_weights_after_loading raises AttributeError."""
    norm = WeightPreprocessedGemmaRMSNorm(64)
    x = torch.randn(2, 64)
    with pytest.raises(AttributeError):
        _ = norm(x)


def test_force_process_weights_after_loading(vllm_cpu_config):
    """Test force_process_weights_after_loading processes all instances in a model."""

    class DummyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.norm1 = WeightPreprocessedGemmaRMSNorm(64)
            self.linear = nn.Linear(64, 64)
            self.sub = nn.Sequential(
                WeightPreprocessedGemmaRMSNorm(64),
                nn.Linear(64, 64),
            )

    model = DummyModel()
    model.norm1.weight.data.fill_(0.25)
    model.sub[0].weight.data.fill_(0.75)

    # Verify not processed initially
    assert not hasattr(model.norm1, "processed_weight")
    assert not hasattr(model.sub[0], "processed_weight")

    # Run force processing
    force_process_weights_after_loading(model)

    # Verify both were processed
    assert hasattr(model.norm1, "processed_weight")
    assert hasattr(model.sub[0], "processed_weight")
    torch.testing.assert_close(model.norm1.processed_weight,
                               torch.full((64, ), 1.25))
    torch.testing.assert_close(model.sub[0].processed_weight,
                               torch.full((64, ), 1.75))
