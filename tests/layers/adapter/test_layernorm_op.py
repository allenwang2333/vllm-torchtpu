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
from vllm import ir
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_torchtpu.layers.vllm.custom_ops.layernorm_op import \
    WeightPreprocessedGemmaRMSNorm

HIDDEN_SIZE = 256
EPS = 1e-6


@pytest.fixture
def vllm_cpu_config():
    """Provides a minimal VllmConfig configured for CPU."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _load(norm: WeightPreprocessedGemmaRMSNorm,
          checkpoint_weight: torch.Tensor) -> None:
    """Load a weight the way vLLM's model loaders do.

    Every Gemma loader resolves the parameter's own hook first and only falls
    back to `default_weight_loader`; see `AutoWeightsLoader._load_param` in
    vllm/model_executor/models/utils.py and gemma4's `load_weights`. Going
    through `getattr` here rather than calling the fold directly is what makes
    this a test of the integration and not just of the arithmetic.
    """
    param = norm.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, checkpoint_weight)


def test_registration():
    """Test that WeightPreprocessedGemmaRMSNorm subclasses GemmaRMSNorm."""
    assert issubclass(WeightPreprocessedGemmaRMSNorm, GemmaRMSNorm)


def test_weight_is_fp32_regardless_of_default_dtype(vllm_cpu_config):
    """The folded weight stays fp32 even when built under a bf16 default.

    Models are constructed inside `set_default_torch_dtype(model_config.dtype)`,
    so the upstream parameter would be bf16 -- too coarse to hold `1 + w` at
    the precision the per-step `.float()` upcast used to provide.
    """
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)
    finally:
        torch.set_default_dtype(default)

    assert norm.weight.dtype == torch.float32


def test_weight_loader_folds_the_implicit_one(vllm_cpu_config):
    """Loading through the parameter hook stores `1 + w`, upcast to fp32."""
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)
    checkpoint_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)

    _load(norm, checkpoint_weight)

    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(norm.weight.data,
                               checkpoint_weight.float() + 1.0)


def test_weight_loader_rejects_shape_mismatch(vllm_cpu_config):
    """The fold keeps `default_weight_loader`'s shape assertion."""
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)

    with pytest.raises(AssertionError):
        _load(norm, torch.randn(HIDDEN_SIZE + 1))


def test_forward_without_residual_matches_upstream(vllm_cpu_config):
    """Forward matches stock GemmaRMSNorm fed the same checkpoint weight."""
    checkpoint_weight = torch.randn(HIDDEN_SIZE)

    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)
    _load(norm, checkpoint_weight)

    x = torch.randn(4, 16, HIDDEN_SIZE)
    out = norm(x)

    expected_out = ir.ops.rms_norm(x, checkpoint_weight.float() + 1.0, EPS)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected_out)


def test_forward_with_residual_matches_upstream(vllm_cpu_config):
    """Residual path matches stock GemmaRMSNorm on the same weight."""
    checkpoint_weight = torch.randn(HIDDEN_SIZE)

    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)
    _load(norm, checkpoint_weight)

    x = torch.randn(4, 16, HIDDEN_SIZE)
    residual = torch.randn(4, 16, HIDDEN_SIZE)
    out, out_residual = norm(x, residual)

    expected_out, expected_residual = ir.ops.fused_add_rms_norm(
        x, residual,
        checkpoint_weight.float() + 1.0, EPS)
    assert out.shape == x.shape
    assert out_residual.shape == residual.shape
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(out_residual, expected_residual)


def test_forward_needs_no_post_load_pass(vllm_cpu_config):
    """A freshly constructed norm forwards without any load-time hook.

    The fold lives in the parameter, so there is no second tensor a caller
    could forget to populate -- the failure mode that made the target and
    draft load sites each need their own post-load sweep.
    """
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)

    out = norm(torch.randn(2, HIDDEN_SIZE))

    assert out.shape == (2, HIDDEN_SIZE)
