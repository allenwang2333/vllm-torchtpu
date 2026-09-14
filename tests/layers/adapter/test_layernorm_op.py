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

import types

import pytest
import torch
import torch.nn as nn
from vllm import ir
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.model_loader.utils import \
    process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_torchtpu.layers.adapter.custom_ops.layernorm_op import \
    WeightPreprocessedGemmaRMSNorm
from vllm_torchtpu.layers.adapter.quantization.unquantized import \
    VllmUnquantizedGemmaNormWeightFold

HIDDEN_SIZE = 256
EPS = 1e-6

# The sweep only reads `dtype` (for attention layers) and `quantization` (for
# the torchao reload hook); a real ModelConfig would pull in a model download.
_MODEL_CONFIG = types.SimpleNamespace(dtype=torch.bfloat16, quantization=None)


@pytest.fixture
def vllm_cpu_config():
    """Provides a minimal VllmConfig configured for CPU."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _sweep(norm: WeightPreprocessedGemmaRMSNorm) -> None:
    """Run vLLM's real post-load sweep over a model holding `norm`.

    `BaseModelLoader.load_model` calls this for every loader, right after
    `load_weights`. Going through the upstream function rather than calling
    the fold directly is what makes these tests check that a norm layer is
    actually reachable from the sweep -- the whole premise of attaching a
    `quant_method` to it.
    """
    model = nn.Module()
    model.norm = norm
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))


def _load(norm: WeightPreprocessedGemmaRMSNorm,
          checkpoint_weight: torch.Tensor) -> None:
    """Load a checkpoint weight and then post-process it, as a loader does."""
    param = norm.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, checkpoint_weight)
    _sweep(norm)


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


def test_upstream_sweep_folds_the_implicit_one(vllm_cpu_config):
    """vLLM's own post-load sweep reaches the norm and stores `1 + w`."""
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)
    checkpoint_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)

    _load(norm, checkpoint_weight)

    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(norm.weight.data,
                               checkpoint_weight.float() + 1.0)


def test_fold_is_idempotent(vllm_cpu_config):
    """A second sweep must not turn `1 + w` into `2 + w`.

    `process_weights_after_loading` is re-entered by vLLM's reload paths, and
    a double fold would skew every normalization instead of failing.
    """
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)
    checkpoint_weight = torch.randn(HIDDEN_SIZE)

    _load(norm, checkpoint_weight)
    _sweep(norm)
    _sweep(norm)

    torch.testing.assert_close(norm.weight.data,
                               checkpoint_weight.float() + 1.0)


def test_fold_survives_a_loader_that_bypasses_weight_loader(vllm_cpu_config):
    """Weights assigned in place, as under `load_format=dummy`, still fold.

    `initialize_dummy_weights` writes through `model.state_dict()` and never
    consults `param.weight_loader`. Driving the fold from the post-load sweep
    instead of from the parameter hook is what keeps that path correct: the
    weight must end up near 1, not near 0, or every Gemma norm scales its
    output to nothing.
    """
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)

    # What `initialize_dummy_weights` does, minus its TPU-only `torch._sync`.
    for tensor in norm.state_dict().values():
        tensor.uniform_(-1e-3, 1e-3)
    dummy_weight = norm.weight.data.clone()

    _sweep(norm)

    torch.testing.assert_close(norm.weight.data, dummy_weight + 1.0)


def test_quant_method_refuses_the_linear_entry_points(vllm_cpu_config):
    """`create_weights`/`apply` exist only to satisfy QuantizeMethodBase."""
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE)
    assert isinstance(norm.quant_method, VllmUnquantizedGemmaNormWeightFold)

    with pytest.raises(NotImplementedError):
        norm.quant_method.create_weights(norm)
    with pytest.raises(NotImplementedError):
        norm.quant_method.apply(norm, torch.randn(2, HIDDEN_SIZE))


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


def test_forward_needs_no_extra_call_at_the_load_site(vllm_cpu_config):
    """The norm is driven entirely by the sweep every loader already runs.

    There is no second tensor a caller could forget to populate -- the failure
    mode that made the target and draft load sites each need their own
    post-load pass.
    """
    norm = WeightPreprocessedGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)

    out = norm(torch.randn(2, HIDDEN_SIZE))

    assert out.shape == (2, HIDDEN_SIZE)
