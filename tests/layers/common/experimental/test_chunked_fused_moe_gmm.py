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
"""Tests for experimental chunked MoE sequence token execution."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common.experimental.chunked_fused_moe_gmm import \
    chunked_fused_moe_func
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


def test_vllm_tpu_moe_chunk_size_env_default(monkeypatch):
    """Verifies that VLLM_TPU_MOE_CHUNK_SIZE defaults to 0 when unset."""
    monkeypatch.delenv("VLLM_TPU_MOE_CHUNK_SIZE", raising=False)
    getter = envs.environment_variables["VLLM_TPU_MOE_CHUNK_SIZE"]
    assert getter() == 0


def test_vllm_tpu_moe_chunk_size_env_override(monkeypatch):
    """Verifies that VLLM_TPU_MOE_CHUNK_SIZE parses integer environment values."""
    monkeypatch.setenv("VLLM_TPU_MOE_CHUNK_SIZE", "256")
    getter = envs.environment_variables["VLLM_TPU_MOE_CHUNK_SIZE"]
    assert getter() == 256


@pytest.mark.parametrize("chunk_size", [0, -1, 10, 20])
def test_chunked_fused_moe_func_passthrough(monkeypatch, chunk_size):
    """Verifies that chunk_size <= 0 or chunk_size >= num_tokens passes through directly."""
    num_tokens = 10
    hidden_size = 64
    hidden_states = jnp.ones((num_tokens, hidden_size), dtype=jnp.float32)
    w1 = jnp.ones((2, hidden_size, 128), dtype=jnp.float32)
    w2 = jnp.ones((2, 64, hidden_size), dtype=jnp.float32)
    topk_weights = jnp.ones((num_tokens, 1), dtype=jnp.float32)
    topk_ids = jnp.zeros((num_tokens, 1), dtype=jnp.int32)

    mock_fused = MagicMock(
        return_value=jnp.zeros((num_tokens, hidden_size), dtype=jnp.float32))
    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.experimental.chunked_fused_moe_gmm.fused_moe_func",
        mock_fused,
    )

    out = chunked_fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        chunk_size=chunk_size,
    )

    assert mock_fused.call_count == 1
    call_args = mock_fused.call_args[0]
    assert call_args[0].shape == (num_tokens, hidden_size)
    assert out.shape == (num_tokens, hidden_size)


@pytest.mark.parametrize(
    ("num_tokens", "chunk_size", "expected_chunk_sizes"),
    [
        (10, 4, [4, 3, 3]),
        (10, 3, [3, 3, 2, 2]),
        (12, 4, [4, 4, 4]),
        (100, 32, [25, 25, 25, 25]),
    ],
)
def test_chunked_fused_moe_func_balanced_slicing(monkeypatch, num_tokens,
                                                 chunk_size,
                                                 expected_chunk_sizes):
    """Verifies balanced token chunking slices and concatenation order."""
    hidden_size = 32
    hidden_states = jnp.arange(num_tokens * hidden_size,
                               dtype=jnp.float32).reshape(
                                   (num_tokens, hidden_size))
    w1 = jnp.ones((2, hidden_size, 64), dtype=jnp.float32)
    w2 = jnp.ones((2, 32, hidden_size), dtype=jnp.float32)
    topk_weights = jnp.ones((num_tokens, 1), dtype=jnp.float32)
    topk_ids = jnp.zeros((num_tokens, 1), dtype=jnp.int32)

    recorded_slice_sizes = []

    def mock_fused(h, *args, **kwargs):
        recorded_slice_sizes.append(h.shape[0])
        # Return unique indicator based on slice content to verify concatenation order
        return h * 2.0

    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.experimental.chunked_fused_moe_gmm.fused_moe_func",
        mock_fused,
    )

    out = chunked_fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        chunk_size=chunk_size,
    )

    assert recorded_slice_sizes == expected_chunk_sizes
    assert out.shape == (num_tokens, hidden_size)
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(hidden_states * 2.0))


def _synthesize_moe_inputs(num_tokens=8,
                           hidden_size=2048,
                           intermediate_size=768,
                           num_experts=4,
                           topk=2,
                           seed=42):
    """Generate deterministic inputs compatible with fused_moe_func."""
    key = jax.random.key(seed)
    hk, w1k, w2k, b1k, b2k = jax.random.split(key, 5)
    hidden_states = (
        jax.random.normal(hk, (num_tokens, hidden_size), dtype=jnp.float32) /
        10).astype(jnp.bfloat16)
    w1 = (jax.random.normal(w1k,
                            (num_experts, hidden_size, intermediate_size * 2),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (jax.random.normal(w2k, (num_experts, intermediate_size, hidden_size),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w1_bias = (jax.random.normal(b1k, (num_experts, 1, intermediate_size * 2),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(b2k, (num_experts, 1, hidden_size),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)

    # Simple uniform routing
    topk_weights = jnp.tile(jnp.array([[0.6, 0.4]], dtype=jnp.float32),
                            (num_tokens, 1))
    topk_ids = jnp.tile(jnp.array([[0, 1]], dtype=jnp.int32), (num_tokens, 1))

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "w1_scale": None,
        "w2_scale": None,
        "w1_bias": w1_bias,
        "w2_bias": w2_bias,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "topk": topk,
        "activation": "silu",
    }


@pytest.mark.parametrize("chunk_size", [2, 3, 4, 8])
def test_chunked_fused_moe_func_matches_fused_moe(chunk_size):
    """Verifies that chunked_fused_moe_func output numerically matches unchunked fused_moe_func."""
    inputs = _synthesize_moe_inputs(num_tokens=8)

    expected = fused_moe_func(**inputs)
    actual = chunked_fused_moe_func(**inputs, chunk_size=chunk_size)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-3,
                               rtol=1e-3)
