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
"""Tests for W8A16 on small FP8 dense-linear steps (TPU_FP8_LINEAR_W8A16_MAX_ROWS)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax

_K, _N = 256, 128


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    scale = np.abs(w).max(axis=0) / 448.0
    w_q = jnp.asarray(w / scale, dtype=jnp.float8_e4m3fn)
    return w_q, jnp.asarray(scale, dtype=jnp.float32)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows,max_rows,expect_quantized",
    [
        (16, 0, True),  # 0: W8A8 everywhere
        (16, 16, False),
        (17, 16, True),
        (64, 64, False),
        (256, 64, True),
    ],
)
def test_activation_quantization_follows_threshold(
    monkeypatch, rows, max_rows, expect_quantized
):
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", str(max_rows))
    seen = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True):
        seen["quantize_activation"] = quantize_activation
        return real(x, w_q, w_s, quantize_activation=quantize_activation)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    out = _quantized_matmul_jax(jnp.ones((rows, _K), jnp.bfloat16), w_q, w_s)
    assert out.shape == (rows, _N)
    assert out.dtype == jnp.bfloat16
    assert seen["quantize_activation"] is expect_quantized


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows,expect_quantized", [(64, False), (65, True)])
def test_default_threshold_is_64_rows(monkeypatch, rows, expect_quantized):
    monkeypatch.delenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", raising=False)
    seen = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True):
        seen["quantize_activation"] = quantize_activation
        return real(x, w_q, w_s, quantize_activation=quantize_activation)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    _quantized_matmul_jax(jnp.ones((rows, _K), jnp.bfloat16), w_q, w_s)
    assert seen["quantize_activation"] is expect_quantized


@pytest.mark.cpu_test
def test_w8a16_is_at_least_as_accurate_as_w8a8(monkeypatch):
    w_q, w_s = _weights()
    x = jnp.asarray(
        np.random.default_rng(1).standard_normal((16, _K)), dtype=jnp.bfloat16
    )
    # Reference: bf16 activations against the dequantized FP8 weights, in f32.
    ref = np.asarray(x, np.float32) @ (
        np.asarray(w_q.astype(jnp.float32)) * np.asarray(w_s)[None, :]
    )

    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "0")
    w8a8 = np.asarray(_quantized_matmul_jax(x, w_q, w_s), np.float32)
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "16")
    w8a16 = np.asarray(_quantized_matmul_jax(x, w_q, w_s), np.float32)

    err_w8a8 = np.abs(w8a8 - ref).max()
    err_w8a16 = np.abs(w8a16 - ref).max()
    assert err_w8a16 <= err_w8a8
