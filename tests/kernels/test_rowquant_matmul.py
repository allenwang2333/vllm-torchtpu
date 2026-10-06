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
"""rowquant_matmul against the XLA per-channel fp8 path it replaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.rowquant_kernel import rowquant_matmul
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

FP8 = jnp.float8_e4m3fn


def _inputs(m, k, n, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(k1, (m, k), jnp.float32)
    x = x * jnp.exp(jax.random.normal(k3, (m, 1)) * 2.0)
    x = x.astype(jnp.bfloat16)
    if m > 2:
        x = x.at[1].set(0)  # an all-zero (padding) row takes scale 1
    wf = jax.random.normal(k2, (k, n), jnp.float32) * 0.02
    ws = jnp.max(jnp.abs(wf), axis=0) / 448.0
    w = jnp.clip(wf / ws, -448, 448).astype(FP8)
    return x, w, ws.astype(jnp.float32)


def _ulps(a, b):
    au = np.asarray(a).view(np.uint16).astype(np.int32)
    bu = np.asarray(b).view(np.uint16).astype(np.int32)
    return np.abs(au - bu)


# (m, k, n): GDN in_proj_qkvz / attn qkv, o_proj, shared expert in / out, at
# decode and prefill row counts of Qwen3.5-397B at TP8.
SHAPES = [
    (4, 4096, 2560),
    (8, 4096, 2560),
    (48, 4096, 2560),
    (8, 1024, 4096),
    (16, 4096, 256),
    (48, 128, 4096),
    (1024, 4096, 2560),
]


@pytest.mark.parametrize("w_kn", [True, False])
@pytest.mark.parametrize("m,k,n", SHAPES)
def test_matches_xla_path(m, k, n, w_kn):
    x, w, ws = _inputs(m, k, n, seed=m + n)
    ref = jax.jit(xla_quantized_matmul)(x, w, ws)
    out = rowquant_matmul(x, w if w_kn else jnp.asarray(w.T), ws, w_kn=w_kn)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert np.isfinite(np.asarray(out, np.float32)).all()
    np.testing.assert_array_equal(np.asarray(out[1]), np.zeros((n,), out.dtype))
    # Same quantized operands, scales and accumulation: bit-exact on v7x.
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), np.asarray(ref).view(np.uint16)
    )


def test_positive_control_catches_a_scale_error():
    """The comparison above must fail on a kernel fed one wrong channel scale."""
    m, k, n = 8, 4096, 2560
    x, w, ws = _inputs(m, k, n)
    ref = jax.jit(xla_quantized_matmul)(x, w, ws)
    bad = rowquant_matmul(x, w, ws.at[n // 3].multiply(1.03), w_kn=True)
    assert _ulps(bad, ref).max() > 0
