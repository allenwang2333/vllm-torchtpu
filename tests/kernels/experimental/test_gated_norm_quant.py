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
"""The fused gated norm + fp8 quantization against the two-step path, in the
Pallas interpreter on the host."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import gated_norm_quant as gnq
from vllm_torchtpu.kernels.experimental.gated_norm_quant import kernel
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _on_the_host():
    # The reference must round as the interpreter does, even in a session that
    # has brought up the TPU backend.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _inputs(tokens=64, heads=4, dim=128, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    o = (jax.random.normal(k[0], (tokens, heads, dim)) * 3).astype(jnp.bfloat16)
    z = jax.random.normal(k[1], (tokens, heads, dim)).astype(jnp.bfloat16)
    w = (1.0 + 0.1 * jax.random.normal(k[2], (dim,))).astype(jnp.bfloat16)
    return o, z, w


def _assert_matches_reference(o, z, w, tile):
    q, s = gnq.gated_norm_quant(o, z, w, 1e-6, tile=tile, interpret=True)
    q_ref, s_ref = gnq.reference(o, z, w, 1e-6)
    np.testing.assert_array_equal(np.asarray(s), np.asarray(s_ref))
    np.testing.assert_array_equal(
        np.asarray(q, np.float32), np.asarray(q_ref, np.float32)
    )


def test_kernel_matches_the_two_step_path():
    _assert_matches_reference(*_inputs(), tile=16)


# 36, 44 and 52 have no multiple-of-8 divisor and run as one block.
@pytest.mark.parametrize("tokens", [4, 12, 20, 36, 40, 44, 48, 52, 80])
def test_row_counts_the_tile_does_not_divide(tokens):
    _assert_matches_reference(*_inputs(tokens=tokens, heads=2), tile=32)


@pytest.mark.parametrize(
    "rows, cap, tile", [(64, 32, 32), (48, 32, 24), (12, 32, 12), (36, 32, 36)]
)
def test_legal_tile(rows, cap, tile):
    assert kernel.legal_tile(rows, cap) == tile


def test_zero_rows_get_unit_scale():
    o, z, w = _inputs(tokens=16, heads=2)
    o = o.at[3].set(0)
    q, s = gnq.gated_norm_quant(o, z, w, 1e-6, tile=16, interpret=True)
    assert float(s[3, 0]) == 1.0
    assert float(jnp.abs(q[3].astype(jnp.float32)).max()) == 0.0


def test_prequant_matmul_equals_the_xla_path():
    o, z, w = _inputs(tokens=32, heads=2)
    q, s = gnq.reference(o, z, w, 1e-6)
    x = o.astype(jnp.float32)
    zz = z.astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
    x = (x * w.astype(jnp.float32) * zz * jax.nn.sigmoid(zz)).astype(jnp.bfloat16)
    x = x.reshape(x.shape[0], -1)
    w_q = (jax.random.normal(jax.random.PRNGKey(7), (x.shape[1], 96)) * 0.05).astype(
        kernel.FP8
    )
    w_s = jnp.full((96,), 0.02, jnp.float32)
    out = gnq.prequant_matmul_in_out(q.reshape(x.shape), s, w_q, w_s)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32),
        np.asarray(xla_quantized_matmul(x, w_q, w_s), np.float32),
    )
