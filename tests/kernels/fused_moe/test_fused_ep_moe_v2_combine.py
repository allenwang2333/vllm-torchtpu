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
"""Single-device route and scalar-memory regression tests for MoE combine."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import kernel

FP8 = jnp.float8_e4m3fn


@pytest.fixture(autouse=True)
def require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("MoE combine uses TPU scalar-memory DMA")


@pytest.mark.parametrize(
    "tokens,topk,hidden",
    [
        (256, 4, 512),
        (256, 7, 512),
        (512, 1, 256),
        (8192, 10, 2560),
        (16384, 10, 2560),
    ],
)
def test_combine_preserves_token_routes_above_scalar_memory_capacity(
    tokens, topk, hidden
):
    """Large routing tables must compile and keep each token's coefficients."""
    rng = np.random.default_rng(46)
    arrivals = rng.integers(-7, 8, size=(512, hidden)).astype(np.float32)
    positions = rng.integers(0, 512, size=(tokens, topk), dtype=np.int32)
    coefficients = np.exp2(rng.integers(-5, 4, size=(tokens, topk))).astype(np.float32)
    coefficients *= rng.choice([-1, 1], size=(tokens, topk))
    expected = np.zeros((tokens, hidden), dtype=np.float32)
    for k in range(topk):
        expected += arrivals[positions[:, k]] * coefficients[:, k, None]
    combine = kernel.build_combine_kernel(
        t_local=tokens,
        topk=topk,
        hidden=hidden,
        lane_blocks=hidden // 128,
        wire_dtype=FP8,
        out_dtype=jnp.bfloat16,
    )
    actual = jax.jit(combine)(
        jnp.asarray(arrivals, dtype=FP8).reshape(512, hidden // 128, 128),
        jnp.asarray(coefficients),
        jnp.asarray(positions),
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(expected, dtype=jnp.bfloat16)
    )
