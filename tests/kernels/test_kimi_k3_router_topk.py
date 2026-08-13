# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-interpreter correctness tests for Kimi-K3's fused router kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.router_topk import kimi_k3_router_topk


@pytest.mark.parametrize("blockwise", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 41])
def test_kimi_k3_router_topk_matches_exact_reference(seed: int,
                                                     blockwise: bool) -> None:
    key = jax.random.key(seed)
    logits = jax.random.normal(key, (1, 896), dtype=jnp.float32)
    bias = 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (896, ),
                                   dtype=jnp.float32)

    weights, ids = kimi_k3_router_topk(logits,
                                       bias,
                                       interpret=True,
                                       blockwise=blockwise)
    scores = jax.nn.sigmoid(logits)
    _, expected_ids = jax.lax.top_k(scores + bias, 16)
    expected_weights = jnp.take_along_axis(scores, expected_ids, axis=1)
    expected_weights /= expected_weights.sum(axis=1, keepdims=True)

    np.testing.assert_array_equal(np.asarray(ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(np.asarray(weights),
                                  np.asarray(expected_weights))
