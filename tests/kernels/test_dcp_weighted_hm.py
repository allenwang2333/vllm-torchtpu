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
"""`weighted_heads_major` (TPU_DCP_MERGE_WEIGHTED_HM) against the torch
merge's weight multiply, transposed: bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import dcp_weighted_hm

pytestmark = pytest.mark.tpu_test


def _case(t, h, d=512, seed=0):
    rng = np.random.default_rng(seed)
    o = rng.normal(size=(t, h, d)).astype(np.float32) * 4
    o[1, :, :7] = np.nan  # NaN partial rows: weighted 0 where w == 0
    o[2, 3, 5] = np.inf
    lse = rng.normal(size=(t, h)).astype(np.float32) * 3
    lse[1] = -np.inf
    lse[5, ::3] = -np.inf
    merged = np.logaddexp(lse, rng.normal(size=(t, h)).astype(np.float32) * 3)
    with np.errstate(invalid="ignore"):
        w = np.exp(lse - merged)
    w[7, 0] = np.nan  # exp(-inf - -inf)
    return jnp.asarray(o, jnp.bfloat16), jnp.asarray(w, jnp.float32)


def _reference(o, w):
    x = o.astype(jnp.float32) * w[..., None]
    x = jnp.where(w[..., None] > 0, x, jnp.zeros_like(x))
    return jnp.transpose(x, (1, 0, 2))


@pytest.mark.parametrize("t,h", [(512, 64), (256, 64), (128, 64), (512, 32), (8, 64)])
def test_weighted_heads_major_bit_identical(t, h):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Pallas TPU kernel")
    o, w = _case(t, h, seed=t + h)
    got = np.asarray(dcp_weighted_hm.weighted_heads_major(o, w)).view(np.uint32)
    want = np.asarray(jax.jit(_reference)(o, w)).view(np.uint32)
    np.testing.assert_array_equal(got, want)
    # Positive control: a head-swapped write is caught.
    bad = np.asarray(
        dcp_weighted_hm.weighted_heads_major(o, w, fault="swap_heads")
    ).view(np.uint32)
    assert not np.array_equal(bad, want)
