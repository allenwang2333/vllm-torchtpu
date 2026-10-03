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
"""kernels/top_p.py: the proven-prefilter threshold IS the bisection's.

Both sides run on the same JAX backend with the same predicate, so equality is
exact on any platform. The torch-vs-JAX predicate match (what makes the op
equal to topp_mask's torch bisection) is gated on TPU in
tests/layers/adapter/test_top_k_top_p.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import top_p

V = 4096
P = np.array([0.95, 0.9, 0.5, 0.99, 0.1, 1.0, 0.95, 0.7], np.float32)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _cases():
    rng = np.random.default_rng(0)
    for s in (0.5, 2.0, 4.0, 8.0):
        yield f"randn{s}", rng.standard_normal((8, V)) * s
    x = np.zeros((8, V))
    x[:, :10] = 5.0
    yield "ties10", x
    yield "flat", np.zeros((8, V))
    x = np.full((8, V), -30.0)
    x[:, 3] = 0.0
    yield "onehot", x
    yield "rounded", np.round(rng.standard_normal((8, V)) * 3)
    # 2000 tied leaders: the tie group is wider than K.
    x = np.full((8, V), -5.0)
    x[:, :2000] = 3.0
    yield "wide_ties", x
    # Zipf-like LM tail.
    r = np.arange(1, V + 1)
    yield "zipf", np.stack([-1.1 * np.log(rng.permutation(r)) for _ in range(8)])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("accum", ["native", "f32"])
@pytest.mark.parametrize("method,k", [("topk", 1024), ("topk", 16), ("approx", 512)])
@pytest.mark.parametrize("window", [0, 2])
def test_fast_threshold_is_bisection(dtype, accum, method, k, window):
    active = P < 1.0
    proven_any = False
    for name, x in _cases():
        probs = jnp.asarray(_softmax(x).astype(np.float32)).astype(dtype)
        p = jnp.asarray(P)
        want = np.asarray(top_p.bisect_threshold(probs, p, accum))
        got, proven, fell_back = top_p.fast_threshold_jit(
            probs, p, k, method, accum, True, window
        )
        got = np.asarray(got)
        np.testing.assert_array_equal(got[active], want[active], err_msg=name)
        proven = np.asarray(proven).astype(bool)
        # A proven row is exact without the fallback too.
        proven_any |= bool(proven[active].any())
        if not bool(fell_back):
            assert proven[active].all(), name
    assert proven_any


def test_fast_threshold_proves_peaked_rows():
    # Sharp rows: nucleus far below K, the proof must succeed (no fallback).
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, V)) * 8.0
    probs = jnp.asarray(_softmax(x).astype(np.float32))
    p = jnp.asarray(P)
    _, proven, fell_back = top_p.fast_threshold_jit(probs, p, 1024, "topk", "f32", True)
    assert not bool(fell_back)
    assert np.asarray(proven)[P < 1.0].all()
