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
"""gmm_v2_ffn must equal the two gmm_v2 calls it replaces, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.megablox.gmm_v2_ffn import gmm_v2_ffn, make_ffn_configs

NUM_GROUPS, SIZE_K, SIZE_I, SIZE_N = 8, 512, 256, 512


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _group_sizes(rng: np.random.Generator, size_m: int) -> np.ndarray:
    """Random sizes over a random subset of groups, summing to at most size_m."""
    valid = int(rng.integers(1, size_m + 1))
    active = np.sort(
        rng.choice(NUM_GROUPS, size=int(rng.integers(1, NUM_GROUPS + 1)), replace=False)
    )
    cuts = np.sort(rng.integers(0, valid + 1, size=len(active) - 1))
    sizes = np.zeros(NUM_GROUPS, np.int32)
    sizes[active] = np.diff(np.concatenate([[0], cuts, [valid]]))
    return sizes


def _bits(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint16)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("size_m", [40, 160, 320, 480])
@pytest.mark.parametrize("seed", [0, 1])
def test_gmm_v2_ffn_matches_two_gmm_v2_calls(has_bias, size_m, seed):
    _require_tpu()
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((size_m, SIZE_K)), jnp.bfloat16)
    w1 = jnp.asarray(
        rng.standard_normal((NUM_GROUPS, SIZE_K, 2 * SIZE_I)) * 0.5, jnp.float8_e4m3fn
    )
    w2 = jnp.asarray(
        rng.standard_normal((NUM_GROUPS, SIZE_I, SIZE_N)) * 0.5, jnp.float8_e4m3fn
    )
    w1_scale = jnp.asarray(
        rng.uniform(0.02, 0.2, (NUM_GROUPS, 1, 1, 2 * SIZE_I)), jnp.float32
    )
    w2_scale = jnp.asarray(
        rng.uniform(0.01, 0.1, (NUM_GROUPS, 1, 1, SIZE_N)), jnp.float32
    )
    w1_bias = w2_bias = None
    if has_bias:
        w1_bias = jnp.asarray(
            rng.standard_normal((NUM_GROUPS, 1, 2 * SIZE_I)), jnp.float32
        )
        w2_bias = jnp.asarray(rng.standard_normal((NUM_GROUPS, 1, SIZE_N)), jnp.float32)
    group_sizes = _group_sizes(rng, size_m)
    gs = jnp.asarray(group_sizes)
    offset = jnp.zeros((1,), jnp.int32)

    assert (
        make_ffn_configs(
            x,
            w1,
            w1_scale,
            w1_bias,
            w2,
            w2_scale,
            w2_bias,
            gs,
            fuse_act="silu",
            maybe_quantize_lhs=True,
            rhs_quant_dtype=None,
        )
        is not None
    )
    hidden = gmm_v2(
        x,
        w1,
        gs,
        rhs_scale=w1_scale,
        rhs_bias=w1_bias,
        group_offset=offset,
        zero_initialize=False,
        fuse_act="silu",
    )
    expected = gmm_v2(
        hidden,
        w2,
        gs,
        rhs_scale=w2_scale,
        rhs_bias=w2_bias,
        group_offset=offset,
        zero_initialize=False,
        preferred_element_type=jnp.bfloat16,
    )
    actual = gmm_v2_ffn(
        x,
        w1,
        w2,
        gs,
        w1_scale=w1_scale,
        w1_bias=w1_bias,
        w2_scale=w2_scale,
        w2_bias=w2_bias,
        group_offset=offset,
        fuse_act="silu",
        preferred_element_type=jnp.bfloat16,
    )

    assert actual.shape == expected.shape
    valid = int(group_sizes.sum())
    np.testing.assert_array_equal(_bits(actual[:valid]), _bits(expected[:valid]))


def test_make_ffn_configs_declines_unmatched_widths():
    _require_tpu()
    x = jax.ShapeDtypeStruct((40, SIZE_K), jnp.bfloat16)
    w1 = jax.ShapeDtypeStruct((NUM_GROUPS, SIZE_K, 2 * SIZE_I), jnp.float8_e4m3fn)
    w2 = jax.ShapeDtypeStruct((NUM_GROUPS, SIZE_I // 2, SIZE_N), jnp.float8_e4m3fn)
    gs = jax.ShapeDtypeStruct((NUM_GROUPS,), jnp.int32)
    assert (
        make_ffn_configs(
            x,
            w1,
            None,
            None,
            w2,
            None,
            None,
            gs,
            fuse_act="silu",
            maybe_quantize_lhs=True,
            rhs_quant_dtype=None,
        )
        is None
    )
