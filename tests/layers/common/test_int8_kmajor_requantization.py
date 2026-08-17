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
"""
Tests for the int8 K-major MoE weight quantizer.

`quantize_to_native_int8_kmajor` is the int8 sibling of
`quantize_to_native_fp4_kmajor`: same blocking along the contracting axis and
the same scale layout, so gmm_v2 reads both through one code path. These tests
pin the layout contract, the numerics, and the relationship to the fp4 path it
stands beside -- int8 must not be a quality regression against the format the
expert weights would otherwise be requantized to.

Runs on CPU:
    pytest tests/layers/common/test_int8_kmajor_requantization.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.common.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor, quantize_to_native_int8_kmajor)


def _kmajor_weight(experts=2, size_k=512, size_n=128, seed=0) -> jax.Array:
    """A [E, K, N] K-major weight with per-expert scale spread."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.02, size=(experts, size_k, size_n)).astype(np.float32)
    # Give each expert a different dynamic range so a single global scale
    # cannot pass the round-trip test by accident.
    w *= np.array([1.0, 37.0], dtype=np.float32)[:experts, None, None]
    return jnp.asarray(w, dtype=jnp.bfloat16)


def _dequantize(q: jax.Array, scale: jax.Array, block: int) -> jax.Array:
    """Undo the block scaling: q is [..., K, N], scale is [..., K/block, 1, N]."""
    size_k, size_n = q.shape[-2], q.shape[-1]
    blocked = q.astype(jnp.float32).reshape(*q.shape[:-2], size_k // block,
                                            block, size_n)
    return (blocked * scale).reshape(*q.shape[:-2], size_k, size_n)


@pytest.mark.parametrize("block", [128, 256, 512])
def test_layout_matches_the_fp4_quantizer(block):
    """Values stay [E, K, N] and the scale keeps the [E, K/block, 1, N] shape
    gmm_v2 expects. The fp4 quantizer is the reference for the scale layout;
    only the value tensor differs, because fp4 packs two to a byte and int8
    does not."""
    w = _kmajor_weight(size_k=512)
    q_int8, scale_int8 = quantize_to_native_int8_kmajor(w, block=block)
    _, scale_fp4 = quantize_to_native_fp4_kmajor(w, block=block)

    assert q_int8.dtype == jnp.int8
    assert q_int8.shape == w.shape, "int8 must not pack along K"
    assert scale_int8.dtype == jnp.float32
    assert scale_int8.shape == scale_fp4.shape
    assert scale_int8.shape == (w.shape[0], w.shape[1] // block, 1, w.shape[2])


def test_round_trip_error_is_bounded_by_the_quantization_step():
    """Dequantizing must land within half a step of the original, per block.

    This is the real correctness claim: the scale is derived from the block's
    own abs-max, so the bound is per-block and holds for both experts despite
    their 37x difference in dynamic range.
    """
    block = 512
    w = _kmajor_weight(size_k=512)
    q, scale = quantize_to_native_int8_kmajor(w, block=block)
    recovered = _dequantize(q, scale, block)

    err = jnp.abs(recovered - w.astype(jnp.float32))
    # scale is the step size; broadcast it back over the block axis.
    step = jnp.broadcast_to(scale, (w.shape[0], w.shape[1] // block, block,
                                    w.shape[2])).reshape(w.shape)
    # bf16 inputs are themselves rounded, so allow a small slack over 0.5 ULP.
    assert jnp.all(err <= step * 0.51 + 1e-6)


def test_int8_is_not_a_regression_against_fp4():
    """int8 carries strictly more mantissa than e2m1 at the same block size, so
    on identical weights its round-trip error must be lower. This is the
    substantive reason the expert path can move to int8 without a quality
    argument of its own."""
    block = 512
    w = _kmajor_weight(size_k=512)

    q_int8, scale_int8 = quantize_to_native_int8_kmajor(w, block=block)
    err_int8 = jnp.mean(
        jnp.square(_dequantize(q_int8, scale_int8, block) -
                   w.astype(jnp.float32)))

    packed_fp4, scale_fp4 = quantize_to_native_fp4_kmajor(w, block=block)
    # Unpack the two-per-byte e2m1 values back to [E, K, N] to compare.
    fp4 = jax.lax.bitcast_convert_type(packed_fp4, jnp.float4_e2m1fn)
    fp4 = jnp.swapaxes(fp4, -1, -2).reshape(w.shape)
    err_fp4 = jnp.mean(
        jnp.square(_dequantize(fp4, scale_fp4, block) - w.astype(jnp.float32)))

    assert err_int8 < err_fp4


def test_rounds_to_nearest_rather_than_truncating():
    """`astype(int8)` truncates toward zero; the quantizer rounds explicitly.

    A block whose abs-max is 127 makes the scale exactly 1.0, so a value of
    2.6 must come back as 3, not 2. Truncation would silently bias every
    expert weight toward zero.
    """
    block = 4
    w = jnp.asarray([[[127.0], [2.6], [-2.6], [0.4]]], dtype=jnp.float32)
    q, scale = quantize_to_native_int8_kmajor(w, block=block)

    np.testing.assert_allclose(np.asarray(scale).reshape(-1), [1.0], rtol=0)
    np.testing.assert_array_equal(
        np.asarray(q).reshape(-1), np.array([127, 3, -3, 0], dtype=np.int8))


def test_all_zero_block_stays_zero():
    """A zero block gives a zero abs-max; the reciprocal must not become inf
    and produce NaN values, matching the fp4 quantizer's `where` guard."""
    block = 128
    w = jnp.zeros((1, 256, 8), dtype=jnp.bfloat16)
    q, scale = quantize_to_native_int8_kmajor(w, block=block)

    assert jnp.all(q == 0)
    assert jnp.all(scale == 0)
    assert not jnp.any(jnp.isnan(_dequantize(q, scale, block)))
