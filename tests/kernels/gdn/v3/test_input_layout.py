# SPDX-License-Identifier: Apache-2.0
"""Compact GDN inputs preserve the existing cast and reshape contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper


@pytest.mark.parametrize(
    "tokens,width,dtype",
    [
        (16, 10240, jnp.bfloat16),
        (127, 10240, jnp.bfloat16),
        pytest.param(128, 10240, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (257, 10240, jnp.bfloat16),
        pytest.param(8192, 10240, jnp.bfloat16, marks=pytest.mark.tpu_test),
        pytest.param(128, 12288, jnp.bfloat16, marks=pytest.mark.tpu_test),
        pytest.param(128, 16384, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (128, 24576, jnp.bfloat16),
        pytest.param(128, 256, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (128, 133, jnp.bfloat16),
        (128, 10240, jnp.float32),
    ],
)
def test_compact_qkv_preserves_value_bits(tokens, width, dtype):
    # A wrong token/feature mapping or narrowing before conversion changes
    # these bits, including on the fallback paths and across tile boundaries.
    rng = np.random.default_rng(54)
    values = rng.standard_normal((tokens, width), dtype=np.float32)
    values *= np.exp2(rng.integers(-10, 11, size=(tokens, 1)))
    values[0, :4] = [0.0, -0.0, 1.0, -1.0]
    qkv = jnp.asarray(values, dtype=dtype)
    expected = qkv.astype(jnp.float32).reshape(tokens, 1, width)

    actual = jax.jit(wrapper._compact_qkv_input)(qkv)

    assert actual.shape == expected.shape
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
    )
