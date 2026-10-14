# SPDX-License-Identifier: Apache-2.0
"""Layer flags cover the whole round without retaining hidden tensors."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("layers", [3, 60])
def test_layer_flags_accumulate_chunks_and_include_later_bad_layers(dtype, layers):
    flags = jnp.zeros(layers, jnp.bool_)
    armed = jnp.array([True])
    good = jnp.ones((3, 128), dtype)
    bad = jnp.ones((5, 128), dtype).at[4, 127].set(jnp.nan)
    for slot, hidden in ((0, good), (1, bad), (1, good), (layers - 1, bad)):
        flags = record_hidden_nan(flags, armed, hidden, jnp.array(slot, jnp.int32))
    expected = np.zeros(layers, np.bool_)
    expected[[1, layers - 1]] = True
    np.testing.assert_array_equal(flags, expected)
    assert flags.shape == (layers,) and flags.dtype == jnp.bool_


def test_unarmed_warmup_does_not_inspect_nan():
    flags = jnp.zeros(3, jnp.bool_)
    result = record_hidden_nan(
        flags, jnp.array([False]), jnp.full((1, 128), jnp.nan), jnp.array(1)
    )
    np.testing.assert_array_equal(result, flags)
