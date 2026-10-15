# SPDX-License-Identifier: Apache-2.0
"""First-hit snapshots and sampling are device-side diagnostic contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("phase", [0, 2, 3, 99])
def test_first_nan_copies_hidden_and_actual_tokens_only_in_sample_window(dtype, phase):
    state = jnp.array([-1, phase, 0, 0], jnp.int32)
    hidden_snapshot = jnp.zeros((8, 128), dtype)
    token_snapshot = jnp.zeros(8, jnp.int32)
    tokens = jnp.array([15, 27, 39], jnp.int32)
    good = jnp.ones((3, 128), dtype)
    bad = good.at[2, 127].set(jnp.nan)
    armed = jnp.array([True])
    before = (state, hidden_snapshot, token_snapshot)
    result = record_hidden_nan(
        *before, armed, good, tokens, jnp.array([3]), jnp.array(0)
    )
    for actual, expected in zip(result, before):
        np.testing.assert_array_equal(actual, expected)
    result = record_hidden_nan(
        *result, armed, bad, tokens, jnp.array([3]), jnp.array(1)
    )
    np.testing.assert_array_equal(result[0], [1, (phase + 1) % 100, 3, 3])
    expected_hidden = bad if phase < 3 else hidden_snapshot[:3]
    np.testing.assert_array_equal(
        np.asarray(result[1][:3], dtype=np.float32),
        np.asarray(expected_hidden, dtype=np.float32),
    )
    np.testing.assert_array_equal(result[2][:3], tokens if phase < 3 else [0, 0, 0])
    later = record_hidden_nan(
        *result,
        armed,
        jnp.full_like(bad, jnp.nan),
        tokens + 1,
        jnp.array([3]),
        jnp.array(2),
    )
    for actual, expected in zip(later, result):
        np.testing.assert_array_equal(
            np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
        )


def test_only_nan_rounds_advance_sampling_and_window_reopens_after_one_hundred():
    result = (
        jnp.array([-1, 0, 0, 0], jnp.int32),
        jnp.zeros((1, 128)),
        jnp.zeros(1, jnp.int32),
    )
    bad = jnp.full((1, 128), jnp.nan)
    selected = []
    record = jax.jit(record_hidden_nan)
    for event in range(1, 206):
        state, hidden, tokens = result
        result = (state.at[0].set(-1), hidden, tokens)
        result = record(
            *result,
            jnp.array([True]),
            jnp.ones_like(bad),
            jnp.array([-9]),
            jnp.array([1]),
            jnp.array(0),
        )
        assert int(result[0][1]) == (event - 1) % 100
        result = record(
            *result,
            jnp.array([True]),
            bad,
            jnp.array([event]),
            jnp.array([1]),
            jnp.array(0),
        )
        if int(result[2][0]) == event:
            selected.append(event)
    assert selected == [1, 2, 3, 101, 102, 103, 201, 202, 203]


def test_unarmed_warmup_preserves_snapshot_and_sampling_phase():
    before = (
        jnp.array([-1, 0, 0, 0], jnp.int32),
        jnp.zeros((1, 128)),
        jnp.zeros(1, jnp.int32),
    )
    after = record_hidden_nan(
        *before,
        jnp.array([False]),
        jnp.full((1, 128), jnp.nan),
        jnp.array([42]),
        jnp.array([1]),
        jnp.array(0),
    )
    for actual, expected in zip(after, before):
        np.testing.assert_array_equal(actual, expected)
