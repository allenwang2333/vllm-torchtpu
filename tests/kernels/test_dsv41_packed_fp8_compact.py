# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact candidate columns preserve dense masked TopK semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_compact import (
    candidate_membership,
    candidate_topk,
    compact_candidate_scores,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("block_size", [8, 96, 192])
def test_compact_scores_keep_logical_ties_and_remove_duplicates(block_size):
    rng = np.random.default_rng(581)
    rows, width, capacity = 7, 4096, 16
    count = (width + block_size - 1) // block_size
    candidates = rng.integers(-2, count + 3, (rows, capacity), dtype=np.int32)
    candidates[0] = -1
    candidates[1, :5] = [1, 1, 0, 0, np.iinfo(np.int32).max]
    scores = rng.integers(-3, 4, (rows, width)).astype(np.float32)
    scores[2, 19:] = -np.inf
    mask, canonical = candidate_membership(jnp.asarray(candidates), count)
    compact = compact_candidate_scores(
        jnp.asarray(scores), canonical, block_size=block_size
    )
    assert compact.shape == (rows, capacity * block_size)
    values, selected = jax.lax.top_k(compact, 16)
    blocks = jnp.take_along_axis(canonical, selected // block_size, axis=1)
    actual = jnp.where(
        jnp.isfinite(values), blocks * block_size + selected % block_size, -1
    )
    expected = np.full((rows, 16), -1, np.int32)
    for row in range(rows):
        logical = np.arange(width)
        valid = np.asarray(mask[row])[logical // block_size] != 0
        valid &= np.isfinite(scores[row])
        chosen = sorted(np.flatnonzero(valid), key=lambda i: (-scores[row, i], i))[:16]
        expected[row, : len(chosen)] = chosen
    np.testing.assert_array_equal(actual, expected)
    result = candidate_topk(compact, canonical, k=16, block_size=block_size)
    for row in range(rows):
        valid = expected[row][expected[row] >= 0]
        expected[row, : valid.size] = np.sort(valid)
    np.testing.assert_array_equal(result, expected)
