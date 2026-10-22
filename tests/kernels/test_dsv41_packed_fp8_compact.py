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
def test_sparse_entry_point_returns_candidate_width(monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_sparse
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        PackedFP8Layout,
    )

    rows, width, capacity = 6, 98304, 2048
    candidates = jnp.tile(jnp.arange(capacity)[None, ::-1] * 3, (rows, 1))
    lengths = jnp.arange(rows, dtype=jnp.int32) * (width // rows)

    def scorer(q, weights, pool, pages, lengths, offsets, **kwargs):
        logical = jnp.arange(pages.shape[1] * 128, dtype=jnp.float32)[None, :]
        return jnp.where(logical < lengths[:, None], logical, -jnp.inf)

    monkeypatch.setattr(packed_fp8_sparse, "packed_fp8_mqa_logits", scorer)
    scores, canonical = packed_fp8_sparse.packed_fp8_sparse_mqa_logits(
        jnp.zeros((rows, 32, 128), jnp.float8_e4m3fn),
        jnp.ones((rows, 32), jnp.float32),
        jnp.empty((1, 128), jnp.int32),
        jnp.arange(width // 128, dtype=jnp.int32)[None, :],
        lengths,
        jnp.zeros((rows,), jnp.int32),
        candidates,
        jnp.array([0, rows], jnp.int32),
        q_scale=jnp.ones((rows, 32), jnp.float32),
        cache_layout=PackedFP8Layout(width // 128),
        block_size=8,
    )
    assert scores.shape == (rows, 16384)
    expected_ids = np.tile(np.arange(capacity, dtype=np.int32) * 3, (rows, 1))
    np.testing.assert_array_equal(canonical, expected_ids)
    logical = (expected_ids[:, :, None] * 8 + np.arange(8)).reshape(rows, -1)
    expected = np.where(logical < np.asarray(lengths)[:, None], logical, -np.inf)
    np.testing.assert_array_equal(scores, expected)


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
