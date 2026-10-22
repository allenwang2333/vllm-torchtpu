# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-row prefill streaming top-k: stable ties and finite padding contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.sparse_mqa_logits import (
    sparse_topk_remap,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    streaming_topk,
)


@pytest.mark.parametrize("block,threshold", [(8192, 8192), (16384, 32768)])
@pytest.mark.parametrize("width", [65536, 65408])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "masked",
    ],
)
def test_streaming_topk_contract(block, threshold, width, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(32, width)).astype(np.float32)
    if distribution == "ascending":
        scores[:] = np.arange(width, dtype=np.float32)
    elif distribution == "descending":
        scores[:] = -np.arange(width, dtype=np.float32)
    elif distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        # Equal Kth values survive repeated candidate compactions, with later
        # blocks also supplying larger winners that displace earlier ties.
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:4] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 16383, 50176, width], np.int32), 32
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            block_size=block,
            reconstruct_threshold=threshold,
        )
    )
    for row, length in enumerate(lengths):
        ids = actual[row]
        ids = ids[(ids >= 0) & (scores[row, np.maximum(ids, 0)] > -np.inf)]
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        order = np.lexsort((eligible, ~keys))[:512]
        np.testing.assert_array_equal(np.sort(ids), np.sort(eligible[order]))
        assert np.unique(ids).size == ids.size


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("ragged_owners", [False, True])
def test_sparse_topk_remap_maps_flat_request_rows(start, ragged_owners):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    requests, queries, width, k = 3, 6, 1024, 128
    rng = np.random.default_rng(917)
    values = rng.integers(-4, 5, (requests * queries, width)).astype(np.float32)
    values[:, ::7] = np.float32(-0.0)
    values[:, ::11] = -np.inf
    lengths = np.resize(np.array([0, 1, 129, 777, 1024], np.int32), (requests, queries))
    logical = (
        np.arange((requests + start) * (width // 8), dtype=np.int32).reshape(
            requests + start, -1
        )
        * 2
    )
    owners = (
        rng.integers(start, start + requests, requests * queries, dtype=np.int32)
        if ragged_owners
        else start + np.arange(requests * queries) // queries
    )
    actual = np.asarray(
        sparse_topk_remap(
            jnp.asarray(values),
            valid_lengths=jnp.asarray(
                lengths.reshape(-1) if ragged_owners else lengths
            ),
            k=k,
            block_size=256,
            reconstruct_threshold=256,
            logical_blocks=jnp.asarray(logical),
            row_requests=jnp.asarray(owners, jnp.int32),
        )
    )
    expected = np.full((requests * queries, k), -1, np.int32)
    for row, length in enumerate(lengths.reshape(-1)):
        eligible = np.flatnonzero((np.arange(width) < length) & (values[row] > -np.inf))
        bits = values[row, eligible].view(np.uint32)
        ordered = bits ^ np.where(
            bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000)
        )
        selected = np.sort(eligible[np.lexsort((eligible, ~ordered))[:k]])
        mapped = logical[owners[row], selected // 8] * 8 + selected % 8
        expected[row, : mapped.size] = mapped
    np.testing.assert_array_equal(actual, expected)
