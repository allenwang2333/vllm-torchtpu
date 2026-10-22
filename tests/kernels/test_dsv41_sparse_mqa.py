# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    sparse_mqa_logits,
    sparse_mqa_metadata,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "counts", [(6, 6, 6, 6), (7, 5, 6, 6), (4, 0, 4, 0), (0, 17, 1, 0)]
)
@pytest.mark.parametrize("inactive", [False, True])
def test_ragged_union_preserves_ownership_membership_and_overflow(counts, inactive):
    rng = np.random.default_rng(593)
    requests = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    lengths = np.full(requests.size, 128, np.int32)
    if inactive:
        lengths[::2] = 0
    candidates = rng.integers(-2, 18, (requests.size, 12), dtype=np.int32)
    offsets = jnp.asarray(np.cumsum([0, *counts]), jnp.int32)
    grouped = [np.flatnonzero(requests == r) for r in range(len(counts))]
    for capacity in (4, 16, 35):
        meta = jax.jit(
            sparse_mqa_metadata.union_metadata,
            static_argnames=("logical_blocks", "capacity_blocks"),
        )(
            jnp.asarray(candidates),
            jnp.asarray(lengths),
            offsets,
            logical_blocks=16,
            capacity_blocks=capacity,
        )
        for request, rows in enumerate(grouped):
            values = candidates[rows[lengths[rows] > 0]].reshape(-1)
            expected = np.unique(values[(values >= 0) & (values < 16)])
            assert int(meta.union_block_count[request]) == expected.size
            assert bool(meta.overflow[request]) == (expected.size > capacity)
            padded = np.full(capacity, -1, np.int32)
            padded[: min(expected.size, capacity)] = expected[:capacity]
            np.testing.assert_array_equal(meta.union_block_ids[request], padded)
        word_offsets = np.asarray(sparse_mqa_metadata.membership_offsets(offsets))
        for request, rows in enumerate(grouped):
            for local_query, row in enumerate(rows):
                for slot, logical in enumerate(
                    np.asarray(meta.union_block_ids[request])
                ):
                    word = int(word_offsets[request]) + local_query // 32
                    bit = (int(meta.membership[word, slot]) >> (local_query % 32)) & 1
                    assert bit == int(
                        lengths[row] > 0 and logical >= 0 and logical in candidates[row]
                    )
        assert not np.any(np.asarray(meta.membership[int(word_offsets[-1]) :]))


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [0, 5])
def test_ragged_empty_queries_keep_static_shapes(tokens):
    offsets = jnp.zeros(5, jnp.int32)
    meta = sparse_mqa_metadata.union_metadata(
        jnp.zeros((tokens, 12), jnp.int32),
        jnp.zeros(tokens, jnp.int32),
        offsets,
        logical_blocks=16,
        capacity_blocks=8,
    )
    assert not np.any(np.asarray(meta.union_block_count))
    np.testing.assert_array_equal(meta.union_block_ids, -1)
    np.testing.assert_array_equal(meta.membership, 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("start", [0, 2, 3])
@pytest.mark.parametrize("candidate_limit", [5, 16])
def test_flat_ragged_scores_read_one_key_cache_per_request(
    heads, start, candidate_limit
):
    rng = np.random.default_rng(731)
    # Request 0 crosses a membership word boundary; the batch ends in padding.
    tokens = 42 - start
    offsets = jnp.maximum(jnp.array([0, 37, 40, 40], jnp.int32) - start, 0)
    lengths = jnp.full(42, 128, jnp.int32).at[5].set(0).at[40:].set(0)[start:]
    values = jnp.asarray(rng.integers(-1, candidate_limit, (tokens, 12)), jnp.int32)
    meta = sparse_mqa_metadata.union_metadata(
        values, lengths, offsets, logical_blocks=16, capacity_blocks=16
    )
    native = jnp.asarray(rng.normal(size=(3, 128, 128)), jnp.bfloat16)
    slots = jnp.maximum(meta.union_block_ids, 0)[..., None] * 8 + jnp.arange(8)
    gathered = native[jnp.arange(3)[:, None], slots.reshape(3, -1)]
    encoded = index_key_layout.encode_records(gathered.reshape(-1, 128)).reshape(
        3, 128, 128
    )
    keys = (encoded[:, ::2] | (encoded[:, 1::2] << 16)).astype(jnp.int32)
    q = jnp.asarray(rng.normal(size=(tokens, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(tokens, heads)), jnp.float32)
    owners = jnp.minimum(
        jnp.searchsorted(offsets[1:], jnp.arange(tokens), side="right"), 2
    )
    bits = sparse_mqa_metadata.membership_offsets(offsets)[owners] * 32
    bits += jnp.arange(tokens) - offsets[owners]
    rows = jnp.stack(
        (
            owners,
            bits,
            jnp.minimum(meta.union_block_count[owners], 16) * 8,
            lengths,
        )
    )
    scores = sparse_mqa_logits.sparse_mqa_logits(
        q,
        weights,
        keys,
        meta.union_block_ids,
        meta.membership,
        rows,
        block_size=8,
        key_tile=64,
        compute_tile=32,
        interpret=True,
    )
    split_keys = jnp.concatenate((gathered[..., ::2], gathered[..., 1::2]), axis=-1)
    logits = jnp.einsum(
        "thd,tkd->thk", q, split_keys[owners], preferred_element_type=jnp.float32
    )
    expected = np.asarray((jnp.maximum(logits, 0) * weights[..., None]).sum(1)).copy()
    for row in range(tokens):
        for index in range(128):
            block = int(meta.union_block_ids[owners[row], index // 8])
            if lengths[row] == 0 or block < 0 or block not in np.asarray(values[row]):
                expected[row, index] = -np.inf
    assert scores.shape == (tokens, 128)
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=4e-5)


@pytest.mark.cpu_test
@pytest.mark.parametrize("capacity", [35, 96])
def test_ragged_bitsets_cross_word_boundaries_with_more_than_32_queries(capacity):
    rng = np.random.default_rng(945)
    owners = np.repeat(np.arange(5, dtype=np.int32), [37, 3, 0, 2, 1])
    owners = np.concatenate((owners, [-1, -1])).astype(np.int32)
    lengths = np.where(owners >= 0, 768, 0).astype(np.int32)
    lengths[31] = 0
    values = rng.integers(-2, 98, (owners.size, 128), dtype=np.int32)
    offsets = jnp.array([0, 37, 40, 40, 42, 43], jnp.int32)
    meta = sparse_mqa_metadata.union_metadata(
        jnp.asarray(values),
        jnp.asarray(lengths),
        offsets,
        logical_blocks=96,
        capacity_blocks=capacity,
    )
    for request in range(5):
        rows = np.flatnonzero(owners == request)
        union = np.unique(values[rows[lengths[rows] > 0]])
        union = union[(union >= 0) & (union < 96)]
        expected_ids = np.full(capacity, -1, np.int32)
        expected_ids[: min(capacity, union.size)] = union[:capacity]
        np.testing.assert_array_equal(meta.union_block_ids[request], expected_ids)
        assert int(meta.union_block_count[request]) == union.size
        first_word = int(sparse_mqa_metadata.membership_offsets(offsets)[request])
        for local_query, row in enumerate(rows):
            words = np.asarray(meta.membership[first_word + local_query // 32])
            bits = (words >> (local_query % 32)) & 1
            expected = (
                (lengths[row] > 0)
                & (expected_ids >= 0)
                & np.isin(expected_ids, values[row])
            )
            np.testing.assert_array_equal(bits, expected)
    used_words = int(sparse_mqa_metadata.membership_offsets(offsets)[-1])
    np.testing.assert_array_equal(meta.membership[used_words:], 0)


@pytest.mark.cpu_test
def test_query_metadata_keeps_unique_candidates_and_skips_missing_cache_blocks():
    candidates = jnp.array(
        [[7, 2, 7, -1, 4, 8, 2, 9], [13, 9, 1, 9, -1, 12, 22, 1], [7] * 8],
        jnp.int32,
    )
    ids, starts, counts = sparse_mqa_metadata.query_metadata(
        candidates,
        jnp.array([[2, -1, 7, -1], [1, 9, 13, -1]], jnp.int32),
        jnp.array([0, 1, 0], jnp.int32),
        jnp.array([60, 77, 0], jnp.int32),
        block_size=8,
    )
    np.testing.assert_array_equal(counts, [2, 2, 0])
    np.testing.assert_array_equal(ids[:, :2], [[2, 7], [1, 9], [-1, -1]])
    np.testing.assert_array_equal(starts[:, :2], [[0, 16], [32, 40], [-1, -1]])
    np.testing.assert_array_equal(ids[:, 2:], -1)
    np.testing.assert_array_equal(starts[:, 2:], -1)


@pytest.mark.cpu_test
def test_sparse_mqa_indexer_requires_flat_queries():
    with pytest.raises(ValueError, match="flat Q"):
        sparse_mqa_logits.sparse_mqa_indexer(
            jnp.zeros((2, 6, 4, 128), jnp.bfloat16),
            jnp.zeros((12, 4), jnp.float32),
            jnp.zeros((2, 64, 128), jnp.int32),
            jnp.zeros((2, 16), jnp.int32),
            jnp.zeros((3, 16), jnp.int32),
            jnp.zeros((12,), jnp.int32),
            query_start_loc=jnp.array([0, 6, 12], jnp.int32),
            union_counts=jnp.zeros((2,), jnp.int32),
            block_size=8,
        )


def _cpu_indexer(monkeypatch, *, key_tile=64, compute_tile=32):
    """Use the real Pallas scorer and CPU selection in place of SparseCore."""
    score = sparse_mqa_logits.sparse_mqa_logits
    score_widths = []

    def interpreted(*args, **kwargs):
        values = score(*args, **kwargs, interpret=True)
        score_widths.append(values.shape[1])
        return values

    def cpu_topk(values, *, valid_lengths, logical_blocks, row_requests, k, **kwargs):
        valid = jnp.arange(values.shape[1]) < valid_lengths[:, None]
        winners, selected = jax.lax.top_k(jnp.where(valid, values, -jnp.inf), k)
        blocks = logical_blocks[row_requests[:, None], selected // 8]
        logical = blocks * 8 + selected % 8
        sentinel = jnp.iinfo(jnp.int32).max
        logical = jnp.sort(jnp.where(jnp.isfinite(winners), logical, sentinel))
        return jnp.where(logical == sentinel, -1, logical)

    monkeypatch.setattr(sparse_mqa_logits, "sparse_mqa_logits", interpreted)
    monkeypatch.setattr(sparse_mqa_logits, "sparse_topk_remap", cpu_topk)
    monkeypatch.setattr(
        sparse_mqa_logits.tuned_block_sizes,
        "get_sparse_mqa_block_sizes",
        lambda key, rows: sparse_mqa_logits.tuned_block_sizes.SparseMQABlockSizes(
            key_tile, rows, compute_tile=compute_tile
        ),
    )
    return score_widths


@pytest.mark.cpu_test
@pytest.mark.parametrize("row_tile", [1, 5, 16])
@pytest.mark.parametrize("per_query_candidates", [False, True])
def test_flat_indexer_keeps_query_order_across_batch_and_request_boundaries(
    monkeypatch, row_tile, per_query_candidates
):
    rng = np.random.default_rng(113)
    offsets = jnp.array([0, 1, 8, 8, 11], jnp.int32)
    lengths = jnp.array(
        [40, 128, 0, 97, 128, 31, 128, 5, 128, 64, 128, 0, 0], jnp.int32
    )
    candidates = jnp.asarray(rng.integers(-1, 18, (13, 8)), jnp.int32)
    meta = sparse_mqa_metadata.union_metadata(
        candidates, lengths, offsets, logical_blocks=16, capacity_blocks=16
    )
    native = jnp.asarray(rng.normal(size=(4, 128, 128)), jnp.bfloat16)
    slots = jnp.maximum(meta.union_block_ids, 0)[..., None] * 8 + jnp.arange(8)
    gathered = native[jnp.arange(4)[:, None], slots.reshape(4, -1)]
    encoded = index_key_layout.encode_records(gathered.reshape(-1, 128)).reshape(
        4, 128, 128
    )
    keys = (encoded[:, ::2] | (encoded[:, 1::2] << 16)).astype(jnp.int32)
    q = jnp.asarray(rng.normal(size=(13, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(13, 4)), jnp.float32)
    score_widths = _cpu_indexer(monkeypatch)
    actual = sparse_mqa_logits.sparse_mqa_indexer(
        q,
        weights,
        keys,
        meta.union_block_ids,
        meta.membership,
        lengths,
        query_start_loc=offsets,
        union_counts=meta.union_block_count,
        block_size=8,
        row_tile=row_tile,
        k=8,
        candidate_block_ids=candidates if per_query_candidates else None,
    )
    owners = np.minimum(
        np.searchsorted(np.asarray(offsets[1:]), np.arange(13), side="right"), 3
    )
    split_keys = jnp.concatenate((native[..., ::2], native[..., 1::2]), axis=-1)
    logits = jnp.einsum(
        "thd,tkd->thk", q, split_keys[owners], preferred_element_type=jnp.float32
    )
    values = np.asarray((jnp.maximum(logits, 0) * weights[..., None]).sum(1))
    expected = np.full((13, 8), -1, np.int32)
    for row in range(13):
        valid = (np.arange(128) < lengths[row]) & np.isin(
            np.arange(128) // 8, candidates[row]
        )
        eligible = np.flatnonzero(valid)
        selected = eligible[np.argsort(-values[row, eligible], kind="stable")[:8]]
        expected[row, : selected.size] = np.sort(selected)
    np.testing.assert_array_equal(actual, expected)
    assert score_widths and set(score_widths) == {64 if per_query_candidates else 128}


@pytest.mark.cpu_test
@pytest.mark.parametrize("per_query_candidates", [False, True])
def test_2048_candidates_select_same_ids_with_different_score_widths(
    monkeypatch, per_query_candidates
):
    score_widths = _cpu_indexer(monkeypatch, key_tile=4096, compute_tile=4096)
    candidates = jnp.arange(2048, dtype=jnp.int32)[None, ::-1]
    membership = (jnp.arange(6144) < 2048)[None].astype(jnp.int32)
    actual = sparse_mqa_logits.sparse_mqa_indexer(
        jnp.zeros((1, 4, 128), jnp.bfloat16),
        jnp.ones((1, 4), jnp.float32),
        jnp.zeros((1, 49152 // 2, 128), jnp.int32),
        jnp.arange(6144, dtype=jnp.int32)[None],
        membership,
        jnp.array([49152], jnp.int32),
        query_start_loc=jnp.array([0, 1], jnp.int32),
        union_counts=jnp.array([6144], jnp.int32),
        block_size=8,
        row_tile=1,
        candidate_block_ids=candidates if per_query_candidates else None,
    )
    np.testing.assert_array_equal(actual, np.arange(512)[None])
    assert score_widths == [16384 if per_query_candidates else 49152]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "candidates",
    [
        jnp.zeros((1, 8), jnp.float32),
        jnp.zeros((2, 8), jnp.int32),
        jnp.zeros((1, 0), jnp.int32),
    ],
)
def test_per_query_candidates_require_one_int32_list_per_query(candidates):
    with pytest.raises(ValueError, match="candidate_block_ids must be int32"):
        sparse_mqa_logits.sparse_mqa_indexer(
            jnp.zeros((1, 4, 128), jnp.bfloat16),
            jnp.zeros((1, 4), jnp.float32),
            jnp.zeros((1, 64, 128), jnp.int32),
            jnp.zeros((1, 16), jnp.int32),
            jnp.zeros((1, 16), jnp.int32),
            jnp.ones((1,), jnp.int32),
            query_start_loc=jnp.array([0, 1], jnp.int32),
            union_counts=jnp.ones((1,), jnp.int32),
            block_size=8,
            candidate_block_ids=candidates,
        )
