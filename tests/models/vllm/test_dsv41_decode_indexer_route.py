# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode routing preserves request ownership and candidate semantics."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import sparse_mqa_indexer as mqa
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import IndexCacheRows
from vllm_torchtpu.models.vllm.deepseek_v41.v2.candidate_cache import SparseMQACache


@pytest.fixture
def sparse_mqa_model():
    candidates = torch.arange(32 * 512, dtype=torch.int32).reshape(32, 512)
    indexer = mqa.SparseMQAIndexer(
        SimpleNamespace(prefix="index-keys"),
        512,
        candidates,
        8,
        candidate_source_layer=20,
    )
    cache = IndexCacheRows(
        torch.zeros((128, 128), dtype=torch.int32),
        ((2, 128, 128), torch.bfloat16, 128 * 256, 0),
    )
    inputs = dict(
        cache=cache,
        records=torch.zeros((2, 65536), dtype=torch.int32),
        batch_max=torch.tensor(65536, dtype=torch.int32),
    )
    return indexer, inputs


@pytest.mark.cpu_test
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("per_query_candidates", [False, True])
def test_sparse_mqa_module_compiles_with_dynamic_query_rows(
    monkeypatch, sparse_mqa_model, banks, per_query_candidates
):
    indexer, inputs = sparse_mqa_model
    indexer.per_query_candidates = per_query_candidates
    if banks == 2:
        inputs["cache"] = IndexCacheRows(
            (inputs["cache"].raw, inputs["cache"].raw.clone()),
            ((4, 128, 128), torch.bfloat16, 128 * 256, 0),
        )

    def prepare(cache, records, candidates, offsets, lengths, **kwargs):
        return SparseMQACache(
            cache.raw + 1, candidates, candidates, lengths, lengths > 0, offsets
        )

    def select(
        q, pool, records, weights, lengths, candidates, requests, maximum, keys, *args
    ):
        return candidates[:, :512] + keys[0, 0] + weights[:, :1].int() + int(args[-1])

    def select_banked(q, lo, hi, *args):
        return select(q, lo, *args) + hi[0, 0]

    monkeypatch.setattr(mqa, "prepare_sparse_mqa_cache", prepare)
    monkeypatch.setattr(mqa, "_select_op", select)
    monkeypatch.setattr(mqa, "_select_2bank_op", select_banked)

    def run(q):
        rows = q.shape[0]
        offsets = torch.stack(
            (
                torch.zeros((), dtype=torch.int32),
                torch.zeros((), dtype=torch.int32) + rows // 2,
                torch.zeros((), dtype=torch.int32) + rows,
            )
        )
        return indexer(
            q,
            q.sum(-1).float(),
            inputs["cache"],
            inputs["records"],
            torch.full((rows,), 65536, dtype=torch.int32),
            (torch.arange(rows) >= rows // 2).int(),
            offsets,
            batch_max=inputs["batch_max"],
            page_records=128,
            capacity=49152,
            stage_cache={},
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    for rows in (6, 7, 12):
        q = torch.ones((rows, 32, 128), dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(q, 0, min=2, max=32)
        torch.testing.assert_close(
            compiled(q), indexer.candidate_blocks[:rows] + 129 + per_query_candidates
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("width", [6144, 49408, 98816])
@pytest.mark.parametrize("mode", ["regular", "padded", "draft", "ordered", "publish"])
def test_decode_route(monkeypatch, width, mode):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: tpu_info)
    count, queries, heads = 4, 6, 32
    rows = count * queries
    q = jnp.arange(rows * heads * 128, dtype=jnp.float32).reshape(rows, heads, 128)
    q = q.astype(jnp.bfloat16)
    weights = jnp.arange(rows * heads, dtype=jnp.float32).reshape(rows, heads)
    records = jnp.arange(count * width, dtype=jnp.int32).reshape(count, width)
    lengths = jnp.full(rows, width - 1, jnp.int32)
    requests = jnp.repeat(jnp.arange(count, dtype=jnp.int32), queries)
    if mode == "padded":
        lengths = lengths.at[-3:].set(0)
        requests = requests.at[-3:].set(0)
    elif mode == "draft":
        requests = jnp.arange(rows, dtype=jnp.int32) % count
    candidates = jnp.zeros((rows, 128 if mode == "publish" else 0), jnp.int32)
    pool = (jnp.zeros((128, 128), jnp.int32),) * 2
    calls = []

    def new_select(
        grouped_q,
        grouped_weights,
        pool,
        pages,
        limits,
        *,
        page_records,
        candidate_count,
        candidate_block_size,
    ):
        calls.append(True)
        assert page_records == 128
        assert pages.shape == (count, (1 << (width - 1).bit_length()) // 128)
        np.testing.assert_array_equal(pages[:, : width // 128], records[:, ::128])
        np.testing.assert_array_equal(pages[:, width // 128 :], -1)
        np.testing.assert_array_equal(grouped_q.reshape(q.shape), ak.split_halves(q))
        assert grouped_weights.shape == (count, queries, heads)
        np.testing.assert_array_equal(grouped_weights.reshape(weights.shape), weights)
        np.testing.assert_array_equal(limits.reshape(-1), lengths)
        result = jnp.where(lengths[:, None] > 0, jnp.full((rows, 512), 2), -1)
        assert candidate_block_size == 8
        if mode == "publish":
            assert candidate_count == 128
            return result, jnp.full(candidates.shape, 3, jnp.int32)
        assert candidate_count == 0
        return result

    def old_score(q, pool, records, weights, lengths, requests, page_records):
        return jnp.ones((q.shape[0], records.shape[1]), jnp.float32)

    def old_select(scores, lengths, candidates, *args):
        result = jnp.where(lengths[:, None] > 0, scores[:, :512].astype(jnp.int32), -1)
        return result, candidates

    monkeypatch.setattr(ak, "_decode_select", new_select)
    monkeypatch.setattr(ak, "paged_index_scores_jax", old_score)
    monkeypatch.setattr(ak, "select_indices_jax", old_select)
    with jax.disable_jit():
        selected, published = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            8,
            mode == "publish",
            False,
            mode != "ordered",
            page_records=128,
        )
    optimized = mode in ("regular", "padded", "publish")
    assert bool(calls) == optimized
    expected = np.where(np.asarray(lengths)[:, None] > 0, 2 if optimized else 1, -1)
    np.testing.assert_array_equal(selected, np.broadcast_to(expected, (rows, 512)))
    np.testing.assert_array_equal(published, 3 if mode == "publish" else candidates)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU decode pipeline")
@pytest.mark.parametrize("width", [49408, 98816])
@pytest.mark.parametrize("candidate_count", [0, 2048])
def test_decode_route_tpu_matches_existing(monkeypatch, width, candidate_count):
    rng = np.random.default_rng(721)
    count, queries, page = 4, 6, 128
    keys = jnp.asarray(rng.normal(size=(count * width, 128)), jnp.bfloat16)
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout

    encoded = layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << layout.KEY_BITS)).astype(jnp.int32)
    bank_rows = words.shape[0] // 2
    pool = (words[:bank_rows], words[bank_rows:])
    page_order = rng.permutation(count * width // page).reshape(count, -1)
    records = jnp.asarray(
        (page_order[:, :, None] * page + np.arange(page)).reshape(count, width),
        jnp.int32,
    )
    records = records.at[0, page : 2 * page].set(-1)
    q = jnp.asarray(rng.normal(size=(count * queries, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(0.1, 1, size=(count * queries, 32)), jnp.float32)
    lengths = jnp.full(count * queries, width - 200, jnp.int32)
    lengths = lengths.at[-queries:].set(jnp.array([0, 1, 127, 511, 512, 513]))
    requests = jnp.repeat(jnp.arange(count, dtype=jnp.int32), queries)
    candidates = jnp.zeros((count * queries, candidate_count), jnp.int32)

    def run(q, pool, records, weights, lengths, candidates, requests):
        return ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            8,
            candidate_count > 0,
            False,
            True,
            page_records=page,
        )

    args = (q, pool, records, weights, lengths, candidates, requests)
    candidate = jax.tree.map(np.asarray, jax.jit(run)(*args))
    monkeypatch.setattr(ak.decode_index_selection, "supports_decode", lambda *a: False)
    baseline = jax.tree.map(np.asarray, jax.jit(lambda *a: run(*a))(*args))
    np.testing.assert_array_equal(candidate[0], baseline[0])
    np.testing.assert_array_equal(np.sort(candidate[1]), np.sort(baseline[1]))


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize("per_query_candidates", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "regular",
        "overflow",
        "uneven",
        "short",
        "ties",
        "invalid",
        "ragged_rows",
        "shifted_pages",
    ],
)
def test_sparse_mqa_cache_matches_existing_with_fallback(mode, per_query_candidates):
    _check_sparse_mqa_cache(
        mode,
        groups=2,
        width=8192,
        capacity=4096,
        per_query_candidates=per_query_candidates,
    )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize(
    "groups,mode",
    [
        (1, "regular"),
        (1, "overflow"),
        (1, "invalid"),
        (1, "short"),
        (4, "regular"),
        (4, "uneven"),
    ],
)
def test_long_sparse_mqa_cache_matches_existing(groups, mode):
    _check_sparse_mqa_cache(mode, groups=groups, width=1048576, capacity=49152)


def _check_sparse_mqa_cache(
    mode, *, groups, width, capacity, queries=6, per_query_candidates=False
):
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.candidate_cache import (
        prepare_sparse_mqa_cache_jax,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.sparse_mqa_indexer import (
        _select_indices_jax,
    )

    rng = np.random.default_rng(104)
    rows = groups * queries
    if mode == "ragged_rows":
        rows += 1
    keys = jnp.asarray(rng.normal(size=(groups * width, 128)), jnp.bfloat16)
    encoded = layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = (words[:4096], words[4096:]) if groups > 1 else words
    pages = rng.permutation(groups * width // 128).reshape(groups, -1) * 128
    records = jnp.asarray(
        (pages[..., None] + np.arange(128)).reshape(groups, width), jnp.int32
    )
    if mode == "shifted_pages":
        # Native pages may start on any pool word row, not only block8.
        records += 2
        pool = (jnp.pad(words[:4096], ((1, 0), (0, 0))), words[4096:])
    records = records.at[-1, 128:256].set(-1)
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(rows, 32)), jnp.float32)
    candidate_count = 2048 if width > 131072 else 128
    offset = width // 8 - 16384 if width > 131072 else 0
    candidates = jnp.broadcast_to(
        jnp.arange(candidate_count, dtype=jnp.int32) + offset,
        (rows, candidate_count),
    )
    lengths = jnp.full((rows,), width - 3, jnp.int32)
    if rows > 1:
        lengths = lengths.at[-1].set(0)
    requests = jnp.repeat(jnp.arange(groups, dtype=jnp.int32), queries)
    if mode == "ragged_rows":
        requests = jnp.concatenate((requests, jnp.array([groups - 1], jnp.int32)))
    if mode == "overflow":
        candidates += jnp.tile(jnp.arange(queries) * candidate_count, groups)[:, None]
    elif mode == "uneven":
        requests = requests.at[queries].set(0)
    elif mode == "short":
        lengths = jnp.arange(rows, dtype=jnp.int32) * 7
    elif mode == "ties":
        q = jnp.zeros_like(q)
    elif mode == "invalid":
        candidates = candidates.at[:, :4].set(jnp.array([-1, width // 8, 8, 8]))
        candidates = candidates.at[0].set(-1)
        lengths = lengths.at[min(1, rows - 1)].set(519)
    batch_max = lengths.max().reshape(1)
    prepare = jax.jit(prepare_sparse_mqa_cache_jax, static_argnums=(5, 6))
    query_offsets = jnp.asarray(
        np.r_[0, np.cumsum(np.bincount(np.asarray(requests), minlength=groups))],
        jnp.int32,
    )
    dense, ids, bits, actual, usable = prepare(
        pool, records, candidates, query_offsets, lengths, 8, capacity
    )
    assert bool(usable[0]) == (mode != "overflow")
    if mode == "overflow":
        assert np.all(np.asarray(actual) * 8 > capacity)
    select = jax.jit(_select_indices_jax, static_argnums=(14, 15, 16, 17))
    result = select(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        dense,
        ids,
        bits,
        actual,
        usable,
        query_offsets,
        8,
        128,
        None,
        per_query_candidates,
    )
    expected, _ = ak.paged_indexer_jax(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        512,
        8,
        False,
        True,
        True,
        batch_max,
        128,
    )
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize("groups,queries", [(1, 1), (3, 2), (4, 3), (7, 5), (8, 8)])
@pytest.mark.parametrize("mode", ["regular", "invalid"])
def test_sparse_mqa_cache_basic_config_covers_non_k5_batches(groups, queries, mode):
    _check_sparse_mqa_cache(
        mode, groups=groups, queries=queries, width=8192, capacity=4096
    )
