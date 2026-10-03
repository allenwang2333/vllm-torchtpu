"""CPU checks for DeepSeek V4.1 cache addressing and gather paths."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
    _request_row_slots,
    _row_slots,
)
from vllm_torchtpu.models.vllm.deepseek_v41.attention_kernels import (
    paged_index_scores_jax,
    paged_indexer_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.cache import gather_cache_values


@pytest.mark.parametrize("width", [7, 128, 300, 2560])
def test_row_slots_matches_elementwise_gather(width):
    rows_per_page = 32
    generator = torch.Generator().manual_seed(width)
    table = torch.randint(0, 1000, (4, 90), generator=generator, dtype=torch.int32)
    requests = torch.randint(0, 4, (17,), generator=generator)
    logical = torch.randint(
        -5, 90 * rows_per_page + 5, (17, width), generator=generator
    )
    cache = torch.empty((1, rows_per_page))
    safe = logical.clamp(0, table.shape[1] * rows_per_page - 1)
    expected = (
        table[requests[:, None], safe // rows_per_page].long() * rows_per_page
        + safe % rows_per_page
    )
    torch.testing.assert_close(_row_slots(cache, table, requests, logical), expected)


@pytest.mark.parametrize("width", [7, 300, 2560, 4000])
def test_request_row_slots_matches_token_row_slots(width):
    rows_per_page = 32
    generator = torch.Generator().manual_seed(width)
    table = torch.randint(0, 1000, (4, 90), generator=generator, dtype=torch.int32)
    requests = torch.randint(0, 4, (17,), generator=generator)
    cache = torch.empty((1, rows_per_page))
    logical = torch.arange(width)[None, :].expand(requests.shape[0], -1)
    expected = _row_slots(cache, table, requests, logical)
    got = _request_row_slots(cache, table, width).index_select(0, requests)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("elements,fp32", [(512, False), (128, False), (64, True)])
def test_gather_cache_values_negative_records(elements, fp32):
    pool = jnp.asarray(
        np.random.default_rng(0).integers(0, 2**31, (64, 128)), dtype=jnp.int32
    ) & jnp.int32(0x3FFF3FFF)  # finite BF16/FP32 values
    record_words = elements if fp32 else elements // 2
    count = 64 * 128 // record_words
    records = jnp.asarray([3, -1, count - 1, -1, -7, 0], dtype=jnp.int32)
    got = gather_cache_values(pool, records, elements, fp32)
    valid = records >= 0
    expected = gather_cache_values(pool, jnp.where(valid, records, 0), elements, fp32)
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])
    assert got.shape == (records.size, elements)


@pytest.mark.parametrize("tokens,width", [(1, 300), (8, 2560), (64, 640)])
def test_paged_index_scores_matches_dense(tokens, width):
    rng = np.random.default_rng(tokens)
    dim, heads = 128, 4
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    # Keep BF16 keys finite: clear exponent-overflow patterns.
    pool = pool & jnp.int32(0x3FFF3FFF)
    records = rng.integers(0, 256 * 2, (tokens, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    got = np.asarray(
        paged_index_scores_jax(q, pool, jnp.asarray(records), weights, lengths)
    )
    keys = gather_cache_values(
        pool, jnp.asarray(np.maximum(records, 0)).reshape(-1), dim
    ).reshape(tokens, width, dim)
    logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
    dense = np.asarray((jnp.maximum(logits, 0) * weights[:, :, None]).sum(1))
    valid = (records >= 0) & (np.arange(width)[None] < lengths[:, None])
    np.testing.assert_allclose(got[valid], dense[valid], rtol=1e-5, atol=1e-3)


def _request_inputs(rng, requests_count, tokens, width, heads=4, dim=128):
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    pool = pool & jnp.int32(0x3FFF3FFF)  # finite BF16 keys
    records = rng.integers(0, 256 * 2, (requests_count, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    requests = np.sort(rng.integers(0, requests_count, tokens)).astype(np.int32)
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    return pool, records, requests, lengths, q, weights


@pytest.mark.parametrize(
    "requests_count,tokens,width", [(1, 1, 300), (8, 8, 2560), (2, 64, 640)]
)
def test_paged_index_scores_shared_tiles_match_per_token(requests_count, tokens, width):
    rng = np.random.default_rng(tokens)
    pool, records, requests, lengths, q, weights = _request_inputs(
        rng, requests_count, tokens, width
    )
    expected = paged_index_scores_jax(
        q, pool, jnp.asarray(records[requests]), weights, lengths
    )
    got = paged_index_scores_jax(
        q, pool, jnp.asarray(records), weights, lengths, jnp.asarray(requests)
    )
    # Scores at -1 records are undefined (callers mask them); others are exact.
    valid = records[requests] >= 0
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])


@pytest.mark.parametrize(
    "candidate_write,candidate_filter",
    [(False, False), (True, False), (False, True)],
)
def test_paged_indexer_request_records_match_per_token(
    candidate_write, candidate_filter
):
    rng = np.random.default_rng(7)
    width, topk, block_size = 640, 64, 16
    pool, records, requests, lengths, q, weights = _request_inputs(rng, 3, 24, width)
    lengths = np.maximum(lengths, topk + 1)  # long enough for candidate_context
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (24, 8)), dtype=jnp.int32
    )
    args = (topk, block_size, candidate_write, candidate_filter)
    tokens = jnp.arange(24, dtype=jnp.int32)
    expected = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records[requests]),
        weights,
        jnp.asarray(lengths),
        candidates,
        tokens,
        *args,
    )
    got = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.asarray(requests),
        *args,
    )
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
