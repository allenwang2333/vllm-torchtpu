"""CPU checks for DeepSeek V4.1 cache addressing and gather paths."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.attention import _row_slots
from vllm_torchtpu.models.vllm.deepseek_v41.attention_kernels import (
    paged_index_scores_jax,
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
