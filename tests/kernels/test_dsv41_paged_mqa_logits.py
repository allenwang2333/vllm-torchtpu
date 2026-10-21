# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.dsv41_bench.paged_mqa_cache import pack_pages
from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection.paged_mqa_logits import (
    paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
)


def make_inputs(counts, width, page_records, banks, seed=2026, heads=32):
    rng = np.random.default_rng(seed)
    requests, tokens = len(counts), sum(counts)
    native = jnp.asarray(rng.normal(size=(requests * width, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(native)
    packed = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = packed if banks == 1 else tuple(jnp.split(packed, 2))
    pages = rng.permutation(requests * width // page_records).reshape(requests, -1)
    pages = pages.astype(np.int32)
    pages[0, 1] = -1
    pages[-1, -1] = requests * width // page_records
    offsets = np.cumsum([0, *counts], dtype=np.int32)
    q = jnp.asarray(rng.normal(size=(tokens, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(tokens, heads)), jnp.float32)
    lengths = np.full(tokens, width - 5, np.int32)
    lengths[0] = 0
    if tokens > 2:
        lengths[1] = page_records + 7
    return (
        (
            q,
            weights,
            pool,
            jnp.asarray(pages),
            jnp.asarray(lengths),
            jnp.asarray(offsets),
        ),
        native,
    )


def reference(inputs, native, page_records, q_scale=None, k_scale=None):
    q, weights, _, pages, lengths, offsets = inputs
    owners = jnp.searchsorted(offsets[1:], jnp.arange(q.shape[0]), side="right")
    logical = jnp.arange(pages.shape[1] * page_records)
    physical_pages = pages[owners[:, None], logical[None, :] // page_records]
    records = jnp.clip(physical_pages, 0) * page_records + logical % page_records
    keys = native[jnp.minimum(records, native.shape[0] - 1)]
    if q_scale is None:
        keys = jnp.concatenate((keys[..., ::2], keys[..., 1::2]), axis=-1)
    logits = jnp.einsum("thd,tkd->thk", q, keys, preferred_element_type=jnp.float32)
    if q_scale is not None:
        scales = k_scale[jnp.minimum(records, native.shape[0] - 1)]
        weights = weights * q_scale
    logits = jnp.maximum(logits, 0).astype(jnp.bfloat16).astype(jnp.float32)
    scores = (logits * weights[..., None]).sum(1)
    if q_scale is not None:
        scores = scores * scales
    valid = (physical_pages >= 0) & (physical_pages < native.shape[0] // page_records)
    valid &= logical[None, :] < lengths[:, None]
    return jnp.where(valid, scores, -jnp.inf)


@pytest.mark.parametrize("banks,page_records", [(1, 128), (2, 128)])
@pytest.mark.parametrize("key_buffers", [2, 4])
def test_paged_mqa_fp8_scales(banks, page_records, key_buffers):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires native TPU v7 FP8 MXU")
    inputs, native = make_inputs((0, 1, 0, 3, 1, 0), 1024, page_records, banks)
    q, weights, _, pages, _, offsets = inputs
    rng = np.random.default_rng(451)
    q = q.astype(jnp.float8_e4m3fn)
    keys = native.astype(jnp.float8_e4m3fn)
    q_scale = jnp.asarray(rng.uniform(0.25, 2, q.shape[:2]), jnp.float32)
    key_scales = jnp.asarray(rng.uniform(0.25, 2, keys.shape[:1]), jnp.float32)
    pool = keys if banks == 1 else tuple(jnp.split(keys, 2))
    lengths = jnp.array([0, 1024, 513, 37, 257], jnp.int32)
    inputs = q, weights, pool, pages, lengths, offsets
    packed = pack_pages(keys, key_scales)
    packed_pool = packed if banks == 1 else tuple(jnp.split(packed, 2))
    actual = paged_mqa_logits(
        q,
        weights,
        packed_pool,
        pages,
        lengths,
        offsets,
        page_records=page_records,
        q_scale=q_scale,
        config=PagedMQAConfig(
            query_tile=2,
            query_group=1,
            key_tile=512,
            issue_depth=2,
            key_buffers=key_buffers,
        ),
    )
    np.testing.assert_allclose(
        actual,
        reference(inputs, keys, page_records, q_scale, key_scales),
        rtol=2e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize("banks,key_buffers", [(1, 2), (2, 4)])
def test_paired_paged_mqa(banks, key_buffers):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires native TPU v7 FP8 MXU")
    inputs, native = make_inputs((0, 1, 3, 2, 0), 1024, 128, banks)
    q, weights, _, pages, _, offsets = inputs
    q, keys = q.astype(jnp.float8_e4m3fn), native.astype(jnp.float8_e4m3fn)
    rng = np.random.default_rng(452)
    q_scale = jnp.asarray(rng.uniform(0.25, 2, q.shape[:2]), jnp.float32)
    key_scales = jnp.asarray(rng.uniform(0.25, 2, keys.shape[:1]), jnp.float32)
    packed = pack_pages(keys, key_scales)
    pool = packed if banks == 1 else tuple(jnp.split(packed, 2))
    lengths = jnp.array([0, 1024, 513, 37, 257, 1024], jnp.int32)
    candidates = jnp.array(
        [
            [-1, -1, -1, -1],
            [0, 1, 3, 6],
            [1, 2, 3, 7],
            [0, 2, 5, -1],
            [1, 3, -1, -1],
            [1, 3, 6, 7],
        ],
        jnp.int32,
    )
    candidates = jnp.pad(candidates, ((0, 0), (0, 124)), constant_values=-1)
    pages = jnp.pad(pages, ((0, 0), (0, 120)), constant_values=-1)
    actual = paged_mqa_logits(
        q,
        weights,
        pool,
        pages,
        lengths,
        offsets,
        page_records=128,
        q_scale=q_scale,
        candidates=candidates,
        config=PagedMQAConfig(2, 512, 2, 2, key_buffers),
    )
    dense = reference(
        (q, weights, pool, pages, lengths, offsets), keys, 128, q_scale, key_scales
    )
    positions = candidates[..., None] * 128 + jnp.arange(128)
    expected = jnp.take_along_axis(
        dense, jnp.maximum(positions, 0).reshape(q.shape[0], -1), axis=1
    )
    expected = jnp.where(jnp.repeat(candidates >= 0, 128, axis=1), expected, -jnp.inf)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-4)


def test_empty_candidate_queries():
    result = paged_mqa_logits(
        jnp.empty((0, 32, 128), jnp.float8_e4m3fn),
        jnp.empty((0, 32), jnp.float32),
        jnp.zeros((33, 128), jnp.int32),
        jnp.zeros((1, 128), jnp.int32),
        jnp.empty((0,), jnp.int32),
        jnp.array([0, 0], jnp.int32),
        page_records=128,
        q_scale=jnp.empty((0, 32), jnp.float32),
        candidates=jnp.empty((0, 128), jnp.int32),
    )
    assert result.shape == (0, 16384)
