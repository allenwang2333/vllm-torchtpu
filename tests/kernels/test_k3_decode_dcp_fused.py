# SPDX-License-Identifier: Apache-2.0
"""Interpret the pipelined DMA/mask schedule without TPU hardware."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = Path(__file__).resolve(
).parents[2] / 'src/vllm_torchtpu/kernels/mla/decode_dcp_fused.py'
_SPEC = importlib.util.spec_from_file_location('k3_cp_fused', _PATH)
kernel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernel)


@pytest.mark.parametrize('group_size', [1, 2, 4])
@pytest.mark.parametrize('tile_pages', [1, 2, 4])
def test_pipeline_uneven_noncontiguous_pages(group_size, tile_pages):
    rng = np.random.default_rng(17)
    lens = [65, 17, 1, 0]
    tables = np.array(
        [[9, 2, 7, 4, 11], [3, 10, 0, 0, 0], [5, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        np.int32)
    cache = np.full((12, 16, 640), np.nan, np.float32)
    refs = []
    q = jnp.asarray(rng.normal(0, .2, (4, 24, 640)), jnp.bfloat16)
    for b, n in enumerate(lens):
        rows = jnp.asarray(rng.normal(0, .2, (n, 640)), jnp.bfloat16)
        for i in range(n):
            cache[tables[b, i // 16], i % 16] = rows[i]
        scores = q[b].astype(jnp.float32) @ rows.astype(
            jnp.float32).T * 192**-.5
        if n:
            refs.append(
                (jax.nn.softmax(scores) @ rows[:, :512].astype(jnp.float32),
                 jax.scipy.special.logsumexp(scores, axis=-1)))
    cache = jnp.asarray(cache.reshape(12, 8, 2, 640), jnp.bfloat16)
    actual, lse = kernel.local_attention(cache,
                                         q,
                                         jnp.array(lens),
                                         jnp.array(tables),
                                         scale=192**-.5,
                                         tile_pages=tile_pages,
                                         group_size=group_size,
                                         interpret=True)
    for b, (expected, expected_lse) in enumerate(refs):
        np.testing.assert_allclose(actual[b].astype(jnp.float32),
                                   expected,
                                   atol=.005,
                                   rtol=.03)
        np.testing.assert_allclose(lse[b], expected_lse, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual[3], 0)
    assert np.isneginf(lse[3]).all()
    # Cache is read-only, including the deliberately poisoned unused pages.
    assert np.isnan(cache[0]).all()


@pytest.mark.parametrize('tile_pages', [16, 32])
@pytest.mark.parametrize('group_size', [2, 4])
@pytest.mark.parametrize('partial_group', [False, True])
def test_pipeline_partial_dma_chunks(tile_pages, group_size, partial_group):
    # Partial tiles straddle the 8-page DMA chunk boundary. Unused table
    # entries point at a poisoned page, and physical pages are noncontiguous.
    rng = np.random.default_rng(811)
    lengths = [113, 128, 129, 0 if partial_group else 241]
    tables = np.zeros((4, 32), np.int32)
    cache = np.full((65, 16, 640), np.nan, np.float32)
    physical = rng.permutation(64) + 1
    q = jnp.asarray(rng.normal(0, .2, (4, 24, 640)), jnp.bfloat16)
    refs = []
    cursor = 0
    for b, length in enumerate(lengths):
        if not length:
            refs.append(jnp.zeros((24, 512), jnp.float32))
            continue
        rows = jnp.asarray(rng.normal(0, .2, (length, 640)), jnp.bfloat16)
        pages = (length + 15) // 16
        tables[b, :pages] = physical[cursor:cursor + pages]
        cursor += pages
        for token in range(length):
            cache[tables[b, token // 16], token % 16] = rows[token]
        scores = q[b].astype(jnp.float32) @ rows.astype(
            jnp.float32).T * 192**-.5
        refs.append(jax.nn.softmax(scores) @ rows[:, :512].astype(jnp.float32))
    actual, _ = kernel.local_attention(jnp.asarray(
        cache.reshape(65, 8, 2, 640), jnp.bfloat16),
                                       q,
                                       jnp.asarray(lengths),
                                       jnp.asarray(tables),
                                       scale=192**-.5,
                                       tile_pages=tile_pages,
                                       group_size=group_size,
                                       interpret=True)
    np.testing.assert_allclose(actual.astype(jnp.float32),
                               jnp.stack(refs),
                               atol=.005,
                               rtol=.03)


def test_pipeline_all_empty_does_not_read_poisoned_cache():
    cache = jnp.full((1, 8, 2, 640), jnp.nan, jnp.bfloat16)
    q = jnp.ones((4, 24, 640), jnp.bfloat16)
    actual, lse = kernel.local_attention(cache,
                                         q,
                                         jnp.zeros(4, jnp.int32),
                                         jnp.zeros((4, 1), jnp.int32),
                                         scale=192**-.5,
                                         tile_pages=2,
                                         group_size=2,
                                         interpret=True)
    np.testing.assert_array_equal(actual, 0)
    assert np.isneginf(lse).all()


@pytest.mark.parametrize('query_len', [1, 4])
@pytest.mark.parametrize('active', [0, 5])
@pytest.mark.parametrize('group_size', [1, 4])
@pytest.mark.parametrize('tile_pages', [2, 128])
def test_distributed_owner_writes_and_bank_reuse(active, group_size,
                                                 tile_pages, query_len):
    if jax.default_backend() != 'tpu' or jax.device_count() < 2:
        pytest.skip('Requires two TPU devices for remote DMA')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('dcp', ))
    rng = np.random.default_rng(817)
    lengths = np.array(([1, 17, 33, 65, 129, 0, 0, 0] if query_len == 1 else
                        [18, 32, 35, 65, 129, 0, 0, 0]) if active else [0] * 8,
                       np.int32)
    starts = jnp.minimum(jnp.arange(9, dtype=jnp.int32), active) * query_len
    tokens = 8 * query_len
    q = jnp.asarray(rng.normal(size=(tokens, 24, 576)), jnp.bfloat16)
    new = jnp.asarray(rng.normal(size=(tokens, 576)), jnp.bfloat16)
    rows = [
        jnp.asarray(rng.normal(size=(max(int(n), 1), 640)), jnp.bfloat16)
        for n in lengths
    ]
    caches, expected_caches, tables = [], [], []
    for rank in range(2):
        cache = jnp.full((40, 16, 640), jnp.nan, jnp.bfloat16)
        table = np.zeros((8, 8), np.int32)
        physical_pages = rng.permutation(39) + 1
        next_page = 0
        for b, length in enumerate(lengths):
            local = 0
            for token in range(length):
                if token // 16 % 2 != rank:
                    continue
                if local % 16 == 0:
                    table[b, local // 16] = physical_pages[next_page]
                    next_page += 1
                cache = cache.at[table[b, local // 16],
                                 local % 16].set(rows[b][token])
                local += 1
        expected = cache
        for b, length in enumerate(lengths[:active]):
            for offset in range(query_len):
                position = length - query_len + offset
                if position // 16 % 2 == rank:
                    pos = position // 32 * 16 + position % 16
                    expected = expected.at[table[b, pos // 16], pos % 16].set(
                        jnp.pad(new[b * query_len + offset], (0, 64)))
        caches.append(cache.reshape(40, 8, 2, 640))
        expected_caches.append(expected.reshape(40, 8, 2, 640))
        tables.append(table)
    cache = jnp.concatenate(caches)
    table = jnp.asarray(np.concatenate(tables))
    expected_cache = jnp.concatenate(expected_caches)
    references = []
    for b, length in enumerate(lengths[:active]):
        kv = rows[b][:length, :576].astype(jnp.float32).at[-query_len:].set(
            new[b * query_len:(b + 1) * query_len].astype(jnp.float32))
        for offset in range(query_len):
            causal = kv[:length - query_len + offset + 1]
            scores = q[b * query_len + offset].astype(jnp.float32) @ causal.T
            references.append(
                jnp.matmul(jax.nn.softmax(scores * 192**-.5),
                           causal[:, :512],
                           precision=jax.lax.Precision.HIGHEST))
    fn = jax.jit(lambda c: kernel.decode_attention(c,
                                                   q[:, :, :512],
                                                   q[:, :, 512:],
                                                   new[:, :512],
                                                   new[:, 512:],
                                                   jnp.asarray(lengths),
                                                   table,
                                                   starts,
                                                   mesh=mesh,
                                                   scale=192**-.5,
                                                   group_size=group_size,
                                                   tile_pages=tile_pages))
    for _ in range(3):
        updated, actual = jax.block_until_ready(fn(cache))
        np.testing.assert_allclose(updated.astype(jnp.float32),
                                   expected_cache.astype(jnp.float32),
                                   atol=0,
                                   rtol=0,
                                   equal_nan=True)
        if active:
            np.testing.assert_allclose(actual[:active * query_len].astype(
                jnp.float32),
                                       jnp.stack(references),
                                       atol=.005,
                                       rtol=.03)
        np.testing.assert_array_equal(actual[active * query_len:], 0)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("flatten_tables", [False, True])
def test_mtp_suffix_positions_and_causal_lengths(rank, flatten_tables):
    # Mixed query counts, an empty request, stripe boundaries and padded tokens.
    lengths = jnp.array([18, 32, 0, 35, 0], jnp.int32)
    starts = jnp.array([0, 4, 5, 5, 8, 8], jnp.int32)
    tables = jnp.array([[3, 7], [2, 6], [0, 0], [1, 5], [0, 0]])
    cache = jnp.full((8, 8, 2, 640), -7, jnp.bfloat16)
    q = jnp.zeros((12, 12, 512), jnp.bfloat16)
    qp = jnp.zeros((12, 12, 64), jnp.bfloat16)
    values = jnp.broadcast_to(jnp.arange(12)[:, None], (12, 576))
    updated, _, causal, token_tables, count = kernel._prepare_decode(
        cache,
        q,
        qp,
        values[:, :512],
        values[:, 512:],
        lengths,
        tables.reshape(-1) if flatten_tables else tables,
        starts,
        rank=rank)
    positions = [14, 15, 16, 17, 31, 32, 33, 34]
    requests = [0, 0, 0, 0, 1, 3, 3, 3]
    expected = np.asarray(cache).copy()
    expected_lengths = np.zeros(12, np.int32)
    for token, (position, req) in enumerate(zip(positions, requests)):
        n = position + 1
        expected_lengths[token] = n // 32 * 16 + min(
            max(n % 32 - rank * 16, 0), 16)
        if position // 16 % 2 == rank:
            local = position // 32 * 16 + position % 16
            expected[tables[req, local // 16], local % 16 // 2,
                     local % 2, :576] = token
            expected[tables[req, local // 16], local % 16 // 2, local % 2,
                     576:] = 0
    np.testing.assert_array_equal(updated, expected)
    np.testing.assert_array_equal(causal, expected_lengths)
    np.testing.assert_array_equal(token_tables[:8],
                                  tables[jnp.array(requests)])
    np.testing.assert_array_equal(count, [8])
