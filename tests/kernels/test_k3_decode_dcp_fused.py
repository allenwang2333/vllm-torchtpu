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


@pytest.mark.parametrize('active', [0, 5])
@pytest.mark.parametrize('group_size', [1, 4])
@pytest.mark.parametrize('tile_pages', [2, 128])
def test_distributed_owner_writes_and_bank_reuse(active, group_size,
                                                 tile_pages):
    if jax.default_backend() != 'tpu' or jax.device_count() < 2:
        pytest.skip('Requires two TPU devices for remote DMA')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('dcp', ))
    rng = np.random.default_rng(817)
    lengths = np.array([1, 17, 33, 65, 129, 0, 0, 0] if active else [0] * 8,
                       np.int32)
    starts = jnp.minimum(jnp.arange(9, dtype=jnp.int32), active)
    q = jnp.asarray(rng.normal(size=(8, 24, 576)), jnp.bfloat16)
    new = jnp.asarray(rng.normal(size=(8, 576)), jnp.bfloat16)
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
        for b, length in enumerate(lengths):
            if length and (length - 1) // 16 % 2 == rank:
                pos = (length - 1) // 32 * 16 + (length - 1) % 16
                expected = expected.at[table[b, pos // 16],
                                       pos % 16].set(jnp.pad(new[b], (0, 64)))
        caches.append(cache.reshape(40, 8, 2, 640))
        expected_caches.append(expected.reshape(40, 8, 2, 640))
        tables.append(table)
    cache = jnp.concatenate(caches)
    table = jnp.asarray(np.concatenate(tables))
    expected_cache = jnp.concatenate(expected_caches)
    references = []
    for b, length in enumerate(lengths[:active]):
        kv = rows[b][:length, :576].astype(jnp.float32).at[-1].set(
            new[b].astype(jnp.float32))
        scores = q[b].astype(jnp.float32) @ kv.T * 192**-.5
        references.append(jax.nn.softmax(scores) @ kv[:, :512])
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
            np.testing.assert_allclose(actual[:active].astype(jnp.float32),
                                       jnp.stack(references),
                                       atol=.005,
                                       rtol=.03)
        np.testing.assert_array_equal(actual[active:], 0)
