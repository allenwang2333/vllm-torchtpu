# SPDX-License-Identifier: Apache-2.0
"""Dense MLA DCP: striped writes, causal masking and remote DMA."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/kernels/mla/dcp.py"
_SPEC = importlib.util.spec_from_file_location("mla_dcp", _PATH)
kernel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernel)


@pytest.mark.parametrize(
    "tile_pages,group_size,active",
    [
        (64, 8, 0),
        (64, 8, 25),
        (64, 8, 32),
        (128, 4, 25),
    ],
)
def test_decode_attention(tile_pages, group_size, active):
    # Four independent CP pairs; full/partial groups and repeated bank reuse.
    devices, bucket, query_len = 8, 32, 1
    if jax.default_backend() != "tpu" or jax.device_count() < devices:
        pytest.skip(f"Requires {devices} TPU devices for remote DMA")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:devices]), ("dcp",))
    rng = np.random.default_rng(817)
    pattern = [1, 16, 17, 129, 257, 1025, 1152, 4097]
    lengths = np.array(
        [pattern[i % len(pattern)] if i < active else 0 for i in range(bucket)],
        np.int32,
    )
    starts = jnp.minimum(jnp.arange(bucket + 1, dtype=jnp.int32), active) * query_len
    tokens = bucket * query_len
    q = jnp.asarray(rng.normal(size=(tokens, devices * 12, 576)), jnp.bfloat16)
    new = jnp.asarray(rng.normal(size=(tokens, 576)), jnp.bfloat16)
    rows = [
        jnp.asarray(rng.normal(size=(max(int(n), 1), 640)), jnp.bfloat16)
        for n in lengths
    ]
    caches, expected_caches, tables = [], [], []
    pages_per_request = (int(lengths.max()) + 31) // 32
    pool_pages = bucket * pages_per_request + 1
    for rank in range(devices):
        cache = jnp.full((pool_pages, 16, 640), jnp.nan, jnp.bfloat16)
        table = np.zeros((bucket, max(pages_per_request, 1)), np.int32)
        physical_pages = rng.permutation(pool_pages - 1) + 1
        next_page = 0
        for b, length in enumerate(lengths):
            local = 0
            for token in range(length):
                if token // 16 % 2 != rank % 2:
                    continue
                if local % 16 == 0:
                    table[b, local // 16] = physical_pages[next_page]
                    next_page += 1
                cache = cache.at[table[b, local // 16], local % 16].set(rows[b][token])
                local += 1
        expected = cache
        for b, length in enumerate(lengths[:active]):
            for offset in range(query_len):
                position = length - query_len + offset
                if position // 16 % 2 == rank % 2:
                    pos = position // 32 * 16 + position % 16
                    expected = expected.at[table[b, pos // 16], pos % 16].set(
                        jnp.pad(new[b * query_len + offset], (0, 64))
                    )
        caches.append(cache.reshape(pool_pages, 8, 2, 640))
        expected_caches.append(expected.reshape(pool_pages, 8, 2, 640))
        tables.append(table)
    cache = jnp.concatenate(caches)
    table = jnp.asarray(np.concatenate(tables))
    expected_cache = jnp.concatenate(expected_caches)
    references = []
    for b, length in enumerate(lengths[:active]):
        kv = (
            rows[b][:length, :576]
            .astype(jnp.float32)
            .at[-query_len:]
            .set(new[b * query_len : (b + 1) * query_len].astype(jnp.float32))
        )
        for offset in range(query_len):
            causal = kv[: length - query_len + offset + 1]
            scores = q[b * query_len + offset].astype(jnp.float32) @ causal.T
            references.append(
                jnp.matmul(
                    jax.nn.softmax(scores * 192**-0.5),
                    causal[:, :512],
                    precision=jax.lax.Precision.HIGHEST,
                )
            )
    fn = jax.jit(
        lambda c: kernel.decode_attention(
            c,
            q[:, :, :512],
            q[:, :, 512:],
            new[:, :512],
            new[:, 512:],
            jnp.asarray(lengths),
            table,
            starts,
            mesh=mesh,
            scale=192**-0.5,
            group_size=group_size,
            tile_pages=tile_pages,
        )
    )
    for _ in range(3):
        updated, actual = jax.block_until_ready(fn(cache))
        np.testing.assert_allclose(
            updated.astype(jnp.float32),
            expected_cache.astype(jnp.float32),
            atol=0,
            rtol=0,
            equal_nan=True,
        )
        if active:
            np.testing.assert_allclose(
                actual[: active * query_len].astype(jnp.float32),
                jnp.stack(references),
                atol=0.005,
                rtol=0.03,
            )
        np.testing.assert_array_equal(actual[active * query_len :], 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_decode_stripe_boundaries(rank):
    lengths = jnp.array([16, 17, 32, 33, 0], jnp.int32)
    starts = jnp.array([0, 1, 2, 3, 4, 4], jnp.int32)
    tables = jnp.array([[1, 5], [2, 6], [3, 7], [4, 8], [0, 0]])
    cache = jnp.full((9, 8, 2, 640), -7, jnp.bfloat16)
    q = jnp.zeros((8, 12, 512), jnp.bfloat16)
    qp = jnp.zeros((8, 12, 64), jnp.bfloat16)
    values = jnp.broadcast_to(jnp.arange(8)[:, None], (8, 576))
    updated, _, causal, _, count = kernel._prepare_decode(
        cache,
        q,
        qp,
        values[:, :512],
        values[:, 512:],
        lengths,
        tables.reshape(-1),
        starts,
        rank=rank,
    )
    expected = np.asarray(cache).copy()
    for token, position in enumerate([15, 16, 31, 32]):
        if position // 16 % 2 == rank:
            local = position // 32 * 16 + position % 16
            expected[tables[token, local // 16], local % 16 // 2, local % 2, :576] = (
                token
            )
            expected[tables[token, local // 16], local % 16 // 2, local % 2, 576:] = 0
    np.testing.assert_array_equal(updated, expected)
    np.testing.assert_array_equal(
        causal,
        [16, 16, 16, 17, 0, 0, 0, 0] if rank == 0 else [0, 1, 16, 16, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(count, [4])
