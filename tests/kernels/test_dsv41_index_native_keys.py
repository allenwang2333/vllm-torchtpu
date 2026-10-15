# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native (bitcast-ready) index-key layout: writer and bitwise reader parity.

Each test builds the legacy and the native pool from the same BF16 keys and
checks that every converted reader returns bitwise-identical scores. Runs in
Pallas interpret mode on CPU (`pltpu.bitcast` lowers to its XLA reference).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout, index_word_scores
from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate
from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode import (
    grouped_decode_scores,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.scores import (
    prefill_index_scores,
)


def _legacy_scatter(cache, records, keys):
    """The legacy sub-row writer (`cache.scatter_cache_values`, CPU path)."""
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32)
    columns = jnp.arange(64)[None]
    rows = jnp.where(records[:, None] >= 0, records[:, None] // 2, cache.shape[0])
    lanes = (records[:, None] % 2) * 64 + columns
    return cache.at[rows, lanes].set(words, mode="drop")


def _pools(rng, records_total, batches=3):
    """Same keys written in shuffled batches (pairs split and joined) both ways."""
    keys = jnp.asarray(rng.normal(size=(records_total, 128)), jnp.bfloat16)
    garbage = jnp.asarray(
        rng.integers(-(2**31), 2**31 - 1, size=(records_total // 2, 128)), jnp.int32
    )
    legacy, native = garbage, index_key_layout.legacy_to_native_rows(garbage)
    order = rng.permutation(records_total)
    for chunk in np.array_split(order, batches):
        # Invalid (-1) and out-of-range writes must be dropped.
        records = np.concatenate([chunk, [-1, records_total + 3]]).astype(np.int32)
        values = jnp.concatenate(
            [keys[chunk], jnp.asarray(rng.normal(size=(2, 128)), jnp.bfloat16)]
        )
        legacy = _legacy_scatter(legacy, jnp.asarray(records), values)
        native = index_key_layout.scatter_native_keys(
            native, jnp.asarray(records), values
        )
    return keys, legacy, native


@pytest.mark.cpu_test
def test_native_writer_matches_legacy_layout():
    rng = np.random.default_rng(3)
    keys, legacy, native = _pools(rng, 512)
    np.testing.assert_array_equal(
        index_key_layout.native_to_legacy_rows(native), legacy
    )
    np.testing.assert_array_equal(
        index_key_layout.legacy_to_native_rows(legacy), native
    )
    split = jnp.concatenate([keys[:, ::2], keys[:, 1::2]], axis=1)
    expected = jax.lax.bitcast_convert_type(split, jnp.uint16)
    for actual in (
        index_key_layout.native_rows_to_keys(native),
        jax.jit(lambda x: pltpu.bitcast(x, jnp.bfloat16))(native),
    ):
        np.testing.assert_array_equal(
            jax.lax.bitcast_convert_type(actual, jnp.uint16), expected
        )


def _bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.int32))


@pytest.mark.cpu_test
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("depth", [2, 4])
def test_native_decode_scores_bitwise(banks, depth):
    rng = np.random.default_rng(11)
    requests, queries, heads, width, page = 3, 6, 4, 256, 16
    _, legacy, native = _pools(rng, requests * width)
    order = rng.permutation(requests * width // page).reshape(requests, -1)
    pages = jnp.asarray(order.astype(np.int32) * page)
    pages = pages.at[0, 3].set(-1)
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(requests, queries, heads, 1)), jnp.float32)
    lengths = jnp.array(
        [[256, 255, 128, 73, 0, 5], [0] * 6, [3, 64, 129, 230, 254, 256]], jnp.int32
    )
    out = []
    for pool, native_keys in ((legacy, False), (native, True)):
        pool = (pool[:176], pool[176:]) if banks == 2 else pool
        out.append(
            grouped_decode_scores(
                q,
                w,
                pool,
                pages,
                lengths,
                page_records=page,
                key_tile=64,
                interpret=True,
                depth=depth,
                native_keys=native_keys,
            )
        )
    np.testing.assert_array_equal(_bits(out[0]), _bits(out[1]))
    assert np.isfinite(np.asarray(out[0])).any()


@pytest.mark.cpu_test
@pytest.mark.parametrize("banks", [1, 2])
def test_native_prefill_scores_bitwise(banks):
    rng = np.random.default_rng(12)
    tokens, heads, width, page = 16, 4, 256, 16
    _, legacy, native = _pools(rng, width)
    order = rng.permutation(width // page).astype(np.int32)
    pages = jnp.asarray((order * page)[None]).at[0, -1].set(-1)
    q = jnp.asarray(rng.normal(size=(tokens, heads, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(tokens, heads, 1)), jnp.float32)
    out = []
    for pool, native_keys in ((legacy, False), (native, True)):
        pool = (pool[:48], pool[48:]) if banks == 2 else pool
        out.append(
            prefill_index_scores(
                q,
                w,
                pool,
                pages,
                jnp.array([200], jnp.int32),
                page_records=page,
                query_tile=8,
                key_tile=64,
                interpret=True,
                native_keys=native_keys,
            )
        )
    np.testing.assert_array_equal(_bits(out[0]), _bits(out[1]))


@pytest.mark.cpu_test
def test_native_word_rows_bitwise():
    rng = np.random.default_rng(13)
    groups, rows = 2, 64
    _, legacy, native = _pools(rng, groups * rows * 2)
    q = jnp.asarray(rng.normal(size=(groups * 3, 4, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(groups * 3, 4, 1)), jnp.float32)
    out = [
        index_word_scores.score_word_rows(
            q,
            w,
            pool.reshape(groups, rows, 128),
            interpret=True,
            native_keys=native_keys,
        )
        for pool, native_keys in ((legacy, False), (native, True))
    ]
    np.testing.assert_array_equal(_bits(out[0]), _bits(out[1]))


@pytest.mark.cpu_test
def test_native_candidate_scores_bitwise():
    rng = np.random.default_rng(14)
    groups, queries, heads, block, blocks = 2, 3, 4, 8, 16
    width = blocks * block
    _, legacy, native = _pools(rng, groups * width)
    q = jnp.asarray(rng.normal(size=(groups, queries, heads, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(groups, queries, heads)), jnp.float32)
    ids = jnp.asarray(rng.permutation(64)[: groups * blocks].reshape(groups, blocks))
    ids = ids.astype(jnp.int32).at[1, -1].set(-1)
    membership = jnp.asarray(
        rng.integers(0, 1 << queries, size=(groups, blocks)), jnp.int32
    )
    lengths = jnp.array([[512, 100, 7], [300, 0, 512]], jnp.int32)
    results = []
    for pool, native_keys in ((legacy, False), (native, True)):
        words = pool.reshape(groups, width // 2, 128)
        unpacked = candidate.unpack_candidate_words(
            words, interpret=True, native_keys=native_keys
        )
        results.append(_bits(unpacked.astype(jnp.float32)))
        for keys in (words, unpacked):
            results.append(
                _bits(
                    candidate.candidate_scores(
                        q,
                        w,
                        keys,
                        ids,
                        membership,
                        lengths,
                        block_size=block,
                        key_tile=64,
                        interpret=True,
                        native_keys=native_keys,
                    )
                )
            )
    legacy_results, native_results = results[:3], results[3:]
    for a, b in zip(legacy_results, native_results, strict=True):
        np.testing.assert_array_equal(a, b)
    # Packed and pre-unpacked scoring agree too.
    np.testing.assert_array_equal(native_results[1], native_results[2])
