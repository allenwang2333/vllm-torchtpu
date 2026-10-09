# SPDX-License-Identifier: Apache-2.0
"""TD8.64: fused 2-bank full-row KV write (`v2/cache._scatter_rows_2bank`).

The fused kernel must match the per-bank path (one 1-bank kernel per bank)
bitwise, including bank-boundary, padding (-1) and out-of-range records.
Runs the TPU Pallas kernel in interpret mode on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache

ROW = v2_cache.ROW_WORDS


def _per_bank(lo, hi, records, words):
    bank = lo.shape[0] * ROW // words.shape[1]
    r_lo = jnp.where(records < bank, records, -1)
    r_hi = jnp.where(records >= bank, records - bank, -1)
    return (
        v2_cache.scatter_cache_values(lo, r_lo, words),
        v2_cache.scatter_cache_values(hi, r_hi, words),
    )


def _inputs(rows_lo, rows_hi, n, chunks, seed):
    rng = np.random.default_rng(seed)

    def rand(shape):
        return jnp.asarray(
            rng.integers(-(2**31), 2**31 - 1, shape, np.int64), jnp.int32
        )

    total = (rows_lo + rows_hi) // chunks
    rec = rng.choice(total, size=n, replace=False).astype(np.int32)
    rec[rng.random(n) < 0.1] = -1
    rec[0] = rows_lo // chunks - 1  # last lo record
    rec[1] = rows_lo // chunks  # first hi record
    rec[2] = total + 5  # out of range: dropped
    return (
        rand((rows_lo, ROW)),
        rand((rows_hi, ROW)),
        jnp.asarray(rec),
        rand((n, chunks * ROW)),
    )


@pytest.fixture
def tpu_kernel_path(monkeypatch):
    # Take the TPU Pallas branch and run it in the TPU interpreter on CPU.
    monkeypatch.setattr(v2_cache.jax, "default_backend", lambda: "tpu")
    with pltpu.force_tpu_interpret_mode():
        yield


@pytest.mark.parametrize(
    "rows_lo,rows_hi,n,chunks",
    [(256, 256, 70, 1), (256, 512, 130, 2), (128, 64, 40, 4)],
)
def test_fused_2bank_matches_per_bank(tpu_kernel_path, rows_lo, rows_hi, n, chunks):
    lo, hi, records, words = _inputs(rows_lo, rows_hi, n, chunks, n)
    want = _per_bank(lo, hi, records, words)
    got = v2_cache._scatter_rows_2bank(lo, hi, records, words)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_scatter_cache_values_tuple_dispatch(tpu_kernel_path, monkeypatch):
    lo, hi, records, words = _inputs(256, 256, 70, 1, 3)
    calls = []
    real = v2_cache._scatter_rows_2bank
    monkeypatch.setattr(
        v2_cache,
        "_scatter_rows_2bank",
        lambda *a: calls.append(1) or real(*a),
    )
    got = v2_cache.scatter_cache_values((lo, hi), records, words)
    assert calls == [1]
    want = _per_bank(lo, hi, records, words)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_scatter_cache_values_tuple_sub_row_keeps_per_bank(monkeypatch):
    # Sub-row widths (< ROW_WORDS) still go through the per-bank path.
    monkeypatch.setattr(
        v2_cache,
        "_scatter_rows_2bank",
        lambda *a: pytest.fail("fused path must not handle sub-row writes"),
    )
    lo = jnp.zeros((4, ROW), jnp.int32)
    hi = jnp.zeros((4, ROW), jnp.int32)
    words = jnp.ones((2, ROW // 2), jnp.int32)
    out_lo, out_hi = v2_cache.scatter_cache_values(
        (lo, hi), jnp.asarray([1, 9], jnp.int32), words
    )
    assert int(out_lo[0, ROW // 2]) == 1 and int(out_hi[0, ROW // 2]) == 1
    assert jax.default_backend() == "cpu"


def test_scatter_cache_values_single_bank_full_row_cpu():
    cache = jnp.zeros((8, ROW), jnp.int32)
    records = jnp.asarray([0, 2, -1], jnp.int32)
    words = jnp.ones((3, ROW * 2), jnp.int32)
    out = v2_cache.scatter_cache_values(cache, records, words)
    assert out.shape == cache.shape
    np.testing.assert_array_equal(np.asarray(out[0]), 1)
    np.testing.assert_array_equal(np.asarray(out[1]), 1)
    np.testing.assert_array_equal(np.asarray(out[4]), 1)
    np.testing.assert_array_equal(np.asarray(out[5]), 1)
    np.testing.assert_array_equal(np.asarray(out[2]), 0)


def test_scatter_cache_values_sub_row_tpu_merge(monkeypatch):
    cache = jnp.zeros((8, ROW), jnp.int32)
    records = jnp.asarray([0, 1], jnp.int32)
    words = jnp.ones((2, ROW // 2), jnp.int32)
    monkeypatch.setattr(v2_cache.jax, "default_backend", lambda: "tpu")
    calls = []
    real = v2_cache.merge_sub_rows
    monkeypatch.setattr(
        v2_cache, "merge_sub_rows", lambda c, r, w: calls.append(1) or real(c, r, w)
    )
    with pltpu.force_tpu_interpret_mode():
        out = v2_cache.scatter_cache_values(cache, records, words)
    assert calls == [1]
    assert out.shape == cache.shape
