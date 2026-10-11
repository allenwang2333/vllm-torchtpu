# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for TD7.33: cache_write.py record deduplication and 100% coverage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import cache_write as cw


def test_write_index_cache_empty_and_validation_errors():
    pool = jnp.zeros((64, cw.HEAD_DIM), dtype=jnp.int32)
    scales = jnp.zeros((2, 1, cw.SLAB_KEYS), dtype=jnp.int32)
    empty_keys = jnp.zeros((0, cw.HEAD_DIM), dtype=jnp.bfloat16)
    empty_records = jnp.zeros((0,), dtype=jnp.int32)

    out_pool, out_scales = cw.write_index_cache(
        pool, scales, empty_records, empty_keys, cw.SLAB_KEYS, interpret=True
    )
    assert out_pool is pool
    assert out_scales is scales

    with pytest.raises(ValueError, match="BF16"):
        cw.write_index_cache(
            pool,
            scales,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, 64), dtype=jnp.bfloat16),
            cw.SLAB_KEYS,
            interpret=True,
        )

    with pytest.raises(ValueError, match="BF16"):
        cw.write_index_cache(
            pool,
            scales,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, cw.HEAD_DIM), dtype=jnp.float32),
            cw.SLAB_KEYS,
            interpret=True,
        )

    with pytest.raises(ValueError, match="128-key aligned"):
        cw.write_index_cache(
            pool,
            scales,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, cw.HEAD_DIM), dtype=jnp.bfloat16),
            64,
            interpret=True,
        )


def test_write_index_cache_interpret_preserves_untouched_and_sorts_records():
    rng = np.random.default_rng(42)
    page_stride = 256
    num_pages = 2
    total_keys = num_pages * page_stride
    pool_init = jnp.asarray(
        rng.integers(-1000, 1000, (total_keys // 4, cw.HEAD_DIM), dtype=np.int32)
    )
    scales_init = jnp.full(
        (num_pages, page_stride // cw.SLAB_KEYS, cw.SLAB_KEYS), 77, dtype=jnp.int32
    )

    records = jnp.asarray([130, -1, 5, 1, -3, 129], dtype=jnp.int32)
    keys = jnp.asarray(
        rng.standard_normal((records.shape[0], cw.HEAD_DIM)), dtype=jnp.bfloat16
    )

    out_pool, out_scales = cw.write_index_cache(
        pool_init, scales_init, records, keys, page_stride, interpret=True
    )

    # Untouched slabs (slab 2 and slab 3 on page 1) remain bitwise identical.
    np.testing.assert_array_equal(np.asarray(out_pool[64:]), np.asarray(pool_init[64:]))
    np.testing.assert_array_equal(np.asarray(out_scales[1]), np.asarray(scales_init[1]))
    # Untouched keys inside slab 0 (e.g. key 0, lane 0) keep original byte & scale.
    assert int(out_scales[0, 0, 0]) == 77
    assert int(out_pool[0, 0] & 0xFF) == int(pool_init[0, 0] & 0xFF)

    # Verify quantized bytes and scales for written records (1, 5, 129, 130).
    vals_f32 = keys.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(vals_f32).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scale = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    q_bytes = jax.lax.bitcast_convert_type(
        (vals_f32 / scale[:, None]).astype(jnp.float8_e4m3fn), jnp.uint8
    )
    q_scales = scale_bytes.astype(jnp.int32)
    for src_idx, rec in enumerate(np.asarray(records)):
        if rec < 0:
            continue
        word_row = rec // 4
        lane = rec % 4
        slab = (rec % page_stride) // cw.SLAB_KEYS
        key_in_slab = rec % cw.SLAB_KEYS
        got_byte = (np.asarray(out_pool[word_row]) >> (8 * lane)) & 0xFF
        np.testing.assert_array_equal(got_byte, np.asarray(q_bytes[src_idx]))
        assert int(out_scales[0, slab, key_in_slab]) == int(q_scales[src_idx])


def test_write_cache_words_empty_and_validation_errors():
    cache = jnp.zeros((16, 128), dtype=jnp.int32)
    empty_records = jnp.zeros((0,), dtype=jnp.int32)
    empty_words = jnp.zeros((0, 32), dtype=jnp.int32)

    assert (
        cw.write_cache_words(cache, empty_records, empty_words, interpret=True) is cache
    )

    with pytest.raises(ValueError, match="Sub-row writes"):
        cw.write_cache_words(
            jnp.zeros((16, 64), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, 32), dtype=jnp.int32),
            interpret=True,
        )

    with pytest.raises(ValueError, match="Sub-row writes"):
        cw.write_cache_words(
            cache,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, 48), dtype=jnp.int32),
            interpret=True,
        )


def test_write_cache_words_interpret_preserves_untouched_and_writes_subrows():
    rng = np.random.default_rng(7)
    width = 32
    per_row = 128 // width  # 4 sub-rows per row
    cache_init = jnp.asarray(rng.integers(1, 1000, (8, 128), dtype=np.int32))
    records = jnp.asarray([9, -1, 2, 0, -5, 11], dtype=jnp.int32)
    words = jnp.asarray(
        rng.integers(2000, 5000, (records.shape[0], width), dtype=np.int32)
    )

    out = cw.write_cache_words(cache_init, records, words, interpret=True)

    expected = np.array(cache_init, copy=True)
    for src_idx, rec in enumerate(np.asarray(records)):
        if rec < 0:
            continue
        row = rec // per_row
        sub = rec % per_row
        expected[row, sub * width : (sub + 1) * width] = np.asarray(words[src_idx])

    np.testing.assert_array_equal(np.asarray(out), expected)


def test_td8327_cache_write_jnp_arange_explicit_int32(monkeypatch):
    orig_arange = jnp.arange
    seen_dtypes: list[object] = []

    def _spy_arange(*args, **kwargs):
        seen_dtypes.append(kwargs.get("dtype"))
        return orig_arange(*args, **kwargs)

    monkeypatch.setattr(cw.jnp, "arange", _spy_arange)

    pool_init = jnp.zeros((64, cw.HEAD_DIM), dtype=jnp.int32)
    scales_init = jnp.zeros((2, 1, cw.SLAB_KEYS), dtype=jnp.int32)
    records = jnp.asarray([0, 1], dtype=jnp.int32)
    keys = jnp.ones((2, cw.HEAD_DIM), dtype=jnp.bfloat16)
    cw.write_index_cache(
        pool_init, scales_init, records, keys, cw.SLAB_KEYS, interpret=True
    )

    cache_init = jnp.zeros((4, 128), dtype=jnp.int32)
    words = jnp.ones((2, 32), dtype=jnp.int32)
    cw.write_cache_words(cache_init, records, words, interpret=True)

    assert len(seen_dtypes) >= 2
    assert all(dt == jnp.int32 for dt in seen_dtypes)
