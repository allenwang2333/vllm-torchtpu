# SPDX-License-Identifier: Apache-2.0
"""100% statement & branch coverage and bit-exactness suite for cache_write.py."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vllm_torchtpu.kernels.deepseek_v41 import cache_write as cw  # noqa: E402


def _ref_quantize_np(keys_bf16: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    vals_f32 = keys_bf16.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(vals_f32).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scale = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    q_bytes = jax.lax.bitcast_convert_type(
        (vals_f32 / scale[:, None]).astype(jnp.float8_e4m3fn), jnp.uint8
    )
    return np.asarray(q_bytes), np.asarray(scale_bytes, dtype=np.int32)


def test_quantize_bit_exact_across_dynamic_ranges(monkeypatch):
    monkeypatch.setattr(cw.pltpu, "bitcast", jax.lax.bitcast_convert_type)
    rng = np.random.default_rng(123)
    raw = rng.standard_normal((16, cw.HEAD_DIM)).astype(np.float32)
    raw[0] = 0.0
    raw[1] *= 1e-5
    raw[2] *= 1e3
    raw[3, ::2] *= -250.0
    keys = jnp.asarray(raw, dtype=jnp.bfloat16)

    q_opt, s_opt = cw._quantize(keys)
    q_ref, s_ref = _ref_quantize_np(keys)
    np.testing.assert_array_equal(np.asarray(q_opt), q_ref)
    np.testing.assert_array_equal(np.asarray(s_opt), s_ref)


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


def test_write_index_cache_bit_exact_all_lanes_and_untouched_slabs():
    rng = np.random.default_rng(42)
    page_stride = 256
    num_pages = 2
    total_keys = num_pages * page_stride
    pool_init = jnp.asarray(
        rng.integers(
            -1_000_000_000,
            1_000_000_000,
            (total_keys // 4, cw.HEAD_DIM),
            dtype=np.int32,
        )
    )
    scales_init = jnp.full(
        (num_pages, page_stride // cw.SLAB_KEYS, cw.SLAB_KEYS), 77, dtype=jnp.int32
    )

    # Exercises all 4 lanes (0, 1, 2, 3), multiple slabs, and negative sentinels
    records = jnp.asarray([130, -1, 5, 0, 1, 2, 3, -3, 129, 255], dtype=jnp.int32)
    keys = jnp.asarray(
        rng.standard_normal((records.shape[0], cw.HEAD_DIM)) * 8.0, dtype=jnp.bfloat16
    )

    out_pool, out_scales = cw.write_index_cache(
        pool_init, scales_init, records, keys, page_stride, interpret=True
    )

    # Page 1 (slabs 2 & 3) is completely untouched
    np.testing.assert_array_equal(np.asarray(out_pool[64:]), np.asarray(pool_init[64:]))
    np.testing.assert_array_equal(np.asarray(out_scales[1]), np.asarray(scales_init[1]))

    # Untouched lanes inside word row 1 (rec=5 is lane 1; lanes 0, 2, 3 untouched)
    for untouched_lane in (0, 2, 3):
        got_u = (np.asarray(out_pool[1]) >> (8 * untouched_lane)) & 0xFF
        ref_u = (np.asarray(pool_init[1]) >> (8 * untouched_lane)) & 0xFF
        np.testing.assert_array_equal(got_u, ref_u)

    q_bytes, q_scales = _ref_quantize_np(keys)
    for src_idx, rec in enumerate(np.asarray(records)):
        if rec < 0:
            continue
        word_row = rec // 4
        lane = rec % 4
        slab = (rec % page_stride) // cw.SLAB_KEYS
        key_in_slab = rec % cw.SLAB_KEYS
        got_byte = (np.asarray(out_pool[word_row]) >> (8 * lane)) & 0xFF
        np.testing.assert_array_equal(got_byte, q_bytes[src_idx])
        assert int(out_scales[0, slab, key_in_slab]) == int(q_scales[src_idx])


@pytest.mark.parametrize("width", [4, 8, 16, 32, 64])
def test_write_cache_words_bit_exact_across_widths_and_errors(width: int):
    cache = jnp.zeros((8, 128), dtype=jnp.int32)
    empty_records = jnp.zeros((0,), dtype=jnp.int32)
    empty_words = jnp.zeros((0, width), dtype=jnp.int32)

    assert (
        cw.write_cache_words(cache, empty_records, empty_words, interpret=True) is cache
    )

    with pytest.raises(ValueError, match="Sub-row writes"):
        cw.write_cache_words(
            jnp.zeros((8, 64), dtype=jnp.int32),
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

    rng = np.random.default_rng(100 + width)
    per_row = 128 // width
    cache_init = jnp.asarray(rng.integers(-500_000, 500_000, (8, 128), dtype=np.int32))
    records = jnp.asarray(
        [per_row * 3 + 1, -1, 1, 0, -5, per_row * 3 + (per_row - 1)], dtype=jnp.int32
    )
    words = jnp.asarray(
        rng.integers(-900_000, 900_000, (records.shape[0], width), dtype=np.int32)
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
