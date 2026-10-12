# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch unit tests for DeepSeek-V4.1 FP8 indexer kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.indexer import (
    HEAD_DIM,
    native_paged_scores,
    paged_scores,
    pipeline_score_topk,
)


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax._src.tpu_info as tpu_info_mod

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setitem(tpu_info_mod.registry, "cpu", lambda: info)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _make_valid_pool(
    rng: np.random.Generator, num_pages: int, page_size: int
) -> np.ndarray:
    fp8_keys = jnp.asarray(
        rng.standard_normal((num_pages * page_size, HEAD_DIM), dtype=np.float32),
        jnp.float8_e4m3fn,
    )
    fp8_bytes = jax.lax.bitcast_convert_type(fp8_keys, jnp.uint8).reshape(
        num_pages * page_size // 4, 4, HEAD_DIM
    )
    return np.asarray(
        jax.lax.bitcast_convert_type(jnp.swapaxes(fp8_bytes, 1, 2), jnp.int32)
    )


def _reference_paged_scores(
    q: np.ndarray,
    weights: np.ndarray,
    pool: np.ndarray,
    scales: np.ndarray,
    block_table: np.ndarray,
    requests: np.ndarray,
    lengths: np.ndarray,
    capacity: int,
    page_size: int,
) -> np.ndarray:
    rows, _, _ = q.shape
    num_pages = scales.shape[0]
    key_bytes = np.stack(
        [((pool >> (8 * b)) & 255).astype(np.uint8) for b in range(4)],
        axis=1,
    ).reshape(num_pages * page_size, HEAD_DIM)
    all_keys = np.asarray(
        jax.lax.bitcast_convert_type(jnp.asarray(key_bytes), jnp.float8_e4m3fn),
        dtype=np.float32,
    )
    all_scales = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray(scales.astype(np.uint8)), jnp.float8_e8m0fnu
        ).astype(jnp.float32),
        dtype=np.float32,
    ).reshape(num_pages, page_size)

    out = np.full((rows, capacity), -np.inf, dtype=np.float32)
    q_f32 = q.astype(np.float32)
    w_f32 = weights.astype(np.float32)
    max_pages = block_table.shape[1]
    clipped_lengths = np.clip(lengths, 0, min(capacity, max_pages * page_size))

    for r in range(rows):
        req = int(requests[r])
        L = int(clipped_lengths[r])
        if L <= 0:
            continue
        for pos in range(capacity):
            if pos >= L:
                continue
            lp = pos // page_size
            phys = int(block_table[req, lp])
            if phys < 0 or phys >= num_pages:
                continue
            k_idx = phys * page_size + (pos % page_size)
            sim = q_f32[r] @ all_keys[k_idx]
            val = float(np.sum(np.maximum(sim, 0.0) * w_f32[r]))
            out[r, pos] = val * float(all_scales[phys, pos % page_size])
    return out


def test_paged_scores_validation_and_interpret_bit_exact() -> None:
    rng = np.random.default_rng(101)
    q_bad_rows = jnp.zeros((33, 2, HEAD_DIM), dtype=jnp.float8_e4m3fn)
    w = jnp.ones((2, 2), dtype=jnp.bfloat16)
    pool = jnp.zeros((64, HEAD_DIM), dtype=jnp.int32)
    scales = jnp.full((2, 1, 128), 127, dtype=jnp.int32)
    bt = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)
    reqs = jnp.asarray([0, 0], dtype=jnp.int32)
    lens = jnp.asarray([192, 128], dtype=jnp.int32)

    with pytest.raises(ValueError, match="1..32 rows"):
        paged_scores(
            q_bad_rows,
            w,
            pool,
            scales,
            bt,
            reqs,
            lens,
            256,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
    with pytest.raises(ValueError, match="FP8 E4M3"):
        paged_scores(
            jnp.zeros((2, 2, HEAD_DIM), dtype=jnp.bfloat16),
            w,
            pool,
            scales,
            bt,
            reqs,
            lens,
            256,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
    with pytest.raises(ValueError, match="Positive capacity"):
        paged_scores(
            jnp.zeros((2, 2, HEAD_DIM), dtype=jnp.float8_e4m3fn),
            w,
            pool,
            scales,
            bt,
            reqs,
            lens,
            0,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
    with pytest.raises(ValueError, match="128-key aligned"):
        paged_scores(
            jnp.zeros((2, 2, HEAD_DIM), dtype=jnp.float8_e4m3fn),
            w,
            pool,
            scales,
            bt,
            reqs,
            lens,
            256,
            page_size=130,
            page_stride=128,
            interpret=True,
        )

    q_np = rng.standard_normal((2, 2, HEAD_DIM)).astype(np.float32)
    q_fp8 = jnp.asarray(q_np, dtype=jnp.float8_e4m3fn)
    w_np = rng.uniform(0.1, 1.0, size=(2, 2)).astype(np.float32)
    w_bf16 = jnp.asarray(w_np, dtype=jnp.bfloat16)
    pool_np = _make_valid_pool(rng, num_pages=2, page_size=128)
    scales_np = rng.integers(120, 128, size=(2, 1, 128), dtype=np.int32)

    got = paged_scores(
        q_fp8,
        w_bf16,
        jnp.asarray(pool_np),
        jnp.asarray(scales_np),
        bt,
        reqs,
        lens,
        256,
        page_size=128,
        page_stride=128,
        interpret=True,
    )
    expected = _reference_paged_scores(
        np.asarray(q_fp8),
        np.asarray(w_bf16),
        pool_np,
        scales_np,
        np.asarray(bt),
        np.asarray(reqs),
        np.asarray(lens),
        256,
        128,
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_native_paged_scores_uniform_ragged_and_multi_tile() -> None:
    rng = np.random.default_rng(202)
    num_pages = 34
    page_size = 128
    pool_np = _make_valid_pool(rng, num_pages=num_pages, page_size=page_size)
    scales_np = rng.integers(121, 128, size=(num_pages, 1, 128), dtype=np.int32)
    q_bf16 = jnp.asarray(rng.standard_normal((2, 2, HEAD_DIM)), dtype=jnp.bfloat16)
    w_bf16 = jnp.asarray(rng.uniform(0.1, 1.0, size=(2, 2)), dtype=jnp.bfloat16)
    bt_multi = jnp.asarray(
        [np.arange(32, dtype=np.int32), np.zeros(32, dtype=np.int32)],
        dtype=jnp.int32,
    )

    with pytest.raises(ValueError, match="1..32 BF16 queries"):
        native_paged_scores(
            jnp.zeros((2, 2, HEAD_DIM), dtype=jnp.float32),
            w_bf16,
            jnp.asarray(pool_np),
            jnp.asarray(scales_np),
            bt_multi,
            jnp.asarray([0, 0], dtype=jnp.int32),
            jnp.asarray([256, 256], dtype=jnp.int32),
            256,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
    with pytest.raises(ValueError, match="128-aligned pages"):
        native_paged_scores(
            q_bf16,
            w_bf16,
            jnp.asarray(pool_np),
            jnp.asarray(scales_np),
            bt_multi,
            jnp.asarray([0, 0], dtype=jnp.int32),
            jnp.asarray([256, 256], dtype=jnp.int32),
            256,
            page_size=2560,
            page_stride=128,
            interpret=True,
        )

    # 1. Uniform active req 0 across 2 tiles (`steps == 2` -> `following()`)
    # plus inactive req 1 (`steps == 0` -> `steps > 0` False branch).
    uniform_scores = native_paged_scores(
        q_bf16,
        w_bf16,
        jnp.asarray(pool_np),
        jnp.asarray(scales_np),
        bt_multi,
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([2304, 2176], dtype=jnp.int32),
        4096,
        page_size=128,
        page_stride=128,
        interpret=True,
    )
    assert uniform_scores.shape == (2, 4096)
    assert np.all(np.isfinite(np.asarray(uniform_scores)[0, :2304]))
    assert np.all(np.isneginf(np.asarray(uniform_scores)[0, 2304:]))

    # 2. Ragged batch across reqs 0 & 1 (`uniform == False` -> `ragged_store()`).
    bt_ragged = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    ragged_scores = native_paged_scores(
        q_bf16,
        w_bf16,
        jnp.asarray(pool_np),
        jnp.asarray(scales_np),
        bt_ragged,
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([200, 150], dtype=jnp.int32),
        256,
        page_size=128,
        page_stride=128,
        interpret=True,
    )
    assert ragged_scores.shape == (2, 256)
    assert np.all(np.isfinite(np.asarray(ragged_scores)[0, :200]))
    assert np.all(np.isneginf(np.asarray(ragged_scores)[0, 200:]))
    assert np.all(np.isfinite(np.asarray(ragged_scores)[1, :150]))
    assert np.all(np.isneginf(np.asarray(ragged_scores)[1, 150:]))


def test_pipeline_score_topk_even_and_odd_tiles() -> None:
    def run_pipeline(num_tiles: int) -> np.ndarray:
        out = jnp.zeros((num_tiles,), dtype=jnp.int32)
        res = pipeline_score_topk(
            num_tiles=num_tiles,
            prepare=lambda t: t + 10,
            score=lambda p: p * 2,
            select=lambda t, s: s + t,
            write=lambda t, sel, buf: buf.at[t].set(sel),
            output=out,
        )
        return np.asarray(res)

    even_res = run_pipeline(4)
    np.testing.assert_array_equal(
        even_res, np.array([(t + 10) * 2 + t for t in range(4)], dtype=np.int32)
    )
    odd_res = run_pipeline(3)
    np.testing.assert_array_equal(
        odd_res, np.array([(t + 10) * 2 + t for t in range(3)], dtype=np.int32)
    )
