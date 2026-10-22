# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate page coverage, logical token identity and sparse DMA accounting."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_sparse import (
    candidate_pages,
    mask_candidate_scores,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "following, nonempty", [(False, False), (True, False), (True, True)]
)
def test_empty_tile_keeps_prefetch_and_output_lifetimes(following, nonempty):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
        _packed_fp8_pipeline as pipeline,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
        PACKED_FP8_MQA_CONFIG,
    )

    calls = []
    plan = pipeline.BufferPlan(
        PACKED_FP8_MQA_CONFIG, 32, 768, 256, jnp.dtype(jnp.float8_e4m3fn)
    )
    ops = SimpleNamespace(
        state=pipeline.QueryState(pipeline.QueryTile(0, 6, 0), 0, 0),
        live_tiles=3,
        next_tiles=3 if following else 0,
        next_group=1,
        tile_flags=jnp.array([[0, 0, 0], [128 if nonempty else 0, 0, 0]], jnp.int32),
        metadata_slot=0,
        skip_invalid_pages=True,
        key_slot=lambda tile: tile & 3,
        wait_keys=lambda tile: calls.append(("wait", tile)),
        fetch=lambda *args, **kwargs: calls.append(("fetch", *map(int, args), kwargs)),
        clear_output=lambda tile: calls.append(("clear", tile)),
        publish=lambda tile, **kwargs: calls.append(("publish", tile, kwargs)),
    )
    with jax.disable_jit():
        pipeline.Pipeline(plan).step(ops, jnp.int32(0))
    expected = [("wait", 0)]
    if following and nonempty:
        expected.append(("fetch", 0, 1, 3, {"unroll": True}))
    expected.extend((("clear", 0), ("publish", 0, {})))
    assert calls == expected


@pytest.mark.parametrize("block_size", [8, 64, 96, 128, 192, 256])
def test_sparse_tasks_match_token_level_candidate_reference(block_size):
    rng = np.random.default_rng(911)
    rows, width, cache_pages = 17, 128 * 11 - 19, 32
    requests = np.array([0] * 7 + [1] + [2] * 9, np.int32)
    table = rng.integers(0, cache_pages, (3, 11), dtype=np.int32)
    table[0, 2], table[1, 3], table[2, 5] = -1, cache_pages, -1
    lengths = np.minimum(np.arange(rows, dtype=np.int32) * 113, width)
    lengths[3] = 0
    candidates = rng.integers(
        -2, (width + block_size - 1) // block_size + 3, (rows, 4), dtype=np.int32
    )
    candidates[0] = -1
    candidates[1] = [0, 0, 1, np.iinfo(np.int32).max]
    candidates[2, 0] = 0
    candidates[-1] = -1
    tasks = candidate_pages(
        jnp.asarray(candidates),
        jnp.asarray(lengths),
        jnp.asarray(requests),
        jnp.asarray(table),
        block_size=block_size,
        cache_pages=cache_pages,
    )
    physical, offsets, task_lengths, block_mask = map(np.asarray, tasks[:4])
    allowed = np.zeros((rows, width), bool)
    for row in range(rows):
        for candidate in candidates[row]:
            if candidate < 0 or int(candidate) * block_size >= lengths[row]:
                continue
            for token in range(
                int(candidate) * block_size,
                min((int(candidate) + 1) * block_size, lengths[row]),
            ):
                if 0 <= table[requests[row], token // 128] < cache_pages:
                    allowed[row, token] = True
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        assert 0 <= end - start <= 6
        expected = np.flatnonzero(
            np.pad(allowed[start:end].any(axis=0), (0, 11 * 128 - width))
            .reshape(11, 128)
            .any(axis=1)
        )
        actual = np.flatnonzero(physical[group] >= 0)
        np.testing.assert_array_equal(actual, expected)
        if start < end:
            assert np.all(requests[start:end] == requests[start])
            np.testing.assert_array_equal(
                physical[group, actual], table[requests[start], actual]
            )
        assert np.all(physical[group, 11:] == -1)
    assert np.all((task_lengths == 0) | (task_lengths == lengths))

    # Model the scorer's page-hole mask; tied scores must retain logical ID order.
    scores = rng.integers(-3, 4, (rows, physical.shape[1] * 128)).astype(np.float32)
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        scores[start:end, np.repeat(physical[group] < 0, 128)] = -np.inf
    masked = mask_candidate_scores(
        jnp.asarray(scores),
        tasks.block_mask,
        jnp.asarray(lengths),
        block_size=block_size,
        width=width,
    )
    values, indices = jax.lax.top_k(masked, 5)
    actual = jnp.where(jnp.isfinite(values), indices, -1)
    expected = np.full((rows, 5), -1, np.int32)
    for row in range(rows):
        chosen = sorted(
            np.flatnonzero(allowed[row]), key=lambda i: (-scores[row, i], i)
        )[:5]
        expected[row, : len(chosen)] = chosen
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("candidate_count", [0, 4])
def test_sparse_tasks_empty_candidates_produce_no_live_tiles(candidate_count):
    tasks = candidate_pages(
        jnp.full((6, candidate_count), -1, jnp.int32),
        jnp.full((6,), 1024, jnp.int32),
        jnp.zeros((6,), jnp.int32),
        jnp.arange(8, dtype=jnp.int32)[None, :],
        block_size=8,
        cache_pages=8,
    )
    assert np.all(np.asarray(tasks.physical) == -1)
    assert np.all(np.asarray(tasks.lengths) == 0)


@pytest.mark.parametrize("banks", [1, 2])
def test_sparse_dma_skips_padding_and_waits_only_for_issued_bytes(monkeypatch, banks):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
        _packed_fp8_pipeline as pipeline,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_mqa
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
        PACKED_FP8_MQA_CONFIG,
    )

    class Ref:
        def __init__(self, name, shape, path=()):
            self.name, self.shape, self.path = name, shape, path

        @property
        def at(self):
            return self

        def __getitem__(self, index):
            index = index if isinstance(index, tuple) else (index,)
            shape = tuple(int(i.size) for i in index if hasattr(i, "size"))
            shape += self.shape[len(index) :]
            return Ref(self.name, shape, (*self.path, index))

        def __setitem__(self, index, value):
            raise AssertionError("K buffers must not receive initialization stores")

    started, waited = [], []

    def copy(src, dst, sem):
        return SimpleNamespace(
            start=lambda: started.append((src, dst, sem)),
            wait=lambda: waited.append((src, dst, sem)),
        )

    monkeypatch.setattr(packed_fp8_mqa.pltpu, "make_async_copy", copy)
    ops = object.__new__(packed_fp8_mqa.MQAOps)
    ops.plan = pipeline.BufferPlan(
        PACKED_FP8_MQA_CONFIG, 32, 256, 256, jnp.dtype(jnp.float8_e4m3fn)
    )
    ops.state = pipeline.QueryState(pipeline.QueryTile(0, 6, 0), 0, 0)
    ops.skip_invalid_pages = True
    # Three valid page tasks, across both banks; padding retains arbitrary
    # addresses and bank bits so only the validity predicate can exclude it.
    ops.pages = jnp.array(
        [(5 << 2) | 2, (45 << 2) | (2 + banks - 1), (10 << 2) | 2]
        + [(67 << 2) | (page % 2) for page in range(253)],
        jnp.int32,
    )
    ops.pool = (Ref("lo", (80, 128)), Ref("hi", (80, 128)))[:banks]
    ops.keys = Ref("keys", (4, 256 * 33, 128))
    ops.read_sem = Ref("sem", (4,))
    ops.tile_flags = jnp.array([[3]], jnp.int32)
    with jax.disable_jit():
        for page in range(256):
            ops.load_page(0, page, 0, 0)
        ops.wait_keys(0)
    assert [src.name for src, _, _ in started] == [
        "lo",
        "hi" if banks == 2 else "lo",
        "lo",
    ]
    assert len(waited) == 1
    assert all(dst.shape == (33, 128) for _, dst, _ in started)
    assert waited[0][1].shape == (3 * 33, 128)
    assert sum(np.prod(dst.shape) * 4 for _, dst, _ in started) == (
        np.prod(waited[0][1].shape) * 4
    )


@pytest.mark.parametrize("fragment", [0, 1, 127])
@pytest.mark.parametrize("poison", ["nan", "inf_scale", "random_bits"])
def test_unloaded_page_bits_cannot_change_valid_scores(fragment, poison, monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
        _packed_fp8_pipeline as pipeline,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_mqa
    from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_math import (
        head_order,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        pack_fp8_pages,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
        PACKED_FP8_MQA_CONFIG,
    )

    # CPU lacks lowering for TPU pack/unpack/roll. Model their elementwise
    # bit semantics; retain the production QK, reduction order and output mask.
    def pack(xs, *, packed_dtype):
        assert packed_dtype == jnp.bfloat16 and len(xs) == 2
        bits = [
            jax.lax.bitcast_convert_type(x.astype(jnp.bfloat16), jnp.uint16).astype(
                jnp.uint32
            )
            for x in xs
        ]
        return bits[0] | (bits[1] << 16)

    def unpack(x, *, index, packed_dtype, unpacked_dtype):
        assert packed_dtype == jnp.bfloat16 and unpacked_dtype == jnp.float32
        bits = ((x >> (16 * index)) & 65535) << 16
        return jax.lax.bitcast_convert_type(bits, jnp.float32)

    monkeypatch.setattr(packed_fp8_mqa.pltpu, "pack_elementwise", pack)
    monkeypatch.setattr(packed_fp8_mqa.pltpu, "unpack_elementwise", unpack)
    monkeypatch.setattr(packed_fp8_mqa.pltpu, "roll", jnp.roll)

    class Keys:
        def __init__(self, words):
            self.words = words

        def __getitem__(self, index):
            slot, rows = index
            assert slot == 0
            start = rows.start - fragment * 66
            return self.words[start : start + rows.size]

    class Scores:
        def __init__(self):
            self.parts = []

        def __setitem__(self, index, value):
            self.parts.append(value)

    rng = np.random.default_rng(803)
    q = jnp.asarray(rng.normal(size=(192, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.1, 0.5, (192, 1)), jnp.bfloat16)
    keys = jnp.asarray(rng.normal(size=(256, 128)), jnp.float8_e4m3fn)
    packed = np.asarray(pack_fp8_pages(keys, jnp.ones((256,), jnp.float32)))
    poisoned = rng.integers(0, 1 << 32, packed.shape, dtype=np.uint32).view(np.int32)
    if poison == "nan":
        poisoned[:] = 0x7F7F7F7F  # All FP8 bytes are NaN.
        poisoned[[32, 65]] = 0x7FC00000
    elif poison == "inf_scale":
        poisoned[:] = 0x7E7E7E7E  # Finite FP8 max, with infinite FP32 K scales.
        poisoned[[32, 65]] = 0x7F800000
    base = fragment * 256
    lengths = base + jnp.array([256, 243, 117, 1, 0, 128], jnp.int32)

    @jax.jit
    def score(words, valid_pages):
        ops = object.__new__(packed_fp8_mqa.MQAOps)
        ops.plan = pipeline.BufferPlan(
            PACKED_FP8_MQA_CONFIG, 32, 256, 256, jnp.dtype(jnp.float8_e4m3fn)
        )
        ops.state = pipeline.QueryState(pipeline.QueryTile(0, 6, 0), 0, 0)
        ops.skip_invalid_pages = True
        ops.q = SimpleNamespace(dtype=q.dtype)
        ops.lhs = head_order(q)
        ops.reduction_weights = head_order(weights)
        rows = jnp.arange(192)
        owners = jnp.where(rows < 128, (rows // 2) % 4, 4 + ((rows - 128) // 2) % 2)
        ops.reduction_selector = (jnp.arange(6)[:, None] == owners).astype(jnp.bfloat16)
        ops.keys, ops.scores = Keys(words), Scores()
        ops.pages = (
            jnp.zeros((256,), jnp.int32)
            .at[fragment * 2 : fragment * 2 + 2]
            .set(valid_pages.astype(jnp.int32) << 1)
        )
        ops.row_lengths = lengths
        ops.prepare_output_mask(0)
        logits = ops.qk(0, fragment)
        values = ops.reduce(fragment, logits)
        ops.store(0, fragment, values)
        return jnp.concatenate(ops.scores.parts, axis=1)

    for valid_pages in ((True, False), (False, True), (False, False)):
        valid_pages = np.asarray(valid_pages)
        row_mask = np.repeat(valid_pages, 33)[:, None]
        clean = np.where(row_mask, packed, 0)
        dirty = np.where(row_mask, packed, poisoned)
        expected = np.asarray(score(clean, valid_pages))
        actual = np.asarray(score(dirty, valid_pages))
        allowed = np.arange(256)[None, :] + base < np.asarray(lengths)[:, None]
        allowed &= np.repeat(valid_pages, 128)[None, :]
        assert np.all(np.isfinite(expected[allowed]))
        assert np.all(np.isneginf(actual[~allowed]))
        np.testing.assert_array_equal(actual, expected)
