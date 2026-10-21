# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.paired_mqa_metadata import (
    _merge_partition,
    paired_mqa_metadata,
)


@pytest.mark.parametrize("capacity", [128, 256, 384])
@pytest.mark.parametrize("overlap", [0, 0.5, 1])
def test_paired_metadata(capacity, overlap):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires native SparseCore metadata")
    counts = [0, 1, 0, 3, 2, 0, 5, 0]
    offsets = np.cumsum([0, *counts], dtype=np.int32)
    tokens, width = offsets[-1], 4 * capacity
    rng = np.random.default_rng(61)
    pages = rng.permutation(len(counts) * width).reshape(len(counts), width)
    candidates = np.full((tokens, capacity), -1, np.int32)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        for query in range(start, end, 2):
            size = int(rng.integers(capacity + 1))
            a = np.sort(rng.choice(width, size, replace=False))
            candidates[query, :size] = a
            if query + 1 < end:
                shared = a[: int(size * overlap)]
                other = rng.choice(
                    np.setdiff1d(np.arange(width), a), size - len(shared), replace=False
                )
                candidates[query + 1, :size] = np.sort(np.concatenate((shared, other)))
    result = jax.device_get(
        paired_mqa_metadata(
            jnp.asarray(candidates), jnp.asarray(pages, jnp.int32), jnp.asarray(offsets)
        )
    )
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        for query in range(start, end, 2):
            a = candidates[query]
            a = a[a >= 0]
            b = candidates[query + 1] if query + 1 < end else np.array([], np.int32)
            b = b[b >= 0]
            union = np.union1d(a, b)
            n = len(union)
            np.testing.assert_array_equal(result.counts[query], n)
            np.testing.assert_array_equal(
                result.pages[query, 0, :n], pages[group, union]
            )
            np.testing.assert_array_equal(result.pages[query, 1, :n], union)
            for local, source in enumerate((a, b)):
                slots = [
                    int(np.searchsorted(source, value)) if value in source else -1
                    for value in union
                ]
                np.testing.assert_array_equal(result.pages[query, 2 + local, :n], slots)
            np.testing.assert_array_equal(result.pages[query, :, n:], -1)
            if query + 1 < end:
                np.testing.assert_array_equal(result.pages[query + 1], -1)


@pytest.mark.parametrize("capacity", [1, 3, 7, 128])
def test_merge_path_slots(capacity):
    rng = np.random.default_rng(37)
    for overlap in (0, 0.5, 1):
        a = np.sort(rng.choice(4 * capacity, capacity, replace=False)).astype(np.int32)
        shared = a[: int(capacity * overlap)]
        b = np.sort(
            np.concatenate(
                (
                    shared,
                    rng.choice(
                        np.setdiff1d(np.arange(4 * capacity), a),
                        capacity - len(shared),
                        replace=False,
                    ),
                )
            )
        ).astype(np.int32)

        def reader(values):
            values = jnp.asarray(values)
            return lambda position: jnp.where(
                (position >= 0) & (position < capacity),
                values[jnp.clip(position, 0, capacity - 1)],
                jnp.iinfo(jnp.int32).max,
            )

        i, j = jax.device_get(
            _merge_partition(
                reader(a),
                reader(b),
                jnp.arange(2 * capacity, dtype=jnp.int32),
                capacity,
                capacity,
                capacity,
            )
        )
        merged = sorted(
            [(int(value), 0) for value in a] + [(int(value), 1) for value in b]
        )
        prefix_a = np.cumsum([0, *[owner == 0 for _, owner in merged]])[:-1]
        np.testing.assert_array_equal(i, prefix_a)
        np.testing.assert_array_equal(j, np.arange(2 * capacity) - prefix_a)
