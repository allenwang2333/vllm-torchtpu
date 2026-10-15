# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool word-row layouts of the DSV4.1 BF16 ``[128]`` index keys.

Both layouts store record ``2 * r + s`` (``s`` in ``{0, 1}``) in pool word
row ``r``, so record addresses, pages and banks are unchanged. Scores use the
split element order (`cache.split_halves`): lane ``i`` of a key is element
``2 * i`` for ``i < 64`` and element ``2 * (i - 64) + 1`` otherwise.

* legacy (packed pairs): slot ``s`` owns words ``[64 * s, 64 * s + 64)``; word
  ``j`` holds elements ``2 * j`` (low half) and ``2 * j + 1`` (high half).
  Readers rebuild split-order keys with stride-2 stores, rolls and selects.
* native (``TPU_DSV41_INDEX_NATIVE_KEYS=1``): slot ``s`` owns 16-bit half
  ``s`` of all 128 words; word ``l`` holds split-order lane ``l``. This is the
  TPU packed BF16 ``(16, 128)`` tiling, so ``pltpu.bitcast(int32[R, 128],
  bfloat16)`` is directly the ``[2 * R, 128]`` key tile in record order (BF16
  row ``2 * r`` = low halves of word row ``r``, row ``2 * r + 1`` = high
  halves). Writing one key is a read-modify-write of its word row.

Both layouts produce identical BF16 key values, hence bitwise-identical
scores. The native layout requires every record pair ``(2 * r, 2 * r + 1)`` to
belong to one index cache page (even page size, stride and offset).
"""

import os

import jax
import jax.numpy as jnp

NATIVE_KEYS = os.environ.get("TPU_DSV41_INDEX_NATIVE_KEYS", "0") == "1"

ROW_WORDS = 128
DIM = 128
HALF = ROW_WORDS // 2
_LOW = jnp.int32(0xFFFF)
_HIGH = jnp.int32(-65536)  # 0xFFFF0000


def key_bits(keys: jax.Array) -> jax.Array:
    """BF16 ``[n, 128]`` -> int32 ``[n, 128]`` split-order 16-bit patterns."""
    if keys.dtype != jnp.bfloat16 or keys.ndim != 2 or keys.shape[1] != DIM:
        raise ValueError("Expected BF16 [n, 128] index keys")
    # Same packing as the legacy writer (word j = elements 2j | 2j+1 << 16),
    # then even elements (low halves) followed by odd elements (high halves).
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, HALF, 2), jnp.int32)
    return jnp.concatenate(
        [words & _LOW, jax.lax.shift_right_logical(words, 16)], axis=1
    )


def native_to_legacy_rows(rows: jax.Array) -> jax.Array:
    """Native word rows ``[..., 128]`` -> the legacy packed rows (exact)."""
    first, second = rows[..., :HALF], rows[..., HALF:]
    slot0 = (first & _LOW) | (second << 16)
    slot1 = jax.lax.shift_right_logical(first, 16) | (second & _HIGH)
    return jnp.concatenate([slot0, slot1], axis=-1)


def legacy_to_native_rows(rows: jax.Array) -> jax.Array:
    """Inverse of `native_to_legacy_rows`."""
    slot0, slot1 = rows[..., :HALF], rows[..., HALF:]
    first = (slot0 & _LOW) | (slot1 << 16)
    second = jax.lax.shift_right_logical(slot0, 16) | (slot1 & _HIGH)
    return jnp.concatenate([first, second], axis=-1)


def native_rows_to_keys(rows: jax.Array) -> jax.Array:
    """XLA reference of the in-kernel bitcast: int32 ``[R, 128]`` -> BF16
    ``[2R, 128]`` split-order keys in record order (`pltpu.bitcast` semantics)."""

    def bf16(bits):
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)

    keys = jnp.stack([bf16(rows << 16), bf16(rows & _HIGH)], axis=-2)
    return keys.reshape(*rows.shape[:-2], 2 * rows.shape[-2], rows.shape[-1])


def merge_native_rows(
    cache: jax.Array, records: jax.Array, keys: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Full native word rows for unique record writes (`merge_sub_rows` analog).

    Each valid record yields ``(row, merged row)``: the current row with every
    half written in this batch replaced. Records sharing a row yield identical
    rows, so the full-row writes may land in any order. Invalid records -> -1.
    """
    bits = key_bits(keys)
    valid = (records >= 0) & (records < cache.shape[0] * 2)
    rows = jnp.where(valid, records // 2, -1)
    slots = records % 2
    merged = cache[jnp.maximum(rows, 0)]
    same_row = (rows[:, None] == rows[None, :]) & valid[:, None] & valid[None, :]
    for slot, keep in ((0, _HIGH), (1, _LOW)):
        writers = same_row & (slots[None, :] == slot)
        source = jnp.take(bits, jnp.argmax(writers, axis=1), axis=0)
        merged = jnp.where(
            writers.any(axis=1)[:, None],
            (merged & keep) | (source << (16 * slot)),
            merged,
        )
    return rows, merged


def scatter_native_keys(
    cache: jax.Array, records: jax.Array, keys: jax.Array, write_rows=None
) -> jax.Array:
    """Write BF16 keys into a native single-bank pool.

    ``write_rows(cache, rows, words)`` publishes full word rows (rows may
    repeat with identical contents; -1 drops); defaults to an XLA scatter.
    """
    if not keys.shape[0]:
        return cache
    rows, merged = merge_native_rows(cache, records, keys)
    if write_rows is not None:
        return write_rows(cache, rows, merged)
    rows = jnp.where(rows >= 0, rows, cache.shape[0])
    return cache.at[rows].set(merged, mode="drop")


def rows_to_keys(words, scratch, native: bool):
    """In-kernel: VMEM word rows ``[R, 128]`` -> split-order BF16 ``[2R, 128]``."""
    if native:
        from jax.experimental.pallas import tpu as pltpu

        return pltpu.bitcast(words, jnp.bfloat16)
    from .index_word_scores import unpack_word_rows

    return unpack_word_rows(words, scratch)
