# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host helpers for the native SparseCore GLM-5.2 cache layout.

These conversions define the standalone native-SC kernel's byte contract. They
are intended for tests and offline cache construction; converting a serving
cache immediately before attention would defeat the layout's purpose.
"""

from __future__ import annotations

import numpy as np

NOPE_SUBROWS = 4
LANE_BYTES = 128
ROPE_WORDS = LANE_BYTES // np.dtype(np.uint32).itemsize
# PACKED ROPE: the 64 B fp8 rope key alone, 16 words, eight tokens per row.
ROPE_BYTES_PACKED = 64
ROPE_WORDS_PACKED = ROPE_BYTES_PACKED // np.dtype(np.uint32).itemsize
ROPE_TOKENS_PER_ROW_PACKED = LANE_BYTES // ROPE_WORDS_PACKED


def _require_u8(name: str, value: np.ndarray, suffix: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.shape[-len(suffix) :] != suffix:
        raise ValueError(
            f"{name} must be uint8 with trailing shape {suffix}, got "
            f"{value.dtype} {value.shape}"
        )
    return value


def pack_nope(nope_u8: np.ndarray) -> np.ndarray:
    """Pack ``[..., token, 4, 128] u8`` as ``[..., token, 128] u32``.

    Native word ``w`` contains the four bytes ``nope_u8[..., :, w]``. The
    transpose is needed for a conventional row-major host array; the deployed
    TPU ``u8[4,128]`` tile has the same raw HBM byte image already.
    """
    nope_u8 = _require_u8("nope_u8", nope_u8, (NOPE_SUBROWS, LANE_BYTES))
    lane_major = np.ascontiguousarray(np.swapaxes(nope_u8, -2, -1))
    return lane_major.view(np.uint32).reshape(*nope_u8.shape[:-2], LANE_BYTES)


def unpack_nope(nope_i32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_nope`."""
    nope_i32 = np.asarray(nope_i32)
    if nope_i32.dtype != np.uint32 or nope_i32.shape[-1] != LANE_BYTES:
        raise ValueError(
            f"nope_i32 must be uint32[..., 128], got {nope_i32.dtype} {nope_i32.shape}"
        )
    lane_major = (
        np.ascontiguousarray(nope_i32)
        .view(np.uint8)
        .reshape(*nope_i32.shape[:-1], LANE_BYTES, NOPE_SUBROWS)
    )
    return np.ascontiguousarray(np.swapaxes(lane_major, -2, -1))


def pack_rope_banded(rope_u8: np.ndarray) -> np.ndarray:
    """Pack native-SC ROPE words into four-token, 128-word HBM rows.

    ``word[token, w].byte[band] = rope_u8[token, 32 * band + w]``. Four
    tokens occupy four consecutive 32-word quarters of the returned row, so
    an SC gather reads exactly 128 bytes per selected token.
    """
    rope_u8 = _require_u8("rope_u8", rope_u8, (NOPE_SUBROWS, LANE_BYTES))
    groups = rope_u8.shape[-3]
    banded = np.ascontiguousarray(
        rope_u8.reshape(*rope_u8.shape[:-1], NOPE_SUBROWS, ROPE_WORDS).swapaxes(-2, -1)
    )
    return banded.view(np.uint32).reshape(*rope_u8.shape[:-3], groups, LANE_BYTES)


def unpack_rope_banded(rope_i32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rope_banded`."""
    rope_i32 = np.asarray(rope_i32)
    if rope_i32.dtype != np.uint32 or rope_i32.shape[-1] != LANE_BYTES:
        raise ValueError(
            "rope_i32 must be uint32[..., groups, 128], got "
            f"{rope_i32.dtype} {rope_i32.shape}"
        )
    groups = rope_i32.shape[-2]
    banded = (
        np.ascontiguousarray(rope_i32)
        .view(np.uint8)
        .reshape(*rope_i32.shape[:-2], groups, NOPE_SUBROWS, ROPE_WORDS, NOPE_SUBROWS)
    )
    return np.ascontiguousarray(
        banded.swapaxes(-2, -1).reshape(
            *rope_i32.shape[:-2], groups, NOPE_SUBROWS, LANE_BYTES
        )
    )


def pack_rope_packed(rope_u8: np.ndarray) -> np.ndarray:
    """Pack ``[..., token, 64] u8`` rope keys as PACKED ``[..., token/8, 128] u32``.

    ``word[token, w].byte[band] = rope_u8[token, 16 * band + w]`` -- the same
    four-band interleave as :func:`pack_rope_banded` (and the writer's
    ``kv_cache_utils.pack_tokens``) at a 16-word band stride. Eight tokens
    occupy eight consecutive 16-word eighths of a row, so an SC gather reads
    exactly 64 bytes per selected token.
    """
    rope_u8 = _require_u8("rope_u8", rope_u8, (ROPE_BYTES_PACKED,))
    tokens = rope_u8.shape[-2]
    if tokens % ROPE_TOKENS_PER_ROW_PACKED:
        raise ValueError(
            f"token count {tokens} must be a multiple of {ROPE_TOKENS_PER_ROW_PACKED}"
        )
    banded = np.ascontiguousarray(
        rope_u8.reshape(*rope_u8.shape[:-1], NOPE_SUBROWS, ROPE_WORDS_PACKED).swapaxes(
            -2, -1
        )
    )
    return banded.view(np.uint32).reshape(
        *rope_u8.shape[:-2], tokens // ROPE_TOKENS_PER_ROW_PACKED, LANE_BYTES
    )


def unpack_rope_packed(rope_i32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rope_packed`: ``[..., token, 64] u8``."""
    rope_i32 = np.asarray(rope_i32)
    if rope_i32.dtype != np.uint32 or rope_i32.shape[-1] != LANE_BYTES:
        raise ValueError(
            "rope_i32 must be uint32[..., rows, 128], got "
            f"{rope_i32.dtype} {rope_i32.shape}"
        )
    tokens = rope_i32.shape[-2] * ROPE_TOKENS_PER_ROW_PACKED
    banded = (
        np.ascontiguousarray(rope_i32)
        .view(np.uint8)
        .reshape(*rope_i32.shape[:-2], tokens, ROPE_WORDS_PACKED, NOPE_SUBROWS)
    )
    return np.ascontiguousarray(banded.swapaxes(-2, -1)).reshape(
        *rope_i32.shape[:-2], tokens, ROPE_BYTES_PACKED
    )


def pack_native_caches(
    nope_u8: np.ndarray,
    rope_u8: np.ndarray,
    page_size: int = 256,
    rope_packed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack complete paged caches into the standalone native-SC ABI.

    ``rope_packed`` selects the 64 B/token PACKED ROPE cache
    (``[pages, page_size/8, 128]``); ``rope_u8`` then holds the 64 real bytes
    per token (a 128 B padded row is truncated to them).
    """
    nope_u8 = np.asarray(nope_u8)
    rope_u8 = np.asarray(rope_u8)
    total_tokens = (
        nope_u8.shape[0] if nope_u8.ndim == 3 else nope_u8.shape[0] * nope_u8.shape[1]
    )
    total_pages = total_tokens // page_size
    if nope_u8.ndim == 3:
        nope_u8 = nope_u8.reshape(total_pages, page_size, NOPE_SUBROWS, LANE_BYTES)
    if rope_packed:
        rope_u8 = rope_u8.reshape(total_pages, page_size, -1)[..., :ROPE_BYTES_PACKED]
        return pack_nope(nope_u8), pack_rope_packed(np.ascontiguousarray(rope_u8))
    if rope_u8.ndim in (2, 3):
        rope_u8 = rope_u8.reshape(
            total_pages, page_size // NOPE_SUBROWS, NOPE_SUBROWS, LANE_BYTES
        )
    return pack_nope(nope_u8), pack_rope_banded(rope_u8)


def pack_selected_rope_for_tc(rope_u8: np.ndarray) -> np.ndarray:
    """Host oracle for the gather's four-token register delta-swap."""
    rope_u8 = np.asarray(rope_u8, dtype=np.uint8).reshape(-1, NOPE_SUBROWS, LANE_BYTES)
    swapped = np.ascontiguousarray(rope_u8.swapaxes(1, 2))
    return swapped.view(np.uint32).reshape(-1, LANE_BYTES)


def flatten_nope_rows(nope_i32: np.ndarray) -> np.ndarray:
    nope_i32 = np.asarray(nope_i32)
    if nope_i32.dtype != np.uint32 or nope_i32.shape[-1] != LANE_BYTES:
        raise ValueError("expected uint32 NOPE cache with 128-word rows")
    return nope_i32.reshape(-1, LANE_BYTES)


def rope_words_per_token(nope_i32: np.ndarray, rope_i32: np.ndarray) -> int:
    """ROPE words per token from the caches' size ratio: 32 padded, 16 packed.

    The same rule as ``native_sc_gather.rope_words_for_caches``.
    """
    ratio, rem = divmod(np.asarray(nope_i32).size, np.asarray(rope_i32).size)
    if rem == 0 and ratio == NOPE_SUBROWS:
        return ROPE_WORDS
    if rem == 0 and ratio == 2 * NOPE_SUBROWS:
        return ROPE_WORDS_PACKED
    raise ValueError(
        f"NOPE cache must be 4x (padded) or 8x (packed) the ROPE cache, got "
        f"{np.asarray(nope_i32).size} vs {np.asarray(rope_i32).size}"
    )


def flatten_rope_rows(rope_i32: np.ndarray, words_per_token: int = ROPE_WORDS) -> np.ndarray:
    """One row of ``words_per_token`` words per token (32 padded, 16 packed)."""
    rope_i32 = np.asarray(rope_i32)
    if rope_i32.dtype != np.uint32 or rope_i32.shape[-1] != LANE_BYTES:
        raise ValueError("expected uint32 ROPE cache with 128-word group rows")
    if words_per_token not in (ROPE_WORDS, ROPE_WORDS_PACKED):
        raise ValueError(f"words_per_token must be 32 or 16, got {words_per_token}")
    return rope_i32.reshape(-1, words_per_token)


def source_bytes_per_token(
    nope_i32: np.ndarray, rope_i32: np.ndarray
) -> tuple[int, int]:
    """Return the logical allocation bytes per token for both caches."""
    nope_tokens = np.prod(nope_i32.shape[:-1], dtype=np.int64)
    tokens_per_row = LANE_BYTES // rope_words_per_token(nope_i32, rope_i32)
    rope_tokens = np.prod(rope_i32.shape[:-1], dtype=np.int64) * tokens_per_row
    if nope_tokens != rope_tokens:
        raise ValueError(f"cache token counts differ: {nope_tokens} vs {rope_tokens}")
    return (nope_i32.nbytes // int(nope_tokens), rope_i32.nbytes // int(rope_tokens))
