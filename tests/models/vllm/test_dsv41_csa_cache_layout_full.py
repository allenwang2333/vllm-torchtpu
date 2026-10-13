# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch unit tests for DeepSeek-V4/V4.1 CSA cache layout."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
for _pkg_name, _pkg_dir in (
    ("vllm_torchtpu", _SRC_ROOT),
    ("vllm_torchtpu.kernels", _SRC_ROOT / "kernels"),
    ("vllm_torchtpu.kernels.deepseek_v4", _SRC_ROOT / "kernels" / "deepseek_v4"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]
    sys.modules.setdefault(_pkg_name, _pkg)

from vllm_torchtpu.kernels.deepseek_v4 import (  # noqa: E402
    csa_cache_layout as layout_mod,
)


class _DummyRef:
    def __init__(self, arr: np.ndarray, dtype) -> None:
        self._arr = arr
        self.dtype = dtype
        self.shape = arr.shape

    def bitcast(self, target_dtype) -> np.ndarray:
        return self._arr.view(target_dtype)


def _ref_slabs_to_words(x: jax.Array) -> jax.Array:
    u = x.astype(jnp.uint32)
    words = u[..., 0, :] | (u[..., 1, :] << 8) | (u[..., 2, :] << 16)
    words = words | (u[..., 3, :] << 24)
    return lax.bitcast_convert_type(words, jnp.int32)


def _ref_words_to_slabs(w: jax.Array) -> jax.Array:
    u = lax.bitcast_convert_type(w, jnp.uint32)
    return jnp.stack(
        [((u >> (8 * b)) & 0xFF).astype(jnp.uint8) for b in range(4)],
        axis=-2,
    )


def _ref_rope_words(rope: jax.Array) -> jax.Array:
    bits = lax.bitcast_convert_type(rope, jnp.uint16).astype(jnp.uint32)
    return lax.bitcast_convert_type(bits[..., :32] | (bits[..., 32:] << 16), jnp.int32)


def test_is_word_array_u8_slab_shape_and_as_u8_slabs() -> None:
    assert layout_mod.is_word_array(jnp.int32) is True
    assert layout_mod.is_word_array(jnp.uint32) is True
    assert layout_mod.is_word_array(jnp.float32) is True
    assert layout_mod.is_word_array(jnp.uint8) is False
    assert layout_mod.is_word_array(jnp.int8) is False
    # Exercise fallback branch for non-cached dtypes (bfloat16, int64, complex64)
    assert layout_mod.is_word_array(jnp.bfloat16) is False
    assert layout_mod.is_word_array(jnp.int16) is False

    assert layout_mod.u8_slab_shape((4, 16, 128), jnp.int32) == (4, 16, 4, 128)
    assert layout_mod.u8_slab_shape((4, 16, 4, 128), jnp.uint8) == (4, 16, 4, 128)
    assert layout_mod.u8_slab_shape([4, 16, 4, 128], jnp.uint8) == (  # type: ignore[arg-type]
        4,
        16,
        4,
        128,
    )

    raw_i32 = np.arange(2 * 8 * 128, dtype=np.int32).reshape(2, 8, 128)
    ref_i32 = _DummyRef(raw_i32, jnp.int32)
    slabs = layout_mod.as_u8_slabs(ref_i32)
    assert slabs.shape == (2, 8, 4, 128) and slabs.dtype == np.uint8

    raw_u8 = np.zeros((2, 8, 4, 128), dtype=np.uint8)
    ref_u8 = _DummyRef(raw_u8, jnp.uint8)
    assert layout_mod.as_u8_slabs(ref_u8) is ref_u8


def test_slabs_to_words_and_words_to_slabs_roundtrip() -> None:
    rng = np.random.default_rng(123)
    slabs_np = rng.integers(0, 256, size=(3, 8, 4, 128), dtype=np.uint8)
    slabs = jnp.asarray(slabs_np, dtype=jnp.uint8)

    words = layout_mod.slabs_to_words(slabs)
    ref_words = _ref_slabs_to_words(slabs)
    assert words.dtype == jnp.int32 and words.shape == (3, 8, 128)
    np.testing.assert_array_equal(np.asarray(words), np.asarray(ref_words))

    # Exercise both int32 and uint32 input paths in words_to_slabs
    unpacked_i32 = layout_mod.words_to_slabs(words)
    unpacked_u32 = layout_mod.words_to_slabs(
        lax.bitcast_convert_type(words, jnp.uint32)
    )
    ref_unpacked = _ref_words_to_slabs(words)
    np.testing.assert_array_equal(np.asarray(unpacked_i32), slabs_np)
    np.testing.assert_array_equal(np.asarray(unpacked_u32), slabs_np)
    np.testing.assert_array_equal(np.asarray(unpacked_i32), np.asarray(ref_unpacked))


def test_rope_words_and_encode_rope_branches() -> None:
    rng = np.random.default_rng(456)
    rope_2d = jnp.asarray(
        rng.standard_normal((16, 64), dtype=np.float32), dtype=jnp.bfloat16
    )

    rw = layout_mod.rope_words(rope_2d)
    ref_rw = _ref_rope_words(rope_2d)
    assert rw.dtype == jnp.int32 and rw.shape == (16, 32)
    np.testing.assert_array_equal(np.asarray(rw), np.asarray(ref_rw))

    # Fast 2D aligned path in encode_rope
    enc_2d = layout_mod.encode_rope(rope_2d)
    ref_enc_2d = ref_rw.reshape(-1, 128)
    assert enc_2d.dtype == jnp.int32 and enc_2d.shape == (4, 128)
    np.testing.assert_array_equal(np.asarray(enc_2d), np.asarray(ref_enc_2d))

    # Fallback 3D path in encode_rope (ndim != 2)
    rope_3d = rope_2d.reshape(2, 8, 64)
    enc_3d = layout_mod.encode_rope(rope_3d)
    np.testing.assert_array_equal(np.asarray(enc_3d), np.asarray(ref_enc_2d))
