# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerics tests for the arena-discovered fused RMSNorm + int8 QKV kernel.

Checks that ``rmsnorm_qkv_fuse(x, rms_w, w_q, w_s)`` matches the
composition ``xla_quantized_matmul(rmsnorm_ref(x, rms_w), w_q, w_s)``
within bf16 tolerance across the shape grid the arena optimized against.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.util import (quantize_tensor,
                                                         xla_quantized_matmul)
from vllm_torchtpu.kernels.rmsnorm_qkv_fuse.kernel import rmsnorm_qkv_fuse

jax.config.update("jax_numpy_dtype_promotion", "standard")


def _require_tpu_v7() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Expected JAX TPU backend, got {backend}.")
    device_kind = jax.devices()[0].device_kind
    match = re.match(r"^TPU[^\d]*(\d+)", device_kind)
    if match is None or int(match.group(1)) < 7:
        pytest.skip(f"Expected TPU v7+, got {device_kind}.")


def _rmsnorm_ref(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    x_f = x.astype(jnp.float32)
    w_f = w.astype(jnp.float32)
    ms = jnp.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(ms + eps)
    return (x_f * inv * w_f[None, :]).astype(x.dtype)


@pytest.mark.parametrize(
    ("B", "H", "QKV"),
    [
        # Match the shape grid in kernel_agents/tasks/rmsnorm_qkv_fuse.py.
        (64, 1536, 3 * 1536),
        (128, 1536, 3 * 1536),
        (256, 1536, 3 * 1536),
    ],
)
def test_matches_rmsnorm_plus_xla_quantized_matmul(B: int, H: int,
                                                   QKV: int) -> None:
    _require_tpu_v7()

    prng_key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(prng_key, 3)
    x = jax.random.normal(k0, (B, H), dtype=jnp.bfloat16) * 0.1
    rms_w = (1.0 + jax.random.normal(k1, (H, ), dtype=jnp.bfloat16) * 0.02)
    w = jax.random.normal(k2, (QKV, H), dtype=jnp.bfloat16) * 0.05

    w_q, w_scale = quantize_tensor(w, jnp.int8, block_size=None)
    w_scale = jnp.squeeze(w_scale)
    assert w_scale.shape == (QKV, )

    actual = rmsnorm_qkv_fuse(x, rms_w, w_q, w_scale)
    x_normed = _rmsnorm_ref(x, rms_w)
    expected = xla_quantized_matmul(x_normed,
                                    w_q,
                                    w_scale,
                                    quantize_activation=False)

    # bf16 tolerance — matches the standard used by
    # test_quantized_matmul_kernel.
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=0.5,
                               rtol=0.5)


def test_shape_validation_hidden_mismatch() -> None:
    _require_tpu_v7()
    x = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    rms_w = jnp.zeros((64, ), dtype=jnp.bfloat16)
    w_q = jnp.zeros((32, 128), dtype=jnp.int8)
    w_s = jnp.zeros((32, ), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Hidden dim mismatch"):
        rmsnorm_qkv_fuse(x, rms_w, w_q, w_s)
