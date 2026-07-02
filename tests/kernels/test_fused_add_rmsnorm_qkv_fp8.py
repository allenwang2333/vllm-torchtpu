# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerics tests for the fused (residual+add) + RMSNorm + FP8 QKV kernel."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_add_rmsnorm_qkv_fp8.kernel import \
    fused_add_rmsnorm_qkv_fp8
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

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


def _rmsnorm_ref(x, w, eps=1e-6):
    x_f = x.astype(jnp.float32)
    w_f = w.astype(jnp.float32)
    ms = jnp.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(ms + eps)
    return (x_f * inv * w_f[None, :]).astype(x.dtype)


def _reference(x, residual, rms_w, qkv_w_q, qkv_w_scale):
    """Unfused composition: fused_add → RMSNorm → xla_quantized_matmul."""
    new_residual = (x.astype(jnp.float32) +
                     residual.astype(jnp.float32)).astype(x.dtype)
    normed = _rmsnorm_ref(new_residual, rms_w)
    qkv = xla_quantized_matmul(normed, qkv_w_q, qkv_w_scale,
                                quantize_activation=False)
    return qkv, new_residual


@pytest.mark.parametrize(
    ("M", "H", "OUT"),
    [
        # Qwen3.5-397B-A17B-FP8 TP=8 shard shape and a couple of smaller
        # sanity shapes.
        (64, 4096, 2560),
        (128, 4096, 2560),
        (256, 4096, 2560),
        (128, 1024, 2048),
    ],
)
def test_matches_reference(M, H, OUT):
    _require_tpu_v7()
    prng = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(prng, 4)
    x = jax.random.normal(k0, (M, H), dtype=jnp.bfloat16) * 0.1
    residual = jax.random.normal(k1, (M, H), dtype=jnp.bfloat16) * 0.05
    rms_w = (1.0 + jax.random.normal(k2, (H, ), dtype=jnp.bfloat16) * 0.02)

    # Build fp8 weights + per-channel scale.
    w = jax.random.normal(k3, (OUT, H), dtype=jnp.bfloat16) * 0.05
    w_f32 = w.astype(jnp.float32)
    w_abs = jnp.max(jnp.abs(w_f32), axis=-1, keepdims=True)
    w_scale_val = jnp.clip(w_abs / 448.0, 1e-12, None)
    w_q = jnp.clip(w_f32 / w_scale_val,
                   -448.0, 448.0).astype(jnp.float8_e4m3fn)
    w_scale = jnp.squeeze(w_scale_val, axis=-1).astype(jnp.float32)

    qkv_actual, resid_actual = fused_add_rmsnorm_qkv_fp8(
        x, residual, rms_w, w_q, w_scale)
    qkv_ref, resid_ref = _reference(x, residual, rms_w, w_q, w_scale)

    # Residual is exact (bf16 addition).
    np.testing.assert_allclose(np.asarray(resid_actual),
                                np.asarray(resid_ref),
                                atol=1e-3, rtol=1e-3)
    # QKV output — bf16 tolerance matching the quantized_matmul test.
    np.testing.assert_allclose(np.asarray(qkv_actual),
                                np.asarray(qkv_ref),
                                atol=0.5, rtol=0.5)


def test_shape_validation_hidden_mismatch():
    _require_tpu_v7()
    x = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    residual = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    rms_w = jnp.zeros((64, ), dtype=jnp.bfloat16)
    w_q = jnp.zeros((32, 128), dtype=jnp.float8_e4m3fn)
    w_s = jnp.zeros((32, ), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Hidden dim mismatch"):
        fused_add_rmsnorm_qkv_fp8(x, residual, rms_w, w_q, w_s)


def test_shape_validation_residual_mismatch():
    _require_tpu_v7()
    x = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    residual = jnp.zeros((16, 64), dtype=jnp.bfloat16)
    rms_w = jnp.zeros((64, ), dtype=jnp.bfloat16)
    w_q = jnp.zeros((32, 64), dtype=jnp.float8_e4m3fn)
    w_s = jnp.zeros((32, ), dtype=jnp.float32)
    with pytest.raises(ValueError, match="residual shape"):
        fused_add_rmsnorm_qkv_fp8(x, residual, rms_w, w_q, w_s)
