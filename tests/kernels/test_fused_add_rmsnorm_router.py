# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerics tests for fused_add + RMSNorm + MoE router bf16 matmul."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_add_rmsnorm_router.kernel import \
    fused_add_rmsnorm_router

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


def _reference(x, residual, rms_w, router_w, eps=1e-6):
    new_residual = (x.astype(jnp.float32) +
                     residual.astype(jnp.float32)).astype(x.dtype)
    x_f = new_residual.astype(jnp.float32)
    w_f = rms_w.astype(jnp.float32)
    ms = jnp.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(ms + eps)
    normed = (x_f * inv * w_f[None, :]).astype(x.dtype)
    router = jax.lax.dot_general(
        normed, router_w,
        dimension_numbers=(((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32).astype(x.dtype)
    return router, normed, new_residual


@pytest.mark.parametrize(
    ("M", "H", "NE"),
    [
        # Qwen3.5-397B TP=8 shard shapes; num_experts=512.
        (64, 4096, 512),
        (128, 4096, 512),
        (256, 4096, 512),
        (128, 1024, 256),
    ],
)
def test_matches_reference(M, H, NE):
    _require_tpu_v7()
    prng = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(prng, 4)
    x = jax.random.normal(k0, (M, H), dtype=jnp.bfloat16) * 0.1
    residual = jax.random.normal(k1, (M, H), dtype=jnp.bfloat16) * 0.05
    rms_w = (1.0 + jax.random.normal(k2, (H, ),
                                       dtype=jnp.bfloat16) * 0.02)
    router_w = jax.random.normal(k3, (NE, H), dtype=jnp.bfloat16) * 0.02

    router_a, normed_a, resid_a = fused_add_rmsnorm_router(
        x, residual, rms_w, router_w)
    router_r, normed_r, resid_r = _reference(x, residual, rms_w, router_w)

    np.testing.assert_allclose(np.asarray(resid_a),
                                np.asarray(resid_r),
                                atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(normed_a),
                                np.asarray(normed_r),
                                atol=1e-2, rtol=1e-2)
    # Router bf16 matmul has slightly wider bf16 tolerance.
    np.testing.assert_allclose(np.asarray(router_a),
                                np.asarray(router_r),
                                atol=5e-2, rtol=5e-2)


def test_shape_validation():
    _require_tpu_v7()
    x = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    residual = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    rms_w = jnp.zeros((64, ), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match="Hidden dim mismatch"):
        fused_add_rmsnorm_router(
            x, residual, rms_w,
            jnp.zeros((32, 128), dtype=jnp.bfloat16))

    with pytest.raises(ValueError, match="residual shape"):
        fused_add_rmsnorm_router(
            x, jnp.zeros((16, 64), dtype=jnp.bfloat16), rms_w,
            jnp.zeros((32, 64), dtype=jnp.bfloat16))

    with pytest.raises(ValueError, match="rms_weight shape"):
        fused_add_rmsnorm_router(
            x, residual, jnp.zeros((32, ), dtype=jnp.bfloat16),
            jnp.zeros((32, 64), dtype=jnp.bfloat16))
