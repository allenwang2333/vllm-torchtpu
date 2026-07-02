# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerics tests for the standalone fused_add + RMSNorm Pallas kernel."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_add_rmsnorm.kernel import fused_add_rmsnorm

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


def _reference(x, residual, weight, eps=1e-6):
    new_residual = (x.astype(jnp.float32) +
                     residual.astype(jnp.float32)).astype(x.dtype)
    x_f = new_residual.astype(jnp.float32)
    w_f = weight.astype(jnp.float32)
    ms = jnp.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(ms + eps)
    normed = (x_f * inv * w_f[None, :]).astype(x.dtype)
    return normed, new_residual


@pytest.mark.parametrize(
    ("M", "H"),
    [
        (64, 4096),
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (128, 1024),
    ],
)
def test_matches_reference(M, H):
    _require_tpu_v7()
    prng = jax.random.key(0)
    k0, k1, k2 = jax.random.split(prng, 3)
    x = jax.random.normal(k0, (M, H), dtype=jnp.bfloat16) * 0.1
    residual = jax.random.normal(k1, (M, H), dtype=jnp.bfloat16) * 0.05
    weight = (1.0 + jax.random.normal(k2, (H, ),
                                        dtype=jnp.bfloat16) * 0.02)

    normed_actual, resid_actual = fused_add_rmsnorm(x, residual, weight)
    normed_ref, resid_ref = _reference(x, residual, weight)

    np.testing.assert_allclose(np.asarray(resid_actual),
                                np.asarray(resid_ref),
                                atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(normed_actual),
                                np.asarray(normed_ref),
                                atol=1e-2, rtol=1e-2)


def test_shape_validation():
    _require_tpu_v7()
    x = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    residual = jnp.zeros((16, 64), dtype=jnp.bfloat16)
    weight = jnp.zeros((64, ), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="residual shape"):
        fused_add_rmsnorm(x, residual, weight)

    weight_wrong = jnp.zeros((32, ), dtype=jnp.bfloat16)
    residual_ok = jnp.zeros((8, 64), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="weight shape"):
        fused_add_rmsnorm(x, residual_ok, weight_wrong)
