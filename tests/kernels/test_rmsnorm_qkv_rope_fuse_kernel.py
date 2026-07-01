# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerics tests for the fused RMSNorm + int8 QKV + RoPE(q,k) kernel."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.util import (quantize_tensor,
                                                         xla_quantized_matmul)
from vllm_torchtpu.kernels.rmsnorm_qkv_rope_fuse.kernel import (
    _apply_rope_pair, rmsnorm_qkv_rope_fuse)

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


@pytest.mark.parametrize(
    ("B", "H", "q_dim", "k_dim", "head_dim"),
    [
        # Qwen 30B-ish shapes: 16 q heads * 128 = 2048 q, 2 kv heads * 128
        # = 256 k. QKV = q_dim + k_dim + v_dim.
        (64, 1536, 2048, 256, 128),
        (128, 1536, 2048, 256, 128),
    ],
)
def test_matches_unfused_reference(B, H, q_dim, k_dim, head_dim):
    _require_tpu_v7()
    v_dim = k_dim  # v uses the same head count as k for GQA
    QKV = q_dim + k_dim + v_dim

    prng = jax.random.key(1)
    k0, k1, k2, k3, k4 = jax.random.split(prng, 5)
    x = jax.random.normal(k0, (B, H), dtype=jnp.bfloat16) * 0.1
    rms_w = (1.0 + jax.random.normal(k1, (H, ), dtype=jnp.bfloat16) * 0.02)
    w = jax.random.normal(k2, (QKV, H), dtype=jnp.bfloat16) * 0.05
    w_q, w_scale = quantize_tensor(w, jnp.int8, block_size=None)
    w_scale = jnp.squeeze(w_scale)
    cos = jax.random.uniform(k3, (B, head_dim // 2),
                             dtype=jnp.bfloat16,
                             minval=-1.0,
                             maxval=1.0)
    sin = jax.random.uniform(k4, (B, head_dim // 2),
                             dtype=jnp.bfloat16,
                             minval=-1.0,
                             maxval=1.0)

    actual = rmsnorm_qkv_rope_fuse(x, rms_w, w_q, w_scale, cos, sin, q_dim,
                                   k_dim)

    # Unfused reference: RMSNorm -> quantized matmul -> RoPE on q/k.
    x_normed = _rmsnorm_ref(x, rms_w)
    qkv = xla_quantized_matmul(x_normed,
                               w_q,
                               w_scale,
                               quantize_activation=False)
    q = qkv[:, :q_dim]
    k = qkv[:, q_dim:q_dim + k_dim]
    v = qkv[:, q_dim + k_dim:]
    q_rot = _apply_rope_pair(q, cos.astype(jnp.float32),
                             sin.astype(jnp.float32)).astype(qkv.dtype)
    k_rot = _apply_rope_pair(k, cos.astype(jnp.float32),
                             sin.astype(jnp.float32)).astype(qkv.dtype)
    expected = jnp.concatenate([q_rot, k_rot, v], axis=-1)

    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=0.5,
                               rtol=0.5)


def test_qkv_split_sums_to_qkv():
    _require_tpu_v7()
    x = jnp.zeros((8, 128), dtype=jnp.bfloat16)
    rms_w = jnp.zeros((128, ), dtype=jnp.bfloat16)
    w_q = jnp.zeros((300, 128), dtype=jnp.int8)
    w_s = jnp.zeros((300, ), dtype=jnp.float32)
    cos = jnp.zeros((8, 32), dtype=jnp.bfloat16)
    sin = jnp.zeros((8, 32), dtype=jnp.bfloat16)
    # 128 + 128 = 256, v_dim = 44, all positive.
    with pytest.raises(ValueError, match="no room for v"):
        rmsnorm_qkv_rope_fuse(x, rms_w, w_q, w_s, cos, sin, 200, 200)
