# SPDX-License-Identifier: Apache-2.0
"""Option-B harness: measure fused vs unfused Qwen3-shape pre-attention path.

Qwen3-Coder-30B-A3B-Instruct pre-attention path per layer:

    hidden_states  = input_layernorm(x)                          (or fused-add form)
    qkv            = qkv_proj(hidden_states)                     # int8 mm
    q, k, v        = qkv.split([q_size, k_size, v_size], dim=-1)
    q              = q.view(B, n_q_heads, head_dim); q_normed  = q_norm(q); q = q_normed.view(...)
    k              = k.view(B, n_kv_heads, head_dim); k_normed = k_norm(k); k = k_normed.view(...)
    q, k           = rotary_emb(positions, q, k)
    -> attention

The FUSED variant substitutes the first two lines with:

    qkv            = fused_rmsnorm_qkv(x, input_layernorm_weight,
                                       qkv_proj_weight_q, qkv_proj_weight_scale)

Everything downstream (split, q_norm, k_norm, RoPE) stays identical.

We test:
  1. NUMERICAL EQUIVALENCE: fused and unfused produce the same (q, k, v)
     within bf16 tolerance.
  2. TIMING: median forward time under jax.jit, comparing the two paths.

Shapes: Qwen3-Coder-30B-A3B-Instruct
  hidden       = 2048  (per-shard; TP=1 here)
  num_q_heads  = 16
  num_kv_heads = 2
  head_dim     = 128
  q_size       = 2048
  kv_size      = 256

Batch sizes swept: 64, 128, 256 (typical decode + small-prefill fill).

Reports:
  - per-shape unfused median (us), fused median (us), speedup
  - geomean speedup across shapes
  - max_abs and rel_err of (fused - unfused)
"""

from __future__ import annotations

import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.quantized_matmul.util import (quantize_tensor,
                                                         xla_quantized_matmul)
from vllm_torchtpu.kernels.rmsnorm_qkv_fuse.kernel import rmsnorm_qkv_fuse

jax.config.update("jax_numpy_dtype_promotion", "standard")

HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM  # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 256
QKV_SIZE = Q_SIZE + KV_SIZE + KV_SIZE  # 2560

BATCH_SIZES = (64, 128, 256)


def _rmsnorm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Reference RMSNorm over the last axis. fp32 accumulation."""
    x_f = x.astype(jnp.float32)
    w_f = w.astype(jnp.float32)
    ms = jnp.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(ms + eps)
    return (x_f * inv * w_f).astype(x.dtype)


def _apply_rope_pair(t: jax.Array, cos: jax.Array,
                     sin: jax.Array) -> jax.Array:
    """Neox-style RoPE. ``t`` is (B, H*num_heads); split heads then rotate."""
    B, total = t.shape
    hd = 2 * cos.shape[-1]
    nh = total // hd
    t3 = t.reshape(B, nh, hd)
    a = t3[..., :hd // 2]
    b = t3[..., hd // 2:]
    ra = a * cos[:, None, :] - b * sin[:, None, :]
    rb = a * sin[:, None, :] + b * cos[:, None, :]
    return jnp.concatenate([ra, rb], axis=-1).reshape(B, total)


def _make_inputs(
    B: int, seed: int
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
           jax.Array, jax.Array]:
    """Return (x, rms_w, qkv_w_q, qkv_w_scale, q_norm_w, k_norm_w, cos, sin)."""
    key = jax.random.key(seed)
    kx, krms, kw, kqn, kkn, kc, ks = jax.random.split(key, 7)
    x = jax.random.normal(kx, (B, HIDDEN), dtype=jnp.bfloat16) * 0.1
    rms_w = (1.0 + jax.random.normal(krms,
                                     (HIDDEN, ), dtype=jnp.bfloat16) * 0.02)
    w = jax.random.normal(kw, (QKV_SIZE, HIDDEN), dtype=jnp.bfloat16) * 0.05
    w_q, w_scale = quantize_tensor(w, jnp.int8, block_size=None)
    w_scale = jnp.squeeze(w_scale)
    q_norm_w = (1.0 +
                jax.random.normal(kqn,
                                  (HEAD_DIM, ), dtype=jnp.bfloat16) * 0.02)
    k_norm_w = (1.0 +
                jax.random.normal(kkn,
                                  (HEAD_DIM, ), dtype=jnp.bfloat16) * 0.02)
    cos = jax.random.uniform(kc, (B, HEAD_DIM // 2),
                             dtype=jnp.bfloat16,
                             minval=-1.0,
                             maxval=1.0)
    sin = jax.random.uniform(ks, (B, HEAD_DIM // 2),
                             dtype=jnp.bfloat16,
                             minval=-1.0,
                             maxval=1.0)
    return x, rms_w, w_q, w_scale, q_norm_w, k_norm_w, cos, sin


def unfused_pre_attention(x, rms_w, qkv_w_q, qkv_w_scale, q_norm_w, k_norm_w,
                          cos, sin):
    """Exact Qwen3 pre-attention forward, unfused. Returns (q, k, v)."""
    h = _rmsnorm(x, rms_w)
    qkv = xla_quantized_matmul(h,
                               qkv_w_q,
                               qkv_w_scale,
                               quantize_activation=False)
    q = qkv[:, :Q_SIZE]
    k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE]
    v = qkv[:, Q_SIZE + KV_SIZE:]
    q_h = q.reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    q_h = _rmsnorm(q_h, q_norm_w)
    q = q_h.reshape(-1, Q_SIZE)
    k_h = k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    k_h = _rmsnorm(k_h, k_norm_w)
    k = k_h.reshape(-1, KV_SIZE)
    q = _apply_rope_pair(q, cos.astype(jnp.float32),
                         sin.astype(jnp.float32)).astype(q.dtype)
    k = _apply_rope_pair(k, cos.astype(jnp.float32),
                         sin.astype(jnp.float32)).astype(k.dtype)
    return q, k, v


def fused_pre_attention(x, rms_w, qkv_w_q, qkv_w_scale, q_norm_w, k_norm_w,
                        cos, sin):
    """Qwen3 pre-attention forward using rmsnorm_qkv_fuse for the first
    two ops; q_norm / k_norm / RoPE remain separate."""
    qkv = rmsnorm_qkv_fuse(x, rms_w, qkv_w_q, qkv_w_scale)
    q = qkv[:, :Q_SIZE]
    k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE]
    v = qkv[:, Q_SIZE + KV_SIZE:]
    q_h = q.reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    q_h = _rmsnorm(q_h, q_norm_w)
    q = q_h.reshape(-1, Q_SIZE)
    k_h = k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    k_h = _rmsnorm(k_h, k_norm_w)
    k = k_h.reshape(-1, KV_SIZE)
    q = _apply_rope_pair(q, cos.astype(jnp.float32),
                         sin.astype(jnp.float32)).astype(q.dtype)
    k = _apply_rope_pair(k, cos.astype(jnp.float32),
                         sin.astype(jnp.float32)).astype(k.dtype)
    return q, k, v


def _block_ready(pytree):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), pytree)


def _time_median(fn: Callable,
                 args,
                 warmup: int = 5,
                 iters: int = 30) -> float:
    for _ in range(warmup):
        out = fn(*args)
        _block_ready(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        _block_ready(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[iters // 2]


def _check_equivalence(B: int) -> Tuple[float, float]:
    inputs = _make_inputs(B, seed=1234)
    unfused = jax.jit(unfused_pre_attention)
    fused = jax.jit(fused_pre_attention)
    q_u, k_u, v_u = unfused(*inputs)
    q_f, k_f, v_f = fused(*inputs)
    max_abs = max(
        float(
            jnp.max(jnp.abs(q_u.astype(jnp.float32) -
                            q_f.astype(jnp.float32)))),
        float(
            jnp.max(jnp.abs(k_u.astype(jnp.float32) -
                            k_f.astype(jnp.float32)))),
        float(
            jnp.max(jnp.abs(v_u.astype(jnp.float32) -
                            v_f.astype(jnp.float32)))),
    )
    ref_max = max(
        float(jnp.max(jnp.abs(q_u.astype(jnp.float32)))),
        float(jnp.max(jnp.abs(k_u.astype(jnp.float32)))),
        float(jnp.max(jnp.abs(v_u.astype(jnp.float32)))),
    )
    rel = max_abs / max(ref_max, 1e-6)
    return max_abs, rel


def main() -> None:
    print(f"device: {jax.default_backend()} {jax.devices()[0].device_kind}")
    print(f"Qwen3 shapes: HIDDEN={HIDDEN} Q_SIZE={Q_SIZE} KV_SIZE={KV_SIZE} "
          f"HEAD_DIM={HEAD_DIM}")
    print()

    print("Numerical equivalence (fused vs unfused):")
    for B in BATCH_SIZES:
        max_abs, rel = _check_equivalence(B)
        print(f"  B={B}: max_abs_diff={max_abs:.4f}  rel_err={rel:.4f}")
    print()

    print(f"{'B':>4} {'unfused(us)':>12} {'fused(us)':>12} {'speedup':>8}")
    print("-" * 40)
    speedups = []
    for B in BATCH_SIZES:
        inputs = _make_inputs(B, seed=1234)
        unfused = jax.jit(unfused_pre_attention)
        fused = jax.jit(fused_pre_attention)
        t_u = _time_median(unfused, inputs)
        t_f = _time_median(fused, inputs)
        speedup = t_u / t_f
        speedups.append(speedup)
        print(f"{B:>4} {t_u * 1e6:>12.2f} {t_f * 1e6:>12.2f} "
              f"{speedup:>7.3f}x")
    geomean = float(np.exp(np.mean(np.log(speedups))))
    print("-" * 40)
    print(f"geomean speedup: {geomean:.3f}x")


if __name__ == "__main__":
    main()
