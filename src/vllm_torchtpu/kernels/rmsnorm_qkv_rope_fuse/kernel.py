# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm → int8 QKV projection → RoPE(q, k).

Arena-discovered variant of :mod:`vllm_torchtpu.kernels.rmsnorm_qkv_fuse`
that additionally applies RoPE to the ``q`` and ``k`` slices of the QKV
output. Discovered by the kernel_agents arena as a 1.079× TPOT-geomean
win over the XLA-fused baseline on transformer-layer-chain workloads.

Structure:
- One Pallas call for RMSNorm + int8 QKV projection (identical to
  :func:`vllm_torchtpu.kernels.rmsnorm_qkv_fuse.kernel.rmsnorm_qkv_fuse`).
- RoPE applied outside the kernel on the ``q``/``k`` slices, then
  concatenated with ``v`` back into a single tensor.

The RoPE-outside-kernel structure is intentional: the arena's Optimizer
explored on-kernel RoPE variants but they did not beat the workload
gate. The winning variant is the one materialized here.

Input contract:
- ``x``           : (B, H)                 bf16   activation
- ``rms_w``       : (H,)                   bf16   RMSNorm weight
- ``qkv_w_q``     : (QKV, H)               int8   per-channel-quantized
- ``qkv_w_scale`` : (QKV,)                 fp32   per-output-row scale
- ``cos``, ``sin``: (B, hd/2)              bf16/fp32  RoPE tables
- ``q_dim``, ``k_dim``: ints; ``v_dim = QKV - q_dim - k_dim`` is inferred.

Layout convention: the fused kernel outputs ``[Q | K | V]`` along the
last axis. ``q_dim + k_dim + v_dim == QKV`` is asserted. Both ``q`` and
``k`` are RoPE-rotated with the SAME ``cos``/``sin`` tables (GQA layout).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _apply_rope_pair(t: jax.Array, cos: jax.Array,
                     sin: jax.Array) -> jax.Array:
    """Apply RoPE to ``t`` of shape (B, num_heads*head_dim).

    ``cos``/``sin`` have shape ``(B, head_dim/2)``; broadcast across all
    heads. The rotation splits ``head_dim`` in half (interleaved pairs).
    """
    B, total = t.shape
    hd = 2 * cos.shape[-1]
    if total % hd != 0:
        raise ValueError(
            f"tensor last dim {total} not a multiple of head_dim {hd}.")
    nh = total // hd
    t3 = t.reshape(B, nh, hd)
    a = t3[..., :hd // 2]
    b = t3[..., hd // 2:]
    ra = a * cos[:, None, :] - b * sin[:, None, :]
    rb = a * sin[:, None, :] + b * cos[:, None, :]
    return jnp.concatenate([ra, rb], axis=-1).reshape(B, total)


def _pallas_rmsnorm_qkv(x, rms_w, qkv_w_q, qkv_w_scale):
    """Inner Pallas call — identical in structure to
    :func:`vllm_torchtpu.kernels.rmsnorm_qkv_fuse.kernel.rmsnorm_qkv_fuse`
    but inlined here so the fused module has no cross-package call at
    the JIT boundary."""

    def _kernel(x_ref, w_ref, q_ref, s_ref, o_ref, xn_ref):
        n_idx = pl.program_id(1)

        @pl.when(n_idx == 0)
        def _compute_norm():
            xf = x_ref[...].astype(jnp.float32)
            wf = w_ref[...].astype(jnp.float32)
            v = jnp.mean(xf * xf, axis=1, keepdims=True)
            inv = jax.lax.rsqrt(v + jnp.float32(1e-6))
            xn = xf * inv * wf[None, :]
            xn_ref[...] = xn.astype(jnp.bfloat16)

        xn_bf = xn_ref[...]
        qt = q_ref[...]
        wbf = qt.astype(jnp.bfloat16)
        out = jax.lax.dot_general(
            xn_bf,
            wbf,
            (((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32,
        )
        scale = s_ref[...].astype(jnp.float32)
        o_ref[...] = (out * scale[None, :]).astype(jnp.bfloat16)

    B, H = x.shape
    QKV, H2 = qkv_w_q.shape
    if H != H2:
        raise ValueError(f"Hidden dim mismatch: x has {H}, qkv_w_q has {H2}.")

    # Block sizes tuned by the arena. Note: for very wide H the winning
    # setting is BM=128 regardless of B (VMEM budget).
    if H >= 4096:
        BM = 128 if B >= 128 else B
    elif B >= 512:
        BM = 512
    elif B >= 256:
        BM = 256
    elif B >= 128:
        BM = 128
    else:
        BM = B
    BN = 512 if QKV >= 512 else QKV
    BK = H

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((B, QKV), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(pl.cdiv(B, BM), pl.cdiv(QKV, BN)),
            in_specs=[
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),
                pl.BlockSpec((BK, ), lambda b, n: (0, )),
                pl.BlockSpec((BN, BK), lambda b, n: (n, 0)),
                pl.BlockSpec((BN, ), lambda b, n: (n, )),
            ],
            out_specs=pl.BlockSpec((BM, BN), lambda b, n: (b, n)),
            scratch_shapes=[pltpu.VMEM((BM, BK), jnp.bfloat16)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"), ),
    )(x, rms_w, qkv_w_q, qkv_w_scale)


def rmsnorm_qkv_rope_fuse(x: jax.Array, rms_w: jax.Array, qkv_w_q: jax.Array,
                          qkv_w_scale: jax.Array, cos: jax.Array,
                          sin: jax.Array, q_dim: int, k_dim: int) -> jax.Array:
    """Fused RMSNorm + int8 QKV projection + RoPE(q,k).

    Returns ``(B, QKV)`` bf16 with RoPE applied to the ``q`` and ``k``
    slices. ``q_dim`` and ``k_dim`` specify the split; ``v_dim`` is
    inferred as ``QKV - q_dim - k_dim``.
    """
    B, H = x.shape
    QKV, H2 = qkv_w_q.shape
    if H != H2:
        raise ValueError(f"Hidden dim mismatch: x has {H}, qkv_w_q has {H2}.")
    v_dim = QKV - q_dim - k_dim
    if v_dim <= 0:
        raise ValueError(
            f"q_dim({q_dim}) + k_dim({k_dim}) >= QKV({QKV}); no room for v.")

    qkv = _pallas_rmsnorm_qkv(x, rms_w, qkv_w_q, qkv_w_scale)

    q = qkv[:, :q_dim]
    k = qkv[:, q_dim:q_dim + k_dim]
    v = qkv[:, q_dim + k_dim:]
    q_rot = _apply_rope_pair(q, cos.astype(jnp.float32),
                             sin.astype(jnp.float32)).astype(qkv.dtype)
    k_rot = _apply_rope_pair(k, cos.astype(jnp.float32),
                             sin.astype(jnp.float32)).astype(qkv.dtype)
    return jnp.concatenate([q_rot, k_rot, v], axis=-1)
