# SPDX-License-Identifier: Apache-2.0
"""Fused (residual add) + RMSNorm Pallas kernel.

Standalone version of the fused-add-then-normalize pattern that fires
120× per Qwen3.5 decode step (60 layers × 2 norm sites per layer:
``input_layernorm`` and ``post_attention_layernorm``). The vLLM
``fused_add_rms_norm`` custom op runs on the ``native`` (XLA) path on
TPU today; a Pallas alternative keeps the fused-add result in VMEM
across the mean-square reduction, eliminating one bf16 (M, H) HBM
round-trip per invocation.

Semantics match vLLM's ``RMSNorm(hidden_states, residual)`` returning
``(normed, new_residual)``:

    new_residual = residual + hidden_states     # bf16 elementwise
    normed = RMSNorm(new_residual, weight, eps)  # rmsnorm the sum

Output contract:
- ``normed``       : (M, H) bf16
- ``new_residual`` : (M, H) bf16
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, resid_ref, w_ref, normed_ref, resid_out_ref, *, eps):
    """Single-tile kernel: compute new_resid + rmsnorm in one pass."""
    x_f = x_ref[...].astype(jnp.float32)
    r_f = resid_ref[...].astype(jnp.float32)
    new_resid = x_f + r_f
    resid_out_ref[...] = new_resid.astype(jnp.bfloat16)

    w_f = w_ref[...].astype(jnp.float32)
    mean_sq = jnp.mean(new_resid * new_resid, axis=1, keepdims=True)
    inv = jax.lax.rsqrt(mean_sq + jnp.float32(eps))
    normed = new_resid * inv * w_f[None, :]
    normed_ref[...] = normed.astype(jnp.bfloat16)


def fused_add_rmsnorm(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Fused residual-add + RMSNorm.

    Args:
      x:        ``(M, H)`` bf16 activation.
      residual: ``(M, H)`` bf16 residual to add.
      weight:   ``(H,)``   bf16 RMSNorm weight.
      eps:      RMSNorm variance epsilon.

    Returns:
      ``(normed (M, H) bf16, new_residual (M, H) bf16)``.
    """
    M, H = x.shape
    if residual.shape != x.shape:
        raise ValueError(
            f"residual shape {residual.shape} != x shape {x.shape}.")
    if weight.shape != (H, ):
        raise ValueError(
            f"weight shape {weight.shape} != ({H},).")

    # Block sizes: full-row VMEM buffer is BM * H * 2 (bf16) + fp32 acc.
    # For H=4096, BM=128 → ~512KB tile → fits comfortably in VMEM.
    if M >= 512:
        BM = 512
    elif M >= 256:
        BM = 256
    elif M >= 128:
        BM = 128
    else:
        BM = M
    # Shrink BM when H is very wide.
    if H >= 4096:
        BM = min(BM, 128)
    elif H >= 8192:
        BM = min(BM, 64)

    grid = (pl.cdiv(M, BM), )

    return pl.pallas_call(
        lambda x_r, r_r, w_r, n_o, r_o: _kernel(x_r, r_r, w_r, n_o, r_o,
                                                 eps=eps),
        out_shape=(
            jax.ShapeDtypeStruct((M, H), jnp.bfloat16),
            jax.ShapeDtypeStruct((M, H), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((BM, H), lambda b: (b, 0)),  # x
                pl.BlockSpec((BM, H), lambda b: (b, 0)),  # residual
                pl.BlockSpec((H, ), lambda b: (0, )),      # weight
            ],
            out_specs=[
                pl.BlockSpec((BM, H), lambda b: (b, 0)),  # normed
                pl.BlockSpec((BM, H), lambda b: (b, 0)),  # new_residual
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ), ),
    )(x, residual, weight)
