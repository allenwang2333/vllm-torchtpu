# SPDX-License-Identifier: Apache-2.0
"""Fused (residual add) + RMSNorm + per-channel FP8 QKV+gate projection.

Targets Qwen3.5-397B-A17B-FP8's full-attention layer pre-attention path
where every non-zero layer executes:

    hidden_states, residual = input_layernorm(hidden_states, residual)
    qkv = qkv_proj(hidden_states)          # per-channel FP8

The stock path lowers to five separate XLA ops with three bf16 (M, H)
HBM intermediates. Fusing them into a single Pallas call keeps the
normalized activation in VMEM across the ``n``-loop, saves one HBM
round-trip of the intermediate and one HBM round-trip of the residual
sum vs. the current 2-op path.

Output contract:
- ``qkv_out``      : (M, OUT) bf16 = ``qkv_proj(rmsnorm(residual + x))``
- ``new_residual`` : (M, H)   bf16 = ``residual + x``

Activation numerics: RMSNorm accumulates in fp32; FP8 weights are cast
to bf16 inside VMEM before the ``dot_general``; the accumulator is fp32
and multiplied by the per-channel weight scale before the final cast to
bf16. This is the same numeric contract as the shipped
`quantized_matmul_kernel` per-channel path.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, resid_ref, rms_w_ref, w_q_ref, w_scale_ref, qkv_out_ref,
            resid_out_ref, xn_scratch):
    """Grid = (b_idx, n_idx). Prologue on ``n_idx == 0`` computes the
    fused-add + RMSNorm once per b-tile and stores the result to VMEM
    scratch; the n-loop then re-reads the normalized activation from
    scratch and dispatches an MXU dot with the FP8 weight tile.

    ``new_residual`` is written to HBM once per b-tile (also guarded on
    ``n_idx == 0``). Writes at ``n_idx > 0`` for the same b-tile would
    be idempotent (same value), but skipping them cuts HBM writes."""
    n_idx = pl.program_id(1)

    @pl.when(n_idx == 0)
    def _compute_prologue():
        x_f = x_ref[...].astype(jnp.float32)
        r_f = resid_ref[...].astype(jnp.float32)
        new_resid = x_f + r_f
        # Write updated residual back to HBM (once per b-tile).
        resid_out_ref[...] = new_resid.astype(jnp.bfloat16)
        # RMSNorm on the fused-add result.
        w_f = rms_w_ref[...].astype(jnp.float32)
        mean_sq = jnp.mean(new_resid * new_resid, axis=1, keepdims=True)
        inv = jax.lax.rsqrt(mean_sq + jnp.float32(1e-6))
        xn = new_resid * inv * w_f[None, :]
        # Persist the normalized activation for the n-loop.
        xn_scratch[...] = xn.astype(jnp.bfloat16)

    # Read the normalized activation and the FP8 weight tile, cast, dot.
    xn_bf16 = xn_scratch[...]
    w_bf16 = w_q_ref[...].astype(jnp.bfloat16)
    out = jax.lax.dot_general(
        xn_bf16,
        w_bf16,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    )
    scale = w_scale_ref[...].astype(jnp.float32)
    qkv_out_ref[...] = (out * scale[None, :]).astype(jnp.bfloat16)


def fused_add_rmsnorm_qkv_fp8(
    x: jax.Array,
    residual: jax.Array,
    rms_w: jax.Array,
    qkv_w_q: jax.Array,
    qkv_w_scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused (residual add) + RMSNorm + per-channel FP8 QKV+gate projection.

    Args:
      x: ``(M, H)`` bf16 activation.
      residual: ``(M, H)`` bf16 residual to add.
      rms_w: ``(H,)`` bf16 RMSNorm weight.
      qkv_w_q: ``(OUT, H)`` float8_e4m3fn weight.
      qkv_w_scale: ``(OUT,)`` fp32 per-channel weight scale.

    Returns:
      ``(qkv_out (M, OUT) bf16, new_residual (M, H) bf16)``.
    """
    M, H = x.shape
    OUT, H2 = qkv_w_q.shape
    if H != H2:
        raise ValueError(f"Hidden dim mismatch: x has {H}, qkv_w_q has {H2}.")
    if residual.shape != x.shape:
        raise ValueError(
            f"residual shape {residual.shape} != x shape {x.shape}.")

    # Block sizes matched to the shipped rmsnorm_qkv_fuse pattern.
    # Larger BM amortizes the RMSNorm cost across more n-tiles; BN=512
    # gives a good MXU dispatch cadence on v7 for QKV projections in
    # the range OUT ∈ [512, 8192].
    if M >= 512:
        BM = 512
    elif M >= 256:
        BM = 256
    elif M >= 128:
        BM = 128
    else:
        BM = M
    if H >= 4096:
        BM = min(BM, 128)  # VMEM budget shrinks fast at H=4096
    BN = 512 if OUT >= 512 else OUT
    BK = H  # full K per tile — Qwen3.5 H=4096 fits at BM=128

    grid = (pl.cdiv(M, BM), pl.cdiv(OUT, BN))
    out_shapes = (
        jax.ShapeDtypeStruct((M, OUT), jnp.bfloat16),
        jax.ShapeDtypeStruct((M, H), jnp.bfloat16),
    )

    return pl.pallas_call(
        _kernel,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # x
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # residual
                pl.BlockSpec((BK, ), lambda b, n: (0, )),  # rms_w
                pl.BlockSpec((BN, BK), lambda b, n: (n, 0)),  # qkv_w_q
                pl.BlockSpec((BN, ), lambda b, n: (n, )),  # qkv_w_scale
            ],
            out_specs=[
                pl.BlockSpec((BM, BN), lambda b, n: (b, n)),  # qkv_out
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # new_residual
            ],
            scratch_shapes=[pltpu.VMEM((BM, BK), jnp.bfloat16)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"), ),
    )(x, residual, rms_w, qkv_w_q, qkv_w_scale)
