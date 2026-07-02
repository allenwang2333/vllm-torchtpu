# SPDX-License-Identifier: Apache-2.0
"""Fused (residual add) + RMSNorm + MoE router projection.

Targets Qwen3.5-397B-A17B-FP8's post-attention path where every layer
runs:

    hidden_states, residual = post_attention_layernorm(hidden_states, residual)
    router_logits, _ = self.gate(hidden_states)      # bf16 matmul, replicated

The stock path lowers to 4 separate ops (add, mean-sq reduce,
multiply-by-scale, dense matmul) with two bf16 (M, H) HBM intermediates
plus a (M, num_experts) router output. Fusing them into a single Pallas
call keeps the normalized activation in VMEM across the router matmul,
saving one HBM round-trip of the (M, H) intermediate per invocation.

Three outputs (all needed downstream):
- ``router_logits`` : (M, num_experts) bf16 → topk / softmax
- ``normed``        : (M, H)   bf16 → shared_expert + fused MoE
- ``new_residual``  : (M, H)   bf16 → next-layer residual chain

The router weight is small (``num_experts * H`` ≈ 2 MB for Qwen3.5;
num_experts=512, H=4096) and stays in bf16 — no quantization needed.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, resid_ref, rms_w_ref, router_w_ref, router_out_ref,
            normed_ref, resid_out_ref, xn_scratch, *, eps):
    """Grid = (b_idx, n_idx). Prologue on n_idx==0 computes the fused-
    add + RMSNorm once per b-tile and stashes the normalized activation
    into VMEM scratch; the n-loop re-reads the scratch and dispatches
    an MXU dot with the router weight tile."""
    n_idx = pl.program_id(1)

    @pl.when(n_idx == 0)
    def _compute_prologue():
        x_f = x_ref[...].astype(jnp.float32)
        r_f = resid_ref[...].astype(jnp.float32)
        new_resid = x_f + r_f
        # Persist updated residual to HBM once per b-tile.
        resid_out_ref[...] = new_resid.astype(jnp.bfloat16)
        # RMSNorm.
        w_f = rms_w_ref[...].astype(jnp.float32)
        mean_sq = jnp.mean(new_resid * new_resid, axis=1, keepdims=True)
        inv = jax.lax.rsqrt(mean_sq + jnp.float32(eps))
        xn = new_resid * inv * w_f[None, :]
        xn_bf16 = xn.astype(jnp.bfloat16)
        # Persist normed output to HBM AND to VMEM scratch for the
        # n-loop.
        normed_ref[...] = xn_bf16
        xn_scratch[...] = xn_bf16

    # Router matmul: bf16 x bf16 -> fp32 -> bf16.
    xn_bf16 = xn_scratch[...]
    w_bf16 = router_w_ref[...]  # (BN, H) bf16
    out = jax.lax.dot_general(
        xn_bf16,
        w_bf16,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    )
    router_out_ref[...] = out.astype(jnp.bfloat16)


def fused_add_rmsnorm_router(
    x: jax.Array,
    residual: jax.Array,
    rms_weight: jax.Array,
    router_weight: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused (residual add) + RMSNorm + MoE router bf16 matmul.

    Args:
      x:              ``(M, H)`` bf16 activation.
      residual:       ``(M, H)`` bf16 residual to add.
      rms_weight:     ``(H,)``   bf16 RMSNorm weight.
      router_weight:  ``(num_experts, H)`` bf16 gate weight.
      eps:            RMSNorm variance epsilon.

    Returns:
      ``(router_logits (M, num_experts) bf16,
         normed (M, H) bf16,
         new_residual (M, H) bf16)``.
    """
    M, H = x.shape
    NE, H2 = router_weight.shape
    if H != H2:
        raise ValueError(
            f"Hidden dim mismatch: x has {H}, router_weight has {H2}.")
    if residual.shape != x.shape:
        raise ValueError(
            f"residual shape {residual.shape} != x shape {x.shape}.")
    if rms_weight.shape != (H, ):
        raise ValueError(
            f"rms_weight shape {rms_weight.shape} != ({H},).")

    # Block sizes.
    if M >= 512:
        BM = 512
    elif M >= 256:
        BM = 256
    elif M >= 128:
        BM = 128
    else:
        BM = M
    if H >= 4096:
        BM = min(BM, 128)
    # BN: pick a block that divides num_experts. Qwen3.5 has 512
    # experts; 512 is a natural full tile.
    if NE >= 512 and NE % 512 == 0:
        BN = 512
    elif NE >= 256 and NE % 256 == 0:
        BN = 256
    elif NE >= 128 and NE % 128 == 0:
        BN = 128
    else:
        BN = NE
    BK = H

    grid = (pl.cdiv(M, BM), pl.cdiv(NE, BN))

    return pl.pallas_call(
        lambda *args: _kernel(*args, eps=eps),
        out_shape=(
            jax.ShapeDtypeStruct((M, NE), jnp.bfloat16),
            jax.ShapeDtypeStruct((M, H), jnp.bfloat16),
            jax.ShapeDtypeStruct((M, H), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # x
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # residual
                pl.BlockSpec((BK, ), lambda b, n: (0, )),      # rms_w
                pl.BlockSpec((BN, BK), lambda b, n: (n, 0)),   # router_w
            ],
            out_specs=[
                pl.BlockSpec((BM, BN), lambda b, n: (b, n)),  # router_out
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # normed
                pl.BlockSpec((BM, BK), lambda b, n: (b, 0)),  # new_resid
            ],
            scratch_shapes=[pltpu.VMEM((BM, BK), jnp.bfloat16)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"), ),
    )(x, residual, rms_weight, router_weight)
