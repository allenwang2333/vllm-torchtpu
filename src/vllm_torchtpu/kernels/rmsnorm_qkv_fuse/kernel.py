# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm → per-channel int8 QKV projection.

Originally discovered by the kernel_agents arena (run
``20260701-031641-d1c1b6ed``) as a 1.055× TPOT-geomean win over the
XLA-fused baseline on Qwen-shape workloads. See
``docs/kernel_agents.md`` for the discovery methodology.

Input contract:
- ``x``           : (B, H)         bf16   activation (post-residual)
- ``rms_w``       : (H,)           bf16   RMSNorm weight
- ``qkv_w_q``     : (QKV, H)       int8   per-channel-quantized weight
- ``qkv_w_scale`` : (QKV,)         fp32   per-output-row scale

Output:
- ``(B, QKV)``    bf16 = ``matmul(rmsnorm(x, rms_w), qkv_w_q) * qkv_w_scale``

Numerics: RMSNorm accumulates in fp32; matmul accumulates in fp32; final
cast to bf16 on store. Correctness is verified in
``tests/kernels/test_rmsnorm_qkv_fuse_kernel.py`` against the
composition of ``vllm_torchtpu.kernels.quantized_matmul.util.
xla_quantized_matmul`` and the JAX-native RMSNorm reference.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, w_ref, q_ref, s_ref, o_ref, xn_ref):
    """Pallas kernel. Grid is (b_idx, n_idx); RMSNorm runs once per b_idx
    (n_idx==0) and the normalized tile is reused across the n-loop via a
    VMEM scratch buffer."""
    n_idx = pl.program_id(1)

    @pl.when(n_idx == 0)
    def _compute_norm():
        x_tile = x_ref[...].astype(jnp.float32)
        w_tile = w_ref[...].astype(jnp.float32)
        mean_sq = jnp.mean(x_tile * x_tile, axis=1, keepdims=True)
        inv = jax.lax.rsqrt(mean_sq + jnp.float32(1e-6))
        x_norm = x_tile * inv * w_tile[None, :]
        xn_ref[...] = x_norm.astype(jnp.bfloat16)

    x_norm_bf16 = xn_ref[...]
    q_tile = q_ref[...]
    w_bf16 = q_tile.astype(jnp.bfloat16)
    out = jax.lax.dot_general(
        x_norm_bf16,
        w_bf16,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    )
    scale = s_ref[...].astype(jnp.float32)
    result = out * scale[None, :]
    o_ref[...] = result.astype(jnp.bfloat16)


def rmsnorm_qkv_fuse(x: jax.Array, rms_w: jax.Array, qkv_w_q: jax.Array,
                     qkv_w_scale: jax.Array) -> jax.Array:
    """Fused RMSNorm + int8 QKV projection.

    ``x``: (B, H) bf16. ``rms_w``: (H,) bf16. ``qkv_w_q``: (QKV, H) int8.
    ``qkv_w_scale``: (QKV,) fp32. Returns (B, QKV) bf16.
    """
    B, H = x.shape
    QKV, H2 = qkv_w_q.shape
    if H != H2:
        raise ValueError(f"Hidden dim mismatch: x has {H}, qkv_w_q has {H2}.")

    # Block sizes tuned by the arena (see task docstring above). Larger
    # BM amortizes the RMSNorm cost across more output columns and
    # reduces total x HBM re-fetches.
    if B >= 512:
        BM = 512
    elif B >= 256:
        BM = 256
    elif B >= 128:
        BM = 128
    else:
        BM = B
    BN = 512 if QKV >= 512 else QKV
    BK = H

    grid = (pl.cdiv(B, BM), pl.cdiv(QKV, BN))
    out_shape = jax.ShapeDtypeStruct((B, QKV), jnp.bfloat16)

    return pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
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
