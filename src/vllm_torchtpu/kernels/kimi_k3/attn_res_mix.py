# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused K3 AttentionResidual (hyper-connection) mix kernel.

Replaces the cat -> RMSNorm -> rank-1 scorer -> fp32 softmax -> weighted-sum
chain with one Pallas kernel per stack width. RMSNorm followed by a rank-1
projection contracts to a weighted dot:

    score(x) = <RMSNorm_w(x), w_proj>
             = <x, w_norm * w_proj> * rsqrt(mean(x^2) + eps)

so the normed [T, n+1, H] stack is never materialized. The default path keeps
fp32 through the score — strictly tighter than the unfused path, which rounds
the normed stack to bf16 before the dot. ``exact_emulation`` reproduces the
unfused path's bf16 roundings for debugging-grade comparison.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# The serving layout assigner tiles the bf16 output (16, 128) (Mosaic
# E2002 "CompileTimeMosaicMisalignedBlockAndTiling" if a block neither
# divides that tiling nor matches the full shape) — so the token-dim block
# must be min(T, 16): below 16 the block matches the full (padded) shape,
# at/above 16 it divides the (16, 128) tiling evenly. A whole decode bucket
# still fits one grid step at T <= 16; at n+1 = 9 a step holds ~2.1 MB of
# bf16 rows, far under the VMEM budget at every prefill T too.
BLOCK_T = 16


def _mix_kernel(prefix_ref, blocks_ref, w_score_ref, w_norm_ref, w_proj_ref,
                out_ref, *, n: int, eps: float, exact_emulation: bool):
    rows = [blocks_ref[:, i, :] for i in range(n)] + [prefix_ref[...]]
    w_score = w_score_ref[0, :]
    scores = []
    rows_f32 = []
    for x in rows:
        x_f32 = x.astype(jnp.float32)
        rows_f32.append(x_f32)
        inv_rms = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1) + eps)
        if exact_emulation:
            normed = (x_f32 * inv_rms[:, None]).astype(jnp.bfloat16)
            normed = normed * w_norm_ref[0, :]
            score = jnp.sum(normed.astype(jnp.float32) *
                            w_proj_ref[0, :].astype(jnp.float32),
                            axis=-1).astype(jnp.bfloat16).astype(jnp.float32)
        else:
            score = jnp.sum(x_f32 * w_score, axis=-1) * inv_rms
        scores.append(score)
    probs = jax.nn.softmax(jnp.stack(scores, axis=-1), axis=-1)
    mix = rows_f32[0] * probs[:, 0][:, None]
    for i in range(1, n + 1):
        mix = mix + rows_f32[i] * probs[:, i][:, None]
    out_ref[...] = mix.astype(out_ref.dtype)


def attn_res_mix(prefix_sum: jax.Array, block_residuals: jax.Array,
                 w_score: jax.Array, w_norm: jax.Array, w_proj: jax.Array, *,
                 n: int, eps: float, exact_emulation: bool) -> jax.Array:
    num_tokens, hidden = prefix_sum.shape
    assert block_residuals.shape == (num_tokens, n, hidden)
    assert hidden % 128 == 0
    # Below 16 the block matches the full (padded) token dim; at/above 16 it
    # must divide the bf16 output's native (16, 128) tiling evenly (Mosaic
    # E2002 otherwise) — see the BLOCK_T comment above.
    block_t = num_tokens if num_tokens < 16 else BLOCK_T
    kernel = functools.partial(_mix_kernel,
                               n=n,
                               eps=eps,
                               exact_emulation=exact_emulation)
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_tokens, block_t), ),
        in_specs=[
            pl.BlockSpec((block_t, hidden), lambda i: (i, 0)),
            pl.BlockSpec((block_t, n, hidden), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, hidden), lambda i: (0, 0)),
            pl.BlockSpec((1, hidden), lambda i: (0, 0)),
            pl.BlockSpec((1, hidden), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_t, hidden), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), prefix_sum.dtype),
        name=f"k3_attn_res_mix_n{n}",
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=int(0.9 *
                                 pltpu.get_tpu_info().vmem_capacity_bytes),
        ),
    )(prefix_sum, block_residuals, w_score, w_norm, w_proj)
