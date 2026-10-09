# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP32 Block Attention Residual for token-local K3 activations."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_COMPILER_PARAMS = pltpu.CompilerParams(vmem_limit_bytes=63 * 1024**2)


def _idx_map_2d(i):
    return (i, 0)


def _idx_map_3d(i):
    return (i, 0, 0)


def _idx_map_1d(i):
    return (0,)


def _mix_kernel(prefix, history, weight, output, *, eps):
    values = jnp.concatenate((history[...], prefix[...][:, None, :]), axis=1).astype(
        jnp.float32
    )
    inv_norm = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    scores = (
        jnp.sum(values * weight[...].astype(jnp.float32), axis=-1, keepdims=True)
        * inv_norm
    )
    probabilities = jax.nn.softmax(scores, axis=1)
    output[...] = jnp.sum(values * probabilities, axis=1).astype(output.dtype)


@functools.lru_cache(maxsize=64)
def _get_block_specs(hidden: int, num_blocks: int):
    out_spec = pl.BlockSpec((16, hidden), _idx_map_2d)
    in_specs = (
        out_spec,
        pl.BlockSpec((16, num_blocks, hidden), _idx_map_3d),
        pl.BlockSpec((hidden,), _idx_map_1d),
    )
    return in_specs, out_spec


@functools.lru_cache(maxsize=16)
def _get_mix_kernel(eps: float):
    return functools.partial(_mix_kernel, eps=eps)


def attention_residual(prefix, history, folded_weight, *, eps=1e-5, interpret=False):
    """Mix [tokens,hidden] prefix and [tokens,blocks,hidden] saved blocks.

    Norm and projection weights are folded to one FP32 vector at load time.
    All stored blocks are live; the production model carries their actual
    count in the history shape. No state or cache is modified.
    """
    tokens, hidden = prefix.shape
    if history.shape[0] != tokens or history.shape[-1] != hidden:
        raise ValueError("Attention Residual prefix/history shapes disagree")
    if folded_weight.shape != (hidden,):
        raise ValueError(
            "Attention Residual needs one folded weight per hidden feature"
        )
    num_blocks = history.shape[1]
    if num_blocks == 0 or tokens == 0:
        return prefix
    if tokens % 16:
        raise ValueError("Fused Attention Residual requires 16-token tiles")
    in_specs, out_specs = _get_block_specs(hidden, num_blocks)
    return pl.pallas_call(
        _get_mix_kernel(eps),
        out_shape=jax.ShapeDtypeStruct(prefix.shape, prefix.dtype),
        grid=(tokens // 16,),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=_COMPILER_PARAMS,
        interpret=interpret,
        name="kimi_attnres_compact_m16",
    )(prefix, history, folded_weight)
