# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attention over gathered windows in a shared word cache."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(q_ref, words_ref, mask_ref, max_ref, den_ref, num_ref, *, scale):
    queries, heads, dim = q_ref.shape
    chunks, _, keys, _ = words_ref.shape
    # The persistent owner mixes record types. Decode only this gathered tile
    # as [..., 2, width] for BF16 ([..., 4, width] for byte storage), keeping
    # packing and permutation within VMEM instead of materializing HBM copies.
    words = jnp.concatenate([words_ref[c, 0] for c in range(chunks)], axis=1)
    words = words.reshape(keys, 1, dim // 2)
    packed = pltpu.bitcast(words, q_ref.dtype)
    kv = packed.transpose(0, 2, 1).reshape(keys, dim)
    valid = mask_ref[0]
    kv = jnp.where(valid.any(axis=0)[:, None], kv, 0)
    q = q_ref[...].reshape(queries * heads, dim)
    scores = (
        jax.lax.dot_general(
            q, kv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ).reshape(queries, heads, keys)
        * scale
    )
    mask = valid[:, None, :]
    scores = jnp.where(mask, scores, -jnp.inf)
    maximum = jnp.maximum(scores.max(axis=-1), -1e30)
    probabilities = jnp.exp(scores - maximum[..., None])
    denominator = probabilities.sum(axis=-1)
    numerator = jax.lax.dot_general(
        probabilities.reshape(queries * heads, keys).astype(q_ref.dtype),
        kv,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).reshape(queries, heads, dim)
    max_ref[0, 0, 0] = maximum.reshape(-1)
    den_ref[0, 0, 0] = denominator.reshape(-1)
    num_ref[...] = numerator


@functools.partial(jax.jit, static_argnames=("scale", "interpret"))
def attend_windows(
    q: jax.Array,
    words: jax.Array,
    mask: jax.Array,
    *,
    scale: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return FP32 max, denominator and numerator for independent windows.

    ``words[dim // 256, groups, keys, 128]`` contains consecutive BF16 element
    pairs in each int32 word. Each group owns consecutive queries from
    ``q[tokens, heads, dim]``; ``mask[groups, queries, keys]`` defines their
    exact windows, including request boundaries and padded tokens.
    """
    tokens, heads, dim = q.shape
    groups, queries, keys = mask.shape
    if (
        q.dtype != jnp.bfloat16
        or words.dtype != jnp.int32
        or dim % 256
        or words.shape != (dim // 256, groups, keys, 128)
        or tokens != groups * queries
        or heads % min(heads, 8)
        or keys % 128
    ):
        raise ValueError("Expected BF16 queries and aligned, packed window words")
    head_block = min(heads, 8)
    head_tiles = heads // head_block
    stats_shape = (groups, head_tiles, 1, queries * head_block)
    maximum, denominator, numerator = pl.pallas_call(
        functools.partial(_kernel, scale=scale),
        grid=(groups, head_tiles),
        in_specs=[
            pl.BlockSpec((queries, head_block, dim), lambda g, h: (g, h, 0)),
            pl.BlockSpec((dim // 256, 1, keys, 128), lambda g, h: (0, g, 0, 0)),
            pl.BlockSpec((1, queries, keys), lambda g, h: (g, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, 1, queries * head_block), lambda g, h: (g, h, 0, 0)),
            pl.BlockSpec((1, 1, 1, queries * head_block), lambda g, h: (g, h, 0, 0)),
            pl.BlockSpec((queries, head_block, dim), lambda g, h: (g, h, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(stats_shape, jnp.float32),
            jax.ShapeDtypeStruct(stats_shape, jnp.float32),
            jax.ShapeDtypeStruct(q.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="swa_window_attention",
    )(q, words, mask)

    def stats(x):
        return (
            x.reshape(groups, head_tiles, queries, head_block)
            .transpose(0, 2, 1, 3)
            .reshape(tokens, heads)
        )

    return stats(maximum), stats(denominator), numerator
