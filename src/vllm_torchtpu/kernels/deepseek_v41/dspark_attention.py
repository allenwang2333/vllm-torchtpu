# SPDX-License-Identifier: Apache-2.0
"""DSpark request-shared SWA online softmax in one Pallas call."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as tpu


def _body(qref, kvref, validref, sinkref, outref, *, scale, block, heads):
    q = qref[0]
    keys = kvref.shape[1]
    maximum = jnp.full((q.shape[0],), -1e30, jnp.float32)
    denominator = jnp.zeros((q.shape[0],), jnp.float32)
    numerator = jnp.zeros(q.shape, jnp.float32)
    for tile in range(keys // 64):
        kv = kvref[0, pl.ds(tile * 64, 64), :]
        valid = validref[0, pl.ds(tile * 64, 64)]
        scores = jnp.dot(q, kv.T, preferred_element_type=jnp.float32) * scale
        scores = jnp.where(valid[None, :], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        probability = jnp.exp(scores - new_max[:, None])
        denominator = denominator * correction + probability.sum(-1)
        numerator = numerator * correction[:, None] + jnp.dot(
            probability.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        maximum = new_max
    sink = sinkref[...]
    denominator += jnp.exp(sink - maximum)
    outref[0] = (numerator / denominator[:, None]).astype(jnp.bfloat16)


def fused_attention(q, kv, valid, sink, scale, *, block, interpret=False):
    tokens, heads, dim = q.shape
    if block <= 1 or tokens % block or dim != 512 or q.dtype != jnp.bfloat16:
        raise ValueError("Fused draft attention requires BF16 whole blocks, D=512")
    if kv.shape != (tokens // block, valid.shape[1], dim) or kv.shape[1] % 64:
        raise ValueError("KV must contain one aligned 64-key window per request")
    if not kv.shape[1]:
        return jnp.zeros_like(q)
    kv = jnp.where(valid[..., None], kv, 0)
    requests = tokens // block
    q = q.reshape(requests, block * heads, dim)
    qspec = pl.BlockSpec((1, block * heads, dim), lambda r: (r, 0, 0))
    result = pl.pallas_call(
        functools.partial(_body, scale=scale, block=block, heads=heads),
        grid=(requests,),
        in_specs=(
            qspec,
            pl.BlockSpec((1, kv.shape[1], dim), lambda r: (r, 0, 0)),
            pl.BlockSpec((None, 1, valid.shape[1]), lambda r: (r, 0, 0)),
            pl.BlockSpec((block * heads,), lambda r: (0,)),
        ),
        out_specs=qspec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="dspark_request_swa_online",
    )(q, kv, valid[:, None, :], jnp.tile(sink, block))
    return result.reshape(tokens, heads, dim)
