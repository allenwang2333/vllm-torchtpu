# SPDX-License-Identifier: Apache-2.0
"""Packed FP8 Engram rows through projection and gating in one Pallas call.

The caller gathers packed rows across TP before this kernel. CPU lookup and
H2D remain outside it. Only this rank's SP tokens are read from each head
shard; dequantized embeddings and projected keys never leave VMEM.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.engram.tuned_block_sizes import get_block_sizes
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def unpack_words(words: jax.Array) -> jax.Array:
    shifts = jnp.arange(4, dtype=jnp.int32) * 8
    return (
        ((words[..., None] >> shifts) & 255)
        .astype(jnp.uint8)
        .reshape(*words.shape[:-1], words.shape[-1] * 4)
    )


def dequantize_rows(raw: jax.Array, scales: jax.Array) -> jax.Array:
    values = jax.lax.bitcast_convert_type(unpack_words(raw), jnp.float8_e4m3fn)
    scale_bytes = unpack_words(scales)
    block = values.shape[-1] // scale_bytes.shape[-1]
    values = values.reshape(*scale_bytes.shape, block).astype(jnp.float32)
    power = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.int32) << 23, jnp.float32
    )
    values = jnp.where(scale_bytes[..., None] == 255, 0.0, values * power[..., None])
    return values.reshape(*raw.shape[:-1], raw.shape[-1] * 4).astype(jnp.bfloat16)


def gated_residual(
    hidden: jax.Array,
    projection: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    eps: float,
    clamp_value: float,
) -> jax.Array:
    streams, dim = hidden.shape[-2:]
    keys = projection[:, : streams * dim].reshape(-1, streams, dim).astype(jnp.float32)
    values = projection[:, streams * dim :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = jnp.sum(h * query_weight * key_weight * keys, axis=-1)
    dot *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + eps)
    dot *= jax.lax.rsqrt(jnp.mean(keys * keys, axis=-1) + eps) * dim**-0.5
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp_value))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude))
    gate *= token_mask[:, None]
    return (h + gate[..., None] * values[:, None, :]).astype(hidden.dtype)


def fused_engram(
    raw: jax.Array,
    scales: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    rank: int,
    eps: float,
    clamp_value: float,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Fuse packed-row dequant, FP8 WKV, normalization, gating and residual.

    Supports DP1 with SP tokens, equally sized head shards, per-channel FP8
    projection scales, and four-byte-aligned scale rows. Inputs are already
    padded by the runner. The BF16 roundings before and after WKV match the
    unfused Engram, including invalid embedding scales (byte 255).
    """
    tp, full_tokens, raw_words = raw.shape
    tokens, streams, dim = hidden.shape
    in_features = tp * raw_words * 4
    out_features = (streams + 1) * dim
    assert full_tokens == tokens * tp, (full_tokens, tokens, tp)
    assert 0 <= rank < tp, (rank, tp)
    assert scales.shape[:2] == raw.shape[:2]
    assert raw_words % scales.shape[-1] == 0
    assert weight.shape == (in_features, out_features), weight.shape
    assert weight_scale.shape == (out_features,), weight_scale.shape
    assert query_weight.shape == key_weight.shape == (streams, dim)
    assert token_mask.shape == (tokens,), token_mask.shape
    assert hidden.dtype == jnp.bfloat16, hidden.dtype
    assert weight.dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2), weight.dtype
    tile_m, tile_k = get_block_sizes(tokens, in_features, streams, dim)
    assert tokens % tile_m == 0 and in_features % tile_k == 0
    steps = in_features // tile_k

    def kernel(
        raw_ref,
        scale_ref,
        h_ref,
        w_ref,
        ws_ref,
        q_ref,
        k_ref,
        mask_ref,
        out_ref,
        x_scratch,
        xs_scratch,
        acc_scratch,
    ):
        step = pl.program_id(1)

        @pl.when(step == 0)
        def prepare():
            rows = dequantize_rows(raw_ref[...], scale_ref[...])
            embedded = rows.transpose(1, 0, 2).reshape(tile_m, in_features)
            quantized, activation_scale = quantize_tensor(embedded, weight.dtype)
            x_scratch[...] = quantized
            xs_scratch[...] = activation_scale
            acc_scratch[...] = jnp.zeros((tile_m, out_features), jnp.float32)

        x = x_scratch[:, pl.ds(step * tile_k, tile_k)]
        acc_scratch[...] += jnp.dot(x, w_ref[...], preferred_element_type=jnp.float32)

        @pl.when(step == steps - 1)
        def finish():
            projection = (acc_scratch[...] * xs_scratch[...] * ws_ref[...]).astype(
                hidden.dtype
            )
            out_ref[...] = gated_residual(
                h_ref[...],
                projection,
                q_ref[...],
                k_ref[...],
                mask_ref[:, 0],
                eps,
                clamp_value,
            )

    def token_block(i, k):
        return 0, i + rank * (tokens // tile_m), 0

    def local_block(i, k):
        return i, 0, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile_m, steps),
        in_specs=[
            pl.BlockSpec((tp, tile_m, raw_words), token_block),
            pl.BlockSpec((tp, tile_m, scales.shape[-1]), token_block),
            pl.BlockSpec((tile_m, streams, dim), local_block),
            pl.BlockSpec((tile_k, out_features), lambda i, k: (k, 0)),
            pl.BlockSpec(weight_scale.shape, lambda i, k: (0,)),
            pl.BlockSpec(query_weight.shape, lambda i, k: (0, 0)),
            pl.BlockSpec(key_weight.shape, lambda i, k: (0, 0)),
            pl.BlockSpec((tile_m, 1), lambda i, k: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tile_m, streams, dim), local_block),
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        scratch_shapes=[
            pltpu.VMEM((tile_m, in_features), weight.dtype),
            pltpu.VMEM((tile_m, 1), jnp.float32),
            pltpu.VMEM((tile_m, out_features), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=interpret,
        name="engram_offload_fused",
    )(
        raw,
        scales,
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask.reshape(tokens, 1),
    )
