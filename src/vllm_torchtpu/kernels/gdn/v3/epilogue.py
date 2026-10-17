# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Output epilogue of the pooled GDN op: the gated RMSNorm and the output
projection's per-token activation quantization, written token-major.

The GDN launches write their output a sequence at a time, [tokens, heads,
d_v] with heads on sublanes, because a token-major tile is shared by several
sequences. This launch walks whole token blocks instead: it reads the output
and z (token-major in the qkvz row tiles), gates and norms each head, and
writes [tokens, heads * d_v] quantized with `quantize_tensor`'s arithmetic,
plus the per-token scale, which the projection then consumes as they are.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config

# Tokens per grid step.
EPILOGUE_TOKEN_BLOCK = 128
# Tokens the per-head work covers at once: the rows of a packed 8-bit tile.
EPILOGUE_ROW_GROUP = 32
# Heads swapped to token-major at once: the rows of a packed 16-bit tile.
EPILOGUE_HEAD_CHUNK = 16


def supports(num_tokens: int, num_heads: int, head_dim: int) -> bool:
    """Whether `gate_and_quantize` takes this shape."""
    block = min(EPILOGUE_TOKEN_BLOCK, num_tokens)
    row_group = min(EPILOGUE_ROW_GROUP, block)
    return (
        head_dim == pltpu.get_tpu_info().num_lanes
        and num_heads % EPILOGUE_HEAD_CHUNK == 0
        and num_tokens % block == 0
        and block % row_group == 0
        and row_group % config.QKV_TILE_ROWS == 0
    )


def _kernel(*refs, gate: bool, eps: float, quant_dtype: jnp.dtype, row_group: int):
    if gate:
        out_ref, z_ref, norm_ref, q_ref, scale_ref, gated_ref = refs
    else:
        out_ref, q_ref, scale_ref, gated_ref = refs
    block, num_heads, head_dim = out_ref.shape
    q_max = float(jnp.finfo(quant_dtype).max)
    # `quantize_block`'s `abs_max / dtype_max`, as XLA folds it: a multiply
    # by the f32 reciprocal.
    inv_q_max = np.float32(1) / np.float32(q_max)
    tile_rows = config.QKV_TILE_ROWS

    for t0 in range(0, block, row_group):
        rows = pl.ds(t0, row_group)

        def gate_heads(c, abs_max, t0=t0, rows=rows):
            h0 = pl.multiple_of(c * EPILOGUE_HEAD_CHUNK, EPILOGUE_HEAD_CHUNK)
            heads = pl.ds(h0, EPILOGUE_HEAD_CHUNK)
            # [tokens, heads, d_v] -> [heads, tokens, d_v]
            x = jnp.swapaxes(out_ref[rows, heads].astype(jnp.float32), 0, 1)
            if gate:
                z = z_ref[pl.ds(t0 // tile_rows, row_group // tile_rows), heads]
                z = jnp.swapaxes(z, 0, 1).reshape(x.shape)
                x = compute_gdn.gated_rms_norm(
                    x, z, norm_ref[...], eps, out_dtype=out_ref.dtype
                ).astype(jnp.float32)
            gated_ref[heads, rows] = x
            return jnp.maximum(abs_max, jnp.max(jnp.abs(x), axis=0))

        abs_max = jax.lax.fori_loop(
            0,
            num_heads // EPILOGUE_HEAD_CHUNK,
            gate_heads,
            jnp.zeros((row_group, head_dim), jnp.float32),
            unroll=True,
        )
        scale = jnp.max(abs_max, axis=-1, keepdims=True) * inv_q_max
        scale = jnp.where(scale == 0, 1.0, scale)
        scale_ref[rows] = scale

        @pl.loop(0, num_heads, unroll=EPILOGUE_HEAD_CHUNK)
        def _(h, rows=rows, scale=scale):
            q = jnp.clip(gated_ref[h, rows] / scale, -q_max, q_max)
            lanes = pl.ds(pl.multiple_of(h * head_dim, head_dim), head_dim)
            q_ref[rows, lanes] = q.astype(quant_dtype)


def gate_and_quantize(
    out: jax.Array,
    z_tiles: jax.Array | None,
    norm_weight: jax.Array | None,
    *,
    z_group: int = 0,
    eps: float = 0.0,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """Quantizes the GDN output per token, gating it first when given z.

    Args:
        out: [tokens, heads, d_v] 16-bit GDN output.
        z_tiles: The qkvz row tiles, [tokens / QKV_TILE_ROWS, columns /
            lanes, QKV_TILE_ROWS, lanes], whose `heads` lane groups from
            `z_group` on are z; then `out` is replaced by
            `compute_gdn.gated_rms_norm(out, z, norm_weight, eps)` per head
            before it is quantized. None if `out` is already gated.
        norm_weight: [d_v] f32 norm weight, with `z_tiles`.
        quant_dtype: The activation dtype the projection consumes.

    Returns:
        `quantize_tensor(out.reshape(tokens, -1), quant_dtype)`: the
        [tokens, heads * d_v] activation and its [tokens, 1] f32 scale.
    """
    num_tokens, num_heads, head_dim = out.shape
    assert supports(num_tokens, num_heads, head_dim), out.shape
    block = min(EPILOGUE_TOKEN_BLOCK, num_tokens)
    row_group = min(EPILOGUE_ROW_GROUP, block)
    gate = z_tiles is not None

    in_specs = [pl.BlockSpec((block, num_heads, head_dim), lambda i: (i, 0, 0))]
    operands = [out]
    if gate:
        tile_rows = config.QKV_TILE_ROWS
        in_specs += [
            # z's lane groups start mid-block, so this block is indexed by
            # element offsets.
            pl.BlockSpec(
                (
                    pl.Element(block // tile_rows),
                    pl.Element(num_heads),
                    pl.Element(tile_rows),
                    pl.Element(head_dim),
                ),
                lambda i: (i * (block // tile_rows), z_group, 0, 0),
            ),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ]
        operands += [z_tiles, norm_weight.astype(jnp.float32).reshape(1, head_dim)]

    return pl.pallas_call(
        functools.partial(
            _kernel, gate=gate, eps=eps, quant_dtype=quant_dtype, row_group=row_group
        ),
        grid=(num_tokens // block,),
        in_specs=in_specs,
        out_specs=(
            pl.BlockSpec((block, num_heads * head_dim), lambda i: (i, 0)),
            pl.BlockSpec((block, 1), lambda i: (i, 0)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), quant_dtype),
            jax.ShapeDtypeStruct((num_tokens, 1), jnp.float32),
        ),
        scratch_shapes=(pltpu.VMEM((num_heads, block, head_dim), jnp.float32),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="gdn_gate_and_quantize",
    )(*operands)
