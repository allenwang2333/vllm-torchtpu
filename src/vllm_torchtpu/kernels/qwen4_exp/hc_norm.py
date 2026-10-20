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
"""Optional prefill HC branch normalization with one BF16 output boundary."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _norm_kernel(input_ref, weight_ref, output_ref, *, hidden, branches, eps):
    values = input_ref[...].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)
    pieces = []
    for branch in range(branches):
        start = branch * hidden
        segment = values[:, start : start + hidden]
        variance = jnp.mean(segment * segment, axis=-1, keepdims=True)
        # Match upstream grouped Gemma RMSNorm: round only the stored output.
        normalized = segment * jax.lax.rsqrt(variance + eps)
        scaled = normalized * weight[None, start : start + hidden]
        pieces.append(normalized + scaled)
    output_ref[...] = jnp.concatenate(pieces, axis=-1).astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("branches", "hidden", "eps", "token_tile"))
def _normalize(input_tensor, weight, *, branches, hidden, eps, token_tile):
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError("HC norm requires positive branches and 128-aligned hidden")
    # This vector-only kernel also supports the model's 64-row norm tiles.
    if token_tile <= 0 or token_tile % 64:
        raise ValueError("HC norm requires positive 64-aligned token tiles")
    tokens, width = input_tensor.shape
    if (
        tokens == 0
        or tokens % token_tile
        or width != branches * hidden
        or weight.shape != (width,)
        or input_tensor.dtype != jnp.bfloat16
        or weight.dtype != jnp.bfloat16
    ):
        raise ValueError("HC norm requires BF16 and whole token tiles")
    return pl.pallas_call(
        functools.partial(_norm_kernel, hidden=hidden, branches=branches, eps=eps),
        grid=(tokens // token_tile,),
        in_specs=[
            pl.BlockSpec((token_tile, width), lambda tile: (tile, 0)),
            pl.BlockSpec((width,), lambda tile: (0,)),
        ],
        out_specs=pl.BlockSpec((token_tile, width), lambda tile: (tile, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=60 << 20
        ),
        name=f"qwen4exp_hc_norm_m{token_tile}",
    )(input_tensor, weight)
