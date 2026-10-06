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
"""(Residual add +) Gemma RMS norm and per-row fp8 quantization in one pass.

Mirrors ``GemmaRMSNorm`` (``ir.ops.rms_norm`` / ``fused_add_rms_norm`` with the
f32 weight ``w + 1``) followed by ``xla_quantized_matmul``'s activation quant:

    s   = x + residual                  (f32; the new residual is bf16(s))
    y   = bf16(s * rsqrt(mean(s^2) + eps) * (w + 1))
    sc  = max|y| / 448 (1 where that is 0)
    q   = clip(y / sc, -448, 448) as float8_e4m3fn

and returns (y, new residual, q, sc): the normed rows for the layer's bf16
consumers, and the fp8 rows and scales for its fp8 linear.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)


def _norm_quant(s, w, eps):
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps) * (w + 1.0)).astype(jnp.bfloat16)
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True).astype(jnp.float32)
    sc = amax / FP8_MAX
    sc = jnp.where(sc == 0.0, 1.0, sc)
    q = jnp.clip(y.astype(jnp.float32) / sc, -FP8_MAX, FP8_MAX).astype(FP8)
    return y, q, sc


def _kernel(*refs, eps, has_res, emit_y):
    refs = list(refs)
    x_ref = refs.pop(0)
    r_ref = refs.pop(0) if has_res else None
    w_ref = refs.pop(0)
    y_ref = refs.pop(0) if emit_y else None
    ro_ref = refs.pop(0) if has_res else None
    q_ref, s_ref = refs
    s = x_ref[...].astype(jnp.float32)
    if has_res:
        s = s + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    y, q, sc = _norm_quant(s, w_ref[...].astype(jnp.float32)[None, :], eps)
    if emit_y:
        y_ref[...] = y
    q_ref[...] = q
    s_ref[...] = sc


def _tile(rows: int, cap: int) -> int:
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


@functools.partial(
    jax.jit, static_argnames=("eps", "emit_y", "tile", "interpret")
)
def rms_norm_quant(
    x: jax.Array,
    residual: jax.Array | None,
    w: jax.Array,
    eps: float,
    *,
    emit_y: bool = True,
    tile: int = 256,
    interpret: bool = False,
):
    """x, residual: [m, k] bf16 (residual may be None); w: [k] (the raw Gemma
    weight; the kernel adds 1). Returns (y or None, new_residual or None, q,
    scale)."""
    m, k = x.shape
    t = _tile(m, tile)
    has_res = residual is not None
    row = pl.BlockSpec((t, k), lambda i: (i, 0))
    in_specs = [row] + ([row] if has_res else []) + [pl.BlockSpec((k,), lambda i: (0,))]
    out_specs, out_shape = [], []
    if emit_y:
        out_specs.append(row)
        out_shape.append(jax.ShapeDtypeStruct((m, k), jnp.bfloat16))
    if has_res:
        out_specs.append(row)
        out_shape.append(jax.ShapeDtypeStruct((m, k), residual.dtype))
    out_specs += [row, pl.BlockSpec((t, 1), lambda i: (i, 0))]
    out_shape += [
        jax.ShapeDtypeStruct((m, k), FP8),
        jax.ShapeDtypeStruct((m, 1), jnp.float32),
    ]
    outs = list(
        pl.pallas_call(
            functools.partial(_kernel, eps=float(eps), has_res=has_res, emit_y=emit_y),
            grid=(m // t,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret,
        )(*([x] + ([residual] if has_res else []) + [w]))
    )
    y = outs.pop(0) if emit_y else None
    ro = outs.pop(0) if has_res else None
    q, sc = outs
    return y, ro, q, sc


def reference(x, residual, w, eps):
    """The served path: GemmaRMSNorm, then xla_quantized_matmul's quantization."""
    s = x.astype(jnp.float32)
    ro = None
    if residual is not None:
        s = s + residual.astype(jnp.float32)
        ro = s.astype(residual.dtype)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps) * (w.astype(jnp.float32) + 1.0)).astype(
        jnp.bfloat16
    )
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    sc = amax / jnp.finfo(FP8).max.astype(jnp.float32)
    sc = jnp.where(sc == 0, 1.0, sc)
    q = (y / sc).clip(-FP8_MAX, FP8_MAX).astype(FP8)
    return y, ro, q, sc.astype(jnp.float32)
