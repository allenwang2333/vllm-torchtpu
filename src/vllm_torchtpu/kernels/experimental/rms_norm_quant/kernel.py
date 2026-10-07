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


def _norm_quant(s, w, eps, rstd=None):
    if rstd is None:
        rstd = jax.lax.rsqrt(jnp.mean(s * s, axis=-1, keepdims=True) + eps)
    y = (s * rstd * (w + 1.0)).astype(jnp.bfloat16)
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True).astype(jnp.float32)
    sc = amax / FP8_MAX
    sc = jnp.where(sc == 0.0, 1.0, sc)
    q = jnp.clip(y.astype(jnp.float32) / sc, -FP8_MAX, FP8_MAX).astype(FP8)
    return y, q, sc


def _kernel(*refs, eps, has_res, emit_y, has_rstd, gamma, has_ba):
    refs = list(refs)
    x_ref = refs.pop(0)
    r_ref = refs.pop(0) if has_res else None
    w_ref = refs.pop(0) if gamma != "none" else None
    rstd = refs.pop(0)[...] if has_rstd else None
    wba_ref = refs.pop(0) if has_ba else None
    y_ref = refs.pop(0) if emit_y else None
    ba_ref = refs.pop(0) if has_ba else None
    ro_ref = refs.pop(0) if has_res else None
    q_ref, s_ref = refs[:2]
    if gamma == "hbm":
        # The weight stays in HBM; its DMA overlaps the residual add (and the
        # statistic), instead of XLA staging it into VMEM before the call.
        w_vmem, sem = refs[2:]
        cp = pltpu.make_async_copy(w_ref, w_vmem, sem)
        cp.start()
    s = x_ref[...].astype(jnp.float32)
    if has_res:
        s = s + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    if gamma == "none":  # timing-only removal arm: gamma = 1 (w = 0)
        w = jnp.zeros((1, s.shape[-1]), jnp.float32)
    elif gamma == "hbm":
        if rstd is None:
            rstd = jax.lax.rsqrt(jnp.mean(s * s, axis=-1, keepdims=True) + eps)
        cp.wait()
        w = w_vmem[...].astype(jnp.float32)[None, :]
    else:
        w = w_ref[...].astype(jnp.float32)[None, :]
    y, q, sc = _norm_quant(s, w, eps, rstd)
    if has_ba:
        # The layer's bf16 consumer of the normed rows (GDN in_proj_ba), so the
        # rows need not leave the kernel.
        ba_ref[...] = jax.lax.dot_general(
            y, wba_ref[...], (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(ba_ref.dtype)
    if emit_y:
        y_ref[...] = y
    q_ref[...] = q
    s_ref[...] = sc


def _tile(rows: int, cap: int) -> int:
    for t in range(min(cap, rows), 7, -1):
        if rows % t == 0 and t % 8 == 0:
            return t
    return rows


# Row counts up to this keep the outputs in VMEM (decode buckets), where the
# next ops read them without an HBM round trip.
VMEM_OUT_MAX_ROWS = 64


@functools.partial(
    jax.jit,
    static_argnames=("eps", "emit_y", "out_vmem", "gamma", "tile", "interpret"),
)
def rms_norm_quant(
    x: jax.Array,
    residual: jax.Array | None,
    w: jax.Array,
    eps: float,
    rstd: jax.Array | None = None,
    *,
    emit_y: bool = True,
    out_vmem: bool | None = None,
    gamma: str = "vmem",
    wba: jax.Array | None = None,
    tile: int = 256,
    interpret: bool = False,
):
    """x, residual: [m, k] bf16 (residual may be None); w: [k] (the raw Gemma
    weight; the kernel adds 1). Returns (y or None, new_residual or None, q,
    scale).

    ``rstd`` [m, 1] f32, if given, is ``rsqrt(mean(s^2) + eps)`` computed by
    the caller (XLA), so the normed rows match the XLA norm bit for bit; the
    kernel then only applies it and quantizes."""
    m, k = x.shape
    t = _tile(m, tile)
    if out_vmem is None:
        out_vmem = m <= VMEM_OUT_MAX_ROWS and not interpret
    sds = pltpu.VMEM if out_vmem else jax.ShapeDtypeStruct
    has_res = residual is not None
    row = pl.BlockSpec((t, k), lambda i: (i, 0))
    has_rstd = rstd is not None
    in_specs = [row] + ([row] if has_res else [])
    args = [x] + ([residual] if has_res else [])
    scratch = []
    if gamma == "vmem":
        in_specs.append(pl.BlockSpec((k,), lambda i: (0,)))
        args.append(w)
    elif gamma == "hbm":
        in_specs.append(pl.BlockSpec(memory_space=pl.ANY))
        args.append(w if interpret else pltpu.with_memory_space_constraint(w, pltpu.HBM))
        scratch = [pltpu.VMEM((k,), w.dtype), pltpu.SemaphoreType.DMA(())]
    else:
        assert gamma == "none", gamma
    if has_rstd:
        in_specs.append(pl.BlockSpec((t, 1), lambda i: (i, 0)))
    has_ba = wba is not None
    if has_ba:  # wba: [n_ba, k], lane-dense (a [k, 16] operand gets relaid out per call)
        nba = wba.shape[0]
        in_specs.append(pl.BlockSpec((nba, k), lambda i: (0, 0)))
    out_specs, out_shape = [], []
    if emit_y:
        out_specs.append(row)
        out_shape.append(sds((m, k), jnp.bfloat16))
    if has_ba:
        out_specs.append(pl.BlockSpec((t, nba), lambda i: (i, 0)))
        out_shape.append(sds((m, nba), wba.dtype))
    if has_res:
        out_specs.append(row)
        out_shape.append(sds((m, k), residual.dtype))
    out_specs += [row, pl.BlockSpec((t, 1), lambda i: (i, 0))]
    out_shape += [sds((m, k), FP8), sds((m, 1), jnp.float32)]
    outs = list(
        pl.pallas_call(
            functools.partial(
                _kernel,
                eps=float(eps),
                has_res=has_res,
                emit_y=emit_y,
                has_rstd=has_rstd,
                gamma=gamma,
                has_ba=has_ba,
            ),
            grid=(m // t,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret,
        )(*(args + ([rstd] if has_rstd else []) + ([wba] if has_ba else [])))
    )
    y = outs.pop(0) if emit_y else None
    ba = outs.pop(0) if has_ba else None
    ro = outs.pop(0) if has_res else None
    q, sc = outs
    if has_ba:
        return y, ro, q, sc, ba
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
