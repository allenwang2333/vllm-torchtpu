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
"""Per-row fp8 activation quantization and the fp8 matmul in one kernel.

``xla_quantized_matmul`` quantizes the activation in separate XLA fusions (a
per-row abs-max reduce, then divide + clip + convert) before an XLA fp8 dot.
At decode row counts each of those fusions is launch bound. This kernel takes
the bf16 rows, computes the per-row abs-max, the scale and the fp8 rows in
VMEM once per row tile (cached across the output tiles), and runs the fp8 dot
on them -- the same arithmetic as ``xla_quantized_matmul``:

    scale = max|x| / 448 (1 where that is 0)
    q     = clip(x / scale, -448, 448) as float8_e4m3fn
    out   = bf16((q @ w) * scale * w_scale)

The weight is either the canonical ``(k, n)`` layout or the ``(n, k)`` one a
layer keeps for a raw consumer (``w_kn=False``); both contract on ``k``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)

# Weight bytes per output tile the default tiling aims for.
_TARGET_W_TILE_BYTES = 2 * 1024 * 1024


def _kernel(x_ref, w_ref, ws_ref, o_ref, xq_ref, xs_ref, *, w_kn: bool):
    @pl.when(pl.program_id(1) == 0)
    def _quantize():
        x = x_ref[...]
        amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
        scale = amax / FP8_MAX
        scale = jnp.where(scale == 0.0, 1.0, scale)
        xq_ref[...] = jnp.clip(
            x.astype(jnp.float32) / scale, -FP8_MAX, FP8_MAX
        ).astype(FP8)
        xs_ref[...] = scale

    dims = (((1,), (0,)), ((), ())) if w_kn else (((1,), (1,)), ((), ()))
    acc = jax.lax.dot_general(
        xq_ref[...], w_ref[...], dims, preferred_element_type=jnp.float32
    )
    acc = acc * xs_ref[...]
    acc = acc * ws_ref[...]
    o_ref[...] = acc.astype(o_ref.dtype)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def default_tiles(m: int, k: int, n: int, w_kn: bool) -> tuple[int, int]:
    """(row tile, output tile) for an [m, k] x [k, n] problem."""
    # Rows: one tile up to 512 (decode and small prefill chunks), padded up to
    # the fp8 sublane packing of 32; else the largest of 512/256/128 dividing m.
    if m <= 512:
        tm = _round_up(m, 32)
    else:
        tm = next((t for t in (512, 256, 128) if m % t == 0), 256)
    align = 128 if w_kn else 32
    want = max(align, _TARGET_W_TILE_BYTES // k // align * align)
    tn = n
    if n > want:
        tn = next(
            (t for t in range(want, align - 1, -align) if n % t == 0),
            n,
        )
    return tm, tn


@functools.partial(
    jax.jit, static_argnames=("w_kn", "tm", "tn", "out_dtype", "interpret")
)
def rowquant_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    w_kn: bool = True,
    tm: int | None = None,
    tn: int | None = None,
    out_dtype=None,
    interpret: bool = False,
) -> jax.Array:
    """x [m, k] bf16; w_q fp8 [k, n] (``w_kn``) or [n, k]; w_s [n] -> [m, n]."""
    m, k = x.shape
    if w_kn:
        kw, n = w_q.shape
    else:
        n, kw = w_q.shape
    if kw != k:
        raise ValueError(f"x {x.shape} and w_q {w_q.shape} disagree on k")
    if w_s.shape != (n,):
        raise ValueError(f"w_s must be per-channel [{n}], got {w_s.shape}")
    out_dtype = x.dtype if out_dtype is None else out_dtype
    dtm, dtn = default_tiles(m, k, n, w_kn)
    tm = dtm if tm is None else tm
    tn = dtn if tn is None else tn
    if n % tn:
        raise ValueError(f"output tile {tn} must divide n={n}")
    ws = w_s.astype(jnp.float32).reshape(1, n)
    grid = (pl.cdiv(m, tm), n // tn)
    if w_kn:
        w_spec = pl.BlockSpec((k, tn), lambda i, j: (0, j))
    else:
        w_spec = pl.BlockSpec((tn, k), lambda i, j: (j, 0))
    xb = tm * k * x.dtype.itemsize
    wb = tn * k * w_q.dtype.itemsize
    ob = tm * tn * jnp.dtype(out_dtype).itemsize
    vmem = 2 * (xb + wb + ob + tn * 4) + tm * k + tm * 4 + 2 * tm * tn * 4
    vmem = min(max(int(vmem * 1.5), 16 << 20), 100 << 20)
    return pl.pallas_call(
        functools.partial(_kernel, w_kn=w_kn),
        grid=grid,
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            w_spec,
            pl.BlockSpec((1, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        scratch_shapes=[
            pltpu.VMEM((tm, k), FP8),
            pltpu.VMEM((tm, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem,
        ),
        interpret=interpret,
    )(x, w_q, ws)
