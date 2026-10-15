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
"""The DCP merge's weighted reduce-scatter operand, built heads-major.

The reduce-scatter merge scatters the head axis, so XLA lays its f32 operand
out heads-major: the weight multiply writes `[T, H, D]` tokens-major and a
relayout copy follows (on the TensorCore, or offloaded to the SparseCore as a
data-format call whose completion the scheduler then holds back until the
reduce-scatter). This kernel writes `where(w > 0, o * w, 0)` straight into a
`[H, T, D]` buffer: one pass, the same f32 products, no relayout.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(o_ref, w_ref, out_ref, *, heads: int, order=None):
    # o_ref [bt, heads, D] (tokens-major, as the attention wrote it), w_ref
    # f32[bt, heads] (the weights as the merge computed them), out_ref
    # f32[heads, bt, D]: head j's rows are a strided sublane read of the input
    # block, its weights a lane column, written contiguously.
    w_all = w_ref[...]
    for j in range(heads):
        w = w_all[:, j : j + 1]  # f32[bt, 1]
        x = o_ref[:, j, :].astype(jnp.float32) * w
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        # A bf16 out_ref (TPU_DCP_MERGE_RS_BF16) rounds the same f32 value
        # once, as the torch merge's `weighted.to(bfloat16)` does.
        out_ref[j if order is None else order[j]] = jnp.where(
            w > 0, x, jnp.zeros_like(x)
        ).astype(out_ref.dtype)


# Bytes of one grid step's input (bf16) plus output (f32) block.
_STEP_BYTES = 6 << 20


def _row_block(tokens: int, heads: int, dim: int, min_rows: int = 8) -> int:
    for bt in (256, 128, 64, 32, 16, 8):
        if bt >= min_rows and tokens % bt == 0 and bt * heads * dim * 6 <= _STEP_BYTES:
            return bt
    return min_rows if tokens % min_rows == 0 else tokens


@functools.partial(jax.jit, static_argnames=("fault", "out_dtype"))
def weighted_heads_major(
    o: jax.Array, w: jax.Array, fault: str | None = None, out_dtype=jnp.float32
) -> jax.Array:
    """`o` [T, H, D] (bf16), `w` f32[T, H] -> f32[H, T, D] =
    `where(w > 0, o * w, 0)` transposed to heads-major, bit for bit the
    torch merge's `output.float() * weight` / `where` per element.

    `o` is read in its own [T, H, D] layout (no reshape: a 2-D view of a
    tiled 3-D buffer is a relayout copy). `fault` (test-only): "swap_heads"
    writes the heads in reverse order. `out_dtype` bf16
    (TPU_DCP_MERGE_RS_BF16): each f32 value rounded once to bf16, the operand
    of the bf16 reduce-scatter.
    """
    out_dtype = jnp.dtype(out_dtype)
    t, h, d = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if d % 128:
        raise ValueError(f"head dim {d} must be a multiple of 128")
    # Every head per step: the [bt, H] weight block's lane dim must be all
    # of H (< 128 heads).
    hb = h
    # bf16 output blocks pack 16 rows per sublane tile.
    bt = _row_block(t, hb, d, min_rows=16 if out_dtype.itemsize == 2 else 8)
    order = tuple(range(hb - 1, -1, -1)) if fault == "swap_heads" else None
    kernel = functools.partial(_kernel, heads=hb, order=order)
    return pl.pallas_call(
        kernel,
        grid=(1, t // bt),
        in_specs=[
            pl.BlockSpec((bt, hb, d), lambda g, i: (i, g, 0)),
            pl.BlockSpec((bt, hb), lambda g, i: (i, g)),
        ],
        out_specs=pl.BlockSpec((hb, bt, d), lambda g, i: (g, i, 0)),
        out_shape=jax.ShapeDtypeStruct((h, t, d), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name=f"dcp_weighted_hm_t{t}_h{h}"
        + ("_bf16" if out_dtype.itemsize == 2 else ""),
    )(o, w.astype(jnp.float32))


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def weighted_heads_major_xla(
    o: jax.Array, w: jax.Array, out_dtype=jnp.float32
) -> jax.Array:
    """`weighted_heads_major` in XLA ops: the same f32 product, the same
    `where`, the same single rounding to `out_dtype`, then heads-major. For
    the buckets below TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS, where the
    kernel's per-chunk launch costs more than the fusion XLA emits."""
    t, h, _ = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    w = w.astype(jnp.float32)[:, :, None]
    x = o.astype(jnp.float32) * w
    x = jnp.where(w > 0, x, jnp.zeros_like(x)).astype(out_dtype)
    return jnp.transpose(x, (1, 0, 2))


@functools.partial(jax.jit, static_argnames=("out_dtype", "fault"))
def vspace_weighted_heads_major(
    o: jax.Array,
    w: jax.Array,
    w_uv: jax.Array,
    w_uv_scale: jax.Array | None = None,
    out_dtype=jnp.bfloat16,
    fault: str | None = None,
) -> jax.Array:
    """The V-space merge's reduce-scatter operand (TPU_DCP_MERGE_VSPACE).

    `o` [T, H, C] bf16 (this rank's latent partial over every head of the DCP
    group), `w` f32[T, H] (its merge weights), `w_uv` [H, C, V] (W_UV gathered
    for the same H heads, in the same order), `w_uv_scale` optional
    [H, 1, V] (per-channel scale of a quantized W_UV) -> `[H, T, V]` =
    `where(w > 0, (o_h @ W_UV_h) * w, 0)`, heads-major, rounded once to
    `out_dtype`.

    The projection runs on the MXU with bf16 operands and f32 accumulation
    (the torch path's `bmm` operands); the weight multiplies the f32 product,
    so the only rounding before the collective is the final cast. A NaN
    partial (zero weight) only reaches its own (token, head) row of the
    product, which the where zeroes.

    `fault` (test-only): "swap_heads" applies head h's W_UV to head H-1-h;
    "unweighted" skips the weight multiply (keeps the where).
    """
    out_dtype = jnp.dtype(out_dtype)
    t, h, c = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if w_uv.shape[:2] != (h, c):
        raise ValueError(f"w_uv {w_uv.shape} must be [{h}, {c}, V]")
    wuv = w_uv.astype(o.dtype)
    if fault == "swap_heads":
        wuv = wuv[::-1]
    v = jax.lax.dot_general(
        o,
        wuv,
        (((2,), (1,)), ((1,), (0,))),
        preferred_element_type=jnp.float32,
    )  # [H, T, V]
    if w_uv_scale is not None:
        v = v * w_uv_scale.astype(jnp.float32).reshape(h, 1, -1)
    wt = jnp.transpose(w.astype(jnp.float32), (1, 0))[:, :, None]  # [H, T, 1]
    x = v if fault == "unweighted" else v * wt
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    return jnp.where(wt > 0, x, jnp.zeros_like(x)).astype(out_dtype)
