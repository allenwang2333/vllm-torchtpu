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


def _kernel(o_ref, w_ref, out_ref):
    w = w_ref[0]  # f32[bt, 1]
    x = o_ref[...].astype(jnp.float32) * w
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    out_ref[0] = jnp.where(w > 0, x, jnp.zeros_like(x))


def _row_block(tokens: int) -> int:
    for bt in (512, 256, 128, 64, 32, 16, 8):
        if tokens % bt == 0:
            return bt
    return tokens


@functools.partial(jax.jit, static_argnames=("fault",))
def weighted_heads_major(
    o: jax.Array, w: jax.Array, fault: str | None = None
) -> jax.Array:
    """`o` [T, H, D] (bf16), `w` f32[T, H] -> f32[H, T, D] =
    `where(w > 0, o * w, 0)` transposed to heads-major, bit for bit the
    torch merge's `output.float() * weight` / `where` per element.

    `fault` (test-only): "swap_heads" writes head h's rows to head h ^ 1.
    """
    t, h, d = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if d % 128:
        raise ValueError(f"head dim {d} must be a multiple of 128")
    bt = _row_block(t)
    o2 = o.reshape(t, h * d)  # head hh is columns [hh * d, (hh + 1) * d)
    w3 = jnp.transpose(w.astype(jnp.float32)).reshape(h, t, 1)
    out_head = (lambda hh: hh ^ 1) if fault == "swap_heads" else (lambda hh: hh)
    return pl.pallas_call(
        _kernel,
        grid=(h, t // bt),
        in_specs=[
            pl.BlockSpec((bt, d), lambda hh, i: (i, hh)),
            pl.BlockSpec((1, bt, 1), lambda hh, i: (hh, i, 0)),
        ],
        out_specs=pl.BlockSpec((1, bt, d), lambda hh, i: (out_head(hh), i, 0)),
        out_shape=jax.ShapeDtypeStruct((h, t, d), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name=f"dcp_weighted_hm_t{t}_h{h}",
    )(o2, w3)
