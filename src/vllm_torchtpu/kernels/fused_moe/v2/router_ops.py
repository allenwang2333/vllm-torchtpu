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
"""Top-k expert selection for the fused expert-parallel MoE router:
pallas_select is an in-VMEM max-and-mask pass replacing XLA's lax.top_k.

An all-NaN score row selects the lowest expert on every slot, with NEG as
the weight of each: NaN is flushed to the sentinel before the first max, so
every row has a maximum some column equals and every index returned is a
real expert. What that row's weights MEAN is not decided here. The kernel
layer masks such a row explicitly, on jnp.any(jnp.isfinite(scores)), rather
than relying on the sentinels to sum to something the renormalization turns
into zero -- that outcome was a property of the accumulation dtype and the
association order rather than of the design.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

NEG = -3.0e38  # below any real softmax score (>=0); avoids inf


def _select_kernel(*refs, topk, n, has_values):
    """One block of rows: take the top `topk` by repeated max-and-mask.

    With `has_values` the block carries a second [R, n] operand and the
    reported weight is that array's entry at the selected column rather than
    the selection key's own value. That is the `noaux_tc` shape -- select on
    `score + correction_bias`, weight by the UNBIASED score -- and doing it
    here rather than with a `take_along_axis` outside keeps the [R, n] gather
    in VMEM, on a block that is resident anyway.
    """
    if has_values:
        x_ref, v_ref, w_ref, i_ref = refs
    else:
        x_ref, w_ref, i_ref = refs
        v_ref = None
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])  # [R, n] f32
    v = None if v_ref is None else v_ref[...]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        idx = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        sel = iota == idx[:, None]  # exactly one true column per row
        if v is None:
            w_ref[:, j] = m[:, 0]
        else:
            # A sum, not a max: exactly one lane is live, and adding zeros to
            # one float is exact in IEEE regardless of the reduction order, so
            # this returns v[row, idx] bit for bit -- including a NaN or a
            # value below any sentinel a max would have floored.
            w_ref[:, j] = jnp.sum(jnp.where(sel, v, 0.0), axis=1)
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, NEG, x)


def pallas_select(scores, topk=10, block_rows=256, values=None):
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk].

    `values`, when given, is a second [R, N] f32 array the weights are read
    from at the selected columns; the selection itself still runs on `scores`.
    """
    R, N = scores.shape
    assert R % block_rows == 0, (R, block_rows)
    if values is not None:
        assert values.shape == scores.shape, (values.shape, scores.shape)
    grid = (R // block_rows, )
    out_shapes = (
        jax.ShapeDtypeStruct((R, topk), jnp.float32),
        jax.ShapeDtypeStruct((R, topk), jnp.int32),
    )
    row_block = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    bs = (row_block, ) if values is None else (row_block, row_block)
    os = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
    )
    operands = (scores, ) if values is None else (scores, values)
    return pl.pallas_call(
        functools.partial(_select_kernel,
                          topk=topk,
                          n=N,
                          has_values=values is not None),
        grid=grid,
        in_specs=list(bs),
        out_specs=list(os),
        out_shape=list(out_shapes),
    )(*operands)
