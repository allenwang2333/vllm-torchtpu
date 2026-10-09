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
"""Sort-free top-k expert selection for the MoE router (Pallas).

``torch.topk`` lowers to a full ``stablehlo.sort`` of every expert plus a
slice, so at 512 experts it runs a 45-stage bitonic network over three arrays
to keep 10 values. This kernel selects the same 10 with ``k`` max-and-mask
passes over one VMEM-resident block: ``O(k*N)`` instead of ``O(N*log^2 N)``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512


def _select_step(
    raw: jax.Array,
    choice: jax.Array,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    renormalize: bool = False,
):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``choice > NEG`` rather than ``isnan(choice)`` to catch all
    three spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of
    which a whole row can be made of. Without it such a row would stay tied
    with the ``finfo.min`` mask and every pass would return column 0.

    When ``raw is not choice`` (biased routing), expert selection uses
    ``choice = raw + bias`` while the returned weight is the unbiased score from
    ``raw`` selected with the same one-hot mask ``sel = (iota == idx[:, None])``
    already computed for masking. When ``renormalize=True``, the ``topk``
    selected weights are summed across the ``topk`` VREG vectors and divided in
    registers before writing ``w_ref``.
    """
    x = jnp.where(choice > NEG, choice, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    first_m = None
    w_list = []
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        if j == 0:
            first_m = m
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        sel = iota == idx[:, None]
        w_j = (
            m
            if raw is choice
            else jnp.sum(jnp.where(sel, raw, 0.0), axis=1, keepdims=True)
        )
        if renormalize:
            w_list.append(w_j)
        else:
            w_ref[:, j] = w_j[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, masked, x)
    assert first_m is not None
    if renormalize:
        denom = w_list[0]
        for w_j in w_list[1:]:
            denom = denom + w_j
        denom = jnp.maximum(denom, 1e-20)
        for j, w_j in enumerate(w_list):
            w_ref[:, j] = (w_j / denom)[:, 0]
    # No score above the sentinel means no real maximum: keep the row
    # poisonous rather than letting renormalization make it plausible.
    w_ref[...] = jnp.where(first_m <= NEG, jnp.nan, w_ref[...])


def _select_kernel(
    x_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    renormalize: bool = False,
):
    """Unbiased top-k selection kernel body."""
    raw = x_ref[...]
    _select_step(raw, raw, w_ref, i_ref, topk=topk, n=n, renormalize=renormalize)


def _select_biased_kernel(
    x_ref,
    b_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    renormalize: bool = False,
):
    """Biased top-k selection kernel body (choice by ``x + b``, weights from ``x``)."""
    raw = x_ref[...]
    choice = raw + b_ref[...]
    _select_step(raw, choice, w_ref, i_ref, topk=topk, n=n, renormalize=renormalize)


def select(
    scores: jax.Array,
    topk: int,
    interpret: bool = False,
    *,
    bias: jax.Array | None = None,
    renormalize: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending (or ordered by descending biased score when ``bias`` is given).
    ``interpret`` runs the kernel on the host, for tests without a TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    out_specs = [
        pl.BlockSpec((block, topk), lambda i: (i, 0)),
        pl.BlockSpec((block, topk), lambda i: (i, 0)),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((rows, topk), jnp.float32),
        jax.ShapeDtypeStruct((rows, topk), jnp.int32),
    ]
    if bias is None:
        return pl.pallas_call(
            functools.partial(_select_kernel, topk=topk, n=n, renormalize=renormalize),
            grid=(pl.cdiv(rows, block),),
            in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
            out_specs=out_specs,
            out_shape=out_shape,
            interpret=interpret,
        )(scores)
    bias_2d = bias if bias.shape == (1, n) else jnp.reshape(bias, (1, n))
    return pl.pallas_call(
        functools.partial(
            _select_biased_kernel, topk=topk, n=n, renormalize=renormalize
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
    )(scores, bias_2d)


def router_topk(
    scores: jax.Array, topk: int, renormalize: bool = False
) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk, renormalize=renormalize)


def router_topk_biased(
    scores: jax.Array,
    bias: jax.Array,
    topk: int,
    renormalize: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Biased ``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk, bias=bias, renormalize=renormalize)
