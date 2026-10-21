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
"""DSv4 MoE router as value-level functions usable inside a Pallas kernel.

Reproduces the fused EP MoE v2 serving path (kernels/fused_moe/v2/layer.py and
router_ops.pallas_select): sqrtsoftplus scores, selection on scores + the
noaux_tc correction bias by repeated max-and-mask with ties to the lowest
expert, weights read from the unbiased scores, renormalization, the routed
scaling factor, and the empty-row mask. Hash layers take the selection from a
token-id table and keep the score weights.
"""

import jax
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32
FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
NEG = -3.0e38  # router_ops.NEG


def _finite(v):
    return jnp.abs(v) < jnp.inf


def rowquant_fp8(x):
    """Per-row dynamic fp8 quant (kernels/fused_moe/v2/kernel.rowquant_fp8):
    max-reduce and apply in x's dtype; unusable rows (zero or non-finite max)
    quantize to zero. Returns (q fp8, scale f32 [rows, 1])."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = amax.astype(F32) / FP8_MAX
    usable = (scale > 0) & (scale < jnp.inf)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    q = jnp.where(usable, x * sinv.astype(x.dtype), jnp.zeros_like(x)).astype(FP8)
    return q, scale


def sqrtsoftplus(logits):
    return jnp.sqrt(jax.nn.softplus(logits.astype(F32)))


def select_topk(choice, weight_scores, topk):
    """router_ops._select_kernel on values: choice, weight_scores f32 [M, E] ->
    (ids, weights), each a list of topk [M, 1] columns in pick order."""
    x = jnp.where(jnp.isnan(choice), NEG, choice)
    n = x.shape[-1]
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    ids, ws = [], []
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1, keepdims=True)
        picked = iota == idx
        ws.append(jnp.sum(jnp.where(picked, weight_scores, 0.0), axis=1, keepdims=True))
        ids.append(idx)
        x = jnp.where(picked, NEG, x)
    return ids, ws


def gather_scores(scores, ids):
    """Hash layers: the score at each given id ([M, 1] columns)."""
    iota = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    return [
        jnp.sum(jnp.where(iota == i, scores, 0.0), axis=1, keepdims=True) for i in ids
    ]


def finalize(
    ids, ws, logits, scores, live_rows, *, n_experts, renormalize, routed_scaling_factor
):
    """layer.py's post-selection: renormalize, scale, mask rows that route
    nowhere (no finite score or logit, or not live) to weight 0 and expert id
    n_experts. `live_rows` is a row count (a live prefix) or a bool [M, 1] mask."""
    if getattr(live_rows, "shape", ()) == ():
        rows = lax.broadcasted_iota(jnp.int32, (logits.shape[0], 1), 0)
        live = rows < live_rows
    else:
        live = live_rows
    routes = (
        (jnp.max(jnp.where(_finite(scores), 1, 0), axis=1, keepdims=True) > 0)
        & (jnp.max(jnp.where(_finite(logits), 1, 0), axis=1, keepdims=True) > 0)
        & live
    )
    if renormalize:
        tot = ws[
            0
        ]  # sequential; XLA's row-sum order is platform dependent (<= 1 ulp apart)
        for w in ws[1:]:
            tot = tot + w
        denom = jnp.where(routes, jnp.maximum(tot, 1e-20), 1.0)
        ws = [w / denom for w in ws]
    if routed_scaling_factor != 1.0:
        ws = [w * jnp.float32(routed_scaling_factor) for w in ws]
    ws = [jnp.where(routes, w, 0.0) for w in ws]
    ids = [jnp.where(routes, i, jnp.int32(n_experts)) for i in ids]
    return ids, ws


def route(
    logits,
    *,
    bias=None,
    hash_ids=None,
    live_rows,
    topk,
    n_experts,
    renormalize=True,
    routed_scaling_factor=1.0,
):
    """f32 gate logits [M, E] -> (ids, weights) lists of topk [M, 1] columns.
    `bias` [1, E] f32 (noaux_tc) or `hash_ids` (list of topk [M, 1] i32)."""
    scores = sqrtsoftplus(logits)
    if hash_ids is not None:
        ids, ws = list(hash_ids), gather_scores(scores, hash_ids)
    elif bias is None:
        ids, ws = select_topk(scores, scores, topk)
    else:
        ids, ws = select_topk(scores + bias, scores, topk)
    return finalize(
        ids,
        ws,
        logits,
        scores,
        live_rows,
        n_experts=n_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def combine_matrix(ids, ws, n_experts):
    """Dense [M, E] f32 routing weights and the [1, E] i32 hit mask of the
    experts at least one routed row selected (masked rows carry id E)."""
    shape = (ids[0].shape[0], n_experts)
    iota = lax.broadcasted_iota(jnp.int32, shape, 1)
    dense = jnp.zeros(shape, F32)
    hit = jnp.zeros(shape, jnp.int32)
    for i, w in zip(ids, ws):
        sel = iota == i
        dense = dense + jnp.where(
            sel, w, 0.0
        )  # a repeated hash id adds, as the stock combine does
        hit = jnp.where(sel, 1, hit)
    return dense, jnp.max(hit, axis=0, keepdims=True)
