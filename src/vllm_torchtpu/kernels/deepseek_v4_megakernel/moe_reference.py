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
"""Plain-JAX references for the DSv4 megakernel MoE block.

stock_*  reproduce the served fused EP MoE v2 path (layer.py routing,
         router_ops.pallas_select, kernel.expert_ffn_blockscale,
         kernel.wire_row_and_scale and the fp8-wire combine), calling the stock
         functions wherever they are callable outside the kernel.
model_*  restate moe_experts' own arithmetic (per-slice intermediate scale,
         f32 sum, bf16 all-reduce) from the unsliced weights, so a kernel bug
         and a layout bug both show up against it.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v2 import kernel as stock_kernel
from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn, _row_routes
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select

from .allreduce import reference_allreduce
from .moe_experts import (
    MoEConfig,
    intermediate,
    merge_slices,
    shared_expert,
    slice_columns,
    unpack_fp4_rows,
)

F32, BF16, FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
DN = (((1,), (0,)), ((), ()))


def gate_logits(x, gate_w):
    return lax.dot_general(x, gate_w, DN, preferred_element_type=F32)


def stock_route(
    logits, cfg: MoEConfig, live_rows, *, bias=None, hash_ids=None, interpret=True
):
    """layer.local_fn routing on f32 logits [M, E] -> (ids i32, weights f32) [M, topk].
    Rows >= live_rows are padding: their logits are -inf, as token_layout pads."""
    M = logits.shape[0]
    rows = jnp.arange(M)[:, None]
    gate_l = jnp.where(rows < live_rows, logits, -jnp.inf)
    scores = _apply_scoring_fn(gate_l, "sqrtsoftplus")
    if hash_ids is not None:
        topk_idx = hash_ids.astype(jnp.int32)
        topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
    elif bias is None:
        topk_weights, topk_idx = pallas_select(
            scores, topk=cfg.topk, block_rows=M, interpret=interpret
        )
    else:
        topk_weights, topk_idx = pallas_select(
            scores + bias.reshape(1, -1).astype(F32),
            topk=cfg.topk,
            block_rows=M,
            weight_scores=scores,
            interpret=interpret,
        )
    row_routes = _row_routes(scores, gate_l)
    if cfg.renormalize:
        denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
        topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
    if cfg.routed_scaling_factor != 1.0:
        topk_weights = topk_weights * jnp.float32(cfg.routed_scaling_factor)
    topk_weights = jnp.where(row_routes, topk_weights, 0.0)
    topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(cfg.n_experts))
    return topk_idx, topk_weights


def _pairs_by_expert(ids):
    ids = np.asarray(ids)
    out = {}
    for t in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            out.setdefault(int(ids[t, k]), []).append((t, k))
    return out


def stock_experts(x, ids, wts, w1, w1s, w2, w2s, cfg: MoEConfig):
    """The fused EP MoE v2 expert math and combine for routed (token, slot)
    pairs -> bf16 [M, H]. w1/w2 are the logical fp4 operands."""
    M, H = x.shape
    xq, xscale = stock_kernel.rowquant_fp8(x.astype(BF16))
    wire = jnp.zeros((M, cfg.topk, H), F32)  # fp8 wire row * its scale, per slot
    for e, pairs in sorted(_pairs_by_expert(ids).items()):
        if e >= cfg.n_experts:
            continue
        t_idx = np.array([p[0] for p in pairs])
        k_idx = np.array([p[1] for p in pairs])
        w1e = w1[e].astype(FP8)
        w2e = w2[e].astype(FP8)
        acc2, mid_scale = stock_kernel.expert_ffn_blockscale(
            xq[t_idx],
            xscale[t_idx],
            lambda b, n0, n1: w1e[b * cfg.qb : (b + 1) * cfg.qb, n0:n1],
            lambda b, n0, n1: w2e[b * cfg.qb : (b + 1) * cfg.qb, n0:n1],
            w1s[e],
            w2s[e],
            qb=cfg.qb,
            act_fn="silu_and_mul_with_clamp",
        )
        q, rscale = stock_kernel.wire_row_and_scale(acc2, mid_scale)
        wire = wire.at[t_idx, k_idx].set(q.astype(F32) * rscale)
    # _combine_arrivals: coef = row_scale * weight, summed over slots in order
    terms = [wire[:, k] * wts[:, k : k + 1].astype(F32) for k in range(cfg.topk)]
    return sum(terms).astype(BF16)


def rank_items(cfg: MoEConfig, ids, rank, held=None, copy=None):
    """The kernel's work list on `rank`: [(expert, global slice, copy)] in
    order; `held` / `copy` are this rank's placement rows for the layer
    (moe_phase.placement_tables), copy = (copy_idx, n_copies) or None."""
    hit = {int(e) for e in np.asarray(ids).ravel() if e < cfg.n_experts}
    if cfg.mode == "tp":
        return [(e, rank, None) for e in sorted(hit)]
    if held is None:
        slots = [(e, None) for e in sorted(hit) if e // cfg.EL == rank]
    else:
        slots = [
            (int(e), None if copy is None else (int(c) >> 8, int(c) & 255))
            for e, c in zip(held, copy if copy is not None else [0] * len(held))
            if e >= 0 and int(e) in hit
        ]
    return [  # a copy k of n runs slices k, k + n, ... (all rows)
        (e, s, c)
        for e, c in slots
        for s in range(cfg.n_slices)
        if c is None or s % c[1] == c[0]
    ]


def model_partial(
    x, ids, wts, w1, w1s, w2, w2s, cfg: MoEConfig, rank, held=None, copy=None
):
    """One rank's f32 partial [M, H] by the kernel's arithmetic, from the
    logical (unsliced) weights."""
    M, H = x.shape
    S, inter, qb = cfg.S, cfg.inter, cfg.qb
    from . import moe_router as rt

    xq, xs = rt.rowquant_fp8(x.astype(BF16))
    dense, _ = rt.combine_matrix(
        [ids[:, k : k + 1] for k in range(cfg.topk)],
        [wts[:, k : k + 1] for k in range(cfg.topk)],
        cfg.n_experts,
    )
    acc_out = jnp.zeros((M, H), F32)
    idx, pieces = slice_columns(cfg.n_slices, inter, qb)
    bounds = np.cumsum((0,) + pieces)
    for e, gs, _ in rank_items(cfg, ids, rank, held, copy):
        cols = np.concatenate([idx[gs], inter + idx[gs]])
        w1e = w1[e][:, cols].astype(FP8)
        s1 = w1s[e][:, cols]
        acc = jnp.zeros((M, 2 * S), F32)
        for b in range(H // qb):
            part = lax.dot_general(
                xq[:, b * qb : (b + 1) * qb],
                w1e[b * qb : (b + 1) * qb],
                DN,
                preferred_element_type=F32,
            )
            acc = acc + part * s1[b : b + 1]
        mid_q, mid_scale = intermediate(cfg, acc, xs)
        coef = mid_scale * dense[:, e : e + 1]
        w2e = w2[e][idx[gs]].astype(FP8)
        for c0 in range(0, cfg.hidden, cfg.w2_cols):
            cs = slice(c0, c0 + cfg.w2_cols)
            y = None
            for j in range(len(pieces)):
                k0, k1 = int(bounds[j]), int(bounds[j + 1])
                b = int(idx[gs, k0]) // qb
                t = lax.dot_general(
                    mid_q[:, k0:k1], w2e[k0:k1, cs], DN, preferred_element_type=F32
                )
                t = t * w2s[e][b : b + 1, cs]
                y = t if y is None else y + t
            acc_out = acc_out.at[:, cs].add(y * coef)
    return acc_out


def model_moe(x, ids, wts, w1, w1s, w2, w2s, cfg: MoEConfig, shared=None):
    """What every rank of moe_layer_call returns: bf16 [M, H]. `shared` is
    slice_shared_weights' output when cfg.shared."""
    parts = []
    for r in range(cfg.R):
        p = model_partial(x, ids, wts, w1, w1s, w2, w2s, cfg, r)
        if shared is not None:
            sw13, s13, sw2, s2 = shared
            p = p + shared_expert(cfg, x, sw13[r], s13[r], sw2[r], s2[r])
        parts.append(p[: cfg.ar_rows])
    tot = reference_allreduce(jnp.stack(parts), cfg.ar_mode, cfg.R)
    out = jnp.zeros((cfg.rows, cfg.hidden), BF16)
    return out.at[: cfg.ar_rows].set(tot.astype(BF16))


def stock_shared(x, w13, s13, w2, s2, cfg: MoEConfig):
    """The served shared expert unsharded: xla_quantized_matmul twice with the
    clamped activation (row scale of w2's input over all columns) -> bf16."""
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    inter, lim = cfg.shared_inter, cfg.swiglu_limit
    h = xla_quantized_matmul(x.astype(BF16), w13, s13)
    g = jnp.minimum(h[:, :inter].astype(F32), lim)
    u = jnp.clip(h[:, inter:].astype(F32), -lim, lim)
    a = (jax.nn.silu(g) * u).astype(BF16)
    return xla_quantized_matmul(a, w2, s2)


def synthetic_shared(key, cfg: MoEConfig):
    """Random fp8 shared expert with per-channel scales (unsharded)."""
    H, inter = cfg.hidden, cfg.shared_inter
    k = jax.random.split(key, 4)
    w13 = jax.random.normal(k[0], (H, 2 * inter), F32).astype(FP8)
    w2 = jax.random.normal(k[1], (inter, H), F32).astype(FP8)
    s13 = jax.random.uniform(k[2], (2 * inter,), F32, 0.5, 1.5) / np.sqrt(H)
    s2 = jax.random.uniform(k[3], (H,), F32, 0.5, 1.5) / np.sqrt(inter)
    return w13, s13, w2, s2


def exact_moe(x, ids, wts, w1, w1s, w2, w2s, cfg: MoEConfig):
    """The layer without activation quantization: bf16 rows, dequantized f32
    weights, f32 everywhere, HIGHEST precision dots -> f32 [M, H]."""
    hp = lax.Precision.HIGHEST
    xf = x.astype(F32)
    out = jnp.zeros(xf.shape, F32)
    inter, qb, lim = cfg.inter, cfg.qb, cfg.swiglu_limit
    for e, pairs in sorted(_pairs_by_expert(ids).items()):
        if e >= cfg.n_experts:
            continue
        t_idx = np.array([p[0] for p in pairs])
        k_idx = np.array([p[1] for p in pairs])
        w1f = w1[e].astype(F32) * jnp.repeat(w1s[e], qb, axis=0)
        w2f = w2[e].astype(F32) * jnp.repeat(w2s[e], qb, axis=0)
        h = jnp.dot(xf[t_idx], w1f, precision=hp)
        act = jax.nn.silu(jnp.minimum(h[:, :inter], lim)) * jnp.clip(
            h[:, inter:], -lim, lim
        )
        y = jnp.dot(act, w2f, precision=hp) * wts[t_idx, k_idx][:, None]
        out = out.at[t_idx].add(y)
    return out


def synthetic_weights(key, cfg: MoEConfig, *, w_scale=1.0):
    """Random fp4 experts with f32 block scales sized so activations stay O(1)."""
    E, H, inter, qb = cfg.n_experts, cfg.hidden, cfg.inter, cfg.qb
    k = jax.random.split(key, 4)
    w1 = jax.random.normal(k[0], (E, H, 2 * inter), F32).astype(jnp.float4_e2m1fn)
    w2 = jax.random.normal(k[1], (E, inter, H), F32).astype(jnp.float4_e2m1fn)
    w1s = jax.random.uniform(k[2], (E, H // qb, 2 * inter), F32, 0.5, 1.5) * (
        w_scale / np.sqrt(H)
    )
    w2s = jax.random.uniform(k[3], (E, inter // qb, H), F32, 0.5, 1.5) * (
        w_scale / np.sqrt(inter)
    )
    return w1, w1s, w2, w2s


class _ByExpert:
    def __init__(self, fn):
        self.fn, self.cache = fn, {}

    def __getitem__(self, e):
        if e not in self.cache:
            self.cache[e] = self.fn(e)
        return self.cache[e]


def logical_experts(w1p, w1sp, w2p, w2sp, cfg: MoEConfig):
    """(w1, w1s, w2, w2s) in the stock layout, rebuilt per expert on demand
    from slice_expert_weights tiles (weights as exact f32 values)."""
    S, qb, inter = cfg.S, cfg.qb, cfg.inter
    idx, pieces = slice_columns(cfg.n_slices, inter, qb)
    host = lambda a: jnp.asarray(np.asarray(a))  # noqa: E731

    def cols(tiles):  # [R, rows, 2S] -> [rows, 2I]
        t = jnp.transpose(tiles, (1, 0, 2))
        return jnp.concatenate(
            [merge_slices(t[..., :S], pieces), merge_slices(t[..., S:], pieces)], axis=1
        )

    def w2s_of(e):
        t = np.asarray(w2sp[e])  # [R, 2, H]
        rows = {}
        for r in range(cfg.n_slices):
            for j, k0 in enumerate(np.cumsum((0,) + pieces)[:-1]):
                rows.setdefault(int(idx[r, k0]) // qb, t[r, j])
        return jnp.asarray(np.stack([rows[b] for b in range(inter // qb)]))

    def w2_of(e):
        t = unpack_fp4_rows(host(w2p[e]))  # [R, S, H]
        return merge_slices(jnp.transpose(t, (2, 0, 1)), pieces).T

    return (
        _ByExpert(lambda e: cols(unpack_fp4_rows(host(w1p[e])))),
        _ByExpert(lambda e: cols(host(w1sp[e])[:, : cfg.hidden // qb])),
        _ByExpert(w2_of),
        _ByExpert(w2s_of),
    )
