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
"""Experimental small-batch FP8 MoE: one Pallas call, no sort or permute.

Each token's routing is a dense weight table over experts. The kernel collects
the shard's hit experts, streams their weights through a ring of VMEM slots in
units (the gate|up halves and the down projection of one expert), and applies
each expert to every row with that row's routing weight as the mask. The
output is the shard's partial sum; the combine across shards stays the
caller's. Activations are quantized per row to FP8 before each matmul (W8A8).

Two entries share the streaming body: `moe_lowtok` takes the caller's local
top-k ids, and `moe_lowtok_scores` takes the router's softmax scores and runs
the top-k selection in the kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per call at or below which the MoE entry selects this kernel.
MOE_LOWTOK_MAX_TOKENS = 256
_MAX_LOCAL_EXPERTS = 128

_ROW_TILE = 32  # FP8 sublane packing
LANES = 512  # the routing table's expert axis, by global expert id
_NEG = -3.0e38  # router scores at or below this never win a pass
_LANE = 128
_VMEM_FRACTION = 0.75
_VMEM_MARGIN = 1.25  # over declared scratch and operands, for compiler tiles
_FP8 = jnp.float8_e4m3fn
_FP8_MAX = 448.0


def can_use_moe_lowtok(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Check arithmetic/layout support; the caller applies the token cap."""
    if activation != "silu" or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    h = hidden.shape[1]
    e, i, out = w2.shape
    if not 0 < e <= _MAX_LOCAL_EXPERTS or out != h or w1.shape != (e, h, 2 * i):
        return False
    if (2 * i) % 512:
        return False
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
    ):
        return False
    return s1.shape == (e, 1, 1, 2 * i) and s2.shape == (e, 1, 1, h)


def _vmem_budget(rows, hidden, inter, w13_parts):
    """Declared scratch and VMEM operands plus a compiler margin, capped.

    A shape-dependent reservation leaves room for neighbouring ops to
    prefetch their weights.
    """
    unit = w13_parts + 1
    weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden)
    scales = unit * 4 * (2 * (inter // w13_parts) + hidden)
    accumulators = rows * 4 * (2 * inter + hidden)
    # x in and out, scores, tables, one-hot.
    operands = (
        rows * 2 * hidden * 2 + rows * 4 * LANES + 4 * LANES * 8 + 2 * LANES * LANES
    )
    need = int((weights + scales + accumulators + operands) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


def _quantize_rows(v):
    """Per-row FP8 quantization of an f32 block: values and restoring scale."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / _FP8_MAX
    return (v / scale).astype(_FP8), scale


def _stream(
    gid_s,
    hits_s,
    wtab_ref,
    x_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
):
    """Stream the hit experts of this shard and accumulate their outputs.

    gid_s [1, E] (SMEM) is the table lane of each local expert; hits_s
    [1, LANES] (SMEM) is 1 on lanes some row routes to; wtab_ref [rows, LANES]
    holds the routing weights. Each expert is `w13_parts` gate|up units then
    one down unit; the ring has one slot per unit, so slot == part.
    """
    rows = x_ref.shape[0]
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)

    def collect(e, n):
        h = hits_s[0, gid_s[0, e]]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        expert = ids[e_idx]
        slot = part
        # The down unit builds w13 descriptors it never starts; clamp them.
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        w13_copies = (
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(col, half)],
                w13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(inter + col, half)],
                w13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(col, half)],
                s13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(inter + col, half)],
                s13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
        )
        return part, w13_copies, w2_copies

    def advance(e_idx, part, by):
        """The unit `by` (at most `units`) ahead, without a division."""
        p = part + by
        wrap = p >= units
        return jnp.where(wrap, e_idx + 1, e_idx), jnp.where(wrap, p - units, p)

    def start(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.start()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.start()

    def wait(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.wait()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.wait()

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = _quantize_rows(x_ref[...].astype(jnp.float32))

    for ahead in range(slots - 1):

        @pl.when(ahead < n_units)
        def _prime():
            start(jnp.int32(0), jnp.int32(ahead))

    def body(u, carry):
        e_idx, part = carry
        slot = part
        expert = ids[e_idx]

        @pl.when(u + slots - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, slots - 1))

        wait(e_idx, part)

        # One static branch per half: the lane offsets must be static.
        for p in range(w13_parts):

            @pl.when(part == p)
            def _gate_up_half(p=p):
                h = jax.lax.dot_general(
                    x_q,
                    w13_buf[slot],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                scales = s13_buf[slot]
                col = p * half
                h_acc[:, col : col + half] = h[:, :half] * x_s * scales[:, :half]
                h_acc[:, inter + col : inter + col + half] = (
                    h[:, half:] * x_s * scales[:, half:]
                )

        @pl.when(part == w13_parts)
        def _down():
            gate = h_acc[:, pl.ds(0, inter)]
            up = h_acc[:, pl.ds(inter, inter)]
            a = gate * jax.nn.sigmoid(gate) * up
            a_q, a_s = _quantize_rows(a)
            o = jax.lax.dot_general(
                a_q,
                w2_buf[slot],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            o = o * a_s * s2_buf[slot]
            # Zero weight masks the rows this expert does not serve.
            w = jnp.sum(
                jnp.where(lane == gid_s[0, expert], wtab_ref[...], 0.0),
                axis=1,
                keepdims=True,
            )
            acc[...] += w * o

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _kernel_ids(
    gid_ref,
    hit_ref,
    x_ref,
    wtab_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
):
    _stream(
        gid_ref,
        hit_ref,
        wtab_ref,
        x_ref,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        w13_buf,
        w2_buf,
        s13_buf,
        s2_buf,
        w13_sem,
        w2_sem,
        h_acc,
        acc,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
    )


def _kernel_scores(
    gid_s,
    x_ref,
    scores_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    wtab_ref,
    hits_v,
    hits_s,
    hsem,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
    topk,
):
    """Router top-k in the kernel, then the shared streaming body.

    `topk` max-and-mask passes over the softmax scores (ties to the lowest
    id), weights renormalized and rounded through bf16 as the served router
    does, written as the routing table over global expert ids.
    """
    rows = x_ref.shape[0]
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)
    x = scores_ref[...]
    x = jnp.where(x > _NEG, x, _NEG)
    masked = jnp.finfo(jnp.float32).min
    sel = jnp.zeros_like(x)
    wsum = jnp.zeros((rows, 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, LANES - 1), axis=1, keepdims=True)
        pick = lane == idx
        sel = jnp.where(pick, m, sel)
        wsum = wsum + m
        x = jnp.where(pick, masked, x)
    wt = jnp.where(wsum > 0, sel / wsum, 0.0).astype(jnp.bfloat16).astype(jnp.float32)
    wtab_ref[...] = wt
    # Lanes some row routes to; `collect` keeps the ones in this shard's gid.
    hit = jnp.any(wt != 0, axis=0, keepdims=True)
    hits_v[...] = jnp.where(hit, 1, 0).astype(jnp.int32)
    copy = pltpu.make_async_copy(hits_v, hits_s, hsem.at[0])
    copy.start()
    copy.wait()
    _stream(
        gid_s,
        hits_s,
        wtab_ref,
        x_ref,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        w13_buf,
        w2_buf,
        s13_buf,
        s2_buf,
        w13_sem,
        w2_sem,
        h_acc,
        acc,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
    )


def _layout(x, w13):
    tokens, _ = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if inter % _LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of {_LANE} lanes"
        )
    w13_parts = 2 if (inter // 2) % _LANE == 0 else 1
    rows = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
    return n_experts, inter, w13_parts, rows


def _stream_scratch(rows, hidden, inter, w13_parts):
    slots = w13_parts + 1
    half = inter // w13_parts
    return (
        pltpu.VMEM((slots, hidden, 2 * half), _FP8),
        pltpu.VMEM((slots, inter, hidden), _FP8),
        pltpu.VMEM((slots, 1, 2 * half), jnp.float32),
        pltpu.VMEM((slots, 1, hidden), jnp.float32),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.VMEM((rows, 2 * inter), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
    )


@jax.jit
def moe_lowtok(x, topk_ids_local, topk_weights, w13, s13, w2, s2):
    """This shard's MoE output for a few tokens, from local top-k ids.

    topk_ids_local is -1 for experts the shard does not hold. Scales use
    [E, 1, 1, N]. Callers must first check can_use_moe_lowtok.
    """
    tokens, hidden = x.shape
    n_experts, inter, w13_parts, rows = _layout(x, w13)
    if n_experts > LANES:
        raise ValueError(f"the routing table holds up to {LANES} experts per shard")
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(
        jnp.where(valid, topk_ids_local, 0), n_experts, dtype=jnp.float32
    )
    wtab = jnp.einsum(
        "tke,tk->te", onehot, jnp.where(valid, topk_weights, 0).astype(jnp.float32)
    )
    hit = jnp.pad(
        jnp.any(wtab != 0, axis=0).astype(jnp.int32), (0, LANES - n_experts)
    ).reshape(1, LANES)
    gid = jnp.arange(n_experts, dtype=jnp.int32).reshape(1, n_experts)
    wtab = jnp.pad(wtab, ((0, rows - tokens), (0, LANES - n_experts)))
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel_ids, n_experts=n_experts, inter=inter, w13_parts=w13_parts
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts)
        ),
        name="moe_lowtok",
    )(gid, hit, x_rows, wtab, s13, s2, w13, w2)
    return out[:tokens]


@functools.partial(jax.jit, static_argnames=("topk",))
def moe_lowtok_scores(x, scores, experts_start, w13, s13, w2, s2, *, topk):
    """This shard's MoE output for a few tokens, with the router in the kernel.

    scores [tokens, LANES] f32 is the softmax over all experts; this shard
    holds experts [experts_start, experts_start + E). A row of zero scores
    (a padded token) selects nothing.
    """
    tokens, hidden = x.shape
    n_experts, inter, w13_parts, rows = _layout(x, w13)
    if scores.shape[1] != LANES:
        raise ValueError(f"the routing table is {LANES} experts wide")
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    sc = jnp.pad(scores.astype(jnp.float32), ((0, rows - tokens), (0, 0)))
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    gid = (experts_start + jnp.arange(n_experts, dtype=jnp.int32)).reshape(1, n_experts)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel_scores,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((rows, LANES), jnp.float32),
                pltpu.VMEM((1, LANES), jnp.int32),
                pltpu.SMEM((1, LANES), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts)
        ),
        name="moe_lowtok",
    )(gid, x_rows, sc, s13, s2, w13, w2)
    return out[:tokens]
