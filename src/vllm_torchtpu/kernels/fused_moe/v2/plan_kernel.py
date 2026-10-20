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
"""The sharded routing plan as one Pallas call.

`build_routing_tables_sharded` and the `shard_*` cutters turn the gathered
routing payload into the kernel's operands with about a hundred small XLA
ops per MoE layer. At decode sizes each one moves a few hundred integers and
costs its fixed launch price, so the plan is priced by its op count rather
than by its arithmetic. This module computes the same operands in one call:
every value is identical to the XLA builder's (the tests compare them), only
the number of programs changes.

The arithmetic is integer prefix sums, per-expert lookups and per-pair ranks.
Each is a 0/1 matrix product -- a strictly triangular matrix for a prefix
sum, a one-hot for a lookup, a selector for a transpose -- run on the MXU in
bf16 with an f32 accumulator. That is exact by construction: every operand
is split into bytes, which bf16 holds exactly, each partial sum stays below
2**24 (asserted from the static shapes), and the bytes are recombined in
int32.

The two slab scatters are factored products too: a pair at slab row r lands
at [r >> 7, r & 127] of a [rows, 128] view, which is a product of two
one-hots, of size (rows + 128) x pairs rather than slab rows x pairs.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    ROWBLK,
    ROWBLK_SHIFT,
    act_scale_slab_rows,
    align_up,
    gather_clamp_bounds,
    off_shard_slab_row,
    pow2_shift,
    token_gather_window_rows,
)

_F32_EXACT = 1 << 24
# The one-hot temporaries of the largest bucket this serves (t_local 128).
_VMEM_LIMIT = 48 * 1024 * 1024


class PlanOutputs(NamedTuple):
    """One shard's plan, in the shapes the Pallas call emits.

    The [G, ep] tables, `counts` and the [1, G] rows are the kernel operands
    as they are; `pos` and `mirror_pos` are [1, pair lanes] in pair order;
    `slab_row` and `scale_bits` are [ep, pair lanes], row d holding source
    shard d's pairs, with every pair that does not compute here (and every
    padding lane) at an index past both slabs.
    """

    run_start: jax.Array
    aligned_rows: jax.Array
    contrib_off: jax.Array
    push_dst: jax.Array
    true_rows: jax.Array
    expert_rows: jax.Array
    slab_base: jax.Array
    visit: jax.Array
    counts: jax.Array
    pos: jax.Array
    mirror_pos: jax.Array
    token_gather: jax.Array
    scale_slab: jax.Array


def _iota(shape, dim):
    return lax.broadcasted_iota(jnp.int32, shape, dim)


def _bf16(mask_or_int):
    return mask_or_int.astype(jnp.float32).astype(jnp.bfloat16)


def _byte_planes(x, nbytes):
    """Non-negative int32 -> its low `nbytes` bytes as bf16 planes."""
    return [_bf16(lax.shift_right_logical(x, 8 * k) & 255) for k in range(nbytes)]


def _dot(a, b, *, nt=False):
    dims = (((1,), (1 if nt else 0,)), ((), ()))
    return lax.dot_general(a, b, dims, preferred_element_type=jnp.float32)


def _combine(parts):
    """Recombine per-byte f32 products (each an exact integer) to int32."""
    out = None
    for k, p in enumerate(parts):
        v = jnp.left_shift(p.astype(jnp.int32), 8 * k)
        out = v if out is None else out + v
    return out


def _mm_right(x, m, nbytes):
    """x @ m for int32 x and a 0/1 bf16 m, exactly."""
    return _combine([_dot(p, m) for p in _byte_planes(x, nbytes)])


def _mm_left(m, x, nbytes):
    """m @ x for a 0/1 bf16 m and int32 x, exactly."""
    return _combine([_dot(m, p) for p in _byte_planes(x, nbytes)])


def _mm_sel_t(sel, x, nbytes):
    """sel @ x.T for a 0/1 bf16 selector and int32 x, exactly."""
    return _combine([_dot(sel, p, nt=True) for p in _byte_planes(x, nbytes)])


def _row(x, r):
    """Row `r` (dynamic) of a 2-D int32 value, as [1, lanes].

    A masked sublane sum: a dynamic single-row load needs an aligned index.
    """
    return jnp.sum(jnp.where(_iota(x.shape, 0) == r, x, 0), axis=0, keepdims=True)


def _nbytes(bound):
    """Byte planes a value below `bound` needs."""
    n = 1
    while bound > 1 << (8 * n):
        n += 1
    return n


def _plan_kernel(
    rank_ref,
    x_ref,
    run_start_ref,
    aligned_ref,
    contrib_ref,
    push_dst_ref,
    true_rows_ref,
    expert_rows_ref,
    slab_base_ref,
    visit_ref,
    counts_ref,
    pos_ref,
    mirror_ref,
    tg_ref,
    scale_ref,
    *,
    ep,
    e_total,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    scale_at,
    count_at,
    lanes,
    scale_lanes,
    nb,
    nb_mirror,
    gather_lo,
    gather_hi,
):
    E, G = e_total, e_total // ep
    Gp = align_up(G, 16)
    n = t_local * topk
    me = rank_ref[0, 0]
    tshift = pow2_shift(tile_m, "tile_m")

    # Per-(dest d, expert e) true rows, as the payloads carried them.
    A = x_ref[:, count_at : count_at + E]  # [ep, E]
    Aal = (A + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    lower_ep = _bf16(_iota((ep, ep), 1) < _iota((ep, ep), 0))
    rsa = _mm_left(lower_ep, Aal, nb)  # run_start_aligned^T [ep, E]
    upper_e = _bf16(_iota((E, E), 0) < _iota((E, E), 1))
    rb = _mm_right(Aal, upper_e, nb)  # recv_base [d, (s, g)]
    era = jnp.sum(Aal, axis=0, keepdims=True)  # expert_rows_aligned [1, E]
    # e // G without a vector divide: exact for these small integers.
    grp = lambda i: ((i.astype(jnp.float32) + 0.5) * (1.0 / G)).astype(jnp.int32)  # noqa: E731
    seg = _bf16(
        (_iota((E, E), 0) < _iota((E, E), 1))
        & (grp(_iota((E, E), 0)) == grp(_iota((E, E), 1)))
    )
    local_base = _mm_right(era, seg, nb)  # [1, E]
    last = rsa + jnp.maximum(Aal, 1) - 1
    tiles = jnp.where(
        Aal > 0,
        jnp.right_shift(last, tshift) - jnp.right_shift(rsa, tshift) + 1,
        0,
    )

    # This shard's experts, transposed to [g, d] by a selector product.
    sel = _bf16(
        (_iota((Gp, E), 1) == me * G + _iota((Gp, E), 0)) & (_iota((Gp, E), 0) < G)
    )
    true_rows = _mm_sel_t(sel, A, nb)[:G]  # [G, ep]
    rs = _mm_sel_t(sel, rsa, nb)[:G]
    my_rb = _mm_sel_t(sel, rb, nb)[:G]
    aligned = (true_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    lower_g = _bf16(_iota((G, G), 1) < _iota((G, G), 0))
    grp_off = _mm_left(lower_g, aligned, nb)  # [G, ep]
    out_total = jnp.sum(aligned, axis=0, keepdims=True)  # [1, ep]
    upper_ep = _bf16(_iota((ep, ep), 0) < _iota((ep, ep), 1))
    contrib_base = _mm_right(out_total, upper_ep, nb)  # [1, ep]
    run_start_ref[...] = jnp.right_shift(rs, ROWBLK_SHIFT)
    aligned_ref[...] = jnp.right_shift(aligned, ROWBLK_SHIFT)
    contrib_ref[...] = jnp.right_shift(contrib_base + grp_off, ROWBLK_SHIFT)
    push_dst_ref[...] = jnp.right_shift(my_rb, ROWBLK_SHIFT)
    true_rows_ref[...] = true_rows

    # [ERA; local_base] for this shard's experts, as rows and as columns.
    per_e = jnp.concatenate([era, local_base], axis=0)  # [2, E]
    per_e_cols = _mm_sel_t(sel, per_e, nb)  # [Gp, 2]
    rows_col = per_e_cols[:G, 0:1]  # [G, 1]
    base_col = per_e_cols[:, 1:2]  # [Gp, 1]
    sel_t = _bf16(
        (_iota((E, Gp), 0) == me * G + _iota((E, Gp), 1)) & (_iota((E, Gp), 1) < G)
    )
    per_e_rows = _mm_right(per_e, sel_t, nb)[:, :G]  # [2, G]
    rows_row = per_e_rows[0:1]
    expert_rows_ref[...] = rows_row
    slab_base_ref[...] = per_e_rows[1:2]

    # expert_visit_list: rank the packed (rows desc, index asc) key.
    key_row = _iota((1, G), 1) - rows_row * G
    key_col = _iota((G, 1), 0) - rows_col * G
    rank_col = jnp.sum((key_row < key_col).astype(jnp.int32), axis=1, keepdims=True)
    perm = jnp.sum(
        jnp.where(rank_col == _iota((G, G), 1), _iota((G, G), 0), 0),
        axis=0,
        keepdims=True,
    )
    visit_ref[...] = jnp.minimum(perm, G - 1)

    # shard_count_vector, one row per count.
    mine = _iota((1, ep), 1) == me
    other = jnp.logical_not(mine)
    my_tiles = jnp.where(
        aligned > 0,
        jnp.right_shift(rs + jnp.maximum(aligned, 1) - 1, tshift)
        - jnp.right_shift(rs, tshift)
        + 1,
        0,
    )

    def colsum(t, m):
        return jnp.sum(jnp.where(m, t, 0), axis=0, keepdims=True)

    def my_total(t):
        """The sum of table row `me` over every expert, at lane `me`."""
        row = jnp.sum(t, axis=1, keepdims=True)  # [ep, 1]
        s = jnp.sum(jnp.where(_iota((ep, 1), 0) == me, row, 0), axis=0, keepdims=True)
        return jnp.where(mine, s, 0)

    n_active = jnp.sum((rows_col > 0).astype(jnp.int32), axis=0, keepdims=True)
    counts_ref[...] = jnp.concatenate(
        [
            colsum(true_rows, other),
            my_total(A) - colsum(true_rows, mine),
            colsum(aligned, other),
            my_total(Aal) - colsum(aligned, mine),
            jnp.where(mine, n_active, 0),
            colsum(my_tiles, other),
            my_total(tiles) - colsum(my_tiles, mine),
            colsum(my_tiles, mine),
        ],
        axis=0,
    )

    # This shard's own pairs: arrival row and scale-mirror position.
    valid = _iota((1, lanes), 1) < n
    pairs = x_ref[:, 0:lanes]  # [ep, lanes]: source shard d's pairs on row d
    eid = _row(pairs, me)  # [1, lanes]
    oh = _bf16((_iota((E, lanes), 0) == eid) & valid)  # [E, lanes]
    upper_p = _bf16(_iota((lanes, lanes), 0) < _iota((lanes, lanes), 1))
    before = _dot(oh, upper_p)  # same-expert pairs earlier, per expert
    rank = jnp.sum(oh.astype(jnp.float32) * before, axis=0, keepdims=True).astype(
        jnp.int32
    )
    # A pair routed nowhere carries the id E, which no lookup bin matches;
    # the XLA builder still ranks it among its routing block's other E ids.
    unrouted = (eid == E) & valid
    blk = lambda i: ((i.astype(jnp.float32) + 0.5) * (1.0 / block)).astype(jnp.int32)  # noqa: E731
    same_blk = _bf16(
        (_iota((lanes, lanes), 0) < _iota((lanes, lanes), 1))
        & (blk(_iota((lanes, lanes), 0)) == blk(_iota((lanes, lanes), 1)))
    )
    unrouted_rank = _dot(jnp.broadcast_to(_bf16(unrouted), (8, lanes)), same_blk)[
        0:1
    ].astype(jnp.int32)
    rank = rank + jnp.where(unrouted, unrouted_rank, 0)
    rb_me = _row(rb, me)  # [1, E]
    rsa_me = _row(rsa, me)
    mirror_base = (jnp.right_shift(rb_me, tshift) + 2 * _iota((1, E), 1)) * tile_m + (
        rsa_me & (tile_m - 1)
    )
    looked = _combine(
        [
            _dot(p, oh)
            for p in _byte_planes(
                jnp.concatenate([rb_me, mirror_base], axis=0), nb_mirror
            )
        ]
    )  # [2, lanes]
    pos_ref[...] = looked[0:1] + rank
    mirror_ref[...] = looked[1:2] + rank

    # Every shard's pairs that compute on this shard's experts: slab row and
    # row scale. Rows (d, g) stacked on sublanes, pairs on lanes.
    target = me * G + _iota((ep, Gp, lanes), 1)
    ohg = (
        (pairs[:, None, :] == target)
        & (_iota((ep, Gp, lanes), 1) < G)
        & (_iota((ep, Gp, lanes), 2) < n)
    ).astype(jnp.float32)  # [ep, Gp, lanes]
    before_g = _dot(ohg.reshape(ep * Gp, lanes).astype(jnp.bfloat16), upper_p).reshape(
        ep, Gp, lanes
    )
    sentinel = off_shard_slab_row(shard_stride, ep) - me * shard_stride
    slab_rows = []
    for d in range(ep):
        rs_d = jnp.sum(
            jnp.where(_iota((Gp, ep), 1) == d, _pad_rows(rs, Gp), 0),
            axis=1,
            keepdims=True,
        )
        base_d = (rs_d + base_col).astype(jnp.float32)  # [Gp, 1]
        o = ohg[d]
        row = jnp.sum(o * (before_g[d] + base_d), axis=0, keepdims=True)
        here = jnp.sum(o, axis=0, keepdims=True) > 0
        slab_rows.append(jnp.where(here, row.astype(jnp.int32), sentinel))
    slab = jnp.concatenate(slab_rows, axis=1)  # [1, ep * lanes]

    # Each pair's token, and that token's row-scale bits (repeated topk times).
    tok_of_pair = (
        (_iota((1, lanes), 1).astype(jnp.float32) + 0.5) * (1.0 / topk)
    ).astype(jnp.int32)
    tokens = jnp.concatenate(
        [d * t_local + jnp.where(valid, tok_of_pair, 0) for d in range(ep)], axis=1
    )
    spread = _bf16(
        (_iota((scale_lanes, lanes), 0) == tok_of_pair) & valid
    )  # [token, pair]
    bits = x_ref[:, scale_at : scale_at + scale_lanes]  # [ep, scale_lanes]
    scale_pp = _combine([_dot(p, spread) for p in _byte_planes(bits, 4)])
    scale_bits = jnp.concatenate([scale_pp[d : d + 1] for d in range(ep)], axis=1)

    # The two slab scatters. A pair at slab row r lands at [r >> 7, r & 127] of
    # the [rows, 128] view, so the scatter is (one-hot of r >> 7, times the
    # value) @ (one-hot of r & 127).T, with at most one pair per slot.
    lo = _bf16(_iota((HIDDEN_LANE_BLOCK, ep * lanes), 0) == (slab & 127))
    hi = jnp.right_shift(slab, 7)
    tg = _scatter_rows(hi, lo, tokens, tg_ref.shape[0], 2)
    tg_ref[...] = jnp.clip(tg, gather_lo, gather_hi)
    scale_ref[...] = lax.bitcast_convert_type(
        _scatter_rows(hi, lo, scale_bits, scale_ref.shape[0], 4), jnp.float32
    )


def _scatter_rows(hi, lo, val, rows, nbytes):
    """out[hi[q], j] = val[q] where lo[j, q]; zero where nothing lands."""
    rp = align_up(rows, 8)
    at = (_iota((rp, hi.shape[1]), 0) == hi).astype(jnp.float32)
    parts = [
        _dot((at * p.astype(jnp.float32)).astype(jnp.bfloat16), lo, nt=True)
        for p in _byte_planes(val, nbytes)
    ]
    return _combine(parts)[:rows]


def _pad_rows(x, rows):
    if x.shape[0] == rows:
        return x
    return jnp.concatenate(
        [x, jnp.zeros((rows - x.shape[0],) + x.shape[1:], x.dtype)], axis=0
    )


def pallas_plan(
    exchange,
    rank,
    *,
    ep,
    e_total,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    scale_at,
    count_at,
    interpret=False,
):
    """The sharded plan of shard `rank` from the gathered routing payload.

    `exchange` is the [ep, width] i32 payload `_pack_routing_exchange` laid
    out (pairs at lane 0, row-scale bits at `scale_at`, per-expert counts at
    `count_at`, every section lane-block aligned) and `rank` this shard's
    [1, 1] i32 mesh index. Expert ids must lie in [0, e_total], e_total
    meaning routed nowhere.
    """
    E, G = e_total, e_total // ep
    n = t_local * topk
    lanes = align_up(n, HIDDEN_LANE_BLOCK)
    scale_lanes = align_up(t_local, HIDDEN_LANE_BLOCK)
    width = exchange.shape[1]
    if (
        exchange.shape[0] != ep
        or lanes > scale_at
        or (scale_at != count_at and scale_at + scale_lanes > count_at)
        or count_at + E > width
    ):
        raise ValueError(f"exchange {exchange.shape} does not match the layout")
    # Every table value is below `bound`; one byte plane's partial sums stay
    # below 255 * (longest contraction), which must be f32-exact.
    bound = shard_stride + ep * t_local * topk + 2 * ROWBLK * E * ep
    if 255 * max(E, lanes, ep, G) >= _F32_EXACT:
        raise ValueError("contraction too long for an exact f32 accumulator")
    nb = _nbytes(bound)
    nb_mirror = _nbytes(bound + 2 * E * tile_m + tile_m)
    if nb_mirror > 3:
        raise ValueError(f"plan values up to {bound} need more than 3 byte planes")
    i32 = jnp.int32
    shapes = PlanOutputs(
        run_start=(G, ep),
        aligned_rows=(G, ep),
        contrib_off=(G, ep),
        push_dst=(G, ep),
        true_rows=(G, ep),
        expert_rows=(1, G),
        slab_base=(1, G),
        visit=(1, G),
        counts=(8, ep),
        pos=(1, lanes),
        mirror_pos=(1, lanes),
        token_gather=(
            (shard_stride + token_gather_window_rows(tile_m)) // HIDDEN_LANE_BLOCK,
            HIDDEN_LANE_BLOCK,
        ),
        scale_slab=(act_scale_slab_rows(shard_stride), HIDDEN_LANE_BLOCK),
    )
    dtypes = [i32] * (len(shapes) - 1) + [jnp.float32]
    gather_lo, gather_hi = gather_clamp_bounds(ep * t_local)
    kernel = functools.partial(
        _plan_kernel,
        ep=ep,
        e_total=E,
        t_local=t_local,
        topk=topk,
        block=block,
        tile_m=tile_m,
        shard_stride=shard_stride,
        scale_at=scale_at,
        count_at=count_at,
        lanes=lanes,
        scale_lanes=scale_lanes,
        nb=nb,
        nb_mirror=nb_mirror,
        gather_lo=gather_lo,
        gather_hi=gather_hi,
    )
    outs = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * len(shapes),
        out_shape=[
            jax.ShapeDtypeStruct(s, t) for s, t in zip(shapes, dtypes, strict=True)
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT),
        interpret=interpret,
    )(rank.astype(i32).reshape(1, 1), exchange.astype(i32))
    return PlanOutputs(*outs)
