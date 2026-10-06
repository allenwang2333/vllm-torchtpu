"""Tensor-parallel fused MoE for prefill-sized token counts.

Every rank holds all experts, each sharded on the intermediate axis, and
every rank sees every token. Per rank the layer is

    prep (XLA)  -> expert kernel (Pallas) -> combine (Pallas, v2's) -> psum

The expert kernel is the fused_moe/v2 expert loop with expert parallelism
taken out: it visits the experts that received rows, gathers each expert's
fp8 activation rows from HBM by one DMA per row, runs gate/up, the
activation, the intermediate requantization and down on row tiles, and
writes each tile's partial rows (unweighted, bf16) into an expert-major slab.
The shared expert is one more visit over every token, with contiguous rows.
The combine is v2's destination combine: per token, the router-weighted sum
of its topk routed rows plus its gated shared row. The partial sum over the
intermediate shards is the caller's all-reduce, issued per combine chunk.

Under TP the down projection's output is a partial sum over an intermediate
shard, so the slab is bf16 (production's grouped GEMM writes bf16 there too),
not v2's fp8 wire rows.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    _tile_row_scales,
    build_combine_kernel,
    combine_step_tokens,
    expert_ffn_fp8,
)

FP8 = jnp.float8_e4m3fn
LANES = 128
WEIGHT_DMA_PRIORITY = 1
# Visit-table entries past the shared visit: the kernel reads a few tiles
# ahead (row lookahead plus index window) without branching on the count.
VISIT_SLACK = 8


def _cdiv(a, b):
    return (a + b - 1) // b


def _build_expert_kernel(
    *,
    n_experts,
    hidden,
    inter,
    inter_sh,
    n_tok,
    n_routes,
    nbuf,
    tile_m,
    n_rungs,
    unroll,
    lookahead,
    interpret,
):
    """The per-rank expert kernel for a fixed token count and route count.

    Slab layout (rows): [0, n_routes) routed rows expert-major, then the
    shared expert's n_tok rows at n_routes, then tile_m rows of slack.

    Each tile computes on the smallest of n_rungs heights covering its live
    rows and writes exactly its live rows (the DMA extent is traced).
    Activation rows are gathered `lookahead` tiles ahead of compute.
    """
    n_lhs = lookahead + 1
    window = tile_m + LANES  # a tile's rows start anywhere in an aligned window
    scale_win = tile_m // LANES + 1
    rungs = [tile_m * (i + 1) // n_rungs for i in range(n_rungs)]
    lane_blocks = hidden // LANES
    shared = n_experts  # visit-table id of the shared expert
    slab_rows = n_routes + n_tok + tile_m
    scale_rows = _cdiv(slab_rows, LANES) + 1
    tok_rows = _cdiv(n_routes, LANES) * LANES + window

    def kernel(
        # scalar prefetch
        visit_e_sm,  # [E+VISIT_SLACK] expert id per visit (shared = E)
        visit_base_sm,  # [E+VISIT_SLACK] first slab row
        visit_rows_sm,  # [E+VISIT_SLACK] rows
        n_visit_sm,  # [1] visits incl. the shared one
        # HBM inputs
        x_hbm,  # [n_tok, lane_blocks, 128] fp8
        ls_hbm,  # [scale_rows, 128] f32  one activation scale per slab row
        tok_hbm,  # [tok_rows] int32  token of each routed slab row
        w1_hbm,  # [E, H, 2I] fp8
        w2_hbm,  # [E, I, H] fp8
        s1_hbm,  # [E, 1, 2I] f32
        s2_hbm,  # [E, 1, H] f32
        w1sh_hbm,  # [1, H, 2I]
        w2sh_hbm,  # [1, I, H]
        s1sh_hbm,  # [1, 1, 2I]
        s2sh_hbm,  # [1, 1, H]
        # HBM output
        part_hbm,  # [slab_rows, lane_blocks, 128] bf16
        # scratch
        lhs_vm,  # [n_lhs, tile_m, lane_blocks, 128] fp8
        ls_vm,  # [n_lhs, scale_win, 128] f32
        out_vm,  # [2, tile_m, lane_blocks, 128] bf16
        w1_vm,  # [nbuf, H, 2I]
        w2_vm,  # [nbuf, I, H]
        s1_vm,  # [nbuf, 1, 2I]
        s2_vm,  # [nbuf, 1, H]
        win_sm,  # [2, window] int32
        lhs_sems,  # DMA[n_lhs]
        win_sems,  # DMA[2]
        out_sems,  # DMA[2]
        w_sems,  # DMA[nbuf]
        w1sh_vm,  # [H, 2Is]  the shared expert, loaded once
        w2sh_vm,  # [Is, H]
        s1sh_vm,  # [1, 2Is]
        s2sh_vm,  # [1, H]
        sh_sem,  # DMA[]
    ):
        n_visit = n_visit_sm[0]

        # ---- weights: routed experts through a ring of nbuf slots, nbuf-1
        # visits ahead; the shared expert (its own intermediate size) has its
        # own buffers, loaded once at entry ----
        def w_copies(e_src, slot):
            srcs = (
                w1_hbm.at[e_src],
                w2_hbm.at[e_src],
                s1_hbm.at[e_src],
                s2_hbm.at[e_src],
            )
            dsts = (w1_vm.at[slot], w2_vm.at[slot], s1_vm.at[slot], s2_vm.at[slot])
            return [
                pltpu.make_async_copy(s, d, w_sems.at[slot]) for s, d in zip(srcs, dsts)
            ]

        shared_copies = [
            pltpu.make_async_copy(s, d, sh_sem)
            for s, d in zip(
                (w1sh_hbm.at[0], w2sh_hbm.at[0], s1sh_hbm.at[0], s2sh_hbm.at[0]),
                (w1sh_vm, w2sh_vm, s1sh_vm, s2sh_vm),
            )
        ]

        def start_weights(v):
            e = visit_e_sm[v]

            @pl.when(e != shared)
            def _():
                for c in w_copies(e, lax.rem(v, nbuf)):
                    c.start(priority=WEIGHT_DMA_PRIORITY)

        def wait_weights(v):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                for c in w_copies(0, lax.rem(v, nbuf)):
                    c.wait()

            @pl.when(is_shared(v))
            def _():
                for c in shared_copies:
                    c.wait()

        # ---- tile coordinates: (visit, tile within visit) -> next ----
        def next_tile(v, t):
            more = (t + 1) * tile_m < visit_rows_sm[v]
            return jnp.where(more, v, v + 1), jnp.where(more, t + 1, 0)

        def tile_base(v, t):
            return visit_base_sm[v] + t * tile_m

        def tile_live(v, t):
            return jnp.minimum(visit_rows_sm[v] - t * tile_m, tile_m)

        def is_shared(v):
            return visit_e_sm[v] == shared

        # ---- token-index windows (routed tiles only) ----
        def win_copy(base, slot):
            aligned = pl.multiple_of(base - lax.rem(base, LANES), LANES)
            return pltpu.make_async_copy(
                tok_hbm.at[pl.ds(aligned, window)], win_sm.at[slot], win_sems.at[slot]
            )

        def start_window(v, t, slot):
            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                win_copy(tile_base(v, t), slot).start()

        def wait_window(v, t, slot):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                win_copy(tile_base(v, t), slot).wait()

        # ---- activation rows ----
        def scale_copy(base, slot):
            return pltpu.make_async_copy(
                ls_hbm.at[pl.ds(base // LANES, scale_win)],
                ls_vm.at[slot],
                lhs_sems.at[slot],
            )

        def issue_tile(v, t, slot, wslot):
            """Start tile (v, t)'s rows into lhs slot `slot`, reading token
            numbers from index-window slot `wslot`."""
            base = tile_base(v, t)

            # The shared expert's rows are contiguous tokens; x is not padded,
            # so the last tile moves only the rows it has.
            @pl.when(is_shared(v))
            def _():
                n = tile_live(v, t)
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(t * tile_m, n)],
                    lhs_vm.at[slot, pl.ds(0, n)],
                    lhs_sems.at[slot],
                ).start()

            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                off = lax.rem(base, LANES)

                def block(i, c):
                    for u in range(unroll):
                        r = i * unroll + u
                        tok = win_sm[wslot, off + r]
                        pltpu.make_async_copy(
                            x_hbm.at[tok], lhs_vm.at[slot, r], lhs_sems.at[slot]
                        ).start()
                    return c

                # Whole blocks of `unroll` rows: up to unroll-1 dead rows ride
                # along, reading in-bounds table entries past the visit.
                lax.fori_loop(0, _cdiv(tile_live(v, t), unroll), block, 0)

            scale_copy(base, slot).start()

        def wait_tile(v, t, slot):
            n = jnp.where(
                is_shared(v), tile_live(v, t), _cdiv(tile_live(v, t), unroll) * unroll
            )
            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, n)],
                lhs_vm.at[slot, pl.ds(0, n)],
                lhs_sems.at[slot],
            ).wait()
            scale_copy(tile_base(v, t), slot).wait()

        def out_copy(base, parity, n):
            return pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(0, n)],
                part_hbm.at[pl.ds(base, n)],
                out_sems.at[parity],
            )

        def out_wait(parity, n):
            # Semaphores count bytes: the wait only needs the extent.
            pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(0, n)],
                out_vm.at[parity, pl.ds(0, n)],
                out_sems.at[parity],
            ).wait()

        # ---- prologue ----
        for c in shared_copies:
            c.start(priority=WEIGHT_DMA_PRIORITY)
        for k in range(nbuf - 1):

            @pl.when(k < n_visit)
            def _(k=k):
                start_weights(jnp.int32(k))

        # Activation rows run `lookahead` tiles ahead of compute in a ring of
        # n_lhs slots; index windows run one tile ahead of the row issue in
        # a ring of 2. Tile j uses lhs slot j % n_lhs and window slot j % 2.
        z = jnp.int32(0)
        start_window(z, z, 0)
        vi, ti = z, z  # issue cursor: the next tile whose rows to start
        for k in range(lookahead):

            @pl.when(vi < n_visit)
            def _(k=k, vi=vi, ti=ti):
                wait_window(vi, ti, k % 2)
                issue_tile(vi, ti, k % n_lhs, k % 2)
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, (k + 1) % 2)

            vi, ti = next_tile(vi, ti)

        # ---- main loop over tiles ----
        def tile_body(carry):
            v, t, g, n_out0, n_out1, vi, ti = carry
            p = lax.rem(g, 2)  # out_vm parity
            lslot = lax.rem(g, n_lhs)

            @pl.when(t == 0)
            def _():
                wait_weights(v)

                @pl.when(v + nbuf - 1 < n_visit)
                def _():
                    start_weights(v + nbuf - 1)

            wait_tile(v, t, lslot)

            vn, tn = next_tile(v, t)
            gi = g + lookahead  # global index of the tile at the issue cursor

            @pl.when(vi < n_visit)
            def _():
                wait_window(vi, ti, lax.rem(gi, 2))
                issue_tile(vi, ti, lax.rem(gi, n_lhs), lax.rem(gi, 2))
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, lax.rem(gi + 1, 2))

            vi_n, ti_n = next_tile(vi, ti)
            vi_n = jnp.where(vi < n_visit, vi_n, vi)
            ti_n = jnp.where(vi < n_visit, ti_n, ti)

            slot = lax.rem(v, nbuf)
            base = tile_base(v, t)
            live = tile_live(v, t)

            # out_vm[p] was last written two tiles ago; free it.
            n_prev = jnp.where(p == 0, n_out0, n_out1)

            @pl.when(n_prev > 0)
            def _():
                out_wait(p, n_prev)

            def compute(h, w1, w2, s1, s2):
                act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                act_s = _tile_row_scales(ls_vm[lslot], base, h)
                acc2, mid_s = expert_ffn_fp8(act, act_s, w1, w2, s1)
                # Retile into the per-row slab layout while still f32 (8x8
                # sublane blocks), then pack to bf16: cheaper than retiling the
                # packed bf16 value.
                rows = (acc2 * mid_s * s2).reshape(h, lane_blocks, LANES)
                rows = rows.astype(jnp.bfloat16)
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out_vm[par, pl.ds(0, h)] = rows

            def compute_visit(h):
                @pl.when(is_shared(v))
                def _():
                    compute(h, w1sh_vm[...], w2sh_vm[...], s1sh_vm[...], s2sh_vm[...])

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    compute(h, w1_vm[slot], w2_vm[slot], s1_vm[slot], s2_vm[slot])

            for i, h in enumerate(rungs):
                below = rungs[i - 1] if i else 0
                if len(rungs) == 1:
                    compute_visit(h)
                else:

                    @pl.when(jnp.logical_and(live > below, live <= h))
                    def _(h=h):
                        compute_visit(h)

            out_copy(base, p, live).start()
            n_out0 = jnp.where(p == 0, live, n_out0)
            n_out1 = jnp.where(p == 1, live, n_out1)
            return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n

        def cond(carry):
            return carry[0] < n_visit

        _, _, _, n_out0, n_out1, _, _ = lax.while_loop(
            cond, tile_body, (z, z, z, z, z, vi, ti)
        )

        for par, n in ((0, n_out0), (1, n_out1)):

            @pl.when(n > 0)
            def _(par=par, n=n):
                out_wait(par, n)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * 11,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((n_lhs, tile_m, lane_blocks, LANES), FP8),
                pltpu.VMEM((n_lhs, scale_win, LANES), jnp.float32),
                pltpu.VMEM((2, tile_m, lane_blocks, LANES), jnp.bfloat16),
                pltpu.VMEM((nbuf, hidden, 2 * inter), FP8),
                pltpu.VMEM((nbuf, inter, hidden), FP8),
                pltpu.VMEM((nbuf, 1, 2 * inter), jnp.float32),
                pltpu.VMEM((nbuf, 1, hidden), jnp.float32),
                pltpu.SMEM((2, window), jnp.int32),
                pltpu.SemaphoreType.DMA((n_lhs,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((nbuf,)),
                pltpu.VMEM((hidden, 2 * inter_sh), FP8),
                pltpu.VMEM((inter_sh, hidden), FP8),
                pltpu.VMEM((1, 2 * inter_sh), jnp.float32),
                pltpu.VMEM((1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, LANES), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=None if interpret else v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=(
            f"moe_tp_prefill_t{n_tok}_e{n_experts}_i{inter}_is{inter_sh}_nb{nbuf}"
            f"_u{unroll}_m{tile_m}r{n_rungs}_la{lookahead}"
        ),
    )
    return call, dict(slab_rows=slab_rows, scale_rows=scale_rows, tok_rows=tok_rows)


@functools.cache
def build_expert_kernel(**kw):
    return _build_expert_kernel(**kw)


def routing_plan(topk_ids, n_experts, row_scale, route_valid=None, n_valid=None):
    """Expert-major slab order for the routed rows (XLA), with no scatter or
    gather over the routes.

    One key/value sort orders the routes by expert, carrying each route's id
    and its token's activation scale; a second sort of that permutation gives
    its inverse (each route's slab row). Routes of tokens at or past n_valid,
    and routes where route_valid [T, K] is False, get the sentinel expert E,
    sort last and are never visited.

    Returns tok [R] (token of each slab row), pos [T, K] (slab row of each
    route), s_sorted [R] (scale of each slab row) and the visit tables.
    """
    n_tok, k = topk_ids.shape
    n_routes = n_tok * k
    e_flat = topk_ids.reshape(-1).astype(jnp.int32)
    iota = jnp.arange(n_routes, dtype=jnp.int32)
    if n_valid is not None:
        e_flat = jnp.where(iota // k < n_valid, e_flat, n_experts)
    if route_valid is not None:
        e_flat = jnp.where(route_valid.reshape(-1), e_flat, n_experts)
    s_rep = jnp.broadcast_to(row_scale[:, None], (n_tok, k)).reshape(-1)
    _, order, s_sorted = lax.sort((e_flat, iota, s_rep), num_keys=1, is_stable=True)
    _, pos = lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        (e_flat[None, :] == jnp.arange(n_experts, dtype=jnp.int32)[:, None]).astype(
            jnp.int32
        ),
        axis=1,
    )
    base = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_e,) = jnp.nonzero(active, size=n_experts + VISIT_SLACK, fill_value=n_experts)
    visit_e = visit_e.astype(jnp.int32)
    n_routed_visits = jnp.sum(active).astype(jnp.int32)
    counts_x = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    base_x = jnp.concatenate([base, jnp.zeros((1,), jnp.int32)])
    return (
        order // k,
        pos.reshape(n_tok, k),
        s_sorted,
        visit_e,
        base_x[visit_e],
        counts_x[visit_e],
        n_routed_visits,
    )


def moe_tp_prefill(
    x,
    w1,
    w2,
    s1,
    s2,
    w1_sh,
    w2_sh,
    s1_sh,
    s2_sh,
    wg_sh,
    topk_weights,
    topk_ids,
    *,
    n_valid=None,
    combine_chunks=1,
    reduce_fn=None,
    skip_zero_weight_routes=True,
    nbuf=3,
    tile_m=None,
    n_rungs=4,
    gather_unroll=8,
    lookahead=2,
    expert_start=None,
    interpret=False,
):
    """One rank's routed + shared MoE output for prefill-sized token counts.

    x [T, H] bf16 (replicated), w1 [E, H, 2I] fp8, w2 [E, I, H] fp8,
    s1 [E, 2I]-shaped per-channel f32 (any of [E,2I], [E,1,2I], [E,1,1,2I]),
    s2 likewise [E, H]; shared w1_sh [H, 2I], w2_sh [I, H], s1_sh [2I],
    s2_sh [H], gate wg_sh [H, 1]. topk_weights / topk_ids [T, K].

    With expert parallelism the rank holds experts [expert_start,
    expert_start + E) of the global routing (topk_ids are global); routes to
    other ranks' experts are not computed here. The shared expert's
    intermediate size (w2_sh.shape[0]) may differ from the routed one.

    reduce_fn (e.g. ``lambda y: lax.psum(y, axis)``) is applied per combine
    chunk; each chunk's input is tied to the previous chunk's reduced output,
    so XLA keeps the all-reduces separate and each runs under the next chunk's
    combine. Without reduce_fn the per-rank partial is returned.
    """
    n_tok, hidden = x.shape
    if tile_m is None:
        tile_m = default_tile_m(n_tok)
    n_experts, _, two_i = w1.shape
    inter = two_i // 2
    inter_sh = w2_sh.shape[0]
    n_routes = n_tok * topk_ids.shape[1]

    call, dims = build_expert_kernel(
        n_experts=n_experts,
        hidden=hidden,
        inter=inter,
        inter_sh=inter_sh,
        n_tok=n_tok,
        n_routes=n_routes,
        nbuf=nbuf,
        tile_m=tile_m,
        n_rungs=n_rungs,
        unroll=gather_unroll,
        lookahead=lookahead,
        interpret=interpret,
    )

    # ---- prep ----
    x_q, x_s = rowquant_rows(x)
    # Padded bucket rows keep their expert ids but carry zero routing weight
    # (TPU_MOE_SKIP_PADDED_TOKENS); their routes are not computed.
    route_valid = (topk_weights != 0) if skip_zero_weight_routes else None
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
        local = (topk_ids >= 0) & (topk_ids < n_experts)
        route_valid = local if route_valid is None else route_valid & local
        topk_ids = jnp.where(local, topk_ids, 0)
    tok, pos, s_sorted, visit_e, visit_base, visit_rows, n_rv = routing_plan(
        topk_ids, n_experts, x_s, route_valid=route_valid, n_valid=n_valid
    )
    # The shared expert is one more visit over every token.
    visit_e = visit_e.at[n_rv].set(n_experts)
    visit_base = visit_base.at[n_rv].set(n_routes)
    visit_rows = visit_rows.at[n_rv].set(n_tok)
    ls = jnp.concatenate([s_sorted, x_s])
    ls = jnp.pad(ls, (0, dims["scale_rows"] * LANES - ls.shape[0])).reshape(-1, LANES)
    tok_pad = jnp.pad(tok, (0, dims["tok_rows"] - n_routes))

    part = call(
        visit_e,
        visit_base,
        visit_rows,
        (n_rv + 1).reshape(1),
        x_q,
        ls,
        tok_pad,
        w1,
        w2,
        s1.reshape(n_experts, 1, two_i).astype(jnp.float32),
        s2.reshape(n_experts, 1, hidden).astype(jnp.float32),
        w1_sh.reshape(1, hidden, 2 * inter_sh),
        w2_sh.reshape(1, inter_sh, hidden),
        s1_sh.reshape(1, 1, 2 * inter_sh).astype(jnp.float32),
        s2_sh.reshape(1, 1, hidden).astype(jnp.float32),
    )

    # ---- combine: topk routed rows + the gated shared row per token ----
    gate = jax.nn.sigmoid(
        jnp.dot(x, wg_sh, preferred_element_type=jnp.float32)
    )  # [T,1]
    coef = topk_weights.astype(jnp.float32)
    shared_pos = n_routes + jnp.arange(n_tok, dtype=jnp.int32)[:, None]
    # Unvisited slab rows are never written; point dead routes at a written
    # row so coef 0 never meets uninitialized memory.
    dead = jnp.zeros(pos.shape, jnp.bool_)
    if n_valid is not None:
        valid = (jnp.arange(n_tok) < n_valid)[:, None]
        coef = jnp.where(valid, coef, 0.0)
        gate = jnp.where(valid, gate, 0.0)
        dead = dead | jnp.logical_not(valid)
    if route_valid is not None:
        coef = jnp.where(route_valid, coef, 0.0)
        dead = dead | jnp.logical_not(route_valid)
    pos = jnp.where(dead, shared_pos, pos)
    return combine(
        part,
        jnp.concatenate([pos, shared_pos], 1),
        jnp.concatenate([coef, gate], 1),
        n_tok,
        hidden,
        combine_chunks,
        reduce_fn,
        interpret,
    )


def combine(part, pos, coef, n_tok, hidden, n_chunks, reduce_fn, interpret=False):
    """out[t] = sum_k coef[t, k] * part[pos[t, k]] in n_chunks token chunks,
    each followed by reduce_fn when given."""
    k = pos.shape[1]
    if n_tok % n_chunks:
        n_chunks = 1
    tc = n_tok // n_chunks
    outs = []
    for c in range(n_chunks):
        sl = slice(c * tc, (c + 1) * tc)
        if combine_step_tokens(tc) and not interpret:
            fn = build_combine_kernel(
                t_local=tc,
                topk=k,
                hidden=hidden,
                lane_blocks=hidden // LANES,
                wire_dtype=jnp.bfloat16,
                out_dtype=jnp.bfloat16,
            )
            y = fn(part, coef[sl], pos[sl])
        else:
            rows = part.reshape(part.shape[0], hidden)[pos[sl]].astype(jnp.float32)
            y = jnp.sum(rows * coef[sl][..., None], axis=1).astype(jnp.bfloat16)
        if reduce_fn is not None:
            if outs:
                # Tie chunk c's input to chunk c-1's reduced output: XLA can
                # no longer merge the two all-reduces into one.
                y, _ = lax.optimization_barrier((y, outs[-1]))
            y = reduce_fn(y)
        outs.append(y)
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


def default_combine_chunks(n_tok, max_chunk_tokens=2048):
    """Fewest equal combine chunks of at most max_chunk_tokens tokens.

    The combine's position and coefficient tables live in SMEM (1 MiB): a
    16384-token call with 11 slots per token does not fit in one chunk. More
    chunks let each chunk's all-reduce run under the next chunk's combine.
    """
    for n in range(1, n_tok + 1):
        if n_tok % n == 0 and n_tok // n <= max_chunk_tokens:
            return n
    return 1


def default_tile_m(n_tok):
    """Tile height for a token count.

    Every tile pays a fixed cost (index window, scale window, waits), so tall
    tiles win once experts hold a few hundred rows each; below that a tall
    tile mostly computes and stages dead rows.
    """
    return 256 if n_tok >= 8192 else 128


def rowquant_rows(x):
    """Per-row fp8 quant written straight into the [T, H/128, 128] layout the
    expert kernel gathers rows from (one XLA fusion, no separate retile).
    Same arithmetic as fused_moe/v2's rowquant_fp8. Returns (q, scale [T])."""
    n_tok, hidden = x.shape
    x3 = x.reshape(n_tok, hidden // LANES, LANES)
    amax = jnp.max(jnp.abs(x3), axis=(1, 2), keepdims=True)
    scale = amax.astype(jnp.float32) / v2_host.FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    q = jnp.where(usable, x3 * sinv.astype(x.dtype), jnp.zeros_like(x3)).astype(FP8)
    return q, scale.reshape(-1)
