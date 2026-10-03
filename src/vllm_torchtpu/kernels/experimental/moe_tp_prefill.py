"""Tensor-parallel fused MoE for prefill-sized token counts.

Every rank holds all experts, each sharded on the intermediate axis, and
every rank sees every token. Per rank the layer is

    prep (XLA)  -> expert kernel (Pallas) -> combine (Pallas, v2's) -> psum

The expert kernel is the fused_moe/v2 expert loop with expert parallelism
taken out: it visits the experts that received rows, gathers each expert's
fp8 activation rows from HBM by one DMA per row, runs gate/up, the
activation, the intermediate requantization and down on 128-row tiles, and
writes each tile's partial rows (unweighted, bf16) into an expert-major slab.
The shared expert is one more visit over every token, with contiguous rows.
The combine is v2's destination combine: per token, the router-weighted sum
of its topk routed rows plus its gated shared row. The partial sum over the
intermediate shards is the caller's all-reduce.

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
    rowquant_fp8,
)

FP8 = jnp.float8_e4m3fn
LANES = 128
TILE_M = 128
# Index-window entries one SMEM slot holds: a tile's rows start anywhere
# inside an aligned window, so it spans two lane blocks.
WINDOW = 2 * LANES
WEIGHT_DMA_PRIORITY = 1


def _cdiv(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _cdiv(a, b) * b


def _cdiv_t(a, b):
    return (a + b - 1) // b


def _build_expert_kernel(
    *,
    n_experts,
    hidden,
    inter,
    n_tok,
    n_routes,
    nbuf,
    interpret,
    ablate=frozenset(),
    unroll=1,
    tile_m=TILE_M,
    n_rungs=1,
    relayout=frozenset(),
):
    """The per-rank expert kernel for a fixed token count and route count.

    Slab layout (rows): [0, n_routes) routed rows expert-major, then the
    shared expert's n_tok rows at n_routes, then tile_m rows of slack.

    Each tile computes on the smallest of n_rungs heights covering its live
    rows and writes exactly its live rows (the DMA extent is traced).
    """
    TILE_M = tile_m  # noqa: N806  (shadows the module default)
    window = tile_m + LANES
    scale_win = tile_m // LANES + 1
    rungs = [tile_m * (i + 1) // n_rungs for i in range(n_rungs)]
    lane_blocks = hidden // LANES
    shared = n_experts  # visit-table id of the shared expert
    slab_rows = n_routes + n_tok + TILE_M
    scale_rows = _cdiv(slab_rows, LANES) + 1
    tok_rows = _align(n_routes, LANES) + window
    x_rows = _align(n_tok, TILE_M) + TILE_M

    def kernel(
        # scalar prefetch
        visit_e_sm,  # [E+3] expert id per visit (shared = E)
        visit_base_sm,  # [E+3] first slab row
        visit_rows_sm,  # [E+3] rows
        n_visit_sm,  # [1] visits incl. the shared one
        # HBM inputs
        x_hbm,  # [x_rows, lane_blocks, 128] fp8
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
        lhs_vm,  # [2, TILE_M, lane_blocks, 128] fp8
        ls_vm,  # [2, 2, 128] f32
        out_vm,  # [2, TILE_M, lane_blocks, 128] bf16
        w1_vm,  # [nbuf, H, 2I]
        w2_vm,  # [nbuf, I, H]
        s1_vm,  # [nbuf, 1, 2I]
        s2_vm,  # [nbuf, 1, H]
        win_sm,  # [2, WINDOW] int32
        lhs_sems,  # DMA[2]
        win_sems,  # DMA[2]
        out_sems,  # DMA[2]
        w_sems,  # DMA[nbuf]
        *extra,  # timing-only 2D scratch for the relayout ablations
    ):
        n_visit = n_visit_sm[0]

        # ---- weights: a ring of nbuf slots, nbuf-1 visits ahead ----
        def w_copies(e_src, slot, use_shared):
            if use_shared:
                srcs = (w1sh_hbm.at[0], w2sh_hbm.at[0], s1sh_hbm.at[0], s2sh_hbm.at[0])
            else:
                srcs = (w1_hbm.at[e_src], w2_hbm.at[e_src], s1_hbm.at[e_src], s2_hbm.at[e_src])
            dsts = (w1_vm.at[slot], w2_vm.at[slot], s1_vm.at[slot], s2_vm.at[slot])
            return [pltpu.make_async_copy(s, d, w_sems.at[slot]) for s, d in zip(srcs, dsts)]

        def start_weights(v):
            e = visit_e_sm[v]
            slot = lax.rem(v, nbuf)

            if "weights" in ablate:
                return

            @pl.when(e == shared)
            def _():
                for c in w_copies(0, slot, True):
                    c.start(priority=WEIGHT_DMA_PRIORITY)

            @pl.when(e != shared)
            def _():
                for c in w_copies(e, slot, False):
                    c.start(priority=WEIGHT_DMA_PRIORITY)

        def wait_weights(v):
            if "weights" in ablate:
                return
            for c in w_copies(0, lax.rem(v, nbuf), False):
                c.wait()

        # ---- tile coordinates: (visit, tile within visit) -> next ----
        def next_tile(v, t):
            more = (t + 1) * TILE_M < visit_rows_sm[v]
            return jnp.where(more, v, v + 1), jnp.where(more, t + 1, 0)

        def tile_base(v, t):
            return visit_base_sm[v] + t * TILE_M

        def tile_live(v, t):
            return jnp.minimum(visit_rows_sm[v] - t * TILE_M, TILE_M)

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
                ls_hbm.at[pl.ds(base // LANES, scale_win)], ls_vm.at[slot], lhs_sems.at[slot]
            )

        def issue_tile(v, t, slot):
            """Start tile (v, t)'s rows into lhs slot `slot`; its window is in
            win slot `slot` (both follow the global tile parity)."""
            base = tile_base(v, t)

            contiguous = is_shared(v) if "gather" not in ablate else jnp.bool_(True)

            @pl.when(contiguous)
            def _():
                tok0 = jnp.minimum(t * TILE_M, x_rows - TILE_M)
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(tok0, TILE_M)], lhs_vm.at[slot], lhs_sems.at[slot]
                ).start()

            @pl.when(jnp.logical_not(contiguous))
            def _():
                off = lax.rem(base, LANES)

                def block(i, c):
                    for u in range(unroll):
                        r = i * unroll + u
                        tok = win_sm[slot, off + r]
                        pltpu.make_async_copy(
                            x_hbm.at[tok], lhs_vm.at[slot, r], lhs_sems.at[slot]
                        ).start()
                    return c

                # Whole blocks of `unroll` rows: up to unroll-1 dead rows ride
                # along, reading in-bounds table entries past the visit.
                lax.fori_loop(0, _cdiv_t(tile_live(v, t), unroll), block, 0)

            scale_copy(base, slot).start()

        def wait_tile(v, t, slot):
            contiguous = is_shared(v) if "gather" not in ablate else jnp.bool_(True)
            n = jnp.where(contiguous, TILE_M, _cdiv_t(tile_live(v, t), unroll) * unroll)
            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, n)], lhs_vm.at[slot, pl.ds(0, n)], lhs_sems.at[slot]
            ).wait()
            scale_copy(tile_base(v, t), slot).wait()

        def out_copy(base, parity, n):
            return pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(0, n)], part_hbm.at[pl.ds(base, n)], out_sems.at[parity]
            )

        def out_wait(parity, n):
            # Semaphores count bytes: the wait only needs the extent.
            pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(0, n)], out_vm.at[parity, pl.ds(0, n)], out_sems.at[parity]
            ).wait()

        # ---- prologue ----
        for k in range(nbuf - 1):

            @pl.when(k < n_visit)
            def _(k=k):
                start_weights(jnp.int32(k))

        z = jnp.int32(0)
        start_window(z, z, 0)
        wait_window(z, z, 0)
        issue_tile(z, z, 0)
        v1, t1 = next_tile(z, z)
        start_window(v1, t1, 1)

        # ---- main loop over tiles ----
        def tile_body(carry):
            v, t, g, n_out0, n_out1 = carry
            p = lax.rem(g, 2)
            q = 1 - p
            e_first = t == 0

            @pl.when(e_first)
            def _():
                wait_weights(v)

                @pl.when(v + nbuf - 1 < n_visit)
                def _():
                    start_weights(v + nbuf - 1)

            wait_tile(v, t, p)

            vn, tn = next_tile(v, t)

            @pl.when(vn < n_visit)
            def _():
                wait_window(vn, tn, q)
                issue_tile(vn, tn, q)
                vnn, tnn = next_tile(vn, tn)
                start_window(vnn, tnn, p)

            slot = lax.rem(v, nbuf)
            base = tile_base(v, t)
            live = tile_live(v, t)

            # out_vm[p] was last written two tiles ago; free it.
            if "write" not in ablate:
                n_prev = jnp.where(p == 0, n_out0, n_out1)

                @pl.when(n_prev > 0)
                def _():
                    out_wait(p, n_prev)

            def compute(h):
                if "relayout_in" in ablate:  # timing only: a 2D-layout lhs, no relayout
                    act = extra[0][pl.ds(0, h)]
                elif "in_strided" in relayout:
                    # One strided load per lane block, placed side by side on lanes.
                    act = jnp.concatenate(
                        [lhs_vm[p, pl.ds(0, h), j, :] for j in range(lane_blocks)], axis=-1
                    )
                elif "in_f32" in relayout:
                    # Unpack, retile at 32 bits (8x8 sublane blocks), repack.
                    act = (
                        lhs_vm[p, pl.ds(0, h)].astype(jnp.float32).reshape(h, hidden).astype(FP8)
                    )
                else:
                    act = lhs_vm[p, pl.ds(0, h)].reshape(h, hidden)
                act_s = _tile_row_scales(ls_vm[p], base, h)
                if "compute" in ablate:
                    rows = jnp.zeros((h, hidden), jnp.bfloat16) + act_s[0, 0].astype(jnp.bfloat16)
                else:
                    acc2, mid_s = expert_ffn_fp8(act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot])
                    rows = acc2 * mid_s * s2_vm[slot]
                    if "out_f32" in relayout and "relayout_out" not in ablate:
                        # Retile at 32 bits, then pack: the per-row layout
                        # costs an 8x8 sublane transpose instead of a packed
                        # 16x16 one.
                        rows = rows.reshape(h, lane_blocks, LANES)
                    rows = rows.astype(jnp.bfloat16)
                if "relayout_out" in ablate:  # timing only: store in the value's own layout
                    extra[-1][pl.ds(0, h)] = rows
                    return
                if "out_strided" in relayout:
                    # One strided store per lane block: row r's block j lands
                    # in sublane j of row r's tiles, with no in-register retile.
                    for par in range(2):

                        @pl.when(p == par)
                        def _(par=par):
                            for j in range(lane_blocks):
                                out_vm[par, pl.ds(0, h), j, :] = rows[:, j * LANES : (j + 1) * LANES]

                    return
                rows = rows.reshape(h, lane_blocks, LANES)
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out_vm[par, pl.ds(0, h)] = rows

            for i, h in enumerate(rungs):
                below = rungs[i - 1] if i else 0
                if len(rungs) == 1:
                    compute(h)
                else:

                    @pl.when(jnp.logical_and(live > below, live <= h))
                    def _(h=h):
                        compute(h)

            if "write" not in ablate:
                out_copy(base, p, live).start()
                n_written = live
            else:
                n_written = jnp.int32(0)
            n_out0 = jnp.where(p == 0, n_written, n_out0)
            n_out1 = jnp.where(p == 1, n_written, n_out1)
            return vn, tn, g + 1, n_out0, n_out1

        def cond(carry):
            return carry[0] < n_visit

        _, _, _, n_out0, n_out1 = lax.while_loop(cond, tile_body, (z, z, z, z, z))

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
                pltpu.VMEM((2, TILE_M, lane_blocks, LANES), FP8),
                pltpu.VMEM((2, scale_win, LANES), jnp.float32),
                pltpu.VMEM((2, TILE_M, lane_blocks, LANES), jnp.bfloat16),
                pltpu.VMEM((nbuf, hidden, 2 * inter), FP8),
                pltpu.VMEM((nbuf, inter, hidden), FP8),
                pltpu.VMEM((nbuf, 1, 2 * inter), jnp.float32),
                pltpu.VMEM((nbuf, 1, hidden), jnp.float32),
                pltpu.SMEM((2, window), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((nbuf,)),
            )
            + ((pltpu.VMEM((tile_m, hidden), FP8),) if "relayout_in" in ablate else ())
            + ((pltpu.VMEM((tile_m, hidden), jnp.bfloat16),) if "relayout_out" in ablate else ()),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, LANES), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=None if interpret else v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"moe_tp_prefill_t{n_tok}_e{n_experts}_i{inter}_nb{nbuf}"
        + f"_u{unroll}_m{tile_m}r{n_rungs}"
        + "".join(f"_{r}" for r in sorted(relayout))
        + "".join(f"_no{a}" for a in sorted(ablate)),
    )
    return call, dict(slab_rows=slab_rows, scale_rows=scale_rows, tok_rows=tok_rows, x_rows=x_rows)


@functools.lru_cache(maxsize=None)
def build_expert_kernel(**kw):
    return _build_expert_kernel(**kw)


def routing_plan(
    topk_ids, n_experts, n_valid=None, row_scale=None, plan="sort2", route_valid=None
):
    """Expert-major slab order for the routed rows (XLA).

    Returns tok [R] (token of each slab row), pos [T, K] (slab row of each
    route), the slab rows' activation scales (row_scale [T] carried through
    the sort, or None) and the visit tables. Routes of tokens at or past
    n_valid, and routes where route_valid [T, K] is False, get the sentinel
    expert E, sort last and are never visited.

    plan="sort2" uses two key/value sorts and no scatter or gather over R:
    the first sorts routes by expert carrying the route id and its scale, the
    second sorts the resulting permutation back, which yields its inverse.
    plan="scatter" is the first version (argsort + scatters + a gather).
    """
    n_tok, k = topk_ids.shape
    n_routes = n_tok * k
    e_flat = topk_ids.reshape(-1).astype(jnp.int32)
    iota = jnp.arange(n_routes, dtype=jnp.int32)
    if n_valid is not None:
        e_flat = jnp.where(iota // k < n_valid, e_flat, n_experts)
    if route_valid is not None:
        e_flat = jnp.where(route_valid.reshape(-1), e_flat, n_experts)
    s_sorted = None
    if plan == "scatter":
        order = jnp.argsort(e_flat, stable=True).astype(jnp.int32)
        pos = jnp.zeros((n_routes,), jnp.int32).at[order].set(iota)
        counts = jnp.zeros((n_experts + 1,), jnp.int32).at[e_flat].add(1)[:n_experts]
        if row_scale is not None:
            s_sorted = row_scale[order // k]
    else:
        if row_scale is None:
            e_sorted, order = lax.sort((e_flat, iota), num_keys=1, is_stable=True)
        else:
            s_rep = jnp.broadcast_to(row_scale[:, None], (n_tok, k)).reshape(-1)
            e_sorted, order, s_sorted = lax.sort((e_flat, iota, s_rep), num_keys=1, is_stable=True)
        _, pos = lax.sort((order, iota), num_keys=1, is_stable=False)
        if plan == "sort2_cmp":
            counts = jnp.sum(
                (e_flat[None, :] == jnp.arange(n_experts, dtype=jnp.int32)[:, None]).astype(
                    jnp.int32
                ),
                axis=1,
            )
        else:
            # Group boundaries by binary search on the sorted keys.
            edges = jnp.searchsorted(
                e_sorted, jnp.arange(n_experts + 1, dtype=jnp.int32), side="left"
            ).astype(jnp.int32)
            counts = edges[1:] - edges[:-1]
    tok = order // k
    pos = pos.reshape(n_tok, k)
    base = jnp.cumsum(counts) - counts
    active = counts > 0
    # Two entries of slack past the shared visit: the kernel reads one and two
    # tiles ahead without branching on the visit count.
    (visit_e,) = jnp.nonzero(active, size=n_experts + 3, fill_value=n_experts)
    visit_e = visit_e.astype(jnp.int32)
    n_routed_visits = jnp.sum(active).astype(jnp.int32)
    counts_x = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    base_x = jnp.concatenate([base, jnp.zeros((1,), jnp.int32)])
    visit_base = base_x[visit_e]
    visit_rows = counts_x[visit_e]
    return tok, pos, s_sorted, visit_e, visit_base, visit_rows, n_routed_visits


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
    nbuf=3,
    combine_chunks=1,
    reduce_fn=None,
    interpret=False,
    ablate=frozenset(),
    stop_after=None,
    plan="sort2_cmp",
    gather_unroll=8,
    skip_zero_weight_routes=True,
    tile_m=None,
    n_rungs=4,
    relayout=frozenset({"out_f32"}),
):
    """One rank's routed + shared MoE output for T > the decode cap.

    x [T, H] bf16 (replicated), w1 [E, H, 2I] fp8, w2 [E, I, H] fp8,
    s1 [E, 2I]-shaped per-channel f32 (any of [E,2I], [E,1,2I], [E,1,1,2I]),
    s2 likewise [E, H]; shared w1_sh [H, 2I], w2_sh [I, H], s1_sh [2I],
    s2_sh [H], gate wg_sh [H, 1]. topk_weights / topk_ids [T, K].

    reduce_fn (e.g. ``lambda y: lax.psum(y, axis)``) is applied per combine
    chunk, so chunk c's all-reduce can overlap chunk c+1's combine. Without it
    the per-rank partial is returned.
    """
    n_tok, hidden = x.shape
    if tile_m is None:
        tile_m = default_tile_m(n_tok)
    n_experts, _, two_i = w1.shape
    inter = two_i // 2
    k = topk_ids.shape[1]
    n_routes = n_tok * k
    lane_blocks = hidden // LANES

    call, dims = build_expert_kernel(
        n_experts=n_experts,
        hidden=hidden,
        inter=inter,
        n_tok=n_tok,
        n_routes=n_routes,
        nbuf=nbuf,
        interpret=interpret,
        ablate=frozenset(ablate),
        unroll=gather_unroll,
        tile_m=tile_m,
        n_rungs=n_rungs,
        relayout=frozenset(relayout),
    )

    # ---- prep ----
    x_q, x_s = rowquant_fp8(x)  # [T,H] fp8, [T,1] f32
    x_s = x_s.reshape(-1)
    # Padded bucket rows keep their expert ids but carry zero routing weight
    # (TPU_MOE_SKIP_PADDED_TOKENS); their routes are not computed.
    route_valid = (topk_weights != 0) if skip_zero_weight_routes else None
    tok, pos, s_sorted, visit_e, visit_base, visit_rows, n_rv = routing_plan(
        topk_ids, n_experts, n_valid, row_scale=x_s, plan=plan, route_valid=route_valid
    )
    # The shared expert is one more visit over every token.
    visit_e = visit_e.at[n_rv].set(n_experts)
    visit_base = visit_base.at[n_rv].set(n_routes)
    visit_rows = visit_rows.at[n_rv].set(n_tok)
    n_visit = (n_rv + 1).reshape(1)

    x_pad = jnp.pad(x_q, ((0, dims["x_rows"] - n_tok), (0, 0)))
    x_pad = x_pad.reshape(dims["x_rows"], lane_blocks, LANES)
    ls = jnp.concatenate([s_sorted, x_s])
    ls = jnp.pad(ls, (0, dims["scale_rows"] * LANES - ls.shape[0]))
    ls = ls.reshape(dims["scale_rows"], LANES)
    tok_pad = jnp.pad(tok, (0, dims["tok_rows"] - n_routes))

    if stop_after == "prep":  # timing arm: everything before the kernel
        probe = (jnp.sum(x_pad[:, 0, :1].astype(jnp.float32)) + jnp.sum(ls[:, :1])
                 + jnp.sum(tok_pad[:8]) + jnp.sum(visit_base) + jnp.sum(pos[:, :1]))
        return jnp.zeros((n_tok, hidden), jnp.bfloat16) + probe.astype(jnp.bfloat16) * 0
    s1r = s1.reshape(n_experts, 1, two_i).astype(jnp.float32)
    s2r = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    part = call(
        visit_e,
        visit_base,
        visit_rows,
        n_visit,
        x_pad,
        ls,
        tok_pad,
        w1,
        w2,
        s1r,
        s2r,
        w1_sh.reshape(1, hidden, two_i),
        w2_sh.reshape(1, inter, hidden),
        s1_sh.reshape(1, 1, two_i).astype(jnp.float32),
        s2_sh.reshape(1, 1, hidden).astype(jnp.float32),
    )

    if stop_after == "kernel":  # timing arm: prep + expert kernel
        return part[:n_tok].reshape(n_tok, hidden)
    # ---- combine: topk routed rows + the gated shared row per token ----
    gate = jax.nn.sigmoid(jnp.dot(x, wg_sh, preferred_element_type=jnp.float32))  # [T,1]
    coef = topk_weights.astype(jnp.float32)
    if n_valid is not None:
        valid = (jnp.arange(n_tok) < n_valid)[:, None]
        coef = jnp.where(valid, coef, 0.0)
        gate = jnp.where(valid, gate, 0.0)
    shared_pos = n_routes + jnp.arange(n_tok, dtype=jnp.int32)[:, None]
    # Unvisited slab rows are never written; point dead routes at a written
    # row so coef 0 never meets uninitialized memory.
    dead = jnp.zeros(pos.shape, jnp.bool_)
    if n_valid is not None:
        dead = dead | jnp.logical_not(valid)
    if route_valid is not None:
        dead = dead | jnp.logical_not(route_valid)
        coef = jnp.where(route_valid, coef, 0.0)
    pos = jnp.where(dead, shared_pos, pos)
    pos_all = jnp.concatenate([pos, shared_pos], 1)
    coef_all = jnp.concatenate([coef, gate], 1)
    return combine(part, pos_all, coef_all, n_tok, hidden, combine_chunks, reduce_fn, interpret)


def combine(part, pos, coef, n_tok, hidden, n_chunks, reduce_fn, interpret=False):
    k = pos.shape[1]
    lane_blocks = hidden // LANES
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
                lane_blocks=lane_blocks,
                wire_dtype=jnp.bfloat16,
                out_dtype=jnp.bfloat16,
            )
            y = fn(part, coef[sl], pos[sl])
        else:
            rows = part.reshape(part.shape[0], hidden)[pos[sl]].astype(jnp.float32)
            y = jnp.sum(rows * coef[sl][..., None], axis=1).astype(jnp.bfloat16)
        if reduce_fn is not None:
            y = reduce_fn(y)
        outs.append(y)
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


def default_combine_chunks(n_tok, max_chunk_tokens=4096):
    """Fewest equal combine chunks of at most max_chunk_tokens tokens.

    The combine's position and coefficient tables live in SMEM (1 MiB): a
    16384-token call with 11 slots per token does not fit in one chunk.
    """
    for n in range(1, n_tok + 1):
        if n_tok % n == 0 and n_tok // n <= max_chunk_tokens:
            return n
    return 1


def default_tile_m(n_tok):
    """Tile height for a token count.

    Every tile pays a fixed cost (index window, scale window, waits), so tall
    tiles win once experts hold a few hundred rows each; below that a tall
    tile mostly computes and stages dead rows. The switch point is measured,
    for 512 experts at top-10, in kernel_work/moe-tp-prefill (probe 04).
    """
    return 256 if n_tok >= 8192 else 128
