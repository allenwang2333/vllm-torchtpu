"""Probe-01 variants of the TP MoE prefill block (write-tpu-kernels campaign
kernel_work/moe-tp-prefill, j-8228ff78 close-gaps loop).

The incumbent, ``moe_tp_prefill.py``, is left byte-identical and importable;
everything here sits beside it:

* ``routing_plan_v2`` (lever L1): the incumbent's ``routing_plan`` with
  - counts / base from a searchsorted over the sorted packed keys instead of
    the [E, R] compare-sum;
  - sort #1 as a one-operand sort (no ``s_rep`` payload); the slab-order
    scales are a gather ``row_scale[tok]`` (a value copy, bit-identical);
  - sort #2 (the inverse permutation) as a one-operand packed sort of
    ``(order << rb) | within-expert rank`` when the route count needs more
    than 16 bits, guarded at run time by ``max(counts) < 2**rb``.
  Outputs equal the incumbent's on every live route; a dead route's ``pos``
  may differ (its rank is clamped) and is redirected to its token's shared
  row by the caller, as in the incumbent.
* ``combine_w`` (lever L2): the per-route weight ``w = coef * pscl[pos]`` is
  formed once per op in XLA (the same f32 product the incumbent combine
  forms on the scalar unit), so the combine kernel prefetches only
  ``pos`` and ``w`` into SMEM, not the whole slab-scale table per chunk.
* ``_build_expert_kernel_kw(removal=...)``: the incumbent expert kernel, copied
  verbatim, with timing-only removal arms (results are WRONG by design):
  ``"epi"``  -- the tile epilogue (dequant, row amax, requant) replaced by a
                plain bf16 -> fp8 cast of the down accumulator;
  ``"gather"`` -- routed tiles issue no per-row gather DMAs; compute reads
                whatever the lhs ring slot holds (a resident VMEM buffer);
  ``"wdma"`` -- no weight DMAs; compute reads whatever the weight ring holds.
* ``moe_tp_prefill_kw``: the incumbent's ``moe_tp_prefill`` with the above
  selected by keyword, plus fault injection for the accuracy test.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp
from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
    FP8,
    LANES,
    VISIT_SLACK,
    WEIGHT_DMA_PRIORITY,
    _cdiv,
    default_combine_chunks,
    default_launch,
    rowquant_rows,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    COMBINE_CHUNK,
    COMBINE_GROUP,
    COMBINE_LEAD,
    COMBINE_PARITIES,
    _tile_row_scales,
    combine_step_tokens,
    expert_ffn_fp8,
    wire_row_and_scale,
)

REMOVALS = (None, "epi", "gather", "wdma")


def _build_expert_kernel_kw(
    *,
    n_experts,
    hidden,
    inter,
    n_tok,
    n_routes,
    nbuf,
    tile_m,
    n_rungs,
    unroll,
    lookahead,
    interpret,
    weight_dma_priority=WEIGHT_DMA_PRIORITY,
    removal=None,
):
    """The per-rank expert kernel for a fixed token count and route count.

    Copy of moe_tp_prefill._build_expert_kernel; removal (timing only, wrong
    results) is one of REMOVALS -- see the module docstring.

    Slab layout (rows): [0, n_routes) routed rows expert-major, then the
    shared expert's n_tok rows at shared_base = cdiv(n_routes, 128) * 128,
    then tile_m rows of slack. Each row is quantized to FP8 along with its
    f32 scale at the matching index in pscl_hbm [scale_rows, 128].

    Each tile computes on the smallest of n_rungs heights covering its live
    rows and writes exactly its live rows (the DMA extent is traced).
    Activation rows are gathered `lookahead` tiles ahead of compute.
    """
    assert removal in REMOVALS, removal
    n_lhs = lookahead + 1
    window = tile_m + LANES  # a tile's rows start anywhere in an aligned window
    scale_win = tile_m // LANES + 1
    rungs = [tile_m * (i + 1) // n_rungs for i in range(n_rungs)]
    lane_blocks = hidden // LANES
    shared = n_experts  # visit-table id of the shared expert
    shared_base = _cdiv(n_routes, LANES) * LANES
    slab_rows = shared_base + n_tok + tile_m
    scale_rows = _cdiv(slab_rows, LANES) + 1
    tok_rows = shared_base + window

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
        # HBM outputs
        part_hbm,  # [slab_rows, lane_blocks, 128] fp8
        pscl_hbm,  # [scale_rows, 128] f32  one output row scale per slab row
        # scratch
        lhs_vm,  # [n_lhs, tile_m, lane_blocks, 128] fp8
        ls_vm,  # [n_lhs, scale_win, 128] f32
        out_vm,  # [2, tile_m, lane_blocks, 128] fp8
        oscl_vm,  # [2, scale_win, 128] f32
        partial_scl_vm,  # [1, 128] f32
        w1_vm,  # [nbuf, H, 2I]
        w2_vm,  # [nbuf, I, H]
        s1_vm,  # [nbuf, 1, 2I]
        s2_vm,  # [nbuf, 1, H]
        win_sm,  # [2, window] int32
        lhs_sems,  # DMA[n_lhs]
        win_sems,  # DMA[2]
        out_sems,  # DMA[2]
        oscl_sems,  # DMA[2]
        w_sems,  # DMA[nbuf]
    ):
        n_visit = n_visit_sm[0]

        # ---- weights: a ring of nbuf slots, nbuf-1 visits ahead ----
        def w_copies(e_src, slot, use_shared):
            if use_shared:
                srcs = (w1sh_hbm.at[0], w2sh_hbm.at[0], s1sh_hbm.at[0], s2sh_hbm.at[0])
            else:
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

        def start_weights(v):
            if removal == "wdma":
                return
            e = visit_e_sm[v]
            slot = lax.rem(v, nbuf)

            @pl.when(e == shared)
            def _():
                for c in w_copies(0, slot, True):
                    c.start(priority=weight_dma_priority)

            @pl.when(e != shared)
            def _():
                for c in w_copies(e, slot, False):
                    c.start(priority=weight_dma_priority)

        def wait_weights(v):
            if removal == "wdma":
                return
            for c in w_copies(0, lax.rem(v, nbuf), False):
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
                if removal != "gather":
                    lax.fori_loop(0, _cdiv(tile_live(v, t), unroll), block, 0)

            scale_copy(base, slot).start()

        def wait_tile(v, t, slot):
            if removal == "gather":
                # Only the shared expert's contiguous rows were started.
                @pl.when(is_shared(v))
                def _():
                    n = tile_live(v, t)
                    pltpu.make_async_copy(
                        lhs_vm.at[slot, pl.ds(0, n)],
                        lhs_vm.at[slot, pl.ds(0, n)],
                        lhs_sems.at[slot],
                    ).wait()

                scale_copy(tile_base(v, t), slot).wait()
                return
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

        def oscl_copy(sub_lo, parity, n_sub):
            return pltpu.make_async_copy(
                oscl_vm.at[parity, pl.ds(0, n_sub)],
                pscl_hbm.at[pl.ds(sub_lo, n_sub)],
                oscl_sems.at[parity],
            )

        def oscl_wait(parity, n_sub):
            pltpu.make_async_copy(
                oscl_vm.at[parity, pl.ds(0, n_sub)],
                oscl_vm.at[parity, pl.ds(0, n_sub)],
                oscl_sems.at[parity],
            ).wait()

        # ---- prologue ----
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
            v, t, g, n_out0, n_out1, n_scl0, n_scl1, vi, ti = carry
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

            # out_vm[p] and oscl_vm[p] were last written two tiles ago; free them.
            n_prev = jnp.where(p == 0, n_out0, n_out1)
            n_scl_prev = jnp.where(p == 0, n_scl0, n_scl1)

            @pl.when(n_prev > 0)
            def _():
                out_wait(p, n_prev)

            @pl.when(n_scl_prev > 0)
            def _():
                oscl_wait(p, n_scl_prev)

            resid = lax.rem(base, jnp.int32(LANES))

            def compute(h):
                act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                act_s = _tile_row_scales(ls_vm[lslot], base, h)
                acc2, mid_s = expert_ffn_fp8(
                    act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot]
                )
                if removal == "epi":
                    # Timing only: no dequant / row amax / requant.
                    wire_rows = acc2.astype(jnp.bfloat16).astype(FP8)
                    wire_scales = mid_s
                else:
                    wire_rows, wire_scales = wire_row_and_scale(
                        acc2,
                        mid_s,
                        w2s=s2_vm[slot],
                        wire_dtype=FP8,
                        tile_m=h,
                    )
                rows = wire_rows.reshape(h, lane_blocks, LANES)
                ws_win = jnp.pad(
                    wire_scales.reshape(-1), (0, scale_win * LANES - h)
                ).reshape(scale_win, LANES)
                rot = pltpu.roll(ws_win, resid, 1)
                prev_rows = jnp.concatenate([partial_scl_vm[:], rot[:-1]], axis=0)
                lane = lax.broadcasted_iota(jnp.int32, rot.shape, 1)
                merged = jnp.where(lane < resid, prev_rows, rot)
                adv = (resid + live) // LANES
                nxt_partial = merged[0:1]
                for s in range(1, scale_win):
                    nxt_partial = jnp.where(adv == s, merged[s : s + 1], nxt_partial)
                partial_scl_vm[:] = nxt_partial
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out_vm[par, pl.ds(0, h)] = rows
                        oscl_vm[par] = merged

            for i, h in enumerate(rungs):
                below = rungs[i - 1] if i else 0
                if len(rungs) == 1:
                    compute(h)
                else:

                    @pl.when(jnp.logical_and(live > below, live <= h))
                    def _(h=h):
                        compute(h)

            out_copy(base, p, live).start()
            sub_lo = base // LANES
            flush_tail = (vn >= n_visit) | (
                is_shared(vn) & jnp.logical_not(is_shared(v))
            )
            n_flush = jnp.where(
                flush_tail,
                _cdiv(resid + live, LANES),
                (resid + live) // LANES,
            )

            @pl.when(n_flush > 0)
            def _():
                oscl_copy(sub_lo, p, n_flush).start()

            n_out0 = jnp.where(p == 0, live, n_out0)
            n_out1 = jnp.where(p == 1, live, n_out1)
            n_scl0 = jnp.where(p == 0, n_flush, n_scl0)
            n_scl1 = jnp.where(p == 1, n_flush, n_scl1)
            return vn, tn, g + 1, n_out0, n_out1, n_scl0, n_scl1, vi_n, ti_n

        def cond(carry):
            return carry[0] < n_visit

        _, _, _, n_out0, n_out1, n_scl0, n_scl1, _, _ = lax.while_loop(
            cond, tile_body, (z, z, z, z, z, z, z, vi, ti)
        )

        for par, n, ns in ((0, n_out0, n_scl0), (1, n_out1, n_scl1)):

            @pl.when(n > 0)
            def _(par=par, n=n):
                out_wait(par, n)

            @pl.when(ns > 0)
            def _(par=par, ns=ns):
                oscl_wait(par, ns)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * 11,
            out_specs=[hbm, hbm],
            scratch_shapes=(
                pltpu.VMEM((n_lhs, tile_m, lane_blocks, LANES), FP8),
                pltpu.VMEM((n_lhs, scale_win, LANES), jnp.float32),
                pltpu.VMEM((2, tile_m, lane_blocks, LANES), FP8),
                pltpu.VMEM((2, scale_win, LANES), jnp.float32),
                pltpu.VMEM((1, LANES), jnp.float32),
                pltpu.VMEM((nbuf, hidden, 2 * inter), FP8),
                pltpu.VMEM((nbuf, inter, hidden), FP8),
                pltpu.VMEM((nbuf, 1, 2 * inter), jnp.float32),
                pltpu.VMEM((nbuf, 1, hidden), jnp.float32),
                pltpu.SMEM((2, window), jnp.int32),
                pltpu.SemaphoreType.DMA((n_lhs,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((nbuf,)),
            ),
            grid=(),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((slab_rows, lane_blocks, LANES), FP8),
            jax.ShapeDtypeStruct((scale_rows, LANES), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=None if interpret else v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=(
            f"moe_tp_prefill_t{n_tok}_e{n_experts}_i{inter}_nb{nbuf}"
            f"_u{unroll}_m{tile_m}r{n_rungs}_la{lookahead}_rm{removal or 'none'}"
        ),
    )
    return call, dict(
        shared_base=shared_base,
        slab_rows=slab_rows,
        scale_rows=scale_rows,
        tok_rows=tok_rows,
    )


@functools.cache
def build_expert_kernel_kw(**kw):
    return _build_expert_kernel_kw(**kw)


def build_expert(removal=None, **kw):
    """The incumbent's cached builder when removal is None, else the copy."""
    if removal is None:
        return mtp.build_expert_kernel(**kw)
    return build_expert_kernel_kw(removal=removal, **kw)


# ----------------------------------------------------------------- L1 ----
def routing_plan_v2(topk_ids, n_experts, row_scale, route_valid=None, n_valid=None):
    """routing_plan with searchsorted counts, a one-operand sort #1 and a
    one-operand packed sort #2. Same outputs as mtp.routing_plan on every
    live route (see the module docstring for dead routes)."""
    n_tok, k = topk_ids.shape
    n_routes = n_tok * k
    e_flat = topk_ids.reshape(-1).astype(jnp.int32)
    e_flat = jnp.where((e_flat >= 0) & (e_flat < n_experts), e_flat, n_experts)
    iota = jnp.arange(n_routes, dtype=jnp.int32)
    if n_valid is not None:
        e_flat = jnp.where(iota // k < n_valid, e_flat, n_experts)
    if route_valid is not None:
        e_flat = jnp.where(route_valid.reshape(-1), e_flat, n_experts)
    if n_routes < (1 << 20) and n_experts < (1 << 11):
        key1 = (e_flat.astype(jnp.uint32) << 20) | iota.astype(jnp.uint32)
        key1_sorted = lax.sort(key1, dimension=0, is_stable=False)
        order = (key1_sorted & jnp.uint32((1 << 20) - 1)).astype(jnp.int32)
        e_sorted = (key1_sorted >> 20).astype(jnp.int32)
        bnd = jnp.searchsorted(
            key1_sorted,
            jnp.arange(n_experts + 1, dtype=jnp.uint32) << 20,
            side="left",
            method="scan",
        ).astype(jnp.int32)
    else:
        e_sorted, order = lax.sort(
            (e_flat, iota), dimension=0, num_keys=1, is_stable=True
        )
        bnd = jnp.searchsorted(
            e_sorted,
            jnp.arange(n_experts + 1, dtype=jnp.int32),
            side="left",
            method="scan",
        ).astype(jnp.int32)
    tok = order // k
    s_sorted = row_scale[tok]
    counts = bnd[1:] - bnd[:-1]
    base = bnd[:-1]
    if n_routes <= (1 << 16):
        inv_packed = lax.sort(
            (order.astype(jnp.uint32) << 16) | iota.astype(jnp.uint32),
            is_stable=False,
        )
        pos = (inv_packed & jnp.uint32(0xFFFF)).astype(jnp.int32)
    else:
        ob = (n_routes - 1).bit_length()
        rb = 32 - ob
        rmask = (1 << rb) - 1

        def packed(_):
            # within-expert rank in slab order; dead routes' ranks clamp
            rank = jnp.minimum(iota - bnd[e_sorted], rmask)
            inv = lax.sort(
                (order.astype(jnp.uint32) << rb) | rank.astype(jnp.uint32),
                is_stable=False,
            )
            return bnd[e_flat] + (inv & jnp.uint32(rmask)).astype(jnp.int32)

        def two_operand(_):
            _, p = lax.sort((order, iota), dimension=0, num_keys=1, is_stable=False)
            return p

        if rb <= 0:
            pos = two_operand(None)
        else:
            pos = lax.cond(jnp.max(counts) <= rmask, packed, two_operand, None)
    active = counts > 0
    (visit_e,) = jnp.nonzero(active, size=n_experts + VISIT_SLACK, fill_value=n_experts)
    visit_e = visit_e.astype(jnp.int32)
    n_routed_visits = jnp.sum(active).astype(jnp.int32)
    valid_visit = visit_e < n_experts
    safe_e = jnp.where(valid_visit, visit_e, 0)
    visit_base = jnp.where(valid_visit, base[safe_e], 0)
    visit_counts = jnp.where(valid_visit, counts[safe_e], 0)
    return (
        tok,
        pos.reshape(n_tok, k),
        s_sorted,
        visit_e,
        visit_base,
        visit_counts,
        n_routed_visits,
    )


# ----------------------------------------------------------------- L2 ----
def _build_tp_combine_kernel_w(
    *,
    t_local,
    topk,
    hidden,
    lane_blocks,
    wire_dtype,
    out_dtype,
    interpret=False,
):
    """out[i] = sum_k w[i, k] * part[pos[i, k]], w = coef * pscl[pos] formed
    by the caller. Otherwise moe_tp_prefill._build_tp_combine_kernel verbatim
    (same f32 products, same k order)."""
    tb = combine_step_tokens(t_local)
    steps = t_local // tb
    gt = COMBINE_GROUP
    groups = tb // gt
    lanes = LANES

    def kernel(
        pos_sm,
        w_sm,
        arr_hbm,
        out_hbm,
        rows_vm,
        tile_vm,
        out_vm,
        row_sems,
        out_sems,
    ):
        def start_row(base, parity, i, k):
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[(base + i) * topk + k]],
                rows_vm.at[parity, k * tb + i],
                row_sems.at[parity],
            ).start(priority=k % 2)

        def reduce_token(base, parity, i):
            acc = None
            for k in range(topk):
                idx = (base + i) * topk + k
                w = w_sm[idx]
                term = rows_vm[parity, k * tb + i].astype(jnp.float32) * w
                acc = term if acc is None else acc + term
            tile_vm[i] = acc.astype(out_dtype)

        def one_step(step, parity):
            nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
            base = step * tb
            npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
            pltpu.make_async_copy(
                rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
            ).wait()

            @pl.when(step >= COMBINE_PARITIES)
            def _():
                pltpu.make_async_copy(
                    out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
                ).wait()

            def one_chunk(chunk, carry):
                for u in range(COMBINE_CHUNK):
                    i = chunk * COMBINE_CHUNK + u
                    for k in range(topk):
                        start_row(nxt, npar, i, k)
                    reduce_token(base, parity, i)
                return carry

            lax.fori_loop(0, tb // COMBINE_CHUNK, one_chunk, jnp.int32(0))
            for g in range(groups):
                out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
                    gt, hidden
                )
            pltpu.make_async_copy(
                out_vm.at[parity],
                out_hbm.at[pl.ds(step * tb, tb)],
                out_sems.at[parity],
            ).start()

        def step_pair(pair, carry):
            for parity in range(COMBINE_PARITIES):
                one_step(pair * COMBINE_PARITIES + parity, parity)
            return carry

        for p in range(COMBINE_LEAD):
            for i in range(tb):
                for k in range(topk):
                    start_row(p * tb, p, i, k)
        lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
        for p in range(COMBINE_LEAD):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
        for p in range(COMBINE_PARITIES):
            pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (COMBINE_PARITIES, topk * tb, lane_blocks, lanes), wire_dtype
                ),
                pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
                pltpu.VMEM((COMBINE_PARITIES, tb, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=None if interpret else v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"moe_tp_combine_w_t{t_local}_k{topk}_b{tb}",
    )

    def fn(arrivals, w, pos):
        return call(pos.astype(jnp.int32).reshape(-1), w.reshape(-1), arrivals)

    return fn


@functools.cache
def build_tp_combine_kernel_w(**kw):
    return _build_tp_combine_kernel_w(**kw)


def _chunk_tables(tab, n_chunks, scatter_ranks, c):
    n_tok, k = tab.shape
    tc = n_tok // n_chunks
    if n_chunks > 1 and scatter_ranks > 1:
        tc_local = tc // scatter_ranks
        return tab.reshape(scatter_ranks, n_chunks, tc_local, k)[:, c].reshape(tc, k)
    return tab[c * tc : (c + 1) * tc]


def combine_w(
    part,
    pscl,
    pos,
    coef,
    n_tok,
    hidden,
    n_chunks,
    reduce_fn,
    interpret=False,
    scatter_ranks=1,
    return_chunks=False,
    stale_chunk=False,
    bump=None,
):
    """moe_tp_prefill.combine with the slab-scale table folded into the
    per-route weight once per op (XLA), not prefetched per chunk."""
    k = pos.shape[1]
    if n_tok % n_chunks or (n_tok // n_chunks) % scatter_ranks:
        n_chunks = 1
    tc = n_tok // n_chunks
    pscl_flat = pscl.reshape(-1)
    if bump is not None:
        pos = pos + bump
    w_all = coef * pscl_flat[pos]
    outs = []
    for c in range(n_chunks):
        pos_c = _chunk_tables(pos, n_chunks, scatter_ranks, c)
        w_c = _chunk_tables(w_all, n_chunks, scatter_ranks, c)
        if combine_step_tokens(tc) and not interpret:
            fn = build_tp_combine_kernel_w(
                t_local=tc,
                topk=k,
                hidden=hidden,
                lane_blocks=hidden // LANES,
                wire_dtype=FP8,
                out_dtype=jnp.bfloat16,
            )
            y = fn(part, w_c, pos_c)
        else:
            rows = part.reshape(part.shape[0], hidden)[pos_c].astype(jnp.float32)
            y = jnp.sum(rows * w_c[..., None], axis=1).astype(jnp.bfloat16)
        if stale_chunk and c == n_chunks - 1 and outs:
            y = y_prev_raw  # noqa: F821  (fault: last chunk = previous chunk's rows)
        y_prev_raw = y  # noqa: F841
        if reduce_fn is not None:
            if outs:
                y, _ = lax.optimization_barrier((y, outs[-1]))
            y = reduce_fn(y)
        outs.append(y)
    if return_chunks:
        return outs
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


def combine_v1(
    part,
    pscl,
    pos,
    coef,
    n_tok,
    hidden,
    n_chunks,
    reduce_fn,
    interpret=False,
    scatter_ranks=1,
    return_chunks=False,
    stale_chunk=False,
    bump=None,
):
    """moe_tp_prefill.combine (incumbent kernels), with per-chunk returns,
    a chain bump and the stale-chunk fault for the instruments."""
    if not (return_chunks or stale_chunk or bump is not None):
        return mtp.combine(
            part, pscl, pos, coef, n_tok, hidden, n_chunks, reduce_fn, interpret,
            scatter_ranks=scatter_ranks,
        )
    k = pos.shape[1]
    if n_tok % n_chunks or (n_tok // n_chunks) % scatter_ranks:
        n_chunks = 1
    tc = n_tok // n_chunks
    pscl_flat = pscl.reshape(-1)
    if bump is not None:
        pos = pos + bump
    outs = []
    for c in range(n_chunks):
        pos_c = _chunk_tables(pos, n_chunks, scatter_ranks, c)
        coef_c = _chunk_tables(coef, n_chunks, scatter_ranks, c)
        if combine_step_tokens(tc) and not interpret:
            fn = mtp.build_tp_combine_kernel(
                t_local=tc,
                topk=k,
                hidden=hidden,
                lane_blocks=hidden // LANES,
                scale_len=pscl_flat.shape[0],
                wire_dtype=FP8,
                out_dtype=jnp.bfloat16,
            )
            y = fn(part, pscl_flat, coef_c, pos_c)
        else:
            rows = part.reshape(part.shape[0], hidden)[pos_c].astype(jnp.float32)
            w = coef_c * pscl_flat[pos_c]
            y = jnp.sum(rows * w[..., None], axis=1).astype(jnp.bfloat16)
        if stale_chunk and c == n_chunks - 1 and outs:
            y = y_prev_raw  # noqa: F821
        y_prev_raw = y  # noqa: F841
        if reduce_fn is not None:
            if outs:
                y, _ = lax.optimization_barrier((y, outs[-1]))
            y = reduce_fn(y)
        outs.append(y)
    if return_chunks:
        return outs
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


# ------------------------------------------------------------- stages ----
FAULTS = (None, "ids0", "drop_shared", "slab_off1", "stale_chunk")


def resolve_launch(n_tok, nbuf=None, tile_m=None, n_rungs=None, gather_unroll=None,
                   lookahead=None):
    launch = dict(default_launch(n_tok))
    for key, v in (("nbuf", nbuf), ("tile_m", tile_m), ("n_rungs", n_rungs),
                   ("gather_unroll", gather_unroll), ("lookahead", lookahead)):
        if v is not None:
            launch[key] = v
    return launch


def expert_build_kwargs(n_experts, hidden, inter, n_tok, n_routes, launch, interpret):
    return dict(
        n_experts=n_experts,
        hidden=hidden,
        inter=inter,
        n_tok=n_tok,
        n_routes=n_routes,
        nbuf=launch["nbuf"],
        tile_m=launch["tile_m"],
        n_rungs=launch["n_rungs"],
        unroll=launch["gather_unroll"],
        lookahead=launch["lookahead"],
        interpret=interpret,
        weight_dma_priority=launch["weight_dma_priority"],
    )


def kw_prep(x_s, topk_weights, topk_ids, gate, *, n_experts, dims, n_valid=None,
            skip_zero_weight_routes=True, routing="v1", fault=None):
    """routing_plan + table padding + combine tables, as moe_tp_prefill."""
    n_tok, _ = topk_ids.shape
    n_routes = topk_ids.size
    if fault == "ids0":
        topk_ids = jnp.zeros_like(topk_ids)
    plan = routing_plan_v2 if routing == "v2" else mtp.routing_plan
    route_valid = (topk_weights != 0) if skip_zero_weight_routes else None
    tok, pos, s_sorted, visit_e, visit_base, visit_rows, n_rv = plan(
        topk_ids, n_experts, x_s, route_valid=route_valid, n_valid=n_valid
    )
    shared_base = dims["shared_base"]
    visit_e = visit_e.at[n_rv].set(n_experts)
    visit_base = visit_base.at[n_rv].set(shared_base)
    visit_rows = visit_rows.at[n_rv].set(n_tok)
    s_sorted_pad = jnp.pad(s_sorted, (0, shared_base - n_routes))
    ls = jnp.concatenate([s_sorted_pad, x_s])
    ls = jnp.pad(ls, (0, dims["scale_rows"] * LANES - ls.shape[0])).reshape(-1, LANES)
    tok_pad = jnp.pad(tok, (0, dims["tok_rows"] - n_routes))
    n_visit = (n_rv if fault == "drop_shared" else n_rv + 1).reshape(1)

    coef = topk_weights.astype(jnp.float32)
    shared_pos = shared_base + jnp.arange(n_tok, dtype=jnp.int32)[:, None]
    dead = jnp.zeros(pos.shape, jnp.bool_)
    if n_valid is not None:
        valid = (jnp.arange(n_tok) < n_valid)[:, None]
        coef = jnp.where(valid, coef, 0.0)
        gate = jnp.where(valid, gate, 0.0)
        dead = dead | jnp.logical_not(valid)
    if route_valid is not None:
        coef = jnp.where(route_valid, coef, 0.0)
        dead = dead | jnp.logical_not(route_valid)
    if fault == "slab_off1":
        pos = pos + 1
    pos = jnp.where(dead, shared_pos, pos)
    return dict(
        visit_e=visit_e,
        visit_base=visit_base,
        visit_rows=visit_rows,
        n_visit=n_visit,
        ls=ls,
        tok_pad=tok_pad,
        pos_all=jnp.concatenate([pos, shared_pos], 1),
        coef_all=jnp.concatenate([coef, gate], 1),
    )


PREP_KEYS = ("visit_e", "visit_base", "visit_rows", "n_visit", "ls", "tok_pad",
             "pos_all", "coef_all")


def kw_expert(call, prep, x_q, w1, w2, s1, s2, w1_sh, w2_sh, s1_sh, s2_sh, bump=None):
    n_experts, hidden, two_i = w1.shape
    inter = two_i // 2
    nv = prep["n_visit"] if bump is None else prep["n_visit"] + bump
    return call(
        prep["visit_e"],
        prep["visit_base"],
        prep["visit_rows"],
        nv,
        x_q,
        prep["ls"],
        prep["tok_pad"],
        w1,
        w2,
        s1.reshape(n_experts, 1, two_i).astype(jnp.float32),
        s2.reshape(n_experts, 1, hidden).astype(jnp.float32),
        w1_sh.reshape(1, hidden, two_i),
        w2_sh.reshape(1, inter, hidden),
        s1_sh.reshape(1, 1, two_i).astype(jnp.float32),
        s2_sh.reshape(1, 1, hidden).astype(jnp.float32),
    )


def moe_tp_prefill_kw(
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
    nbuf=None,
    tile_m=None,
    n_rungs=None,
    gather_unroll=None,
    lookahead=None,
    interpret=False,
    x_quantized=None,
    shared_gate=None,
    scatter_ranks=1,
    routing="v1",
    combine_impl="v1",
    removal=None,
    fault=None,
):
    """moe_tp_prefill with the probe-01 levers selected by keyword:
    routing "v1"|"v2" (L1), combine_impl "v1"|"v2" (L2), removal (timing-only
    arms), fault (accuracy-test fault injection, one of FAULTS)."""
    assert fault in FAULTS, fault
    n_experts, hidden, two_i = w1.shape
    if x_quantized is not None:
        x_q, x_s = x_quantized
        n_tok = x_q.shape[0]
    else:
        n_tok, hidden = x.shape
    launch = resolve_launch(n_tok, nbuf, tile_m, n_rungs, gather_unroll, lookahead)
    inter = two_i // 2
    n_routes = n_tok * topk_ids.shape[1]
    call, dims = build_expert(
        removal=removal,
        **expert_build_kwargs(n_experts, hidden, inter, n_tok, n_routes, launch,
                              interpret),
    )
    if x_quantized is None:
        x_q, x_s = rowquant_rows(x)
    if shared_gate is not None:
        gate = shared_gate
    else:
        gate = jax.nn.sigmoid(jnp.dot(x, wg_sh, preferred_element_type=jnp.float32))
    prep = kw_prep(
        x_s, topk_weights, topk_ids, gate, n_experts=n_experts, dims=dims,
        n_valid=n_valid, skip_zero_weight_routes=skip_zero_weight_routes,
        routing=routing, fault=fault,
    )
    part, pscl = kw_expert(call, prep, x_q, w1, w2, s1, s2, w1_sh, w2_sh, s1_sh, s2_sh)
    comb = combine_w if combine_impl == "v2" else combine_v1
    return comb(
        part,
        pscl,
        prep["pos_all"],
        prep["coef_all"],
        n_tok,
        hidden,
        combine_chunks,
        reduce_fn,
        interpret,
        scatter_ranks=scatter_ranks,
        stale_chunk=(fault == "stale_chunk"),
    )


# ------------------------------------------------------- driver check ----
def kernel_driver(visit_rows, n_visit, *, n_tok, n_routes, hidden, inter, launch,
                  removal=None, combine_impl="v1", n_chunks=1, topk_all=11,
                  scatter_ranks=1):
    """Work driver of one expert-kernel + combine call, from the visit table
    (host ints) and the same tiling rules the kernel builders use: tiles,
    gathered rows, DMA descriptors by kind, slab bytes, weight bytes, and the
    combine's SMEM table bytes. visit_rows[:n_visit] includes the shared
    visit (last)."""
    tile_m, unroll, nbuf = launch["tile_m"], launch["gather_unroll"], launch["nbuf"]
    shared_base = _cdiv(n_routes, LANES) * LANES
    slab_rows = shared_base + n_tok + tile_m
    scale_rows = _cdiv(slab_rows, LANES) + 1
    tiles = routed_tiles = rows_gathered = live_rows = 0
    for v in range(n_visit):
        r = int(visit_rows[v])
        shared = v == n_visit - 1
        nt = _cdiv(r, tile_m)
        tiles += nt
        live_rows += r
        if not shared:
            routed_tiles += nt
            for t in range(nt):
                live = min(r - t * tile_m, tile_m)
                rows_gathered += _cdiv(live, unroll) * unroll
    shared_tiles = tiles - routed_tiles
    d = dict(
        visits=n_visit,
        tiles=tiles,
        routed_tiles=routed_tiles,
        live_rows=live_rows,
        rows_gathered=0 if removal == "gather" else rows_gathered,
        desc_row_gather=0 if removal == "gather" else rows_gathered,
        desc_shared_rows=shared_tiles,
        desc_window=routed_tiles,
        desc_scale=tiles,
        desc_out=tiles,
        desc_weights=0 if removal == "wdma" else 4 * n_visit,
        weight_bytes=0 if removal == "wdma" else n_visit * (hidden * 2 * inter + inter * hidden),
        slab_bytes=live_rows * hidden,
        epilogue="cast" if removal == "epi" else "dequant+amax+requant",
    )
    d["desc_total_expert"] = sum(v for k, v in d.items() if k.startswith("desc_"))
    tc = n_tok // n_chunks
    tab = tc * topk_all * 4
    if combine_impl == "v2":
        d["combine_smem_bytes_per_chunk"] = 2 * tab
        d["combine_scale_table_bytes_per_op"] = 0
        d["xla_w_gather_elems"] = n_tok * topk_all
    else:
        d["combine_smem_bytes_per_chunk"] = 2 * tab + scale_rows * LANES * 4
        d["combine_scale_table_bytes_per_op"] = n_chunks * scale_rows * LANES * 4
        d["xla_w_gather_elems"] = 0
    d["combine_desc_rows"] = n_tok * topk_all
    return d
