"""Probe-02 arms of the TP MoE prefill expert kernel (write-tpu-kernels
campaign kernel_work/moe-tp-prefill, j-8228ff78 close-gaps loop): where does
the ~0.4 us/tile excess over the static bundle count go?

The incumbent, ``moe_tp_prefill.py``, is left byte-identical; this file holds
a copy of its expert-kernel builder with one arm selected by ``arm``:

  "inc"  verbatim copy (for traces; timing arm 0 is the incumbent itself).
  "f1"   token-index table resident in SMEM: the whole tok table
         ([tok_rows] int32, 10*T + pad) is copied HBM->SMEM once, blocking,
         in the prologue; no per-tile index-window DMA or wait. Same numerics.
  "f2"   one fixed rung for every tile (no pl.when rung-dispatch chain).
         TIMING ONLY: tiles needing another height compute the wrong rows.
  "f3"   the wait on tile g-2's slab write (out + scale DMA) moved from
         before compute to after this tile's own output DMA start.
         TIMING ONLY: out_vm[p] is overwritten while its DMA may be in flight.
  "f4k"  gate/up contracts over hidden/2 only (w1 rows [0, H/2)): 1/3 fewer
         weight bytes pushed into the MXU per tile. TIMING ONLY.
  "f4n"  intermediate width I/2 (w1 columns [0, I), w2 rows [0, I/2)): half
         the weight bytes pushed per tile. TIMING ONLY.
  "seg"  same arithmetic, the tile body staged through VMEM scratch in
         predicated blocks (gate/up | activation+requant | down | epilogue)
         so the LLO trace carries a bundle marker between stages. Same
         numerics (intended bit-exact); instrumentation, not a lever.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
    FP8,
    LANES,
    WEIGHT_DMA_PRIORITY,
    _cdiv,
)
from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    DN as _DN,
    _intermediate_rows,
    _tile_row_scales,
    expert_ffn_fp8,
    wire_row_and_scale,
)

ARMS = ("inc", "f1", "f2", "f3", "f4k", "f4n", "seg")
TIMING_ONLY = ("f2", "f3", "f4k", "f4n")


def _build_expert_kernel_p02(
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
    arm="inc",
    fixed_rung=None,
):
    """The per-rank expert kernel for a fixed token count and route count.

    Slab layout (rows): [0, n_routes) routed rows expert-major, then the
    shared expert's n_tok rows at shared_base = cdiv(n_routes, 128) * 128,
    then tile_m rows of slack. Each row is quantized to FP8 along with its
    f32 scale at the matching index in pscl_hbm [scale_rows, 128].

    Each tile computes on the smallest of n_rungs heights covering its live
    rows and writes exactly its live rows (the DMA extent is traced).
    Activation rows are gathered `lookahead` tiles ahead of compute.

    Copy of moe_tp_prefill._build_expert_kernel with the probe-02 arm
    selected by `arm` (one of ARMS; see the module docstring).
    """
    assert arm in ARMS, arm
    assert (arm == "f2") == (fixed_rung is not None), (arm, fixed_rung)
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
        win_sm,  # [2, window] int32 ; f1: [tok_rows] int32, the whole table
        lhs_sems,  # DMA[n_lhs]
        win_sems,  # DMA[2]
        out_sems,  # DMA[2]
        oscl_sems,  # DMA[2]
        w_sems,  # DMA[nbuf]
        *seg_scratch,  # seg: acc1 [tile_m, 2I] f32, mid [tile_m, I] fp8, mid_s [tile_m, 1] f32, acc2 [tile_m, H] f32
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
            if arm == "f1":
                return

            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                win_copy(tile_base(v, t), slot).start()

        def wait_window(v, t, slot):
            if arm == "f1":
                return

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
                        if arm == "f1":
                            tok = win_sm[base + r]
                        else:
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
        if arm == "f1":
            whole = pltpu.make_async_copy(tok_hbm, win_sm, win_sems.at[0])
            whole.start()
            whole.wait()
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

            def slab_waits():
                @pl.when(n_prev > 0)
                def _():
                    out_wait(p, n_prev)

                @pl.when(n_scl_prev > 0)
                def _():
                    oscl_wait(p, n_scl_prev)

            if arm != "f3":
                slab_waits()

            resid = lax.rem(base, jnp.int32(LANES))

            def compute(h):
                act_s = _tile_row_scales(ls_vm[lslot], base, h)
                if arm == "f4k":
                    # Timing only: gate/up contracts over hidden/2 (w1 rows
                    # [0, H/2)) -- 1/3 fewer weight bytes pushed per tile.
                    act = lhs_vm[lslot, pl.ds(0, h), pl.ds(0, lane_blocks // 2)].reshape(
                        h, hidden // 2
                    )
                    acc2, mid_s = expert_ffn_fp8(
                        act, act_s, w1_vm[slot, pl.ds(0, hidden // 2)], w2_vm[slot],
                        s1_vm[slot],
                    )
                elif arm == "f4n":
                    # Timing only: intermediate width I/2 (w1 columns [0, I),
                    # w2 rows [0, I/2)) -- half the weight bytes per tile.
                    act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                    acc2, mid_s = expert_ffn_fp8(
                        act, act_s, w1_vm[slot, :, pl.ds(0, inter)],
                        w2_vm[slot, pl.ds(0, inter // 2)], s1_vm[slot, :, pl.ds(0, inter)],
                    )
                elif arm == "seg":
                    # Same arithmetic, staged through VMEM in four predicated
                    # blocks so the trace carries a marker between stages.
                    acc1_vm, mid_vm, mids_vm, acc2_vm = seg_scratch
                    go = live > 0

                    @pl.when(go)
                    def _():
                        act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                        acc1_vm[pl.ds(0, h)] = lax.dot_general(
                            act, w1_vm[slot], _DN, preferred_element_type=jnp.float32
                        )

                    @pl.when(go)
                    def _():
                        mid_q, ms = _intermediate_rows(
                            acc1_vm[pl.ds(0, h)], act_s, inter, w1s=s1_vm[slot],
                            act_fn="silu", w1b=None,
                        )
                        mid_vm[pl.ds(0, h)] = mid_q
                        mids_vm[pl.ds(0, h)] = ms

                    @pl.when(go)
                    def _():
                        acc2_vm[pl.ds(0, h)] = lax.dot_general(
                            mid_vm[pl.ds(0, h)], w2_vm[slot], _DN,
                            preferred_element_type=jnp.float32,
                        )

                    acc2 = acc2_vm[pl.ds(0, h)]
                    mid_s = mids_vm[pl.ds(0, h)]
                else:
                    act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                    acc2, mid_s = expert_ffn_fp8(
                        act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot]
                    )
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

            if arm == "f2":
                # Timing only: every tile computes on one rung, no dispatch.
                compute(fixed_rung)
            for i, h in enumerate(rungs if arm != "f2" else []):
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

            if arm == "f3":
                # Timing only: tile g-2's slab write is waited for after this
                # tile's compute (out_vm[p] was overwritten while in flight).
                slab_waits()

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
                pltpu.SMEM((tok_rows,) if arm == "f1" else (2, window), jnp.int32),
                pltpu.SemaphoreType.DMA((n_lhs,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((nbuf,)),
            )
            + (
                (
                    pltpu.VMEM((tile_m, 2 * inter), jnp.float32),
                    pltpu.VMEM((tile_m, inter), FP8),
                    pltpu.VMEM((tile_m, 1), jnp.float32),
                    pltpu.VMEM((tile_m, hidden), jnp.float32),
                )
                if arm == "seg"
                else ()
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
            f"_u{unroll}_m{tile_m}r{n_rungs}_la{lookahead}_p02{arm}"
        ),
    )
    return call, dict(
        shared_base=shared_base,
        slab_rows=slab_rows,
        scale_rows=scale_rows,
        tok_rows=tok_rows,
    )


@functools.cache
def build_expert_kernel_p02(**kw):
    return _build_expert_kernel_p02(**kw)


def build_expert(arm=None, **kw):
    """The incumbent's cached builder when arm is None, else the copy."""
    if arm is None:
        return mtp.build_expert_kernel(**kw)
    return build_expert_kernel_p02(arm=arm, **kw)


def mode_rung(visit_rows, n_routed, tile_m, n_rungs):
    """The rung height most routed tiles compute on (f2's fixed rung)."""
    rungs = [tile_m * (i + 1) // n_rungs for i in range(n_rungs)]
    hist = {h: 0 for h in rungs}
    for v in range(n_routed):
        r = int(visit_rows[v])
        for t in range(_cdiv(r, tile_m)):
            live = min(r - t * tile_m, tile_m)
            hist[min(h for h in rungs if h >= live)] += 1
    return max(hist, key=hist.get), hist
