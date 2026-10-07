"""Probe-03 arms of the TP MoE prefill expert kernel (write-tpu-kernels
campaign kernel_work/moe-tp-prefill, j-8228ff78 close-gaps loop): is the
~800-cycle dynamic stall inside the straight-line rung body XLU result
latency on the per-tile scale path?

The incumbent, ``moe_tp_prefill.py``, is left byte-identical; this file holds
a copy of its expert-kernel builder with one arm selected by ``arm``:

  "inc"  verbatim copy (for traces; timing arm 0 is the incumbent itself).
  "s1"   the per-tile scale path without XLU ops. (a) Activation scales: the
         caller passes the slab-row scale table LANE-REPLICATED,
         ls_rep [slab rows, 128] f32 with ls_rep[r, :] = scale of row r, and
         each tile DMAs its live rows of it; the body multiplies the gate/up
         halves by ls_rep[rows] [h, 128] elementwise -- no lane roll, no
         row->column transpose, no lane broadcast. (b) Wire scales: the
         dense [scale_win, 128] window (rows rolled by resid, merged with
         the previous tile's partial sublane) is built on the VPU as a
         one-hot select-and-sum over a shifted identity, instead of the
         column->row reshape + dynamic lane roll. Same numerics: intended
         bit-exact (one non-zero term per output lane).
  "s1x"  s1 plus the two per-row amax lane reductions (intermediate and
         wire requant) replaced by per-lane maxima over the lane blocks: no
         cross-lane (XLU) reduce left in the body, VPU work kept. TIMING
         ONLY (every row scale is per lane, not per row).
  "s2"   the activation silu(gate) * up replaced by a copy of gate.
         TIMING ONLY.

Probe 03's S3 (the staged SEG body at T=16384) reuses moe_tp_prefill_p02's
"seg" arm unchanged.
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
    QCHUNK as _QCHUNK,
    _absmax_rows,
    _quant_apply,
    _row_scale,
    _tile_row_scales,
    expert_ffn_fp8,
    wire_row_and_scale,
)

ARMS = ("inc", "s1", "s1x", "s2")
TIMING_ONLY = ("s1x", "s2")
REPLICATED_LS = ("s1", "s1x")  # arms that take the lane-replicated scale table


def replicate_ls(ls):
    """[scale_rows, 128] dense one-scale-per-slab-row table -> the
    [scale_rows * 128, 128] lane-replicated table the s1 arms read."""
    flat = ls.reshape(-1)
    return jnp.broadcast_to(flat[:, None], (flat.shape[0], LANES))


def _lane_max(x):
    """Per-lane maximum of |x| over x's 128-lane blocks ([m, k*128] -> [m, 128]);
    the s1x stand-in for a row amax (no cross-lane reduce)."""
    k = x.shape[-1] // LANES
    m = jnp.abs(x[:, 0:LANES])
    for b in range(1, k):
        m = jnp.maximum(m, jnp.abs(x[:, b * LANES : (b + 1) * LANES]))
    return m


def _tile_lanes(v, k):
    return v if k == 1 else jnp.concatenate([v] * k, axis=-1)


def _ffn_p03(arm, act, act_scale, w1, w2, w1s, inter):
    """expert_ffn_fp8 for the probe arms: gate/up, activation, intermediate
    requant, down. act_scale is [h, 128] lane-replicated for s1/s1x, the
    incumbent's [h, 1] column otherwise. Returns (acc2, mid_scale) with
    mid_scale [h, 1] (inc/s1/s2) or [h, 128] per-lane (s1x)."""
    assert inter <= _QCHUNK  # one chunk, as the incumbent at this shape
    if arm in REPLICATED_LS:
        assert inter == LANES, inter  # the replicated scale multiplies a 128-wide half
    acc1 = lax.dot_general(act, w1, _DN, preferred_element_type=jnp.float32)
    gate = acc1[:, 0:inter] * act_scale * w1s[:, 0:inter]
    up = acc1[:, inter : 2 * inter] * act_scale * w1s[:, inter : 2 * inter]
    if arm == "s2":
        chunk = gate.astype(jnp.bfloat16)  # timing only: activation -> copy
    else:
        chunk = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    if arm == "s1x":
        amax = jnp.abs(chunk)  # per lane, no lane reduce (timing only)
    else:
        amax = _absmax_rows(chunk)
    mid_scale, mid_sinv, usable = _row_scale(amax)
    mid_q = _quant_apply(chunk, mid_sinv.astype(jnp.bfloat16), usable)
    acc2 = lax.dot_general(mid_q, w2, _DN, preferred_element_type=jnp.float32)
    return acc2, mid_scale


def _wire_p03_s1x(acc2, mid_scale, w2s):
    """wire_row_and_scale with per-lane scales (s1x, timing only): returns the
    fp8 rows and a [h, 128] per-lane scale."""
    k = acc2.shape[-1] // LANES
    down = acc2 * _tile_lanes(mid_scale, k) * w2s
    down_bf16 = down.astype(jnp.bfloat16)
    amax = _lane_max(down_bf16)
    scale, sinv, usable = _row_scale(amax)
    rows = _quant_apply(
        down_bf16, _tile_lanes(sinv.astype(jnp.bfloat16), k), _tile_lanes(usable, k)
    )
    return rows, scale


def _build_expert_kernel_p03(
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
):
    """Copy of moe_tp_prefill._build_expert_kernel with the probe-03 arm
    selected by `arm` (one of ARMS; see the module docstring)."""
    assert arm in ARMS, arm
    rep_ls = arm in REPLICATED_LS
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
        ls_hbm,  # [scale_rows, 128] f32 (s1/s1x: [scale_rows*128, 128] lane-replicated)
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
        ls_vm,  # [n_lhs, scale_win, 128] f32 (s1/s1x: [n_lhs, tile_m, 128])
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
            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                win_copy(tile_base(v, t), slot).start()

        def wait_window(v, t, slot):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                win_copy(tile_base(v, t), slot).wait()

        # ---- activation rows ----
        def scale_copy(v, t, slot):
            base = tile_base(v, t)
            if rep_ls:
                # this tile's live rows of the lane-replicated table
                n = tile_live(v, t)
                return pltpu.make_async_copy(
                    ls_hbm.at[pl.ds(base, n)],
                    ls_vm.at[slot, pl.ds(0, n)],
                    lhs_sems.at[slot],
                )
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

            scale_copy(v, t, slot).start()

        def wait_tile(v, t, slot):
            n = jnp.where(
                is_shared(v), tile_live(v, t), _cdiv(tile_live(v, t), unroll) * unroll
            )
            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, n)],
                lhs_vm.at[slot, pl.ds(0, n)],
                lhs_sems.at[slot],
            ).wait()
            scale_copy(v, t, slot).wait()

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
                if arm == "inc":
                    act_s = _tile_row_scales(ls_vm[lslot], base, h)
                    acc2, mid_s = expert_ffn_fp8(
                        act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot]
                    )
                else:
                    if rep_ls:
                        act_s = ls_vm[lslot, pl.ds(0, h)]  # [h, 128] replicated
                    else:
                        act_s = _tile_row_scales(ls_vm[lslot], base, h)
                    acc2, mid_s = _ffn_p03(
                        arm, act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot], inter
                    )
                if arm == "s1x":
                    wire_rows, wire_scales = _wire_p03_s1x(acc2, mid_s, s2_vm[slot])
                else:
                    wire_rows, wire_scales = wire_row_and_scale(
                        acc2,
                        mid_s,
                        w2s=s2_vm[slot],
                        wire_dtype=FP8,
                        tile_m=h,
                    )
                rows = wire_rows.reshape(h, lane_blocks, LANES)
                lane = lax.broadcasted_iota(jnp.int32, (scale_win, LANES), 1)
                if rep_ls:
                    # Dense window on the VPU: win[s, l] = ws[s*128 + l - resid]
                    # (0 outside [0, h)), as a one-hot select over a shifted
                    # identity summed down the sublanes; then the previous
                    # tile's partial sublane under lanes < resid of sublane 0.
                    wdt = scale_win * LANES
                    if arm == "s1x":
                        wsb = _tile_lanes(wire_scales, scale_win)  # [h, wdt]
                    else:
                        wsb = jnp.broadcast_to(wire_scales, (h, wdt))
                    ii = lax.broadcasted_iota(jnp.int32, (h, wdt), 0)
                    jj = lax.broadcasted_iota(jnp.int32, (h, wdt), 1)
                    flat = jnp.sum(
                        jnp.where(jj == ii + resid, wsb, jnp.float32(0.0)),
                        axis=0,
                        keepdims=True,
                    )
                    win = jnp.concatenate(
                        [flat[:, s * LANES : (s + 1) * LANES] for s in range(scale_win)],
                        axis=0,
                    )
                    sub = lax.broadcasted_iota(jnp.int32, (scale_win, LANES), 0)
                    prev = jnp.broadcast_to(partial_scl_vm[:], (scale_win, LANES))
                    merged = jnp.where((sub == 0) & (lane < resid), prev, win)
                else:
                    ws_win = jnp.pad(
                        wire_scales.reshape(-1), (0, scale_win * LANES - h)
                    ).reshape(scale_win, LANES)
                    rot = pltpu.roll(ws_win, resid, 1)
                    prev_rows = jnp.concatenate([partial_scl_vm[:], rot[:-1]], axis=0)
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
                pltpu.VMEM(
                    (n_lhs, tile_m, LANES) if rep_ls else (n_lhs, scale_win, LANES),
                    jnp.float32,
                ),
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
            f"_u{unroll}_m{tile_m}r{n_rungs}_la{lookahead}_p03{arm}"
        ),
    )
    return call, dict(
        shared_base=shared_base,
        slab_rows=slab_rows,
        scale_rows=scale_rows,
        tok_rows=tok_rows,
    )


@functools.cache
def build_expert_kernel_p03(**kw):
    return _build_expert_kernel_p03(**kw)


def build_expert(arm=None, **kw):
    """The incumbent's cached builder when arm is None, else the copy."""
    if arm is None:
        return mtp.build_expert_kernel(**kw)
    return build_expert_kernel_p03(arm=arm, **kw)
