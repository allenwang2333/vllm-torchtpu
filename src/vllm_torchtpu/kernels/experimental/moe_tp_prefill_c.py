"""Probe-04 stage-A composite of the TP MoE prefill expert kernel
(write-tpu-kernels campaign kernel_work/moe-tp-prefill, j-8228ff78 close-gaps
loop, WHITEBOARD.v4): C = F1 + S1 + G, every component bit-exact.

The incumbent, ``moe_tp_prefill.py``, is left byte-identical; this file holds
one copy of its expert-kernel builder with three independent levers, each a
build flag, so every rung of the ladder A0 -> F1 -> F1+S1 -> C and the
leave-one-out arms are the same code with flags flipped:

  f1  resident token-index table: the whole tok table ([tok_rows] int32,
      10*T + pad) is copied HBM->SMEM once, blocking, in the prologue; no
      per-tile index-window DMA or wait (probe 02's F1, verbatim).
  s1  per-tile scale path without XLU roll / lane broadcast: the caller passes
      the slab-row activation-scale table lane-replicated
      ([scale_rows*128, 128] f32, see ``replicate_ls``) and each tile DMAs its
      live rows; the wire-scale window is a VPU one-hot select (probe 03's
      S1, verbatim).
  g   lean descriptor-issue loop: the token table is a flat 1-D SMEM array
      (resident under f1, otherwise a [2*window] ring of two windows instead
      of the [2, window] 2-D array whose tiled address costs
      shift/and/shift/add/add per row), and the table pointer of each block of
      `unroll` rows is a loop carry advanced once per block, so each row
      descriptor is one SMEM load at a constant offset from the carry plus the
      DMA start. Same rows, same descriptors, same order as the incumbent.

Gates (WHITEBOARD.v4): f1 only when the resident table fits SMEM beside the
kernel's scalar-prefetch tables (``f1_fits``); s1 only at tile_m <= 128
(``s1_allowed``), i.e. T < 8192 with the default launch. ``resolve_arm``
applies them; an arm name ending in "!" bypasses the s1 gate (measurement of
the gate itself, not a candidate).
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
    WEIGHT_DMA_PRIORITY,
    _cdiv,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    DN as _DN,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    QCHUNK as _QCHUNK,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    _absmax_rows,
    _quant_apply,
    _row_scale,
    _tile_row_scales,
    expert_ffn_fp8,
    wire_row_and_scale,
)

# arm name -> levers (f1, s1, g), before the gates
ARM_LEVERS = {
    "inc": (False, False, False),
    "f1": (True, False, False),
    "f1s1": (True, True, False),
    "g": (False, False, True),
    "c": (True, True, True),
}
# The scalar-prefetch tables the expert kernel also holds in SMEM, and the
# share of SMEM left to Mosaic (scalar spills, semaphores): the convention the
# other SMEM-sized kernels in this tree use (smem_capacity - 32 KiB).
SMEM_RESERVE = 32 * 1024
SMEM_CAPACITY_V7 = 1024 * 1024  # tpu_info TPU_7X smem_capacity_bytes


def smem_budget(n_experts, n_routes, tile_m, smem_capacity=SMEM_CAPACITY_V7):
    """SMEM bytes of the expert kernel with a resident token table (f1) vs
    the incumbent's [2, window] ring, from the builder's own sizes."""
    window = tile_m + LANES
    shared_base = _cdiv(n_routes, LANES) * LANES
    tok_rows = shared_base + window
    prefetch = (3 * (n_experts + mtp.VISIT_SLACK) + 1) * 4
    return dict(
        tok_table_bytes=tok_rows * 4,
        ring_bytes=2 * window * 4,
        prefetch_bytes=prefetch,
        f1_total=tok_rows * 4 + prefetch,
        inc_total=2 * window * 4 + prefetch,
        limit=smem_capacity - SMEM_RESERVE,
    )


def f1_fits(n_experts, n_routes, tile_m, smem_capacity=SMEM_CAPACITY_V7):
    b = smem_budget(n_experts, n_routes, tile_m, smem_capacity)
    return b["f1_total"] <= b["limit"]


def s1_allowed(tile_m):
    return tile_m <= 128


def resolve_arm(arm, *, n_experts, n_routes, tile_m, inter):
    """(f1, s1, g) actually built for `arm` at this launch, after the v4 gates.
    A trailing "!" bypasses the s1 tile_m gate (gate measurement only)."""
    force = arm.endswith("!")
    f1, s1, g = ARM_LEVERS[arm.rstrip("!")]
    f1 = f1 and f1_fits(n_experts, n_routes, tile_m)
    s1 = s1 and inter == LANES and (force or s1_allowed(tile_m))
    return f1, s1, g


def replicate_ls(ls):
    """[scale_rows, 128] dense one-scale-per-slab-row table -> the
    [scale_rows * 128, 128] lane-replicated table the s1 arms read."""
    flat = ls.reshape(-1)
    return jnp.broadcast_to(flat[:, None], (flat.shape[0], LANES))


def _ffn_s1(act, act_scale, w1, w2, w1s, inter):
    """expert_ffn_fp8 with the lane-replicated activation scale (probe 03's
    s1 path of _ffn_p03, verbatim): gate/up, activation, intermediate requant,
    down. act_scale is [h, 128]. Returns (acc2, mid_scale [h, 1])."""
    assert inter <= _QCHUNK  # one chunk, as the incumbent at this shape
    assert inter == LANES, inter  # the replicated scale multiplies a 128-wide half
    acc1 = lax.dot_general(act, w1, _DN, preferred_element_type=jnp.float32)
    gate = acc1[:, 0:inter] * act_scale * w1s[:, 0:inter]
    up = acc1[:, inter : 2 * inter] * act_scale * w1s[:, inter : 2 * inter]
    chunk = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    amax = _absmax_rows(chunk)
    mid_scale, mid_sinv, usable = _row_scale(amax)
    mid_q = _quant_apply(chunk, mid_sinv.astype(jnp.bfloat16), usable)
    acc2 = lax.dot_general(mid_q, w2, _DN, preferred_element_type=jnp.float32)
    return acc2, mid_scale


def _build_expert_kernel_c(
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
    f1=False,
    s1=False,
    g=False,
):
    """Copy of moe_tp_prefill._build_expert_kernel with the levers f1 / s1 /
    g selected by flag (see the module docstring). All flags off is the
    incumbent's kernel, verbatim."""
    rep_ls = s1
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
        ls_hbm,  # [scale_rows, 128] f32 (s1: [scale_rows*128, 128] lane-replicated)
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
        ls_vm,  # [n_lhs, scale_win, 128] f32 (s1: [n_lhs, tile_m, 128])
        out_vm,  # [2, tile_m, lane_blocks, 128] fp8
        oscl_vm,  # [2, scale_win, 128] f32
        partial_scl_vm,  # [1, 128] f32
        w1_vm,  # [nbuf, H, 2I]
        w2_vm,  # [nbuf, I, H]
        s1_vm,  # [nbuf, 1, 2I]
        s2_vm,  # [nbuf, 1, H]
        win_sm,  # [2, window] int32 ; g: [2*window] flat ; f1: [tok_rows] resident
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
            if g:
                # flat ring: window slot `slot` is win_sm[slot*window : +window]
                lo = slot * window
                if not isinstance(lo, int):
                    lo = pl.multiple_of(lo, LANES)
                dst = win_sm.at[pl.ds(lo, window)]
            else:
                dst = win_sm.at[slot]
            return pltpu.make_async_copy(
                tok_hbm.at[pl.ds(aligned, window)], dst, win_sems.at[slot]
            )

        def start_window(v, t, slot):
            if f1:
                return

            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                win_copy(tile_base(v, t), slot).start()

        def wait_window(v, t, slot):
            if f1:
                return

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
                n_blk = _cdiv(tile_live(v, t), unroll)

                if g:
                    # Lean issue loop: the carry is the flat table index of the
                    # block's first row, advanced once per block; each row is
                    # one SMEM load at a constant offset from it + the DMA.
                    ptr0 = base if f1 else wslot * window + off

                    def block_g(i, ptr):
                        r0 = i * unroll
                        for u in range(unroll):
                            tok = win_sm[ptr + u]
                            pltpu.make_async_copy(
                                x_hbm.at[tok],
                                lhs_vm.at[slot, r0 + u],
                                lhs_sems.at[slot],
                            ).start()
                        return ptr + unroll

                    lax.fori_loop(0, n_blk, block_g, ptr0)
                else:

                    def block(i, c):
                        for u in range(unroll):
                            r = i * unroll + u
                            # f1: probe 02's resident 1-D index; else the 2-D ring
                            tok = win_sm[base + r] if f1 else win_sm[wslot, off + r]
                            pltpu.make_async_copy(
                                x_hbm.at[tok], lhs_vm.at[slot, r], lhs_sems.at[slot]
                            ).start()
                        return c

                    # Whole blocks of `unroll` rows: up to unroll-1 dead rows
                    # ride along, reading in-bounds table entries past the visit.
                    lax.fori_loop(0, n_blk, block, 0)

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
        if f1:
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

            @pl.when(n_prev > 0)
            def _():
                out_wait(p, n_prev)

            @pl.when(n_scl_prev > 0)
            def _():
                oscl_wait(p, n_scl_prev)

            resid = lax.rem(base, jnp.int32(LANES))

            def compute(h):
                act = lhs_vm[lslot, pl.ds(0, h)].reshape(h, hidden)
                if not rep_ls:
                    act_s = _tile_row_scales(ls_vm[lslot], base, h)
                    acc2, mid_s = expert_ffn_fp8(
                        act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot]
                    )
                else:
                    act_s = ls_vm[lslot, pl.ds(0, h)]  # [h, 128] replicated
                    acc2, mid_s = _ffn_s1(
                        act, act_s, w1_vm[slot], w2_vm[slot], s1_vm[slot], inter
                    )
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
                    wsb = jnp.broadcast_to(wire_scales, (h, wdt))
                    ii = lax.broadcasted_iota(jnp.int32, (h, wdt), 0)
                    jj = lax.broadcasted_iota(jnp.int32, (h, wdt), 1)
                    flat = jnp.sum(
                        jnp.where(jj == ii + resid, wsb, jnp.float32(0.0)),
                        axis=0,
                        keepdims=True,
                    )
                    win = jnp.concatenate(
                        [
                            flat[:, s * LANES : (s + 1) * LANES]
                            for s in range(scale_win)
                        ],
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
                pltpu.SMEM(
                    (tok_rows,) if f1 else ((2 * window,) if g else (2, window)),
                    jnp.int32,
                ),
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
            f"_u{unroll}_m{tile_m}r{n_rungs}_la{lookahead}"
            f"_c{'f' if f1 else ''}{'s' if s1 else ''}{'g' if g else ''}"
        ),
    )
    return call, dict(
        shared_base=shared_base,
        slab_rows=slab_rows,
        scale_rows=scale_rows,
        tok_rows=tok_rows,
    )


@functools.cache
def build_expert_kernel_c(**kw):
    return _build_expert_kernel_c(**kw)


def build_expert(arm=None, **kw):
    """The incumbent's cached builder when arm is None, else the composite
    builder with `arm`'s levers after the gates. Returns (call, dims) with
    dims["levers"] = (f1, s1, g) as built."""
    if arm is None:
        call, dims = mtp.build_expert_kernel(**kw)
        return call, dict(dims, levers=(False, False, False))
    f1, s1, g = resolve_arm(
        arm,
        n_experts=kw["n_experts"],
        n_routes=kw["n_routes"],
        tile_m=kw["tile_m"],
        inter=kw["inter"],
    )
    call, dims = build_expert_kernel_c(f1=f1, s1=s1, g=g, **kw)
    return call, dict(dims, levers=(f1, s1, g))


def moe_tp_prefill_c(
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
    arm="c",
    fault=None,
):
    """moe_tp_prefill with the composite's expert kernel (`arm`, gated), the
    incumbent's routing_plan and combine, and the accuracy-test faults of
    moe_tp_prefill_kw. Under s1 the lane-replicated scale table is built here,
    inside the op (it is part of the lever's cost at whole-op scope)."""
    from vllm_torchtpu.kernels.experimental import moe_tp_prefill_kw as kw

    assert fault in kw.FAULTS, fault
    n_experts, hidden, two_i = w1.shape
    if x_quantized is not None:
        x_q, x_s = x_quantized
        n_tok = x_q.shape[0]
    else:
        n_tok, hidden = x.shape
    launch = kw.resolve_launch(n_tok, nbuf, tile_m, n_rungs, gather_unroll, lookahead)
    inter = two_i // 2
    n_routes = n_tok * topk_ids.shape[1]
    call, dims = build_expert(
        arm=arm,
        **kw.expert_build_kwargs(
            n_experts, hidden, inter, n_tok, n_routes, launch, interpret
        ),
    )
    if x_quantized is None:
        x_q, x_s = mtp.rowquant_rows(x)
    if shared_gate is not None:
        gate = shared_gate
    else:
        gate = jax.nn.sigmoid(jnp.dot(x, wg_sh, preferred_element_type=jnp.float32))
    prep = kw.kw_prep(
        x_s,
        topk_weights,
        topk_ids,
        gate,
        n_experts=n_experts,
        dims=dims,
        n_valid=n_valid,
        skip_zero_weight_routes=skip_zero_weight_routes,
        fault=fault,
    )
    if dims["levers"][1]:
        prep["ls"] = replicate_ls(prep["ls"])
    part, pscl = kw.kw_expert(
        call, prep, x_q, w1, w2, s1, s2, w1_sh, w2_sh, s1_sh, s2_sh
    )
    return kw.combine_v1(
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
