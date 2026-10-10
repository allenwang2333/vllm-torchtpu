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
    DN,
    _intermediate_rows,
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
        w1sh_hbm,  # [1, H, 2Is]
        w2sh_hbm,  # [1, Is, H]
        s1sh_hbm,  # [1, 1, 2Is]
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


# ---- Blocked mode: experts streamed in intermediate blocks ----------------
#
# For experts whose whole-expert weight slots do not fit VMEM (GLM-5.2 at EP:
# H 6144, I 2048, one slot = 37.7 MB of FP8). Each routed visit is cut into
# chunks of at most `chunk_rows` rows; a chunk's activation rows stay resident
# (double-buffered across chunks) with a float32 running sum of its rows in
# VMEM, and the expert streams through a ring of (expert, block) units: block
# b holds w13's gate columns [b*ib, (b+1)*ib) beside the matching up columns
# and w2's rows [b*ib, (b+1)*ib). Every block adds its partial down product
# for every tile of the chunk; the last block finishes each tile into the
# bf16 slab, which keeps the whole-expert mode's layout, so the combine is
# unchanged. Exact in real arithmetic (SiLU(gate) * up is elementwise in the
# intermediate and the down projection sums over it); the intermediate is
# quantized per row per block instead of per row per expert. Each chunk reads
# its expert's weights once. The shared expert (its own, small, intermediate)
# is loaded once and runs as one unit per chunk of its rows, after the routed
# chunks.


def _build_blocked_expert_kernel(
    *,
    n_experts,
    hidden,
    inter,
    inter_sh,
    n_tok,
    n_routes,
    ib,
    nbuf,
    chunk_rows,
    tile_m,
    unroll,
    interpret,
):
    n_blk = inter // ib
    big_r = chunk_rows
    tiles = big_r // tile_m
    window = big_r + LANES  # a chunk's rows start anywhere in an aligned window
    scale_win = tile_m // LANES + 1
    lane_blocks = hidden // LANES
    shared = n_experts
    slab_rows = n_routes + n_tok + tile_m
    scale_rows = _cdiv(slab_rows, LANES) + 1
    tok_rows = _cdiv(n_routes, LANES) * LANES + window
    n_chunks = n_experts + _cdiv(n_routes, big_r) + _cdiv(n_tok, big_r) + VISIT_SLACK

    def kernel(
        # scalar prefetch
        visit_e_sm,  # [E+VISIT_SLACK] expert per routed visit
        visit_base_sm,  # [E+VISIT_SLACK] first slab row
        visit_rows_sm,  # [E+VISIT_SLACK] rows
        n_visit_sm,  # [1] routed visits
        # HBM inputs
        x_hbm,
        ls_hbm,
        tok_hbm,
        w1_hbm,
        w2_hbm,
        s1_hbm,
        s2_hbm,
        w1sh_hbm,
        w2sh_hbm,
        s1sh_hbm,
        s2sh_hbm,
        # HBM output
        part_hbm,
        # scratch
        lhs_vm,  # [2, R, lane_blocks, 128] fp8
        ls_vm,  # [2, tiles, scale_win, 128] f32
        acc_vm,  # [R, H] f32
        out_vm,  # [R, lane_blocks, 128] bf16
        w1_vm,  # [nbuf, H, 2ib]
        w2_vm,  # [nbuf, ib, H]
        s1_vm,  # [nbuf, 1, 2ib]
        s2_vm,  # [nbuf, 1, H]
        win_sm,  # [2, window] int32
        lhs_sems,  # DMA[2]
        win_sems,  # DMA[2]
        out_sem,  # DMA[]
        w_sems,  # DMA[nbuf]
        w1sh_vm,
        w2sh_vm,
        s1sh_vm,
        s2sh_vm,
        sh_sem,
        chunk_e_sm,  # [n_chunks] expert per chunk (shared = E)
        chunk_base_sm,  # [n_chunks] first slab row
        chunk_rows_sm,  # [n_chunks] live rows
    ):
        # ---- chunk table (SMEM, scalar core): every routed visit cut into
        # chunks of at most big_r rows, in visit order, then the shared
        # expert's rows (slab rows n_routes + [0, n_tok)) ----
        def put(c, e, base, rows):
            chunk_e_sm[c] = e
            chunk_base_sm[c] = base
            chunk_rows_sm[c] = rows
            return c + 1

        def fill_visit(v, c):
            e, base, rows = visit_e_sm[v], visit_base_sm[v], visit_rows_sm[v]

            def fill(q, c):
                return put(c, e, base + q * big_r, jnp.minimum(big_r, rows - q * big_r))

            return lax.fori_loop(0, _cdiv(rows, big_r), fill, c)

        n_routed = lax.fori_loop(0, n_visit_sm[0], fill_visit, jnp.int32(0))

        def fill_shared(q, c):
            return put(
                c, shared, n_routes + q * big_r, jnp.minimum(big_r, n_tok - q * big_r)
            )

        n_chunk = lax.fori_loop(0, _cdiv(n_tok, big_r), fill_shared, n_routed)
        n_units = n_routed * n_blk

        def is_shared(c):
            return chunk_e_sm[c] == shared

        # ---- weights: unit u = block u % n_blk of routed chunk u // n_blk ----
        def unit_copies(u):
            e = chunk_e_sm[u // n_blk]
            b = lax.rem(u, n_blk)
            slot = lax.rem(u, nbuf)
            gate = pl.multiple_of(b * ib, ib)
            up = pl.multiple_of(inter + b * ib, ib)
            sem = w_sems.at[slot]
            copies = []
            for dst, src in ((0, gate), (ib, up)):
                copies.append(
                    pltpu.make_async_copy(
                        w1_hbm.at[e, :, pl.ds(src, ib)],
                        w1_vm.at[slot, :, pl.ds(dst, ib)],
                        sem,
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        s1_hbm.at[e, :, pl.ds(src, ib)],
                        s1_vm.at[slot, :, pl.ds(dst, ib)],
                        sem,
                    )
                )
            copies.append(
                pltpu.make_async_copy(w2_hbm.at[e, pl.ds(gate, ib)], w2_vm.at[slot], sem)
            )
            copies.append(pltpu.make_async_copy(s2_hbm.at[e], s2_vm.at[slot], sem))
            return copies

        def start_unit(u):
            @pl.when(u < n_units)
            def _():
                for c in unit_copies(u):
                    c.start(priority=WEIGHT_DMA_PRIORITY)

        def wait_unit(u):
            for c in unit_copies(u):
                c.wait()

        shared_copies = [
            pltpu.make_async_copy(s, d, sh_sem)
            for s, d in zip(
                (w1sh_hbm.at[0], w2sh_hbm.at[0], s1sh_hbm.at[0], s2sh_hbm.at[0]),
                (w1sh_vm, w2sh_vm, s1sh_vm, s2sh_vm),
            )
        ]

        # ---- a chunk's rows: index window, row gather, per-tile scales ----
        def win_copy(c, slot):
            base = chunk_base_sm[c]
            aligned = pl.multiple_of(base - lax.rem(base, LANES), LANES)
            return pltpu.make_async_copy(
                tok_hbm.at[pl.ds(aligned, window)], win_sm.at[slot], win_sems.at[slot]
            )

        def start_window(c, slot):
            @pl.when(jnp.logical_and(c < n_chunk, jnp.logical_not(is_shared(c))))
            def _():
                win_copy(c, slot).start()

        def n_tiles(c):
            return _cdiv(chunk_rows_sm[c], tile_m)

        def gathered_rows(c):
            live = chunk_rows_sm[c]
            return jnp.where(is_shared(c), live, _cdiv(live, unroll) * unroll)

        def scale_copy(c, buf, t):
            row = (chunk_base_sm[c] + t * tile_m) // LANES
            return pltpu.make_async_copy(
                ls_hbm.at[pl.ds(row, scale_win)], ls_vm.at[buf, t], lhs_sems.at[buf]
            )

        def issue_chunk(c, buf):
            """Start chunk c's rows and scale windows into buffer buf (its
            index window, routed chunks only, is in window slot buf)."""
            base, live = chunk_base_sm[c], chunk_rows_sm[c]

            @pl.when(is_shared(c))
            def _():
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(base - n_routes, live)],
                    lhs_vm.at[buf, pl.ds(0, live)],
                    lhs_sems.at[buf],
                ).start()

            @pl.when(jnp.logical_not(is_shared(c)))
            def _():
                win_copy(c, buf).wait()
                off = lax.rem(base, LANES)

                def block(j, carry):
                    for u in range(unroll):
                        r = j * unroll + u
                        tok = win_sm[buf, off + r]
                        pltpu.make_async_copy(
                            x_hbm.at[tok], lhs_vm.at[buf, r], lhs_sems.at[buf]
                        ).start()
                    return carry

                lax.fori_loop(0, _cdiv(live, unroll), block, 0)

            def scales(t, carry):
                scale_copy(c, buf, t).start()
                return carry

            lax.fori_loop(0, n_tiles(c), scales, 0)

        def wait_chunk(c, buf):
            n = gathered_rows(c)
            pltpu.make_async_copy(
                lhs_vm.at[buf, pl.ds(0, n)], lhs_vm.at[buf, pl.ds(0, n)], lhs_sems.at[buf]
            ).wait()
            nt = n_tiles(c)
            pltpu.make_async_copy(
                ls_vm.at[buf, pl.ds(0, nt)], ls_vm.at[buf, pl.ds(0, nt)], lhs_sems.at[buf]
            ).wait()

        def out_copy(base, n):
            return pltpu.make_async_copy(
                out_vm.at[pl.ds(0, n)], part_hbm.at[pl.ds(base, n)], out_sem
            )

        def out_wait(n):
            pltpu.make_async_copy(
                out_vm.at[pl.ds(0, n)], out_vm.at[pl.ds(0, n)], out_sem
            ).wait()

        def tile_ffn(c, buf, t, w1, w2, s1, s2, mode):
            """Tile t of chunk c through one unit: gate/up, the activation and
            its per-row requantization, then the down projection in column
            chunks of _DOWN_CHUNK straight into the running sum (mode "first"
            stores, "mid" adds) or, finished with the down scale, into the
            bf16 output rows ("last" adds the running sum first, "only" has
            none)."""
            lo = pl.multiple_of(t * tile_m, tile_m)
            act = lhs_vm[buf, pl.ds(lo, tile_m)].reshape(tile_m, hidden)
            act_s = _tile_row_scales(ls_vm[buf, t], chunk_base_sm[c] + lo, tile_m)
            width = w1.shape[-1] // 2
            acc1 = lax.dot_general(act, w1[...], DN, preferred_element_type=jnp.float32)
            mid_q, mid_s = _intermediate_rows(
                acc1, act_s, width, w1s=s1, act_fn="silu", w1b=None
            )
            rows = pl.ds(lo, tile_m)
            nc = min(_DOWN_CHUNK, hidden)
            for n in range(0, hidden, nc):
                cols = slice(n, n + nc)
                part = lax.dot_general(
                    mid_q, w2[:, cols], DN, preferred_element_type=jnp.float32
                )
                part = part * mid_s
                if mode == "first":
                    acc_vm[rows, cols] = part
                elif mode == "mid":
                    acc_vm[rows, cols] += part
                else:
                    if mode == "last":
                        part = acc_vm[rows, cols] + part
                    done = (part * s2[:, cols]).reshape(tile_m, nc // LANES, LANES)
                    out_vm[rows, n // LANES : (n + nc) // LANES] = done.astype(
                        jnp.bfloat16
                    )

        # ---- prologue ----
        for c in shared_copies:
            c.start(priority=WEIGHT_DMA_PRIORITY)
        for k in range(nbuf - 1):
            start_unit(jnp.int32(k))
        z = jnp.int32(0)
        start_window(z, 0)
        issue_chunk(z, 0)  # every call has at least its shared chunk
        start_window(jnp.int32(1), 1)

        def chunk_body(c, n_prev):
            buf = lax.rem(c, 2)
            nt = n_tiles(c)
            wait_chunk(c, buf)

            @pl.when(c + 1 < n_chunk)
            def _():
                issue_chunk(c + 1, 1 - buf)

            start_window(c + 2, buf)

            @pl.when(jnp.logical_not(is_shared(c)))
            def _routed():
                def unit_wait(b):
                    u = c * n_blk + b
                    wait_unit(u)
                    start_unit(u + nbuf - 1)
                    return lax.rem(u, nbuf)

                def block_step(b, carry):
                    slot = unit_wait(b)

                    def tile_sum(t, carry):
                        args = (w1_vm.at[slot], w2_vm.at[slot], s1_vm[slot], None)

                        @pl.when(b == 0)
                        def _():
                            tile_ffn(c, buf, t, *args, "first")

                        @pl.when(b > 0)
                        def _():
                            tile_ffn(c, buf, t, *args, "mid")

                        return carry

                    lax.fori_loop(0, nt, tile_sum, 0)
                    return carry

                lax.fori_loop(0, n_blk - 1, block_step, 0)
                slot = unit_wait(n_blk - 1)

                @pl.when(n_prev > 0)
                def _():
                    out_wait(n_prev)

                def last_tile(t, carry):
                    tile_ffn(
                        c, buf, t, w1_vm.at[slot], w2_vm.at[slot], s1_vm[slot],
                        s2_vm[slot], "only" if n_blk == 1 else "last",
                    )
                    return carry

                lax.fori_loop(0, nt, last_tile, 0)

            @pl.when(is_shared(c))
            def _shared():
                @pl.when(c == n_routed)
                def _():
                    for cp in shared_copies:
                        cp.wait()

                @pl.when(n_prev > 0)
                def _():
                    out_wait(n_prev)

                def shared_tile(t, carry):
                    tile_ffn(
                        c, buf, t, w1sh_vm, w2sh_vm, s1sh_vm[...], s2sh_vm[...], "only"
                    )
                    return carry

                lax.fori_loop(0, nt, shared_tile, 0)

            live = chunk_rows_sm[c]
            out_copy(chunk_base_sm[c], live).start()
            return live

        n_last = lax.fori_loop(0, n_chunk, chunk_body, z)

        @pl.when(n_last > 0)
        def _():
            out_wait(n_last)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * 11,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, big_r, lane_blocks, LANES), FP8),
                pltpu.VMEM((2, tiles, scale_win, LANES), jnp.float32),
                pltpu.VMEM((big_r, hidden), jnp.float32),
                pltpu.VMEM((big_r, lane_blocks, LANES), jnp.bfloat16),
                pltpu.VMEM((nbuf, hidden, 2 * ib), FP8),
                pltpu.VMEM((nbuf, ib, hidden), FP8),
                pltpu.VMEM((nbuf, 1, 2 * ib), jnp.float32),
                pltpu.VMEM((nbuf, 1, hidden), jnp.float32),
                pltpu.SMEM((2, window), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA((nbuf,)),
                pltpu.VMEM((hidden, 2 * inter_sh), FP8),
                pltpu.VMEM((inter_sh, hidden), FP8),
                pltpu.VMEM((1, 2 * inter_sh), jnp.float32),
                pltpu.VMEM((1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SMEM((n_chunks,), jnp.int32),
                pltpu.SMEM((n_chunks,), jnp.int32),
                pltpu.SMEM((n_chunks,), jnp.int32),
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
            f"_ib{ib}_r{big_r}_m{tile_m}_u{unroll}"
        ),
    )
    return call, dict(
        slab_rows=slab_rows, scale_rows=scale_rows, tok_rows=tok_rows, n_chunks=n_chunks
    )


@functools.cache
def build_blocked_expert_kernel(**kw):
    return _build_blocked_expert_kernel(**kw)


# VMEM held back from the planned buffers for Mosaic's internal scratch (the
# live f32 matmul results of a tile).
_BLOCKED_RESERVE = 6 << 20
# Output columns per down-projection matmul in blocked mode: bounds the live
# f32 result, which goes straight into the running sum.
_DOWN_CHUNK = 512
_MAX_CHUNK_ROWS = 512


def _vmem_budget():
    try:
        return v2_host.vmem_limit()
    except Exception:  # no TPU (interpret mode): the v7x figure
        return int(64 * 2**20 * v2_host.VMEM_FRACTION)


def _whole_expert_bytes(hidden, inter, inter_sh, nbuf, tile_m, lookahead):
    """VMEM of the whole-expert kernel: the weight ring, the activation and
    output tiles, the shared expert and the tile's live f32 result."""
    ring = nbuf * (3 * hidden * inter + 4 * (2 * inter + hidden))
    tiles = (lookahead + 1) * tile_m * hidden + 2 * tile_m * hidden * 2
    return ring + tiles + 3 * hidden * inter_sh + tile_m * hidden * 4


def _blocked_bytes(hidden, ib, inter_sh, nbuf, tile_m, chunk_rows):
    ring = nbuf * (3 * hidden * ib + 4 * (2 * ib + hidden))
    rows = chunk_rows * hidden * (2 * 1 + 4 + 2)  # 2 fp8 row buffers, acc, out
    scales = 2 * (chunk_rows // tile_m) * (tile_m // LANES + 1) * LANES * 4
    return ring + rows + scales + 3 * hidden * inter_sh


def plan_blocked(
    hidden, inter, inter_sh, nbuf, tile_m, lookahead, i_block, chunk_rows, block_nbuf
):
    """(ib, ring depth, chunk_rows) of the blocked kernel, or None for the
    whole-expert kernel, chosen at trace time from the shapes.

    Whole experts whenever their ring of nbuf slots fits VMEM beside the
    token-side buffers (every shape whose whole-expert build could compile);
    i_block forces blocked mode. Blocked defaults, by tile height (so by
    token count, default_tile_m), from the GLM-5.2 EP16 sweep: 128-row tiles
    (T < 8192) stream 512-wide blocks through 3 slots, 256-row tiles 1024-wide
    blocks through 2 (half the running-sum passes, which dominate once
    experts hold hundreds of rows); each takes the tallest chunk (a multiple
    of tile_m, at most _MAX_CHUNK_ROWS) that fits, falling back to narrower
    blocks if none does. i_block / block_nbuf / chunk_rows override.
    """
    limit = _vmem_budget()
    budget = limit - _BLOCKED_RESERVE
    if i_block is None:
        if _whole_expert_bytes(hidden, inter, inter_sh, nbuf, tile_m, lookahead) <= limit:
            return None
        prefs = [(1024, 2)] if tile_m >= 256 else []
        prefs += [(512, 3), (256, 3), (128, 3)]
        cands = [(w, n) for w, n in prefs if w < inter and inter % w == 0]
    else:
        if inter % i_block or i_block % LANES:
            raise ValueError(f"i_block {i_block} must divide inter={inter} in lanes")
        cands = [(i_block, 3)]
    for k, (w, nb) in enumerate(cands):
        nb = max(2, block_nbuf or nb)
        if chunk_rows is not None:
            return w, nb, chunk_rows
        rows = (_MAX_CHUNK_ROWS // tile_m) * tile_m
        while rows > tile_m and _blocked_bytes(hidden, w, inter_sh, nb, tile_m, rows) > (
            budget
        ):
            rows -= tile_m
        fits = _blocked_bytes(hidden, w, inter_sh, nb, tile_m, rows) <= budget
        if fits or k == len(cands) - 1:
            return w, nb, rows
    raise AssertionError("unreachable")


# ---- Sparse combine (blocked mode) ----------------------------------------
#
# The whole-expert path's combine (v2's) reads topk + 1 slab rows per token
# whatever their weight. Under expert parallelism a token has topk / EP local
# routes on average (GLM-5.2 at EP16: 0.5 of 8), so most of those reads fetch
# the shared row again for a zero weight. This combine reads only the live
# slots: each token's slot list holds its shared row first, then its local
# routed rows, and a per-token count says how many are live. Same step
# structure as v2's (steps of tb tokens, rows staged a step ahead, the sum in
# the rows' layout, the bf16 tile relaid out in VMEM); the step's row waits
# count the bytes its live slots started.

_SC_TOKENS = 64
_SC_GROUP = 16
_SC_PARITIES = 4  # row buffers
_SC_LEAD = 2  # steps of rows in flight ahead of the reduction


def sparse_combine_tokens(t_local):
    """Tokens per step of the sparse combine, or 0 to use the XLA form."""
    tb = _SC_TOKENS
    steps = t_local // tb
    if t_local % tb or steps < _SC_PARITIES or steps % _SC_PARITIES:
        return 0
    return tb


def _build_sparse_combine_kernel(*, t_local, slots, hidden, interpret=False):
    tb = sparse_combine_tokens(t_local)
    steps = t_local // tb
    lanes = LANES
    lane_blocks = hidden // lanes
    groups = tb // _SC_GROUP

    def kernel(
        cnt_sm, step_rows_sm, pos_sm, coef_sm, part_hbm, out_hbm,
        rows_vm, tile_vm, out_vm, row_sems, out_sems,
    ):
        # Tables are slot-major: slot j of token t at j * t_local + t; a
        # token's live slots come first, cnt[t] of them (the shared row is
        # slot 0, always live). Only live slots are touched: a scalar loop
        # per token, so the cost follows the live rows, not topk.
        def issue_step(step, parity):
            """Start the live rows of step `step` into row buffer `parity`."""
            base = step * tb

            def start(i, t, j, priority):
                pltpu.make_async_copy(
                    part_hbm.at[pos_sm[j * t_local + t]],
                    rows_vm.at[parity, j * tb + i],
                    row_sems.at[parity],
                ).start(priority=priority)

            def token(i, carry):
                t = base + i
                # Shared rows and routed rows on separate DMA queues.
                start(i, t, 0, 0)

                def slot(j, carry):
                    start(i, t, j, 1)
                    return carry

                return lax.fori_loop(1, cnt_sm[t], slot, carry)

            lax.fori_loop(0, tb, token, jnp.int32(0))

        def one_step(step, parity):
            base = step * tb
            out_par = parity % 2
            # Semaphores count bytes: wait for exactly the rows started.
            n = step_rows_sm[step]
            pltpu.make_async_copy(
                rows_vm.at[parity, pl.ds(0, n)],
                rows_vm.at[parity, pl.ds(0, n)],
                row_sems.at[parity],
            ).wait()

            @pl.when(step + _SC_LEAD < steps)
            def _():
                issue_step(step + _SC_LEAD, (parity + _SC_LEAD) % _SC_PARITIES)

            @pl.when(step >= 2)
            def _():
                pltpu.make_async_copy(
                    out_vm.at[out_par], out_vm.at[out_par], out_sems.at[out_par]
                ).wait()

            def token(i, carry):
                t = base + i

                def slot(j, acc):
                    row = rows_vm[parity, j * tb + i].astype(jnp.float32)
                    return acc + row * coef_sm[j * t_local + t]

                acc = rows_vm[parity, i].astype(jnp.float32) * coef_sm[t]
                acc = lax.fori_loop(1, cnt_sm[t], slot, acc)
                tile_vm[i] = acc.astype(jnp.bfloat16)
                return carry

            lax.fori_loop(0, tb, token, jnp.int32(0))
            for g in range(groups):
                out_vm[out_par, pl.ds(g * _SC_GROUP, _SC_GROUP)] = tile_vm[
                    pl.ds(g * _SC_GROUP, _SC_GROUP)
                ].reshape(_SC_GROUP, hidden)
            pltpu.make_async_copy(
                out_vm.at[out_par], out_hbm.at[pl.ds(base, tb)], out_sems.at[out_par]
            ).start()

        def step_group(g, carry):
            for parity in range(_SC_PARITIES):
                one_step(g * _SC_PARITIES + parity, parity)
            return carry

        for p in range(_SC_LEAD):
            issue_step(jnp.int32(p), p)
        lax.fori_loop(0, steps // _SC_PARITIES, step_group, jnp.int32(0))
        for p in range(2):
            pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (_SC_PARITIES, slots * tb, lane_blocks, lanes), jnp.bfloat16
                ),
                pltpu.VMEM((tb, lane_blocks, lanes), jnp.bfloat16),
                pltpu.VMEM((2, tb, hidden), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((_SC_PARITIES,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=None if interpret else _vmem_budget(),
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"moe_tp_sparse_combine_t{t_local}_s{slots}_b{tb}",
    )


@functools.cache
def build_sparse_combine_kernel(**kw):
    return _build_sparse_combine_kernel(**kw)


def sparse_combine(part, pos, coef, live, n_tok, hidden, n_chunks, reduce_fn, interpret):
    """combine() over only the live slots: pos/coef/live [S, T] SLOT-MAJOR
    (a [T, S] table with S = topk + 1 pads its minor dimension to a lane
    block; [S, T] does not), slot 0 the shared row (always live), live
    marking the routed slots that carry a local route. One sort along the
    slot axis moves each token's live slots to the front (stable), so the
    kernel reads cnt[t] rows for token t."""
    _, pos, coef = lax.sort(
        (jnp.logical_not(live).astype(jnp.int32), pos.astype(jnp.int32),
         coef.astype(jnp.float32)),
        dimension=0,
        num_keys=1,
        is_stable=True,
    )
    cnt = jnp.sum(live, axis=0).astype(jnp.int32)
    slots = pos.shape[0]
    if n_tok % n_chunks:
        n_chunks = 1
    tc = n_tok // n_chunks
    outs = []
    for c in range(n_chunks):
        sl = slice(c * tc, (c + 1) * tc)
        tb = sparse_combine_tokens(tc)
        if tb:
            step_rows = jnp.sum(cnt[sl].reshape(-1, tb), axis=1).astype(jnp.int32)
            fn = build_sparse_combine_kernel(
                t_local=tc, slots=slots, hidden=hidden, interpret=interpret
            )
            y = fn(
                cnt[sl], step_rows, pos[:, sl].reshape(-1), coef[:, sl].reshape(-1),
                part.reshape(part.shape[0], hidden // LANES, LANES),
            )
        else:
            m = jnp.arange(slots)[:, None] < cnt[sl][None, :]
            rows = part.reshape(part.shape[0], hidden)[pos[:, sl]].astype(jnp.float32)
            w = jnp.where(m, coef[:, sl], 0.0)
            y = jnp.sum(jnp.where(m[..., None], rows, 0.0) * w[..., None], axis=0)
            y = y.astype(jnp.bfloat16)
        if reduce_fn is not None:
            if outs:
                y, _ = lax.optimization_barrier((y, outs[-1]))
            y = reduce_fn(y)
        outs.append(y)
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


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
    i_block=None,
    chunk_rows=None,
    block_nbuf=None,
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

    Experts whose ring of nbuf whole-expert slots does not fit VMEM (GLM-5.2
    at EP: H 6144, I 2048) run the blocked kernel instead: chunks of at most
    chunk_rows rows per visit, each expert streamed in blocks of i_block
    intermediate columns through a ring of block_nbuf slots (plan_blocked),
    and the sparse combine. i_block forces blocked mode; shapes whose
    whole-expert ring fits run the unblocked program unchanged.
    """
    n_tok, hidden = x.shape
    if tile_m is None:
        tile_m = default_tile_m(n_tok)
    n_experts, _, two_i = w1.shape
    inter = two_i // 2
    inter_sh = w2_sh.shape[0]
    n_routes = n_tok * topk_ids.shape[1]
    blocked = plan_blocked(
        hidden, inter, inter_sh, nbuf, tile_m, lookahead, i_block, chunk_rows, block_nbuf
    )
    if blocked is not None:
        ib, nb_blk, rows_blk = blocked
        call, dims = build_blocked_expert_kernel(
            n_experts=n_experts,
            hidden=hidden,
            inter=inter,
            inter_sh=inter_sh,
            n_tok=n_tok,
            n_routes=n_routes,
            ib=ib,
            nbuf=nb_blk,
            chunk_rows=rows_blk,
            tile_m=tile_m,
            unroll=gather_unroll,
            interpret=interpret,
        )
    else:
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
    if blocked is None:
        # The shared expert is one more visit over every token.
        visit_e = visit_e.at[n_rv].set(n_experts)
        visit_base = visit_base.at[n_rv].set(n_routes)
        visit_rows = visit_rows.at[n_rv].set(n_tok)
    ls = jnp.concatenate([s_sorted, x_s])
    ls = jnp.pad(ls, (0, dims["scale_rows"] * LANES - ls.shape[0])).reshape(-1, LANES)
    tok_pad = jnp.pad(tok, (0, dims["tok_rows"] - n_routes))
    # Blocked: the kernel cuts the routed visits (and the shared rows) into
    # chunks itself, so it takes the routed visit count.
    n_visit = n_rv if blocked is not None else n_rv + 1
    tables = (visit_e, visit_base, visit_rows, n_visit.reshape(1))

    part = call(
        *tables,
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
    if blocked is not None:
        live = jnp.concatenate(
            [jnp.ones((1, n_tok), jnp.bool_), jnp.logical_not(dead).T], 0
        )
        return sparse_combine(
            part,
            jnp.concatenate([shared_pos.T, pos.T], 0),
            jnp.concatenate([gate.T, coef.T], 0),
            live,
            n_tok,
            hidden,
            combine_chunks,
            reduce_fn,
            interpret,
        )
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
