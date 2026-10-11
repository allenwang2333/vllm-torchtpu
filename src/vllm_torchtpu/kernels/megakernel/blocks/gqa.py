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
"""Grouped-Query Attention (`GQABlock`) with sequence-split KV ring and MTP verify support."""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel.ops import (
    BF16,
    F32,
    FP8,
    I32,
    U32,
    bswitch,
    rms_zc,
    rnd,
    rope_partial,
)
from vllm_torchtpu.kernels.megakernel.spec import Block, SmallField, WeightDecl

NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))


class GQABlock(Block):
    """Full GQA block with QK RMSNorm, partial RoPE, group Q/softmax exchange, and output gate."""

    name: str = "gqa"

    def weights(self, c: Any) -> list[WeightDecl]:
        return [
            WeightDecl("att_in", c.H, c.NAI, FP8, layers="attn", stream_ring="static"),
            WeightDecl("att_out", c.nh * c.HD, c.H, FP8, layers="attn", stream_ring="static"),
        ]

    def smalls(self, c: Any) -> list[SmallField]:
        return [
            SmallField("in_s", max(c.NLI, c.NAI)),
            SmallField("out_s", c.H),
            SmallField("qn", c.HD),
            SmallField("kn", c.HD),
        ]

    def prologue(self, ctx: Any) -> None:
        """Build this rank's KV worklist (`wl_b`, `wl_t`, `st[1]`)."""
        c, dp, scr = ctx.c, ctx.dp, ctx.scr
        pos_ref, gi = ctx.pos_ref, ctx.gi
        NS, T, BLK, TS, SPLIT = dp.NS, dp.T, dp.BLK, dp.TS, dp.SPLIT
        eff_ctx = max(c.ctx, c.max_pages_per_seq * c.page_size) if c.paged_kv else c.ctx
        max_nb = -(-eff_ctx // BLK)

        n = jnp.int32(0)
        for b in range(NS):
            nb = jnp.minimum(jnp.maximum((pos_ref[b * T] + BLK - 1) // BLK, 0), max_nb)
            t0 = lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0)
            cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

            def put(j, carry, b=b, t0=t0, n=n):
                scr["wl_b"][n + j] = jnp.int32(b)
                scr["wl_t"][n + j] = t0 + j * TS
                return carry

            lax.fori_loop(0, cnt, put, 0)
            n = n + cnt
        scr["st"][1] = n

    def init_buffers(self, ctx: Any) -> None:
        """Initialize attention-specific VMEM/SMEM scratch buffers after barrier."""
        c, dp, scr = ctx.c, ctx.dp, ctx.scr
        if c.paged_kv:
            scr["knew_o"][...] = jnp.zeros((dp.MR, dp.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((dp.MR, dp.HD), FP8)
        if dp.FUSED:
            for s in range(dp.NS):
                scr["kv_rd_pend"][s] = jnp.int32(0)
                scr["kv_wr_pend"][s] = jnp.int32(0)
        if dp.SPLIT:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if dp.PACK:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)

    def emit(self, ctx: Any, l: Any, slot: Any, g0: Any) -> Any:
        c, dp, tiles, scr, ST, O = ctx.c, ctx.dp, ctx.tiles, ctx.scr, ctx.ST, ctx.O
        pos_ref, raw_refs, gi, skip, sc = ctx.pos_ref, ctx.raw_refs, ctx.gi, ctx.skip, ctx.sc
        stream, comm = ctx.stream, ctx.comm
        AE, NKV, BLK, SUB = dp.AE, dp.NKV, dp.BLK, dp.SUB
        B, T, NS, VER, G, MR = dp.B, dp.T, dp.NS, dp.VER, dp.G, dp.MR
        nh, HD, H, QR, TQ, PW = dp.nh, dp.HD, dp.H, dp.QR, dp.TQ, dp.PW
        SPLIT, PACK, PB, NBP, FUSED, LAZY_ATT = (
            dp.SPLIT,
            dp.PACK,
            dp.PB,
            dp.NBP,
            dp.FUSED,
            dp.LAZY_ATT,
        )
        nt, typed, interpret = dp.nt, dp.typed, ctx.interpret

        a = l // AE
        ksem = scr["kv_sem"]
        am, al, aacc = scr["am"], scr["al"], scr["aacc"]
        n_items = scr["st"][1]

        if c.paged_kv:
            pt_ref = ST["page_table"]
            kvbuf = scr["kvbuf"]
            P_BLK = BLK // c.page_size

            def kv_start(i):
                @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                def _():
                    s = i % NKV
                    b, t = scr["wl_b"][i], scr["wl_t"][i]

                    def start_for_ref(r):
                        def issue_p(p, _):
                            kp = jnp.clip(pt_ref[b, t * P_BLK + p], 0, c.num_pages - 1)
                            dst = kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)]
                            pltpu.make_async_copy(r.at[kp], dst, ksem.at[s]).start()
                            return 0

                        lax.fori_loop(0, P_BLK, issue_p, 0)

                    bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

            def kv_wait(i):
                if "kv" not in skip:
                    s = i % NKV

                    def wait_p(p, _):
                        pltpu.make_async_copy(
                            raw_refs[0].at[0],
                            kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)],
                            ksem.at[s],
                        ).wait()
                        return 0

                    lax.fori_loop(0, P_BLK, wait_p, 0)

            lax.fori_loop(0, NKV, lambda i, _: (kv_start(i), 0)[1], 0)
        else:
            kc, vc = ST["kc"], ST["vc"]
            kbuf, vbuf = scr["kbuf"], scr["vbuf"]

            def kv_item(i, b, t):
                s = i % NKV
                t0 = pl.multiple_of(t * BLK, BLK)
                return (
                    pltpu.make_async_copy(
                        kc.at[a, b, pl.ds(t0, BLK)], kbuf.at[s], ksem.at[0, s]
                    ),
                    pltpu.make_async_copy(
                        vc.at[a, b, pl.ds(t0, BLK)], vbuf.at[s], ksem.at[1, s]
                    ),
                )

            def kv_start(i):
                @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                def _():
                    for d in kv_item(i, scr["wl_b"][i], scr["wl_t"][i]):
                        d.start()

            def kv_wait(i):
                if "kv" not in skip:
                    zero = jnp.int32(0)
                    for d in kv_item(i, zero, zero):
                        d.wait()

            for i in range(NKV):
                kv_start(jnp.int32(i))

        with sc("att_in"):
            y = stream.dmatmul("att_in", g0, scr["xb"], c.NAI, lazy=LAZY_ATT)
            ybuf = scr["ybuf"]
            ybuf[:, : c.NAI] = rnd(y * ctx.sm(slot, "in_s", c.NAI))
        qn = ctx.sm(slot, "qn")
        kn = ctx.sm(slot, "kn")
        half = c.ROT // 2
        inv = 1.0 / (
            c.theta ** (lax.broadcasted_iota(I32, (1, half), 1).astype(F32) * 2.0 / c.ROT)
        )

        koff = nh * 2 * HD
        with sc("att_qk"):
            for b in range(B):
                pb = pos_ref[b]
                ang = pb.astype(F32) * inv
                cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                q = rms_zc(qg[:, :HD], qn, c.eps)
                k = rms_zc(ybuf[b : b + 1, koff : koff + HD], kn, c.eps)
                v = ybuf[b : b + 1, koff + HD : koff + 2 * HD]
                q = rnd(rope_partial(q, cos, sin, c.ROT) * HD**-0.5)
                k = rope_partial(k, cos, sin, c.ROT)
                if PACK:
                    t_, o_ = dp.qrow(b)
                    scr["qg"][gi, t_, o_ : o_ + nh] = q
                else:
                    if nh < 8:
                        q = jnp.concatenate((q, jnp.zeros((8 - nh, HD), F32)), axis=0)
                    if SPLIT or VER:
                        scr["qg"][gi, b] = q
                    else:
                        scr["qb"][b] = q.astype(BF16)
                if c.paged_kv:
                    k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                    v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                    scr["knew_o"][b : b + 1, :] = k_fp8
                    scr["vnew_o"][b : b + 1, :] = v_fp8
                    scr["knew"][b : b + 1, :] = k_fp8.astype(F32)
                    scr["vnew"][b : b + 1, :] = v_fp8.astype(F32)
                else:
                    scr["knew"][b : b + 1, :] = k
                    scr["vnew"][b : b + 1, :] = v
            if not FUSED:
                k_src = scr["knew_o"] if c.paged_kv else scr["knew"]
                v_src = scr["vnew_o"] if c.paged_kv else scr["vnew"]
                kv_out = (
                    pltpu.make_async_copy(k_src, O["knew"].at[a], scr["kv_out_sem"].at[0]),
                    pltpu.make_async_copy(v_src, O["vnew"].at[a], scr["kv_out_sem"].at[1]),
                )
                for d in kv_out:
                    d.start()

        if SPLIT or VER:
            with sc("att_qgather"):
                qgb = scr["qg"]
                if SPLIT:
                    comm.exchange(
                        comm.group_copies(
                            lambda off: qgb.at[gi], qgb.at[gi], scr["sq_ssem"], scr["sq_rsem"]
                        )
                    )

                def group_q(b):
                    if not SPLIT:
                        return qgb[0, b]
                    t_, o_ = dp.qrow(b)
                    qa = jnp.concatenate(
                        [qgb[g, t_, o_ : o_ + nh, :] for g in range(G)], axis=0
                    )
                    if G * nh < QR:
                        qa = jnp.concatenate((qa, jnp.zeros((QR - G * nh, HD), F32)), 0)
                    return qa

                for sq in range(NS):
                    qs = [group_q(sq * T + t) for t in range(T)]
                    qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
                    scr["qb"][sq] = qs.astype(BF16)

        with sc("att_core"):
            AMASK = tiles.amask
            col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
            col_v = (
                lax.broadcasted_iota(I32, (HD, SUB), 1)
                if (c.paged_kv and AMASK == 0)
                else None
            )
            am[...] = jnp.full(am.shape, -1e30, F32)
            al[...] = jnp.zeros(al.shape, F32)
            aacc[...] = jnp.zeros(aacc.shape, F32)

            def item(i, carry):
                s = i % NKV
                kv_wait(i)
                if "attn" not in skip:
                    b, t = scr["wl_b"][i], scr["wl_t"][i]
                    plen = pos_ref[b * T] if VER else pos_ref[b]
                    qrows = scr["qb"][b]
                    if c.paged_kv:
                        if interpret:
                            k_full = kvbuf[s, 0].reshape(HD, BLK)
                            v_full = kvbuf[s, 1].reshape(HD, BLK)
                        else:
                            k_u32 = kvbuf.at[s, 0].bitcast(U32).reshape(64, BLK)[...]
                            v_u32 = kvbuf.at[s, 1].bitcast(U32).reshape(64, BLK)[...]
                            k_full = pltpu.bitcast(k_u32, FP8)
                            if AMASK in (0, 3, 4):
                                v_full = pltpu.bitcast(v_u32, FP8)
                    for j in range(BLK // SUB):
                        off = t * BLK + j * SUB
                        rem = plen - off
                        full_blk = off + SUB <= plen
                        if c.paged_kv:
                            kj = k_full if SUB == BLK else k_full[:, j * SUB : (j + 1) * SUB]
                            if AMASK == 0:
                                vj = v_full if SUB == BLK else v_full[:, j * SUB : (j + 1) * SUB]
                                vj = jnp.where(col_v + off < plen, vj, jnp.zeros_like(vj))
                            elif interpret:
                                vj = v_full if SUB == BLK else v_full[:, j * SUB : (j + 1) * SUB]
                                if AMASK not in (3, 4):
                                    vj = jnp.where(col < rem, vj, jnp.zeros_like(vj))
                            elif AMASK == 1:
                                vu = v_u32 if SUB == BLK else v_u32[:, j * SUB : (j + 1) * SUB]
                                vu = jnp.where(col < rem, vu, jnp.uint32(0))
                                vj = pltpu.bitcast(vu, FP8)
                            elif AMASK == 2:
                                vu = v_u32 if SUB == BLK else v_u32[:, j * SUB : (j + 1) * SUB]
                                vu = lax.cond(
                                    full_blk,
                                    lambda: vu,
                                    lambda: jnp.where(col < rem, vu, jnp.uint32(0)),
                                )
                                vj = pltpu.bitcast(vu, FP8)
                            else:
                                vj = v_full if SUB == BLK else v_full[:, j * SUB : (j + 1) * SUB]
                            scores = stream.mm(qrows, kj, NN)
                        else:
                            kj = kbuf[s] if SUB == BLK else kbuf[s, j * SUB : (j + 1) * SUB]
                            vj = vbuf[s] if SUB == BLK else vbuf[s, j * SUB : (j + 1) * SUB]
                            scores = stream.mm(qrows, kj, NT)
                        if AMASK == 0:
                            scores = jnp.where(col + off < plen, scores, -jnp.inf)
                        elif AMASK in (1, 4):
                            scores = jnp.where(col < rem, scores, -jnp.inf)
                        else:
                            scores = lax.cond(
                                full_blk,
                                lambda: scores,
                                lambda: jnp.where(col < rem, scores, -jnp.inf),
                            )
                        m_old = am[b]
                        m_new = jnp.maximum(m_old, jnp.max(scores, axis=-1, keepdims=True))
                        alpha = jnp.exp(m_old - m_new)
                        pr = jnp.exp(scores - m_new[:, 0:1])
                        pv = stream.mm(pr.astype(BF16), vj, NT if c.paged_kv else NN)
                        al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
                        aacc[b] = aacc[b] * alpha[:, 0:1] + pv
                        am[b] = m_new
                kv_start(i + NKV)
                return carry

            lax.fori_loop(0, n_items, item, 0)

        def seq_same_page(s):
            if T == 1:
                return jnp.bool_(True)
            b0 = s * T
            lp0 = lax.div(pos_ref[b0], c.page_size)
            ok = jnp.bool_(True)
            for t in range(1, T):
                ok = jnp.logical_and(ok, lax.div(pos_ref[b0 + t], c.page_size) == lp0)
            return ok

        if FUSED:
            kv_scat = scr["kv_scat_page"]
            rd_sem, wr_sem = scr["kv_rd_sem"], scr["kv_wr_sem"]
            rd_pend, wr_pend = scr["kv_rd_pend"], scr["kv_wr_pend"]
            for s in range(NS):
                ctx.wait_kv_wr(s)
                act_s = ctx.is_seq_active(s)
                same_s = seq_same_page(s)
                do_rd = jnp.logical_and(act_s, same_s)
                rd_pend[s] = jnp.where(do_rd, 1, 0).astype(I32)

                @pl.when(do_rd)
                def _(s=s):
                    lp0 = lax.div(pos_ref[s * T], c.page_size)
                    kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                    def issue_rd(r):
                        pltpu.make_async_copy(
                            r.at[kp0], kv_scat.at[s], rd_sem.at[s]
                        ).start()

                    bswitch(a, [lambda r=r: issue_rd(r) for r in raw_refs])

        if SPLIT:
            with sc("att_merge"):
                pso, pri = scr["pso"], scr["pri"]
                lane_pw = lax.broadcasted_iota(I32, (nh, 128), 1)
                for b in range(B):
                    t_, o_ = dp.qrow(b)
                    sq, q0 = b // T, (b % T) * QR
                    mb, lb = am[sq, q0 : q0 + QR], al[sq, q0 : q0 + QR]
                    ab = aacc[sq, q0 : q0 + QR]
                    for g in range(G):
                        r0 = g * nh
                        pso[g, t_, o_ : o_ + nh, 0:HD] = ab[r0 : r0 + nh]
                        pso[g, t_, o_ : o_ + nh, HD:PW] = jnp.where(
                            lane_pw < 64, mb[r0 : r0 + nh], lb[r0 : r0 + nh]
                        )
                pri[gi] = pso[gi]
                comm.exchange(
                    comm.group_copies(
                        lambda off: pso.at[gi ^ off], pri.at[gi], scr["sp_ssem"], scr["sp_rsem"]
                    )
                )
                for t in range(NBP):
                    parts = [pri[g, t] for g in range(G)]
                    mg = [x[:, HD : HD + 1] for x in parts]
                    m = mg[0]
                    for x in mg[1:]:
                        m = jnp.maximum(m, x)
                    wg = [jnp.exp(x - m) for x in mg]
                    lsum = sum(w * x[:, HD + 64 : HD + 65] for w, x in zip(wg, parts))
                    asum = sum(w * x[:, :HD] for w, x in zip(wg, parts))
                    if not PACK:
                        sq, q0 = t // T, (t % T) * QR
                        am[sq, q0 : q0 + 8] = jnp.broadcast_to(m, (8, 128))
                        al[sq, q0 : q0 + 8] = jnp.broadcast_to(lsum, (8, 128))
                        aacc[sq, q0 : q0 + 8] = asum
                        continue
                    for b in range(t * PB, min(B, (t + 1) * PB)):
                        o_ = dp.qrow(b)[1]
                        sq, q0 = b // T, (b % T) * QR
                        am[sq, q0 : q0 + nh] = jnp.broadcast_to(m[o_ : o_ + nh], (nh, 128))
                        al[sq, q0 : q0 + nh] = jnp.broadcast_to(lsum[o_ : o_ + nh], (nh, 128))
                        aacc[sq, q0 : q0 + nh] = asum[o_ : o_ + nh]

        with sc("att_new"):
            NR = nh if PACK else 8
            for b in range(B):
                sq, tb = divmod(b, T)
                q0 = tb * QR
                m_old = am[sq, q0 : q0 + NR][:, 0:1]
                l_old = al[sq, q0 : q0 + NR][:, 0:1]
                a_old = aacc[sq, q0 : q0 + NR]
                if PACK:
                    t_, o_ = dp.qrow(b)
                    qf = scr["qg"][gi, t_, o_ : o_ + nh]
                else:
                    qf = scr["qg"][gi, b] if SPLIT or VER else scr["qb"][b].astype(F32)
                if VER:
                    for j in range(tb + 1):
                        bj = sq * T + j
                        s_new = jnp.sum(
                            qf * scr["knew"][bj : bj + 1, :], axis=-1, keepdims=True
                        )
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_old = l_old * alpha + p_new
                        a_old = a_old * alpha + p_new * scr["vnew"][bj : bj + 1, :]
                        m_old = m_new
                    outb = a_old / l_old
                else:
                    s_new = jnp.sum(qf * scr["knew"][b : b + 1, :], axis=-1, keepdims=True)
                    m_new = jnp.maximum(m_old, s_new)
                    alpha = jnp.exp(m_old - m_new)
                    p_new = jnp.exp(s_new - m_new)
                    l_tot = l_old * alpha + p_new
                    outb = (a_old * alpha + p_new * scr["vnew"][b : b + 1, :]) / l_tot
                for h in range(nh):
                    oh = rnd(outb[h : h + 1, :])
                    gh = rnd(jax.nn.sigmoid(scr["gate"][b : b + 1, h * HD : (h + 1) * HD]))
                    scr["mix"][b : b + 1, h * HD : (h + 1) * HD] = rnd(oh * gh)

        with sc("att_out"):
            if FUSED:
                lane_iota_kv = lax.broadcasted_iota(I32, (64, c.page_size), 1)
                k_cols = ctx.batch_to_i32_cols(scr["knew_o"][...])
                v_cols = ctx.batch_to_i32_cols(scr["vnew_o"][...])
                for s in range(NS):
                    act_s = ctx.is_seq_active(s)
                    same_s = seq_same_page(s)
                    do_fast = rd_pend[s] != 0
                    rd_pend[s] = jnp.int32(0)

                    @pl.when(do_fast)
                    def _(s=s):
                        pltpu.make_async_copy(
                            raw_refs[0].at[0], kv_scat.at[s], rd_sem.at[s]
                        ).wait()
                        b0 = s * T
                        if typed:
                            k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(64, c.page_size)
                            v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(64, c.page_size)
                        else:
                            k_view = kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                            v_view = kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                            k_cur = k_view[...]
                            v_cur = v_view[...]
                        for t in range(T):
                            mask_b = lane_iota_kv == (b0 + t)
                            mask_ln = lane_iota_kv == lax.rem(pos_ref[b0 + t], c.page_size)
                            k_col = jnp.sum(
                                jnp.where(mask_b, k_cols, jnp.int32(0)), axis=1, keepdims=True
                            )
                            v_col = jnp.sum(
                                jnp.where(mask_b, v_cols, jnp.int32(0)), axis=1, keepdims=True
                            )
                            k_cur = jnp.where(mask_ln, k_col, k_cur)
                            v_cur = jnp.where(mask_ln, v_col, v_cur)
                        if typed:
                            kv_scat[s, 0] = pltpu.bitcast(
                                k_cur.reshape(64, 1, c.page_size), FP8
                            )
                            kv_scat[s, 1] = pltpu.bitcast(
                                v_cur.reshape(64, 1, c.page_size), FP8
                            )
                        else:
                            k_view[...] = k_cur
                            v_view[...] = v_cur
                        lp0 = lax.div(pos_ref[b0], c.page_size)
                        kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                        def issue_wr(r):
                            pltpu.make_async_copy(
                                kv_scat.at[s], r.at[kp0], wr_sem.at[s]
                            ).start()

                        bswitch(a, [lambda r=r: issue_wr(r) for r in raw_refs])

                    wr_pend[s] = jnp.where(do_fast, 1, 0).astype(I32)
                    if T > 1:
                        @pl.when(jnp.logical_and(act_s, jnp.logical_not(same_s)))
                        def _(s=s):
                            b0 = s * T
                            for t in range(T):
                                p_t = pos_ref[b0 + t]
                                lp_t = lax.div(p_t, c.page_size)
                                ln_t = lax.rem(p_t, c.page_size)
                                kp_t = jnp.clip(pt_ref[s, lp_t], 0, c.num_pages - 1)

                                def rd_one(r, kp_t=kp_t):
                                    pltpu.make_async_copy(
                                        r.at[kp_t], kv_scat.at[s], rd_sem.at[s]
                                    ).start()

                                bswitch(a, [lambda r=r: rd_one(r) for r in raw_refs])
                                pltpu.make_async_copy(
                                    raw_refs[0].at[0], kv_scat.at[s], rd_sem.at[s]
                                ).wait()
                                mask_b = lane_iota_kv == (b0 + t)
                                mask_ln = lane_iota_kv == ln_t
                                k_col = jnp.sum(
                                    jnp.where(mask_b, k_cols, jnp.int32(0)),
                                    axis=1,
                                    keepdims=True,
                                )
                                v_col = jnp.sum(
                                    jnp.where(mask_b, v_cols, jnp.int32(0)),
                                    axis=1,
                                    keepdims=True,
                                )
                                if typed:
                                    k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(
                                        64, c.page_size
                                    )
                                    v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(
                                        64, c.page_size
                                    )
                                    k_cur = jnp.where(mask_ln, k_col, k_cur)
                                    v_cur = jnp.where(mask_ln, v_col, v_cur)
                                    kv_scat[s, 0] = pltpu.bitcast(
                                        k_cur.reshape(64, 1, c.page_size), FP8
                                    )
                                    kv_scat[s, 1] = pltpu.bitcast(
                                        v_cur.reshape(64, 1, c.page_size), FP8
                                    )
                                else:
                                    k_view = (
                                        kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                                    )
                                    v_view = (
                                        kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                                    )
                                    k_view[...] = jnp.where(mask_ln, k_col, k_view[...])
                                    v_view[...] = jnp.where(mask_ln, v_col, v_view[...])

                                def wr_one(r, kp_t=kp_t):
                                    pltpu.make_async_copy(
                                        kv_scat.at[s], r.at[kp_t], wr_sem.at[s]
                                    ).start()

                                bswitch(a, [lambda r=r: wr_one(r) for r in raw_refs])
                                pltpu.make_async_copy(
                                    kv_scat.at[s], raw_refs[0].at[0], wr_sem.at[s]
                                ).wait()
            else:
                for d in kv_out:
                    d.wait()
            NO = nh * HD
            scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
            yo = stream.dmatmul("att_out", g0 + nt["att_in"], scr["xb2"], H, lazy=LAZY_ATT)
        if LAZY_ATT:
            with sc("refill"):
                stream.flush(g0, nt["att_in"] + nt["att_out"])
        return yo * ctx.sm(slot, "out_s")

    def epilogue(self, ctx: Any) -> None:
        if ctx.dp.FUSED:
            for s in range(ctx.dp.NS):
                ctx.wait_kv_wr(s)

