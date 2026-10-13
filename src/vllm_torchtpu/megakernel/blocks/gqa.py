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
"""Grouped-Query Attention (`GQABlock`) with sequence-split KV ring and MTP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    rnd,
    rope_partial,
    rup,
)
from ..spec import (
    Block,
    ExecSpec,
    SmallField,
    StateDecl,
    WeightDecl,
)

NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))


@dataclass(frozen=True)
class GQAParams:
    """Block-owned build-time parameters for `GQABlock` (`direct_hbm` KV cache)."""

    H: int
    nh: int
    HD: int
    ROT: int
    NAI: int
    G: int
    BLK: int
    NKV: int
    SUB: int
    AMASK: int
    SPLIT: bool
    TS: int
    QR: int
    TQ: int
    PW: int
    NWL: int
    PACK: bool
    PB: int
    NBP: int
    LAZY_ATT: bool
    eps: float
    theta: float
    zero_centered_norm: bool

    def qrow(self, b: int) -> tuple[int, int]:
        return b // self.PB, (b % self.PB) * self.nh


class GQABlock(Block):
    """Full GQA block with QK RMSNorm, partial RoPE, Q/softmax exchange, and gate."""

    name: str = "gqa"
    reduction = "allreduce"
    input_norm: str | None = "in_norm"

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            WeightDecl("att_in", K=g.H, N=nai, dtype=FP8, stream_ring="static"),
            WeightDecl("att_out", K=nh * g.HD, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", nai),
            SmallField("out_s", g.H),
            SmallField("qn", g.HD),
            SmallField("kn", g.HD),
        ]

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return [
            StateDecl("kc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
            StateDecl("vc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return [
            StateDecl("knew", (ex.MR, g.HD), F32, access="w"),
            StateDecl("vnew", (ex.MR, g.HD), F32, access="w"),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return {
            "acc": max(nai, g.H),
            "ybuf": nai,
            "xb2": nh * g.HD,
            "mix": nh * g.HD,
        }

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> GQAParams:
        tiles = ex.resolved_tiles
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        G = max(1, ex.R // g.NKV)
        BLK, NKV = tiles.kv_blk, tiles.nkv
        assert ex.ctx % BLK == 0 and BLK % 128 == 0, (ex.ctx, BLK)
        SUB = tiles.kv_sub or BLK
        assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
        SPLIT = tiles.split and G > 1
        assert not SPLIT or (G & (G - 1) == 0), G
        TS = G if SPLIT else 1
        QR = rup(G * nh, 8) if SPLIT else 8
        TQ = ex.T * QR
        PW = g.HD + 128
        max_nb = -(-ex.ctx // BLK)
        NWL = rup(ex.NS * -(-max_nb // TS), 8)
        PACK = (ex.WIDE if tiles.pack < 0 else bool(tiles.pack)) and SPLIT and nh < 8
        PB = 8 // nh if PACK else 1
        NBP = -(-ex.B // PB)
        nt_att = plan.nt["att_in"] + plan.nt["att_out"]
        LAZY_ATT = tiles.lazy and nt_att <= tiles.nbd
        return GQAParams(
            H=g.H,
            nh=nh,
            HD=g.HD,
            ROT=g.ROT,
            NAI=nai,
            G=G,
            BLK=BLK,
            NKV=NKV,
            SUB=SUB,
            AMASK=tiles.amask,
            SPLIT=SPLIT,
            TS=TS,
            QR=QR,
            TQ=TQ,
            PW=PW,
            NWL=NWL,
            PACK=PACK,
            PB=PB,
            NBP=NBP,
            LAZY_ATT=LAZY_ATT,
            eps=g.eps,
            theta=g.theta,
            zero_centered_norm=bool(g.zero_centered_norm),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        ap: GQAParams = plan.block_params[self.name]
        scr: dict[str, Any] = {
            "kbuf": pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16),
            "vbuf": pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16),
            "kv_sem": pltpu.SemaphoreType.DMA((2, ap.NKV)),
            "wl_b": pltpu.SMEM((ap.NWL,), I32),
            "wl_t": pltpu.SMEM((ap.NWL,), I32),
            "wl_cnt": pltpu.SMEM((1,), I32),
            "qb": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), BF16),
            "am": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
            "al": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
            "aacc": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), F32),
            "knew": pltpu.VMEM((ex.MR, ap.HD), F32),
            "vnew": pltpu.VMEM((ex.MR, ap.HD), F32),
            "gate": pltpu.VMEM((ex.MR, ap.nh * ap.HD), F32),
            "kv_out_sem": pltpu.SemaphoreType.DMA((2,)),
        }
        if ap.SPLIT:
            q_rows = ap.NBP if ap.PACK else ex.B
            scr.update(
                {
                    "qg": pltpu.VMEM((ap.G, q_rows, 8, ap.HD), F32),
                    "pso": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                    "pri": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                    "sq_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sq_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sp_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sp_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                }
            )
        elif ex.VER:
            scr["qg"] = pltpu.VMEM((1, ex.B, 8, ap.HD), F32)
        return scr

    def prologue(self, ctx: Any) -> None:
        """Build this rank's KV worklist (`wl_b`, `wl_t`, `wl_cnt[0]`)."""
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, pos_ref = ctx.scr, ctx.pos_ref
        NS, T, BLK, TS, SPLIT = ex.NS, ex.T, ap.BLK, ap.TS, ap.SPLIT
        gi = lax.rem(ctx.me, ap.G) if SPLIT else jnp.int32(0)
        max_nb = -(-ex.ctx // BLK)

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
        scr["wl_cnt"][0] = n

    def init_buffers(self, ctx: Any) -> None:
        """Initialize attention-specific VMEM scratch buffers after barrier."""
        ap: GQAParams = ctx.block_params(self.name)
        scr = ctx.scr
        if ap.SPLIT:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if ap.PACK:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, ST, O = ctx.scr, ctx.ST, ctx.O  # noqa: E741
        pos_ref, skip, sc = ctx.pos_ref, ctx.skip, ctx.sc
        stream, comm = ctx.stream, ctx.comm
        NKV, BLK, SUB, AMASK = ap.NKV, ap.BLK, ap.SUB, ap.AMASK
        B, T, NS, VER = ex.B, ex.T, ex.NS, ex.VER
        G, nh, HD, H, NAI, ROT = ap.G, ap.nh, ap.HD, ap.H, ap.NAI, ap.ROT
        QR, TQ, PW = ap.QR, ap.TQ, ap.PW
        SPLIT, PACK, PB, NBP, LAZY_ATT = (
            ap.SPLIT,
            ap.PACK,
            ap.PB,
            ap.NBP,
            ap.LAZY_ATT,
        )
        gi = lax.rem(ctx.me, G) if SPLIT else jnp.int32(0)

        a = stream.sp.block_ord(l, self.name)
        ksem = scr["kv_sem"]
        am, al, aacc = scr["am"], scr["al"], scr["aacc"]
        n_items = scr["wl_cnt"][0]
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

        with stream.segment(lazy=LAZY_ATT):
            with sc("att_in"):
                y = stream.dmatmul("att_in", scr["xb"], NAI)
                ybuf = scr["ybuf"]
                ybuf[:, :NAI] = rnd(y * ctx.sm(slot, "in_s", NAI))
            qn = ctx.sm(slot, "qn")
            kn = ctx.sm(slot, "kn")
            half = ROT // 2
            inv = 1.0 / (
                ap.theta
                ** (lax.broadcasted_iota(I32, (1, half), 1).astype(F32) * 2.0 / ROT)
            )

            koff = nh * 2 * HD
            with sc("att_qk"):
                for b in range(B):
                    pb = pos_ref[b]
                    ang = pb.astype(F32) * inv
                    cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                    qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                    scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                    q = rmsnorm(
                        qg[:, :HD], qn, ap.eps, zero_centered=ap.zero_centered_norm
                    )
                    k = rmsnorm(
                        ybuf[b : b + 1, koff : koff + HD],
                        kn,
                        ap.eps,
                        zero_centered=ap.zero_centered_norm,
                    )
                    v = ybuf[b : b + 1, koff + HD : koff + 2 * HD]
                    q = rnd(rope_partial(q, cos, sin, ROT) * HD**-0.5)
                    k = rope_partial(k, cos, sin, ROT)
                    if PACK:
                        t_, o_ = ap.qrow(b)
                        scr["qg"][gi, t_, o_ : o_ + nh] = q
                    else:
                        if nh < 8:
                            q = jnp.concatenate(
                                (q, jnp.zeros((8 - nh, HD), F32)), axis=0
                            )
                        if SPLIT or VER:
                            scr["qg"][gi, b] = q
                        else:
                            scr["qb"][b] = q.astype(BF16)
                    scr["knew"][b : b + 1, :] = k
                    scr["vnew"][b : b + 1, :] = v
                kv_out = (
                    pltpu.make_async_copy(
                        scr["knew"], O["knew"].at[a], scr["kv_out_sem"].at[0]
                    ),
                    pltpu.make_async_copy(
                        scr["vnew"], O["vnew"].at[a], scr["kv_out_sem"].at[1]
                    ),
                )
                for d in kv_out:
                    d.start()

            if SPLIT or VER:
                with sc("att_qgather"):
                    qgb = scr["qg"]
                    if SPLIT:
                        comm.exchange(
                            comm.group_copies(
                                lambda off: qgb.at[gi],
                                qgb.at[gi],
                                scr["sq_ssem"],
                                scr["sq_rsem"],
                                G,
                            )
                        )

                    def group_q(b):
                        if not SPLIT:
                            return qgb[0, b]
                        t_, o_ = ap.qrow(b)
                        qa = jnp.concatenate(
                            [qgb[g_idx, t_, o_ : o_ + nh, :] for g_idx in range(G)],
                            axis=0,
                        )
                        if G * nh < QR:
                            qa = jnp.concatenate(
                                (qa, jnp.zeros((QR - G * nh, HD), F32)), 0
                            )
                        return qa

                    for sq in range(NS):
                        qs = [group_q(sq * T + t) for t in range(T)]
                        qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
                        scr["qb"][sq] = qs.astype(BF16)

            with sc("att_core"):
                col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
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
                        for j in range(BLK // SUB):
                            off = t * BLK + j * SUB
                            rem = plen - off
                            full_blk = off + SUB <= plen
                            kj = (
                                kbuf[s]
                                if SUB == BLK
                                else kbuf[s, j * SUB : (j + 1) * SUB]
                            )
                            vj = (
                                vbuf[s]
                                if SUB == BLK
                                else vbuf[s, j * SUB : (j + 1) * SUB]
                            )
                            scores = stream.mm(qrows, kj, NT)
                            if AMASK == 0:
                                scores = jnp.where(col + off < plen, scores, -jnp.inf)
                            elif AMASK in (1, 4):
                                scores = jnp.where(col < rem, scores, -jnp.inf)
                            else:
                                scores = lax.cond(
                                    full_blk,
                                    lambda s_val=scores: s_val,
                                    lambda s_val=scores, r_val=rem: jnp.where(
                                        col < r_val, s_val, -jnp.inf
                                    ),
                                )
                            m_old = am[b]
                            m_new = jnp.maximum(
                                m_old, jnp.max(scores, axis=-1, keepdims=True)
                            )
                            alpha = jnp.exp(m_old - m_new)
                            pr = jnp.exp(scores - m_new[:, 0:1])
                            pv = stream.mm(pr.astype(BF16), vj, NN)
                            al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
                            aacc[b] = aacc[b] * alpha[:, 0:1] + pv
                            am[b] = m_new
                    kv_start(i + NKV)
                    return carry

                lax.fori_loop(0, n_items, item, 0)

            if SPLIT:
                with sc("att_merge"):
                    pso, pri = scr["pso"], scr["pri"]
                    lane_pw = lax.broadcasted_iota(I32, (nh, 128), 1)
                    for b in range(B):
                        t_, o_ = ap.qrow(b)
                        sq, q0 = b // T, (b % T) * QR
                        mb, lb = am[sq, q0 : q0 + QR], al[sq, q0 : q0 + QR]
                        ab = aacc[sq, q0 : q0 + QR]
                        for g_idx in range(G):
                            r0 = g_idx * nh
                            pso[g_idx, t_, o_ : o_ + nh, 0:HD] = ab[r0 : r0 + nh]
                            pso[g_idx, t_, o_ : o_ + nh, HD:PW] = jnp.where(
                                lane_pw < 64, mb[r0 : r0 + nh], lb[r0 : r0 + nh]
                            )
                    pri[gi] = pso[gi]
                    comm.exchange(
                        comm.group_copies(
                            lambda off: pso.at[gi ^ off],
                            pri.at[gi],
                            scr["sp_ssem"],
                            scr["sp_rsem"],
                            G,
                        )
                    )
                    for t in range(NBP):
                        parts = [pri[g_idx, t] for g_idx in range(G)]
                        mg = [x[:, HD : HD + 1] for x in parts]
                        m = mg[0]
                        for x in mg[1:]:
                            m = jnp.maximum(m, x)
                        wg = [jnp.exp(x - m) for x in mg]
                        lsum: Any = sum(
                            w * x[:, HD + 64 : HD + 65] for w, x in zip(wg, parts)
                        )
                        asum: Any = sum(w * x[:, :HD] for w, x in zip(wg, parts))
                        if not PACK:
                            sq, q0 = t // T, (t % T) * QR
                            am[sq, q0 : q0 + 8] = jnp.broadcast_to(m, (8, 128))
                            al[sq, q0 : q0 + 8] = jnp.broadcast_to(lsum, (8, 128))
                            aacc[sq, q0 : q0 + 8] = asum
                            continue
                        for b in range(t * PB, min(B, (t + 1) * PB)):
                            o_ = ap.qrow(b)[1]
                            sq, q0 = b // T, (b % T) * QR
                            am[sq, q0 : q0 + nh] = jnp.broadcast_to(
                                m[o_ : o_ + nh], (nh, 128)
                            )
                            al[sq, q0 : q0 + nh] = jnp.broadcast_to(
                                lsum[o_ : o_ + nh], (nh, 128)
                            )
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
                        t_, o_ = ap.qrow(b)
                        qf = scr["qg"][gi, t_, o_ : o_ + nh]
                    else:
                        qf = (
                            scr["qg"][gi, b]
                            if SPLIT or VER
                            else scr["qb"][b].astype(F32)
                        )
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
                        s_new = jnp.sum(
                            qf * scr["knew"][b : b + 1, :], axis=-1, keepdims=True
                        )
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_tot = l_old * alpha + p_new
                        outb = (
                            a_old * alpha + p_new * scr["vnew"][b : b + 1, :]
                        ) / l_tot
                    for h in range(nh):
                        oh = rnd(outb[h : h + 1, :])
                        gh = rnd(
                            jax.nn.sigmoid(
                                scr["gate"][b : b + 1, h * HD : (h + 1) * HD]
                            )
                        )
                        scr["mix"][b : b + 1, h * HD : (h + 1) * HD] = rnd(oh * gh)

            with sc("att_out"):
                for d in kv_out:
                    d.wait()
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = stream.dmatmul("att_out", scr["xb2"], H)

        return yo * ctx.sm(slot, "out_s")

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Scatter `knew` and `vnew` into `kc` and `vc` inside `shard_map`."""
        kc, vc = states["kc"], states["vc"]
        knew, vnew = raw_outputs["knew"], raw_outputs["vnew"]
        if ex.NS > 16:
            bi = jnp.arange(ex.B) if ex.T == 1 else jnp.arange(ex.B) // ex.T
            kw = dict(
                indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"
            )
            kc = kc.at[:, bi, pos].set(knew[:, : ex.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, : ex.B].astype(BF16), **kw)
            return {"kc": kc, "vc": vc}
        for s_idx in range(ex.NS):
            b0 = s_idx * ex.T
            upd = (0, s_idx, pos[b0], 0)
            kc = lax.dynamic_update_slice(
                kc, knew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
            vc = lax.dynamic_update_slice(
                vc, vnew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
        return {"kc": kc, "vc": vc}
