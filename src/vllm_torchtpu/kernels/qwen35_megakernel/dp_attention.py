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
"""dp attention: one KV pass for the query heads of all virtual ranks of a core.

The NV virtual ranks of a core share its KV head. Per microbatch:

  1. per virtual rank v: att_in (v's q | gate heads and the head's k | v), q norm / RoPE into
     the v rows of qg, the gate into gate[v]; k / v of v = 0 are the new tokens' K / V;
  2. one online-softmax pass over the head's KV blocks with QR = NV * nh query rows per token
     (row t * QR + v * nh + h: token t, virtual rank v, head h), then the new tokens;
  3. per virtual rank v: output gate, att_out of v's heads; the partial is added to dsum.

The dense stream of an attention layer's microbatch is att_in of virtual ranks 0 .. NV-1, then
att_out of 0 .. NV-1 (rings.fetch_wide, ctx.VATT). The math per element is the TP8 block's
QV verify path (same rounding points), so the result matches the per-virtual-rank form up to the
softmax summation order over KV blocks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from . import pool_alias
from .config import BF16, F32, FP8, I32, NN, NT, U32, bswitch, rnd
from .context import MKBuildContext


class DPAttention:
    def __init__(self, ctx: MKBuildContext, rings, kv_dma, att_blk):
        self.ctx = ctx
        self.rings = rings
        self.kv_dma = kv_dma
        self.att_blk = att_blk  # kv_len / init_worklist / init_rope

    def attention(self, l, slot, m, g_mb):
        """Microbatch m (operands bound) of attention layer l; g_mb: the microbatch's first tile
        in the dense stream. Adds the core's mixer partial (sum over its virtual ranks) to
        scr["dsum"]."""
        ctx = self.ctx
        c, tiles, scr = ctx.c, ctx.tiles, ctx.scr
        pos_ref, raw_refs = ctx.pos_ref, ctx.raw_refs
        sc, mm, rms_zc = ctx.sc, ctx.mm, ctx.rms_zc
        dmatmul = self.rings.dmatmul
        kv_dma = self.kv_dma
        AE, B, T, NS, H, NV = ctx.AE, ctx.B, ctx.T, ctx.NS, ctx.H, ctx.NV
        nh, HD, NHH, QR, TQ, SR = ctx.nh, ctx.HD, ctx.NHH, ctx.QR, ctx.TQ, ctx.SR
        BLK, SUB, NKV, NKVR, KVE = ctx.BLK, ctx.SUB, ctx.NKV, ctx.NKVR, ctx.KVE
        KV_ITEM, EBROWS, PAIR_KV = ctx.KV_ITEM, ctx.EBROWS, ctx.PAIR_KV
        nt, skip, interpret = ctx.nt, ctx.skip, ctx.interpret
        assert ctx.VATT and c.paged_kv and ctx.FUSED and ctx.QV and not ctx.SPLIT and T > 1
        assert QR == NV * nh and SR == 8, (QR, NV, nh, SR)
        n_ai, n_ao = nt["att_in"], nt["att_out"]
        a = l // AE
        ksem = scr["kv_sem"]
        am, al, aacc = scr["am"], scr["al"], scr["aacc"]
        n_items = scr["st"][1]
        KR = NKVR
        P_BLK = BLK // c.page_size

        if KVE:
            def kv_slot(s):
                return pool_alias.view(
                    scr["epool"].at[pl.ds(s * EBROWS, KV_ITEM // 128)], (2, 64, BLK), U32)

            def kv_src(r, kp):
                return r.at[kp].bitcast(U32).reshape(2, 64, c.page_size)

            def kv_dst(s, p):
                return kv_slot(s).at[:, :, pl.ds(p * c.page_size, c.page_size)]
        else:
            kvbuf = scr["kvbuf"]

            def kv_src(r, kp):
                return r.at[kp]

            def kv_dst(s, p):
                return kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)]

        def kv_stage(s):
            return kv_slot(s) if KVE else kvbuf.at[s]

        from jax.experimental.pallas import tpu as pltpu

        def kv_start(i):
            @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
            def _():
                s = i % KR
                k0 = i * P_BLK

                def start_for_ref(r):
                    for p in range(P_BLK):
                        kp = scr["wl_kp"][k0 + p]
                        pltpu.make_async_copy(kv_src(r, kp), kv_dst(s, p), ksem.at[s]).start()

                bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

        def kv_wait(i):
            if "kv" not in skip:
                st_ = kv_stage(i % KR)
                pltpu.make_async_copy(st_, st_, ksem.at[i % KR]).wait()

        lax.fori_loop(0, KR, lambda i, _: (kv_start(i), 0)[1], 0)

        half = c.ROT // 2
        koff = nh * 2 * HD
        cos = scr["rope"][0, :B, :half]
        sin = scr["rope"][1, :B, :half]

        def rope(x, cos_, sin_):
            x1, x2, rest = x[:, :half], x[:, half:c.ROT], x[:, c.ROT:]
            return jnp.concatenate(
                (rnd(rnd(x1 * cos_) - rnd(x2 * sin_)), rnd(rnd(x2 * cos_) + rnd(x1 * sin_)), rest),
                axis=-1)

        # ---- 1. per virtual rank: in-projection, q heads, gates (k / v from v = 0) ----
        def vr_in(v, carry):
            with sc("att_in"):
                ctx.bind_virtual(m, v)
                ybuf = scr["ybuf"]
                y = dmatmul("att_in", g_mb + v * n_ai, scr["xb"], c.NAI)
                ybuf[:, :c.NAI] = rnd(y * ctx.sm(slot, "in_s", c.NAI))
            with sc("att_qk"):
                qn = ctx.sm(slot, "qn")
                for h in range(nh):
                    q0_ = h * 2 * HD
                    qh = rms_zc(ybuf[:B, q0_:q0_ + HD], qn)
                    qh = rnd(rope(qh, cos, sin) * HD ** -0.5)
                    for j in range(NHH):  # row b, head h -> rows b * SR + h of qg[v]
                        scr["qg"][v, jnp.int32(j), pl.ds(h, B, stride=SR), :] = (
                            qh[:, j * 128:(j + 1) * 128])
                    scr["gate"][v, :B, h * HD:(h + 1) * HD] = ybuf[:B, q0_ + HD:q0_ + 2 * HD]

            @pl.when(v == 0)
            def _():
                with sc("att_kvnew"):
                    k = rms_zc(ybuf[:B, koff:koff + HD], ctx.sm(slot, "kn"))
                    vv = ybuf[:B, koff + HD:koff + 2 * HD]
                    k = rope(k, cos, sin)
                    k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                    v_fp8 = jnp.clip(vv, -448.0, 448.0).astype(FP8)
                    scr["knew_o"][:B, :] = k_fp8
                    scr["vnew_o"][:B, :] = v_fp8
                    scr["knew"][:B, :] = k_fp8.astype(F32)
                    scr["vnew"][:B, :] = v_fp8.astype(F32)
            return carry

        lax.fori_loop(0, NV, vr_in, 0)
        ctx.bind_microbatch(m)
        with sc("att_kvrd"):
            kv_dma.kvrd_issue(a)
        with sc("att_qgather"):
            def qrows(v, b):  # virtual rank v's nh query rows of row b (bf16-exact f32)
                return jnp.concatenate(
                    [scr["qg"][v, j, b * SR:b * SR + nh, :] for j in range(NHH)], axis=-1)

            for sq in range(NS):  # token t: rows t * QR + v * nh + h of qb[sq]
                qs = [qrows(v, sq * T + t) for t in range(T) for v in range(NV)]
                scr["qb"][sq] = jnp.concatenate(qs, axis=0).astype(BF16)

        # ---- 2. one KV pass (all virtual ranks' heads), then the new tokens ----
        with sc("att_core"):
            AMASK = tiles.amask
            col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
            am[...] = jnp.full(am.shape, -1e30, F32)
            al[...] = jnp.zeros(al.shape, F32)
            aacc[...] = jnp.zeros(aacc.shape, F32)

            def item_front(i):
                s = i % KR
                b, t = scr["wl_b"][i], scr["wl_t"][i]
                plen = self.att_blk.kv_len(pos_ref[b * T])
                qrows_ = scr["qb"][b]
                if interpret:
                    k_full = kvbuf[s, 0].reshape(HD, BLK)
                    v_full = kvbuf[s, 1].reshape(HD, BLK)
                elif KVE:
                    kst = kv_slot(s)
                    k_u32, v_u32 = kst[0], kst[1]
                else:
                    k_u32 = kvbuf.at[s, 0].bitcast(U32).reshape(64, BLK)[...]
                    v_u32 = kvbuf.at[s, 1].bitcast(U32).reshape(64, BLK)[...]
                if not interpret:
                    k_full = pltpu.bitcast(k_u32, FP8)
                outs = []
                for j in range(BLK // SUB):
                    off = t * BLK + j * SUB
                    rem = plen - off
                    kj = k_full if SUB == BLK else k_full[:, j * SUB:(j + 1) * SUB]
                    if interpret:
                        vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                        vj = jnp.where(col < rem, vj, jnp.zeros_like(vj))
                    else:
                        vu = v_u32 if SUB == BLK else v_u32[:, j * SUB:(j + 1) * SUB]
                        vj = pltpu.bitcast(jnp.where(col < rem, vu, jnp.uint32(0)), FP8)
                    scores = mm(qrows_, kj, NN)
                    scores = jnp.where(col < rem, scores, -jnp.inf)
                    outs.append((scores, jnp.max(scores, axis=-1, keepdims=True), vj))
                return b, outs

            def item_back(b, outs):
                for scores, rmax, vj in outs:
                    m_old = am[b]
                    m_new = jnp.maximum(m_old, rmax)
                    alpha = jnp.exp(m_old - m_new)
                    pr = jnp.exp(scores - m_new[:, 0:1])
                    pv = mm(pr.astype(BF16), vj, NT)
                    al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
                    aacc[b] = aacc[b] * alpha[:, 0:1] + pv
                    am[b] = m_new

            def item(i, carry):
                kv_wait(i)
                if "attn" not in skip:
                    item_back(*item_front(i))
                kv_start(i + KR)
                return carry

            def item_pair(p, carry):
                i0 = 2 * p
                i1 = i0 + 1
                kv_wait(i0)
                kv_wait(i1)
                if "attn" not in skip:
                    f0 = item_front(i0)
                    f1 = item_front(i1)
                    item_back(*f0)
                    item_back(*f1)
                kv_start(i0 + KR)
                kv_start(i1 + KR)
                return carry

            if PAIR_KV:
                lax.fori_loop(0, lax.div(n_items, 2), item_pair, 0)
                @pl.when(lax.rem(n_items, 2) == 1)
                def _():
                    item(n_items - 1, 0)
            else:
                lax.fori_loop(0, n_items, item, 0)

        with sc("att_new"):
            for b in range(B):  # token tb of sequence sq sees new tokens 0 .. tb
                sq, tb = divmod(b, T)
                q0 = tb * QR
                m_old = am[sq, q0:q0 + QR][:, 0:1]
                l_old = al[sq, q0:q0 + QR][:, 0:1]
                a_old = aacc[sq, q0:q0 + QR]
                qf = jnp.concatenate([qrows(v, b) for v in range(NV)], axis=0)  # [QR, HD]
                for j in range(tb + 1):
                    bj = sq * T + j
                    s_new = jnp.sum(qf * scr["knew"][bj:bj + 1, :], axis=-1, keepdims=True)
                    m_new = jnp.maximum(m_old, s_new)
                    alpha = jnp.exp(m_old - m_new)
                    p_new = jnp.exp(s_new - m_new)
                    l_old = l_old * alpha + p_new
                    a_old = a_old * alpha + p_new * scr["vnew"][bj:bj + 1, :]
                    m_old = m_new
                ob = rnd(a_old / l_old)  # rows v * nh + h -> ost[v] rows b * SR + h
                for v in range(NV):
                    for j in range(NHH):
                        scr["ost"][v, j, b * SR:b * SR + nh, :] = (
                            ob[v * nh:(v + 1) * nh, j * 128:(j + 1) * 128])
        with sc("att_kvwr"):
            kv_dma.kv_rmw(a)

        # ---- 3. per virtual rank: output gate and out-projection ----
        def vr_out(v, carry):
            with sc("att_out"):
                ctx.bind_virtual(m, v)
                for h in range(nh):
                    oh = jnp.concatenate(
                        [scr["ost"][v, jnp.int32(j), pl.ds(h, B, stride=SR), :]
                         for j in range(NHH)], axis=-1)
                    gh = rnd(jax.nn.sigmoid(scr["gate"][v, :B, h * HD:(h + 1) * HD]))
                    scr["mix"][:B, h * HD:(h + 1) * HD] = rnd(oh * gh)
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = dmatmul("att_out", g_mb + NV * n_ai + v * n_ao, scr["xb2"], H)
                scr["dsum"][...] += yo * ctx.sm(slot, "out_s")
            return carry

        lax.fori_loop(0, NV, vr_out, 0)
        ctx.bind_microbatch(m)
