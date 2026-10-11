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
"""MoE block (`MoEBlock`): router, EP/TP deduplication, shared expert, and piped routed experts."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel.ops import BF16, F32, FP8, I32, rms_zc, rnd
from vllm_torchtpu.kernels.megakernel.spec import Block, SmallField, WeightDecl


class MoEBlock(Block):
    """Sparse + shared MoE block with dynamic `epool` streaming and `PIPE` software pipelining."""

    name: str = "moe"

    def weights(self, c: Any) -> list[WeightDecl]:
        layout_e = "stacked" if c.stack_e else "per_layer"
        return [
            WeightDecl("router", c.H, c.E, BF16, layers="all", stream_ring="static"),
            WeightDecl("sh13", c.H, 2 * c.ISR, FP8, layers="all", stream_ring="static"),
            WeightDecl("sh2", c.ISR, c.H, FP8, layers="all", stream_ring="static"),
            WeightDecl("e13", c.H, 2 * c.IR, FP8, layers="all", stream_ring="dynamic", layout=layout_e),
            WeightDecl("e2", c.IR, c.H, FP8, layers="all", stream_ring="dynamic", layout=layout_e),
            WeightDecl("es", 1, 2 * c.IR + c.H, F32, layers="all", stream_ring="hbm"),
        ]

    def smalls(self, c: Any) -> list[SmallField]:
        fields = [
            SmallField("post_norm", c.H),
            SmallField("sh13_s", 2 * c.ISR),
            SmallField("sh2_s", c.H),
            SmallField("sh_gate", c.H),
        ]
        if not c.moe_tp and c.e_slots > c.ER:
            fields.append(SmallField("ecopy", c.E))
        return fields

    def emit(self, ctx: Any, l: Any, slot: Any, g0: Any, is_att: Any = True) -> Any:
        c, dp, tiles, W, scr = ctx.c, ctx.dp, ctx.tiles, ctx.W, ctx.scr
        stream, me, skip, sc = ctx.stream, ctx.me, ctx.skip, ctx.sc
        nt, tk = dp.nt, dp.tk
        B, MR, MB, H, E, R = dp.B, dp.MR, dp.MB, dp.H, dp.E, dp.R
        IR, ER, NSS, T13, T2, TE, NBE, CH = (
            dp.IR,
            dp.ER,
            dp.NSS,
            dp.T13,
            dp.T2,
            dp.TE,
            dp.NBE,
            dp.CH,
        )
        PIPE, LAZY_MOE, N_MOE, ECOPY, EXTRA, WIDE = (
            dp.PIPE,
            dp.LAZY_MOE,
            dp.N_MOE,
            dp.ECOPY,
            dp.EXTRA,
            dp.WIDE,
        )
        lane_e8, lane128, row8, EM, FLT_MIN = (
            ctx.lane_e8,
            ctx.lane128,
            ctx.row8,
            ctx.EM,
            ctx.FLT_MIN,
        )

        def es_wait(i):
            s = i % NSS
            with sc("swait", 2):
                pltpu.make_async_copy(
                    W["es"].at[0, 0], scr["es"].at[s], scr["es_sem"].at[s]
                ).wait()

        def gate_up(i, w13):
            h = stream.mm(scr["xb"][...], w13[...]) * scr["es"][i % NSS, :, : 2 * IR]
            return rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)

        def down(i, mid, w2):
            e, s, part = scr["elist"][i], i % NSS, scr["part"]
            wcol = jnp.sum(
                jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
            )
            for j in range(H // CH):
                lo = 2 * IR + j * CH
                y = stream.mm(mid, w2[:, j * CH : (j + 1) * CH])[0:MR]
                y = rnd(y * scr["es"][s, :, lo : lo + CH])
                part[0:MR, j * CH : (j + 1) * CH] += wcol * y

        def expert(l_idx, i, carry):
            e = scr["elist"][i]
            s = i % NSS
            base = i * TE
            acc, xb, xb2, part = scr["acc"], scr["xb"], scr["xb2"], scr["part"]
            if T13 == 1:
                h = stream.mm(xb[...], stream.etake("e13", base)[...])
                stream.egive(base, l_idx)
            else:
                acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)

                def b13(t, cc):
                    u = base + t
                    w = stream.etake("e13", u)
                    k0 = pl.multiple_of(t * tk["e13"], tk["e13"])
                    acc[:, : 2 * IR] += stream.mm(xb[:, pl.ds(k0, tk["e13"])], w[...])
                    stream.egive(u, l_idx)
                    return cc

                lax.fori_loop(0, T13, b13, 0)
                h = acc[:, : 2 * IR]
            es_wait(i)
            h = h * scr["es"][s, :, : 2 * IR]
            mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
            xb2[:, :IR] = mid_.astype(BF16)
            wcol = jnp.sum(
                jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
            )
            u2 = base + T13
            if T2 == 1:
                w2 = stream.etake("e2", u2)[...]
                xm = xb2[:, :IR]
                for j in range(H // CH):
                    lo = 2 * IR + j * CH
                    y = stream.mm(xm, w2[:, j * CH : (j + 1) * CH])[0:MR]
                    y = rnd(y * scr["es"][s, :, lo : lo + CH])
                    part[0:MR, j * CH : (j + 1) * CH] += wcol * y
                stream.egive(u2, l_idx)
            else:
                acc[:, :H] = jnp.zeros((MB, H), F32)

                def b2(t, cc):
                    u = u2 + t
                    w = stream.etake("e2", u)
                    k0 = pl.multiple_of(t * tk["e2"], tk["e2"])
                    acc[:, :H] += stream.mm(xb2[:, pl.ds(k0, tk["e2"])], w[...])
                    stream.egive(u, l_idx)
                    return cc

                lax.fori_loop(0, T2, b2, 0)
                y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
                part[0:MR, :] += wcol * y
            return carry

        def experts_piped(l_idx, n_hit):
            xb2 = scr["xb2"]

            @pl.when(n_hit > 0)
            def _():
                w13 = stream.etake("e13", 0)
                es_wait(0)
                xb2[:, :IR] = gate_up(0, w13)
                stream.egive(0, l_idx, 0)

                def body(i, carry):
                    u13, u2 = (i + 1) * TE, i * TE + T13
                    w13 = stream.etake("e13", u13)
                    es_wait(i + 1)
                    w2 = stream.etake("e2", u2)[...]
                    mid = xb2[:, :IR]
                    nxt = gate_up(i + 1, w13)
                    down(i, mid, w2)
                    xb2[:, :IR] = nxt
                    stream.egive(u13, l_idx, 0)
                    stream.egive(u2, l_idx, T13)
                    return carry

                lax.fori_loop(0, n_hit - 1, body, 0)
                last = n_hit - 1
                u2 = last * TE + T13
                down(last, xb2[:, :IR], stream.etake("e2", u2)[...])
                stream.egive(u2, l_idx, T13)

        def route(l_idx, p8):
            choice = p8
            if ctx.forced_routing:
                rs = l_idx % 2
                pltpu.make_async_copy(
                    ctx.rb_hbm.at[0], scr["rb"].at[rs], scr["rb_sem"].at[rs]
                ).wait()

                @pl.when(l_idx + 1 < dp.L)
                def _():
                    pltpu.make_async_copy(
                        ctx.rb_hbm.at[l_idx + 1],
                        scr["rb"].at[1 - rs],
                        scr["rb_sem"].at[1 - rs],
                    ).start()

                choice = scr["rb"][rs, 0:MR, :]
            with sc("route_topk", 2):
                ids = jnp.zeros((MR, 128), I32)
                if tiles.topk:
                    kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
                    key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e8), F32)
                    for k in range(c.TOPK):
                        kmax = jnp.max(key, axis=-1, keepdims=True)
                        am_ = EM - (lax.bitcast_convert_type(kmax, I32) & EM)
                        ids = jnp.where(lane128 == k, am_, ids)
                        key = jnp.where(key == kmax, -1.0, key)
                    sel = key < 0.0
                else:
                    sel = jnp.zeros((MR, E), jnp.bool_)
                    for k in range(c.TOPK):
                        cmax = jnp.max(choice, axis=-1, keepdims=True)
                        am_ = jnp.min(
                            jnp.where(choice == cmax, lane_e8, E), axis=-1, keepdims=True
                        )
                        ids = jnp.where(lane128 == k, am_, ids)
                        hit = lane_e8 == am_
                        sel = jnp.logical_or(sel, hit)
                        choice = jnp.where(hit, -jnp.inf, choice)
                scr["hitv"][...] = ids
                wsel = jnp.where(sel, p8, 0.0)
                scr["wn"][...] = jnp.where(
                    row8 < B, wsel / jnp.sum(wsel, axis=-1, keepdims=True), 0.0
                )
            with sc("route_dma", 2):
                cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
                cp.start()
                cp.wait()
            stamp = l_idx + 1
            slot_ec = l_idx % 2
            if ECOPY:
                for r in range(R):
                    scr["rload"][r] = jnp.int32(0)

                if tiles.ecopy_opt == 1:
                    def pick_uniq(b, k, n_tot):
                        e = scr["hits"][b, k]
                        hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        new = scr["seen"][hi, lo] != stamp
                        scr["seen"][hi, lo] = stamp
                        scr["ulist"][n_tot] = e
                        return n_tot + jnp.where(new, 1, 0).astype(I32)

                    def classify_uniq(i, carry_c):
                        n, n_cp = carry_c
                        e = scr["ulist"][i]
                        code = scr["ecopy"][slot_ec, 0, e].astype(I32)
                        has_cp = code > 0
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        scr["rload"][r0] = scr["rload"][r0] + jnp.int32(1)
                        mine_fixed = jnp.logical_and(jnp.logical_not(has_cp), r0 == me)
                        scr["elist"][n] = e
                        scr["wlist"][n] = s0
                        scr["cplist"][n_cp] = e
                        return (
                            n + jnp.where(mine_fixed, 1, 0).astype(I32),
                            n_cp + jnp.where(has_cp, 1, 0).astype(I32),
                        )

                    n_tot = jnp.int32(0)
                    with sc("route_dedup", 2):
                        if WIDE:
                            def row_uniq(b, nt_in):
                                return lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_uniq, b), nt_in, unroll=True
                                )

                            n_tot = lax.fori_loop(0, B, row_uniq, n_tot, unroll=2)
                        else:
                            for b in range(B):
                                n_tot = lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_uniq, b), n_tot, unroll=True
                                )
                        n_hit, n_cp = lax.fori_loop(
                            0, n_tot, classify_uniq, (jnp.int32(0), jnp.int32(0))
                        )
                else:
                    def pick_ecopy(b, k, carry):
                        n, n_cp, n_tot = carry
                        e = scr["hits"][b, k]
                        hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        new = scr["seen"][hi, lo] != stamp
                        scr["seen"][hi, lo] = stamp
                        code = scr["ecopy"][slot_ec, 0, e].astype(I32)
                        has_cp = code > 0
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        fixed_new = jnp.logical_and(new, jnp.logical_not(has_cp))
                        cp_new = jnp.logical_and(new, has_cp)
                        scr["rload"][r0] = scr["rload"][r0] + jnp.where(new, 1, 0).astype(I32)
                        mine_fixed = jnp.logical_and(fixed_new, r0 == me)
                        scr["elist"][n] = e
                        scr["wlist"][n] = s0
                        scr["cplist"][n_cp] = e
                        return (
                            n + jnp.where(mine_fixed, 1, 0).astype(I32),
                            n_cp + jnp.where(cp_new, 1, 0).astype(I32),
                            n_tot + jnp.where(new, 1, 0).astype(I32),
                        )

                    carry = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
                    with sc("route_dedup", 2):
                        if WIDE:
                            def row_ecopy(b, c_in):
                                return lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_ecopy, b), c_in, unroll=True
                                )

                            n_hit, n_cp, n_tot = lax.fori_loop(0, B, row_ecopy, carry, unroll=2)
                        else:
                            for b in range(B):
                                carry = lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_ecopy, b), carry, unroll=True
                                )
                            n_hit, n_cp, n_tot = carry

                with sc("route_dedup", 2):
                    def po2c_pass1(j, _):
                        e = scr["cplist"][j]
                        r0 = lax.div(e, ER)
                        c0 = scr["ecopy"][slot_ec, 0, e].astype(I32) - 1
                        r1 = lax.div(c0, EXTRA)
                        l0 = scr["rload"][r0]
                        l1 = scr["rload"][r1]
                        use_c = jnp.logical_or(
                            l1 + 1 < l0,
                            jnp.logical_and(
                                l1 < l0, jnp.logical_or(l0 * R > n_tot, l1 * R < n_tot)
                            ),
                        )
                        d = jnp.where(use_c, 1, 0).astype(I32)
                        scr["rload"][r0] = l0 - d
                        scr["rload"][r1] = l1 + d
                        scr["cpchoice"][j] = d
                        return 0

                    def po2c_pass2(j, n):
                        e = scr["cplist"][j]
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        c0 = scr["ecopy"][slot_ec, 0, e].astype(I32) - 1
                        r1 = lax.div(c0, EXTRA)
                        s1 = ER + (c0 - r1 * EXTRA)
                        use_c = scr["cpchoice"][j] != 0
                        cur = jnp.where(use_c, r1, r0)
                        alt = jnp.where(use_c, r0, r1)
                        l_cur = scr["rload"][cur]
                        l_alt = scr["rload"][alt]
                        flip = jnp.logical_or(
                            l_alt + 1 < l_cur,
                            jnp.logical_and(
                                l_alt < l_cur,
                                jnp.logical_and(l_cur * R > n_tot, l_alt * R < n_tot),
                            ),
                        )
                        d = jnp.where(flip, 1, 0).astype(I32)
                        scr["rload"][cur] = l_cur - d
                        scr["rload"][alt] = l_alt + d
                        final_c = jnp.not_equal(use_c, flip)
                        pick_r = jnp.where(final_c, r1, r0)
                        pick_w = jnp.where(final_c, s1, s0)
                        mine = pick_r == me
                        scr["elist"][n] = e
                        scr["wlist"][n] = pick_w
                        return n + jnp.where(mine, 1, 0).astype(I32)

                    lax.fori_loop(0, n_cp, po2c_pass1, 0)
                    n_hit = lax.fori_loop(0, n_cp, po2c_pass2, n_hit)
            else:
                def pick(b, k, n):
                    e = scr["hits"][b, k]
                    hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                    seen = scr["seen"][hi, lo]
                    new = seen != stamp
                    if not c.moe_tp:
                        new = jnp.logical_and(new, lax.div(e, ER) == me)
                    if ctx.row_mask:
                        on = ctx.mr_ref[b] != 0
                        new = jnp.logical_and(new, on)
                        scr["seen"][hi, lo] = jnp.where(on, stamp, seen)
                    else:
                        scr["seen"][hi, lo] = stamp
                    scr["elist"][n] = e
                    return n + jnp.where(new, 1, 0).astype(I32)

                n_hit = jnp.int32(0)
                with sc("route_dedup", 2):
                    if WIDE:
                        def row(b, n):
                            return lax.fori_loop(
                                0, c.TOPK, functools.partial(pick, b), n, unroll=True
                            )

                        n_hit = lax.fori_loop(0, B, row, n_hit, unroll=2)
                    else:
                        for b in range(B):
                            n_hit = lax.fori_loop(
                                0, c.TOPK, functools.partial(pick, b), n_hit, unroll=True
                            )
            if "experts" in skip:
                n_hit = jnp.int32(0)
            scr["st"][0] = n_hit * TE
            with sc("route_fetch", 2):
                if c.stack_e:
                    for k in range(NBE):
                        @pl.when(k < n_hit * TE)
                        def _(k=k):
                            stream.fetch_e(k, l_idx)
                else:
                    lax.fori_loop(
                        0,
                        jnp.minimum(NBE, n_hit * TE),
                        lambda k, _: (stream.fetch_e(k, l_idx), 0)[1],
                        0,
                    )
            return n_hit

        seg_r = g0 + jnp.where(
            is_att, nt["att_in"] + nt["att_out"], nt["lin_in"] + nt["lin_out"]
        )
        with sc("moe_router"):
            x = rms_zc(scr["res"][...], ctx.sm(slot, "post_norm"), c.eps)
            scr["xb"][...] = x.astype(BF16)
            logits = stream.dmatmul("router", seg_r, scr["xb"], E, lazy=LAZY_MOE)[0:MR]
            glog = jnp.sum(x * ctx.sm(slot, "sh_gate"), axis=-1, keepdims=True)
            scr["sg"][...] = jnp.broadcast_to(jax.nn.sigmoid(glog), (MB, 128))
            mx = jnp.max(logits, axis=-1, keepdims=True)
            pexp = jnp.exp(logits - mx)
        with sc("moe_route"):
            n_hit = route(l, pexp)
        with sc("moe_shared"):
            g13 = stream.dmatmul(
                "sh13", seg_r + nt["router"], scr["xb"], 2 * c.ISR, lazy=LAZY_MOE
            )
            g13 = g13 * ctx.sm(slot, "sh13_s")
            mid = rnd(rnd(jax.nn.silu(rnd(g13[:, : c.ISR]))) * rnd(g13[:, c.ISR :]))
            scr["xb2"][:, : c.ISR] = mid.astype(BF16)
            ysh = stream.dmatmul(
                "sh2", seg_r + nt["router"] + nt["sh13"], scr["xb2"], H, lazy=LAZY_MOE
            )
            scr["part"][...] = ysh * ctx.sm(slot, "sh2_s") * scr["sg"][:, 0:1]
        with sc("moe_experts"):
            if PIPE:
                experts_piped(l, n_hit)
            else:
                lax.fori_loop(0, n_hit, functools.partial(expert, l), 0)
        if LAZY_MOE:
            with sc("refill"):
                stream.flush(seg_r, N_MOE)
        return scr["part"][...]

