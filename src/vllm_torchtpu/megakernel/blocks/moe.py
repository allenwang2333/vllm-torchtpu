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
"""MoE block (`MoEBlock`): router, EP/TP dedup, shared and piped routed experts."""

from __future__ import annotations

import functools
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
    bswitch,
    rnd,
    rup,
)
from ..spec import (
    Block,
    ExecSpec,
    SmallField,
    WeightDecl,
)


@dataclass(frozen=True)
class MoEParams:
    """Block-owned build-time parameters for `MoEBlock`."""

    H: int
    E: int
    TOPK: int
    IR: int
    ISR: int
    ER: int
    EW: int
    EXTRA: int
    CH: int
    T13: int
    T2: int
    TE: int
    NBE: int
    NSS: int
    PIPE: bool
    LAZY_MOE: bool
    N_MOE: int
    ECOPY: bool
    moe_tp: bool
    stack_e: bool
    topk_packed: bool
    ecopy_opt: int
    eps: float
    zero_centered_norm: bool


class MoEBlock(Block):
    """Sparse + shared MoE block with dynamic `epool` streaming and `PIPE`."""

    name: str = "moe"
    reduction = "allreduce"
    input_norm: str | None = "post_norm"

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        er = g.E // ex.R
        isr = g.IS // ex.R
        ir = g.I // ex.R if g.moe_tp else g.I
        ew = g.e_slots if g.e_slots > 0 else (g.E if g.moe_tp else er)
        layout_e = "stacked" if ex.stack_e else "per_layer"
        return [
            WeightDecl("router", K=g.H, N=g.E, dtype=BF16, stream_ring="static"),
            WeightDecl("sh13", K=g.H, N=2 * isr, dtype=FP8, stream_ring="static"),
            WeightDecl("sh2", K=isr, N=g.H, dtype=FP8, stream_ring="static"),
            WeightDecl(
                "e13",
                K=g.H,
                N=2 * ir,
                dtype=FP8,
                stream_ring="dynamic",
                layout=layout_e,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "e2",
                K=ir,
                N=g.H,
                dtype=FP8,
                stream_ring="dynamic",
                layout=layout_e,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "es",
                K=1,
                N=2 * ir + g.H,
                dtype=F32,
                stream_ring="hbm",
                lead_dims=(ew,),
            ),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        er = g.E // ex.R
        isr = g.IS // ex.R
        fields = [
            SmallField("post_norm", g.H),
            SmallField("sh13_s", 2 * isr),
            SmallField("sh2_s", g.H),
            SmallField("sh_gate", g.H),
        ]
        if not g.moe_tp and g.e_slots > er:
            fields.append(SmallField("ecopy", g.E))
        return fields

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        isr = g.IS // ex.R
        ir = g.I // ex.R if g.moe_tp else g.I
        return {
            "acc": max(g.H, g.E, 2 * isr, 2 * ir),
            "xb2": max(isr, ir),
        }

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> MoEParams:
        tiles = ex.resolved_tiles
        er = g.E // ex.R
        isr = g.IS // ex.R
        ir = g.I // ex.R if g.moe_tp else g.I
        ew = g.e_slots if g.e_slots > 0 else (g.E if g.moe_tp else er)
        extra = ew - er
        ch = min(1024, g.H)
        assert g.H % ch == 0, (g.H, ch)
        t13, t2 = plan.nt["e13"], plan.nt["e2"]
        te = t13 + t2
        nbe = tiles.nbe
        nss = 3 + (nbe - 1) // te
        pipe = tiles.pipe and t13 == 1 and t2 == 1
        n_moe = plan.nt["router"] + plan.nt["sh13"] + plan.nt["sh2"]
        lazy_moe = tiles.lazy and n_moe <= tiles.nbd
        ecopy = not g.moe_tp and g.e_slots > er
        assert not (ex.row_mask and ecopy), "row_mask: identity expert placement only"
        return MoEParams(
            H=g.H,
            E=g.E,
            TOPK=g.TOPK,
            IR=ir,
            ISR=isr,
            ER=er,
            EW=ew,
            EXTRA=extra,
            CH=ch,
            T13=t13,
            T2=t2,
            TE=te,
            NBE=nbe,
            NSS=nss,
            PIPE=pipe,
            LAZY_MOE=lazy_moe,
            N_MOE=n_moe,
            ECOPY=ecopy,
            moe_tp=g.moe_tp,
            stack_e=ex.stack_e,
            topk_packed=tiles.topk,
            ecopy_opt=tiles.ecopy_opt,
            eps=g.eps,
            zero_centered_norm=bool(g.zero_centered_norm),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        mp: MoEParams = plan.block_params[self.name]
        nb_picks = rup(ex.B * mp.TOPK, 128)
        scr: dict[str, Any] = {
            "wn": pltpu.VMEM((ex.MR, mp.E), F32),
            "sg": pltpu.VMEM((ex.MB, 128), F32),
            "hitv": pltpu.VMEM((ex.MR, 128), I32),
            "hits": pltpu.SMEM((ex.MR, 128), I32),
            "elist": pltpu.SMEM((nb_picks,), I32),
            "seen": pltpu.SMEM((8, 128), I32),
            "misc_sem": pltpu.SemaphoreType.DMA(()),
            "es": pltpu.VMEM((mp.NSS, 1, 2 * mp.IR + mp.H), F32),
            "es_sem": pltpu.SemaphoreType.DMA((mp.NSS,)),
            "st": pltpu.SMEM((1,), I32),
        }
        if mp.ECOPY:
            scr.update(
                {
                    "ecopy": pltpu.SMEM((2, 1, mp.E), F32),
                    "ec_sem": pltpu.SemaphoreType.DMA((2,)),
                    "wlist": pltpu.SMEM((nb_picks,), I32),
                    "cplist": pltpu.SMEM((nb_picks,), I32),
                    "cpchoice": pltpu.SMEM((nb_picks,), I32),
                    "rload": pltpu.SMEM((ex.R,), I32),
                }
            )
            if mp.ecopy_opt == 1:
                scr["ulist"] = pltpu.SMEM((nb_picks,), I32)
        if ex.forced_routing:
            scr["rb"] = pltpu.VMEM((2, ex.M, mp.E), F32)
            scr["rb_sem"] = pltpu.SemaphoreType.DMA((2,))
        return scr

    def prologue(self, ctx: Any) -> None:
        """Start async zeroing of `seen` and prefetch layer-0 `ecopy`/`rb` if active."""
        ex = ctx.ex
        mp: MoEParams = ctx.block_params(self.name)
        scr, W = ctx.scr, ctx.W
        scr["hitv"][...] = jnp.zeros((ex.MR, 128), I32)
        pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        ).start()
        if mp.ECOPY:
            o_ec = ctx.stream.sp.lay["ecopy"][0]
            pltpu.make_async_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, mp.E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            ).start()
        if ex.forced_routing:
            pltpu.make_async_copy(
                ctx.rb_hbm.at[0], scr["rb"].at[0], scr["rb_sem"].at[0]
            ).start()

    def init_buffers(self, ctx: Any) -> None:
        """Wait for initial `seen` zeroing DMA to complete."""
        scr = ctx.scr
        pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        ).wait()

    def on_layer_start(self, ctx: Any, lyr: Any, slot: Any) -> None:
        """Double-buffer `ecopy` SMEM table alongside `smalls`."""
        mp: MoEParams = ctx.block_params(self.name)
        if not mp.ECOPY:
            return
        scr, W = ctx.scr, ctx.W
        o_ec = ctx.stream.sp.lay["ecopy"][0]
        pltpu.make_async_copy(
            W["smalls"].at[0, :, pl.ds(o_ec, mp.E)],
            scr["ecopy"].at[slot],
            scr["ec_sem"].at[slot],
        ).wait()

        @pl.when(lyr + 1 < ctx.stream.sp.L)
        def _():
            pltpu.make_async_copy(
                W["smalls"].at[lyr + 1, :, pl.ds(o_ec, mp.E)],
                scr["ecopy"].at[1 - slot],
                scr["ec_sem"].at[1 - slot],
            ).start()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        mp: MoEParams = ctx.block_params(self.name)
        W, scr = ctx.W, ctx.scr
        stream, me, skip, sc = ctx.stream, ctx.me, ctx.skip, ctx.sc
        sp = stream.sp
        tk = sp.tk
        B, MR, MB, R, WIDE = ex.B, ex.MR, ex.MB, ex.R, ex.WIDE
        H, E, TOPK, IR, ISR, ER, EXTRA, CH = (
            mp.H,
            mp.E,
            mp.TOPK,
            mp.IR,
            mp.ISR,
            mp.ER,
            mp.EXTRA,
            mp.CH,
        )
        NSS, T13, T2, TE, NBE = mp.NSS, mp.T13, mp.T2, mp.TE, mp.NBE
        PIPE, LAZY_MOE, ECOPY = mp.PIPE, mp.LAZY_MOE, mp.ECOPY
        lane_e8 = lax.broadcasted_iota(I32, (MR, E), 1)
        lane128 = lax.broadcasted_iota(I32, (MR, 128), 1)
        row8 = lax.broadcasted_iota(I32, (MR, 1), 0)
        EM = (1 << max(1, (E - 1).bit_length())) - 1
        FLT_MIN = float(jnp.finfo(F32).tiny)

        def fetch_e(u, l_idx, t=None):
            """Issue routed expert unit `u` into dynamic bank `u % NBE`."""
            if t is None and isinstance(u, int):
                t = u % TE
            i = u // TE if t is None else (u - t) // TE
            if ECOPY:
                w = scr["wlist"][i]
            else:
                e = scr["elist"][i]
                w = e if mp.moe_tp else e - me * ER
            bank = u % NBE

            def es_copy():
                s = i % NSS
                pltpu.make_async_copy(
                    W["es"].at[l_idx, w], scr["es"].at[s], scr["es_sem"].at[s]
                ).start()

            def e13_start(tt):
                if mp.stack_e:
                    stream.ecopy("e13", W["e13"].at[l_idx, w, tt], bank).start()
                else:
                    bswitch(
                        l_idx,
                        [
                            lambda r=r: stream.ecopy("e13", r.at[w, tt], bank).start()
                            for r in W["e13"]
                        ],
                    )

            def e2_start(tt):
                if mp.stack_e:
                    stream.ecopy("e2", W["e2"].at[l_idx, w, tt], bank).start()
                else:
                    bswitch(
                        l_idx,
                        [
                            lambda r=r: stream.ecopy("e2", r.at[w, tt], bank).start()
                            for r in W["e2"]
                        ],
                    )

            if t is not None:
                if t < T13:
                    e13_start(t)
                else:
                    e2_start(t - T13)
                if t == 0:
                    es_copy()
                return
            t = u - i * TE

            @pl.when(t < T13)
            def _():
                e13_start(t)

            @pl.when(t >= T13)
            def _():
                e2_start(t - T13)

            pl.when(t == 0)(es_copy)

        def etake(kind: str, u):
            bank = u % NBE
            with sc("ewait", 2):
                w0 = W[kind].at[0, 0, 0] if mp.stack_e else W[kind][0].at[0, 0]
                stream.ecopy(kind, w0, bank).wait()
            return stream.eview(kind, bank)

        def egive(u, l_idx, t=None):
            nu = u + NBE
            nt_ = None if t is None or NBE % TE else t

            @pl.when(nu < scr["st"][0])
            def _():
                fetch_e(nu, l_idx, nt_)

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
                h = stream.mm(xb[...], etake("e13", base)[...])
                egive(base, l_idx)
            else:
                acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)

                def b13(t, cc):
                    u = base + t
                    w = etake("e13", u)
                    k0 = pl.multiple_of(t * tk["e13"], tk["e13"])
                    acc[:, : 2 * IR] += stream.mm(xb[:, pl.ds(k0, tk["e13"])], w[...])
                    egive(u, l_idx)
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
                w2 = etake("e2", u2)[...]
                xm = xb2[:, :IR]
                for j in range(H // CH):
                    lo = 2 * IR + j * CH
                    y = stream.mm(xm, w2[:, j * CH : (j + 1) * CH])[0:MR]
                    y = rnd(y * scr["es"][s, :, lo : lo + CH])
                    part[0:MR, j * CH : (j + 1) * CH] += wcol * y
                egive(u2, l_idx)
            else:
                acc[:, :H] = jnp.zeros((MB, H), F32)

                def b2(t, cc):
                    u = u2 + t
                    w = etake("e2", u)
                    k0 = pl.multiple_of(t * tk["e2"], tk["e2"])
                    acc[:, :H] += stream.mm(xb2[:, pl.ds(k0, tk["e2"])], w[...])
                    egive(u, l_idx)
                    return cc

                lax.fori_loop(0, T2, b2, 0)
                y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
                part[0:MR, :] += wcol * y
            return carry

        def experts_piped(l_idx, n_hit):
            xb2 = scr["xb2"]

            @pl.when(n_hit > 0)
            def _():
                w13 = etake("e13", 0)
                es_wait(0)
                xb2[:, :IR] = gate_up(0, w13)
                egive(0, l_idx, 0)

                def body(i, carry):
                    u13, u2 = (i + 1) * TE, i * TE + T13
                    w13 = etake("e13", u13)
                    es_wait(i + 1)
                    w2 = etake("e2", u2)[...]
                    mid = xb2[:, :IR]
                    nxt = gate_up(i + 1, w13)
                    down(i, mid, w2)
                    xb2[:, :IR] = nxt
                    egive(u13, l_idx, 0)
                    egive(u2, l_idx, T13)
                    return carry

                lax.fori_loop(0, n_hit - 1, body, 0)
                last = n_hit - 1
                u2 = last * TE + T13
                down(last, xb2[:, :IR], etake("e2", u2)[...])
                egive(u2, l_idx, T13)

        def route(l_idx, p8):
            choice = p8
            if ex.forced_routing:
                rs = l_idx % 2
                pltpu.make_async_copy(
                    ctx.rb_hbm.at[0], scr["rb"].at[rs], scr["rb_sem"].at[rs]
                ).wait()

                @pl.when(l_idx + 1 < sp.L)
                def _():
                    pltpu.make_async_copy(
                        ctx.rb_hbm.at[l_idx + 1],
                        scr["rb"].at[1 - rs],
                        scr["rb_sem"].at[1 - rs],
                    ).start()

                choice = scr["rb"][rs, 0:MR, :]
            with sc("route_topk", 2):
                ids = jnp.zeros((MR, 128), I32)
                if mp.topk_packed:
                    kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
                    key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e8), F32)
                    for k in range(TOPK):
                        kmax = jnp.max(key, axis=-1, keepdims=True)
                        am_ = EM - (lax.bitcast_convert_type(kmax, I32) & EM)
                        ids = jnp.where(lane128 == k, am_, ids)
                        key = jnp.where(key == kmax, -1.0, key)
                    sel = key < 0.0
                else:
                    sel = jnp.zeros((MR, E), jnp.bool_)
                    for k in range(TOPK):
                        cmax = jnp.max(choice, axis=-1, keepdims=True)
                        am_ = jnp.min(
                            jnp.where(choice == cmax, lane_e8, E),
                            axis=-1,
                            keepdims=True,
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

                if mp.ecopy_opt == 1:

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
                                    0,
                                    TOPK,
                                    functools.partial(pick_uniq, b),
                                    nt_in,
                                    unroll=True,
                                )

                            n_tot = lax.fori_loop(0, B, row_uniq, n_tot, unroll=2)
                        else:
                            for b in range(B):
                                n_tot = lax.fori_loop(
                                    0,
                                    TOPK,
                                    functools.partial(pick_uniq, b),
                                    n_tot,
                                    unroll=True,
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
                        scr["rload"][r0] = scr["rload"][r0] + jnp.where(
                            new, 1, 0
                        ).astype(I32)
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
                                    0,
                                    TOPK,
                                    functools.partial(pick_ecopy, b),
                                    c_in,
                                    unroll=True,
                                )

                            n_hit, n_cp, n_tot = lax.fori_loop(
                                0, B, row_ecopy, carry, unroll=2
                            )
                        else:
                            for b in range(B):
                                carry = lax.fori_loop(
                                    0,
                                    TOPK,
                                    functools.partial(pick_ecopy, b),
                                    carry,
                                    unroll=True,
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
                    if not mp.moe_tp:
                        new = jnp.logical_and(new, lax.div(e, ER) == me)
                    if ex.row_mask:
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
                                0, TOPK, functools.partial(pick, b), n, unroll=True
                            )

                        n_hit = lax.fori_loop(0, B, row, n_hit, unroll=2)
                    else:
                        for b in range(B):
                            n_hit = lax.fori_loop(
                                0, TOPK, functools.partial(pick, b), n_hit, unroll=True
                            )
            if "experts" in skip:
                n_hit = jnp.int32(0)
            scr["st"][0] = n_hit * TE
            with sc("route_fetch", 2):
                if mp.stack_e:
                    for k in range(NBE):

                        @pl.when(k < n_hit * TE)
                        def _(k=k):
                            fetch_e(k, l_idx)
                else:
                    lax.fori_loop(
                        0,
                        jnp.minimum(NBE, n_hit * TE),
                        lambda k, _: (fetch_e(k, l_idx), 0)[1],
                        0,
                    )
            return n_hit

        with stream.segment(lazy=LAZY_MOE):
            with sc("moe_router"):
                assert ctx.norm_x is not None, (
                    "MoEBlock requires input_norm='post_norm' to populate ctx.norm_x"
                )
                x = ctx.norm_x
                logits = stream.dmatmul("router", scr["xb"], E)[0:MR]
                glog = jnp.sum(x * ctx.sm(slot, "sh_gate"), axis=-1, keepdims=True)
                scr["sg"][...] = jnp.broadcast_to(jax.nn.sigmoid(glog), (MB, 128))
                mx = jnp.max(logits, axis=-1, keepdims=True)
                pexp = jnp.exp(logits - mx)
            with sc("moe_route"):
                n_hit = route(l, pexp)
            with sc("moe_shared"):
                g13 = stream.dmatmul("sh13", scr["xb"], 2 * ISR)
                g13 = g13 * ctx.sm(slot, "sh13_s")
                mid = rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
                scr["xb2"][:, :ISR] = mid.astype(BF16)
                ysh = stream.dmatmul("sh2", scr["xb2"], H)
                scr["part"][...] = ysh * ctx.sm(slot, "sh2_s") * scr["sg"][:, 0:1]
            with sc("moe_experts"):
                if PIPE:
                    experts_piped(l, n_hit)
                else:
                    lax.fori_loop(0, n_hit, functools.partial(expert, l), 0)
        return scr["part"][...]
