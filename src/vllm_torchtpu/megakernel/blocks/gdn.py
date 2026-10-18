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
"""Composable GatedDeltaNet (`GDNBlock`) megakernel block."""

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
    rnd,
    rup,
)
from ..spec import (
    Block,
    ExecSpec,
    SmallField,
    StateDecl,
    WeightDecl,
)


@dataclass(frozen=True)
class GDNLinInLayout:
    """Column layout `[qkv | z | b (0:vh) | a (64:64+vh)]` of `lin_in` / `ybuf`."""

    kh: int
    vh: int
    KD: int
    VD: int
    QKV: int
    Z: int
    BA: int
    A_OFF: int
    BA_PAD: int
    NLI: int


@dataclass(frozen=True)
class GDNParams:
    """Block-local parameters owned and computed by `GDNBlock.plan_params`."""

    kh: int
    vh: int
    KD: int
    VD: int
    QKV: int
    Z: int
    BA: int
    A_OFF: int
    BA_PAD: int
    NLI: int
    H: int
    CR: int
    HR: int
    GCH: bool
    CB: int
    NF: int
    TR: int
    NCH: int
    NRB: int
    CBV: int
    NGDN: int
    eps: float
    zero_centered_norm: bool
    l2_eps: float = 1e-6


class GDNBlock(Block):
    """GatedDeltaNet block supporting decode (`T=1`), verify (`T>1`), and `GCH`."""

    name = "gdn"
    reduction = "allreduce"
    input_norm: str | None = "in_norm"

    @staticmethod
    def lin_in_layout(g: Any, ex_or_r: ExecSpec | int) -> GDNLinInLayout:
        """Return shared column offsets for `lin_in` (`[qkv | z | b | a]`)."""
        R = ex_or_r.R if isinstance(ex_or_r, ExecSpec) else int(ex_or_r)
        kh = g.KH // R
        vh = g.VH // R
        KD, VD = g.KD, g.VD
        qkv = 2 * kh * KD + vh * VD
        z = vh * VD
        ba = qkv + z
        ba_pad = 128
        return GDNLinInLayout(
            kh=kh,
            vh=vh,
            KD=KD,
            VD=VD,
            QKV=qkv,
            Z=z,
            BA=ba,
            A_OFF=ba + 64,
            BA_PAD=ba_pad,
            NLI=ba + ba_pad,
        )

    @staticmethod
    def accept(
        states: dict[str, Any],
        ckpts: dict[str, Any],
        acc: Any,
        *,
        T: int,
        vh: int,
    ) -> dict[str, Any]:
        """Jittable on-device selection of post-verify GDN states from checkpoints."""
        c_ckpt = ckpts["conv_ckpt"] if "conv_ckpt" in ckpts else ckpts.get("conv")
        r_ckpt = ckpts["rec_ckpt"] if "rec_ckpt" in ckpts else ckpts.get("rec")
        out = {k: v for k, v in states.items() if k not in ("conv_ckpt", "rec_ckpt")}
        if c_ckpt is None or r_ckpt is None:
            return out

        NS = c_ckpt.shape[-3]
        acc_arr = jnp.asarray(acc, dtype=I32)
        assert acc_arr.shape == (NS,), (acc_arr.shape, NS)

        def _accept_conv_seq(c_seq, a_s):
            taps = lax.dynamic_slice_in_dim(c_seq, a_s, 3, axis=-2)
            return c_seq.at[..., 0:3, :].set(taps)

        conv_next = jax.vmap(_accept_conv_seq, in_axes=(-3, 0), out_axes=-3)(
            c_ckpt, acc_arr
        )

        lead = r_ckpt.shape[:-3]
        KD, VD = r_ckpt.shape[-2], r_ckpt.shape[-1]
        r_unflat = r_ckpt.reshape((*lead, NS, T, vh, KD, VD))
        r_sel = jax.vmap(
            lambda r_seq, a_s: lax.dynamic_index_in_dim(
                r_seq, a_s - 1, axis=-4, keepdims=False
            ),
            in_axes=(-5, 0),
            out_axes=-4,
        )(r_unflat, acc_arr)
        rec_next = r_sel.reshape((*lead, NS * vh, KD, VD))

        out["conv"] = conv_next
        out["rec"] = rec_next
        return out

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        lay = self.lin_in_layout(g, ex)
        return [
            WeightDecl("lin_in", K=g.H, N=lay.NLI, dtype=FP8, stream_ring="static"),
            WeightDecl("lin_out", K=lay.Z, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        lay = self.lin_in_layout(g, ex)
        conv_shape = (
            (rup(3 * ex.B, 8), lay.QKV)
            if ex.T == 1
            else (ex.NS, rup(3 + ex.T, 8), lay.QKV)
        )
        return [
            StateDecl("conv", shape=conv_shape, dtype=F32, access="rw"),
            StateDecl(
                "rec", shape=(ex.NS * lay.vh, lay.KD, lay.VD), dtype=BF16, access="rw"
            ),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        lay = self.lin_in_layout(g, ex)
        if ex.VER:
            return [
                StateDecl(
                    "conv_ckpt",
                    shape=(ex.NS, rup(3 + ex.T, 8), lay.QKV),
                    dtype=F32,
                    access="w",
                ),
                StateDecl(
                    "rec_ckpt",
                    shape=(ex.B * lay.vh, lay.KD, lay.VD),
                    dtype=BF16,
                    access="w",
                ),
            ]
        return [
            StateDecl("conv", shape=(rup(3 * ex.B, 8), lay.QKV), dtype=F32, access="w"),
            StateDecl(
                "rec", shape=(ex.B * lay.vh, lay.KD, lay.VD), dtype=BF16, access="w"
            ),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        lay = self.lin_in_layout(g, ex)
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", lay.NLI),
            SmallField("out_s", g.H),
            SmallField("conv", 4 * lay.QKV),
            SmallField("alog", 128),
            SmallField("dtb", 128),
            SmallField("gnorm", lay.VD),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        lay = self.lin_in_layout(g, ex)
        return {
            "acc": lay.NLI,
            "ybuf": lay.NLI,
            "xb2": lay.Z,
            "mix": lay.Z,
        }

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> GDNParams:
        tiles = ex.resolved_tiles
        R, B, T = ex.R, ex.B, ex.T
        assert not (ex.VER and tiles.inplace), (
            "verify returns per-token checkpoints: no aliasing"
        )
        assert g.KH % R == 0 and g.VH % R == 0, (g.KH, g.VH, R)
        assert g.VH % g.KH == 0, (g.VH, g.KH)
        lay = self.lin_in_layout(g, ex)
        assert lay.vh <= 64, ("b|a gate lanes require vh <= 64", lay.vh)
        for dim_name, dim_val in (
            ("H", g.H),
            ("KD", lay.KD),
            ("VD", lay.VD),
            ("QKV", lay.QKV),
            ("Z", lay.Z),
        ):
            assert dim_val % 128 == 0, (
                f"GDN {dim_name} must be a multiple of 128",
                dim_val,
            )

        CR = rup(3 * B, 8)
        HR = rup(3 + T, 8)

        ngdn = plan.block_layer_counts.get(self.name, 0)
        GCH = ex.WIDE and tiles.gdn_rows > 0 and ngdn > 0
        CB = tiles.gdn_rows if GCH else B
        assert not GCH or CB % 8 == 0, (
            "gdn_rows must be a multiple of 8",
            tiles.gdn_rows,
        )
        assert not (GCH and ex.VER) or CB % T == 0, (
            "gdn_rows must be a multiple of T",
            CB,
            T,
        )
        NF, TR = divmod(B, CB)
        NCH = NF + int(TR > 0)
        NRB = min(tiles.gdn_bufs, NCH)
        assert not GCH or NCH == 1 or NRB >= 2, ("gdn_bufs >= 2", tiles.gdn_bufs)
        CBV = CB * lay.vh

        return GDNParams(
            kh=lay.kh,
            vh=lay.vh,
            KD=lay.KD,
            VD=lay.VD,
            QKV=lay.QKV,
            Z=lay.Z,
            BA=lay.BA,
            A_OFF=lay.A_OFF,
            BA_PAD=lay.BA_PAD,
            NLI=lay.NLI,
            H=g.H,
            CR=CR,
            HR=HR,
            GCH=GCH,
            CB=CB,
            NF=NF,
            TR=TR,
            NCH=NCH,
            NRB=NRB,
            CBV=CBV,
            NGDN=ngdn,
            eps=g.eps,
            zero_centered_norm=bool(g.zero_centered_norm),
            l2_eps=float(g.gdn_l2_eps),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        gp: GDNParams = plan.block_params[self.name]
        scr: dict[str, Any] = {
            "recb": (
                pltpu.VMEM((gp.NRB, gp.CBV, gp.KD, gp.VD), BF16)
                if gp.GCH
                else pltpu.VMEM((ex.B * gp.vh, gp.KD, gp.VD), BF16)
            ),
            "sbuf": pltpu.VMEM((gp.vh, gp.KD, gp.VD), F32),
            "convb": pltpu.VMEM(
                (ex.NS, gp.HR, gp.QKV) if ex.VER else (gp.CR, gp.QKV), F32
            ),
            "st_sem": pltpu.SemaphoreType.DMA((4,)),
        }
        if ex.VER and not gp.GCH:
            scr.update(
                {
                    "ub": pltpu.VMEM((ex.NS, rup(ex.T, 8), gp.QKV), F32),
                    "rin_sem": pltpu.SemaphoreType.DMA((ex.NS,)),
                }
            )
        if gp.GCH:
            scr["rec_sem"] = pltpu.SemaphoreType.DMA((2, gp.NRB))
        return scr

    def _rec_chunk(
        self, ctx: Any, gp: GDNParams, o: Any, j: Any, s: Any, out: bool
    ) -> Any:
        """Issue DMA for row chunk `j` of GDN layer `o` and ring buffer `s`."""
        ex = ctx.ex
        rec_out_key = "rec_ckpt" if ex.VER else "rec"
        n = gp.TR * gp.vh if gp.TR and isinstance(j, int) and j == gp.NF else gp.CBV
        r0 = j * gp.CBV if isinstance(j, int) else pl.multiple_of(j * gp.CBV, gp.CBV)
        buf = (
            ctx.scr["recb"].at[s] if n == gp.CBV else ctx.scr["recb"].at[s, pl.ds(0, n)]
        )
        if out:
            return pltpu.make_async_copy(
                buf,
                ctx.O[rec_out_key].at[o, pl.ds(r0, n)],
                ctx.scr["rec_sem"].at[1, s],
            )
        if ex.VER:
            ni = n // ex.T
            ri = (
                j * (gp.CBV // ex.T)
                if isinstance(j, int)
                else pl.multiple_of(j * (gp.CBV // ex.T), gp.CBV // ex.T)
            )
            return pltpu.make_async_copy(
                ctx.ST["rec"].at[o, pl.ds(ri, ni)],
                ctx.scr["recb"].at[s, pl.ds(n - ni, ni)],
                ctx.scr["rec_sem"].at[0, s],
            )
        return pltpu.make_async_copy(
            ctx.ST["rec"].at[o, pl.ds(r0, n)], buf, ctx.scr["rec_sem"].at[0, s]
        )

    def _rec_drain(self, ctx: Any, gp: GDNParams, o: Any) -> None:
        """Wait for layer `o`'s in-flight chunked `rec` write-backs."""
        for j_ in range(gp.NCH - gp.NRB, gp.NCH):
            self._rec_chunk(ctx, gp, o, j_, j_ % gp.NRB, True).wait()

    def _gdn_rec(self, ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
        """Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`)."""
        ex = ctx.ex
        scr = ctx.scr
        B, T, VER = ex.B, ex.T, ex.VER
        kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
        QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
        l2_eps = gp.l2_eps
        recb, sb = scr["recb"], scr["sbuf"]
        alog = ctx.sm(slot, "alog")[0, :vh]
        dtb = ctx.sm(slot, "dtb")[0, :vh]
        gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
        expand = vh // kh

        def cols(x):
            xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
            return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(
                vh, KD, VD
            )

        for b in range(B):
            src = b
            if VER:
                s_, t_ = divmod(b, T)
                urow_ = scr["ub"][s_, t_ : t_ + 1, :]
                q = urow_[0, : kh * KD].reshape(kh, KD)
                k = urow_[0, kh * KD : 2 * kh * KD].reshape(kh, KD)
                v = urow_[0, 2 * kh * KD :].reshape(vh, VD)
                src = b if t_ == 0 else b - 1
            else:
                q = u[b, : kh * KD].reshape(kh, KD)
                k = u[b, kh * KD : 2 * kh * KD].reshape(kh, KD)
                v = u[b, 2 * kh * KD :].reshape(vh, VD)
            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
            q = q * KD**-0.5
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
            qc, kc_ = cols(q), cols(k)
            beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
            a = ybuf[b, A_OFF : A_OFF + vh]
            decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            sb[...] = (
                recb[src * vh : (src + 1) * vh].astype(F32)
                * jnp.exp(decay)[:, None, None]
            )
            pred = jnp.sum(sb[...] * kc_, axis=1)
            delta = (v - pred) * beta[:, None]
            sb[...] = sb[...] + kc_ * delta[:, None, :]
            out = rnd(jnp.sum(sb[...] * qc, axis=1))
            recb[b * vh : (b + 1) * vh] = sb[...].astype(BF16)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            scr["mix"][b, :Z] = rnd(val * jax.nn.silu(zz)).reshape(Z)

    def _gdn_rec_chunked(self, ctx: Any, gp: GDNParams, o: Any, slot: Any) -> None:
        """Gated delta rule over `NRB`-buffered row chunks (`GCH=True`)."""
        ex = ctx.ex
        scr, skip, sc = ctx.scr, ctx.skip, ctx.sc
        T, VER = ex.T, ex.VER
        kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
        QKV, Z, BA, A_REL, BA_PAD = (
            gp.QKV,
            gp.Z,
            gp.BA,
            gp.A_OFF - gp.BA,
            gp.BA_PAD,
        )
        l2_eps = gp.l2_eps
        CB, NF, TR, NCH, NRB = gp.CB, gp.NF, gp.TR, gp.NCH, gp.NRB
        recb, sb, ybuf = scr["recb"], scr["sbuf"], scr["ybuf"]
        alog = ctx.sm(slot, "alog")[0, :vh]
        dtb = ctx.sm(slot, "dtb")[0, :vh]
        gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
        expand = vh // kh

        def cols(x):
            xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
            return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(
                vh, KD, VD
            )

        def rows(r0, n: int, s):
            n_pad = rup(n, 8)
            uc = ybuf[pl.ds(r0, n_pad), 0:QKV]
            bac = ybuf[pl.ds(r0, n_pad), BA : BA + BA_PAD]
            zc = ybuf[pl.ds(r0, n_pad), QKV : QKV + Z]
            st = recb.at[s]
            row_i = lax.broadcasted_iota(I32, (n_pad, 1), 0)
            blk = jnp.zeros((n_pad, Z), F32)
            for bb in range(n):
                h0 = bb * vh
                q = uc[bb, : kh * KD].reshape(kh, KD)
                k = uc[bb, kh * KD : 2 * kh * KD].reshape(kh, KD)
                v = uc[bb, 2 * kh * KD :].reshape(vh, VD)
                q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
                q = q * KD**-0.5
                k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
                qc, kc_ = cols(q), cols(k)
                beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
                a = bac[bb, A_REL : A_REL + vh]
                decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                src = (
                    ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh
                    if VER
                    else h0
                )
                sb[...] = st[src : src + vh].astype(F32) * jnp.exp(decay)[:, None, None]
                pred = jnp.sum(sb[...] * kc_, axis=1)
                delta = (v - pred) * beta[:, None]
                sb[...] = sb[...] + kc_ * delta[:, None, :]
                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                st[h0 : h0 + vh] = sb[...].astype(BF16)
                val = out * lax.rsqrt(
                    jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps
                )
                val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                zz = zc[bb].reshape(vh, VD)
                row = rnd(val * jax.nn.silu(zz)).reshape(1, Z)
                blk = jnp.where(row_i == bb, row, blk)
            scr["mix"][pl.ds(r0, n_pad), 0:Z] = blk

        def chunk(j, carry):
            s = lax.rem(j, NRB)
            if "gdn_state" not in skip:
                with sc("gdn_rec_wait"):
                    self._rec_chunk(ctx, gp, o, j, s, False).wait()
            if "gdn" not in skip:
                rows(pl.multiple_of(j * CB, CB), CB, s)
            if "gdn_state" not in skip:
                self._rec_chunk(ctx, gp, o, j, s, True).start()
                jp, jn = j - 1, j - 1 + NRB

                @pl.when(jnp.logical_and(j >= 1, jn < NCH))
                def _():
                    sp = lax.rem(jp, NRB)
                    self._rec_chunk(ctx, gp, o, jp, sp, True).wait()
                    if TR:

                        @pl.when(jn == NF)
                        def _():
                            self._rec_chunk(ctx, gp, o, NF, sp, False).start()

                        @pl.when(jn < NF)
                        def _():
                            self._rec_chunk(ctx, gp, o, jn, sp, False).start()
                    else:
                        self._rec_chunk(ctx, gp, o, jn, sp, False).start()

            return carry

        lax.fori_loop(0, NF, chunk, 0)
        if TR:
            s = NF % NRB
            if "gdn_state" not in skip:
                with sc("gdn_rec_wait"):
                    self._rec_chunk(ctx, gp, o, NF, s, False).wait()
            if "gdn" not in skip:
                rows(NF * CB, TR, s)
            if "gdn_state" not in skip:
                self._rec_chunk(ctx, gp, o, NF, s, True).start()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        gp: GDNParams = ctx.block_params(self.name)
        scr, ST, O = ctx.scr, ctx.ST, ctx.O  # noqa: E741
        skip, sc, stream = ctx.skip, ctx.sc, ctx.stream
        B, T, NS, VER = ex.B, ex.T, ex.NS, ex.VER
        vh, QKV, Z, NLI, H = gp.vh, gp.QKV, gp.Z, gp.NLI, gp.H
        GCH, NRB = gp.GCH, gp.NRB
        conv_out_key = "conv_ckpt" if VER else "conv"
        rec_out_key = "rec_ckpt" if VER else "rec"

        o = stream.sp.block_ord(l, self.name)
        convb, recb, sem = scr["convb"], scr["recb"], scr["st_sem"]

        def rec_in():
            if not VER:
                return [pltpu.make_async_copy(ST["rec"].at[o], recb, sem.at[1])]
            return [
                pltpu.make_async_copy(
                    ST["rec"].at[o, pl.ds(s * vh, vh)],
                    recb.at[pl.ds(s * T * vh, vh)],
                    scr["rin_sem"].at[s],
                )
                for s in range(NS)
            ]

        with stream.segment(lazy=False):
            with sc("gdn_in"):
                if "gdn_state" not in skip:

                    @pl.when(o > 0)
                    def _():
                        pltpu.make_async_copy(
                            convb, O[conv_out_key].at[0], sem.at[2]
                        ).wait()
                        if GCH:
                            self._rec_drain(ctx, gp, 0)
                        else:
                            pltpu.make_async_copy(
                                recb, O[rec_out_key].at[0], sem.at[3]
                            ).wait()

                    pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).start()
                    if GCH:
                        for s_ in range(NRB):
                            self._rec_chunk(ctx, gp, o, s_, s_, False).start()
                    else:
                        for d in rec_in():
                            d.start()

                y = stream.dmatmul("lin_in", scr["xb"], NLI)
                ybuf = scr["ybuf"]
                ybuf[:, :NLI] = rnd(y * ctx.sm(slot, "in_s", NLI))

            with sc("gdn_conv"):
                if "gdn_state" not in skip:
                    pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).wait()
                ow, _ = stream.sp.lay["conv"]
                taps = [
                    scr["sm"][slot, :, ow + t * QKV : ow + (t + 1) * QKV]
                    for t in range(4)
                ]
                u = None
                if VER:
                    us = []
                    for s in range(NS):
                        convb[s, 3 : 3 + T, :] = ybuf[s * T : (s + 1) * T, 0:QKV]
                        w = [convb[s, k : k + T, :] for k in range(4)]
                        u_ = rnd(
                            jax.nn.silu(
                                w[0] * taps[0]
                                + w[1] * taps[1]
                                + w[2] * taps[2]
                                + w[3] * taps[3]
                            )
                        )
                        if GCH:
                            us.append(u_)
                        else:
                            scr["ub"][s, 0:T, :] = u_
                    if GCH:
                        u = jnp.concatenate(us, axis=0)
                else:
                    x = ybuf[0:B, 0:QKV]
                    s0 = convb[0:B, :]
                    s1 = convb[B : 2 * B, :]
                    s2 = convb[2 * B : 3 * B, :]
                    u = rnd(
                        jax.nn.silu(
                            s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x * taps[3]
                        )
                    )
                    convb[0:B, :] = s1
                    convb[B : 2 * B, :] = s2
                    convb[2 * B : 3 * B, :] = x
                if "gdn_state" not in skip:
                    pltpu.make_async_copy(
                        convb, O[conv_out_key].at[o], sem.at[2]
                    ).start()
                if GCH:
                    u_pad = jnp.pad(u, ((0, ex.MR - B), (0, 0))) if B < ex.MR else u
                    ybuf[0 : ex.MR, 0:QKV] = u_pad

            if GCH:
                with sc("gdn_rec"):
                    self._gdn_rec_chunked(ctx, gp, o, slot)
            else:
                if "gdn_state" not in skip:
                    with sc("gdn_rec_wait"):
                        for d in rec_in():
                            d.wait()
                if "gdn" not in skip:
                    with sc("gdn_rec"):
                        self._gdn_rec(ctx, gp, u, ybuf, slot)

            with sc("gdn_out"):
                if not GCH and "gdn_state" not in skip:
                    pltpu.make_async_copy(recb, O[rec_out_key].at[o], sem.at[3]).start()
                scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                yo = stream.dmatmul("lin_out", scr["xb2"], H)
                return yo * ctx.sm(slot, "out_s")

    def epilogue(self, ctx: Any) -> None:
        gp: GDNParams = ctx.block_params(self.name)
        if gp.NGDN > 0 and "gdn_state" not in ctx.skip:
            conv_out_key = "conv_ckpt" if ctx.ex.VER else "conv"
            rec_out_key = "rec_ckpt" if ctx.ex.VER else "rec"
            o_last = gp.NGDN - 1
            pltpu.make_async_copy(
                ctx.scr["convb"],
                ctx.O[conv_out_key].at[o_last],
                ctx.scr["st_sem"].at[2],
            ).wait()
            if gp.GCH:
                self._rec_drain(ctx, gp, o_last)
            else:
                pltpu.make_async_copy(
                    ctx.scr["recb"],
                    ctx.O[rec_out_key].at[o_last],
                    ctx.scr["st_sem"].at[3],
                ).wait()

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        if ex.VER:
            return {
                "conv_ckpt": raw_outputs["conv_ckpt"],
                "rec_ckpt": raw_outputs["rec_ckpt"],
            }
        return {"conv": raw_outputs["conv"], "rec": raw_outputs["rec"]}
