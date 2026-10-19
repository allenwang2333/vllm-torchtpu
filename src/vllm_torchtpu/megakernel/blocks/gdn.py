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
    U32,
    bswitch,
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
from . import fused_pool_io


def _kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)


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
    NGI: int = 0
    NGO: int = 0
    G_PER_ATT: int = 0
    REC_PAGES: int = 4
    GDN_PAGES: int = 5
    SSM_U32_CAP: int = 512
    ssm_u32_rows: int = 0
    qkv_128: int = 0
    A_conv: int = 16
    conv8_rows: int = 0
    ckpt_is_3d: bool = True
    roff_is_2d: bool = True
    K_ckpt: int = 1
    ckpt_t_off: int = 0
    GPR: bool = False
    GCONV_ROWS: int = 0
    conv_stage: bool = False
    head_major: bool = False


class _GprRecIn:
    """In-kernel checkpoint-to-`recb` gather handle for `GPR=True`."""

    def __init__(
        self,
        blk: GDNBlock,
        ctx: Any,
        gp: GDNParams,
        o: Any,
        j: Any,
        s: Any,
        n: int,
    ) -> None:
        self.blk = blk
        self.ctx = ctx
        self.gp = gp
        self.o = o
        self.s = s
        self.n = n
        self.nseq = n // (ctx.ex.T * gp.vh)
        self.seqs = [j * (gp.CB // ctx.ex.T) + q for q in range(self.nseq)]
        self.dsts = [ctx.scr["gpr_rec"].at[s, q] for q in range(self.nseq)]

    def start(self) -> None:
        self.blk._gpr_start(
            self.ctx,
            self.gp,
            self.o,
            self.seqs,
            self.dsts,
            self.ctx.scr["rec_sem"].at[0, self.s],
            0,
            self.gp.REC_PAGES,
        )

    def wait(self) -> None:
        ctx, gp = self.ctx, self.gp
        scr = ctx.scr
        T = ctx.ex.T
        vh, KD, VD = gp.vh, gp.KD, gp.VD
        SSM_U32_CAP, ssm_u32_rows = gp.SSM_U32_CAP, gp.ssm_u32_rows
        s, n = self.s, self.n
        for sq, dst in zip(self.seqs, self.dsts):
            self.blk._gpr_wait(
                ctx, gp, self.o, sq, dst, scr["rec_sem"].at[0, s], gp.REC_PAGES
            )
        for q, sq in enumerate(self.seqs):
            if ctx.ex.interpret:
                u32 = pltpu.bitcast(scr["gpr_rec"][s, q], U32).reshape(
                    SSM_U32_CAP, 128
                )[:ssm_u32_rows, :]
            else:
                u32 = (
                    scr["gpr_rec"]
                    .at[s, q]
                    .bitcast(U32)
                    .reshape(SSM_U32_CAP, 128)[:ssm_u32_rows, :]
                )
            u32 = jnp.where(
                self.blk._has_gpr_input(ctx, gp, self.o, sq), u32, jnp.uint32(0)
            )
            scr["recb"][s, pl.ds(n - n // T + q * vh, vh)] = pltpu.bitcast(
                u32, BF16
            ).reshape(vh, KD, VD)


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
        if ex.FUSED or ex.GPR:
            g_per_att = g.att_every - 1
            k_ckpt = ex.ckpt_k if ex.ckpt_k > 0 else ex.T
            ckpt_sh = (ex.NS, k_ckpt) if ex.ckpt_2d else (g_per_att, ex.NS, k_ckpt)
            roff_sh = (ex.NS,) if ex.roff_1d else (g_per_att, ex.NS)
            st = [
                StateDecl(
                    "ckpt_indices",
                    shape=ckpt_sh,
                    dtype=I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
                StateDecl(
                    "read_offsets",
                    shape=roff_sh,
                    dtype=I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
            ]
            if ex.act_mode == "ref":
                st.append(
                    StateDecl(
                        "active_seqs",
                        shape=(ex.NS,),
                        dtype=I32,
                        access="r",
                        per_layer=False,
                        memory_space="smem",
                    )
                )
            return st
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
        if ex.FUSED:
            return []
        lay = self.lin_in_layout(g, ex)
        if ex.VER:
            return [
                StateDecl(
                    "conv_ckpt",
                    shape=(ex.NS, rup(3 + ex.T, 8), lay.QKV),
                    dtype=BF16 if ex.GPR else F32,
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
        tiles = ex.resolve_tiles(g)
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
        GCH = (
            (ex.WIDE or bool(ex.GPR))
            and tiles.gdn_rows > 0
            and ngdn > 0
            and not ex.FUSED
        )
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

        GPR = bool(ex.GPR and GCH and ex.VER)
        assert not ex.GPR or GPR, (
            "pool_gather needs the paged chunked verify path",
            B,
            T,
        )
        GCONV_ROWS = (HR * lay.QKV * 2) // (4 * 128)
        assert not GPR or ex.page_size == 128, ex.page_size
        assert not GPR or (HR * lay.QKV * 2) % (4 * 128) == 0, (HR, lay.QKV)
        assert not GPR or GCONV_ROWS <= 64, GCONV_ROWS

        g_per_att = max(0, getattr(g, "att_every", 1) - 1)
        ssm_u32_rows = (lay.vh * lay.KD) // 2
        REC_PAGES = max(4, (ssm_u32_rows + 127) // 128)
        GDN_PAGES = REC_PAGES + 1
        SSM_U32_CAP = REC_PAGES * 128
        qkv_128 = lay.QKV // 128
        A_conv = max(16, rup(3 + T, 8))
        conv8_rows = A_conv * qkv_128
        NGI = min(ex.NS, 8) if ex.FUSED else 0
        NGO = min(ex.NS, 2) if ex.FUSED else 0
        K_ckpt = ex.ckpt_k if ex.ckpt_k > 0 else T
        ckpt_t_off = K_ckpt - T if K_ckpt > T else 0
        if ex.FUSED or GPR:
            assert ex.page_size == 128, (
                f"pool I/O requires page_size == 128, got {ex.page_size}"
            )
            assert g_per_att > 0, (
                f"pool I/O on GDNBlock requires att_every > 1, got {g.att_every}"
            )

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
            NGI=NGI,
            NGO=NGO,
            G_PER_ATT=g_per_att,
            REC_PAGES=REC_PAGES,
            GDN_PAGES=GDN_PAGES,
            SSM_U32_CAP=SSM_U32_CAP,
            ssm_u32_rows=ssm_u32_rows,
            qkv_128=qkv_128,
            A_conv=A_conv,
            conv8_rows=conv8_rows,
            ckpt_is_3d=not ex.ckpt_2d,
            roff_is_2d=not ex.roff_1d,
            K_ckpt=K_ckpt,
            ckpt_t_off=ckpt_t_off,
            GPR=GPR,
            GCONV_ROWS=GCONV_ROWS,
            conv_stage=bool(tiles.conv_stage),
            head_major=bool(tiles.gdn_head_major),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        gp: GDNParams = plan.block_params[self.name]
        if ex.FUSED:
            return fused_pool_io.gdn_fused_scratch(ex, gp, plan)
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
            if gp.GPR:
                scr["gpr_rec"] = pltpu.VMEM(
                    (gp.NRB, gp.CB // ex.T, gp.REC_PAGES, 2, 64, 4, ex.page_size),
                    FP8,
                )
                scr["gpr_conv"] = pltpu.VMEM(
                    (ex.NS, gp.GCONV_ROWS, 4, ex.page_size), FP8
                )
                scr["convb16"] = pltpu.VMEM((ex.NS, gp.HR, gp.QKV), BF16)
        return scr

    def init_buffers(self, ctx: Any) -> None:
        if not ctx.ex.FUSED:
            return
        gp: GDNParams = ctx.block_params(self.name)
        fused_pool_io.gdn_fused_init_buffers(ctx, gp)

    def _is_seq_active(self, ctx: Any, gp: GDNParams, s: Any) -> Any:
        ex, ST = ctx.ex, ctx.ST
        if ex.act_mode == "all":
            return jnp.bool_(True)
        if ex.act_mode == "ref":
            return ST["active_seqs"][s] != 0
        return (
            (ST["ckpt_indices"][0, s, 0] > 0)
            if gp.ckpt_is_3d
            else (ST["ckpt_indices"][s, 0] > 0)
        )

    def _gdn_roff(self, ctx: Any, gp: GDNParams, sub: Any, s: Any) -> Any:
        ST = ctx.ST
        return ST["read_offsets"][sub, s] if gp.roff_is_2d else ST["read_offsets"][s]

    def _has_gpr_input(self, ctx: Any, gp: GDNParams, o: Any, s: Any) -> Any:
        sub = lax.rem(o, gp.G_PER_ATT)
        return jnp.logical_and(
            self._is_seq_active(ctx, gp, s),
            self._gdn_roff(ctx, gp, sub, s) >= 0,
        )

    def _gdn_pb_in(self, ctx: Any, gp: GDNParams, sub: Any, s: Any) -> Any:
        ex, ST = ctx.ex, ctx.ST
        GDN_PAGES, G_PER_ATT = gp.GDN_PAGES, gp.G_PER_ATT
        roff = jnp.maximum(self._gdn_roff(ctx, gp, sub, s), jnp.int32(0))
        if gp.ckpt_is_3d:
            pb = ST["ckpt_indices"][sub, s, roff] * GDN_PAGES
        else:
            pb = ST["ckpt_indices"][s, roff] * (GDN_PAGES * G_PER_ATT) + sub * GDN_PAGES
        _kernel_debug_check(
            jnp.logical_and(pb >= 0, pb + GDN_PAGES <= ex.num_pages),
            "GDN gather checkpoint slot maps outside [0, num_pages)",
            interpret=bool(ex.interpret),
        )
        return jnp.clip(pb, 0, ex.num_pages - GDN_PAGES)

    def _gpr_start(
        self,
        ctx: Any,
        gp: GDNParams,
        o: Any,
        seqs: list[Any],
        dsts: list[Any],
        sem: Any,
        first: int,
        npages: int,
    ) -> None:
        raw_refs = ctx.raw_refs
        sub = lax.rem(o, gp.G_PER_ATT)
        g_pool = lax.div(o, gp.G_PER_ATT)

        def issue(r):
            for sq, dst in zip(seqs, dsts):

                @pl.when(self._has_gpr_input(ctx, gp, o, sq))
                def _(sq=sq, dst=dst):
                    pb = self._gdn_pb_in(ctx, gp, sub, sq) + first
                    pltpu.make_async_copy(r.at[pl.ds(pb, npages)], dst, sem).start()

        bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])

    def _gpr_wait(
        self,
        ctx: Any,
        gp: GDNParams,
        o: Any,
        sq: Any,
        dst: Any,
        sem: Any,
        npages: int,
    ) -> None:
        raw_refs = ctx.raw_refs

        @pl.when(self._has_gpr_input(ctx, gp, o, sq))
        def _():
            pltpu.make_async_copy(raw_refs[0].at[pl.ds(0, npages)], dst, sem).wait()

    def _gpr_conv_copy(
        self, ctx: Any, gp: GDNParams, r: Any, sq: int, pb: Any, sem: Any
    ) -> Any:
        return pltpu.make_async_copy(
            r.at[pb, 0, pl.ds(0, gp.GCONV_ROWS)], ctx.scr["gpr_conv"].at[sq], sem
        )

    def _gpr_conv_start(self, ctx: Any, gp: GDNParams, o: Any) -> None:
        raw_refs = ctx.raw_refs
        sub = lax.rem(o, gp.G_PER_ATT)
        g_pool = lax.div(o, gp.G_PER_ATT)

        def issue(r):
            for sq in range(ctx.ex.NS):

                @pl.when(self._has_gpr_input(ctx, gp, o, sq))
                def _(sq=sq):
                    self._gpr_conv_copy(
                        ctx,
                        gp,
                        r,
                        sq,
                        self._gdn_pb_in(ctx, gp, sub, sq) + gp.REC_PAGES,
                        ctx.scr["st_sem"].at[0],
                    ).start()

        bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])

    def _conv_out_src(self, ctx: Any, gp: GDNParams) -> Any:
        return ctx.scr["convb16"] if gp.GPR else ctx.scr["convb"]

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
        if gp.GPR:
            return _GprRecIn(self, ctx, gp, o, j, s, n)
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

        def colsk(x):
            return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

        def cols(x):
            return jnp.broadcast_to(colsk(x)[:, None], (kh, expand, KD, VD)).reshape(
                vh, KD, VD
            )

        HEAD_MAJOR = gp.head_major
        assert not HEAD_MAJOR or 2 * kh <= vh, (kh, vh)
        NCB = 2 if 4 * kh <= vh else 1
        hrow = lax.broadcasted_iota(I32, (vh, VD), 0)

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
                beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
                a = bac[bb, A_REL : A_REL + vh]
                decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                src = (
                    ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh
                    if VER
                    else h0
                )
                if not HEAD_MAJOR:
                    # Frozen-parity recurrence: whole-state update staged in `sb`.
                    qc, kc_ = cols(q), cols(k)
                    sb[...] = (
                        st[src : src + vh].astype(F32) * jnp.exp(decay)[:, None, None]
                    )
                    pred = jnp.sum(sb[...] * kc_, axis=1)
                    delta = (v - pred) * beta[:, None]
                    sb[...] = sb[...] + kc_ * delta[:, None, :]
                    out = rnd(jnp.sum(sb[...] * qc, axis=1))
                    st[h0 : h0 + vh] = sb[...].astype(BF16)
                else:
                    # Head-major recurrence (opt-in `Tiles.gdn_head_major`):
                    # per-head update with `colsk` K/Q double-buffered in `sb`.
                    cb = (bb % NCB) * 2 * kh
                    sb[cb : cb + kh] = colsk(k)
                    sb[cb + kh : cb + 2 * kh] = colsk(q)
                    eg = jnp.exp(decay)[:, None, None]
                    beta2 = beta[:, None]
                    out_acc = jnp.zeros((vh, VD), F32)
                    for h in range(vh):
                        j = h // expand
                        kc_h = sb[cb + j : cb + j + 1]
                        s_h = st[src + h : src + h + 1].astype(F32) * eg[h : h + 1]
                        pred_h = jnp.sum(s_h * kc_h, axis=1)
                        delta_h = (v[h : h + 1] - pred_h) * beta2[h : h + 1]
                        s_h = s_h + kc_h * delta_h[:, None, :]
                        qc_h = sb[cb + kh + j : cb + kh + j + 1]
                        out_acc = jnp.where(
                            hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc
                        )
                        st[h0 + h : h0 + h + 1] = s_h.astype(BF16)
                    out = rnd(out_acc)
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
        o = ctx.stream.sp.block_ord(l, self.name)
        if ex.FUSED:
            return fused_pool_io.gdn_fused_emit(ctx, gp, o, slot)
        scr, ST, O = ctx.scr, ctx.ST, ctx.O  # noqa: E741
        raw_refs = ctx.raw_refs
        skip, sc, stream = ctx.skip, ctx.sc, ctx.stream
        B, T, NS, VER = ex.B, ex.T, ex.NS, ex.VER
        vh, QKV, Z, NLI, H = gp.vh, gp.QKV, gp.Z, gp.NLI, gp.H
        GCH, NRB, GPR = gp.GCH, gp.NRB, gp.GPR
        conv_out_key = "conv_ckpt" if VER else "conv"
        rec_out_key = "rec_ckpt" if VER else "rec"

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
                    if GPR:
                        self._gpr_conv_start(ctx, gp, o)
                        for s_ in range(NRB):
                            self._rec_chunk(ctx, gp, o, s_, s_, False).start()

                    @pl.when(o > 0)
                    def _():
                        pltpu.make_async_copy(
                            self._conv_out_src(ctx, gp),
                            O[conv_out_key].at[0],
                            sem.at[2],
                        ).wait()
                        if GCH:
                            self._rec_drain(ctx, gp, 0)
                        else:
                            pltpu.make_async_copy(
                                recb, O[rec_out_key].at[0], sem.at[3]
                            ).wait()

                    if not GPR:
                        pltpu.make_async_copy(
                            ST["conv"].at[o], convb, sem.at[0]
                        ).start()
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
                    if GPR:
                        hr_rows_128 = gp.HR * gp.qkv_128
                        conv_row_iota = lax.broadcasted_iota(I32, (gp.HR, QKV), 0)
                        for s_ in range(NS):

                            @pl.when(self._has_gpr_input(ctx, gp, o, s_))
                            def _(s_=s_):
                                self._gpr_conv_copy(
                                    ctx, gp, raw_refs[0], s_, 0, sem.at[0]
                                ).wait()

                        for s_ in range(NS):
                            if ex.interpret:
                                raw_hr = pltpu.bitcast(
                                    scr["gpr_conv"][s_], BF16
                                ).reshape(2 * gp.GCONV_ROWS, 128)
                            else:
                                raw_hr = (
                                    scr["gpr_conv"]
                                    .at[s_]
                                    .bitcast(BF16)
                                    .reshape(2 * gp.GCONV_ROWS, 128)
                                )
                            raw_hr = (
                                raw_hr[:hr_rows_128, :].reshape(gp.HR, QKV).astype(F32)
                            )
                            keep = jnp.logical_and(
                                self._has_gpr_input(ctx, gp, o, s_), conv_row_iota < 3
                            )
                            convb[s_] = jnp.where(keep, raw_hr, 0.0)
                    else:
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
                if GPR:
                    scr["convb16"][...] = convb[...].astype(BF16)
                if "gdn_state" not in skip:
                    pltpu.make_async_copy(
                        self._conv_out_src(ctx, gp),
                        O[conv_out_key].at[o],
                        sem.at[2],
                    ).start()
                if GCH:
                    b_pad = rup(B, 8)
                    u_pad = jnp.pad(u, ((0, b_pad - B), (0, 0))) if b_pad > B else u
                    ybuf[0:b_pad, 0:QKV] = u_pad

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
        if ctx.ex.FUSED:
            fused_pool_io.gdn_fused_epilogue(ctx, gp)
            return
        if gp.NGDN > 0 and "gdn_state" not in ctx.skip:
            conv_out_key = "conv_ckpt" if ctx.ex.VER else "conv"
            rec_out_key = "rec_ckpt" if ctx.ex.VER else "rec"
            o_last = gp.NGDN - 1
            pltpu.make_async_copy(
                self._conv_out_src(ctx, gp),
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
        if ex.FUSED:
            return {}
        if ex.VER:
            return {
                "conv_ckpt": raw_outputs["conv_ckpt"],
                "rec_ckpt": raw_outputs["rec_ckpt"],
            }
        return {"conv": raw_outputs["conv"], "rec": raw_outputs["rec"]}
