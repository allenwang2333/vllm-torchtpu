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

from .. import pool_alias
from ..errors import MegakernelConfigError
from ..fused_io_hook import fused_io
from ..ops import (
    BF16,
    F32,
    FP8,
    I32,
    U32,
    bswitch,
    rmsnorm,
    rnd,
    rope_partial,
    rup,
)
from ..spec import (
    POOL_STATE_NAME,
    Block,
    ExecSpec,
    SmallField,
    StateDecl,
    WeightDecl,
)

NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))

# Geometry (layout) requirements, not per-bucket tuning:
# - KVE views each KV ring stage as `(2, HD // 4, BLK)` u32 rows of the untyped
#   expert pool; that view is validated for HD == 256 only (the frozen /
#   candidate30 layout). Other head dims keep the separate `kvbuf` ring.
# - GQ sends the partial-softmax `(m | l)` pair as two 128-lane halves of one
#   HD-wide row of the exchange buffer, so it needs HD == 2 * 128.
KVE_HD = 256
GQ_ML_LANES = 2 * 128


@dataclass(frozen=True)
class GQAParams:
    """Block-owned build-time parameters for `GQABlock` (`direct_hbm` & `paged_kv`)."""

    H: int
    nh: int
    HD: int
    ROT: int
    NAI: int
    G: int
    BLK: int
    NKV: int
    NKVR: int
    KVE: bool
    KV_ITEM: int
    EBROWS: int
    PAIR_KV: bool
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
    QV: bool
    GQ: bool
    NHH: int
    SR: int
    QG: int
    MLW: int
    QVR: int
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

    def __init__(self, pool_layout: Any = None) -> None:
        # `PoolLayout` of the model's paged pools (None: the default page
        # packing `(2, HD // 4, 4, page_size)`, the only one the paged reads use).
        self.pool_layout = pool_layout

    def kv_page_shape(self, g: Any, ex: ExecSpec) -> tuple[int, ...]:
        """Shape of one pool page, from `pool_layout` when the model sets one.

        Raises `MegakernelConfigError` if the layout's page packing differs from
        the `(2, HD // 4, 4, page_size)` packing the paged KV reads assume.
        """
        shape = (2, g.HD // 4, 4, ex.page_size)
        if self.pool_layout is None:
            return shape
        lay_shape = tuple(self.pool_layout.kv_page_shape)
        if lay_shape != shape:
            raise MegakernelConfigError(
                f"GQABlock paged KV reads need pool pages {shape}, but the pool "
                f"layout has {lay_shape}"
            )
        return lay_shape

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
        if ex.paged_kv:
            st = [
                StateDecl(
                    POOL_STATE_NAME,
                    (ex.num_pages, *self.kv_page_shape(g, ex)),
                    FP8,
                    access="r",
                    per_layer=True,
                    memory_space="hbm",
                    tupled=True,
                ),
                StateDecl(
                    "page_table",
                    (ex.NS, ex.resolved_max_pages_per_seq),
                    I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
            ]
            if ex.FUSED:
                st.extend(fused_io().gqa_fused_states(ex))
            return st
        return [
            StateDecl("kc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
            StateDecl("vc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if ex.FUSED:
            return fused_io().gqa_fused_outputs(g, ex)
        kv_dt = FP8 if ex.paged_kv else F32
        return [
            StateDecl("knew", (ex.MR, g.HD), kv_dt, access="w"),
            StateDecl("vnew", (ex.MR, g.HD), kv_dt, access="w"),
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
        tiles = ex.resolve_tiles(g)
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        G = max(1, ex.R // g.NKV)
        BLK = tiles.kv_blk
        PAIR_KV = tiles.pair_kv
        NKV = max(tiles.nkv, 4) if PAIR_KV else tiles.nkv
        assert ex.ctx % BLK == 0 and BLK % 128 == 0, (ex.ctx, BLK)
        if ex.paged_kv:
            assert g.HD % 4 == 0, f"paged_kv requires HD % 4 == 0, got {g.HD}"
            assert (
                ex.page_size > 0 and ex.page_size % 128 == 0 and BLK % ex.page_size == 0
            ), (
                BLK,
                ex.page_size,
            )
            assert (
                ex.ctx % ex.page_size == 0
                and ex.num_pages >= 1
                and ex.resolved_max_pages_per_seq >= ex.ctx // ex.page_size
            ), (
                ex.page_size,
                ex.ctx,
                ex.num_pages,
                ex.resolved_max_pages_per_seq,
            )
            assert tiles.amask not in (3, 4), (
                f"paged_kv requires amask in {{0, 1, 2}}, got {tiles.amask}"
            )
        KV_ITEM = 2 * (g.HD // 4) * BLK * 4
        EBROWS = tiles.ebank // 128
        # KVE stages KV blocks in the (idle) expert ring. On TPU this aliases the
        # untyped byte pool `epool` via `pool_alias.view`; in CPU interpret mode
        # `vmem.build_scratch` allocates a separate typed `epool`, so the
        # aliasing itself is only exercised on TPU (the KV math is shared).
        KVE = bool(
            ex.paged_kv
            and tiles.kv_ering
            and bool(plan.dynamic_kinds)
            and tiles.nbe >= NKV
            and tiles.ebank >= KV_ITEM
            and g.HD == KVE_HD
        )
        NKVR = tiles.nbe if KVE else NKV
        SUB = tiles.kv_sub or BLK
        assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
        SPLIT = tiles.split and G > 1
        assert not SPLIT or (G & (G - 1) == 0), G
        TS = G if SPLIT else 1
        QR = rup(G * nh, 8) if SPLIT else 8
        TQ = ex.T * QR
        PW = g.HD + 128
        eff_ctx = (
            max(ex.ctx, ex.resolved_max_pages_per_seq * ex.page_size)
            if ex.paged_kv
            else ex.ctx
        )
        max_nb = -(-eff_ctx // BLK)
        NWL = rup(ex.NS * -(-max_nb // TS), 8)
        PACK = (ex.WIDE if tiles.pack < 0 else bool(tiles.pack)) and SPLIT and nh < 8
        PB = 8 // nh if PACK else 1
        NBP = -(-ex.B // PB)
        NHH = g.HD // 128
        SR = 8 // PB
        QV = (
            tiles.gqa_qv
            and (SPLIT or ex.VER)
            and g.HD % 128 == 0
            and nh <= 8
            and (not PACK or PB * nh == 8)
        )
        if tiles.gq_pack and not tiles.gqa_qv:
            raise MegakernelConfigError("Tiles.gq_pack requires gqa_qv")
        QG = ex.T * nh
        GQ = bool(
            tiles.gq_pack
            and QV
            and SPLIT
            and ex.VER
            and G * nh == QR
            and QG % 16 == 0
            and ex.NS <= 64
            and g.HD == GQ_ML_LANES
        )
        MLW = 128 // ex.NS if GQ else 0
        if GQ:
            SR = nh
        QVR = ex.B * nh + 8 if GQ else NBP * 8
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
            NKVR=NKVR,
            KVE=KVE,
            KV_ITEM=KV_ITEM,
            EBROWS=EBROWS,
            PAIR_KV=PAIR_KV,
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
            QV=QV,
            GQ=GQ,
            NHH=NHH,
            SR=SR,
            QG=QG,
            MLW=MLW,
            QVR=QVR,
            LAZY_ATT=LAZY_ATT,
            eps=g.eps,
            theta=g.theta,
            zero_centered_norm=bool(g.zero_centered_norm),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        ap: GQAParams = plan.block_params[self.name]
        scr: dict[str, Any] = {}
        if not ex.paged_kv:
            scr["kbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
            scr["vbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
        scr.update(
            {
                "kv_sem": pltpu.SemaphoreType.DMA(
                    (ap.NKVR,) if ex.paged_kv else (2, ap.NKV)
                ),
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
                "rcs": pltpu.VMEM((2, ex.MR, ap.ROT // 2), F32),
            }
        )
        if not ex.FUSED:
            scr["kv_out_sem"] = pltpu.SemaphoreType.DMA((2,))
        else:
            scr.update(fused_io().gqa_fused_scratch(ex, ap.HD))
        if ex.paged_kv:
            if not ap.KVE:
                scr["kvbuf"] = pltpu.VMEM((ap.NKV, 2, ap.HD // 4, 4, ap.BLK), FP8)
            scr["knew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
            scr["vnew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
            scr["wl_kp"] = pltpu.SMEM((ap.NWL * (ap.BLK // ex.page_size),), I32)
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
        if ap.QV:
            scr["rope"] = pltpu.VMEM((2, ex.MR, 128), F32)  # QV-only cos / sin table
            scr["qg"] = pltpu.VMEM(
                (ap.G if ap.SPLIT and not ap.GQ else 1, ap.NHH, ap.QVR, 128), F32
            )
            scr["ost"] = pltpu.VMEM((ap.NHH, ap.QVR, 128), F32)
        if ap.GQ:
            del scr["pso"]
            scr.update(
                {
                    "qxb": pltpu.VMEM((ap.G, ex.NS, ap.QG, ap.HD), BF16),
                    "psend": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
                    "pri": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
                }
            )
        return scr

    def prologue(self, ctx: Any) -> None:
        """Build this rank's KV worklist (`wl_b`, `wl_t`, `wl_kp`, `wl_cnt[0]`)."""
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, ST, pos_ref = ctx.scr, ctx.ST, ctx.pos_ref
        NS, T, BLK, TS, SPLIT = ex.NS, ex.T, ap.BLK, ap.TS, ap.SPLIT
        gi = lax.rem(ctx.me, ap.G) if SPLIT else jnp.int32(0)
        eff_ctx = (
            max(ex.ctx, ex.resolved_max_pages_per_seq * ex.page_size)
            if ex.paged_kv
            else ex.ctx
        )
        max_nb = -(-eff_ctx // BLK)
        max_p = ex.resolved_max_pages_per_seq

        n = jnp.int32(0)
        for b in range(NS):
            nb = jnp.minimum(jnp.maximum((pos_ref[b * T] + BLK - 1) // BLK, 0), max_nb)
            t0 = lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0)
            cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

            def put(j, carry, b=b, t0=t0, n=n):
                scr["wl_b"][n + j] = jnp.int32(b)
                scr["wl_t"][n + j] = t0 + j * TS
                if ex.paged_kv:
                    pt_ref_ = ST["page_table"]
                    pb = BLK // ex.page_size
                    p0 = (t0 + j * TS) * pb
                    k0 = (n + j) * pb
                    for p in range(pb):
                        col = jnp.minimum(p0 + p, max_p - 1)
                        scr["wl_kp"][k0 + p] = jnp.clip(
                            pt_ref_[b, col], 0, ex.num_pages - 1
                        )
                return carry

            lax.fori_loop(0, cnt, put, 0)
            n = n + cnt
        scr["wl_cnt"][0] = n

    def init_buffers(self, ctx: Any) -> None:
        """Initialize attention-specific VMEM scratch buffers after barrier."""
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, pos_ref = ctx.scr, ctx.pos_ref
        scr["knew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["vnew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        if ex.paged_kv:
            scr["knew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
        if ex.FUSED:
            fused_io().gqa_fused_init_buffers(ctx)
        if ap.SPLIT and not ap.GQ:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if ap.PACK or ap.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        if ap.QV:
            half_ = ap.ROT // 2
            inv_ = 1.0 / (
                ap.theta
                ** (lax.broadcasted_iota(I32, (1, half_), 1).astype(F32) * 2.0 / ap.ROT)
            )
            rows_ = lax.broadcasted_iota(I32, (ex.MR, half_), 0)
            pf = jnp.zeros((ex.MR, half_), F32)
            for b in range(ex.B):
                pf = jnp.where(rows_ == b, pos_ref[b].astype(F32), pf)
            ang_ = pf * inv_
            scr["rope"][0, :, :half_] = rnd(jnp.cos(ang_))
            scr["rope"][1, :, :half_] = rnd(jnp.sin(ang_))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, ST, O = ctx.scr, ctx.ST, ctx.O  # noqa: E741
        pos_ref, skip, sc = ctx.pos_ref, ctx.skip, ctx.sc
        stream, comm = ctx.stream, ctx.comm
        NKV, NKVR, KVE, PAIR_KV = ap.NKV, ap.NKVR, ap.KVE, ap.PAIR_KV
        KV_ITEM, EBROWS = ap.KV_ITEM, ap.EBROWS
        BLK, SUB, AMASK = ap.BLK, ap.SUB, ap.AMASK
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
        QV, GQ, NHH, SR, QG, MLW = (
            ap.QV,
            ap.GQ,
            ap.NHH,
            ap.SR,
            ap.QG,
            ap.MLW,
        )
        interpret = bool(ex.interpret)
        gi = lax.rem(ctx.me, G) if SPLIT else jnp.int32(0)

        a = stream.sp.block_ord(l, self.name)
        ksem = scr["kv_sem"]
        am, al, aacc = scr["am"], scr["al"], scr["aacc"]
        n_items = scr["wl_cnt"][0]
        KR = NKVR if ex.paged_kv else NKV

        kvbuf: Any = None
        kbuf: Any = None
        vbuf: Any = None
        if ex.paged_kv:
            raw_refs = ctx.raw_refs
            P_BLK = BLK // ex.page_size

            if KVE:
                epool_ref = stream.borrow_epool()
                if interpret:

                    def kv_slot(s):
                        return epool_ref.at[s]

                    def kv_src(r, kp):
                        return r.at[kp]

                    def kv_dst(s, p):
                        return kv_slot(s).at[
                            :, :, :, pl.ds(p * ex.page_size, ex.page_size)
                        ]

                else:

                    def kv_slot(s):
                        return pool_alias.view(
                            epool_ref.at[pl.ds(s * EBROWS, KV_ITEM // 128)],
                            (2, HD // 4, BLK),
                            U32,
                        )

                    def kv_src(r, kp):
                        return r.at[kp].bitcast(U32).reshape(2, HD // 4, ex.page_size)

                    def kv_dst(s, p):
                        return kv_slot(s).at[
                            :, :, pl.ds(p * ex.page_size, ex.page_size)
                        ]

            else:
                kvbuf = scr["kvbuf"]

                def kv_src(r, kp):
                    return r.at[kp]

                def kv_dst(s, p):
                    return kvbuf.at[s, :, :, :, pl.ds(p * ex.page_size, ex.page_size)]

            def kv_stage(s):
                return kv_slot(s) if KVE else kvbuf.at[s]

            def kv_start(i):
                @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                def _():
                    s = i % KR
                    k0 = i * P_BLK
                    sem = ksem.at[s]
                    kp_tab = scr["wl_kp"]

                    def start_for_ref(r):
                        for p in range(P_BLK):
                            kp = kp_tab[k0 + p]
                            pltpu.make_async_copy(
                                kv_src(r, kp), kv_dst(s, p), sem
                            ).start()

                    bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

            def kv_wait(i):
                if "kv" not in skip:
                    s = i % KR
                    if interpret:
                        for p in range(P_BLK):
                            pltpu.make_async_copy(
                                raw_refs[0].at[0], kv_dst(s, p), ksem.at[s]
                            ).wait()
                    else:
                        st_ = kv_stage(s)
                        pltpu.make_async_copy(st_, st_, ksem.at[s]).wait()

            lax.fori_loop(0, KR, lambda i, _: (kv_start(i), 0)[1], 0)
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

            def rms_zc(x_in, w_in):
                return rmsnorm(x_in, w_in, ap.eps, zero_centered=ap.zero_centered_norm)

            def rope(x_in, cos_in, sin_in):
                return rope_partial(x_in, cos_in, sin_in, ROT)

            koff = nh * 2 * HD
            rcs = scr["rcs"]
            with sc("att_qk"):
                if QV:
                    cos = scr["rope"][0, :B, :half]
                    sin = scr["rope"][1, :B, :half]
                    for h in range(nh):
                        q0_ = h * 2 * HD
                        qh = rms_zc(ybuf[:B, q0_ : q0_ + HD], qn)
                        qh = rnd(rope(qh, cos, sin) * HD**-0.5)
                        for j in range(NHH):
                            scr["qg"][
                                0 if GQ else gi,
                                jnp.int32(j),
                                pl.ds(h, B, stride=SR),
                                :,
                            ] = qh[:, j * 128 : (j + 1) * 128]
                    if GQ:
                        for sq in range(NS):
                            r0 = sq * QG
                            scr["qxb"][gi, sq] = jnp.concatenate(
                                [scr["qg"][0, j, r0 : r0 + QG, :] for j in range(NHH)],
                                axis=-1,
                            ).astype(BF16)
                else:
                    for b in range(B):
                        pb = pos_ref[b]
                        ang = pb.astype(F32) * inv
                        cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                        rcs[0, b : b + 1, :] = cos
                        rcs[1, b : b + 1, :] = sin
                        qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                        q = rms_zc(qg[:, :HD], qn)
                        q = rnd(rope(q, cos, sin) * HD**-0.5)
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

            qx = []
            if SPLIT:
                with sc("att_qgather"):
                    qgb = scr["qxb"] if GQ else scr["qg"]
                    qx = comm.group_copies(
                        lambda off: qgb.at[gi],
                        qgb.at[gi],
                        scr["sq_ssem"],
                        scr["sq_rsem"],
                        G,
                    )
                    for d in qx:
                        d.start()
            if ex.FUSED:
                with sc("att_kvrd"):
                    fused_io().gqa_fused_kvrd_issue(ctx, a)
            with sc("att_kvnew"):
                if QV:
                    cos = scr["rope"][0, :B, :half]
                    sin = scr["rope"][1, :B, :half]
                    for h in range(nh):
                        q0_ = h * 2 * HD
                        scr["gate"][:B, h * HD : (h + 1) * HD] = ybuf[
                            :B, q0_ + HD : q0_ + 2 * HD
                        ]
                    k = rms_zc(ybuf[:B, koff : koff + HD], kn)
                    v = ybuf[:B, koff + HD : koff + 2 * HD]
                    k = rope(k, cos, sin)
                    if ex.paged_kv:
                        k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                        v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                        scr["knew"][:B, :] = k_fp8.astype(F32)
                        scr["vnew"][:B, :] = v_fp8.astype(F32)
                    else:
                        scr["knew"][:B, :] = k
                        scr["vnew"][:B, :] = v
                else:
                    for b in range(B):
                        cos, sin = rcs[0, b : b + 1, :], rcs[1, b : b + 1, :]
                        qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                        scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                        k = rms_zc(ybuf[b : b + 1, koff : koff + HD], kn)
                        v = ybuf[b : b + 1, koff + HD : koff + 2 * HD]
                        k = rope(k, cos, sin)
                        if ex.paged_kv:
                            k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                            v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                            scr["knew"][b : b + 1, :] = k_fp8.astype(F32)
                            scr["vnew"][b : b + 1, :] = v_fp8.astype(F32)
                        else:
                            scr["knew"][b : b + 1, :] = k
                            scr["vnew"][b : b + 1, :] = v
                if ex.paged_kv:
                    scr["knew_o"][...] = scr["knew"][...].astype(FP8)
                    scr["vnew_o"][...] = scr["vnew"][...].astype(FP8)
                if not ex.FUSED:
                    k_src = scr["knew_o"] if ex.paged_kv else scr["knew"]
                    v_src = scr["vnew_o"] if ex.paged_kv else scr["vnew"]
                    kv_out = (
                        pltpu.make_async_copy(
                            k_src, O["knew"].at[a], scr["kv_out_sem"].at[0]
                        ),
                        pltpu.make_async_copy(
                            v_src, O["vnew"].at[a], scr["kv_out_sem"].at[1]
                        ),
                    )
                    for d in kv_out:
                        d.start()
                else:
                    kv_out = ()

            if SPLIT or VER:
                with sc("att_qgather"):
                    qgb = scr["qg"]
                    for d in qx:
                        d.wait_recv()
                    for d in qx:
                        d.wait_send()
                    if GQ:
                        for sq in range(NS):
                            for g_idx in range(G):
                                scr["qb"][sq, g_idx * QG : (g_idx + 1) * QG] = scr[
                                    "qxb"
                                ][g_idx, sq]

                    def qg_rows(g_idx, r0, n):
                        if not QV:
                            t_, o_ = divmod(r0, 8)
                            return qgb[g_idx, t_, o_ : o_ + n, :]
                        return jnp.concatenate(
                            [qgb[g_idx, j, r0 : r0 + n, :] for j in range(NHH)],
                            axis=-1,
                        )

                    def group_q(b):
                        if not SPLIT:
                            return qg_rows(0, b * 8, 8)
                        t_, o_ = ap.qrow(b)
                        qa = jnp.concatenate(
                            [qg_rows(g_idx, t_ * 8 + o_, nh) for g_idx in range(G)],
                            axis=0,
                        )
                        if G * nh < QR:
                            qa = jnp.concatenate(
                                (qa, jnp.zeros((QR - G * nh, HD), F32)), 0
                            )
                        return qa

                    for sq in range(NS if not GQ else 0):
                        qs = [group_q(sq * T + t) for t in range(T)]
                        qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
                        scr["qb"][sq] = qs.astype(BF16)

            with sc("att_core"):
                col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
                col_v: Any = (
                    lax.broadcasted_iota(I32, (HD, SUB), 1)
                    if (ex.paged_kv and AMASK == 0)
                    else None
                )
                am[...] = jnp.full(am.shape, -1e30, F32)
                al[...] = jnp.zeros(al.shape, F32)
                aacc[...] = jnp.zeros(aacc.shape, F32)

                def item_front(i):
                    s = i % KR
                    b, t = scr["wl_b"][i], scr["wl_t"][i]
                    plen = pos_ref[b * T] if VER else pos_ref[b]
                    qrows = scr["qb"][b]
                    k_full: Any = None
                    v_full: Any = None
                    v_u32: Any = None
                    if ex.paged_kv:
                        if interpret:
                            kst = kv_slot(s) if KVE else kvbuf.at[s]
                            k_u32 = lax.bitcast_convert_type(
                                jnp.swapaxes(kst[0], 1, 2), U32
                            )
                            v_u32 = lax.bitcast_convert_type(
                                jnp.swapaxes(kst[1], 1, 2), U32
                            )
                        elif KVE:
                            kst = kv_slot(s)
                            k_u32 = kst[0]
                            v_u32 = kst[1]
                        else:
                            k_u32 = (
                                kvbuf.at[s, 0].bitcast(U32).reshape(HD // 4, BLK)[...]
                            )
                            v_u32 = (
                                kvbuf.at[s, 1].bitcast(U32).reshape(HD // 4, BLK)[...]
                            )
                        k_full = pltpu.bitcast(k_u32, FP8)
                        if AMASK in (0, 3, 4):
                            v_full = pltpu.bitcast(v_u32, FP8)
                    outs = []
                    for j in range(BLK // SUB):
                        off = t * BLK + j * SUB
                        rem = plen - off
                        full_blk = off + SUB <= plen
                        if ex.paged_kv:
                            kj = (
                                k_full
                                if SUB == BLK
                                else k_full[:, j * SUB : (j + 1) * SUB]
                            )
                            if AMASK == 0:
                                vj = (
                                    v_full
                                    if SUB == BLK
                                    else v_full[:, j * SUB : (j + 1) * SUB]
                                )
                                vj = jnp.where(
                                    col_v + off < plen, vj, jnp.zeros_like(vj)
                                )
                            elif AMASK == 1:
                                vu = (
                                    v_u32
                                    if SUB == BLK
                                    else v_u32[:, j * SUB : (j + 1) * SUB]
                                )
                                vu = jnp.where(col < rem, vu, jnp.uint32(0))
                                vj = pltpu.bitcast(vu, FP8)
                            else:
                                assert AMASK == 2
                                vu = (
                                    v_u32
                                    if SUB == BLK
                                    else v_u32[:, j * SUB : (j + 1) * SUB]
                                )
                                vu = lax.cond(
                                    full_blk,
                                    lambda v_val=vu: v_val,
                                    lambda v_val=vu, r_val=rem: jnp.where(
                                        col < r_val, v_val, jnp.uint32(0)
                                    ),
                                )
                                vj = pltpu.bitcast(vu, FP8)
                            scores = stream.mm(qrows, kj, NN)
                        else:
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
                        rmax = jnp.max(scores, axis=-1, keepdims=True)
                        outs.append((scores, rmax, vj))
                    return b, outs

                def item_back(b, outs):
                    for scores, rmax, vj in outs:
                        m_old = am[b]
                        m_new = jnp.maximum(m_old, rmax)
                        alpha = jnp.exp(m_old - m_new)
                        pr = jnp.exp(scores - m_new[:, 0:1])
                        pv = stream.mm(pr.astype(BF16), vj, NT if ex.paged_kv else NN)
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

            def gq_merge_start():
                psend, pri = scr["psend"], scr["pri"]
                with sc("att_merge"):
                    lane = lax.broadcasted_iota(I32, (QG, 128), 1)
                    for g_idx in range(G):
                        rs = slice(g_idx * QG, (g_idx + 1) * QG)
                        for s_ in range(NS):
                            psend[g_idx, s_] = aacc[s_, rs]
                        xm, xl = am[0, rs], al[0, rs]
                        for s_ in range(1, NS):
                            hi = lane >= s_ * MLW
                            xm = jnp.where(hi, am[s_, rs], xm)
                            xl = jnp.where(hi, al[s_, rs], xl)
                        psend[g_idx, NS] = jnp.concatenate((xm, xl), axis=-1)
                    pri[gi] = psend[gi]
                    mx = comm.group_copies(
                        lambda off: psend.at[gi ^ off],
                        pri.at[gi],
                        scr["sp_ssem"],
                        scr["sp_rsem"],
                        G,
                    )
                    for d in mx:
                        d.start()
                return mx

            def slab_merge_start():
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
                    mx = comm.group_copies(
                        lambda off: pso.at[gi ^ off],
                        pri.at[gi],
                        scr["sp_ssem"],
                        scr["sp_rsem"],
                        G,
                    )
                    for d in mx:
                        d.start()
                return mx

            if SPLIT:
                mx = gq_merge_start() if GQ else slab_merge_start()
                pri = scr["pri"]
                if ex.FUSED:
                    with sc("att_kvwr"):
                        fused_io().gqa_fused_kv_rmw(ctx, ap, a)
                with sc("att_gate"):
                    scr["gate"][0:B, :] = rnd(jax.nn.sigmoid(scr["gate"][0:B, :]))
                with sc("att_merge"):
                    for d in mx:
                        d.wait_recv()
                    for d in mx:
                        d.wait_send()
                    for t in range(NBP if not GQ else 0):
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
                if GQ:
                    pri = scr["pri"]
                    mlt = [pri[g_idx, NS] for g_idx in range(G)]
                    row_q = lax.broadcasted_iota(I32, (QG, 1), 0)
                    for sq in range(NS):
                        f = sq * MLW
                        parts = [pri[g_idx, sq] for g_idx in range(G)]
                        mg = [x[:, f : f + 1] for x in mlt]
                        lg = [x[:, 128 + f : 128 + f + 1] for x in mlt]
                        m = mg[0]
                        for x in mg[1:]:
                            m = jnp.maximum(m, x)
                        wg = [jnp.exp(x - m) for x in mg]
                        lsum_gq: Any = sum(w * x for w, x in zip(wg, lg))
                        asum_gq: Any = sum(w * x for w, x in zip(wg, parts))
                        r0 = sq * QG
                        qblk = jnp.concatenate(
                            [scr["qg"][0, j, r0 : r0 + QG, :] for j in range(NHH)],
                            axis=-1,
                        )
                        m_old, l_old, a_old = m, lsum_gq, asum_gq
                        for j in range(T):
                            bj = sq * T + j
                            s_new = jnp.sum(
                                qblk * scr["knew"][bj : bj + 1, :],
                                axis=-1,
                                keepdims=True,
                            )
                            m_new = jnp.maximum(m_old, s_new)
                            alpha = jnp.exp(m_old - m_new)
                            p_new = jnp.exp(s_new - m_new)
                            l_new = l_old * alpha + p_new
                            a_new = a_old * alpha + p_new * scr["vnew"][bj : bj + 1, :]
                            if j == 0:
                                m_old, l_old, a_old = m_new, l_new, a_new
                            else:
                                upd = row_q >= j * nh
                                m_old = jnp.where(upd, m_new, m_old)
                                l_old = jnp.where(upd, l_new, l_old)
                                a_old = jnp.where(upd, a_new, a_old)
                        ob = rnd(a_old / l_old)
                        for j in range(NHH):
                            scr["ost"][j, r0 : r0 + QG, :] = ob[
                                :, j * 128 : (j + 1) * 128
                            ]
                else:
                    NR = nh if PACK else 8
                    for b in range(B):
                        sq, tb = divmod(b, T)
                        q0 = tb * QR
                        m_old = am[sq, q0 : q0 + NR][:, 0:1]
                        l_old = al[sq, q0 : q0 + NR][:, 0:1]
                        a_old = aacc[sq, q0 : q0 + NR]
                        if QV:
                            qf = jnp.concatenate(
                                [
                                    scr["qg"][gi, j, b * SR : b * SR + NR, :]
                                    for j in range(NHH)
                                ],
                                axis=-1,
                            )
                        elif PACK:
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
                                    qf * scr["knew"][bj : bj + 1, :],
                                    axis=-1,
                                    keepdims=True,
                                )
                                m_new = jnp.maximum(m_old, s_new)
                                alpha = jnp.exp(m_old - m_new)
                                p_new = jnp.exp(s_new - m_new)
                                l_old = l_old * alpha + p_new
                                a_old = (
                                    a_old * alpha + p_new * scr["vnew"][bj : bj + 1, :]
                                )
                                m_old = m_new
                            outb = a_old / l_old
                        else:
                            s_new = jnp.sum(
                                qf * scr["knew"][b : b + 1, :],
                                axis=-1,
                                keepdims=True,
                            )
                            m_new = jnp.maximum(m_old, s_new)
                            alpha = jnp.exp(m_old - m_new)
                            p_new = jnp.exp(s_new - m_new)
                            l_tot = l_old * alpha + p_new
                            outb = (
                                a_old * alpha + p_new * scr["vnew"][b : b + 1, :]
                            ) / l_tot
                        if QV:
                            ob = rnd(outb[:nh])
                            for j in range(NHH):
                                scr["ost"][j, b * SR : b * SR + nh, :] = ob[
                                    :, j * 128 : (j + 1) * 128
                                ]
                            continue
                        for h in range(nh):
                            oh = rnd(outb[h : h + 1, :])
                            gh = scr["gate"][b : b + 1, h * HD : (h + 1) * HD]
                            if not SPLIT:
                                gh = rnd(jax.nn.sigmoid(gh))
                            scr["mix"][b : b + 1, h * HD : (h + 1) * HD] = rnd(oh * gh)
                if QV:
                    for h in range(nh):
                        oh = jnp.concatenate(
                            [
                                scr["ost"][jnp.int32(j), pl.ds(h, B, stride=SR), :]
                                for j in range(NHH)
                            ],
                            axis=-1,
                        )
                        gh = scr["gate"][:B, h * HD : (h + 1) * HD]
                        if not SPLIT:
                            gh = rnd(jax.nn.sigmoid(gh))
                        scr["mix"][:B, h * HD : (h + 1) * HD] = rnd(oh * gh)

            with sc("att_out"):
                if ex.FUSED and not SPLIT:
                    fused_io().gqa_fused_kv_rmw(ctx, ap, a)
                elif not ex.FUSED:
                    for d in kv_out:
                        d.wait()
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = stream.dmatmul("att_out", scr["xb2"], H)

        return yo * ctx.sm(slot, "out_s")

    def epilogue(self, ctx: Any) -> None:
        if ctx.ex.FUSED:
            fused_io().gqa_fused_epilogue(ctx)

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Scatter `knew` and `vnew` into `kc` and `vc` inside `shard_map`."""
        if ex.FUSED:
            if POOL_STATE_NAME in raw_outputs:
                return {POOL_STATE_NAME: raw_outputs[POOL_STATE_NAME]}
            return {}
        knew, vnew = raw_outputs["knew"], raw_outputs["vnew"]
        if ex.paged_kv or "kc" not in states:
            return {"knew": knew, "vnew": vnew}
        kc, vc = states["kc"], states["vc"]
        if ex.resolve_tiles(g).kv_scatter_vec_ns < ex.NS:
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
