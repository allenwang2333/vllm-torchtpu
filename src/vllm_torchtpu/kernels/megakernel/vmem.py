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
"""VMEM, SMEM, and Semaphore allocation and double-buffered `smalls` management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel import pool_alias
from vllm_torchtpu.kernels.megakernel.ops import BF16, F32, FP8, I32, rup
from vllm_torchtpu.kernels.megakernel.planner import DENSE_KINDS, EXPERT_KINDS, plan
from vllm_torchtpu.kernels.megakernel.spec import Tiles

U8 = jnp.uint8


@dataclass(frozen=True)
class DerivedParams:
    """Precomputed build-time parameters shared by stream, comm, vmem, and blocks."""

    R: int
    B: int
    M: int
    H: int
    L: int
    E: int
    AE: int
    MR: int
    G: int
    MB: int
    T: int
    NS: int
    VER: bool
    tk: dict[str, int]
    nt: dict[str, int]
    kinds: dict[str, tuple[int, int, Any]]
    t_gdn: int
    t_att: int
    t_block: int
    T_TOTAL: int
    segs_gdn: list[tuple[str, int, int, int]]
    segs_att: list[tuple[str, int, int, int]]
    lay: dict[str, tuple[int, int]]
    S: int
    NBD: int
    NBE: int
    BROWS: int
    EBROWS: int
    T13: int
    T2: int
    TE: int
    NSS: int
    PIPE: bool
    BLK: int
    NKV: int
    nh: int
    kh: int
    vh: int
    HD: int
    KD: int
    VD: int
    QKV: int
    Z: int
    BA: int
    IR: int
    ER: int
    PR: int
    PC: int
    HPC: int
    CH: int
    NACC: int
    NY: int
    KX2: int
    NMIX: int
    typed: bool
    N_MOE: int
    LAZY_ATT: bool
    LAZY_MOE: bool
    SPLIT: bool
    TS: int
    QR: int
    TQ: int
    SUB: int
    PW: int
    NWL: int
    FUSED: bool
    WIDE: bool
    GCH: bool
    CB: int
    NF: int
    TR: int
    NCH: int
    NRB: int
    CBV: int
    NGI: int
    NGO: int
    G_PER_ATT: int
    qkv_128: int
    A_conv: int
    conv8_rows: int
    ssm_u32_rows: int
    ckpt_is_3d: bool
    roff_is_2d: bool
    K_ckpt: int
    ckpt_t_off: int
    HIER: int
    RSAG: bool
    WAR: bool
    AR_R: int
    AR_C: int
    CW: int
    HC2: int
    AR_ROUNDS: tuple[int, ...]
    AR_PEERS: int
    PACK: bool
    PB: int
    NBP: int
    ECOPY: bool
    EXTRA: int
    vshape: dict[str, tuple[int, int]]
    vdtype: dict[str, Any]

    def qrow(self, b: int) -> tuple[int, int]:
        return b // self.PB, (b % self.PB) * self.nh


def build_scratch(
    c: Any,
    tiles: Tiles,
    *,
    interpret: bool = False,
    forced_routing: bool = False,
    row_mask: bool = False,
) -> tuple[DerivedParams, dict[str, Any]]:
    """Compute `DerivedParams` and allocate all VMEM/SMEM/Semaphore scratch specs."""
    R, B, M, H, L, E = c.R, c.B, c.M, c.H, c.L, c.E
    AE, MR, G = c.att_every, c.MR, c.G
    MB = MR if tiles.mb8 == 2 else (max(16, MR) if tiles.mb8 else M)
    p = plan(c, tiles)
    tk, nt, kinds = p["tk"], p["nt"], p["kinds"]
    t_gdn, t_att, t_block = p["t_gdn"], p["t_att"], p["t_block"]
    T_TOTAL = (L // AE) * t_block
    lay, S = p["smalls_layout"], p["smalls_width"]
    NBD, NBE = tiles.nbd, tiles.nbe
    BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128
    T13, T2 = nt["e13"], nt["e2"]
    TE = T13 + T2
    NSS = 3 + (NBE - 1) // TE
    PIPE = tiles.pipe and T13 == 1 and T2 == 1
    BLK, NKV = tiles.kv_blk, tiles.nkv
    assert (c.paged_kv or c.ctx % BLK == 0) and BLK % 128 == 0
    if c.paged_kv:
        assert BLK % c.page_size == 0, (BLK, c.page_size)
    nh, kh, vh, HD, KD, VD = c.nh, c.kh, c.vh, c.HD, c.KD, c.VD
    QKV, Z, BA, IR, ER = c.QKV, c.Z, c.BA, c.IR, c.ER
    PR, PC = c.PR, c.PC
    HPC = H // PC
    CH = min(1024, H)
    assert H % CH == 0
    NACC = max(c.NLI, c.NAI, H, E, 2 * c.ISR, 2 * IR)
    NY = max(c.NLI, c.NAI)
    KX2 = max(Z, nh * HD, c.ISR, IR)
    NMIX = max(Z, nh * HD)
    typed = bool(interpret)
    N_MOE = nt["router"] + nt["sh13"] + nt["sh2"]
    LAZY_ATT = tiles.lazy and nt["att_in"] + nt["att_out"] <= NBD
    LAZY_MOE = tiles.lazy and N_MOE <= NBD
    SPLIT = tiles.split and G > 1
    assert not SPLIT or G & (G - 1) == 0
    TS = G if SPLIT else 1
    QR = rup(G * nh, 8) if SPLIT else 8
    T, NS = c.T, c.NS
    VER = T > 1
    TQ = T * QR
    assert not (VER and tiles.inplace), "verify returns per-token states: no aliasing"
    SUB = tiles.kv_sub or BLK
    assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
    PW = HD + 128
    eff_ctx = max(c.ctx, c.max_pages_per_seq * c.page_size) if c.paged_kv else c.ctx
    max_nb = -(-eff_ctx // BLK)
    NWL = rup(NS * -(-max_nb // TS), 8)
    FUSED = c.paged_kv and c.fused_pool_io
    WIDE = B > 16
    GCH = WIDE and tiles.gdn_rows > 0 and c.NGDN > 0 and not FUSED
    CB = tiles.gdn_rows if GCH else B
    assert not GCH or CB % 8 == 0, ("gdn_rows must be a multiple of 8", tiles.gdn_rows)
    assert not (GCH and VER) or CB % T == 0, ("gdn_rows must be a multiple of T", CB, T)
    NF, TR = divmod(B, CB)
    NCH = NF + (TR > 0)
    NRB = min(tiles.gdn_bufs, NCH)
    assert not GCH or NCH == 1 or NRB >= 2, ("gdn_bufs >= 2", tiles.gdn_bufs)
    CBV = CB * vh
    NGI = min(NS, 8)
    NGO = min(NS, 2)
    G_PER_ATT = max(1, AE - 1)
    qkv_128 = QKV // 128
    A_conv = 16
    conv8_rows = A_conv * qkv_128
    ssm_u32_rows = (vh * KD) // 2
    ckpt_is_3d = not c.ckpt_2d
    roff_is_2d = not c.roff_1d
    K_ckpt = c.ckpt_k if c.ckpt_k > 0 else T
    ckpt_t_off = 0 if K_ckpt == T else 1
    HIER = tiles.hier if R > 1 else 0
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (WIDE or MR * H <= PR * PC) if tiles.war < 0 else bool(tiles.war)
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    CW = AR_C // R
    HC2 = AR_C // 2
    assert not RSAG or CW % 128 == 0
    assert HIER in (0, 1, 2, 3) and (HIER == 0 or R == 8)
    assert HIER != 2 or HC2 % 128 == 0
    AR_ROUNDS = (
        (0, 1, 2) if HIER in (2, 3) else ((0, 1) if (RSAG or HIER == 1) else (0,))
    )
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))
    PACK = (WIDE if tiles.pack < 0 else bool(tiles.pack)) and SPLIT and nh < 8
    PB = 8 // nh if PACK else 1
    NBP = -(-B // PB)
    ECOPY = not c.moe_tp and c.e_slots > ER
    assert not (row_mask and ECOPY), "row_mask: identity expert placement only"
    EXTRA = c.EW - ER

    vshape = {k: (tk[k], kinds[k][1]) for k in kinds}
    vdtype = {k: kinds[k][2] for k in kinds}
    for k in kinds:
        assert pool_alias.storage_size(vshape[k], jnp.dtype(vdtype[k])) <= (
            tiles.ebank if k in EXPERT_KINDS else tiles.bank
        ), k

    scratch: dict[str, Any] = {
        "dsem": pltpu.SemaphoreType.DMA((NBD,)),
        "esem": pltpu.SemaphoreType.DMA((NBE,)),
        "sm": pltpu.VMEM((2, 1, S), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "res": pltpu.VMEM((MB, H), F32),
        "xb": pltpu.VMEM((MB, H), BF16),
        "xb2": pltpu.VMEM((MB, KX2), BF16),
        "acc": pltpu.VMEM((MB, NACC), F32),
        "ybuf": pltpu.VMEM((MB, NY), F32),
        "part": pltpu.VMEM((MB, H), F32),
        "mix": pltpu.VMEM((MB, NMIX), F32),
        "send": pltpu.VMEM(
            (3, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, R, AR_R, CW)
            if RSAG
            else (2, AR_R, AR_C),
            BF16,
        ),
        "recv": pltpu.VMEM(
            (3, 2, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, 2, AR_R, AR_C)
            if HIER == 1
            else (2, R, AR_R, CW)
            if RSAG
            else (2, R, AR_R, AR_C),
            BF16,
        ),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        # GatedDeltaNet
        "recb": pltpu.VMEM((B * vh, KD, VD), BF16),
        "sbuf": pltpu.VMEM((vh, KD, VD), F32),
        "convb": pltpu.VMEM((NS, c.HR, QKV) if VER else (c.CR, QKV), F32),
        "st_sem": pltpu.SemaphoreType.DMA((4,)),
        # Attention
        "kbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
        "vbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
        "kv_sem": pltpu.SemaphoreType.DMA((2, NKV)),
        "wl_b": pltpu.SMEM((NWL,), I32),
        "wl_t": pltpu.SMEM((NWL,), I32),
        "qb": pltpu.VMEM((NS, TQ, HD), BF16),
        "am": pltpu.VMEM((NS, TQ, 128), F32),
        "al": pltpu.VMEM((NS, TQ, 128), F32),
        "aacc": pltpu.VMEM((NS, TQ, HD), F32),
        "knew": pltpu.VMEM((MR, HD), F32),
        "vnew": pltpu.VMEM((MR, HD), F32),
        "gate": pltpu.VMEM((MR, nh * HD), F32),
        "kv_out_sem": pltpu.SemaphoreType.DMA((2,)),
        # MoE
        "wn": pltpu.VMEM((MR, E), F32),
        "sg": pltpu.VMEM((MB, 128), F32),
        "hitv": pltpu.VMEM((MR, 128), I32),
        "hits": pltpu.SMEM((MR, 128), I32),
        "elist": pltpu.SMEM((rup(B * c.TOPK, 128),), I32),
        "seen": pltpu.SMEM((8, 128), I32),
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        "es": pltpu.VMEM((NSS, 1, 2 * IR + H), F32),
        "es_sem": pltpu.SemaphoreType.DMA((NSS,)),
        "st": pltpu.SMEM((8,), I32),
        "fn": pltpu.VMEM((1, H), F32),
    }
    if c.paged_kv:
        del scratch["kbuf"], scratch["vbuf"]
        scratch["kvbuf"] = pltpu.VMEM((NKV, 2, 64, 4, BLK), FP8)
        scratch["kv_sem"] = pltpu.SemaphoreType.DMA((NKV,))
        scratch["knew_o"] = pltpu.VMEM((MR, HD), FP8)
        scratch["vnew_o"] = pltpu.VMEM((MR, HD), FP8)
    if FUSED:
        for k in ("recb", "convb", "st_sem", "kv_out_sem"):
            del scratch[k]
        if c.NGDN > 0:
            scratch.update({
                "gdn_in_page": pltpu.VMEM((NGI, 5, 2, 64, 4, c.page_size), FP8),
                "gdn_out_page": pltpu.VMEM((NGO, 5 * T, 2, 64, 4, c.page_size), FP8),
                "gdn_in_sem": pltpu.SemaphoreType.DMA((NGI,)),
                "gdn_out_sem": pltpu.SemaphoreType.DMA((NGO, T)),
                "gdn_in_pend": pltpu.SMEM((NGI,), I32),
                "gdn_out_mode": pltpu.SMEM((NGO,), I32),
            })
        else:
            del scratch["sbuf"]
        scratch.update({
            "kv_scat_page": pltpu.VMEM((NS, 2, 64, 4, c.page_size), FP8),
            "kv_rd_sem": pltpu.SemaphoreType.DMA((NS,)),
            "kv_wr_sem": pltpu.SemaphoreType.DMA((NS,)),
            "kv_rd_pend": pltpu.SMEM((NS,), I32),
            "kv_wr_pend": pltpu.SMEM((NS,), I32),
        })
    if ECOPY:
        nb_picks = rup(B * c.TOPK, 128)
        scratch.update({
            "ecopy": pltpu.SMEM((2, 1, E), F32),
            "ec_sem": pltpu.SemaphoreType.DMA((2,)),
            "wlist": pltpu.SMEM((nb_picks,), I32),
            "cplist": pltpu.SMEM((nb_picks,), I32),
            "cpchoice": pltpu.SMEM((nb_picks,), I32),
            "rload": pltpu.SMEM((R,), I32),
        })
        if tiles.ecopy_opt == 1:
            scratch["ulist"] = pltpu.SMEM((nb_picks,), I32)
    if SPLIT:
        scratch.update({
            "qg": pltpu.VMEM((G, B, 8, HD), F32),
            "pso": pltpu.VMEM((G, B, 8, PW), F32),
            "pri": pltpu.VMEM((G, B, 8, PW), F32),
            "sq_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sq_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sp_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sp_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
        })
    elif VER:
        scratch["qg"] = pltpu.VMEM((1, B, 8, HD), F32)
    if VER and not GCH and not FUSED:
        scratch.update({
            "ub": pltpu.VMEM((NS, rup(T, 8), QKV), F32),
            "rin_sem": pltpu.SemaphoreType.DMA((NS,)),
        })
    if RSAG:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, CW), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, R, AR_R, CW), BF16)
    elif HIER == 1:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, AR_C), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, AR_C), BF16)
    elif HIER == 2:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, HC2), BF16)
        scratch["s2_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["s2_recv"] = pltpu.VMEM((2, 2, AR_R, HC2), BF16)
    if forced_routing:
        scratch["rb"] = pltpu.VMEM((2, M, E), F32)
        scratch["rb_sem"] = pltpu.SemaphoreType.DMA((2,))
    if GCH:
        scratch["recb"] = pltpu.VMEM((NRB, CBV, KD, VD), BF16)
        scratch["rec_sem"] = pltpu.SemaphoreType.DMA((2, NRB))
    elif WIDE and c.NGDN == 0 and not FUSED:
        for k in ("recb", "sbuf", "convb"):
            del scratch[k]
    if PACK:
        scratch["qg"] = pltpu.VMEM((G, NBP, 8, HD), F32)
        scratch["pso"] = pltpu.VMEM((G, NBP, 8, PW), F32)
        scratch["pri"] = pltpu.VMEM((G, NBP, 8, PW), F32)
    if typed:
        for k in DENSE_KINDS:
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
        for k in EXPERT_KINDS:
            scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
    else:
        scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
        scratch["epool"] = pltpu.VMEM((NBE * EBROWS, 128), U8)

    dp = DerivedParams(
        R=R,
        B=B,
        M=M,
        H=H,
        L=L,
        E=E,
        AE=AE,
        MR=MR,
        G=G,
        MB=MB,
        T=T,
        NS=NS,
        VER=VER,
        tk=tk,
        nt=nt,
        kinds=kinds,
        t_gdn=t_gdn,
        t_att=t_att,
        t_block=t_block,
        T_TOTAL=T_TOTAL,
        segs_gdn=p["segs_gdn"],
        segs_att=p["segs_att"],
        lay=lay,
        S=S,
        NBD=NBD,
        NBE=NBE,
        BROWS=BROWS,
        EBROWS=EBROWS,
        T13=T13,
        T2=T2,
        TE=TE,
        NSS=NSS,
        PIPE=PIPE,
        BLK=BLK,
        NKV=NKV,
        nh=nh,
        kh=kh,
        vh=vh,
        HD=HD,
        KD=KD,
        VD=VD,
        QKV=QKV,
        Z=Z,
        BA=BA,
        IR=IR,
        ER=ER,
        PR=PR,
        PC=PC,
        HPC=HPC,
        CH=CH,
        NACC=NACC,
        NY=NY,
        KX2=KX2,
        NMIX=NMIX,
        typed=typed,
        N_MOE=N_MOE,
        LAZY_ATT=LAZY_ATT,
        LAZY_MOE=LAZY_MOE,
        SPLIT=SPLIT,
        TS=TS,
        QR=QR,
        TQ=TQ,
        SUB=SUB,
        PW=PW,
        NWL=NWL,
        FUSED=FUSED,
        WIDE=WIDE,
        GCH=GCH,
        CB=CB,
        NF=NF,
        TR=TR,
        NCH=NCH,
        NRB=NRB,
        CBV=CBV,
        NGI=NGI,
        NGO=NGO,
        G_PER_ATT=G_PER_ATT,
        qkv_128=qkv_128,
        A_conv=A_conv,
        conv8_rows=conv8_rows,
        ssm_u32_rows=ssm_u32_rows,
        ckpt_is_3d=ckpt_is_3d,
        roff_is_2d=roff_is_2d,
        K_ckpt=K_ckpt,
        ckpt_t_off=ckpt_t_off,
        HIER=HIER,
        RSAG=RSAG,
        WAR=WAR,
        AR_R=AR_R,
        AR_C=AR_C,
        CW=CW,
        HC2=HC2,
        AR_ROUNDS=AR_ROUNDS,
        AR_PEERS=AR_PEERS,
        PACK=PACK,
        PB=PB,
        NBP=NBP,
        ECOPY=ECOPY,
        EXTRA=EXTRA,
        vshape=vshape,
        vdtype=vdtype,
    )
    return dp, scratch

