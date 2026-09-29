"""GLM-5.2 decode megakernel, v4_pipe: v2_tpo (DP attention + EP MoE, o_proj TP over heads) with ONE
continuous weight stream: every streamed weight tile goes through a single ring of `nbank` byte banks in
consumption order, and a cursor-driven producer refills each freed bank from wherever the stream is
(next op, next layer, the routed experts once routing publishes the hit list), so HBM never drains at
an op or layer boundary (upstream Qwen decode_megakernel global tile ring; Kimi expert prefetch).

Per rank r (one decode token per rank, R ranks on a ("d",) mesh):

  for each layer l (fori_loop, per-layer operands picked by a binary `when` cascade):
    input RMSNorm -> fused_qkv_a (fp8 stream) -> q_a / kv_a norms -> k_pe rope
    q_b (fp8 stream) -> heads -> rope -> W_UK absorb (VPU, bf16 stream)
    KV write (nope row, rope row RMW) into the served SPARSECORE caches
    [full-indexer layers] indexer q/k/weights, idx-cache record write, page scan, TC top-k
    TC gather of the top-k nope / rope rows, sparse MLA softmax (two pass, served formula)
    W_UV (VPU) -> quantize the row (per-token scale) -> A2A head slices (remote DMA) ->
      o_proj on this rank's HR heads for all R tokens (fp8 stream, 1/R of the weight) ->
      RS unscaled f32 partials to the owner -> x sx x o_proj_s -> residual + post RMSNorm
    dense layers: fp8 dense MLP on the own token
    MoE layers: AG normed rows (remote DMA) -> route all R tokens -> hit local experts on all
      tokens -> RS partial rows (remote DMA) -> owner sum; shared expert on the own token
  final RMSNorm -> out

Activations for per-token ops are carried as [M=32, X] with only row 0 live, because an fp8
matmul operand block needs >= 32 sublane rows. Numerics follow NUMERICS.md (served formulas).
"""

import dataclasses
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias, vtopk
from .layout import rup, smalls_layout

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
U8 = jnp.uint8
M = 32                      # row padding of every activation block
FP8_MAX = 448.0
INT_MIN = np.int32(-(2 ** 31))
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))


# ------------------------------------------------------------------------------------------
# Operand plan
# ------------------------------------------------------------------------------------------

def weight_kinds(cfg):
    """kind -> (layers, per-rank shape, dtype)."""
    all_l = list(range(cfg.L))
    full = [l for l in all_l if cfg.full_indexer(l)]
    moe = [l for l in all_l if cfg.is_moe(l)]
    dense = [l for l in all_l if not cfg.is_moe(l)]
    H = cfg.H
    return {
        "qkv_a": (all_l, (H, rup(cfg.QKVA)), FP8),     # N padded: an unaligned minor dim
                                                        # gets a non-default layout = a copy/step
        "q_b": (all_l, (cfg.QL, cfg.HR * cfg.DQ), FP8),   # v9: columns of heads [r*HR, (r+1)*HR)
        "w_uk": (all_l, (cfg.HR, cfg.DN, cfg.KL), BF16),  # v9: heads [r*HR, (r+1)*HR)
        "w_uv": (all_l, (cfg.NH, cfg.KL, cfg.DV), BF16),
        "o_proj": (all_l, (cfg.HR * cfg.DV, H), FP8),  # rows of heads [r*HR, (r+1)*HR)
        "idx_wq": (full, (cfg.QL, cfg.IH * cfg.IDIM), FP8),
        "idx_wk": (full, (H, rup(cfg.IDIM + cfg.IH)), BF16),
        "gate": (moe, (H, cfg.E), BF16),
        "ew13": (moe, (cfg.ER, H, 2 * cfg.I), FP8),
        "ew2": (moe, (cfg.ER, cfg.I, H), FP8),
        "es13": (moe, (cfg.ER, 2 * cfg.I), F32),
        "es2": (moe, (cfg.ER, H), F32),
        "sw13": (moe, (H, 2 * cfg.IS), FP8),
        "sw2": (moe, (cfg.IS, H), FP8),
        "w13": (dense, (H, 2 * cfg.ID), FP8),
        "w13_s": (dense, (1, 2 * cfg.ID), F32),
        "w2": (dense, (cfg.ID, H), FP8),
    }


def cache_kinds(cfg, nb, rope_packed=False):
    """rope_packed: the served ROPE cache holds 64 B/token, eight tokens per 128-word row
    (kv_cache_utils SparseMLAKVCacheSpec.rope_packed), instead of 128 B/token, four per row."""
    all_l = list(range(cfg.L))
    full = [l for l in all_l if cfg.full_indexer(l)]
    pg = cfg.page
    return {
        "nope": (all_l, (nb, pg, 128), U32),
        "rope": (all_l, (nb, pg // (8 if rope_packed else 4), 128), U32),
        "idx": (full, (nb, pg // 4, 4, 256), U8),
    }


@dataclasses.dataclass(frozen=True)
class Tiles:
    head_group: int = 8            # heads per W_UK / W_UV tile
    # Mosaic's fp8 x fp8 MXU dot on TPU7x is off by ~0.4% rel (j-da524a3e); the bf16 upcast is
    # exact (e4m3 is a subset of bf16). "bf16": upcast both operands; "mixed": bf16 activation x
    # fp8 weight (Mosaic picks the lowering); "fp8": native fp8 x fp8 (inexact on TPU7x).
    dot: str = "mixed"
    # v4: all streamed kinds share ONE ring of `nbank` banks of `bank` bytes (u8 VMEM pool, typed per
    # tile through pool_alias views). Each kind's tile is the largest 128-row multiple that fits a bank.
    bank: int = 3 << 20
    nbank: int = 12
    # Per-phase jax.named_scope markers -> tpu.trace_start/stop (level 10) for an xprof capture.
    trace: bool = False
    # Indexer-scan page ring: scan_depth - 1 pages (256 KiB each) in flight. 2 -> 8 takes the 256k
    # page stream from 130 us to 37 us, under the 68 us page-body compute (j-f17f2938).
    scan_depth: int = 8
    # Gather prefetch: a layer that reuses the last top-k set (no indexer) has its rows known once
    # the previous layer's scores are done; each streamed matmul tile then issues `gpre` of them
    # while it waits on its weight DMA (scalar-issue-bound gather, 54 us/layer j-f9a1bd50). 0: off.
    gpre: int = 0
    # With gpre: issue gather rows only while take() would block (the next tile's DMA semaphore is
    # short of a full tile), not on every tile -- resident tiles are compute-bound and the issue
    # cost adds there (j-a067267f: m_shared 7.7 -> 29.5 us). `gpre` rows per poll.
    gwait: bool = False
    # ---- probe 27 arms (all off = the stack3b incumbent; er + hd on by default, bitwise-equal) ----
    # er: MoE stream order sw13, gate, sw2, exp -- route before the shared expert's w2 so the routed
    # expert tiles stream under sw2 instead of after routing (m_experts first-tile wait).
    early_route: bool = True    # probe 27: -91 us @L40 R=8 [measured: j-098f75f1]
    # kva: the KV-write rope-row read is issued at layer start, the row writes are waited one layer
    # later (double-buffered rows) instead of a blocking RMW round trip in L_kvw.
    kvw_async: bool = False
    # r1: packed rope re-laid to ONE aligned segment (lanes 0..31): 2 rope dots instead of 8, no select.
    rope1: bool = False
    # hd: fp8 bytes decoded by the hardware fp8->bf16 convert (u8 truncate + bitcast) instead of
    # the bit-twiddling fp8_decode; 0x7F/0xFF (NaN, never written: writes clip to +-448) -> +-448.
    hwdec: bool = True          # probe 27: -46 us @L40 R=8 [measured: j-098f75f1]
    # tb: top-k extraction per 128-row block over only the blocks each k chunk spans.
    topk_blk: bool = False
    topk_kc: int = 256

    def tk(self, K, N, dtype):
        """Largest 128-multiple divisor of K whose (tk, N) tile fits one bank. tk doubles as the
        dynamic lane offset into the activation row: 128-aligned (E2003)."""
        best = None
        for tk in range(128, K + 1, 128):
            if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= self.bank:
                best = tk
        assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, self.bank)
        return best


# ------------------------------------------------------------------------------------------
# In-kernel helpers
# ------------------------------------------------------------------------------------------

def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def shr(x, n):
    return lax.shift_right_logical(x, jnp.asarray(n, x.dtype))


def bswitch(idx, fns):
    """Run fns[k]() for k == idx (traced scalar); keys sorted ints. Binary cascade of whens."""
    keys = sorted(fns)

    def rec(ks):
        if len(ks) == 1:
            fns[ks[0]]()
            return
        mid = len(ks) // 2

        @pl.when(idx < ks[mid])
        def _():
            rec(ks[:mid])

        @pl.when(idx >= ks[mid])
        def _():
            rec(ks[mid:])

    rec(keys)


def quant_rows(x):
    """x f32 [M, K] -> (fp8 [M, K], scale f32 [M, 1]); quantize_block semantics."""
    s = jnp.max(jnp.abs(x), axis=1, keepdims=True) / FP8_MAX
    s = jnp.where(s == 0, 1.0, s)
    return jnp.clip(x / s, -FP8_MAX, FP8_MAX).astype(FP8), s


def rmsnorm(x, w, eps):
    """x f32 [M, H] (already residual-summed and bf16-rounded); w f32 [1, H]."""
    y = (x * lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + eps)).astype(BF16)
    return (y.astype(F32) * w).astype(BF16).astype(F32)


def lane_swap(x):
    """Interleaved-rope partner: out[2i] = x[2i+1], out[2i+1] = x[2i]."""
    n = x.shape[-1]
    lane = lax.broadcasted_iota(I32, x.shape, x.ndim - 1)
    return jnp.where(lane % 2 == 0, pltpu.roll(x, n - 1, x.ndim - 1), pltpu.roll(x, 1, x.ndim - 1))


def rope(x, cos, sin_signed):
    """bf16(x*cos + swap(x)*sin); cos/sin_signed [1, W] (identity lanes carry cos=1, sin=0)."""
    return (x * cos + lane_swap(x) * sin_signed).astype(BF16).astype(F32)


def packed_rope_to_padded(p, iv):
    """Gathered PACKED rope rows -> the padded rope lane layout the score math reads.

    p: [128, 128] i32, row r = the 128-word cache row holding gathered token t_r = iv[0, r] in its
    eighth s8 = t_r & 7 (lanes 16*s8 + j, word j = key bytes (j, 16+j, 32+j, 48+j)).
    Returns rows with that token's key in quarter s4 = t_r & 3 (lanes 32*s4 + j', word j' = bytes
    (j', 32+j') in bytes 0 and 1) -- exactly the lanes and bytes a padded row carries there, so
    the per-quarter q0s / q1s dots see identical operands. Other lanes hold finite fp8 garbage that
    only meets zero q lanes (fp8_decode has no NaN), as the padded layout's other tokens do."""
    s8 = jnp.transpose(jnp.broadcast_to(iv & 7, (128, 128)))         # per-row (sublane) eighth
    lane = lax.broadcasted_iota(I32, (128, 128), 1)
    b0, b1, b2 = (s8 & 1) == 1, (s8 & 2) == 2, (s8 & 4) == 4
    p = jnp.where(b2, pltpu.roll(p, 64, 1), p)                         # eighth s8 -> s4 (x16 lanes)
    lo = (p & 0xFF) | shl(shr(p, 16) & 0xFF, 8)                        # bytes (j, 32+j)
    hi = pltpu.roll((shr(p, 8) & 0xFF) | shl(shr(p, 24) & 0xFF, 8), 16, 1)   # (16+j, 48+j), +16 lanes
    c = jnp.where((shr(lane, 4) & 1) == (s8 & 1), lo, hi)             # key at lanes 16*s4 .. +32
    c = jnp.where(b0, pltpu.roll(c, 16, 1), c)
    return jnp.where(b1, pltpu.roll(c, 32, 1), c)                      # key at lanes 32*s4 .. +32


def fp8_decode(b):
    """int32 byte (e4m3fn bit pattern) -> f32 value."""
    s = shr(b, 7) & 1
    e = shr(b, 3) & 15
    m = b & 7
    bits = shl(s, 31) | shl(e + 120, 23) | shl(m, 20)
    normal = lax.bitcast_convert_type(bits, F32)
    sub = m.astype(F32) * (2.0 ** -9) * (1 - 2 * s).astype(F32)
    return jnp.where(e == 0, sub, normal)


def fp8_decode_hw(b):
    """probe 27 hd: int32 byte (0..255, e4m3fn bits) -> bf16 through the hardware fp8 convert.
    0x7F / 0xFF (NaN in e4m3fn; fp8_decode reads them as +-480) become 0x7E / 0xFE (+-448) so no
    NaN reaches a dot: the cache writer clips to +-448 and never emits them, so written bytes
    decode identically; they only occur in garbage lanes that meet zero q lanes."""
    b = jnp.where((b & 0x7F) == 0x7F, b ^ 1, b)
    return lax.bitcast_convert_type(b.astype(U8), FP8).astype(BF16)


def packed_rope_to_seg0(p, iv):
    """probe 27 r1: gathered PACKED rope rows -> the token's key in lanes 0..31 of every row,
    word j' = bytes (j', 32+j') -- quarter 0 of the padded layout whatever t_r & 3 is, so one
    q0 / q1 dot pair serves every row. Lanes 32..127 hold finite fp8 garbage that only meets zero
    q lanes (as in packed_rope_to_padded)."""
    s8 = jnp.transpose(jnp.broadcast_to(iv & 7, (128, 128)))         # per-row (sublane) eighth
    lane = lax.broadcasted_iota(I32, (128, 128), 1)
    p = jnp.where((s8 & 4) == 4, pltpu.roll(p, 64, 1), p)             # lanes 16*s8 -> 16*(s8&3)
    p = jnp.where((s8 & 2) == 2, pltpu.roll(p, 96, 1), p)             # -> 16*(s8&1)
    p = jnp.where((s8 & 1) == 1, pltpu.roll(p, 112, 1), p)            # -> lanes 0..15
    a = (p & 0xFF) | shl(shr(p, 16) & 0xFF, 8)                         # bytes (j, 32+j)
    b = (shr(p, 8) & 0xFF) | shl(shr(p, 24) & 0xFF, 8)                 # bytes (16+j, 48+j)
    return jnp.where(lane < 16, a, pltpu.roll(b, 16, 1))


def mxu(a, b, tiles):
    """dot operands per tiles.dot (see Tiles)."""
    if tiles.dot == "fp8" or (a.dtype != FP8 and b.dtype != FP8):
        return a, b
    if tiles.dot == "mixed":
        return a.astype(BF16), b
    return a.astype(BF16), b.astype(BF16)


class Phases:
    """Sequential trace regions without reindenting: mark("a") ... mark("b") ... end().
    A no-op unless enabled; each traced body (layer, pl.when branch) keeps its own instance."""

    def __init__(self, on):
        self.on, self.cur = on, None

    def mark(self, name):
        self.end()
        if self.on:
            self.cur = jax.named_scope(name)
            self.cur.__enter__()

    def end(self):
        if self.cur is not None:
            self.cur.__exit__(None, None, None)
            self.cur = None


def fp8_bytes(x8):
    return lax.bitcast_convert_type(x8, U8).astype(I32)


# ------------------------------------------------------------------------------------------
# The kernel
# ------------------------------------------------------------------------------------------

SEM_UNIT = 32         # bytes per DMA-semaphore count (probe 06, j-232ba686: counts up bytes/32, wait -> 0)
GATHER_UNROLL = int(os.environ.get("MK_GUNROLL", "16"))  # A/B knob; 16 taken (probe 07)
# "ag"/"a2a"/"rs" drop one collective each ("coll" drops all): serving-hang bisect at R=16.
# "agdiag": one AG right after the barrier, arrivals and both semaphores polled (bounded), then
# exactly the landed counts drained (never blocks); out row 1
# lanes = [me, spins/256, arrived, recv sem (3 base-256 digits, pre-drain), send sem (same),
# then from AGDIAG_TAG0 the tag in xall[r] (src + 1, 0 = never arrived) for r < R].
ABLATIONS = ("qtp", "weights", "gather", "index", "attn", "experts", "coll", "uk_uv", "topk", "ag", "a2a", "rs",
             "agdiag")
AGDIAG_SPIN = int(os.environ.get("MK_AGDIAG_SPIN", "1000000"))
AGDIAG_TAG0 = 9
# The barrier semaphore is keyed by collective_id. 0 is also the fused EP MoE kernel's (fused_moe/v2), so a
# peer still inside that kernel satisfied this kernel's entry barrier and the two kernels' remote DMAs landed
# on each other's semaphores (j-c4e6dce4, j-bf0ef082). Unique per kernel, as upstream does (21 / 25 / 45).
COLLECTIVE_ID = int(os.environ.get("MK_COLLECTIVE_ID", "37"))


def build(cfg, *, nb, max_pages, tiles=Tiles(), interpret=False, debug=False, skip=frozenset(),
          axis_name=None, rope_packed=False):
    """Returns (step_fn, operand names). step_fn(meta, bt, h0, rope_tab, smalls, *weights,
    *caches) -> (out [M, H] bf16, *caches) for ONE rank; wrap in shard_map.
    skip: timing ablations only (numerics are garbage), a subset of ABLATIONS.
    rope_packed: the ROPE caches are the 64 B/token PACKED layout (see cache_kinds). The gather,
    the KV write and the post-call scatter address row sl >> 3; attention re-lays each gathered
    chunk into the padded lane layout first, so the score math is unchanged (bit-exact)."""
    assert set(skip) <= set(ABLATIONS), skip
    typed = interpret or os.environ.get("MK_TYPED") == "1"   # diag: typed banks, no pool_alias
    R, L, H = cfg.R, cfg.L, cfg.H
    pg = cfg.page
    lg_pg = int(math.log2(pg))
    assert 1 << lg_pg == pg
    KT = cfg.KT
    NC = KT // 128
    ROWS = rup(max_pages * pg // 128, 128)
    assert ROWS <= 32 * 128, "top-k row summary is one (1, 128) word row"
    lay, S = smalls_layout(cfg)
    wk = weight_kinds(cfg)
    ck = cache_kinds(cfg, nb, rope_packed)
    # rope row of token slot sl: sl >> RSH; RSEG tokens of RW words per row
    RSH = 3 if rope_packed else 2
    RSEG, RW = 1 << RSH, 128 >> RSH
    G = tiles.head_group
    assert cfg.NH % G == 0
    GU = min(G, cfg.HR)                              # v9: heads per W_UK tile (of this rank's HR)
    assert cfg.HR % GU == 0 and R <= M
    NQB = cfg.HR * cfg.DQ                            # v9: q_b N per rank
    QW = cfg.QL + 128                                # v9: AG row = q_lora fp8 values (f32) | scale
    QA = cfg.KL + 128                                # v9: A2A block = absorbed q | q lanes DQ-128..DQ
    sm_scale = cfg.DQ ** -0.5

    def kdim(kind):
        return {"ew13": H, "ew2": cfg.I}.get(kind) or wk[kind][1][0]

    # --- tile sizes per streamed kind ---
    tk = {}
    for kind in ("qkv_a", "q_b", "o_proj", "idx_wq", "sw13", "sw2", "w13", "w2", "idx_wk", "gate"):
        K, N = wk[kind][1]
        tk[kind] = tiles.tk(K, N, wk[kind][2])
    tk["ew13"] = tiles.tk(H, 2 * cfg.I, FP8)
    tk["ew2"] = tiles.tk(cfg.I, H, FP8)

    NACC = max(cfg.QKVA + 128, cfg.NH * cfg.DQ, H, 2 * cfg.ID, 2 * cfg.I, 2 * cfg.IS,
               cfg.IH * cfg.IDIM, rup(cfg.IDIM + cfg.IH, 128), cfg.E, cfg.NH * cfg.DV)
    NACC = rup(NACC)
    KX = max(H, cfg.QL, cfg.NH * cfg.DV, cfg.ID, cfg.I, cfg.IS)
    KR = cfg.HR * cfg.DV                              # o_proj K per rank
    assert KR % 128 == 0

    # --- operand lists ---
    w_names = [(k, l) for k, (ls, _, _) in wk.items() for l in ls]
    c_names = [(k, l) for k, (ls, _, _) in ck.items() for l in ls]
    n_fixed = 5
    n_w = len(w_names)

    # --- scratch ---
    scratch = {
        "sm": pltpu.VMEM((2, 1, S), F32), "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "h": pltpu.VMEM((M, H), F32), "res": pltpu.VMEM((M, H), F32),
        "hn": pltpu.VMEM((M, H), F32),
        "xq": pltpu.VMEM((M, KX), FP8), "xqm": pltpu.VMEM((M, H), FP8),
        "xbf": pltpu.VMEM((M, H), BF16),
        "acc": pltpu.VMEM((M, NACC), F32),
        "qc": pltpu.VMEM((M, cfg.QL), F32),
        "qh": pltpu.VMEM((cfg.NH, cfg.DQ), F32),
        "ql": pltpu.VMEM((cfg.NH, cfg.KL), F32),
        "att": pltpu.VMEM((cfg.NH // G, 8, cfg.KL), F32),   # head group g in rows 0..G of [g]
        "orow": pltpu.VMEM((M, cfg.NH * cfg.DV), F32),
        "wrow": pltpu.VMEM((8, 128), U32), "rrow": pltpu.VMEM((8, 128), U32),
        "kv_sem": pltpu.SemaphoreType.DMA((3 if tiles.kvw_async else 2,)),
        "ih": pltpu.VMEM((M, cfg.IDIM), F32),
        "rec": pltpu.VMEM((M, 256), U8), "igrp": pltpu.VMEM((1, 4, 256), U8),
        "ipage": pltpu.VMEM((tiles.scan_depth, pg // 4, 4, 256), U8),
        "ip_sem": pltpu.SemaphoreType.DMA((tiles.scan_depth,)),
        "keys": pltpu.VMEM((ROWS, 128), I32),
        **({"tk_vt": pltpu.VMEM((ROWS, 256), BF16),
            "tk_acc": pltpu.VMEM((min(tiles.topk_kc, KT), 256), F32),
            "tk_c": pltpu.VMEM((2, 128, 128), F32)} if tiles.topk_blk else {}),
        "idx_s": pltpu.SMEM((NC, 128), I32),
        "idx_v": pltpu.VMEM((NC, 128), I32),
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        "g_nope": pltpu.VMEM((KT, 128), U32), "g_rope": pltpu.VMEM((KT, 128), U32),
        "g_sem": pltpu.SemaphoreType.DMA((2,)),
        "kvb": pltpu.VMEM((KT, cfg.KL), BF16),
        "s_all": pltpu.VMEM((NC, cfg.NH, 128), F32),
        # MoE
        "hrow": pltpu.VMEM((1, H), BF16), "xall": pltpu.VMEM((R, 1, H), BF16),
        "part": pltpu.VMEM((M, H), F32), "rsrecv": pltpu.VMEM((R, 1, H), F32),
        "psend": pltpu.VMEM((R, 1, H), F32),
        "shared": pltpu.VMEM((M, H), F32),
        # o_proj A2A: fp8 values carried as bf16 (exact), one (1, KR) row per peer
        "osend": pltpu.VMEM((R, 1, KR), BF16), "xrecv": pltpu.VMEM((R, 1, KR), BF16),
        "xrq": pltpu.VMEM((M, KR), FP8),
        "a2a_send": pltpu.SemaphoreType.DMA(()), "a2a_recv": pltpu.SemaphoreType.DMA(()),
        "wloc": pltpu.VMEM((M, 128), F32),
        # v9 q head-TP: AG q_lora rows, A2A (absorbed q | pre-rope q window) back to the owner
        "qsend": pltpu.VMEM((1, QW), F32), "qall": pltpu.VMEM((R, 1, QW), F32),
        "qasend": pltpu.VMEM((R, cfg.HR, QA), F32), "qarecv": pltpu.VMEM((R, cfg.HR, QA), F32),
        "qag_send": pltpu.SemaphoreType.DMA(()), "qag_recv": pltpu.SemaphoreType.DMA(()),
        "qa2a_send": pltpu.SemaphoreType.DMA(()), "qa2a_recv": pltpu.SemaphoreType.DMA(()),
        "hitv": pltpu.VMEM((8, 128), I32), "hits": pltpu.SMEM((8, 128), I32),
        "es13": pltpu.VMEM((cfg.ER, 2 * cfg.I), F32), "es2": pltpu.VMEM((cfg.ER, H), F32),
        "es_sem": pltpu.SemaphoreType.DMA((2,)),
        "w13s": pltpu.VMEM((1, 2 * cfg.ID), F32),
        "ag_send": pltpu.SemaphoreType.DMA(()), "ag_recv": pltpu.SemaphoreType.DMA(()),
        "rs_send": pltpu.SemaphoreType.DMA(()), "rs_recv": pltpu.SemaphoreType.DMA(()),
    }
    wshape = {
        "qkv_a": (tk["qkv_a"], rup(cfg.QKVA)), "q_b": (tk["q_b"], NQB),
        "o_proj": (tk["o_proj"], H), "idx_wq": (tk["idx_wq"], cfg.IH * cfg.IDIM),
        "idx_wk": (tk["idx_wk"], rup(cfg.IDIM + cfg.IH)), "gate": (tk["gate"], cfg.E),
        "sw13": (tk["sw13"], 2 * cfg.IS), "sw2": (tk["sw2"], H),
        "ew13": (tk["ew13"], 2 * cfg.I), "ew2": (tk["ew2"], H),
        "w13": (tk["w13"], 2 * cfg.ID), "w2": (tk["w2"], H),
        "w_uk": (GU, cfg.DN, cfg.KL), "w_uv": (G, cfg.KL, cfg.DV),
    }
    wdtype = {k: (BF16 if k in ("idx_wk", "gate", "w_uk", "w_uv") else FP8) for k in wshape}
    NB = tiles.nbank
    BROWS = tiles.bank // 128                          # u8 rows per bank
    for k, shp in wshape.items():
        assert pool_alias.storage_size(shp, jnp.dtype(wdtype[k])) <= tiles.bank, (k, shp)
    # pool_alias is a Mosaic lowering hack: interpret mode gets typed per-kind buffers with the
    # same slot schedule, and so does any kind whose lane dim is not whole 128-lane columns
    # (pool_alias lays out shape[-1]//128 columns; TINY's gate is (tk, E=8)).
    tkinds = [k for k, shp in wshape.items() if typed or shp[-1] % 128]
    for k in tkinds:
        scratch["ibuf_" + k] = pltpu.VMEM((NB,) + wshape[k], wdtype[k])
    if len(tkinds) < len(wshape):
        scratch["pool"] = pltpu.VMEM((NB * BROWS, 128), U8)
    scratch["pool_sem"] = pltpu.SemaphoreType.DMA((NB,))
    scratch["st"] = pltpu.SMEM((8,), I32)              # stream cursor, see `pump`
    scratch["gst"] = pltpu.SMEM((2,), I32)             # gather prefetch: rows issued, target layer
    GPRE = 0 if "gather" in skip else tiles.gpre
    NOIDX = [l for l in range(L) if not cfg.full_indexer(l)]
    assert GPRE == 0 or KT % GPRE == 0, (KT, GPRE)

    # --- the stream: one layer's tiles in consumption order, segment by segment ---
    SEGS = ("qkv_a", "q_b", "w_uk", "idx_wq", "idx_wk", "w_uv", "o_proj", "w13", "w2",
            "sw13", "sw2", "gate", "exp")
    if tiles.early_route:                              # probe 27 er: route before the shared w2
        SEGS = ("qkv_a", "q_b", "w_uk", "idx_wq", "idx_wk", "w_uv", "o_proj", "w13", "w2",
                "sw13", "gate", "sw2", "exp")
    SID = {k: i for i, k in enumerate(SEGS)}
    EXP = SID["exp"]
    KSEG = {k: (EXP if k in ("ew13", "ew2") else SID[k]) for k in wshape}

    def ntiles(kind):
        if kind == "w_uk":
            return cfg.HR // GU
        if kind == "w_uv":
            return cfg.NH // G
        return kdim(kind) // tk[kind]
    NTS = {k: ntiles(k) for k in SEGS if k != "exp"}
    T13, T2 = ntiles("ew13"), ntiles("ew2")
    TE = T13 + T2                                      # tiles per routed expert

    def bits(ks):
        return sum(1 << SID[k] for k in ks)
    M_BASE = bits(["qkv_a", "q_b", "o_proj"] + ([] if "uk_uv" in skip else ["w_uk", "w_uv"]))
    M_IDX = 0 if "index" in skip else bits(["idx_wq", "idx_wk"])
    M_DENSE, M_MOE = bits(["w13", "w2"]), bits(["sw13", "sw2", "gate", "exp"])

    def segs_py(l):
        m = M_BASE | (M_IDX if cfg.full_indexer(l) else 0) | (M_MOE if cfg.is_moe(l) else M_DENSE)
        return [k for k in SEGS if m >> SID[k] & 1]

    # Which segments can a pump at each consume site reach (<= NB tiles ahead)? Only those get DMA
    # code at that site (each is a when-cascade over the layers). Correctness does not depend on it:
    # `take` pumps its own kind if the producer stopped short of it.
    reach = {k: {SID[k]} for k in SEGS}
    reach["init"] = set()
    for nh in sorted({0, 1, cfg.ER}):
        seq = [SID[k] for l in range(L) for k in segs_py(l)
               for _ in range(nh * TE if k == "exp" else NTS[k])]
        reach["init"] |= set(seq[:NB])
        for p_, k in enumerate(seq):
            reach[SEGS[k]] |= set(seq[p_ + 1:p_ + 1 + NB])
    if "agdiag" in skip:
        scratch["agdiag"] = pltpu.VMEM((1, 128), F32)
    if debug:                                    # debug outputs live in HBM: VMEM is full
        scratch["dbg_sem"] = pltpu.SemaphoreType.DMA((2,))
    s_names = list(scratch)

    def kernel(*refs):
        meta, bt, h0, rope_tab, smalls = refs[:n_fixed]
        w_in = refs[n_fixed:n_fixed + n_w]
        c_in = refs[n_fixed + n_w:n_fixed + n_w + len(c_names)]
        out_ref, nrow_out, rrow_out, irow_out = refs[n_fixed + n_w + len(c_names):
                                                     n_fixed + n_w + len(c_names) + 4]
        rest = refs[n_fixed + n_w + len(c_names) + 4:]
        if debug:
            dbg_h, dbg_idx = rest[:2]
            rest = rest[2:]
        scr = dict(zip(s_names, rest))
        W = {}
        for (k, l), r in zip(w_names, w_in):
            W.setdefault(k, {})[l] = r
        C = {}
        for (k, l), r in zip(c_names, c_in):
            C.setdefault(k, {})[l] = r

        # axis_name: the rank from Mosaic's device id (upstream collectives32.py), not meta[0] -- the
        # served torch graph must not carry a rank constant (j-c4e6dce4), and XLA's partition-id is
        # rejected on that path (j-a28e43a5)
        me = meta[0] if axis_name is None else lax.axis_index(axis_name).astype(I32)
        pos = meta[1]
        seq_len = meta[2]
        npages = meta[3]
        active = seq_len > 0

        lane128 = lax.broadcasted_iota(I32, (1, 128), 1)
        row_iota = lax.broadcasted_iota(I32, (M, 1), 0)
        row0 = (row_iota == 0).astype(F32)            # [M, 1] mask of the live row

        # ---------------- weight streaming (one ring, one producer) ----------------
        st, psem = scr["st"], scr["pool_sem"]

        def src_of(kind, ref, t, j=None):
            if kind == "w_uk":
                return ref.at[pl.ds(t * GU, GU)]
            if kind == "w_uv":
                return ref.at[pl.ds(t * G, G)]
            if kind in ("ew13", "ew2"):
                return ref.at[j, pl.ds(t * tk[kind], tk[kind])]
            return ref.at[pl.ds(t * tk[kind], tk[kind])]

        def view(kind, slot):
            if kind in tkinds:
                return scr["ibuf_" + kind].at[slot]
            shp, dt = wshape[kind], jnp.dtype(wdtype[kind])
            return pool_alias.view(
                scr["pool"].at[pl.ds(slot * BROWS, pool_alias.storage_size(shp, dt) // 128)], shp, dt)

        def copy(kind, src, slot):
            return pltpu.make_async_copy(src, view(kind, slot), psem.at[slot])

        def seg_mask(l):
            full = jnp.logical_or(l < cfg.idx_skip, (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1)
            return (jnp.int32(M_BASE) | jnp.where(full, jnp.int32(M_IDX), 0)
                    | jnp.where(l >= cfg.first_dense, jnp.int32(M_MOE), jnp.int32(M_DENSE)))

        def nt_dyn(sg):
            v = st[6] * TE                             # exp: this layer's hits x tiles per expert
            for k, n in NTS.items():
                v = jnp.where(sg == SID[k], n, v)
            return v

        def issue_seg(a, l, t, slot):
            kind = SEGS[a]
            if kind != "exp":
                bswitch(l, {ll: (lambda r=r: copy(kind, src_of(kind, r, t), slot).start())
                            for ll, r in W[kind].items()})
                return
            i = t // TE
            u = t - i * TE
            j = scr["hits"][1, i]

            @pl.when(u < T13)
            def _():
                bswitch(l, {ll: (lambda r=r: copy("ew13", src_of("ew13", r, u, j), slot).start())
                            for ll, r in W["ew13"].items()})

            @pl.when(u >= T13)
            def _():
                bswitch(l, {ll: (lambda r=r: copy("ew2", src_of("ew2", r, u - T13, j), slot).start())
                            for ll, r in W["ew2"].items()})

        def pump(site, allowed=None):
            """Issue stream tiles while a bank is free. Cursor st: 0 issued, 1 consumed, 2 layer,
            3 segment, 4 tile in segment, 5 layer whose hit list is published, 6 its hit count.
            The producer stops at a layer's routed experts until routing publishes them."""
            if "weights" in skip:
                return
            allowed = sorted(reach[site] if allowed is None else allowed)
            amask = jnp.int32(sum(1 << a for a in allowed))

            def cond(c):
                l, sg = st[2], st[3]
                ok = jnp.logical_and(st[0] - st[1] < NB, l < L)
                ok = jnp.logical_and(ok, (lax.shift_right_logical(amask, sg) & 1) == 1)
                return jnp.logical_and(ok, jnp.logical_or(sg != EXP, st[5] == l))

            def body(c):
                l, sg, t = st[2], st[3], st[4]
                nt = nt_dyn(sg)

                @pl.when(t < nt)
                def _():
                    slot = st[0] % NB
                    bswitch(sg, {a: (lambda a=a: issue_seg(a, l, t, slot)) for a in allowed})
                    st[0] = st[0] + 1
                t1 = jnp.where(t < nt, t + 1, t)
                st[4] = t1

                @pl.when(t1 >= nt)
                def _():
                    rest = lax.shift_right_logical(seg_mask(l), sg + 1)
                    low = rest & (-rest)
                    has = rest != 0
                    st[3] = jnp.where(has, sg + 1 + (jnp.int32(31) - lax.clz(low)), 0)
                    st[2] = jnp.where(has, l, l + 1)
                    st[4] = 0
                return c + 1                           # a passthrough carry mis-lowers (scf.while)
            lax.while_loop(cond, body, jnp.int32(0))

        def take(kind, poll=False):
            """Wait for the next stream tile (must be of `kind`); its typed view. poll: while the
            tile is still landing, issue prefetched gather rows (gwait)."""
            if "weights" in skip:
                return view(kind, 0)

            @pl.when(st[0] <= st[1])                   # producer stopped short: pump this kind
            def _():
                pump(kind if kind in SID else "exp", allowed={KSEG[kind]})
            slot = st[1] % NB
            if poll and GPRE and not interpret:
                gst = scr["gst"]
                full = sem_full(kind)

                def pcond(c):
                    armed = jnp.logical_and(gst[1] >= 0, gst[0] < KT)
                    return jnp.logical_and(armed, pl.semaphore_read(psem.at[slot]) < full)

                def pbody(c):
                    gpump()
                    return c + 1
                lax.while_loop(pcond, pbody, jnp.int32(0))
            ref0 = next(iter(W[kind].values()))
            copy(kind, src_of(kind, ref0, 0, 0), slot).wait()
            return view(kind, slot)

        def give(kind):
            """Release the tile just used; refill from wherever the stream is."""
            st[1] = st[1] + 1
            pump(kind if kind in SID else "exp")

        def gather_rows(ref_n, ref_r, i0, n):
            """Issue top-k rows [i0, i0 + n) (n static) of one layer's nope / rope cache into
            g_nope / g_rope. Issue-bound, ~14 ns per row DMA: 59 -> 56 us per 2048 tokens at
            unroll 4 -> 16 (probe 07, j-9f61168b); the DMA count, not the address math, is the cost."""
            def gbody(k, c):
                i = i0 + k
                t = jnp.maximum(scr["idx_s"][i >> 7, i & 127], 0)
                page = bt[t >> lg_pg]
                sl = t & (pg - 1)
                pltpu.make_async_copy(ref_n.at[page, pl.ds(sl, 1)],
                                      scr["g_nope"].at[pl.ds(i, 1)], scr["g_sem"].at[0]).start()
                pltpu.make_async_copy(ref_r.at[page, pl.ds(sl >> RSH, 1)],
                                      scr["g_rope"].at[pl.ds(i, 1)], scr["g_sem"].at[1]).start()
                return c
            lax.fori_loop(0, n, gbody, 0, unroll=min(GATHER_UNROLL, n))

        def sem_full(kind):
            """DMA-semaphore value of a landed `kind` tile (probe 06: SEM_UNIT bytes per count)."""
            shp, dt = wshape[kind], jnp.dtype(wdtype[kind])
            return math.prod(shp) * dt.itemsize // SEM_UNIT

        def gpump():
            """Issue the next GPRE prefetched gather rows, if a layer is armed (gst[1] >= 0)."""
            if not GPRE:
                return
            gst = scr["gst"]

            @pl.when(jnp.logical_and(gst[1] >= 0, gst[0] < KT))
            def _():
                i0 = gst[0]
                bswitch(gst[1], {ll: (lambda ll=ll: gather_rows(C["nope"][ll], C["rope"][ll], i0,
                                                                 GPRE)) for ll in NOIDX})
                gst[0] = i0 + GPRE

        def matmul(kind, l, xref, K, N, j=None):
            """acc[:, :N] = xref[:, :K] @ W, W's tiles taken off the stream in order."""
            nt = ntiles(kind)
            t_k = tk[kind]
            acc = scr["acc"]
            acc[:, :N] = jnp.zeros((M, N), F32)

            def body(t, c):
                if not tiles.gwait:
                    gpump()
                w_ = take(kind, poll=tiles.gwait)
                k0 = pl.multiple_of(t * t_k, t_k)
                xs, ws = mxu(xref[:, pl.ds(k0, t_k)], w_[...], tiles)
                acc[:, :N] += lax.dot_general(xs, ws, NN, preferred_element_type=F32)
                give(kind)
                return c

            lax.fori_loop(0, nt, body, 0)
            return acc[:, :N]

        def sm(slot, name):
            o, n = lay[name]
            return scr["sm"][slot, :, o:o + n]

        # the stream starts before anything else (upstream Qwen: first fetches precede the barrier)
        for i_, v_ in enumerate((0, 0, 0, 0, 0, -1, 0, 0)):
            st[i_] = jnp.int32(v_)
        scr["gst"][0] = jnp.int32(0)
        scr["gst"][1] = jnp.int32(-1)
        pump("init")

        # ---------------- collectives ----------------
        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(R):
                pl.semaphore_signal(barrier, inc=1, device_id=(jnp.int32(d),),
                                    device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, R)

        if "agdiag" in skip and R > 1:
            # arrivals are counted from the source tags, not the semaphore: its per-row unit is what
            # this measures (raw values reported), so the drain waits exactly one desc per arrival
            scr["xall"][...] = jnp.zeros(scr["xall"].shape, BF16)
            scr["hrow"][...] = jnp.full((1, H), me + 1, I32).astype(F32).astype(BF16)
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=scr["hrow"], dst_ref=scr["xall"].at[me],
                        send_sem=scr["ag_send"], recv_sem=scr["ag_recv"],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()

            def arrived():
                n = jnp.int32(0)
                for r_ in range(R):
                    n = n + (jnp.max(scr["xall"][r_][:, 0:128].astype(F32)) > 0).astype(I32)
                return n

            # a remote row counts bytes/16 on the semaphores (j-480a968f: 64 per 1 KiB row at R=8), not
            # the local stream's bytes/32
            U = H * 2 // 16

            def cond(c):
                done = jnp.logical_and(c[1] >= R - 1, jnp.minimum(pl.semaphore_read(scr["ag_send"]),
                                                                 pl.semaphore_read(scr["ag_recv"])) >= (R - 1) * U)
                return jnp.logical_and(c[0] < AGDIAG_SPIN, jnp.logical_not(done))

            def body(c):
                return c[0] + 1, arrived()
            spins, n_arr = lax.while_loop(cond, body, (jnp.int32(0), arrived()))
            vr = pl.semaphore_read(scr["ag_recv"])
            vs = pl.semaphore_read(scr["ag_send"])
            desc = pltpu.make_async_remote_copy(
                src_ref=scr["hrow"], dst_ref=scr["xall"].at[0],
                send_sem=scr["ag_send"], recv_sem=scr["ag_recv"],
                device_id=(me,), device_id_type=pl.DeviceIdType.MESH)

            # DMA semaphores cannot be signalled (j-42a0905c), so a timeout drains exactly what landed:
            # whole rows, then the rest in 128-lane chunks. Anything landing after the read halts the
            # core at exit (nonzero semaphore, j-1cc2e439), which is itself the report.
            small = pltpu.make_async_remote_copy(
                src_ref=scr["hrow"].at[:, 0:128], dst_ref=scr["xall"].at[0, :, 0:128],
                send_sem=scr["ag_send"], recv_sem=scr["ag_recv"],
                device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
            US = 128 * 2 // 16

            def drain(v_, row_wait, small_wait):
                def w1(i, c):
                    row_wait()
                    return c

                def w2(i, c):
                    small_wait()
                    return c
                lax.fori_loop(0, v_ // U, w1, 0)
                lax.fori_loop(0, (v_ % U) // US, w2, 0)
            drain(vr, desc.wait_recv, small.wait_recv)
            drain(vs, desc.wait_send, small.wait_send)
            vals = (me, spins // 256, n_arr, vr % 256, (vr // 256) % 256, vr // 65536,
                    vs % 256, (vs // 256) % 256, vs // 65536)
            acc = jnp.zeros((1, 128), F32)
            for j_, v_ in enumerate(vals):
                acc = jnp.where(lane128 == j_, jnp.asarray(v_).astype(F32), acc)
            for r_ in range(R):
                acc = jnp.where(lane128 == AGDIAG_TAG0 + r_,
                                jnp.max(scr["xall"][r_][:, 0:128].astype(F32)), acc)
            scr["agdiag"][...] = acc

        def all_gather_rows():
            """hrow (1,H) bf16 on every rank -> xall[r] = rank r's row."""
            scr["xall"][me] = scr["hrow"][...]
            if R == 1 or "coll" in skip or "ag" in skip:
                return
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=scr["hrow"], dst_ref=scr["xall"].at[me],
                        send_sem=scr["ag_send"], recv_sem=scr["ag_recv"],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()

        def ag_wait():
            if R == 1 or "coll" in skip or "ag" in skip:
                return
            desc = pltpu.make_async_remote_copy(
                src_ref=scr["hrow"], dst_ref=scr["xall"].at[0],
                send_sem=scr["ag_send"], recv_sem=scr["ag_recv"],
                device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
            for _ in range(R - 1):
                desc.wait_recv()
            for _ in range(R - 1):
                desc.wait_send()

        def all_to_all_heads(row):
            """row (1, NH*DV) bf16 on every rank -> xrecv[src] on rank d = src's head slice d.
            No slot double-buffering (cf. upstream collectives32): a peer can only write xrecv
            again at layer l+1, after it received this rank's layer-l o_proj/MoE RS rows, which
            this rank sends only after it has consumed xrecv. Same argument for rsrecv."""
            for d in range(R):
                scr["osend"][d] = row[:, d * KR:(d + 1) * KR]
            scr["xrecv"][me] = scr["osend"][me]
            if R == 1 or "coll" in skip or "a2a" in skip:
                return
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=scr["osend"].at[d], dst_ref=scr["xrecv"].at[me],
                        send_sem=scr["a2a_send"], recv_sem=scr["a2a_recv"],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()
            desc = pltpu.make_async_remote_copy(
                src_ref=scr["osend"].at[0], dst_ref=scr["xrecv"].at[0],
                send_sem=scr["a2a_send"], recv_sem=scr["a2a_recv"],
                device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
            for _ in range(R - 1):
                desc.wait_recv()
            for _ in range(R - 1):
                desc.wait_send()

        def exchange(src_of_d, dst, ss, rs, wait=True):
            """v9: start src_of_d(d) -> dst[me] on every peer d; wait R-1 receives and sends.
            v10: wait=False returns the wait as a callable, so independent work (indexer, gather
            issue) overlaps the transfer; the caller must run it before the next collective.
            Race freedom as all_to_all_heads: a peer rewrites dst at layer l+1 only after this
            rank's next collective's rows, sent after dst was consumed (chain per layer:
            qall -> qarecv -> xrecv -> rsrecv -> xall -> rsrecv -> qall)."""
            if R == 1 or "coll" in skip or "qtp" in skip:
                return None if wait else (lambda: None)
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=src_of_d(d), dst_ref=dst.at[me], send_sem=scr[ss], recv_sem=scr[rs],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()

            def finish():
                desc = pltpu.make_async_remote_copy(
                    src_ref=src_of_d(0), dst_ref=dst.at[0], send_sem=scr[ss], recv_sem=scr[rs],
                    device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
                for _ in range(R - 1):
                    desc.wait_recv()
                for _ in range(R - 1):
                    desc.wait_send()
            if not wait:
                return finish
            finish()

        def reduce_scatter_rows():
            """part[t] (row per token/rank t) -> rsrecv[src] on owner t."""
            # rows are staged into a [R, 1, H] buffer: a single-row slice of a 2D VMEM ref at
            # an unaligned (or dynamic) offset is not a legal DMA source
            for d in range(R):
                scr["psend"][d] = scr["part"][d:d + 1, :]
            scr["rsrecv"][me] = scr["psend"][me]
            if R == 1 or "coll" in skip or "rs" in skip:
                return
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=scr["psend"].at[d], dst_ref=scr["rsrecv"].at[me],
                        send_sem=scr["rs_send"], recv_sem=scr["rs_recv"],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()
            desc = pltpu.make_async_remote_copy(
                src_ref=scr["psend"].at[0], dst_ref=scr["rsrecv"].at[0],
                send_sem=scr["rs_send"], recv_sem=scr["rs_recv"],
                device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
            for _ in range(R - 1):
                desc.wait_recv()
            for _ in range(R - 1):
                desc.wait_send()

        # ---------------- top-k (P3 design) ----------------
        def topk():
            K = KT
            idx_s, idx_v = scr["idx_s"], scr["idx_v"]

            @pl.when(seq_len <= K)
            def _fast():
                i = (lax.broadcasted_iota(I32, (NC, 128), 0) * 128
                     + lax.broadcasted_iota(I32, (NC, 128), 1))
                idx_v[...] = jnp.where(i < seq_len, i, -1)
                cp = pltpu.make_async_copy(idx_v, idx_s, scr["misc_sem"])
                cp.start()
                cp.wait()

            @pl.when(seq_len > K)
            def _slow():
                # the scan wrote order-preserving keys (INT_MIN = invalid); rows past its pages hold
                # stale keys from an earlier layer
                keys = scr["keys"]
                p_ = (lax.broadcasted_iota(I32, (ROWS, 128), 0) * 128
                      + lax.broadcasted_iota(I32, (ROWS, 128), 1))
                keys[...] = jnp.where(p_ >= seq_len, INT_MIN, keys[...])

                # threshold search, 2 bits per pass: the 3 candidates share one read of the keys and
                # counts fall with the candidate, so the digit is how many have cnt >= K
                # (topk_probe j-34587ead: 32 -> 16 passes, 35.8 -> 31.8 us)
                def pass_(i, p):
                    sh = jnp.int32(30) - 2 * i
                    kk_ = keys[...]
                    dig = jnp.int32(0)
                    for k in (1, 2, 3):
                        cs = (p | shl(jnp.int32(k), sh)) ^ INT_MIN
                        dig = dig + (jnp.sum((kk_ >= cs).astype(I32)) >= K).astype(I32)
                    return p | shl(dig, sh)

                p = lax.fori_loop(0, 16, pass_, jnp.int32(0))
                T = p ^ INT_MIN
                # vectorized extraction: counting prefix sums + one one-hot matmul per k chunk, no scalar
                # loop (topk_vec_probe j-5c210072, kc 512: thr+extract 90 -> 38 us rand, 106 -> 44 ties, exact)
                if tiles.topk_blk:
                    vtopk.extract_blk(keys, T, K, idx_v, scr["tk_vt"], scr["tk_acc"], scr["tk_c"],
                                      kc=tiles.topk_kc)
                else:
                    vtopk.extract(keys, T, K, idx_v, kc=tiles.topk_kc)
                cp = pltpu.make_async_copy(idx_v, idx_s, scr["misc_sem"])
                cp.start()
                cp.wait()

        KVA = tiles.kvw_async

        def kv_wait_writes():
            pltpu.make_async_copy(scr["wrow"], nrow_out.at[0], scr["kv_sem"].at[0]).wait()
            pltpu.make_async_copy(scr["rrow"], rrow_out.at[0], scr["kv_sem"].at[1]).wait()

        # ---------------- one layer ----------------
        def layer(l, carry):
            ph = Phases(tiles.trace)
            ph.mark("L_qkva")
            slot = l % 2
            is_moe = l >= cfg.first_dense
            full_idx = jnp.logical_or(
                l < cfg.idx_skip,
                jnp.logical_and(l >= cfg.idx_skip,
                                (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1))

            # smalls: this layer's already in flight; prefetch next
            pltpu.make_async_copy(smalls.at[0], scr["sm"].at[slot],
                                  scr["sm_sem"].at[slot]).wait()

            @pl.when(l + 1 < L)
            def _():
                pltpu.make_async_copy(smalls.at[l + 1], scr["sm"].at[1 - slot],
                                      scr["sm_sem"].at[1 - slot]).start()

            if KVA:
                # probe 27 kva: last layer's row writes landed long ago (wrow / rrow are free), and
                # this layer's rope row read flies under L_qkva
                @pl.when(active)
                def _():
                    @pl.when(l > 0)
                    def _():
                        kv_wait_writes()
                    page_ = bt[pos >> lg_pg]
                    sl_ = pos & (pg - 1)
                    bswitch(l, {ll: (lambda r=C["rope"][ll]: pltpu.make_async_copy(
                        r.at[page_, pl.ds(sl_ >> RSH, 1)], scr["rrow"].at[pl.ds(0, 1)],
                        scr["kv_sem"].at[2]).start()) for ll in range(L)})

            # input norm
            x = scr["h"][...] + scr["res"][...]
            x = x.astype(BF16).astype(F32)
            scr["res"][...] = x
            hn = rmsnorm(x, sm(slot, "in_ln"), cfg.eps) * row0
            scr["hn"][...] = hn

            # fused_qkv_a
            xq, sx = quant_rows(hn)
            scr["xq"][:, :H] = xq
            a = matmul("qkv_a", l, scr["xq"], H, rup(cfg.QKVA))
            o_s, _ = lay["qkv_a_s"]
            qkv_s = scr["sm"][slot, :, o_s:o_s + cfg.QKVA]
            qc_raw = (a[:, :cfg.QL] * sx * qkv_s[:, :cfg.QL]).astype(BF16).astype(F32)
            kv_raw = (a[:, cfg.QL:cfg.QL + cfg.KL] * sx
                      * qkv_s[:, cfg.QL:cfg.QL + cfg.KL]).astype(BF16).astype(F32)
            o_pe = cfg.QL + cfg.KL
            kpe_s = scr["sm"][slot, :, o_s + o_pe:o_s + o_pe + 128]
            kpe_raw = (scr["acc"][:, o_pe:o_pe + 128] * sx * kpe_s).astype(BF16).astype(F32)
            qc = rmsnorm(qc_raw, sm(slot, "q_a_ln"), cfg.eps) * row0
            kvc = rmsnorm(kv_raw, sm(slot, "kv_a_ln"), cfg.eps) * row0
            kpe = jnp.where(lane128 < cfg.DR,
                            rope(kpe_raw, rope_tab[2:3, 0:128], rope_tab[3:4, 0:128]), 0.0) * row0
            scr["qc"][...] = qc
            # v10: the q all-gather (L_qb below) starts here, so it flies under the KV write
            xq, sx = quant_rows(qc)
            scr["qsend"][...] = jnp.concatenate(
                [xq[0:1].astype(F32), jnp.where(lane128 == 0, sx[0:1], 0.0)], axis=1)
            scr["qall"][me] = scr["qsend"][...]
            qag_wait = exchange(lambda d: scr["qsend"], scr["qall"], "qag_send", "qag_recv",
                                wait=False)

            ph.mark("L_kvw")
            # KV write: nope row and rope row (RMW), SPARSECORE band packing. Caches are read-only
            # here (an aliased cache under a custom barrier costs XLA a full copy in and out); the
            # merged rows go to nrow_out / rrow_out and `step` scatters them after the call.
            @pl.when(active)
            def _():
                page = bt[pos >> lg_pg]
                sl = pos & (pg - 1)
                nb8 = fp8_bytes(jnp.clip(kvc, -FP8_MAX, FP8_MAX).astype(FP8))[0:1]
                word = (nb8[:, 0:128] | shl(nb8[:, 128:256], 8)
                        | shl(nb8[:, 256:384], 16) | shl(nb8[:, 384:512], 24))
                scr["wrow"][0:1, :] = lax.bitcast_convert_type(word, U32)
                rb = fp8_bytes(jnp.clip(kpe, -FP8_MAX, FP8_MAX).astype(FP8))[0:1]
                if rope_packed:   # word j = bytes (j, 16+j, 32+j, 48+j) of the 64 B key
                    rw = jnp.where(lane128 < 16, rb | shl(pltpu.roll(rb, 112, 1), 8)
                                   | shl(pltpu.roll(rb, 96, 1), 16)
                                   | shl(pltpu.roll(rb, 80, 1), 24), 0)
                else:             # word j = bytes (j, 32+j, 64+j, 96+j); 64..127 are zero
                    rw = jnp.where(lane128 < 32, rb | shl(pltpu.roll(rb, 96, 1), 8), 0)
                seg = sl & (RSEG - 1)

                cn = pltpu.make_async_copy(scr["wrow"], nrow_out.at[l], scr["kv_sem"].at[0])
                cn.start()

                def kvw(ref_r):
                    def f():
                        if KVA:
                            cr = pltpu.make_async_copy(ref_r.at[page, pl.ds(sl >> RSH, 1)],
                                                       scr["rrow"].at[pl.ds(0, 1)],
                                                       scr["kv_sem"].at[2])
                        else:
                            cr = pltpu.make_async_copy(ref_r.at[page, pl.ds(sl >> RSH, 1)],
                                                       scr["rrow"].at[pl.ds(0, 1)],
                                                       scr["kv_sem"].at[1])
                            cr.start()
                        cr.wait()
                        old = lax.bitcast_convert_type(scr["rrow"][0:1, :], I32)
                        new = old
                        for sg in range(RSEG):
                            new = jnp.where((seg == sg) & (lane128 // RW == sg),
                                            pltpu.roll(rw, RW * sg, 1) if sg else rw, new)
                        scr["rrow"][0:1, :] = lax.bitcast_convert_type(new, U32)
                    return f
                if KVA:                  # the read wait needs no layer switch (sem + bytes only)
                    kvw(C["rope"][0])()
                else:
                    bswitch(l, {ll: kvw(C["rope"][ll]) for ll in range(L)})
                cw = pltpu.make_async_copy(scr["rrow"], rrow_out.at[l], scr["kv_sem"].at[1])
                cw.start()
                if not KVA:              # kva: waited at the next layer's start / after the loop
                    cw.wait()
                    cn.wait()

            ph.mark("L_qb")
            # v9: q_b + W_UK absorb TP over heads (v3_tpa's q half). The owner quantizes its q_lora
            # row (served per-token scale); fp8 values (as f32, exact) and the scale are
            # all-gathered; every rank computes q for its HR heads for all R tokens, absorbs the
            # nope lanes through W_UK on the MXU, and A2As (absorbed q | pre-rope q lanes
            # DQ-128..DQ) back to each owner, which applies its own rope. Idle ranks send a zero
            # row (qc is masked to row 0 and zero when idle) and run every collective.
            qag_wait()
            ph.mark("L_qbmm")
            qa = [scr["qall"][t] for t in range(R)]                             # (1, QW) each
            scr["xq"][:, :cfg.QL] = jnp.concatenate(
                [r[:, :cfg.QL] for r in qa] + ([jnp.zeros((M - R, cfg.QL), F32)] if M > R else []),
                axis=0).astype(FP8)                                             # exact: fp8 values
            sxc = jnp.concatenate([r[:, cfg.QL:cfg.QL + 1] for r in qa]
                                  + ([jnp.ones((M - R, 1), F32)] if M > R else []), axis=0)  # [M, 1]
            a = matmul("q_b", l, scr["xq"], cfg.QL, NQB)
            o_qb, _ = lay["q_b_s"]
            qbs = scr["sm"][slot, :, pl.ds(pl.multiple_of(o_qb + me * NQB, 128), NQB)]
            q = (a * sxc * qbs).astype(BF16).astype(F32)                        # [M, NQB], row = token

            ph.mark("L_uk")
            # W_UK absorb (MXU): ql[t, h] = q_nope[t, h] @ W_UK_T[h]; bf16 operands (both exact),
            # f32 accumulation -- the served formula up to f32 summation order
            for g in range(cfg.HR // GU):
                wb = take("w_uk") if "uk_uv" not in skip else None
                for hh in range(GU):
                    j = g * GU + hh
                    qj = q[:, j * cfg.DQ:(j + 1) * cfg.DQ]                         # [M, DQ]
                    if wb is not None:
                        y = lax.dot_general(qj[:, 0:cfg.DN].astype(BF16), wb[hh], NN,
                                            preferred_element_type=F32).astype(BF16).astype(F32)
                    else:
                        y = jnp.zeros((M, cfg.KL), F32)
                    for t in range(R):
                        scr["qasend"][t, j:j + 1, 0:cfg.KL] = y[t:t + 1]
                        scr["qasend"][t, j:j + 1, cfg.KL:] = qj[t:t + 1, cfg.DQ - 128:]
                if wb is not None:
                    give("w_uk")
            ph.mark("L_qa2a")
            scr["qarecv"][me] = scr["qasend"][me]
            qa2a_wait = exchange(lambda d: scr["qasend"].at[d], scr["qarecv"], "qa2a_send",
                                 "qa2a_recv", wait=False)
            # v10: the return A2A is waited after L_index / L_gather (neither reads ql / qh), so the
            # transfer and the peers' skew hide under the indexer and the gather issue loop

            ph.mark("L_index")
            # indexer (full layers) -> idx_s / idx_v
            @pl.when(jnp.logical_and(full_idx, "index" not in skip))
            def _():
                phi = Phases(tiles.trace)
                phi.mark("i_proj")
                xq, sx = quant_rows(scr["qc"][...])
                scr["xq"][:, :cfg.QL] = xq
                a = matmul("idx_wq", l, scr["xq"], cfg.QL, cfg.IH * cfg.IDIM)
                iq = (a[0:1] * sx[0:1] * sm(slot, "idx_wq_s")).astype(BF16).astype(F32)
                scr["ih"][...] = jnp.zeros((M, cfg.IDIM), F32)
                for hh in range(cfg.IH):
                    scr["ih"][hh:hh + 1, :] = iq[:, hh * cfg.IDIM:(hh + 1) * cfg.IDIM]
                iqh = rope(scr["ih"][...], rope_tab[2:3, 0:128], rope_tab[3:4, 0:128])
                sq = jnp.max(jnp.abs(iqh), axis=1, keepdims=True) / FP8_MAX
                sq = jnp.where(sq == 0, 1.0, sq)
                iq8 = jnp.clip(iqh / sq, -FP8_MAX, FP8_MAX).astype(FP8)
                scr["xbf"][...] = scr["hn"][...].astype(BF16)
                NW = rup(cfg.IDIM + cfg.IH)
                a = matmul("idx_wk", l, scr["xbf"], H, NW)
                kw = scr["acc"][:, 0:256].astype(BF16).astype(F32)
                kx = kw[:, 0:128]
                mu = jnp.mean(kx, axis=1, keepdims=True)
                var = jnp.mean((kx - mu) ** 2, axis=1, keepdims=True)
                kn = ((kx - mu) * lax.rsqrt(var + 1e-6) * sm(slot, "idx_kn_w")
                      + sm(slot, "idx_kn_b")).astype(BF16).astype(F32)
                kr = rope(kn, rope_tab[2:3, 0:128], rope_tab[3:4, 0:128])
                wts = jnp.where(lane128 < cfg.IH, kw[0:1, 128:256], 0.0)             # [1,128]
                eye = (lax.broadcasted_iota(I32, (M, 128), 0)
                       == lax.broadcasted_iota(I32, (M, 128), 1))
                wcol = jnp.sum(jnp.where(eye, wts, 0.0), axis=1, keepdims=True)       # [M,1]
                wcol = (wcol * sq * (cfg.IDIM ** -0.5) * (cfg.IH ** -0.5)).astype(BF16).astype(F32)
                # ue8m0 k quant (exact exponent from the f32 bits)
                s = jnp.max(jnp.abs(kr[0:1]), axis=1, keepdims=True) / FP8_MAX       # [1,1]
                sb = lax.bitcast_convert_type(s, I32)
                e = (shr(sb, 23) & 0xFF) - 127
                e = jnp.where((sb & 0x7FFFFF) != 0, e + 1, e)
                e = jnp.clip(e, -126, 127)
                sc = lax.bitcast_convert_type(shl(e + 127, 23), F32)
                k8 = jnp.clip(kr / sc, -FP8_MAX, FP8_MAX).astype(FP8)                 # [M,128]
                scr["rec"][:, 0:128] = lax.bitcast_convert_type(k8, U8)
                eb = jnp.where((lax.broadcasted_iota(I32, (M, 128), 1) == 0), e + 127, 0)
                scr["rec"][:, 128:256] = eb.astype(U8)

                def scan(ref):
                    def f():
                        page = bt[pos >> lg_pg]
                        sl = pos & (pg - 1)
                        # u8 HBM is (4,128)-tiled on the token-in-group axis: merge the record into
                        # its 4-token group -> irow_out[l] (scattered by `step`); the scan patches
                        # the group into the page it reads, since the cache itself is stale.
                        grp = ref.at[page, pl.ds(sl >> 2, 1)]

                        @pl.when(active)
                        def _():
                            c = pltpu.make_async_copy(grp, scr["igrp"], scr["misc_sem"])
                            c.start()
                            c.wait()
                            g = scr["igrp"][0].astype(I32)
                            r0 = scr["rec"][0:8].astype(I32)[0:1]
                            row = lax.broadcasted_iota(I32, (4, 256), 0)
                            scr["igrp"][0] = jnp.where(row == (sl & 3), r0, g).astype(U8)
                            c = pltpu.make_async_copy(scr["igrp"], irow_out.at[pl.ds(l, 1)],
                                                      scr["misc_sem"])
                            c.start()
                            c.wait()

                        def pcopy(p, sl_):
                            return pltpu.make_async_copy(ref.at[bt[p]], scr["ipage"].at[sl_],
                                                         scr["ip_sem"].at[sl_])

                        SD = tiles.scan_depth
                        for q in range(SD - 1):
                            @pl.when(q < npages)
                            def _():
                                pcopy(q, q).start()

                        # scan_llo j-e39a2fd5/j-3d41a563: 79.5 -> 56-58 us/layer, keys bit-exact. The
                        # page is read as u32 (512,128): even rows are fp8 keys (no relayout), odd rows
                        # carry the ue8m0 byte, gathered to (8,128) token order; two-stage software
                        # pipeline (the matmul of page p overlaps the key math of page p-1).
                        NR, NCH = pg // 128, max(1, pg // 512)     # key rows per page, 128-lane chunks
                        assert pg % 256 == 0, pg
                        sub8 = lax.broadcasted_iota(I32, (NR, 128), 0)
                        ln8 = lax.broadcasted_iota(I32, (NR, 128), 1)

                        def stage1(p):
                            sl_ = p % SD

                            @pl.when(p + SD - 1 < npages)     # refills the slot freed last page
                            def _():
                                pcopy(p + SD - 1, (p + SD - 1) % SD).start()
                            pcopy(0, sl_).wait()

                            @pl.when(jnp.logical_and(active, p == pos >> lg_pg))
                            def _():
                                scr["ipage"][sl_, pl.ds(sl >> 2, 1)] = scr["igrp"][...]
                            if interpret:    # the interpreter has no raw views: same values, old load
                                blk = scr["ipage"][sl_].reshape(pg, 256)
                                keys8 = pltpu.bitcast(blk[:, 0:128], FP8)
                                dots = lax.dot_general(*mxu(iq8, keys8, tiles), NT,
                                                       preferred_element_type=F32)
                                return dots, blk[:, 128:129].astype(I32).reshape(NR, 128)
                            v = pool_alias.view(scr["ipage"].at[sl_], (pg // 2, 128), U32)
                            keys8 = pltpu.bitcast(pltpu.bitcast(v[pl.ds(0, pg // 4, stride=2), :], U8),
                                                  FP8)
                            dots = lax.dot_general(*mxu(iq8, keys8, tiles), NT,
                                                   preferred_element_type=F32)
                            # lane 0 of odd u32 row r packs the scale bytes of tokens 4r .. 4r+3
                            tw = lax.bitcast_convert_type(v[pl.ds(1, pg // 4, stride=2), :], I32).T[0:1]
                            if pg < 512:
                                tw = jnp.concatenate([tw, jnp.zeros((1, 128 - pg // 4), I32)], axis=1)
                            rr = jnp.broadcast_to(tw[:, 0:128], (NR, 128))
                            for ch in range(1, NCH):
                                rr = jnp.where(sub8 // 4 == ch,
                                               jnp.broadcast_to(tw[:, ch * 128:(ch + 1) * 128], (NR, 128)),
                                               rr)
                            g = jnp.take_along_axis(rr, 32 * (sub8 % 4) + ln8 // 4, axis=1,
                                                    mode="promise_in_bounds")
                            return dots, shr(g, 8 * (ln8 % 4)) & 255              # token 128 r + l

                        def stage2(p, dots, sbyte):
                            scv = jnp.sum(jnp.maximum(dots, 0.0) * wcol, axis=0, keepdims=True)
                            scv = jnp.concatenate([scv[:, j * 128:(j + 1) * 128] for j in range(NR)],
                                                  axis=0)
                            score = scv * lax.bitcast_convert_type(shl(sbyte, 23), F32)
                            pp = p * pg + 128 * sub8 + ln8
                            # order-preserving i32 key, written in place of the f32 score (one
                            # (ROWS, 128) buffer fewer: v11's top-k temporaries overflowed VMEM)
                            b = lax.bitcast_convert_type(score, I32)
                            key = jnp.where(b >= 0, b, b ^ jnp.int32(0x7FFFFFFF))
                            key = jnp.where((pp < seq_len) & (score != -jnp.inf), key, INT_MIN)
                            scr["keys"][pl.ds(pl.multiple_of(p * NR, NR), NR), :] = key

                        @pl.when(npages > 0)
                        def _():
                            carry = lax.fori_loop(
                                1, npages, lambda p, c: (stage1(p), stage2(p - 1, *c))[0], stage1(0))
                            stage2(npages - 1, *carry)
                    return f
                phi.mark("i_scan")
                bswitch(l, {ll: scan(r) for ll, r in C["idx"].items()})
                phi.mark("i_topk")
                if "topk" not in skip:
                    topk()
                phi.end()

            ph.mark("L_gather")
            # gather top-k rows of nope / rope
            if "gather" not in skip:
                # rows already issued by the prefetch (armed for this layer), then the rest
                def rest(ref_n, ref_r):
                    def f():
                        if not GPRE:
                            gather_rows(ref_n, ref_r, 0, KT)
                            return
                        c0 = jnp.where(scr["gst"][1] == l, scr["gst"][0], 0) // GPRE
                        lax.fori_loop(c0, KT // GPRE, lambda c, x: (gather_rows(
                            ref_n, ref_r, pl.multiple_of(c * GPRE, GPRE), GPRE), x)[1], 0)
                    return f
                # an idle rank (seq_len 0) gathers nothing: its KT row DMAs all hit one row and
                # were the slowest rank's gather in the one-busy proxy, stalling o_proj everywhere
                @pl.when(active)
                def _():
                    bswitch(l, {ll: rest(C["nope"][ll], C["rope"][ll]) for ll in range(L)})
                    pltpu.make_async_copy(scr["g_nope"], scr["g_nope"], scr["g_sem"].at[0]).wait()
                    pltpu.make_async_copy(scr["g_rope"], scr["g_rope"], scr["g_sem"].at[1]).wait()
                if GPRE:
                    scr["gst"][1] = jnp.int32(-1)
                # This step's row is not in HBM yet (caches are read-only here). top-k emits
                # ascending token ids and fills min(seq_len, KT) slots, and pos is the largest
                # valid token: if selected it is the last filled slot. Patch it from VMEM.
                il = jnp.maximum(jnp.minimum(seq_len, KT) - 1, 0)

                @pl.when(jnp.logical_and(active, scr["idx_s"][il >> 7, il & 127] == pos))
                def _():
                    scr["g_nope"][pl.ds(il, 1), :] = scr["wrow"][0:1, :]
                    scr["g_rope"][pl.ds(il, 1), :] = scr["rrow"][0:1, :]

            ph.mark("L_qa2a_w")
            qa2a_wait()
            ph.mark("L_qasm")
            for src in range(R):                     # owner: heads [src*HR, (src+1)*HR) of its q
                blk = scr["qarecv"][src]
                scr["ql"][src * cfg.HR:(src + 1) * cfg.HR, :] = blk[:, 0:cfg.KL]
                scr["qh"][src * cfg.HR:(src + 1) * cfg.HR, cfg.DQ - 128:] = blk[:, cfg.KL:]
            # rope on the window only: lane pairs are aligned (DQ - 128 even), and nope lanes are
            # rope-identity, so this equals v8's full-width rope on the lanes attention reads
            scr["qh"][:, cfg.DQ - 128:] = rope(scr["qh"][:, cfg.DQ - 128:],
                                               rope_tab[0:1, cfg.DQ - 128:cfg.DQ],
                                               rope_tab[1:2, cfg.DQ - 128:cfg.DQ])

            ph.mark("L_attn_s")
            # sparse MLA attention (served formula, two pass)
            qn = scr["ql"][...].astype(BF16)                                      # [NH,KL]
            xr = scr["qh"][:, cfg.DQ - 128:cfg.DQ]                                # rope in lanes 64..127
            lane_h = lax.broadcasted_iota(I32, (cfg.NH, 128), 1)
            q0 = jnp.where(lane_h < 32, pltpu.roll(xr, 64, 1), 0.0).astype(BF16)
            q1 = jnp.where(lane_h < 32, pltpu.roll(xr, 32, 1), 0.0).astype(BF16)
            R1 = tiles.rope1 and rope_packed
            if not R1:
                q0s = [q0] + [pltpu.roll(q0.astype(F32), 32 * sg, 1).astype(BF16) for sg in (1, 2, 3)]
                q1s = [q1] + [pltpu.roll(q1.astype(F32), 32 * sg, 1).astype(BF16) for sg in (1, 2, 3)]
            if tiles.hwdec:
                dec = fp8_decode_hw
            else:
                def dec(b_):
                    return fp8_decode(b_).astype(BF16)

            def s_body(c, carry):
                r0 = pl.multiple_of(c * 128, 128)
                w = lax.bitcast_convert_type(scr["g_nope"][pl.ds(r0, 128), :], I32)
                s = jnp.zeros((cfg.NH, 128), F32)
                for b in range(4):
                    band = dec(shr(w, 8 * b) & 0xFF)
                    scr["kvb"][pl.ds(r0, 128), b * 128:(b + 1) * 128] = band
                    s = s + lax.dot_general(qn[:, b * 128:(b + 1) * 128], band, NT,
                                            preferred_element_type=F32)
                rw_ = lax.bitcast_convert_type(scr["g_rope"][pl.ds(r0, 128), :], I32)
                iv = scr["idx_v"][pl.ds(c, 1), :]                                  # [1,128]
                if R1:
                    rw_ = packed_rope_to_seg0(rw_, iv)
                    B0 = dec(rw_ & 0xFF)
                    B1 = dec(shr(rw_, 8) & 0xFF)
                    s = s + (lax.dot_general(q0, B0, NT, preferred_element_type=F32)
                             + lax.dot_general(q1, B1, NT, preferred_element_type=F32))
                else:
                    if rope_packed:
                        rw_ = packed_rope_to_padded(rw_, iv)
                    B0 = dec(rw_ & 0xFF)
                    B1 = dec(shr(rw_, 8) & 0xFF)
                    seg = iv & 3
                    for sg in range(4):
                        ss = (lax.dot_general(q0s[sg], B0, NT, preferred_element_type=F32)
                              + lax.dot_general(q1s[sg], B1, NT, preferred_element_type=F32))
                        s = s + jnp.where(seg == sg, ss, 0.0)
                s = s * sm_scale
                s = jnp.where(iv >= 0, s, jnp.finfo(F32).min)
                scr["s_all"][c] = s
                return carry
            nc_att = jnp.where(active, 0 if "attn" in skip else NC, 0)
            lax.fori_loop(0, nc_att, s_body, 0)

            # g_nope / g_rope are consumed (kvb holds the decoded nope): arm the next layer's gather
            # when it reuses this top-k set
            if GPRE:
                nxt = l + 1
                reuse = jnp.logical_and(active & (nxt < L), jnp.logical_not(jnp.logical_or(
                    nxt < cfg.idx_skip, (nxt - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1)))

                @pl.when(reuse)
                def _():
                    scr["gst"][0] = jnp.int32(0)
                    scr["gst"][1] = nxt

            ph.mark("L_attn_pv")

            def m_body(c, m):
                return jnp.maximum(m, jnp.max(scr["s_all"][c], axis=1, keepdims=True))
            mx = lax.fori_loop(0, nc_att, m_body, jnp.full((cfg.NH, 1), -jnp.inf, F32))

            def pv_body(c, carry):
                lsum, pv = carry
                p = jnp.exp(scr["s_all"][c] - mx)
                r0 = pl.multiple_of(c * 128, 128)
                kv = scr["kvb"][pl.ds(r0, 128), :].astype(F32)
                return (lsum + jnp.sum(p, axis=1, keepdims=True),
                        pv + lax.dot_general(p, kv, NN, preferred_element_type=F32))
            lsum, pv = lax.fori_loop(0, nc_att, pv_body, (jnp.zeros((cfg.NH, 1), F32),
                                                     jnp.zeros((cfg.NH, cfg.KL), F32)))
            att = jnp.where(active, pv / jnp.maximum(lsum, 1e-30), 0.0).astype(BF16).astype(F32)
            if "attn" in skip:
                att = jnp.zeros_like(pv)
            for gg in range(cfg.NH // G):
                scr["att"][gg, 0:G, :] = att[gg * G:(gg + 1) * G]

            ph.mark("L_uv")
            # W_UV (VPU): o[h] = att[h] @ W_UV[h]
            scr["orow"][...] = jnp.zeros((M, cfg.NH * cfg.DV), F32)

            def uv_body(g, c):
                wb = take("w_uv")
                agT = scr["att"][g].T                                               # [KL, 8]
                for hh in range(G):
                    y = jnp.sum(agT[:, hh:hh + 1] * wb[hh].astype(F32), axis=0, keepdims=True)
                    o0 = pl.multiple_of((g * G + hh) * cfg.DV, 128)
                    scr["orow"][0:1, pl.ds(o0, cfg.DV)] = y.astype(BF16).astype(F32)
                give("w_uv")
                return c
            if "uk_uv" not in skip:
                lax.fori_loop(0, cfg.NH // G, uv_body, 0)

            ph.mark("L_oproj")
            # o_proj, TP over heads (K split). The owner quantizes its whole row (the served
            # per-token scale sx stays on the owner), rank d gets head slice d of every token and
            # multiplies them by its o_proj rows; the unscaled f32 partials reduce-scatter back and
            # the owner applies sx * o_proj_s once: the served formula up to f32 summation order.
            xq, sx = quant_rows(scr["orow"][...])
            all_to_all_heads(xq[0:1].astype(BF16))
            scr["xrq"][...] = jnp.concatenate(
                [scr["xrecv"][r] for r in range(R)] + [jnp.zeros((M - R, KR), BF16)], axis=0
            ).astype(FP8)
            scr["part"][...] = matmul("o_proj", l, scr["xrq"], KR, H)
            reduce_scatter_rows()
            a = scr["rsrecv"][0]
            for rr in range(1, R):
                a = a + scr["rsrecv"][rr]
            attn = (a * sx[0:1] * sm(slot, "o_proj_s")).astype(BF16).astype(F32)
            attn = jnp.concatenate([attn, jnp.zeros((M - 1, H), F32)], axis=0)

            # post-attention norm
            x = (attn + scr["res"][...]).astype(BF16).astype(F32)
            scr["res"][...] = x
            hn = rmsnorm(x, sm(slot, "post_ln"), cfg.eps) * row0
            scr["hn"][...] = hn

            def silu_mul(gu, n):
                g_, u_ = gu[:, :n], gu[:, n:2 * n]
                return (g_ * jax.nn.sigmoid(g_) * u_).astype(BF16).astype(F32)

            ph.mark("L_dense")

            @pl.when(jnp.logical_not(is_moe))
            def _dense():

                def ld(ref):
                    return lambda: pltpu.make_async_copy(ref, scr["w13s"], scr["misc_sem"]).start()
                bswitch(l, {ll: ld(r) for ll, r in W["w13_s"].items()})
                pltpu.make_async_copy(next(iter(W["w13_s"].values())), scr["w13s"],
                                      scr["misc_sem"]).wait()
                xq, sx = quant_rows(scr["hn"][...])
                scr["xq"][:, :H] = xq
                a = matmul("w13", l, scr["xq"], H, 2 * cfg.ID)
                mid = silu_mul((a * sx * scr["w13s"][...]).astype(BF16).astype(F32), cfg.ID) * row0
                xq, sx = quant_rows(mid)
                scr["xq"][:, :cfg.ID] = xq
                a = matmul("w2", l, scr["xq"], cfg.ID, H)
                scr["h"][...] = (a * sx * sm(slot, "w2_s")).astype(BF16).astype(F32) * row0

            ph.mark("L_moe")

            @pl.when(is_moe)
            def _moe():
                phm = Phases(tiles.trace)
                phm.mark("m_shared")
                def ldes(r13, r2):
                    def f():
                        pltpu.make_async_copy(r13, scr["es13"], scr["es_sem"].at[0]).start()
                        pltpu.make_async_copy(r2, scr["es2"], scr["es_sem"].at[1]).start()
                    return f
                bswitch(l, {ll: ldes(W["es13"][ll], W["es2"][ll]) for ll in W["es13"]})
                # an idle rank publishes a zero row: routing drops it (below), so its dummy token
                # no longer pulls experts onto the busy ranks' critical path
                scr["hrow"][...] = jnp.where(active, scr["hn"][0:1, :], 0.0).astype(BF16)
                all_gather_rows()

                # shared expert on the own token
                xq, sx = quant_rows(scr["hn"][...])
                scr["xq"][:, :H] = xq
                a = matmul("sw13", l, scr["xq"], H, 2 * cfg.IS)
                mid = silu_mul((a * sx * sm(slot, "sw13_s")).astype(BF16).astype(F32), cfg.IS) * row0
                xq, sx_sh = quant_rows(mid)
                scr["xq"][:, :cfg.IS] = xq

                def shared_w2():
                    a_ = matmul("sw2", l, scr["xq"], cfg.IS, H)
                    scr["shared"][...] = (a_ * sx_sh * sm(slot, "sw2_s")).astype(BF16).astype(F32)
                if not tiles.early_route:
                    shared_w2()

                phm.mark("m_agwait")
                ag_wait()
                xall = jnp.concatenate(
                    [scr["xall"][r].astype(F32) for r in range(R)] + [jnp.zeros((M - R, H), F32)], axis=0)
                scr["xbf"][...] = xall.astype(BF16)
                xqm, sxm = quant_rows(xall)
                scr["xqm"][...] = xqm

                phm.mark("m_router")
                # router: sigmoid, +bias, top-k (ties -> lowest id), renorm
                logits = matmul("gate", l, scr["xbf"], H, cfg.E)
                scv = jax.nn.sigmoid(logits)
                choice = scv + sm(slot, "gate_bias")
                eio = lax.broadcasted_iota(I32, (M, cfg.E), 1)
                wsel = jnp.zeros((M, cfg.E), F32)
                for _ in range(cfg.TOPK):
                    mxv = jnp.max(choice, axis=1, keepdims=True)
                    am = jnp.min(jnp.where(choice == mxv, eio, cfg.E), axis=1, keepdims=True)
                    sel = eio == am
                    wsel = jnp.where(sel, scv, wsel)
                    choice = jnp.where(sel, -jnp.inf, choice)
                wsum = jnp.sum(wsel, axis=1, keepdims=True)
                wn = (wsel / jnp.maximum(wsum, 1e-20)).astype(BF16).astype(F32)
                live = jnp.max(jnp.abs(xall), axis=1, keepdims=True) > 0
                valid_tok = jnp.logical_and(row_iota < R, live)
                wn = jnp.where(valid_tok, wn, 0.0)
                # local expert columns -> wloc lanes j, hit counts -> SMEM
                wloc = jnp.zeros((M, 128), F32)
                hitv = jnp.zeros((1, 128), I32)
                for j in range(cfg.ER):
                    e = me * cfg.ER + j
                    col = jnp.sum(jnp.where(eio == e, wn, 0.0), axis=1, keepdims=True)
                    cnt = jnp.sum(jnp.where(eio == e, (wsel > 0).astype(I32), 0) * valid_tok.astype(I32))
                    wloc = jnp.where(lane128 == j, col, wloc)
                    hitv = jnp.where(lane128 == j, cnt, hitv)
                scr["wloc"][...] = wloc
                scr["hitv"][0:1, :] = hitv
                cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
                cp.start()
                cp.wait()

                # compact hit list into hits[1, :]
                def cbody(j, n):
                    h_ = scr["hits"][0, j] > 0

                    @pl.when(h_)
                    def _():
                        scr["hits"][1, n] = j
                    return n + h_.astype(I32)
                n_hit = lax.fori_loop(0, cfg.ER, cbody, jnp.int32(0))
                if "experts" in skip:
                    n_hit = jnp.int32(0)
                # publish the hit list: the producer may now stream this layer's experts
                st[6] = n_hit
                st[5] = l
                pump("gate")
                if tiles.early_route:        # the routed tiles stream under the shared w2
                    phm.mark("m_sw2")
                    shared_w2()

                phm.mark("m_experts")
                pltpu.make_async_copy(next(iter(W["es13"].values())), scr["es13"],
                                      scr["es_sem"].at[0]).wait()
                pltpu.make_async_copy(next(iter(W["es2"].values())), scr["es2"],
                                      scr["es_sem"].at[1]).wait()
                scr["part"][...] = jnp.zeros((M, H), F32)

                def ebody(i, c):
                    j = scr["hits"][1, i]
                    a = matmul("ew13", l, scr["xqm"], H, 2 * cfg.I, j)
                    s13 = scr["es13"][pl.ds(j, 1), :]
                    mid = silu_mul(a * sxm * s13, cfg.I)
                    xq, sx = quant_rows(mid)
                    scr["xq"][:, :cfg.I] = xq
                    a = matmul("ew2", l, scr["xq"], cfg.I, H, j)
                    y = a * sx * scr["es2"][pl.ds(j, 1), :]
                    wcol = jnp.sum(jnp.where(lane128 == j, scr["wloc"][...], 0.0), axis=1,
                                   keepdims=True)
                    scr["part"][...] += y * wcol
                    return c
                lax.fori_loop(0, n_hit, ebody, 0)

                phm.mark("m_rs")
                reduce_scatter_rows()
                routed = scr["rsrecv"][0]
                for rr in range(1, R):
                    routed = routed + scr["rsrecv"][rr]
                routed = (routed.astype(BF16).astype(F32) * cfg.routed_scale).astype(BF16).astype(F32)
                out = (scr["shared"][0:1] + routed).astype(BF16).astype(F32)
                scr["h"][...] = jnp.concatenate([out, jnp.zeros((M - 1, H), F32)], axis=0)
                phm.end()

            ph.end()
            if debug:
                dcs = [pltpu.make_async_copy(scr["h"].at[pl.ds(0, 1)], dbg_h.at[pl.ds(2 * l, 1)],
                                             scr["dbg_sem"].at[0]),
                       pltpu.make_async_copy(scr["res"].at[pl.ds(0, 1)],
                                             dbg_h.at[pl.ds(2 * l + 1, 1)], scr["dbg_sem"].at[0]),
                       pltpu.make_async_copy(scr["idx_v"], dbg_idx.at[l], scr["dbg_sem"].at[1])]
                for d_ in dcs:
                    d_.start()
                for d_ in dcs:
                    d_.wait()
            return carry

        # ---------------- step ----------------
        pltpu.make_async_copy(smalls.at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        scr["h"][...] = h0[...].astype(F32) * row0
        scr["res"][...] = jnp.zeros((M, H), F32)
        scr["idx_v"][...] = jnp.full((NC, 128), -1, I32)
        if "weights" in skip:       # keep the ablated numerics finite (routing stays sane)
            for k in tkinds:
                scr["ibuf_" + k][...] = jnp.zeros(scr["ibuf_" + k].shape, wdtype[k])
            if len(tkinds) < len(wshape):
                scr["pool"][...] = jnp.zeros(scr["pool"].shape, U8)
        lax.fori_loop(0, L, layer, 0)
        if KVA:
            @pl.when(active)
            def _():
                kv_wait_writes()
        x = (scr["h"][...] + scr["res"][...]).astype(BF16).astype(F32)
        lay_o = smalls_layout(cfg)[0]
        # final norm weight rides in the last smalls row's "in_ln" slot of an extra row
        pltpu.make_async_copy(smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        pltpu.make_async_copy(smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]).wait()
        o, n = lay_o["in_ln"]
        out_ref[...] = rmsnorm(x, scr["sm"][0, :, o:o + n], cfg.eps).astype(BF16)
        if "agdiag" in skip and R > 1:
            out_ref[1:2, 0:128] = scr["agdiag"][...].astype(BF16)

    # ---------------- pallas_call ----------------
    in_specs = ([pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.SMEM),
                 pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.VMEM),
                 pl.BlockSpec(memory_space=pl.ANY)]
                + [pl.BlockSpec(memory_space=pl.ANY)] * (n_w + len(c_names)))
    out_shape = [jax.ShapeDtypeStruct((M, H), BF16), jax.ShapeDtypeStruct((L, 8, 128), U32),
                 jax.ShapeDtypeStruct((L, 8, 128), U32), jax.ShapeDtypeStruct((L, 4, 256), U8)]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [pl.BlockSpec(memory_space=pl.ANY)] * 3
    if debug:
        out_shape += [jax.ShapeDtypeStruct((2 * L, H), F32), jax.ShapeDtypeStruct((L, NC, 128), I32)]
        out_specs += [pl.BlockSpec(memory_space=pl.ANY)] * 2
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            # No has_side_effects: every output is consumed, and a side-effecting custom call
            # without a sharding is rejected by XLA's SPMD partitioner on the torch_tpu path.
            # No DMA bounds checks: each DMA start otherwise carries two ~7-bundle serial check
            # chains, which bound the 4096-DMA sparse gather (gather_llo j-050f4e3f: 86 -> 47 us per
            # layer, byte-identical). Every data-dependent index is clamped before it is used.
            collective_id=COLLECTIVE_ID if R > 1 else None, vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    def step(meta, bt, h0, rope_tab, smalls, *rest):
        """The kernel, then this token's cache rows scattered into the (donated) caches: a DUS on
        a donated buffer updates in place, where an alias through the kernel would copy it."""
        outs = call(meta, bt, h0, rope_tab, smalls, *rest)
        out, nrow, rrow, irow = outs[:4]
        pos, active = meta[1], meta[2] > 0
        page = bt[jnp.clip(pos >> lg_pg, 0, max_pages - 1)]
        sl = pos & (pg - 1)
        z = jnp.int32(0)
        new = []
        for (k, l), x in zip(c_names, rest[n_w:]):
            if k == "nope":
                row, at = nrow[l, 0:1][None], (page, sl, z)
            elif k == "rope":
                row, at = rrow[l, 0:1][None], (page, sl >> RSH, z)
            else:
                row, at = irow[l][None, None], (page, sl >> 2, z, z)
            old = lax.dynamic_slice(x, at, row.shape)
            new.append(lax.dynamic_update_slice(x, jnp.where(active, row, old), at))
        return (out, *new, *outs[4:])

    vmem_by = {k: int(np.prod(v.shape)) * jnp.dtype(v.dtype).itemsize for k, v in scratch.items()
               if str(getattr(v, "memory_space", "")) == "vmem"}
    vmem = sum(vmem_by.values())
    info = dict(s_names=s_names, w_names=w_names, c_names=c_names, tk=tk, S=S, ROWS=ROWS, wk=wk, ck=ck,
                vmem_scratch_bytes=vmem, vmem_by=vmem_by)
    return step, info
