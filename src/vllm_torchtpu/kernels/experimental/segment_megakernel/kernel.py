# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode segment megakernel: one pallas_call per MoE -> skip decoder boundary.

GLM-5.2 (DeepSeek-V3.2 MLA + MoE) at TP16 x DCP8, decode token buckets of at
most 32 rows. One call covers the tail of MoE layer N and the head of the
following skip layer N+1 (a layer with no indexer, reusing the shared top-k):

  entry: layer N's DCP LSE-merge all-to-all receive buffer, int16
         [dcp, 1 + 4T, v_head] (the TPU_DCP_LSE_MERGE_A2A_PACK +
         _REAL_ROWS layout: row 0 the lse words, rows 1.. the bf16 partials)
   -> online LSE merge with the -inf / NaN guards of `_merge_received`
   -> o_proj, fp8_skinny numerics (4 column tiles through 2 VMEM slots)
   -> AR1: attention TP all-reduce (moe_tp_decode's in-kernel all-reduce)
   -> post_attention_layernorm (fused add, `fused_add_rmsnorm.reference`)
   -> router as GateLinear: bf16 x bf16 -> bf16 logits, f32 sigmoid;
      top-k as argmax-and-mask
   -> moe_tp_decode's body: compaction, ring of `nb` expert slots, shared
      expert, combine
   -> AR2: FFN all-reduce (own scratch; the barrier AR1 waited covers it)
   -> input_layernorm of N+1 (fused add) -> residual exit
   -> fused q_a/kv_a (xla_quantized_matmul numerics, weight padded to a lane
      multiple at load) -> q_a_layernorm -> q_c exit
   -> kv_a_layernorm, k_pe NeoX rope, fp8, lane-dense pack -> the SparseCore
      gather's KV-write operand (`native_sc_gather.pack_write_rows` layout)

The weight stream is the pipelining axis: the data-independent weights
(router gate, shared expert, q_a/kv_a) are issued right after the last o_proj
tile, so they stream under the o_proj tail and AR1; the expert ring is primed
after routing.

Ported from the segment-megakernel campaign's rung 2b for skip boundaries
(close-gaps miliu_1016_curve, kernel_work/segment-mk/variants/rung2_segment.py:
rung 2, md5 daa29cb4, plus the L3 MXU expert-id compaction, WHITEBOARD.v6 md5
94d747cd; measured -7.7/-8.5/-6.8/-9.0/-3.8 us per skip boundary at
decode_first/decode_last/b2/b4/b8 against the incumbent, j-b722fa4c), skip
exit only, without the diagnostic arms and fault switches. Code reused from
this package's siblings:
  moe_tp_decode/kernel.py: `_route` constants, `_shared_expert`,
    `_quantize_blocks`, `_combine`, the expert body;
  moe_tp_decode/all_reduce.py: prologue / exchange / fold pieces;
  quantized_matmul/fp8_skinny.py:126-148: activation quant + bf16 x fp8 dot;
  quantized_matmul/util.py quantize_block; fused_add_rmsnorm.py reference.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce as MAR
from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as MK
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

F8 = jnp.float8_e4m3fn
BF16, F32, I32, U32 = jnp.bfloat16, jnp.float32, jnp.int32, jnp.uint32
FP8_MAX = 448.0
LANES = 128
MIB = 1 << 20
# Token rows of the KV-write operand are padded to whole SparseCore subcore
# rounds (native_sc_gather.NUM_SUBCORES).
KV_TOKEN_ROUND = 16
# Token buckets the kernel serves (the verified decode buckets; at most two
# subcore rounds of the gather's KV write, AR row buckets 16 / 32).
TOKEN_BUCKETS = (4, 8, 16, 32)
MAX_TOKENS = max(TOKEN_BUCKETS)


@dataclasses.dataclass(frozen=True)
class SegCfg:
    """Static shape and tuning of one segment call (one per token bucket)."""

    T: int  # token bucket: rows of every exit
    hidden: int = 6144
    tp: int = 16  # ranks of the in-kernel all-reduces
    dcp: int = 8  # DCP group size: partials merged at entry
    heads_rank: int = 4  # attention heads per rank after the merge
    v_head: int = 256
    q_lora: int = 2048
    kv_lora: int = 512
    rope: int = 64
    qkv_pad: int = 2688  # q_lora + kv_lora + rope (2624) padded to lanes at load
    experts: int = 256
    topk: int = 8
    moe_i: int = 128  # routed intermediate columns per rank
    shared_i: int = 128  # shared-expert intermediate columns per rank
    routed_scale: float = 2.5
    rms_eps: float = 1e-5
    nb: int = 6  # expert ring depth (6 measured best at T <= 32 on skip)
    o_tile: int = 1536
    o_slots: int = 2
    qkv_chunks: int = 4

    @property
    def m(self) -> int:  # moe_tp_decode row bucket of this token bucket
        return 16 if self.T <= 16 else 32

    @property
    def e_pad(self) -> int:
        return -(-self.experts // LANES) * LANES

    @property
    def R4(self) -> int:  # merge rows (t, h)
        return self.T * self.heads_rank

    @property
    def o_in(self) -> int:
        return self.heads_rank * self.v_head

    @property
    def tok_rows(self) -> int:  # KV-write token padding
        return -(-self.T // KV_TOKEN_ROUND) * KV_TOKEN_ROUND

    @property
    def a2a_rows(self) -> int:  # entry rows per source: lse words + partials
        return -(-2 * self.R4 // self.v_head) + self.R4

    def fields(self) -> tuple[tuple[str, int, int], ...]:
        """(name, offset, width) of each small f32 constant in one row of the
        all-layer constant stack `fstack` [boundaries, 1, row_width]."""
        widths = (
            ("oproj_s", self.hidden),
            ("g_post", self.hidden),
            ("g_in", self.hidden),
            ("g_qa", self.q_lora),
            ("g_kva", self.kv_lora),
            ("bias", self.e_pad),
            ("ss13", 2 * self.shared_i),
            ("ss2", self.hidden),
            ("qkv_s", self.qkv_pad),
        )
        out, off = [], 0
        for name, w in widths:
            out.append((name, off, w))
            off += w
        return tuple(out)

    @property
    def row_width(self) -> int:
        _, off, w = self.fields()[-1]
        return off + w

    def field(self, name: str) -> tuple[int, int]:
        for n, off, w in self.fields():
            if n == name:
                return off, w
        raise KeyError(name)


def supported(cfg: SegCfg) -> str | None:
    """None if the kernel serves `cfg`, else the reason it does not."""
    if cfg.T not in TOKEN_BUCKETS:
        return f"token bucket {cfg.T} not in {TOKEN_BUCKETS}"
    if cfg.a2a_rows != 1 + cfg.R4:
        return "the merge entry expects one lse row per source (2 * 4T <= v_head)"
    if cfg.e_pad != cfg.experts:
        return "the router gate must hold a lane multiple of experts"
    if (
        cfg.hidden % cfg.o_tile
        or cfg.hidden % cfg.qkv_chunks
        or cfg.hidden % MK._N_CHUNK
    ):
        return "hidden size does not tile"
    if cfg.qkv_pad % LANES or cfg.qkv_pad < cfg.q_lora + cfg.kv_lora + cfg.rope:
        return "q_a/kv_a padding is not a lane multiple covering q|kv|rope"
    if cfg.kv_lora != 4 * LANES or cfg.rope != 64:
        return "KV pack layout assumes kv_lora 512 and rope 64"
    if (cfg.q_lora + cfg.kv_lora) % LANES:
        return "the k_pe block must start on a lane boundary"
    if cfg.moe_i % LANES or cfg.shared_i % MK.PAIR:
        return "expert intermediate size does not tile"
    if not MAR.supported(cfg.tp, cfg.m, cfg.hidden, pltpu.get_tpu_info()):
        return "the in-kernel all-reduce does not support this geometry"
    if vmem_need_bytes(cfg)["limit"] > 64 * MIB:
        return "VMEM plan above 64 MiB"
    return None


# =====================================================================================
# all-reduce epilogue: moe_tp_decode/all_reduce.py epilogue pieces with the barrier
# wait optional (AR2 runs after AR1's exchange proved every peer is inside the call)
# =====================================================================================
def _exchange(spec, me, buf, rows, cols, s):
    cw = cols // spec.n

    def slot(q):
        return s["recv"].at[q, pl.ds(0, rows), pl.ds(0, cw)]

    def chunk(t):
        return MAR._cols(buf, t, cw)

    red = s["red"].at[pl.ds(0, rows), pl.ds(0, cw)]
    own = pltpu.make_async_copy(chunk(me), slot(me), s["loc"].at[0])
    own.start()
    rs = MAR._send_all(spec, me, chunk, slot, s["rs_send"], s["rs_recv"])
    own.wait()
    MAR._recv_all(spec, me, slot, s["rs_recv"])

    def load(q):
        return s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)]

    red[...] = MAR._sum_slots(load, spec.n).astype(red.dtype)
    own_ag = pltpu.make_async_copy(red, chunk(me), s["loc"].at[1])
    own_ag.start()
    ag = MAR._send_all(spec, me, lambda t: red, chunk, s["ag_send"], s["ag_recv"])
    own_ag.wait()
    MAR._recv_all(spec, me, chunk, s["ag_recv"])
    for copy in (*rs, *ag):
        copy.wait_send()


def _run_path(spec, kind, me, nv, partial, out_ref, s, barrier):
    m, h, tr = spec.m, spec.h, spec.tile_rows
    if kind == "fold":
        q = MAR._fold(spec, MAR._masked(partial, nv, tr, F32))
        s["q"][...] = q.astype(s["q"].dtype)
        buf, rows, cols = s["q"], tr, h // spec.lane_groups
    else:
        rows, cols = (m if kind == "full" else tr), h
        MAR._stage_rows(out_ref, partial, nv)
        buf = out_ref if rows == m else out_ref.at[pl.ds(0, rows)]
    if barrier is not None:
        pl.semaphore_wait(barrier, spec.n)
    _exchange(spec, me, buf, rows, cols, s)
    if kind == "fold":
        MAR._unfold(spec, s["q"][...].astype(F32), nv, out_ref)


def _ar_epilogue(spec, rank, n_valid, partial, out_ref, scr, *, wait_barrier):
    s = dict(zip(MAR._scratch_names(spec), scr, strict=True))
    barrier = pltpu.get_barrier_semaphore() if wait_barrier else None
    m, tr = spec.m, spec.tile_rows
    nv = jnp.minimum(jnp.maximum(n_valid, 0), m)

    @pl.when(nv == 0)
    def _zero():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        if barrier is not None:
            pl.semaphore_wait(barrier, spec.n)

    lo = 0
    if spec.fold:

        @pl.when((nv > 0) & (nv <= spec.fold_rows))
        def _fold_path():
            _run_path(spec, "fold", rank, nv, partial, out_ref, s, barrier)

        lo = spec.fold_rows
    if tr < m:

        @pl.when((nv > lo) & (nv <= tr))
        def _trim_path():
            _run_path(spec, "trim", rank, nv, partial, out_ref, s, barrier)

        lo = tr

    @pl.when(nv > lo)
    def _full_path():
        _run_path(spec, "full", rank, nv, partial, out_ref, s, barrier)


# =====================================================================================
# numerics helpers (each the incumbent op's formula)
# =====================================================================================
def rms_add(x32, res32, w, eps):
    """fused_add_rmsnorm.reference in f32 terms: s = x + res; new_res = bf16(s);
    y = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * w). Both roundings are what the
    served model computes: torch_tpu compiles with allow_excess_precision off,
    so vllm.ir's RMSNorm keeps the middle bf16 cast (jax.jit, with excess
    precision on, would fold it away -- not the production numerics)."""
    s = x32 if res32 is None else x32 + res32
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * lax.rsqrt(var + eps)).astype(BF16).astype(F32) * w.astype(F32)
    return y.astype(BF16), s


def quant_rows_skinny(x):
    """fp8_skinny.py:126-136: f32 scale = amax / 448, 0 -> 1; x_q =
    fp8(clip(x / scale)) held as bf16 (exact)."""
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(F32) * jnp.float32(
        1.0 / FP8_MAX
    )
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    xq = jnp.clip(x.astype(F32) * (jnp.float32(1.0) / scale), -FP8_MAX, FP8_MAX).astype(
        F8
    )
    return xq.astype(BF16), scale


def quant_rows_block(x_bf16):
    """util.quantize_block(x, axis=1, fp8) (xla_quantized_matmul's activation
    quant), the quantized x held as bf16 (exact), scale f32."""
    xq, sc = quantize_block(x_bf16, axis=1, target_dtype=F8)
    return xq.astype(BF16), sc.astype(F32)


def rope_lanes(x, cfull, sfull, half=32):
    """NeoX rope on lanes [0, 2*half) of each 128-lane block in bf16 semantics:
    o1 = bf16(bf16(x1 c) - bf16(x2 s)), o2 = bf16(bf16(x2 c) + bf16(x1 s));
    lanes >= 2*half pass through. cfull/sfull: per-lane cos/sin
    (cos[l % half] for l < 2*half)."""
    n = x.shape[-1]
    lane = lax.broadcasted_iota(I32, x.shape, 1) % LANES
    xf = x.astype(F32)
    partner = jnp.where(
        lane < half, pltpu.roll(xf, n - half, 1), pltpu.roll(xf, half, 1)
    )
    xc = (xf * cfull.astype(F32)).astype(BF16).astype(F32)
    xs = (partner * sfull.astype(F32)).astype(BF16).astype(F32)
    o = jnp.where(lane < half, xc - xs, xc + xs).astype(BF16)
    return jnp.where(lane < 2 * half, o, x.astype(BF16))


def fp8_code_u32(v):
    """e4m3fn byte of an fp8-representable f32 value (|v| <= 448) by 32-bit
    integer ops: normal -> ((e+7) << 3) | mant3; subnormal (|v| < 2^-6) ->
    |v| * 2^9; NaN -> 0x7f."""
    bits = pltpu.bitcast(v, I32)
    sign = (bits >> 31) & 1
    a = jnp.abs(v)
    ab = pltpu.bitcast(a, I32)
    e = ((ab >> 23) & 0xFF) - 127
    mant3 = (ab >> 20) & 0x7
    normal = ((e + 7) << 3) | mant3
    sub = (a * jnp.float32(512.0)).astype(I32)
    code = jnp.where(e >= -6, normal, sub)
    code = jnp.where(a == 0, 0, code)
    code = jnp.where(a != a, 0x7F, code)
    return pltpu.bitcast(code | (sign << 7), U32)


def router_scores(h, gate):
    """The production router up to the scores, [rows, E] f32: GateLinear on TPU
    is F.linear in bf16 (f32 accumulation, ONE rounding of the logits to bf16,
    kept because torch_tpu compiles with allow_excess_precision off), then
    `moe_routing._apply_scoring_fn` takes .float().sigmoid(). The bf16
    rounding is load-bearing: with e_score_correction_bias (mean ~4-10, std
    ~0.04 on GLM-5.2) f32 logits pick a different top-k set on 2-3% of rows
    per layer. The sigmoid is written 1 / (1 + exp(-x)): in Mosaic that is bit
    for bit XLA's jax.nn.sigmoid over every bf16 input, where Mosaic's own
    logistic differs on 4612 of them (by up to 1 f32 ulp)."""
    logits = lax.dot_general(h, gate, _NT, preferred_element_type=F32)
    x = logits.astype(BF16).astype(F32)
    return 1.0 / (1.0 + jnp.exp(-x))


def route_weights(h, gate, bias, n_valid, *, topk):
    """router_scores + _route_argmax: the kernel's routing table [rows, E] f32
    (exposed so the test can compare the top-k sets with production's)."""
    return _route_argmax(router_scores(h, gate), bias, n_valid, topk=topk)


def _route_argmax(scores, bias, n_valid, *, topk):
    """moe_tp_decode `_route` semantics (biased selection, unbiased weights
    renormalized over the picks, floor SUM_FLOOR, bf16 weights, rows >= n_valid
    zero) with the selection as lax.argmax + mask."""
    rows, n = scores.shape
    lane = lax.broadcasted_iota(I32, (rows, n), 1)
    masked = float(jnp.finfo(F32).min)
    x = scores + bias
    x = jnp.where(x > MK.ROUTER_NEG, x, MK.ROUTER_NEG)
    wsum = jnp.zeros((rows, 1), F32)
    for _ in range(topk):
        idx = jnp.argmax(x, axis=1, keepdims=True)
        pick = lane == idx
        wsum = wsum + jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
        x = jnp.where(pick, masked, x)
    chosen = x == masked
    denom = jnp.where(wsum < MK.SUM_FLOOR, MK.SUM_FLOOR, wsum)
    w = (jnp.where(chosen, scores, 0.0) / denom).astype(BF16)
    valid = lax.broadcasted_iota(I32, (rows, n), 0) < n_valid
    return jnp.where(jnp.logical_and(chosen, valid), w.astype(F32), 0.0)


def _compact_mxu(cfg, r):
    """Ascending ids of the hit experts without a scalar loop over experts:
    incl = hits . U (U upper-triangular ones) is each expert's inclusive rank;
    P[j, e] = hit[e] & (incl[e] - 1 == j); ids[j] = sum_e e P[j, e] (one
    nonzero term). Every operand is 0/1 or an integer <= 256, exact in bf16
    with f32 accumulation: the same ids in the same order as moe_tp_decode's
    branch-free compaction. Rung 2b of the campaign (lever L3, WHITEBOARD.v6,
    variants/rung2_segment.py `_compact_mxu`). Returns the hit count."""
    E = cfg.e_pad
    hit = r["hits_v"][...] != 0  # [1, E]
    hit8 = jnp.broadcast_to(jnp.where(hit, 1.0, 0.0), (8, E)).astype(BF16)
    U = _sel(E, E, lambda i, j: i <= j)
    incl = jnp.max(
        lax.dot_general(hit8, U, _NN, preferred_element_type=F32), axis=0, keepdims=True
    )  # [1, E]
    jj = lax.broadcasted_iota(I32, (E, E), 0)
    P = jnp.where((jj == (incl.astype(I32) - 1)) & hit, 1.0, 0.0).astype(BF16)
    er = lax.broadcasted_iota(I32, (8, E), 1)
    erow = jnp.where(lax.broadcasted_iota(I32, (8, E), 0) == 0, er, 0)
    erow = erow.astype(F32).astype(BF16)
    ids = lax.dot_general(erow, P, _NT, preferred_element_type=F32)  # [8, E]; row 0
    r["ids_v"][...] = ids.astype(I32)
    cnt = jnp.sum(jnp.where(hit, 1, 0), axis=1, keepdims=True)  # [1, 1]
    r["cnt_v"][...] = jnp.broadcast_to(cnt, (8, LANES)).astype(I32)
    a = pltpu.make_async_copy(r["ids_v"], r["ids_s"], r["hit_sem"].at[0])
    b = pltpu.make_async_copy(r["cnt_v"], r["cnt_s"], r["hit_sem"].at[1])
    a.start()
    b.start()
    a.wait()
    b.wait()
    return r["cnt_s"][0, 0]


def _sel(rows, cols, fn):
    """0/1 bf16 selection matrix S[i, j] = fn(i, j) (an exact row permutation
    on the MXU)."""
    i = lax.broadcasted_iota(I32, (rows, cols), 0)
    j = lax.broadcasted_iota(I32, (rows, cols), 1)
    return jnp.where(fn(i, j), 1.0, 0.0).astype(BF16)


_NN = (((1,), (0,)), ((), ()))
_NT = (((1,), (1,)), ((), ()))


# =====================================================================================
# operand tables (name order = pallas_call operand order)
# =====================================================================================
SMEM_IN = ("meta",)  # [n_valid, rank, boundary slot, 0 x5, KV dst rows x T]
META_DST = 8
# Taken in HBM / anywhere and copied to VMEM in the kernel. XLA places small
# custom-call operands in VMEM itself (memory space S(1)) with a copy each,
# and at T = 4 those copies ran serially in front of the call (j-7694e3a5,
# 16 of them, 12.7 us): so the per-step inputs come as two operands and the
# per-layer constants as rows of all-layer stacks too large to be placed.
STAGED_IN = ("a2a", "act")  # act = [res (T rows); cfull (m); sfull (m)], width H
HBM_IN = (
    "oproj_w",
    "gate",
    "w13",
    "w2",
    "s2",
    "qkv_w",
    "fstack",  # f32 [boundaries, 1, row_width]: SegCfg.fields() per boundary
    "s13s",  # f32 [boundaries, E, 1, 2I]: routed gate/up scales
    "sw13s",  # fp8 [boundaries, H, 2 Fs]
    "sw2s",  # fp8 [boundaries, Fs, H]
)
# fstack fields, each copied to VMEM
VEC_FIELDS = (
    "oproj_s",
    "g_post",
    "g_in",
    "g_qa",
    "g_kva",
    "bias",
    "ss13",
    "ss2",
    "qkv_s",
)
OUTS = ("res_out", "qc_out", "kvp_out")


def scratch_table(cfg: SegCfg):
    """[(name, shape-object)] in kernel order, and the all-reduce spec."""
    H, m, T = cfg.hidden, cfg.m, cfg.T
    nb, ic = cfg.nb, cfg.moe_i
    lanes = pltpu.get_tpu_info().num_lanes
    spec = MAR.make_spec(cfg.tp, m, H)
    sc = [
        ("o_tiles", pltpu.VMEM((cfg.o_slots, cfg.o_in, cfg.o_tile), F8)),
        ("gate_v", pltpu.VMEM((cfg.e_pad, H), BF16)),
        ("sw13_v", pltpu.VMEM((H, 2 * cfg.shared_i), F8)),
        ("sw2_v", pltpu.VMEM((cfg.shared_i, H), F8)),
        ("qkv_v", pltpu.VMEM((H, cfg.qkv_pad), F8)),
        ("w1_r", pltpu.VMEM((nb, H, 2 * ic), F8)),
        ("w2_r", pltpu.VMEM((nb, ic, H), F8)),
        ("s1_r", pltpu.VMEM((nb, 1, 2 * ic), F32)),
        ("s2_r", pltpu.VMEM((nb, 1, H), F32)),
        ("res_pad", pltpu.VMEM((m, H), F32)),
        ("y_pad", pltpu.VMEM((m, H), F32)),
        ("ar1_out", pltpu.VMEM((m, H), BF16)),
        ("ar2_out", pltpu.VMEM((m, H), BF16)),
        ("res1_v", pltpu.VMEM((m, H), F32)),
        ("h_v", pltpu.VMEM((m, H), BF16)),
        ("sxq", pltpu.VMEM((m, H), F8)),
        ("xq", pltpu.VMEM((m, H), F8)),
        ("xs", pltpu.VMEM((H // MK._Q_BLOCK, m, lanes), BF16)),
        ("owner", pltpu.VMEM((m, H), F32)),
        ("shared", pltpu.VMEM((m, H), F32)),
        ("wtab", pltpu.VMEM((m, cfg.e_pad), F32)),
        ("hits_v", pltpu.VMEM((1, cfg.e_pad), I32)),
        ("ids_v", pltpu.VMEM((8, cfg.e_pad), I32)),
        ("cnt_v", pltpu.VMEM((8, LANES), I32)),
        ("ids_s", pltpu.SMEM((8, cfg.e_pad), I32)),
        ("cnt_s", pltpu.SMEM((8, LANES), I32)),
        ("res_m", pltpu.VMEM((m, H), BF16)),
        ("res_x", pltpu.VMEM((T, H), BF16)),
        ("qc_m", pltpu.VMEM((m, cfg.q_lora), BF16)),
        ("qc_x", pltpu.VMEM((T, cfg.q_lora), BF16)),
        ("kvp_x", pltpu.VMEM((3 * cfg.tok_rows, LANES), U32)),
        ("in_sem", pltpu.SemaphoreType.DMA((len(STAGED_IN),))),
        ("vec_sem", pltpu.SemaphoreType.DMA((len(VEC_FIELDS),))),
        ("o_sem", pltpu.SemaphoreType.DMA((cfg.o_slots,))),
        ("w_sem", pltpu.SemaphoreType.DMA((1 + cfg.qkv_chunks,))),  # gate, qkv
        ("shared_sem", pltpu.SemaphoreType.DMA((2,))),
        ("ring_sem", pltpu.SemaphoreType.DMA((nb, 4))),
        ("hit_sem", pltpu.SemaphoreType.DMA((2,))),
        ("out_sem", pltpu.SemaphoreType.DMA((3,))),
    ]
    shapes = operand_shapes(cfg)
    sc += [(n, pltpu.VMEM(*shapes[n])) for n in STAGED_IN]
    sc += [(f"{n}_v", pltpu.VMEM((1, cfg.field(n)[1]), F32)) for n in VEC_FIELDS]
    sc += [(f"ar1_{i}", s) for i, s in enumerate(MAR.scratch_shapes(spec))]
    sc += [(f"ar2_{i}", s) for i, s in enumerate(MAR.scratch_shapes(spec))]
    return sc, spec


def vmem_need_bytes(cfg: SegCfg) -> dict:
    """Static VMEM sum of every buffer + the two AR epilogues' f32 temporaries
    + the VMEM inputs; the limit is need + 2 MiB."""
    info = pltpu.get_tpu_info()
    t = functools.partial(MAR.tiled_bytes, info=info)
    sc, spec = scratch_table(cfg)
    tot = 0
    for _, s in sc:
        ms = str(getattr(s, "memory_space", ""))
        if "VMEM" in ms.upper() and hasattr(s, "shape"):
            tot += t(tuple(s.shape), s.dtype)
    tot += 2 * (
        MAR.vmem_bytes(spec, info)
        - t((spec.n, spec.m, spec.chunk), BF16)
        - t((spec.m, spec.chunk), BF16)
    )
    return {"need": tot, "limit": tot + 2 * MIB}


# =====================================================================================
# the body
# =====================================================================================
def _segment_kernel(*refs, cfg: SegCfg, spec):
    sc_t, _ = scratch_table(cfg)
    names = (
        list(SMEM_IN)
        + [f"in_{n}" for n in STAGED_IN]
        + list(HBM_IN + OUTS)
        + [n for n, _ in sc_t]
    )
    assert len(names) == len(refs), (len(names), len(refs))
    r = dict(zip(names, refs, strict=True))
    n_ar = len(MAR.scratch_shapes(spec))
    ar1 = [r[f"ar1_{i}"] for i in range(n_ar)]
    ar2 = [r[f"ar2_{i}"] for i in range(n_ar)]

    H, T, m, nb = cfg.hidden, cfg.T, cfg.m, cfg.nb
    meta = r["meta"]
    n_valid = meta[0]
    rank = meta[1]
    slot_k = meta[2]  # this boundary's row of the constant stacks
    eps = jnp.float32(cfg.rms_eps)

    def frow(name):
        off, w = cfg.field(name)
        return r["fstack"].at[slot_k, :, pl.ds(off, w)]

    # ---- 1. barrier signals for BOTH in-kernel all-reduces (one prologue;
    #         AR2 skips its wait)
    MAR.prologue(spec)

    # ---- 2. o_proj weight stream: tiles 0..o_slots-1 + its scale
    n_ot = H // cfg.o_tile

    def o_copy(j):
        return pltpu.make_async_copy(
            r["oproj_w"].at[:, pl.ds(j * cfg.o_tile, cfg.o_tile)],
            r["o_tiles"].at[j % cfg.o_slots],
            r["o_sem"].at[j % cfg.o_slots],
        )

    for j in range(min(cfg.o_slots, n_ot)):
        o_copy(j).start()
    w_sem = r["w_sem"]
    vec = {}
    for i, n in enumerate(VEC_FIELDS):
        vec[n] = pltpu.make_async_copy(frow(n), r[f"{n}_v"], r["vec_sem"].at[i])
        vec[n].start()

    qc_rows = H // cfg.qkv_chunks
    pre = {}

    def issue_prefetch():
        """The data-independent weights: router gate, shared expert, q_a/kv_a."""
        gate = pltpu.make_async_copy(r["gate"], r["gate_v"], w_sem.at[0])
        gate.start()
        shared = []
        for i, (nm, dst) in enumerate((("sw13s", "sw13_v"), ("sw2s", "sw2_v"))):
            c = pltpu.make_async_copy(r[nm].at[slot_k], r[dst], r["shared_sem"].at[i])
            c.start()
            shared.append(c)
        qkv = []
        for c in range(cfg.qkv_chunks):
            cp = pltpu.make_async_copy(
                r["qkv_w"].at[pl.ds(c * qc_rows, qc_rows)],
                r["qkv_v"].at[pl.ds(c * qc_rows, qc_rows)],
                w_sem.at[1 + c],
            )
            cp.start()
            qkv.append(cp)
        pre.update(gate=gate, shared=shared, qkv=qkv)

    # ---- 3. entry: online LSE merge with guards -> attn [T, o_in] bf16
    staged = {}
    for i, n in enumerate(STAGED_IN):
        staged[n] = pltpu.make_async_copy(r[f"in_{n}"], r[n], r["in_sem"].at[i])
        staged[n].start()
    staged["a2a"].wait()
    attn = _entry_merge(cfg, r)

    # ---- 4. o_proj (fp8_skinny numerics), tile ring; prefetch issued after the
    #         last tile
    r["y_pad"][...] = jnp.zeros((m, H), F32)
    xq_o, xs_o = quant_rows_skinny(attn)
    for j in range(n_ot):
        o_copy(j).wait()
        if j == 0:
            vec["oproj_s"].wait()
        y = lax.dot_general(
            xq_o, r["o_tiles"][j % cfg.o_slots], _NN, preferred_element_type=F32
        )
        cols = pl.ds(j * cfg.o_tile, cfg.o_tile)
        yb = (y * xs_o * r["oproj_s_v"][:, cols]).astype(BF16)
        r["y_pad"][pl.ds(0, T), cols] = yb.astype(F32)
        if j + cfg.o_slots < n_ot:
            o_copy(j + cfg.o_slots).start()
            if j + cfg.o_slots == n_ot - 1:
                issue_prefetch()  # after the last o_proj tile is queued
    if not pre:
        issue_prefetch()

    # ---- 5. AR1 (attention TP all-reduce), barrier waited here
    _ar_epilogue(
        spec, rank, n_valid, r["y_pad"][...], r["ar1_out"], ar1, wait_barrier=True
    )

    # ---- 6. post_attention_layernorm(attn_sum, residual)
    r["res_pad"][...] = jnp.zeros((m, H), F32)
    staged["act"].wait()
    vec["g_post"].wait()
    r["res_pad"][pl.ds(0, T), :] = r["act"][pl.ds(0, T), :].astype(F32)
    h, s1 = rms_add(
        r["ar1_out"][...].astype(F32), r["res_pad"][...], r["g_post_v"][...], eps
    )
    r["res1_v"][...] = s1.astype(BF16).astype(F32)
    r["h_v"][...] = h

    # ---- 7. router: GateLinear's bf16 logits, f32 sigmoid
    pre["gate"].wait()
    scores = router_scores(r["h_v"][...], r["gate_v"][...])

    # ---- 8a. routing
    vec["bias"].wait()
    r["wtab"][...] = _route_argmax(scores, r["bias_v"][...], n_valid, topk=cfg.topk)
    hit = jnp.any(r["wtab"][...] != 0, axis=0, keepdims=True)
    r["hits_v"][...] = jnp.where(hit, 1, 0).astype(I32)
    ring_sem = r["ring_sem"]
    # ---- 8b. compaction (two exact 0/1 matmuls) + ring prime
    items = _compact_mxu(cfg, r)
    ids = r["ids_s"]

    def transfers(step):
        e = ids[0, step]
        slot = step % nb
        return [
            pltpu.make_async_copy(
                r["w13"].at[e], r["w1_r"].at[slot], ring_sem.at[slot, 0]
            ),
            pltpu.make_async_copy(
                r["w2"].at[e], r["w2_r"].at[slot], ring_sem.at[slot, 1]
            ),
            pltpu.make_async_copy(
                r["s13s"].at[slot_k, e],
                r["s1_r"].at[slot],
                ring_sem.at[slot, 2],
            ),
            pltpu.make_async_copy(
                r["s2"].at[e, 0], r["s2_r"].at[slot], ring_sem.at[slot, 3]
            ),
        ]

    for s_ in range(nb):

        @pl.when(jnp.int32(s_) < items)
        def _(s_=s_):
            for c in transfers(jnp.int32(s_)):
                c.start()

    # ---- 8c. shared expert + x quant
    for c in pre["shared"]:
        c.wait()
    vec["ss13"].wait()
    vec["ss2"].wait()
    x_bf = r["h_v"][...]
    r["owner"][...] = jnp.zeros((m, H), F32)
    sxq, sxs = quantize_block(x_bf, axis=1, target_dtype=F8)
    r["sxq"][...] = sxq
    MK._shared_expert(
        r["sxq"], sxs, r["sw13_v"], r["ss13_v"], r["sw2_v"], r["ss2_v"], r["shared"]
    )
    MK._quantize_blocks(x_bf, r["xq"], r["xs"])

    # ---- 8d. routed experts (ring)
    expert_columns = lax.broadcasted_iota(I32, r["wtab"].shape, 1)
    lanes = r["xs"].shape[-1]
    ic = cfg.moe_i

    def gate_up(slot):
        w_scale = r["s1_r"][slot].astype(BF16)
        acc = jnp.zeros((m, 2 * ic), BF16)
        for k in range(0, H, MK._Q_BLOCK):
            part = jnp.matmul(
                r["xq"][:, k : k + MK._Q_BLOCK],
                r["w1_r"][slot, k : k + MK._Q_BLOCK, :],
                preferred_element_type=F32,
            ).astype(BF16)
            part *= jnp.concatenate(
                [r["xs"][k // MK._Q_BLOCK]] * (2 * ic // lanes), axis=1
            )
            part *= w_scale
            acc += part
        return acc

    def finish(step, slot, acc1):
        e = ids[0, step]
        coeff = jnp.sum(
            jnp.where(expert_columns == e, r["wtab"][...], 0.0), axis=1, keepdims=True
        )
        act = jax.nn.silu(acc1[:, :ic]) * acc1[:, ic:]
        mid = jnp.where(coeff != 0, act.astype(F32) * coeff, 0.0)
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(F8)
        for n in range(0, H, MK._N_CHUNK):
            acc2 = jnp.matmul(
                mq, r["w2_r"][slot, :, n : n + MK._N_CHUNK], preferred_element_type=F32
            )
            r["owner"][:, n : n + MK._N_CHUNK] += (
                acc2 * r["s2_r"][slot, :, n : n + MK._N_CHUNK] * ms
            )

    @pl.when(items > 0)
    def _run_experts():
        for c in transfers(jnp.int32(0)):
            c.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < items)
            def _():
                for c in transfers(nxt):
                    c.wait()

            acc1_next = gate_up(nxt % nb)
            finish(step, step % nb, acc1)

            @pl.when(step + nb < items)
            def _():
                for c in transfers(step + nb):
                    c.start()

            return acc1_next

        lax.fori_loop(0, items, expert_body, gate_up(0))

    combined = MK._combine(r["owner"][...], r["shared"][...], cfg.routed_scale)

    # ---- 9. AR2 (FFN all-reduce), own scratch, barrier already consumed by AR1
    _ar_epilogue(spec, rank, n_valid, combined, r["ar2_out"], ar2, wait_barrier=False)

    # ---- 10. input_layernorm of layer N+1; residual exit (async)
    vec["g_in"].wait()
    h2, s2 = rms_add(
        r["ar2_out"][...].astype(F32), r["res1_v"][...], r["g_in_v"][...], eps
    )
    r["res_m"][...] = s2.astype(BF16)
    r["res_x"][...] = r["res_m"][pl.ds(0, T), :]
    out_sem = r["out_sem"]
    res_copy = pltpu.make_async_copy(r["res_x"], r["res_out"], out_sem.at[0])
    res_copy.start()

    # ---- 11. fused q_a/kv_a (xla_quantized_matmul numerics), K-chunked
    xq2, xs2 = quant_rows_block(h2)
    vec["qkv_s"].wait()
    acc = None
    for c in range(cfg.qkv_chunks):
        pre["qkv"][c].wait()
        part = lax.dot_general(
            xq2[:, c * qc_rows : (c + 1) * qc_rows],
            r["qkv_v"][pl.ds(c * qc_rows, qc_rows), :],
            _NN,
            preferred_element_type=F32,
        )
        acc = part if acc is None else acc + part
    qkv = ((acc * xs2) * r["qkv_s_v"][...]).astype(BF16)  # [m, qkv_pad]
    q_pre = qkv[:, : cfg.q_lora]
    kv_c = qkv[:, cfg.q_lora : cfg.q_lora + cfg.kv_lora]
    k0 = cfg.q_lora + cfg.kv_lora
    pe_blk = qkv[:, k0 : k0 + LANES]  # k_pe in lanes 0..63

    # ---- 12. q_a_layernorm -> q_c exit (async); KV pack exit
    vec["g_qa"].wait()
    vec["g_kva"].wait()
    q_c, _ = rms_add(q_pre.astype(F32), None, r["g_qa_v"][...], eps)
    r["qc_m"][...] = q_c
    r["qc_x"][...] = r["qc_m"][pl.ds(0, T), :]
    qc_copy = pltpu.make_async_copy(r["qc_x"], r["qc_out"], out_sem.at[1])
    qc_copy.start()
    cfull = r["act"][pl.ds(T, m), pl.ds(0, LANES)]
    sfull = r["act"][pl.ds(T + m, m), pl.ds(0, LANES)]
    _skip_exit(cfg, r, kv_c, pe_blk, cfull, sfull, eps)
    kvp_copy = pltpu.make_async_copy(r["kvp_x"], r["kvp_out"], out_sem.at[2])
    kvp_copy.start()
    for c in (res_copy, qc_copy, kvp_copy):
        c.wait()


def _entry_merge(cfg: SegCfg, r):
    """Per (t, h) row, online max / denominator / accumulator over the dcp
    received partials, with the guards that make it equal `_zero_nan` +
    `_merge_received` (non-finite lse -> weight 0; NaN partial -> 0), then the
    [4T, v_head] result placed as attn[t, h*v:(h+1)*v] by exact 0/1 selection
    matmuls."""
    R, R4, T = cfg.dcp, cfg.R4, cfg.T
    m_run = None
    den = acc = None
    for src in range(R):
        # row 0 of the packed block = lse words (hi halves at lanes [0, R4), lo at
        # [R4, 2 R4)); rows 1..R4 = the bf16 partial's bits.
        head = r["a2a"][src, pl.ds(0, 8), :].astype(I32)  # [8, v_head]
        words = jnp.transpose(head)  # [v_head, 8]; column 0 = the lse words
        lane8 = lax.broadcasted_iota(I32, (R4, 8), 1)
        hi = jnp.sum(jnp.where(lane8 == 0, words[0:R4], 0), axis=1, keepdims=True)
        lo = jnp.sum(
            jnp.where(lane8 == 0, words[R4 : 2 * R4], 0), axis=1, keepdims=True
        )
        lse = pltpu.bitcast((hi << 16) | (lo & 0xFFFF), F32)  # [R4, 1]
        o = pltpu.bitcast(
            r["a2a"][src, pl.ds(1, R4), :].astype(I32) << 16, F32
        )  # bf16 -> f32
        fin = jnp.isfinite(lse)
        lse = jnp.where(fin, lse, -jnp.inf)
        o = jnp.where(jnp.isnan(o), 0.0, o)
        if m_run is None:
            m_new = lse
            m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
            p = jnp.where(fin, jnp.exp(lse - m_safe), 0.0)
            den = p
            acc = p * o
        else:
            m_new = jnp.maximum(m_run, lse)
            m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
            alpha = jnp.where(jnp.isfinite(m_run), jnp.exp(m_run - m_safe), 0.0)
            p = jnp.where(fin, jnp.exp(lse - m_safe), 0.0)
            den = den * alpha + p
            acc = acc * alpha + p * o
        m_run = m_new
    merged = (acc / jnp.where(den > 0, den, 1.0)).astype(BF16)  # [R4, v_head]
    heads = []
    for hh in range(cfg.heads_rank):
        P = _sel(T, R4, lambda i, j, hh=hh: j == i * cfg.heads_rank + hh)  # [T, R4]
        heads.append(
            lax.dot_general(P, merged, _NN, preferred_element_type=F32).astype(BF16)
        )
    return jnp.concatenate(heads, axis=1)  # [T, o_in]


def _skip_exit(cfg, r, kv_c, pe_blk, cfull, sfull, eps):
    """kv_a_layernorm, k_pe NeoX rope, clamp-convert to fp8, `_pack_lane_dense`
    words, `pack_write_rows` layout u32[3 Tp, 128] (rows >= T zero; the
    destination rows come in meta)."""
    T, m, tp = cfg.T, cfg.m, cfg.tok_rows
    kv_n, _ = rms_add(kv_c.astype(F32), None, r["g_kva_v"][...], eps)
    pe_r = rope_lanes(pe_blk, cfull, sfull)
    row = lax.broadcasted_iota(I32, (m, LANES), 0)
    lane = lax.broadcasted_iota(I32, (m, LANES), 1)
    kv8 = jnp.clip(kv_n.astype(F32), -FP8_MAX, FP8_MAX).astype(F8).astype(F32)
    b = fp8_code_u32(kv8)  # [m, 512]
    nope = (
        b[:, 0:128]
        | (b[:, 128:256] << 8)
        | (b[:, 256:384] << 16)
        | (b[:, 384:512] << 24)
    )
    pe8 = jnp.clip(pe_r.astype(F32), -FP8_MAX, FP8_MAX).astype(F8).astype(F32)
    pb = fp8_code_u32(pe8)  # [m, 128], lanes 0..63 valid
    ropew = jnp.where(
        lane < 32, pb | (pltpu.roll(pb, LANES - 32, 1) << 8), jnp.uint32(0)
    )
    live = row < T
    nope = jnp.where(live, nope, jnp.uint32(0))
    ropew = jnp.where(live, ropew, jnp.uint32(0))
    kvp = r["kvp_x"]
    kvp[pl.ds(0, tp), :] = nope[:tp]
    kvp[pl.ds(tp, tp), :] = ropew[:tp]
    # The gather's destination block: enc = dst + 1 (0 = skip) on all lanes.
    rows = lax.broadcasted_iota(I32, (tp, LANES), 0)
    enc = jnp.zeros((tp, LANES), I32)
    for t in range(T):
        d = r["meta"][META_DST + t]
        enc = jnp.where(rows == t, jnp.where(d >= 0, d + 1, 0), enc)
    kvp[pl.ds(2 * tp, tp), :] = pltpu.bitcast(enc, U32)


# =====================================================================================
# wrapper
# =====================================================================================
def kernel_name(cfg: SegCfg) -> str:
    return f"segment_megakernel_skip-T_{cfg.T}-m_{cfg.m}-nb_{cfg.nb}-e_{cfg.experts}"


def segment_call(cfg: SegCfg, ops: dict, *, interpret=None) -> dict:
    """One boundary on one device. `ops`: name -> array for every name of
    SMEM_IN + STAGED_IN + HBM_IN, shapes as `operand_shapes(cfg)`. Returns
    {res_out bf16 [T, H], qc_out bf16 [T, q_lora], kvp_out u32 [3 Tp, 128]}."""
    reason = supported(cfg)
    if reason is not None:
        raise ValueError(f"segment megakernel: {reason}")
    sc_t, spec = scratch_table(cfg)
    T, H = cfg.T, cfg.hidden
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)  # moe_tp_decode kernel.py form
    anyspace = pl.BlockSpec(memory_space=pl.ANY)  # outputs written by async DMA
    out_shapes = (
        jax.ShapeDtypeStruct((T, H), BF16),
        jax.ShapeDtypeStruct((T, cfg.q_lora), BF16),
        jax.ShapeDtypeStruct((3 * cfg.tok_rows, LANES), U32),
    )
    limit = vmem_need_bytes(cfg)["limit"]
    call = pl.pallas_call(
        functools.partial(_segment_kernel, cfg=cfg, spec=spec),
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(SMEM_IN),
            in_specs=[anyspace] * len(STAGED_IN) + [hbm] * len(HBM_IN),
            out_specs=tuple([anyspace] * len(out_shapes)),
            scratch_shapes=[s for _, s in sc_t],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(limit), collective_id=MAR.COLLECTIVE_ID
        ),
        name=kernel_name(cfg),
        **({} if interpret is None else {"interpret": interpret}),
    )
    outs = call(*[ops[n] for n in SMEM_IN + STAGED_IN + HBM_IN])
    return dict(zip(OUTS, outs, strict=True))


def operand_shapes(cfg: SegCfg, boundaries: int = 1) -> dict:
    """name -> (shape, dtype) the kernel takes (per device); `boundaries`
    sizes the constant stacks."""
    H, T, m = cfg.hidden, cfg.T, cfg.m
    return {
        "meta": ((META_DST + T,), I32),
        "a2a": ((cfg.dcp, 1 + cfg.R4, cfg.v_head), jnp.int16),
        "act": ((T + 2 * m, H), BF16),
        "oproj_w": ((cfg.o_in, H), F8),
        "gate": ((cfg.e_pad, H), BF16),
        "w13": ((cfg.experts, H, 2 * cfg.moe_i), F8),
        "w2": ((cfg.experts, cfg.moe_i, H), F8),
        "s2": ((cfg.experts, 1, 1, H), F32),
        "qkv_w": ((H, cfg.qkv_pad), F8),
        "fstack": ((boundaries, 1, cfg.row_width), F32),
        "s13s": ((boundaries, cfg.experts, 1, 2 * cfg.moe_i), F32),
        "sw13s": ((boundaries, H, 2 * cfg.shared_i), F8),
        "sw2s": ((boundaries, cfg.shared_i, H), F8),
    }


def pack_act(res, cfull, sfull):
    """The `act` operand: [res (T rows); cfull (m rows); sfull (m rows)], the
    rope tables zero-padded to the residual's width."""
    H = res.shape[1]
    pad = lambda x: jnp.pad(x, ((0, 0), (0, H - x.shape[1])))  # noqa: E731
    return jnp.concatenate([res, pad(cfull), pad(sfull)], axis=0)


def pack_meta(n_valid, rank, slot, dst_rows):
    """The SMEM `meta` operand."""
    head = jnp.stack([jnp.asarray(x, I32).reshape(()) for x in (n_valid, rank, slot)])
    return jnp.concatenate(
        [head, jnp.zeros((META_DST - 3,), I32), dst_rows.astype(I32)]
    )


def fstack_row(cfg: SegCfg, consts: dict):
    """One boundary's `fstack` row (f32 [row_width]) from its constants:
    oproj_s, g_post, g_in, g_qa, g_kva, bias, ss13, ss2, qkv_s (padded to
    qkv_pad)."""
    parts = []
    for name, _, w in cfg.fields():
        v = consts[name].astype(F32).reshape(-1)
        parts.append(jnp.pad(v, (0, w - v.shape[0])))
    return jnp.concatenate(parts)


def rope_lane_tables(cos, sin, m):
    """[T, rope/2] cos / sin rows -> the kernel's per-lane [m, 128] tables
    (cos[l % 32] on lanes 0..63, zero above; rows >= T zero)."""
    T, half = cos.shape
    pad = jnp.zeros((T, LANES - 2 * half), cos.dtype)
    cs = jnp.concatenate([cos, cos, pad], axis=1)
    sn = jnp.concatenate([sin, sin, pad], axis=1)
    return jnp.pad(cs, ((0, m - T), (0, 0))), jnp.pad(sn, ((0, m - T), (0, 0)))


def kv_dst_block(dst_rows, tok_rows):
    """`pack_write_rows`' destination block: enc = dst + 1 (0 = skip) on all 128
    lanes, rows >= T zero."""
    T = dst_rows.shape[0]
    enc = jnp.where(dst_rows >= 0, dst_rows + 1, 0).astype(U32)
    return jnp.pad(
        jnp.broadcast_to(enc[:, None], (T, LANES)), ((0, tok_rows - T), (0, 0))
    )
