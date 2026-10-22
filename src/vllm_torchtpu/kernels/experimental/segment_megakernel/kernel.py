# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode segment megakernel: one pallas_call per MoE -> skip decoder boundary.

GLM-5.2 (DeepSeek-V3.2 MLA + MoE) at TP16 x DCP8, decode token buckets of at
most 32 rows. One call covers the tail of MoE layer N and the head of the
following skip layer N+1 (a layer with no indexer, reusing the shared top-k):

  entry: layer N's merged DCP attention output in latent space, bf16
         [T, heads * kv_lora] (what the send-layout / fused small all-to-all
         MLA ops return after their LSE merge, W_UV not yet applied)
   -> W_UV per head, as the backend's torch.bmm: bf16 x bf16, f32
      accumulation, one rounding to bf16
   -> o_proj, fp8_skinny numerics (4 column tiles through 2 VMEM slots; the
      fused o_proj + all-reduce kernel's numerics too)
   -> AR1: attention TP all-reduce (moe_tp_decode's in-kernel all-reduce)
   -> post_attention_layernorm (fused add, `fused_add_rmsnorm.reference`)
   -> router as GateLinear: bf16 x bf16 -> bf16 logits, f32 sigmoid;
      top-k as argmax-and-mask
   -> moe_tp_decode's body with its numerics: per-row fp8 x over the full
      K, compaction, ring of `nb` expert slots, its shared expert, combine
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
exit only, without the diagnostic arms and fault switches; then carried onto
the trial-6 stack (kernel_work/decode-mk/port-t6/opmap.md): the entry moved
from the packed merge receive buffer to the merged latent + W_UV (that stack
merges in latent space, inside its MLA kernel at b1), and the MoE body takes
moe_tp_decode's per-row full-K quantization and f32 expert arithmetic. Code
reused from this package's siblings:
  moe_tp_decode/kernel.py: `_route` constants, `_shared_expert`, `_combine`,
    the routed expert body (gate_up / finish);
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
    dcp: int = 8  # DCP group size (the KV write's owner ranks)
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
    # Speculative expert prefetch (TPU_SEGMK_EXPERT_PREFETCH): `e_pre` pool
    # slots beside the `nb` ring slots of one expert buffer. 0 = the kernel
    # without it (every code path below gated on `prefetch`).
    e_pre: int = 0
    # Where the pool's expert ids come from: "self" = probe P15 rule (b) run in
    # this call on its own input residual (router_N(post_attention_layernorm_N
    # (r_in_N)): its own gate / norm / bias), while AR1's reduce-scatter is in
    # flight; "input" = the `pref` operand (tests: controlled pools).
    pf_src: str = "self"
    # False: the removal arm -- the pool DMAs are issued and waited, every
    # routed expert still streams through the ring (timing only; exact).
    pf_use: bool = True
    # False: no pool DMA at all (the layout + predictor cost alone; timing).
    pf_issue: bool = True
    # q_a/kv_a column blocks streamed early, into their own buffer, right
    # behind the pool DMAs (the rest go into free expert slots after the
    # routed experts, under AR2): trades pool slots for hidden q_a/kv_a bytes.
    qkv_early: int = 0
    # Self-predicted pool with the incumbent q_a/kv_a layout: issue q_a/kv_a
    # before AR1's barrier, as the prefetch-off kernel does (it streams under
    # the barrier's rank-skew wait), instead of behind the pool.
    pf_qkv_first: bool = False
    # Test-only positive control: "stale_slot" fills pool slot p with the
    # expert predicted for slot p + 1 (the map still says p).
    pf_fault: str = ""

    @property
    def m(self) -> int:  # moe_tp_decode row bucket of this token bucket
        return 16 if self.T <= 16 else 32

    @property
    def prefetch(self) -> bool:
        return self.e_pre > 0

    @property
    def slots(self) -> int:  # expert buffer slots: ring + pool
        return self.nb + self.e_pre

    @property
    def qkv_blk(self) -> int:  # q_a/kv_a column block = one expert slot's w13
        return 2 * self.moe_i

    @property
    def qkv_nblk(self) -> int:
        return -(-self.qkv_pad // self.qkv_blk)

    @property
    def qkv_blocked(self) -> bool:
        """q_a/kv_a held as column blocks (some streamed after the routed
        experts into free expert slots); with every block early the
        incumbent's buffer, layout and matmul are kept as they are."""
        return self.prefetch and self.qkv_early < self.qkv_nblk

    @property
    def e_pad(self) -> int:
        return -(-self.experts // LANES) * LANES

    @property
    def o_in(self) -> int:
        return self.heads_rank * self.v_head

    @property
    def tok_rows(self) -> int:  # KV-write token padding
        return -(-self.T // KV_TOKEN_ROUND) * KV_TOKEN_ROUND

    @property
    def lat_w(self) -> int:  # entry width: every head's latent
        return self.heads_rank * self.kv_lora

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
    if cfg.v_head % LANES:
        return "W_UV's output width is not a lane multiple"
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
    if cfg.prefetch:
        if cfg.e_pre > PRED_LANES:
            return f"prefetch pool above {PRED_LANES} slots"
        if cfg.nb < 2:
            return "prefetch needs at least 2 ring slots"
        if not 0 <= cfg.qkv_early <= cfg.qkv_nblk:
            return "qkv_early outside [0, q_a/kv_a blocks]"
        if cfg.qkv_blocked and cfg.slots < cfg.qkv_nblk - cfg.qkv_early:
            return "late q_a/kv_a blocks do not fit the expert buffer's slots"
        if cfg.pf_fault not in ("", "stale_slot", "no_dma"):
            return f"unknown prefetch fault {cfg.pf_fault}"
        if cfg.pf_src not in ("self", "input"):
            return f"unknown prefetch source {cfg.pf_src}"
    if vmem_need_bytes(cfg)["limit"] > VMEM_CAP:
        return f"VMEM plan above {VMEM_CAP // MIB} MiB (prefetch pool overflow?)"
    return None


# XLA's largest valid scoped-VMEM request on tpu7x is 64 MiB - 64 KiB
# (67043328 B, "max valid bytes"); keep a margin under it.
VMEM_CAP = 63 * MIB
PRED_LANES = 16  # lanes of the prediction row that carry expert ids


def prefetch_fit(
    cfg: SegCfg, e_pre: int, qkv_early: int = 0, min_nb: int = 6
) -> SegCfg | None:
    """`cfg` with the largest pool <= e_pre slots that fits VMEM beside
    `qkv_early` early q_a/kv_a blocks (clamped to the block count), the ring
    kept at nb (shrunk to min_nb only when that buys pool slots); None when
    not even one slot fits."""
    k = min(max(qkv_early, 0), cfg.qkv_nblk)
    for p in range(min(e_pre, PRED_LANES), 0, -1):
        for nb in range(cfg.nb, min_nb - 1, -1):
            c = dataclasses.replace(cfg, e_pre=p, nb=nb, qkv_early=k)
            if supported(c) is None:
                return c
    return None


# =====================================================================================
# all-reduce epilogue: moe_tp_decode/all_reduce.py epilogue pieces with the barrier
# wait optional (AR2 runs after AR1's exchange proved every peer is inside the call)
# =====================================================================================
def _exchange(spec, me, buf, rows, cols, s, mid=None):
    cw = cols // spec.n

    def slot(q):
        return s["recv"].at[q, pl.ds(0, rows), pl.ds(0, cw)]

    def chunk(t):
        return MAR._cols(buf, t, cw)

    red = s["red"].at[pl.ds(0, rows), pl.ds(0, cw)]
    own = pltpu.make_async_copy(chunk(me), slot(me), s["loc"].at[0])
    own.start()
    rs = MAR._send_all(spec, me, chunk, slot, s["rs_send"], s["rs_recv"])
    if mid is not None:  # independent work while the reduce-scatter is in flight
        mid()
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


def _run_path(spec, kind, me, nv, partial, out_ref, s, barrier, mid=None):
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
    _exchange(spec, me, buf, rows, cols, s, mid)
    if kind == "fold":
        MAR._unfold(spec, s["q"][...].astype(F32), nv, out_ref)


def _ar_epilogue(spec, rank, n_valid, partial, out_ref, scr, *, wait_barrier, mid=None):
    """`mid`: traced once per exclusive path, run once per call (between the
    reduce-scatter sends and their waits; on the no-valid-rows path after
    the barrier)."""
    s = dict(zip(MAR._scratch_names(spec), scr, strict=True))
    barrier = pltpu.get_barrier_semaphore() if wait_barrier else None
    m, tr = spec.m, spec.tile_rows
    nv = jnp.minimum(jnp.maximum(n_valid, 0), m)

    @pl.when(nv == 0)
    def _zero():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        if barrier is not None:
            pl.semaphore_wait(barrier, spec.n)
        if mid is not None:
            mid()

    lo = 0
    if spec.fold:

        @pl.when((nv > 0) & (nv <= spec.fold_rows))
        def _fold_path():
            _run_path(spec, "fold", rank, nv, partial, out_ref, s, barrier, mid)

        lo = spec.fold_rows
    if tr < m:

        @pl.when((nv > lo) & (nv <= tr))
        def _trim_path():
            _run_path(spec, "trim", rank, nv, partial, out_ref, s, barrier, mid)

        lo = tr

    @pl.when(nv > lo)
    def _full_path():
        _run_path(spec, "full", rank, nv, partial, out_ref, s, barrier, mid)


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


# =====================================================================================
# speculative expert prefetch (cfg.prefetch)
# =====================================================================================
def _pref_valid(cfg, e):
    return (e >= 0) & (e < cfg.experts)


def _pool_src(cfg, r, p):
    """The expert id pool slot p is filled with: pref[p] (the positive
    control "stale_slot": pref[p + 1] when that is a valid id, so the slot
    holds another expert's weights than the map says)."""
    e = r["pref_s"][0, p]
    if cfg.pf_fault == "stale_slot":
        e2 = r["pref_s"][0, (p + 1) % cfg.e_pre]
        e = jnp.where(_pref_valid(cfg, e2), e2, e)
    return e


def _pool_started(cfg, r, p):
    """Whether pool slot p's DMAs were issued (pref[p] a valid expert id;
    never under the positive control "no_dma", which leaves the slot holding
    whatever an earlier call left there while the map still says hit)."""
    if cfg.pf_fault == "no_dma" or not cfg.pf_issue:
        return jnp.bool_(False)
    return _pref_valid(cfg, r["pref_s"][0, p])


def _pool_copies(cfg, r, p, slot_k, e=None):
    """Pool slot p's four DMAs (buffer slot nb + p), as the ring's. `e`: the
    source expert (only the semaphore and the destination matter to a wait)."""
    if e is None:
        e = _pool_src(cfg, r, p)
    e = jnp.clip(e, 0, cfg.experts - 1)
    slot = cfg.nb + p
    sem = r["pool_sem"]
    return [
        pltpu.make_async_copy(r["w13"].at[e], r["w1_r"].at[slot], sem.at[p, 0]),
        pltpu.make_async_copy(r["w2"].at[e], r["w2_r"].at[slot], sem.at[p, 1]),
        pltpu.make_async_copy(
            r["s13s"].at[slot_k, e], r["s1_r"].at[slot], sem.at[p, 2]
        ),
        pltpu.make_async_copy(r["s2"].at[e, 0], r["s2_r"].at[slot], sem.at[p, 3]),
    ]


def _qkv_block_copy(cfg, r, j):
    """q_a/kv_a column block j: early blocks into their own buffer, the rest
    into expert slot j - qkv_early (free after the routed experts)."""
    if j < cfg.qkv_early:
        dst = r["qkv_e"].at[j]
    else:
        dst = r["w1_r"].at[j - cfg.qkv_early]
    return pltpu.make_async_copy(r["qkv_w"].at[j], dst, r["qkvb_sem"].at[j])


def _issue_pool(cfg, r, slot_k):
    """Start every pool slot's DMAs (slots whose id is not an expert stay
    empty), then the early q_a/kv_a blocks behind them."""
    for p in range(cfg.e_pre):

        @pl.when(_pool_started(cfg, r, p))
        def _(p=p):
            for c in _pool_copies(cfg, r, p, slot_k):
                c.start()

    for j in range(cfg.qkv_early if cfg.qkv_blocked else 0):
        _qkv_block_copy(cfg, r, j).start()


def _compact_pool(cfg, r):
    """`_compact_mxu` with the pool: the same ascending ids of the hit experts
    (row 0), and per step j its buffer slot (row 1: nb + p for an expert the
    pool holds in slot p, else ring slot (miss ordinal % nb)), pool flag (row
    2), miss ordinal (row 3); row 4 the misses' ids in ascending order (the
    ring's stream). Every
    operand of the 0/1 matmuls is an integer <= 256: exact in bf16 with f32
    accumulation. Returns (hit count, miss count)."""
    E, nb, P = cfg.e_pad, cfg.nb, cfg.e_pre
    lane = lax.broadcasted_iota(I32, (1, E), 1)
    hit = r["hits_v"][...] != 0  # [1, E]
    # pool map: pm[e] = the pool slot holding expert e (last slot wins on a
    # duplicate id), -1 if none; the removal arm (pf_use False) maps nothing.
    pm = jnp.full((1, E), -1, I32)
    if cfg.pf_use and cfg.pf_issue:
        for p in range(P):
            e = r["pref_s"][0, p]
            pm = jnp.where((lane == e) & _pref_valid(cfg, e), p, pm)
    in_pool = hit & (pm >= 0)
    miss = hit & jnp.logical_not(in_pool)
    row8 = lax.broadcasted_iota(I32, (8, E), 0)
    lhs = jnp.where(row8 == 0, jnp.where(hit, 1.0, 0.0), 0.0)
    lhs = jnp.where(row8 == 1, jnp.where(miss, 1.0, 0.0), lhs).astype(BF16)
    U = _sel(E, E, lambda i, j: i <= j)
    incl = lax.dot_general(lhs, U, _NN, preferred_element_type=F32).astype(I32)
    incl_hit, incl_miss = incl[0:1], incl[1:2]  # inclusive ranks [1, E]
    jj = lax.broadcasted_iota(I32, (E, E), 0)
    P_hit = jnp.where((jj == incl_hit - 1) & hit, 1.0, 0.0).astype(BF16)
    P_miss = jnp.where((jj == incl_miss - 1) & miss, 1.0, 0.0).astype(BF16)
    ordm = jnp.where(miss, incl_miss - 1, 0)
    bslot = jnp.where(in_pool, nb + pm, lax.rem(ordm, nb))
    vals = (lane, bslot, jnp.where(in_pool, 1, 0), ordm)
    lhs2 = jnp.zeros((8, E), I32)
    for i, v in enumerate(vals):
        lhs2 = jnp.where(row8 == i, v, lhs2)
    lhs2 = lhs2.astype(F32).astype(BF16)
    steps = lax.dot_general(lhs2, P_hit, _NT, preferred_element_type=F32)  # [8, E]
    erow = jnp.where(row8 == 0, lane, 0).astype(F32).astype(BF16)
    mids = lax.dot_general(erow, P_miss, _NT, preferred_element_type=F32)  # row 0
    out = jnp.where(row8 < 4, steps.astype(I32), 0)
    out = jnp.where(row8 == 4, jnp.broadcast_to(mids[0:1].astype(I32), (8, E)), out)
    r["ids_v"][...] = out
    n_hit = jnp.sum(jnp.where(hit, 1, 0), axis=1, keepdims=True)  # [1, 1]
    n_miss = jnp.sum(jnp.where(miss, 1, 0), axis=1, keepdims=True)
    cl = lax.broadcasted_iota(I32, (8, LANES), 1)
    r["cnt_v"][...] = jnp.where(cl == 0, n_hit, jnp.where(cl == 1, n_miss, 0))
    a = pltpu.make_async_copy(r["ids_v"], r["ids_s"], r["hit_sem"].at[0])
    b = pltpu.make_async_copy(r["cnt_v"], r["cnt_s"], r["hit_sem"].at[1])
    a.start()
    b.start()
    a.wait()
    b.wait()
    return r["cnt_s"][0, 0], r["cnt_s"][0, 1]


def predict_mass(scores, bias, n_valid, *, topk):
    """The predictor's per-expert mass [1, E]: the production routing rule
    (biased top-k selection, unbiased weights renormalized, bf16) applied to
    the predicted scores, summed over the valid rows."""
    w = _route_argmax(scores, bias, n_valid, topk=topk)
    return jnp.sum(w, axis=0, keepdims=True)


def rank_experts(mass, n):
    """[8, 128] int32: row 0 lanes 0..n-1 = the n experts of largest mass
    (ties to the lower id; -1 once the mass is 0), every other entry -1."""
    E = mass.shape[1]
    lane = lax.broadcasted_iota(I32, (8, E), 1)
    m8 = jnp.broadcast_to(mass, (8, E))
    orow = lax.broadcasted_iota(I32, (8, LANES), 0)
    olane = lax.broadcasted_iota(I32, (8, LANES), 1)
    out = jnp.full((8, LANES), -1, I32)
    for p in range(n):
        mx = jnp.max(m8, axis=1, keepdims=True)  # [8, 1]
        idx = jnp.argmax(m8, axis=1, keepdims=True).astype(I32)
        idv = jnp.where(mx > 0, idx, -1)
        out = jnp.where((orow == 0) & (olane == p), idv, out)
        m8 = jnp.where(lane == idx, -1.0, m8)
    return out


def _predict_ids(cfg, r, n_valid, eps):
    """Probe P15 rule (b) on this call's input residual r_in (res_pad rows,
    bf16 values): post_attention_layernorm -> GateLinear (bf16 logits, f32
    sigmoid) -> + e_score_correction_bias, top-k per valid row -> experts
    ranked by routed mass. The real routing input is norm(r_in + attention)."""
    hp, _ = rms_add(r["res_pad"][...], None, r["g_post_v"][...], eps)
    scores = router_scores(hp, r["gate_v"][...])
    mass = predict_mass(scores, r["bias_v"][...], n_valid, topk=cfg.topk)
    return rank_experts(mass, cfg.e_pre)


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
STAGED_IN = ("lat", "act")  # act = [res (T rows); cfull (m); sfull (m)], width H
HBM_IN = (
    "wuv",  # bf16 [heads, kv_lora, v_head]: layer N's W_UV
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
# The prefetch variant's extra operand (pf_src "input" only): `pref` int32
# [8, 128], row 0 lanes 0..e_pre-1 = the expert ids to stage into the pool
# (-1 / out of range = none).
PF_HBM_IN = ("pref",)


def hbm_in(cfg: SegCfg) -> tuple[str, ...]:
    return HBM_IN + (PF_HBM_IN if cfg.prefetch and cfg.pf_src == "input" else ())


def vec_fields(cfg: SegCfg) -> tuple[str, ...]:
    return VEC_FIELDS


def outs(cfg: SegCfg) -> tuple[str, ...]:
    return OUTS


def scratch_table(cfg: SegCfg):
    """[(name, shape-object)] in kernel order, and the all-reduce spec."""
    H, m, T = cfg.hidden, cfg.m, cfg.T
    nb, ic = cfg.slots, cfg.moe_i  # ring (+ pool) slots
    spec = MAR.make_spec(cfg.tp, m, H)
    # Prefetch: no q_a/kv_a buffer -- its column blocks stream into the
    # expert slots after the routed experts are done.
    qkv = [] if cfg.qkv_blocked else [("qkv_v", pltpu.VMEM((H, cfg.qkv_pad), F8))]
    sc = [
        ("wuv_v", pltpu.VMEM((cfg.heads_rank, cfg.kv_lora, cfg.v_head), BF16)),
        ("o_tiles", pltpu.VMEM((cfg.o_slots, cfg.o_in, cfg.o_tile), F8)),
        ("gate_v", pltpu.VMEM((cfg.e_pad, H), BF16)),
        ("sw13_v", pltpu.VMEM((H, 2 * cfg.shared_i), F8)),
        ("sw2_v", pltpu.VMEM((cfg.shared_i, H), F8)),
        *qkv,
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
        ("xs", pltpu.VMEM((m, 2 * ic), F32)),
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
        ("in_sem", pltpu.SemaphoreType.DMA((len(STAGED_IN) + 1,))),  # + W_UV
        ("vec_sem", pltpu.SemaphoreType.DMA((len(vec_fields(cfg)),))),
        ("o_sem", pltpu.SemaphoreType.DMA((cfg.o_slots,))),
        ("w_sem", pltpu.SemaphoreType.DMA((1 + cfg.qkv_chunks,))),  # gate, qkv
        ("shared_sem", pltpu.SemaphoreType.DMA((2,))),
        ("ring_sem", pltpu.SemaphoreType.DMA((cfg.nb, 4))),
        ("hit_sem", pltpu.SemaphoreType.DMA((2,))),
        ("out_sem", pltpu.SemaphoreType.DMA((len(outs(cfg)),))),
    ]
    if cfg.prefetch:
        sc += [
            ("pref_s", pltpu.SMEM((8, LANES), I32)),
            ("pred_v", pltpu.VMEM((8, LANES), I32)),
            ("pool_sem", pltpu.SemaphoreType.DMA((cfg.e_pre, 4))),
            *(
                [
                    ("qkvb_sem", pltpu.SemaphoreType.DMA((cfg.qkv_nblk,))),
                    ("qkv_e", pltpu.VMEM((cfg.qkv_early, H, cfg.qkv_blk), F8)),
                ]
                if cfg.qkv_blocked and cfg.qkv_early
                else [("qkvb_sem", pltpu.SemaphoreType.DMA((cfg.qkv_nblk,)))]
                if cfg.qkv_blocked
                else []
            ),
            ("pf_sem", pltpu.SemaphoreType.DMA((1,))),  # pool ids -> SMEM
        ]
    shapes = operand_shapes(cfg)
    sc += [(n, pltpu.VMEM(*shapes[n])) for n in STAGED_IN]
    sc += [(f"{n}_v", pltpu.VMEM((1, cfg.field(n)[1]), F32)) for n in vec_fields(cfg)]
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
        + list(hbm_in(cfg) + outs(cfg))
        + [n for n, _ in sc_t]
    )
    assert len(names) == len(refs), (len(names), len(refs))
    r = dict(zip(names, refs, strict=True))
    n_ar = len(MAR.scratch_shapes(spec))
    ar1 = [r[f"ar1_{i}"] for i in range(n_ar)]
    ar2 = [r[f"ar2_{i}"] for i in range(n_ar)]

    H, T, m, nb = cfg.hidden, cfg.T, cfg.m, cfg.nb
    pf = cfg.prefetch
    meta = r["meta"]
    n_valid = meta[0]
    rank = meta[1]
    slot_k = meta[2]  # this boundary's row of the constant stacks
    eps = jnp.float32(cfg.rms_eps)

    def frow(name):
        off, w = cfg.field(name)
        return r["fstack"].at[slot_k, :, pl.ds(off, w)]

    # ---- 1. barrier signals for BOTH in-kernel all-reduces (one prologue;
    #         AR2 skips its wait); W_UV, the first weight the call reads
    MAR.prologue(spec)
    wuv_copy = pltpu.make_async_copy(
        r["wuv"], r["wuv_v"], r["in_sem"].at[len(STAGED_IN)]
    )
    wuv_copy.start()
    pf_in = pf and cfg.pf_src == "input"
    pf_self = pf and cfg.pf_src == "self"
    qkv_first = pf_self and cfg.pf_qkv_first and not cfg.qkv_blocked
    if pf_in:  # the pool's expert ids, needed when the pool DMAs are issued
        pref_copy = pltpu.make_async_copy(r["pref"], r["pref_s"], r["pf_sem"].at[0])
        pref_copy.start()

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
    for i, n in enumerate(vec_fields(cfg)):
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
        for c in range(0 if cfg.qkv_blocked else cfg.qkv_chunks):
            qkv.append(
                pltpu.make_async_copy(
                    r["qkv_w"].at[pl.ds(c * qc_rows, qc_rows)],
                    r["qkv_v"].at[pl.ds(c * qc_rows, qc_rows)],
                    w_sem.at[1 + c],
                )
            )
        pre.update(gate=gate, shared=shared, qkv=qkv)
        if pf_in:  # the pool first; q_a/kv_a behind it
            pref_copy.wait()
            issue_pool()
        elif not pf or qkv_first:
            for cp in qkv:
                cp.start()

    def issue_pool():
        """The pool's DMAs, then the q_a/kv_a bytes due early behind them."""
        _issue_pool(cfg, r, slot_k)
        if not cfg.qkv_blocked and not qkv_first:
            for cp in pre["qkv"]:
                cp.start()

    # ---- 3. entry: W_UV of the merged latent -> attn [T, o_in] bf16
    staged = {}
    for i, n in enumerate(STAGED_IN):
        staged[n] = pltpu.make_async_copy(r[f"in_{n}"], r[n], r["in_sem"].at[i])
        staged[n].start()
    staged["lat"].wait()
    wuv_copy.wait()
    attn = _entry_wuv(cfg, r)

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

    def fill_residual():
        r["res_pad"][...] = jnp.zeros((m, H), F32)
        staged["act"].wait()
        vec["g_post"].wait()
        r["res_pad"][pl.ds(0, T), :] = r["act"][pl.ds(0, T), :].astype(F32)

    def self_predict():
        """P15 rule (b) on this call's input residual with its own router,
        then the pool DMAs -- while AR1's reduce-scatter is in flight."""
        fill_residual()
        pre["gate"].wait()
        vec["bias"].wait()
        r["pred_v"][...] = _predict_ids(cfg, r, n_valid, eps)
        cp = pltpu.make_async_copy(r["pred_v"], r["pref_s"], r["pf_sem"].at[0])
        cp.start()
        cp.wait()
        issue_pool()

    # ---- 5. AR1 (attention TP all-reduce), barrier waited here
    _ar_epilogue(
        spec,
        rank,
        n_valid,
        r["y_pad"][...],
        r["ar1_out"],
        ar1,
        wait_barrier=True,
        **({"mid": self_predict} if pf_self else {}),
    )

    # ---- 6. post_attention_layernorm(attn_sum, residual)
    if not pf_self:
        fill_residual()
    h, s1 = rms_add(
        r["ar1_out"][...].astype(F32), r["res_pad"][...], r["g_post_v"][...], eps
    )
    r["res1_v"][...] = s1.astype(BF16).astype(F32)
    r["h_v"][...] = h

    # ---- 7. router: GateLinear's bf16 logits, f32 sigmoid
    if not pf_self:
        pre["gate"].wait()
    scores = router_scores(r["h_v"][...], r["gate_v"][...])

    # ---- 8a. routing
    if not pf_self:
        vec["bias"].wait()
    r["wtab"][...] = _route_argmax(scores, r["bias_v"][...], n_valid, topk=cfg.topk)
    hit = jnp.any(r["wtab"][...] != 0, axis=0, keepdims=True)
    r["hits_v"][...] = jnp.where(hit, 1, 0).astype(I32)
    ring_sem = r["ring_sem"]
    # ---- 8b. compaction (two exact 0/1 matmuls) + ring prime. Prefetch:
    #          the ring streams the misses only (in ascending id order, a
    #          subsequence of the steps); ids rows 1-3 give each step its
    #          buffer slot, pool flag and miss ordinal, row 4 the miss ids.
    if pf:
        items, n_ring = _compact_pool(cfg, r)
        ring_row = 4
    else:
        items = _compact_mxu(cfg, r)
        n_ring, ring_row = items, 0
    ids = r["ids_s"]

    def transfers(step):  # the step-th expert the ring streams
        e = ids[ring_row, step]
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

        @pl.when(jnp.int32(s_) < n_ring)
        def _(s_=s_):
            for c in transfers(jnp.int32(s_)):
                c.start()

    # ---- 8c. x quant + shared expert, as moe_tp_decode's kernel: x per row
    #          over the full K (multiply by the inverse, no clip) for the
    #          routed experts; its `_shared_expert` (true division + clip)
    ic = cfg.moe_i
    r["owner"][...] = jnp.zeros((m, H), F32)
    x = r["h_v"][...].astype(F32)
    xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / FP8_MAX
    r["xq"][...] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(F8)
    r["xs"][...] = jnp.broadcast_to(xsc, (m, 2 * ic))
    for c in pre["shared"]:
        c.wait()
    vec["ss13"].wait()
    vec["ss2"].wait()
    MK._shared_expert(
        r["h_v"],
        r["sxq"],
        r["sw13_v"],
        r["ss13_v"],
        r["sw2_v"],
        r["ss2_v"],
        r["shared"],
    )

    # ---- 8d. routed experts (ring), moe_tp_decode's gate_up / finish
    expert_columns = lax.broadcasted_iota(I32, r["wtab"].shape, 1)

    def gate_up(slot):
        return jnp.matmul(r["xq"][...], r["w1_r"][slot], preferred_element_type=F32)

    def finish(step, slot, acc1):
        e = ids[0, step]
        coeff = jnp.sum(
            jnp.where(expert_columns == e, r["wtab"][...], 0.0), axis=1, keepdims=True
        )
        acc1 = acc1 * r["xs"][...] * r["s1_r"][slot]
        # Mask, not multiply: a padded row with non-finite hidden would give
        # NaN * 0 = NaN. Rows this expert does not own contribute exactly 0.
        mid = jnp.where(
            coeff != 0, jax.nn.silu(acc1[:, :ic]) * acc1[:, ic:] * coeff, 0.0
        )
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(F8)
        for n in range(0, H, MK._N_CHUNK):
            acc2 = jnp.matmul(
                mq, r["w2_r"][slot, :, n : n + MK._N_CHUNK], preferred_element_type=F32
            )
            r["owner"][:, n : n + MK._N_CHUNK] += (
                acc2 * r["s2_r"][slot, :, n : n + MK._N_CHUNK] * ms
            )

    if not pf:

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

    else:
        # The same steps in the same (ascending id) order as above -- so the
        # same arithmetic in the same order: only where a step's weights
        # come from differs (pool slot nb + p, or ring slot ordinal % nb).
        # Every issued pool slot is waited here, once, after the ring is
        # primed (issued under AR1, they have normally landed by now).
        for p_ in range(cfg.e_pre):

            @pl.when(_pool_started(cfg, r, p_))
            def _(p_=p_):
                for c in _pool_copies(cfg, r, p_, slot_k):
                    c.wait()

        def wait_step(step):
            @pl.when(ids[2, step] == 0)
            def _():
                for c in transfers(ids[3, step]):
                    c.wait()

        @pl.when(items > 0)
        def _run_experts_pf():
            wait_step(jnp.int32(0))
            last = jnp.int32(cfg.e_pad - 1)

            def expert_body(step, acc1):
                nxt = step + 1

                @pl.when(nxt < items)
                def _():
                    wait_step(nxt)

                acc1_next = gate_up(ids[1, jnp.minimum(nxt, last)])
                finish(step, ids[1, step], acc1)
                k = ids[3, step]

                @pl.when((ids[2, step] == 0) & (k + nb < n_ring))
                def _():
                    for c in transfers(k + nb):
                        c.start()

                return acc1_next

            lax.fori_loop(0, items, expert_body, gate_up(ids[1, 0]))

        # Every slot is free: q_a/kv_a's column blocks stream into slots
        # 0..qkv_nblk-1 under AR2; the predictor's gate into gate_v.
        for j in range(cfg.qkv_early, cfg.qkv_nblk if cfg.qkv_blocked else 0):
            _qkv_block_copy(cfg, r, j).start()

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
    if not cfg.qkv_blocked:
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
    else:
        # Per column block, the same K chunks summed in the same order.
        blocks = []
        for j in range(cfg.qkv_nblk):
            _qkv_block_copy(cfg, r, j).wait()
            src, jj = (
                (r["qkv_e"], j) if j < cfg.qkv_early else (r["w1_r"], j - cfg.qkv_early)
            )
            acc_j = None
            for c in range(cfg.qkv_chunks):
                part = lax.dot_general(
                    xq2[:, c * qc_rows : (c + 1) * qc_rows],
                    src[jj, pl.ds(c * qc_rows, qc_rows), :],
                    _NN,
                    preferred_element_type=F32,
                )
                acc_j = part if acc_j is None else acc_j + part
            blocks.append(acc_j)
        acc = jnp.concatenate(blocks, axis=1)[:, : cfg.qkv_pad]
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
    copies = [res_copy, qc_copy, kvp_copy]
    for c in copies:
        c.wait()


def _entry_wuv(cfg: SegCfg, r):
    """attn[t, h*v:(h+1)*v] = bf16(lat[t, h*kv:(h+1)*kv] . W_UV[h]) for each
    head: the backend's `torch.bmm(outputs_t, W_UV)` after the DCP merge (bf16
    operands, f32 accumulation, one rounding to bf16)."""
    kv = cfg.kv_lora
    heads = []
    for hh in range(cfg.heads_rank):
        lat = r["lat"][:, pl.ds(hh * kv, kv)]  # [T, kv_lora] bf16
        heads.append(
            lax.dot_general(
                lat, r["wuv_v"][hh], _NN, preferred_element_type=F32
            ).astype(BF16)
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
    name = f"segment_megakernel_skip-T_{cfg.T}-m_{cfg.m}-nb_{cfg.nb}-e_{cfg.experts}"
    if cfg.prefetch:
        name += f"-pf_{cfg.e_pre}_{cfg.pf_src}_qe{cfg.qkv_early}"
        name += "_qf" if cfg.pf_qkv_first else ""
        name += "" if cfg.pf_use else "-issueonly"
        name += "" if cfg.pf_issue else "-noissue"
    return name


def segment_call(cfg: SegCfg, ops: dict, *, interpret=None) -> dict:
    """One boundary on one device. `ops`: name -> array for every name of
    SMEM_IN + STAGED_IN + hbm_in(cfg), shapes as `operand_shapes(cfg)`. Returns
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
            in_specs=[anyspace] * len(STAGED_IN) + [hbm] * len(hbm_in(cfg)),
            out_specs=tuple([anyspace] * len(out_shapes)),
            scratch_shapes=[s for _, s in sc_t],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(limit), collective_id=MAR.COLLECTIVE_ID
        ),
        name=kernel_name(cfg),
        **({} if interpret is None else {"interpret": interpret}),
    )
    res = call(*[ops[n] for n in SMEM_IN + STAGED_IN + hbm_in(cfg)])
    return dict(zip(outs(cfg), res, strict=True))


def operand_shapes(cfg: SegCfg, boundaries: int = 1) -> dict:
    """name -> (shape, dtype) the kernel takes (per device); `boundaries`
    sizes the constant stacks."""
    H, T, m = cfg.hidden, cfg.T, cfg.m
    return {
        "meta": ((META_DST + T,), I32),
        "lat": ((T, cfg.lat_w), BF16),
        "wuv": ((cfg.heads_rank, cfg.kv_lora, cfg.v_head), BF16),
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
        **(
            {
                # column blocks of the padded q_a/kv_a weight (block_qkv)
                "qkv_w": ((cfg.qkv_nblk, H, cfg.qkv_blk), F8),
            }
            if cfg.qkv_blocked
            else {}
        ),
        **(
            {"pref": ((8, LANES), I32)}
            if cfg.prefetch and cfg.pf_src == "input"
            else {}
        ),
    }


def block_qkv(qkv_w, blk: int):
    """[H, qkv_pad] -> [nblk, H, blk]: the prefetch variant's q_a/kv_a layout
    (column blocks the shape of one expert slot's w13, zero-padded)."""
    H, n = qkv_w.shape
    nblk = -(-n // blk)
    w = jnp.pad(qkv_w, ((0, 0), (0, nblk * blk - n)))
    return jnp.transpose(w.reshape(H, nblk, blk), (1, 0, 2))


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
