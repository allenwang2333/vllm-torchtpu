# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""segment_megakernel: MoE layer N's tail + skip layer N+1's head in one call.

The oracle (ported from the segment-megakernel campaign's instruments): every
output is compared, per device and per real token row, with an f32 reference
computed at HIGHEST precision from the same inputs (dequantized fp8 weights,
unquantized activations, f32 all-reduces) -- the residual delta the segment
adds (attention + MoE), q_c, and the KV write rows decoded back to values.
Routing is planted (each token's residual leans on 8 router directions), so
no row has a near-tie in its top-k and every row is compared.

  * 16 CPU devices (pytest): this file runs itself as a subprocess with 16
    host devices, in TPU interpret mode with the race detector, at reduced
    dims. That checks the DMA / semaphore / all-reduce protocols, the operand
    layouts and the arithmetic -- not Mosaic lowering, VMEM limits or timing.
    Positive controls (a misaligned o_proj scale, swapped norm weights, a wrong
    KV destination, the next layer's weights in the wrong slot) must fail it.
  * TPU 2x2x2 (16 cores, 2 hosts), full GLM-5.2 TP16 x DCP8 dims:
        python3 test_segment_megakernel.py --tpu
    under jax.distributed (simple-run-command-2node exports COORD / NPROC /
    JOBSET_JOB_INDEX). Runs the kernel AND the incumbent composition of the
    production ops it replaces (the `_merge_received` merge, the fp8_skinny
    o_proj, the small all-reduce, fused_add_rmsnorm.reference, the router dot,
    moe_tp_decode, xla_quantized_matmul, the bf16 NeoX rope, quantize_kv,
    pack_write_rows) on decode buckets b1/b2/b4/b8 (T = 4/8/16/32) and a
    padded b3, judged by the campaign's rule: tol = max(1.5 x the incumbent's
    error, floor) per metric; the kernel passes if under tol everywhere.
"""

import dataclasses
import functools
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import segment_megakernel as seg

F8, BF16, F32, I32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32, jnp.int32
AXIS = "tp"
HI = lax.Precision.HIGHEST
N_RANKS = 16
DCP = 8


@dataclasses.dataclass(frozen=True)
class Dims:
    hidden: int
    experts: int
    q_lora: int
    o_tile: int
    topk: int = 8
    moe_i: int = 128
    shared_i: int = 128
    kv_lora: int = 512
    rope: int = 64
    heads_rank: int = 4
    v_head: int = 256
    routed_scale: float = 2.5
    eps: float = 1e-5
    page_size: int = 256
    num_pages: int = 64
    interleave: int = 4

    @property
    def qkv_width(self):
        return self.q_lora + self.kv_lora + self.rope

    @property
    def qkv_pad(self):
        return -(-self.qkv_width // 128) * 128

    def cfg(self, T):
        return seg.SegCfg(
            T=T,
            hidden=self.hidden,
            tp=N_RANKS,
            dcp=DCP,
            heads_rank=self.heads_rank,
            v_head=self.v_head,
            q_lora=self.q_lora,
            kv_lora=self.kv_lora,
            rope=self.rope,
            qkv_pad=self.qkv_pad,
            experts=self.experts,
            topk=self.topk,
            moe_i=self.moe_i,
            shared_i=self.shared_i,
            routed_scale=self.routed_scale,
            rms_eps=self.eps,
            o_tile=self.o_tile,
        )


# One rank's slice of GLM-5.2-FP8 at TP16 x DCP8.
GLM52 = Dims(hidden=6144, experts=256, q_lora=2048, o_tile=1536)
# Interpret mode: a small geometry with every code path (the all-reduce fold
# path needs hidden / 16 >= 2 lanes; the router needs a lane of experts).
SMALL = Dims(hidden=4096, experts=128, q_lora=512, o_tile=1024)
# SEGMK_TEST_TINY=1: a quicker local smoke (no all-reduce fold path).
TINY = Dims(hidden=2048, experts=128, q_lora=512, o_tile=512)
# (T, n_valid): decode b1 / b2 / b4 / b8 and a padded b3 (12 real rows in 16).
TPU_CASES = ((4, 4), (8, 8), (16, 16), (32, 32), (16, 12))
# Interpret mode is slow (~minutes per call at SMALL): the fold path with the
# positive controls, and a padded bucket on the full path.
CPU_CASES = ((4, 4), (16, 12))
# Harness tolerance rule (instruments spec.TOL_POLICY): factor x the
# incumbent's error vs the f32 reference, floored per metric.
TOL_FACTOR = 1.5
TOL_FLOOR = dict(res_delta=2e-3, q_c=2e-3, kv_nope=2e-3, kv_rope=2e-3)
# Interpret mode (no incumbent run there): the kernel's own fp8 activation
# error vs the f32 reference. The campaign measured the incumbent at 0.050
# (res_delta), 0.050 (q_c), 0.059 (kv_nope), 0.068 (kv_rope) on its cells
# (j-679e2b07); a fault moves a metric by O(0.3) or more.
CPU_TOL = dict(res_delta=0.12, q_c=0.12, kv_nope=0.12, kv_rope=0.12)
METRICS = tuple(TOL_FLOOR)
FAULTS = ("oproj_scale_roll", "norms_swapped", "wrong_dst", "wrong_qkv")


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


# =====================================================================================
# Inputs, generated per rank on device
# =====================================================================================
def _pack_entry(o, lse, real, vdim):
    """The packed real-rows receive buffer (cp_mla_attention._all_to_all_packed,
    real_tokens path): o bf16 [S, T, h, V], lse f32 [S, T, h] -> int16
    [S, extra + T*h, V]: lse high halves, low halves, zero pad, then the
    partial rows; rows past the real tokens' are zero."""
    S, T, h, V = o.shape
    rows = T * h
    extra = -(-2 * rows // vdim)
    bits = lax.bitcast_convert_type(lse.reshape(S, rows), I32)
    hi = (bits >> 16).astype(jnp.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(jnp.int16)
    words = jnp.pad(
        jnp.concatenate([hi, lo], axis=1), ((0, 0), (0, extra * V - 2 * rows))
    )
    ow = lax.bitcast_convert_type(o.reshape(S, rows, V), jnp.int16)
    live = jnp.arange(rows)[None, :, None] < real * h
    ow = jnp.where(live, ow, jnp.int16(0))
    return jnp.concatenate([words.reshape(S, extra, V), ow], axis=1)


def _rank_key(seed, rank):
    return jax.random.fold_in(jax.random.PRNGKey(seed), rank)


def _per_rank_inputs(d: Dims, T: int, n_valid: int, seed: int, rank):
    """This rank's operands (inside shard_map; `rank` int32 [1])."""
    H, E, Im, Fs = d.hidden, d.experts, d.moe_i, d.shared_i
    ks = iter(jax.random.split(_rank_key(seed, rank[0]), 16))
    o_in = d.heads_rank * d.v_head

    def fp8(key, shape):
        """Normal fp8 values; one leading slice at a time (small RNG transients)."""
        if len(shape) == 2:
            return jax.random.normal(key, shape, F32).astype(F8)
        keys = jax.random.split(key, shape[0])
        return lax.map(lambda k: jax.random.normal(k, shape[1:], F32).astype(F8), keys)

    o = jax.random.normal(next(ks), (DCP, T, d.heads_rank, d.v_head), F32)
    lse = jax.random.normal(next(ks), (DCP, T, d.heads_rank), F32) * 2.0
    # Ranks owning none of a row's positions: -inf lse and a NaN partial
    # (the attention kernel ran on a dummy key there); source 0 always owns.
    unowned = jax.random.uniform(next(ks), (DCP, T, d.heads_rank)) < 0.25
    unowned = unowned & (jnp.arange(DCP) > 0)[:, None, None]
    lse = jnp.where(unowned, -jnp.inf, lse)
    o = jnp.where(unowned[..., None], jnp.nan, o).astype(BF16)
    return dict(
        o=o,
        lse=lse,
        a2a=_pack_entry(o, lse, n_valid, d.v_head),
        oproj_w=fp8(next(ks), (o_in, H)),
        oproj_s=jax.random.uniform(next(ks), (H,), F32, 0.5, 1.5)
        * (0.25 / (4 * np.sqrt(o_in))),
        w13=fp8(next(ks), (E, H, 2 * Im)),
        w2=fp8(next(ks), (E, Im, H)),
        s13=jax.random.uniform(next(ks), (E, 1, 1, 2 * Im), F32, 0.5, 1.5) / np.sqrt(H),
        s2=jax.random.uniform(next(ks), (E, 1, 1, H), F32, 0.5, 1.5)
        * (0.5 / np.sqrt(Im)),
        sw13=fp8(next(ks), (H, 2 * Fs)),
        ss13=jax.random.uniform(next(ks), (2 * Fs,), F32, 0.5, 1.5) / np.sqrt(H),
        sw2=fp8(next(ks), (Fs, H)),
        ss2=jax.random.uniform(next(ks), (H,), F32, 0.5, 1.5) * (0.25 / np.sqrt(Fs)),
    )


def _host(x, dtype):
    """A host (numpy) array in a JAX dtype: identical on every process, so it can
    be placed on a multi-host mesh replicated."""
    return np.asarray(x, np.float32).astype(jnp.dtype(dtype))


def _shared_inputs(d: Dims, T: int, n_valid: int, seed: int):
    """Replicated operands and per-step metadata (host-side numpy)."""
    rng = np.random.default_rng(seed + 1000)
    H, E = d.hidden, d.experts
    gate = rng.standard_normal((E, H)).astype(np.float32)
    gate /= np.linalg.norm(gate, axis=1, keepdims=True)  # unit router rows
    # Planted routing: token t's residual leans on 8 router directions.
    planted = np.stack([rng.choice(E, d.topk, replace=False) for _ in range(T)])
    lean = np.stack([gate[p].sum(0) for p in planted]) / np.sqrt(d.topk)
    res = lean * np.sqrt(H) * 0.6 + rng.standard_normal((T, H)) * 0.6
    width, pad = d.qkv_width, d.qkv_pad
    qkv_w = np.zeros((H, pad), np.float32)
    qkv_w[:, :width] = rng.standard_normal((H, width))
    qkv_s = np.zeros((pad,), np.float32)
    qkv_s[:width] = rng.uniform(0.5, 1.5, width) / np.sqrt(H)
    pos = rng.integers(0, 4096, T)
    inv = 1.0 / (10000.0 ** (np.arange(0, d.rope, 2) / d.rope))
    ang = pos[:, None] * inv[None, :]
    # KV destination rows per DCP rank: each token's row on its one owner,
    # -1 (skip) on every other rank.
    owner = rng.integers(0, DCP, T)
    row_of = rng.choice(d.num_pages * d.page_size, T, replace=False)
    dst = np.where(owner[None, :] == np.arange(DCP)[:, None], row_of[None, :], -1)
    return dict(
        res=_host(res, BF16),
        gate=_host(gate, BF16),
        bias=_host(rng.standard_normal(E) * 0.01, F32),
        g_post=_host(1 + 0.3 * rng.standard_normal(H), BF16),
        g_in=_host(1 + 0.3 * rng.standard_normal(H), BF16),
        g_qa=_host(1 + 0.1 * rng.standard_normal(d.q_lora), BF16),
        g_kva=_host(1 + 0.1 * rng.standard_normal(d.kv_lora), BF16),
        qkv_w=_host(qkv_w, F8),
        qkv_s=_host(qkv_s, F32),
        cos=_host(np.cos(ang), BF16),
        sin=_host(np.sin(ang), BF16),
        dst=dst.astype(np.int32),  # [DCP, T]
        planted=planted,
    )


# =====================================================================================
# The three computations, per device
# =====================================================================================
def _kernel_device(d, T, n_valid, sh, pr, rank, dst_rows, fault=None, interpret=None):
    cfg = d.cfg(T)
    g_post, g_in = sh["g_post"], sh["g_in"]
    oproj_s = pr["oproj_s"]
    qkv_w, qkv_s = sh["qkv_w"], sh["qkv_s"]
    if fault == "oproj_scale_roll":
        oproj_s = jnp.roll(oproj_s, 1)
    elif fault == "norms_swapped":
        g_post, g_in = g_in, g_post
    elif fault == "wrong_dst":
        dst_rows = jnp.where(dst_rows >= 0, dst_rows + 1, dst_rows)
    elif fault == "wrong_qkv":
        qkv_w = jnp.roll(qkv_w, 1, axis=0)
    cfull, sfull = seg.rope_lane_tables(sh["cos"], sh["sin"], cfg.m)
    consts = dict(
        oproj_s=oproj_s,
        g_post=g_post,
        g_in=g_in,
        g_qa=sh["g_qa"],
        g_kva=sh["g_kva"],
        bias=sh["bias"],
        ss13=pr["ss13"],
        ss2=pr["ss2"],
        qkv_s=qkv_s,
        s13=pr["s13"],
    )
    # Two boundaries in the stacks, this one in row 1 (row 0 a decoy: wrong
    # row indexing shows up as garbage constants).
    row = seg.fstack_row(cfg, consts)
    decoy = jnp.flip(row) + 1.0
    ops = dict(
        meta=seg.pack_meta(n_valid, rank[0], 1, dst_rows),
        a2a=pr["a2a"],
        act=seg.pack_act(sh["res"], cfull, sfull),
        oproj_w=pr["oproj_w"],
        gate=sh["gate"],
        w13=pr["w13"],
        w2=pr["w2"],
        s2=pr["s2"],
        qkv_w=qkv_w,
        fstack=jnp.stack([decoy, row]),
        sw13s=jnp.stack([jnp.flip(pr["sw13"], 0), pr["sw13"]]),
        sw2s=jnp.stack([jnp.flip(pr["sw2"], 0), pr["sw2"]]),
    )
    out = seg.segment_call(cfg, ops, interpret=interpret)
    return out["res_out"], out["qc_out"], out["kvp_out"]


def _rms(x, w, eps):
    x = x.astype(F32)
    return x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w.astype(F32)


def _rope_f32(x, cos, sin):
    half = cos.shape[-1]
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    c, s = cos.astype(F32), sin.astype(F32)
    return jnp.concatenate(
        [x1 * c - x2 * s, x2 * c + x1 * s, x[..., 2 * half :]], axis=-1
    )


def _reference_device(d, T, n_valid, sh, pr, rank):
    """The segment in f32 at HIGHEST precision: no activation quantization,
    dequantized weights, the exact routing (planted, so unambiguous)."""
    dot = functools.partial(jnp.dot, precision=HI, preferred_element_type=F32)
    o = jnp.where(jnp.isnan(pr["o"].astype(F32)), 0.0, pr["o"].astype(F32))
    real = (jnp.arange(T) < n_valid)[None, :, None, None]
    o = jnp.where(real, o, 0.0)
    lse = pr["lse"]
    fin = jnp.isfinite(lse)
    m = jnp.max(jnp.where(fin, lse, -jnp.inf), axis=0)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.where(fin, jnp.exp(lse - m[None]), 0.0)
    w = w / jnp.where(w.sum(0) > 0, w.sum(0), 1.0)[None]
    attn = jnp.sum(o * w[..., None], axis=0).reshape(T, -1)
    y = dot(attn, pr["oproj_w"].astype(F32)) * pr["oproj_s"][None]
    attn_sum = lax.psum(y, AXIS)
    s1 = attn_sum + sh["res"].astype(F32)
    h = _rms(s1, sh["g_post"], d.eps)
    scores = jax.nn.sigmoid(dot(h, sh["gate"].astype(F32).T))
    biased = scores + sh["bias"][None]
    _, ids = lax.top_k(biased, d.topk)  # [T, k]
    picked = jnp.take_along_axis(scores, ids, axis=1)
    wts = picked / jnp.sum(picked, axis=1, keepdims=True)
    w13 = pr["w13"][ids].astype(F32) * pr["s13"][ids][:, :, 0, 0][:, :, None, :]
    gu = jnp.einsum("th,tkhj->tkj", h, w13, precision=HI)
    act = jax.nn.silu(gu[..., : d.moe_i]) * gu[..., d.moe_i :]
    w2 = pr["w2"][ids].astype(F32) * pr["s2"][ids][:, :, 0, 0][:, :, None, :]
    routed = jnp.einsum("tki,tkih,tk->th", act, w2, wts, precision=HI)
    sgu = dot(h, pr["sw13"].astype(F32)) * pr["ss13"][None]
    sact = jax.nn.silu(sgu[:, : d.shared_i]) * sgu[:, d.shared_i :]
    shared = dot(sact, pr["sw2"].astype(F32)) * pr["ss2"][None]
    moe = lax.psum(shared + d.routed_scale * routed, AXIS)
    s2 = moe + s1
    h2 = _rms(s2, sh["g_in"], d.eps)
    qkv = dot(h2, sh["qkv_w"].astype(F32)) * sh["qkv_s"][None]
    q_c = _rms(qkv[:, : d.q_lora], sh["g_qa"], d.eps)
    kv = _rms(qkv[:, d.q_lora : d.q_lora + d.kv_lora], sh["g_kva"], d.eps)
    k0 = d.q_lora + d.kv_lora
    pe = _rope_f32(qkv[:, k0 : k0 + d.rope], sh["cos"], sh["sin"])
    return s2, q_c, kv, pe, s2 - sh["res"].astype(F32)


def _incumbent_device(d, T, n_valid, sh, pr, rank, dst_rows):
    """The production ops the kernel replaces, composed as the decoder runs
    them (instruments incumbent.block_body, skip exit)."""
    from vllm_torchtpu.kernels import fused_add_rmsnorm
    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as moe_kernel
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce as small_ar,
    )
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    H = d.hidden
    # cp_mla_attention._all_to_all_packed receive side + _merge_received, .to(bf16)
    recv = pr["a2a"]
    rows = T * d.heads_rank
    extra = -(-2 * rows // d.v_head)
    o_recv = lax.bitcast_convert_type(recv[:, extra:], BF16).reshape(
        DCP, T, d.heads_rank, -1
    )
    lw = recv[:, :extra].reshape(DCP, -1)
    hi = lw[:, :rows].astype(I32)
    lo = lw[:, rows : 2 * rows].astype(I32)
    l_recv = lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), F32).reshape(
        DCP, T, -1
    )
    fin = jnp.isfinite(l_recv)
    shift = jnp.max(jnp.where(fin, l_recv, -jnp.inf), axis=0)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    w = jnp.where(fin, jnp.exp(l_recv - shift[None]), 0.0)
    w = w / jnp.where(w.sum(0) > 0, w.sum(0), 1.0)
    of = o_recv.astype(F32)
    of = jnp.where(jnp.isnan(of), 0.0, of)
    w_hi = w.astype(BF16).astype(F32)
    w_lo = w - w_hi
    attn = (jnp.sum(of * w_hi[..., None], 0) + jnp.sum(of * w_lo[..., None], 0)).astype(
        BF16
    )
    attn = attn.reshape(T, -1)
    # o_proj: linear_common._fp8_linear_pallas
    w_o, s_o = pr["oproj_w"], pr["oproj_s"]
    if fp8_skinny.supports(T, w_o, s_o):
        y = fp8_skinny.fp8_skinny_matmul(
            attn,
            w_o,
            s_o,
            n_logical=H,
            scale_dma=fp8_skinny.scale_dma_applies(*w_o.shape),
        )
    else:
        y = xla_quantized_matmul(attn, w_o, s_o)
    # small_all_reduce_op._small_all_reduce
    tile_rows = int(pltpu.get_tpu_info().get_sublane_tiling(BF16))
    nv = jnp.full((1,), n_valid, I32)
    if small_ar.fold_geometry(T, H, N_RANKS, tile_rows) is not None:
        attn_sum = small_ar.small_all_reduce_valid_rows(
            y, nv, rank, num_cores=N_RANKS, axis=AXIS, tile_rows=tile_rows
        )
    else:
        attn_sum = small_ar.small_all_reduce(y, num_cores=N_RANKS, axis=AXIS, rank=rank)
    h, res1 = fused_add_rmsnorm.reference(attn_sum, sh["res"], sh["g_post"], d.eps)
    scores = jax.nn.sigmoid(
        lax.dot_general(
            h, sh["gate"], (((1,), (1,)), ((), ())), preferred_element_type=F32
        )
    )
    m = max(moe_kernel.ROW_BUCKETS[0], T)
    moe = moe_kernel.moe_tp_decode(
        jnp.pad(h, ((0, m - T), (0, 0))),
        jnp.pad(scores, ((0, m - T), (0, 0))),
        sh["bias"],
        jnp.int32(n_valid),
        rank.reshape(()),
        pr["w13"],
        pr["w2"],
        pr["s13"],
        pr["s2"],
        pr["sw13"],
        pr["ss13"],
        pr["sw2"],
        pr["ss2"],
        jnp.int32(0),
        routed_scale=d.routed_scale,
        topk=d.topk,
        n_ranks=N_RANKS,
    )[:T]
    h2, res2 = fused_add_rmsnorm.reference(moe, res1, sh["g_in"], d.eps)
    width = d.qkv_width
    qkv = xla_quantized_matmul(h2, sh["qkv_w"][:, :width], sh["qkv_s"][:width])
    q_c = fused_add_rmsnorm.reference(qkv[:, : d.q_lora], None, sh["g_qa"], d.eps)
    kv_c = qkv[:, d.q_lora : d.q_lora + d.kv_lora]
    k_pe = qkv[:, d.q_lora + d.kv_lora :]
    kv_n = fused_add_rmsnorm.reference(kv_c, None, sh["g_kva"], d.eps)
    half = d.rope // 2
    c, s = sh["cos"], sh["sin"]
    x1, x2 = k_pe[:, :half], k_pe[:, half:]
    k_pe = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(BF16)
    kv8 = jnp.clip(kv_n.astype(F32), -448.0, 448.0).astype(F8)
    pe8 = jnp.clip(k_pe.astype(F32), -448.0, 448.0).astype(F8)

    def lane_dense(values, token_bytes):  # attention_interface._pack_lane_dense
        u8 = lax.bitcast_convert_type(values, jnp.uint8)
        band = token_bytes // 4
        words = None
        for b in range(4):
            if b * band >= u8.shape[-1]:
                break
            part = u8[:, b * band : (b + 1) * band].astype(jnp.uint32) << (8 * b)
            words = part if words is None else words | part
        return jnp.pad(words, ((0, 0), (0, 128 - words.shape[-1])))

    kv_pack = native_sc_gather.pack_write_rows(
        dst_rows, lane_dense(kv8, d.kv_lora), lane_dense(pe8, 128)
    )
    return res2, q_c, kv_pack


# =====================================================================================
# Decoding and metrics
# =====================================================================================
def _decode_pack(kvp, T, tp):
    """u32 [3 Tp, 128] -> (nope fp8 values [T, 512], rope [T, 64], enc [Tp, 128])."""
    u = np.asarray(kvp).view(np.uint32)
    nope_w, rope_w, enc = u[:tp], u[tp : 2 * tp], u[2 * tp : 3 * tp]
    nb = np.concatenate([((nope_w >> (8 * b)) & 0xFF) for b in range(4)], axis=1)
    rb = np.concatenate(
        [((rope_w[:, :32] >> (8 * b)) & 0xFF) for b in range(2)], axis=1
    )
    f8 = lambda b: np.asarray(  # noqa: E731
        jnp.asarray(b.astype(np.uint8)).view(F8).astype(F32)
    )
    return f8(nb[:T]), f8(rb[:T]), enc


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float(np.linalg.norm(a))


def _metrics(d, T, n_valid, sh, out, ref, dst):
    """Per device: worst row-relative error per metric over the real rows,
    plus exact checks (KV destinations, padding rows of the pack)."""
    res_out, q_c, kvp = (
        np.asarray(x, np.float32) if i < 2 else np.asarray(x) for i, x in enumerate(out)
    )
    s2, q_ref, kv_ref, pe_ref, delta_ref = (np.asarray(x, np.float64) for x in ref)
    res_in = np.asarray(sh["res"], np.float64)
    tp = -(-T // 16) * 16
    nope, rope, enc = _decode_pack(kvp, T, tp)
    kv_ref8 = np.asarray(
        jnp.asarray(np.clip(kv_ref, -448, 448), F32).astype(F8).astype(F32)
    )
    pe_ref8 = np.asarray(
        jnp.asarray(np.clip(pe_ref, -448, 448), F32).astype(F8).astype(F32)
    )
    m = {k: 0.0 for k in METRICS}
    for t in range(n_valid):
        m["res_delta"] = max(m["res_delta"], _rel(res_out[t] - res_in[t], delta_ref[t]))
        m["q_c"] = max(m["q_c"], _rel(q_c[t], q_ref[t]))
        m["kv_nope"] = max(m["kv_nope"], _rel(nope[t], kv_ref8[t]))
        m["kv_rope"] = max(m["kv_rope"], _rel(rope[t], pe_ref8[t]))
    want_enc = np.zeros((tp, 128), np.uint32)
    want_enc[:T] = np.where(dst >= 0, dst + 1, 0).astype(np.uint32)[:, None]
    u = kvp.view(np.uint32)
    flags = {
        "enc_exact": bool(np.array_equal(enc, want_enc)),
        "pad_rows_zero": bool((u[T:tp] == 0).all() and (u[tp + T : 2 * tp] == 0).all()),
        "finite": bool(
            np.isfinite(res_out[:n_valid]).all() and np.isfinite(q_c[:n_valid]).all()
        ),
    }
    return m, flags


# =====================================================================================
# Drivers
# =====================================================================================
def _mesh(devices):
    return Mesh(np.asarray(devices), (AXIS,))


def _run_case(mesh, d, T, n_valid, seed, *, interpret=None, faults=(), incumbent=False):
    """-> {build: per-device [(metrics, flags)]} for "kernel", each fault, and
    "incumbent" when asked. Inputs are built on device, per rank."""
    rank_ids = jax.device_put(
        np.arange(N_RANKS, dtype=np.int32).reshape(N_RANKS, 1),
        NamedSharding(mesh, P(AXIS)),
    )
    shared = _shared_inputs(d, T, n_valid, seed)
    sh = {k: v for k, v in shared.items() if k not in ("dst", "planted")}
    sh_dev = {k: jax.device_put(v, NamedSharding(mesh, P())) for k, v in sh.items()}
    # [16, T]: device r is DCP rank r % DCP (vLLM's DCP groups inside TP).
    dst_all = shared["dst"][np.arange(N_RANKS) % DCP]
    dst_sh = jax.device_put(dst_all, NamedSharding(mesh, P(AXIS)))
    sh_specs = {k: P() for k in sh}

    def body(build, rank, dst, sh_):
        pr = _per_rank_inputs(d, T, n_valid, seed, rank[0])
        dst_rows = dst[0]
        if build == "reference":
            outs = _reference_device(d, T, n_valid, sh_, pr, rank[0])
        elif build == "incumbent":
            outs = _incumbent_device(d, T, n_valid, sh_, pr, rank[0], dst_rows)
        else:
            fault = None if build == "kernel" else build
            outs = _kernel_device(
                d,
                T,
                n_valid,
                sh_,
                pr,
                rank[0],
                dst_rows,
                fault=fault,
                interpret=interpret,
            )
        return tuple(x[None] for x in outs)

    def run(build):
        n_out = 5 if build == "reference" else 3
        fn = jax.jit(
            jax.shard_map(
                functools.partial(body, build),
                mesh=mesh,
                in_specs=(P(AXIS), P(AXIS), sh_specs),
                out_specs=tuple(P(AXIS) for _ in range(n_out)),
                check_vma=False,
            )
        )
        return fn(rank_ids, dst_sh, sh_dev)

    ref = run("reference")
    builds = ["kernel", *faults] + (["incumbent"] if incumbent else [])
    results = {}
    local = [
        i
        for i, dv in enumerate(mesh.devices)
        if dv.process_index == jax.process_index()
    ]
    for b in builds:
        out = run(b)
        rows = []
        for i in local:
            shard = lambda x, i=i: np.asarray(  # noqa: E731
                [s for s in x.addressable_shards if s.device == mesh.devices[i]][
                    0
                ].data[0]
            )
            o_i = tuple(shard(x) for x in out)
            r_i = tuple(shard(x) for x in ref)
            rows.append(_metrics(d, T, n_valid, sh, o_i, r_i, np.asarray(dst_all[i])))
        results[b] = rows
    return results


def _worst(rows):
    m = {k: max(r[0][k] for r in rows) for k in METRICS}
    flags = {k: all(r[1][k] for r in rows) for k in rows[0][1]}
    return m, flags


def _cpu_worker():
    """Subprocess body: 16 host devices, interpret mode with races detected."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.get_tpu_info = _tpu7x
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.get_tpu_info = _tpu7x
    mesh = _mesh(sorted(jax.devices("cpu"), key=lambda dv: dv.id)[:N_RANKS])
    out = {}
    tiny = os.environ.get("SEGMK_TEST_TINY") == "1"
    dims = TINY if tiny else SMALL
    cases = CPU_CASES[:1] if tiny else CPU_CASES
    for i, (T, nv) in enumerate(cases):
        t0 = time.time()
        pltpu.reset_tpu_interpret_mode_state()
        faults = FAULTS if i == 0 and not tiny else ()
        res = _run_case(mesh, dims, T, nv, seed=T * 100 + nv, faults=faults)
        races = bool(interpret_pallas_call.races.races_found)
        out[f"T{T}_nv{nv}"] = {
            "races": races,
            "seconds": round(time.time() - t0, 1),
            **{b: _worst(rows) for b, rows in res.items()},
        }
    print("RESULT " + json.dumps(out), flush=True)


@pytest.fixture(scope="module")
def cpu_results():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=16",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "--cpu-worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_admits_the_decode_buckets_at_glm52_dims(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    for T in seg.TOKEN_BUCKETS:
        assert seg.supported(GLM52.cfg(T)) is None, T
    for T in (2, 12, 64):
        assert seg.supported(GLM52.cfg(T)) is not None, T
    assert seg.supported(dataclasses.replace(GLM52, rope=32).cfg(4)) is not None


def test_kernel_matches_the_oracle_on_16_devices(cpu_results):
    for case, r in cpu_results.items():
        m, flags = r["kernel"]
        assert not r["races"], case
        assert all(flags.values()), (case, flags)
        bad = {k: v for k, v in m.items() if v > CPU_TOL[k]}
        assert not bad, (case, m)


@pytest.mark.parametrize("fault", FAULTS)
def test_positive_controls_fail_the_oracle(cpu_results, fault):
    runs = [r[fault] for r in cpu_results.values() if fault in r]
    assert runs, fault
    for m, flags in runs:
        assert any(v > CPU_TOL[k] for k, v in m.items()) or not all(flags.values()), (
            fault,
            m,
            flags,
        )


def _tpu_main():
    pid = int(os.environ.get("JOBSET_JOB_INDEX", os.environ.get("TPU_WORKER_ID", "0")))
    nproc = int(os.environ.get("NPROC", "1"))
    if nproc > 1:
        jax.distributed.initialize(os.environ["COORD"], nproc, pid)
    devices = sorted(jax.devices(), key=lambda dv: (dv.process_index, dv.id))
    assert len(devices) == N_RANKS, f"need {N_RANKS} devices, have {len(devices)}"
    mesh = _mesh(devices)
    from jax.experimental import multihost_utils

    summary, ok = {}, True
    for T, nv in TPU_CASES:
        t0 = time.time()
        res = _run_case(
            mesh, GLM52, T, nv, seed=T * 100 + nv, faults=FAULTS, incumbent=True
        )
        mine = {b: _worst(rows) for b, rows in res.items()}
        allm = multihost_utils.process_allgather(
            np.asarray([[mine[b][0][k] for k in METRICS] for b in res]), tiled=False
        )
        allf = multihost_utils.process_allgather(
            np.asarray([[all(mine[b][1].values())] for b in res]), tiled=False
        )
        worst = {
            b: dict(zip(METRICS, np.max(allm[:, j], axis=0).tolist()))
            for j, b in enumerate(res)
        }
        flags = {b: bool(np.all(allf[:, j])) for j, b in enumerate(res)}
        tol = {
            k: max(TOL_FACTOR * worst["incumbent"][k], TOL_FLOOR[k]) for k in METRICS
        }
        kernel_pass = flags["kernel"] and all(
            worst["kernel"][k] <= tol[k] for k in METRICS
        )
        caught = {
            f: (not flags[f]) or any(worst[f][k] > tol[k] for k in METRICS)
            for f in FAULTS
        }
        ok = ok and kernel_pass and flags["incumbent"] and all(caught.values())
        summary[f"T{T}_nv{nv}"] = dict(
            tol=tol,
            worst=worst,
            flags=flags,
            kernel_pass=kernel_pass,
            caught=caught,
            seconds=round(time.time() - t0, 1),
        )
        if jax.process_index() == 0:
            print(
                f"[segmk] T={T} n_valid={nv} kernel_pass={kernel_pass} caught={caught}",
                flush=True,
            )
            for b in res:
                print(
                    f"    {b:18s} "
                    + " ".join(f"{k}={worst[b][k]:.4f}" for k in METRICS)
                    + f" flags={flags[b]}",
                    flush=True,
                )
            print(
                "    tol                "
                + " ".join(f"{k}={tol[k]:.4f}" for k in METRICS),
                flush=True,
            )
    if jax.process_index() == 0:
        print(
            "SEGMK_TPU_RESULT " + json.dumps({"pass": ok, "cases": summary}), flush=True
        )
    return 0 if ok else 1


if __name__ == "__main__":
    if "--cpu-worker" in sys.argv:
        # Around everything: pallas_call reads the mode when it is built.
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            _cpu_worker()
    elif "--tpu" in sys.argv:
        sys.exit(_tpu_main())
    else:
        sys.exit(pytest.main([__file__, "-v"]))
