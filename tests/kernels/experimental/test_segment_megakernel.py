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
    KV destination, the next layer's weights in the wrong slot, W_UV's heads
    rotated) must fail it.
  * TPU 2x2x2 (16 cores, 2 hosts), full GLM-5.2 TP16 x DCP8 dims:
        python3 test_segment_megakernel.py --tpu
    under jax.distributed (simple-run-command-2node exports COORD / NPROC /
    JOBSET_JOB_INDEX). Runs the kernel AND the incumbent composition of the
    production ops it replaces (the backend's bf16 W_UV bmm after the merge,
    the fused fp8 o_proj + all-reduce kernel at <= 16 rows / fp8_skinny + the
    small valid-rows all-reduce at 32, fused_add_rmsnorm.reference, the router
    dot, moe_tp_decode on its fast schedule, xla_quantized_matmul, the bf16
    NeoX rope, quantize_kv, pack_write_rows) on decode buckets b1/b2/b4/b8
    (T = 4/8/16/32) and a
    padded b3, compiled without excess precision as torch_tpu compiles the
    served model, judged by the campaign's rule: tol = max(1.5 x the incumbent's
    error, floor) per metric; the kernel passes if under tol everywhere.
  * Routing (both modes): on real-bias-like router inputs (GLM-5.2's
    e_score_correction_bias offset and spread) the kernel's routing must pick
    the same top-k sets, with the same weights, as production's -- GateLinear's
    bf16 logits, XLA sigmoid, moe_tp_decode's `_route` -- while the f32-logit
    form must not (the positive control).
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
FAULTS = ("oproj_scale_roll", "norms_swapped", "wrong_dst", "wrong_qkv", "wuv_heads_roll")


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


# =====================================================================================
# Inputs, generated per rank on device
# =====================================================================================
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

    # The merged DCP attention output (latent, a convex mix of KV rows) and
    # this rank's heads of W_UV, bf16 as the backend keeps them.
    lat = jax.random.normal(next(ks), (T, d.heads_rank, d.kv_lora), F32)
    wuv = jax.random.normal(next(ks), (d.heads_rank, d.kv_lora, d.v_head), F32)
    return dict(
        lat=lat.astype(BF16),
        wuv=(wuv / np.sqrt(d.kv_lora)).astype(BF16),
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
def _kernel_device(
    d,
    T,
    n_valid,
    sh,
    pr,
    rank,
    dst_rows,
    fault=None,
    interpret=None,
    pf=None,
    ops_only=False,
):
    """pf (speculative expert prefetch): dict(cfg=<prefetch SegCfg>, pref=int32
    [8, 128] when cfg.pf_src == "input")."""
    cfg = d.cfg(T) if pf is None else pf["cfg"]
    g_post, g_in = sh["g_post"], sh["g_in"]
    oproj_s = pr["oproj_s"]
    qkv_w, qkv_s = sh["qkv_w"], sh["qkv_s"]
    wuv = pr["wuv"]
    if fault == "wuv_heads_roll":
        wuv = jnp.roll(wuv, 1, axis=0)
    elif fault == "oproj_scale_roll":
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
    )
    # Two boundaries in the stacks, this one in row 1 (row 0 a decoy: wrong
    # row indexing shows up as garbage constants).
    row = seg.fstack_row(cfg, consts)
    decoy = jnp.flip(row) + 1.0
    ops = dict(
        meta=seg.pack_meta(n_valid, rank[0], 1, dst_rows),
        lat=pr["lat"].reshape(T, -1),
        act=seg.pack_act(sh["res"], cfull, sfull),
        wuv=wuv,
        oproj_w=pr["oproj_w"],
        gate=sh["gate"],
        w13=pr["w13"],
        w2=pr["w2"],
        s2=pr["s2"],
        qkv_w=qkv_w,
        fstack=jnp.stack([decoy, row])[:, None, :],
        s13s=jnp.stack([jnp.flip(pr["s13"][:, 0], 0), pr["s13"][:, 0]]),
        sw13s=jnp.stack([jnp.flip(pr["sw13"], 0), pr["sw13"]]),
        sw2s=jnp.stack([jnp.flip(pr["sw2"], 0), pr["sw2"]]),
    )
    if pf is not None:
        if cfg.qkv_blocked:
            ops["qkv_w"] = seg.block_qkv(qkv_w, cfg.qkv_blk)
        if cfg.pf_src == "input":
            ops["pref"] = pf["pref"]
    if ops_only:
        return cfg, ops, (cfull, sfull)
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
    attn = jnp.einsum(
        "thk,hkv->thv",
        pr["lat"].astype(F32),
        pr["wuv"].astype(F32),
        precision=HI,
    ).reshape(T, -1)
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


def _production_scores(h, gate):
    """GateLinear on TPU (F.linear in bf16: bf16 logits) then
    `moe_routing._apply_scoring_fn`: .float().sigmoid()."""
    logits = lax.dot_general(h, gate, (((1,), (1,)), ((), ())))  # bf16 out
    assert logits.dtype == BF16
    return jax.nn.sigmoid(logits.astype(F32))


def _incumbent_device(d, T, n_valid, sh, pr, rank, dst_rows):
    """The production ops the kernel replaces, composed as the decoder runs
    them (instruments incumbent.block_body, skip exit)."""
    from vllm_torchtpu.kernels import fused_add_rmsnorm
    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as moe_kernel
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce as small_ar,
    )
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
    from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce, fp8_skinny
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    H = d.hidden
    # The backend after the merge: torch.bmm(outputs_t, W_UV) in bf16 (f32
    # accumulation, the output rounded to bf16), heads back to [T, h * v].
    lat_t = jnp.transpose(pr["lat"], (1, 0, 2))  # [h, T, kv]
    attn = lax.dot_general(
        lat_t, pr["wuv"], (((2,), (1,)), ((0,), (0,)))
    )  # bf16 [h, T, v]
    assert attn.dtype == BF16
    attn = jnp.transpose(attn, (1, 0, 2)).reshape(T, -1)
    w_o, s_o = pr["oproj_w"], pr["oproj_s"]
    if fp8_oproj_all_reduce.supports(T, w_o.shape[0], w_o.shape[1], N_RANKS):
        # TPU_FUSED_OPROJ_AR: small_all_reduce_op.row_parallel_or
        attn_sum = fp8_oproj_all_reduce.fp8_matmul_all_reduce(
            attn, w_o, s_o, rank, num_cores=N_RANKS, axis=AXIS
        )
    else:
        # o_proj: linear_common._fp8_linear_pallas
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
            attn_sum = small_ar.small_all_reduce(
                y, num_cores=N_RANKS, axis=AXIS, rank=rank
            )
    h, res1 = fused_add_rmsnorm.reference(attn_sum, sh["res"], sh["g_post"], d.eps)
    scores = _production_scores(h, sh["gate"])
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
        fast=True,  # TPU_MOE_TP_DECODE_FAST (the same bits as the plain schedule)
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
# Routing against production's, on real-bias-like inputs
# =====================================================================================
# GLM-5.2-FP8 layers 10/30/60 (probes/numerics-ab/router_cpu.py on the
# checkpoint): router rows |W| 1.8-4.5, e_score_correction_bias mean 4-10 and
# std 0.03-0.05, post_attention_layernorm |g| 0.05-0.44. The bias offset makes
# the f32 sum coarse and the bias spread is the size of the score gaps, so the
# bf16 rounding of the logits decides the top-k set on 2-3% of rows.
ROUTE_ROWS, ROUTE_CALLS = 32, 32
ROUTE_LAYERS = (
    (4.465, 6.986, 0.0402, 0.055),
    (3.128, 3.984, 0.0481, 0.186),
    (1.824, 10.447, 0.0311, 0.440),
)  # (|W row|, bias mean, std, |g|)


def _routing_inputs(d: Dims, layer: int, call: int):
    w_norm, b_mean, b_std, g_abs = ROUTE_LAYERS[layer]
    rng = np.random.default_rng(7919 * layer + call)
    H, E, n = d.hidden, d.experts, ROUTE_ROWS
    gate = rng.standard_normal((E, H))
    gate *= w_norm / np.linalg.norm(gate, axis=1, keepdims=True)
    gate = _host(gate, BF16)
    bias = (b_mean + b_std * rng.standard_normal(E)).astype(np.float32)
    g = g_abs * (1 + 0.3 * rng.standard_normal(H))
    # Half the rows unaligned, half leaning on 8 router directions.
    s = rng.standard_normal((n, H))
    pick = np.stack([rng.choice(E, d.topk, replace=False) for _ in range(n // 2)])
    lean = np.asarray(gate, np.float32)[pick].sum(1)
    lean /= np.linalg.norm(lean, axis=1, keepdims=True)
    s[n // 2 :] += lean * np.sqrt(H) * rng.uniform(0.3, 1.0, (n // 2, 1)) / g_abs
    h = s / np.sqrt(np.mean(s * s, axis=1, keepdims=True)) * g
    return _host(h, BF16), gate, bias[None, :]


@functools.cache
def _routing_fns(topk: int, n: int, E: int, interpret: bool):
    """Jitted (kernel, production router, production `_route`, f32-logit
    control) for [n, H] tokens and E experts."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as moe_kernel
    from vllm_torchtpu.kernels.experimental.segment_megakernel import kernel as sk

    nt = (((1,), (1,)), ((), ()))

    def pcall(fn):
        def body(*refs):
            refs[-1][...] = fn(*(r[...] for r in refs[:-1]))

        return jax.jit(
            pl.pallas_call(
                body,
                out_shape=jax.ShapeDtypeStruct((n, E), F32),
                compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 2**20),
                interpret=interpret,
            )
        )

    def prod_router(a, g):  # GateLinear (bf16 logits) + .float().sigmoid()
        lg = lax.dot_general(a, g, nt).astype(F32)
        return lg, jax.nn.sigmoid(lg)

    def f32_logits(a, g, b):
        lg = lax.dot_general(a, g, nt, preferred_element_type=F32)
        return sk._route_argmax(jax.nn.sigmoid(lg), b, n, topk=topk)

    return (
        pcall(lambda a, g, b: sk.route_weights(a, g, b, n, topk=topk)),
        jax.jit(prod_router),
        pcall(lambda sc, b: moe_kernel._route(sc, b, jnp.int32(n), topk=topk)),
        pcall(f32_logits),
    )


def _routing_tables(d: Dims, h, gate, bias, *, interpret):
    """-> (kernel, production, kernel-with-f32-logits) routing tables [rows, E]:
    the kernel's routing (router_scores + _route_argmax in Mosaic), production's
    (XLA GateLinear + sigmoid, then moe_tp_decode's `_route` in Mosaic), and the
    pre-fix f32-logit form as the positive control."""
    kern_fn, router_fn, route_fn, old_fn = _routing_fns(
        d.topk, h.shape[0], gate.shape[0], bool(interpret)
    )
    lg32, scores = router_fn(h, gate)
    # torch_tpu compiles with allow_excess_precision off, so the served logits
    # are bf16 values; this process must too (XLA_FLAGS), or XLA folds the
    # rounding away and the "production" side is not production.
    lg32 = np.asarray(lg32)
    assert np.array_equal(
        np.asarray(jnp.asarray(lg32).astype(BF16).astype(F32)), lg32
    ), "logits not bf16: run with XLA_FLAGS=--xla_allow_excess_precision=false"
    kern = kern_fn(h, gate, bias)
    prod = route_fn(scores, bias)
    old = old_fn(h, gate, bias)
    return tuple(np.asarray(x) for x in (kern, prod, old))


def routing_vs_production(d: Dims, *, interpret, calls=ROUTE_CALLS):
    """Rows whose top-k SET differs from production's, for the kernel and the
    f32-logit control; the max routing-weight difference where the sets agree."""
    out = dict(rows=0, kernel_set_diff=0, f32_logit_set_diff=0, kernel_w_maxdiff=0.0)
    for layer in range(len(ROUTE_LAYERS)):
        for call in range(calls):
            kern, prod, old = _routing_tables(
                d, *_routing_inputs(d, layer, call), interpret=interpret
            )
            ks, ps, os_ = kern != 0, prod != 0, old != 0
            assert (ps.sum(1) == d.topk).all()
            out["rows"] += ks.shape[0]
            out["kernel_set_diff"] += int(np.any(ks != ps, axis=1).sum())
            out["f32_logit_set_diff"] += int(np.any(os_ != ps, axis=1).sum())
            same = np.all(ks == ps, axis=1)
            if same.any():
                out["kernel_w_maxdiff"] = max(
                    out["kernel_w_maxdiff"],
                    float(np.abs(kern[same] - prod[same]).max()),
                )
    return out


def test_routing_matches_production_gate_linear():
    """Identical top-k sets to production's bf16-logit router on real-bias-like
    inputs; the f32-logit form must differ somewhere (the check can fail).
    Interpret mode on CPU in a subprocess without excess precision: XLA:CPU
    otherwise drops the bf16 rounding of the logits that XLA:TPU keeps."""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "--routing-worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("ROUTING ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    r = json.loads(lines[-1][len("ROUTING ") :])
    assert r["f32_logit_set_diff"] > 0, r
    assert r["kernel_set_diff"] == 0, r
    # XLA:CPU's logistic is not 1 / (1 + exp(-x)) (on TPU they agree bit for
    # bit and --tpu asserts equal weights): at most one bf16 ulp of a weight.
    assert r["kernel_w_maxdiff"] <= 2**-8, r


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


# =====================================================================================
# Speculative expert prefetch (TPU_SEGMK_EXPERT_PREFETCH): only data movement
# changes, so the oracle is bit-identity with the prefetch-off kernel
# =====================================================================================
PF_KINDS = ("hit", "mixed", "wrong", "garbage", "none")
PF_FAULTS = ("stale_slot", "no_dma")


def _prefs(d: Dims, planted, n_valid, P, seed):
    """Host-side pool id lists [8, 128] int32 per kind, from the planted
    routing of the valid rows: hit = the P most-picked planted experts; mixed =
    half of those + experts no row picks; wrong = only experts no row picks;
    garbage = duplicates, -1, out-of-range ids among planted ones; none = -1."""
    rng = np.random.default_rng(seed + 77)
    pk = planted[:n_valid].ravel()
    ids, cnt = np.unique(pk, return_counts=True)
    order = ids[np.lexsort((ids, -cnt))]
    unpicked = np.setdiff1d(np.arange(d.experts), ids)
    rng.shuffle(unpicked)
    lists = {
        "hit": list(order[:P]),
        "mixed": list(order[: (P + 1) // 2]) + list(unpicked[: P // 2]),
        "wrong": list(unpicked[:P]),
        "garbage": [order[0], order[0], -1, d.experts + 44, -7]
        + list(order[1:P])
        + list(unpicked[:P]),
        "none": [],
    }
    out = {}
    for k, v in lists.items():
        a = np.full((8, 128), -1, np.int32)
        v = np.asarray(v[:P], np.int32)
        a[0, : len(v)] = v
        out[k] = a
    picked = set(ids.tolist())
    hits = {
        k: len({int(e) for e in a[0, :P] if 0 <= e < d.experts} & picked)
        for k, a in out.items()
    }
    return out, hits, len(picked)


def predictor_vs_reference(d: Dims, *, interpret, calls=8):
    """The in-kernel predictor (`_predict_ids`' arithmetic: rms_add,
    router_scores, predict_mass, rank_experts in Mosaic) against the same
    rule in XLA (production GateLinear + sigmoid), on real-bias-like router
    inputs at every bucket: the kernel's ids must be a top-P set of the
    reference mass up to near-ties (_pred_ok)."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.experimental.segment_megakernel import kernel as sk

    out = {}
    for T in seg.TOKEN_BUCKETS:
        m = 16 if T <= 16 else 32
        n_ids = 12
        ok, n = True, 0
        for layer in range(len(ROUTE_LAYERS)):
            for call in range(calls):
                h, gate, bias = _routing_inputs(d, layer, call)
                x = jnp.pad(jnp.asarray(h[:T]).astype(F32), ((0, m - T), (0, 0)))
                g = jnp.asarray(
                    1 + 0.3 * np.random.default_rng(call).standard_normal(d.hidden),
                    F32,
                ).astype(BF16)[None]

                def body(x_ref, g_ref, w_ref, b_ref, o_ref, T=T):
                    hp, _ = sk.rms_add(x_ref[...], None, g_ref[...], jnp.float32(d.eps))
                    sc = sk.router_scores(hp, w_ref[...])
                    o_ref[...] = sk.rank_experts(
                        sk.predict_mass(sc, b_ref[...], T, topk=d.topk), n_ids
                    )

                got = np.asarray(
                    pl.pallas_call(
                        body,
                        out_shape=jax.ShapeDtypeStruct((8, 128), I32),
                        compiler_params=pltpu.CompilerParams(
                            vmem_limit_bytes=64 * 2**20
                        ),
                        interpret=interpret,
                    )(x, g, jnp.asarray(gate), jnp.asarray(bias))
                )
                hp, _ = sk.rms_add(x, None, g, jnp.float32(d.eps))
                scores = _production_scores(hp, jnp.asarray(gate))
                mass = np.asarray(
                    sk.predict_mass(scores, jnp.asarray(bias), T, topk=d.topk)
                )[0]
                ok = ok and _pred_ok(got[0, :n_ids], None, mass) and (got[1:] == -1).all()
                n += 1
        out[f"T{T}"] = dict(ok=bool(ok), calls=n)
    return out


PRED_TIE_TOL = 2e-3  # mass units: a routed weight is O(1/topk)


def _pred_ok(got, _ref_ids, mass, tol=PRED_TIE_TOL):
    """The kernel's ids are a top-P set of the reference mass up to near-ties
    (the two sum the bf16 routing weights in different orders): distinct,
    valid, every chosen expert's mass >= every unchosen one's - tol, and the
    order non-increasing in mass up to tol."""
    pos = [int(e) for e in got if e >= 0]
    n_pos = int((mass > 0).sum())
    if len(set(pos)) != len(pos) or len(pos) != min(len(got), n_pos):
        return False
    if not pos:
        return True
    chosen = mass[pos]
    rest = np.delete(mass, pos)
    if rest.size and chosen.min() < rest.max() - tol:
        return False
    return bool(np.all(np.diff(chosen) <= tol))


def _pf_cfg(d: Dims, T: int, e_pre: int, qkv_early: int = 0, **kw):
    cfg = seg.prefetch_fit(d.cfg(T), e_pre, qkv_early)
    assert cfg is not None, (T, e_pre)
    return dataclasses.replace(cfg, **kw)


def _run_prefetch_case(
    mesh, d, T, n_valid, seed, e_pre, *, interpret=None, timing=(), sweep=()
):
    """Bit-identity with the prefetch-off kernel on every device: the
    production form (pool ids predicted in the call, "self") with its
    removal arms, and controlled pools ("input": hit / mixed / wrong /
    garbage / none) with the positive controls. timing: M values for a
    chained per-call clock of the arms (TPU only)."""
    rank_ids = jax.device_put(
        np.arange(N_RANKS, dtype=np.int32).reshape(N_RANKS, 1),
        NamedSharding(mesh, P(AXIS)),
    )
    shared = _shared_inputs(d, T, n_valid, seed)
    sh = {k: v for k, v in shared.items() if k not in ("dst", "planted")}
    sh_dev = {k: jax.device_put(v, NamedSharding(mesh, P())) for k, v in sh.items()}
    dst_all = shared["dst"][np.arange(N_RANKS) % DCP]
    dst_sh = jax.device_put(dst_all, NamedSharding(mesh, P(AXIS)))
    sh_specs = {k: P() for k in sh}
    e_pre, qkv_early = e_pre if isinstance(e_pre, tuple) else (e_pre, 0)
    self_cfg = _pf_cfg(d, T, e_pre, qkv_early)
    in_cfg = dataclasses.replace(self_cfg, pf_src="input")
    Pn = self_cfg.e_pre
    prefs, hits, n_routed = _prefs(d, shared["planted"], n_valid, Pn, seed)
    local = [
        i
        for i, dv in enumerate(mesh.devices)
        if dv.process_index == jax.process_index()
    ]

    def build(cfg):
        def body(rank, dst, sh_, pref):
            pr = _per_rank_inputs(d, T, n_valid, seed, rank[0])
            pf = None if cfg is None else dict(cfg=cfg, pref=pref[0])
            outs = _kernel_device(
                d, T, n_valid, sh_, pr, rank[0], dst[0], interpret=interpret, pf=pf
            )
            return tuple(x[None] for x in outs)

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P(AXIS), P(AXIS), sh_specs, P(AXIS)),
                out_specs=tuple(P(AXIS) for _ in range(3)),
                check_vma=False,
            )
        )

    def pref_dev(kind):
        a = np.broadcast_to(prefs[kind][None], (N_RANKS, 8, 128))
        return jax.device_put(np.ascontiguousarray(a), NamedSharding(mesh, P(AXIS)))

    def shards(out):
        res = []
        for i in local:
            res.append(
                tuple(
                    np.asarray(
                        [s for s in x.addressable_shards if s.device == mesh.devices[i]][
                            0
                        ].data[0]
                    )
                    for x in out
                )
            )
        return res

    def same(a, b):  # bit identity of res / q_c / kv rows on every local device
        return bool(
            all(
                all(
                    np.array_equal(
                        np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)
                    )
                    for x, y in zip(da, db)
                )
                for da, db in zip(a, b)
            )
        )

    def run(cfg, kind="none"):
        return shards(build(cfg)(rank_ids, dst_sh, sh_dev, pref_dev(kind)))

    base = run(None)
    out = dict(
        e_pre=Pn,
        qkv_early=self_cfg.qkv_early,
        nb=self_cfg.nb,
        vmem_limit_mib=round(seg.kernel.vmem_need_bytes(self_cfg)["limit"] / 2**20, 2),
        routed_experts=n_routed,
        planned_hits=hits,
        exact={},
        caught={},
    )
    out["exact"]["self"] = same(run(self_cfg), base)
    out["exact"]["self_issue_only"] = same(
        run(dataclasses.replace(self_cfg, pf_use=False)), base
    )
    out["exact"]["self_no_issue"] = same(
        run(dataclasses.replace(self_cfg, pf_issue=False)), base
    )
    for kind in PF_KINDS:
        out["exact"][f"input_{kind}"] = same(run(in_cfg, kind), base)
    out["exact"]["input_issue_only"] = same(
        run(dataclasses.replace(in_cfg, pf_use=False), "hit"), base
    )
    for f in PF_FAULTS:
        fcfg = dataclasses.replace(in_cfg, pf_fault=f)
        run(in_cfg, "wrong")  # VMEM holds other experts first ("no_dma")
        out["caught"][f] = not same(run(fcfg, "hit"), base)

    def pref_for(c, kind):
        a = _prefs(d, shared["planted"], n_valid, c.e_pre, seed)[0][kind]
        return jax.device_put(
            np.ascontiguousarray(np.broadcast_to(a[None], (N_RANKS, 8, 128))),
            NamedSharding(mesh, P(AXIS)),
        )

    sweep_cfgs = {}
    for P_, K_ in sweep:
        c = _pf_cfg(d, T, P_, K_)
        name = f"P{c.e_pre}K{c.qkv_early}"
        sweep_cfgs[name] = c
        ci = dataclasses.replace(c, pf_src="input")
        out["exact"][f"{name}_self"] = same(run(c), base)
        out["exact"][f"{name}_input_hit"] = same(
            shards(build(ci)(rank_ids, dst_sh, sh_dev, pref_for(c, "hit"))), base
        )
    if timing:
        if jax.process_index() == 0:
            print(f"[segmk-pf] T={T} nv={n_valid} correctness {json.dumps(out)}", flush=True)
        out["timing_us"] = _pf_timing(
            mesh,
            d,
            T,
            n_valid,
            seed,
            sweep_cfgs or {"main": self_cfg},
            timing,
            (rank_ids, dst_sh, sh_dev),
            pref_for,
            sh_specs,
        )
    return out


def _pf_timing(mesh, d, T, n_valid, seed, cfgs, Ms, args, pref_for, sh_specs):
    """Per-call device time of each arm: M chained calls in one program (each
    call's residual input depends on the previous call's output through an
    optimization barrier), every operand built outside the loop, slope of
    the median wall time over M (M2). Arms: A0 the prefetch-off kernel; per
    (pool, early q_a/kv_a blocks) config: L the layout alone (empty pool, no
    predictor), S the production form (self-predicted pool), S_io its
    removal arm (pool DMAs issued, never used), I_hit a planted pool."""
    first = next(iter(cfgs.values()))
    arms = {"A0_off": (None, first, "none")}
    for name, c in cfgs.items():
        ci = dataclasses.replace(c, pf_src="input")
        arms[f"{name}_L_layout"] = (ci, c, "none")
        arms[f"{name}_S_self"] = (c, c, "none")
        arms[f"{name}_S_io"] = (dataclasses.replace(c, pf_use=False), c, "none")
        arms[f"{name}_I_hit"] = (ci, c, "hit")
        if not c.qkv_blocked:
            arms[f"{name}_S_qkvlast"] = (
                dataclasses.replace(c, pf_qkv_first=False),
                c,
                "none",
            )
    # Rank skew in front of every call (served runs have it, the clock does
    # not): ranks >= tp/2 run ~2 x 12 MB of extra matmul before each call.
    skew = os.environ.get("SEGMK_PF_SKEW") == "1"
    if skew:
        for k in list(arms):
            arms[k + "+skew"] = arms[k] + ("skew",)
    keep = [x for x in os.environ.get("SEGMK_PF_ARMS", "").split(",") if x]
    if keep:  # e.g. "A0_off,_L_layout+skew": keep arms whose name ends with one
        arms = {k: v for k, v in arms.items() if any(k.endswith(x) for x in keep)}
    # The per-rank weights once, outside the timed programs.
    pr_keys = tuple(
        jax.eval_shape(
            lambda: _per_rank_inputs(d, T, n_valid, seed, jnp.zeros((1,), I32))
        ).keys()
    )
    pr_all = jax.jit(
        jax.shard_map(
            lambda rank: {
                k: v[None]
                for k, v in _per_rank_inputs(d, T, n_valid, seed, rank[0]).items()
            },
            mesh=mesh,
            in_specs=(P(AXIS),),
            out_specs={k: P(AXIS) for k in pr_keys},
            check_vma=False,
        )
    )(args[0])
    fns = {}
    for name, arm in arms.items():
        cfg, pcfg, kind = arm[:3]
        skewed = len(arm) > 3
        for M in Ms:

            def body(rank, dst, sh_, pref, prs, cfg=cfg, M=M, skewed=skewed):
                pr = {k: v[0] for k, v in prs.items()}
                pf = None if cfg is None else dict(cfg=cfg, pref=pref[0])
                # Every operand built once, outside the loop (block_qkv is an
                # fp8 relayout; nothing but the kernel may sit in the loop).
                kcfg, ops, (cfull, sfull) = _kernel_device(
                    d, T, n_valid, sh_, pr, rank[0], dst[0], pf=pf, ops_only=True
                )

                def heavy(x):
                    w = pr["oproj_w"].astype(BF16)  # [o_in, H]
                    for _ in range(2):
                        y = jnp.dot(x, w.T, preferred_element_type=F32).astype(BF16)
                        x = x + (jnp.dot(y, w, preferred_element_type=F32) * 0).astype(
                            BF16
                        )
                    return x

                def step(i, carry):
                    res_in, _ = lax.optimization_barrier((sh_["res"], carry))
                    if skewed:
                        res_in = lax.cond(
                            rank[0][0] >= N_RANKS // 2, heavy, lambda x: x, res_in
                        )
                    o = seg.segment_call(
                        kcfg, dict(ops, act=seg.pack_act(res_in, cfull, sfull))
                    )
                    return o["res_out"]

                last = lax.fori_loop(0, M, step, sh_["res"])
                return last[None]

            fn = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(
                        P(AXIS),
                        P(AXIS),
                        sh_specs,
                        P(AXIS),
                        {k: P(AXIS) for k in pr_keys},
                    ),
                    out_specs=P(AXIS),
                    check_vma=False,
                )
            )
            a = (*args, pref_for(pcfg, kind), pr_all)
            jax.block_until_ready(fn(*a))  # compile + warm
            fns[(name, M)] = (fn, a)
    rounds = int(os.environ.get("SEGMK_PF_ROUNDS", "3"))
    samples = {k: [] for k in fns}
    for _ in range(rounds):  # arms interleaved, round after round
        for key, (fn, a) in fns.items():
            for _ in range(5):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*a))
                samples[key].append(time.perf_counter() - t0)
    res = {}
    for name in arms:
        xs = np.asarray(Ms, np.float64)
        ys = np.asarray([np.median(samples[(name, M)]) for M in Ms]) * 1e6
        slope, icpt = np.polyfit(xs, ys, 1)
        # per-round slopes: the spread of the estimate
        per = []
        for rd in range(rounds):
            yr = [np.median(samples[(name, M)][rd * 5 : rd * 5 + 5]) * 1e6 for M in Ms]
            per.append(float(np.polyfit(xs, yr, 1)[0]))
        res[name] = dict(
            per_call_us=round(float(slope), 2),
            round_slopes=[round(x, 2) for x in per],
            intercept_us=round(float(icpt), 1),
            med_us={str(M): round(float(y), 1) for M, y in zip(Ms, ys)},
        )
    return res


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
        t0 = time.time()
        route = routing_vs_production(GLM52, interpret=False)
        route_ok = (
            route["kernel_set_diff"] == 0
            and route["kernel_w_maxdiff"] == 0.0
            and route["f32_logit_set_diff"] > 0
        )
        ok = ok and route_ok
        summary["routing"] = dict(
            route, pass_=route_ok, seconds=round(time.time() - t0, 1)
        )
        print(f"[segmk] routing vs production pass={route_ok} {route}", flush=True)
        print(
            "SEGMK_TPU_RESULT " + json.dumps({"pass": ok, "cases": summary}), flush=True
        )
    return 0 if ok else 1


def pool_overflow_checks(d: Dims):
    """Host-side: a pool beyond VMEM or the prediction row is refused, and
    prefetch_fit clamps a request to what fits."""
    out = {}
    for T in seg.TOKEN_BUCKETS:
        c = d.cfg(T)
        big = dataclasses.replace(c, e_pre=16)
        fit = seg.prefetch_fit(c, 64)
        out[f"T{T}"] = dict(
            e_pre16_nb6_refused=seg.supported(big) is not None,
            e_pre17_refused=seg.supported(dataclasses.replace(c, e_pre=17, nb=2))
            is not None,
            fit_for_64=(fit.e_pre, fit.nb) if fit else None,
            fit_ok=fit is not None and seg.supported(fit) is None,
        )
    return out


def _lowered_custom_call(path, cfg, d):
    """The tpu_custom_call payload of kernel module `path` for cfg (E_pre=0):
    proves the prefetch-off kernel is the incumbent's (M4)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("segk_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shapes = mod.operand_shapes(cfg, boundaries=2)
    args = {n: jax.ShapeDtypeStruct(*shapes[n]) for n in shapes}
    names = mod.SMEM_IN + mod.STAGED_IN + mod.HBM_IN
    fn = jax.jit(lambda *a: mod.segment_call(cfg, dict(zip(names, a))))
    import re

    txt = fn.lower(*[args[n] for n in names]).as_text()
    m = re.findall(r'backend_config = "([^"]*)"', txt) or re.findall(
        r"backend_config=([^,}]*)", txt
    )
    import hashlib

    return hashlib.md5("".join(m).encode()).hexdigest(), len(txt)


def _tpu_prefetch_main():
    pid = int(os.environ.get("JOBSET_JOB_INDEX", os.environ.get("TPU_WORKER_ID", "0")))
    nproc = int(os.environ.get("NPROC", "1"))
    if nproc > 1:
        jax.distributed.initialize(os.environ["COORD"], nproc, pid)
    devices = sorted(jax.devices(), key=lambda dv: (dv.process_index, dv.id))
    assert len(devices) == N_RANKS, f"need {N_RANKS} devices, have {len(devices)}"
    mesh = _mesh(devices)
    e_pre = int(os.environ.get("SEGMK_PF_EPRE", "12"))
    qkv_early = int(os.environ.get("SEGMK_PF_QKV_EARLY", "0"))
    sweep = tuple(
        tuple(int(v) for v in x.split(":"))
        for x in os.environ.get("SEGMK_PF_SWEEP", "").split(",")
        if x
    )
    timing_T = {
        int(t) for t in os.environ.get("SEGMK_PF_TIMING_T", "4,16,32").split(",") if t
    }
    Ms = tuple(int(x) for x in os.environ.get("SEGMK_PF_MS", "4,24").split(","))
    summary, ok = {"overflow": pool_overflow_checks(GLM52)}, True
    ok = ok and all(
        v["e_pre16_nb6_refused"] and v["e_pre17_refused"] and v["fit_ok"]
        for v in summary["overflow"].values()
    )
    old = os.environ.get("SEGMK_OLD_KERNEL")
    if old and jax.process_index() == 0:
        new = seg.kernel.__file__
        same = {}
        for T in seg.TOKEN_BUCKETS:
            a = _lowered_custom_call(old, GLM52.cfg(T), GLM52)
            b = _lowered_custom_call(new, GLM52.cfg(T), GLM52)
            same[f"T{T}"] = dict(old=a, new=b, identical=a[0] == b[0])
        summary["e_pre0_lowering_identical_to_incumbent"] = same
        print(f"[segmk-pf] lowering identity {same}", flush=True)
    for T, nv in TPU_CASES:
        t0 = time.time()
        r = _run_prefetch_case(
            mesh,
            GLM52,
            T,
            nv,
            seed=T * 100 + nv,
            e_pre=(e_pre, qkv_early),
            sweep=sweep if (T in timing_T and nv == T) else (),
            timing=Ms if (T in timing_T and nv == T) else (),
        )
        mine = np.asarray(
            [r["exact"][k] for k in sorted(r["exact"])]
            + [r["caught"][k] for k in sorted(r["caught"])]
        )
        allv = multihost_utils_allgather(mine)
        names = [f"exact.{k}" for k in sorted(r["exact"])] + [
            f"caught.{k}" for k in sorted(r["caught"])
        ]
        agg = dict(zip(names, np.all(allv, axis=0).tolist()))
        r["all_hosts"] = agg
        case_ok = all(v for k, v in agg.items() if not k.startswith("caught.no_dma"))
        r["pass"] = case_ok
        ok = ok and case_ok
        r["seconds"] = round(time.time() - t0, 1)
        summary[f"T{T}_nv{nv}"] = r
        if jax.process_index() == 0:
            print(
                f"[segmk-pf] T={T} nv={nv} pass={case_ok} {json.dumps(r)}", flush=True
            )
            _pf_dump(summary)
    if jax.process_index() == 0:
        pv = predictor_vs_reference(GLM52, interpret=False)
        summary["predictor_vs_xla"] = pv
        ok = ok and all(v["ok"] for v in pv.values())
        print(f"[segmk-pf] predictor vs XLA {pv}", flush=True)
        print("SEGMK_PF_RESULT " + json.dumps({"pass": ok, "cases": summary}), flush=True)
        _pf_dump({"pass": ok, "cases": summary})
    return 0 if ok else 1


def _pf_dump(summary):
    """Results also to $CDK_OUTPUT_DIR (the log transport can drop stdout)."""
    d = os.environ.get("CDK_OUTPUT_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "segmk_pf_result.json"), "w") as f:
            json.dump(summary, f)


def multihost_utils_allgather(x):
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(np.asarray(x), tiled=False))


def _cpu_prefetch_worker():
    """Interpret mode, 16 host devices: the prefetch kernel's DMA / semaphore
    protocol (race detector) and bit identity at reduced dims."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.get_tpu_info = _tpu7x
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.get_tpu_info = _tpu7x
    mesh = _mesh(sorted(jax.devices("cpu"), key=lambda dv: dv.id)[:N_RANKS])
    tiny = os.environ.get("SEGMK_TEST_TINY") == "1"
    dims = TINY if tiny else SMALL
    out = {}
    for T, nv in ((4, 4),) if tiny else ((4, 4), (16, 12)):
        t0 = time.time()
        pltpu.reset_tpu_interpret_mode_state()
        r = _run_prefetch_case(mesh, dims, T, nv, seed=T * 100 + nv, e_pre=6)
        r["races"] = bool(interpret_pallas_call.races.races_found)
        r["seconds"] = round(time.time() - t0, 1)
        out[f"T{T}_nv{nv}"] = r
    print("PFRESULT " + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    if "--routing-worker" in sys.argv:
        print(
            "ROUTING "
            + json.dumps(routing_vs_production(GLM52, interpret=True, calls=8)),
            flush=True,
        )
    elif "--cpu-prefetch-worker" in sys.argv:
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            _cpu_prefetch_worker()
    elif "--tpu-prefetch" in sys.argv:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
        )
        sys.exit(_tpu_prefetch_main())
    elif "--cpu-worker" in sys.argv:
        # Around everything: pallas_call reads the mode when it is built.
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            _cpu_worker()
    elif "--tpu" in sys.argv:
        # The served model's numerics: torch_tpu compiles every program with
        # allow_excess_precision off (bf16 casts are kept), so the incumbent
        # composition here must be compiled the same way.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
        )
        sys.exit(_tpu_main())
    else:
        sys.exit(pytest.main([__file__, "-v"]))
