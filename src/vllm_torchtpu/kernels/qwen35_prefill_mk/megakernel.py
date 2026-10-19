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
"""Qwen3.5-397B-A17B prefill megakernel: one pipeline stage's whole model step in ONE pallas_call.

For each of the stage's layers (static: the layer types are known per stage), inside the kernel:

    residual += hidden; x = norm(residual)              norm_add_quant (+ per-token fp8 of x)
    hidden = GDN(x) | GatedAttention(x)                 projections, mixer, gate/norm, out-proj
    residual += hidden; x = norm(residual)              norm_add_quant
    hidden = MoE(x)                                     router + shared gate/up (one fp8 matmul),
                                                        top-k routing, grouped experts, combine

Every operand is in HBM; each phase is an emit_pipeline over its tiles and intermediates live in
HBM workspaces, which are extra outputs of the call and are reused across layers. The running
residual and hidden live in the call's first two outputs. KV caches and GDN states are updated
in place (input/output aliases).

M1 scope: one sequence of T new tokens at positions ctx .. ctx+T-1 (ctx is a runtime scalar,
a multiple of 32), contiguous per-layer KV caches [NKV, S, HD] fp8, dense per-layer GDN states.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen35_prefill_mk import phases as P
from vllm_torchtpu.kernels.qwen35_prefill_mk.reference import BF16, F32, FP8, StageConfig

I32 = jnp.int32

GDN_W = ("in_norm", "qkvz_w", "qkvz_s", "ba_w", "conv_w", "a_log", "dt_bias", "gnorm_w",
         "out_w", "out_s")
ATT_W = ("in_norm", "qkv_w", "qkv_s", "q_norm", "k_norm", "o_w", "o_s")
MOE_W = ("post_norm", "gate_gu_w", "gate_gu_s", "sd_w", "sd_s", "sgate_w", "w13", "w13_s",
         "w2", "w2_s")


@dataclasses.dataclass(frozen=True)
class MKTiles:
    tm_norm: int = 256      # norms, swiglu
    tm_prep: int = 64       # GDN conv/gates prep (wide f32 temporaries)
    tm_gate: int = 128      # GDN gated norm, attention q/k norm + rope, attention gate
    tm: int = 1024           # matmul row tile (j-d714d214-80e9-4d1f-9ec4: 153 vs 157 ms at 512)
    tn: int = 1024
    chunk: int = 64          # GDN chunk
    bq: int = 256            # attention q tile (tokens)
    bkv: int = 2048          # attention kv tile (j-692719db-95ef-4ea4-8b2f: best of 512..4096)
    att_sub: int = 8         # attention row sub-chunks (MXU / softmax overlap)
    att_mxu_sum: int = 0     # 1: softmax row sum from the PV matmul (ones columns in V)
    att_fp8: int = 0         # 1: native fp8 QK / PV (opt-in, changes attention numerics)
    rt: int = 256            # MoE grouped rows per tile (j-964229e0-90a9-4258-a763: 150 vs 153 ms)
    moe_wslots: int = 2      # expert weight buffers (lookahead = wslots - 1)
    moe_interleave: int = 0  # >0: next tile's row DMAs started in this many chunks in the matmuls
    tm_route: int = 256
    tm_combine: int = 64
    inflight: int = 64
    vmem_mib: int = 63


def kernel_weight_layout(c, w, lt):
    """A layer's reference weights -> the kernel's operand layouts (2-D vectors, [ks, QKV]
    conv weight, [E, 1, N] expert scales)."""
    k = {}
    k["in_norm"] = w["in_norm"][None]
    if lt == "gdn":
        k.update(qkvz_w=w["qkvz_w"], qkvz_s=w["qkvz_s"][None], ba_w=w["ba_w"], conv_w=w["conv_w"].T,
                 a_log=w["a_log"][None], dt_bias=w["dt_bias"][None], gnorm_w=w["gnorm_w"][None],
                 out_w=w["out_w"], out_s=w["out_s"][None])
    else:
        k.update(qkv_w=w["qkv_w"], qkv_s=w["qkv_s"][None], q_norm=w["q_norm"][None],
                 k_norm=w["k_norm"][None], o_w=w["o_w"], o_s=w["o_s"][None])
    k.update(post_norm=w["post_norm"][None], gate_gu_w=w["gate_gu_w"],
             gate_gu_s=w["gate_gu_s"][None], sd_w=w["sd_w"], sd_s=w["sd_s"][None],
             sgate_w=w["sgate_w"], w13=w["w13"], w13_s=w["w13_s"][:, None], w2=w["w2"],
             w2_s=w["w2_s"][:, None])
    return k


def rope_table(c, s_max):
    half = c.ROT // 2
    inv = 1.0 / (c.theta ** (np.arange(half) * 2.0 / c.ROT))
    ang = np.arange(s_max)[:, None] * inv[None]
    return jnp.asarray(np.concatenate([np.cos(ang), np.sin(ang)], -1), F32)


def build(c: StageConfig, t: int, s_max: int, tiles: MKTiles = MKTiles(), *, interpret=False,
          debug=False):
    """Returns step(ctx, hidden, residual, weights, states, rope) -> (hidden, residual, states).

    weights / states: per layer, the kernel layouts (kernel_weight_layout; gdn states
    {"conv": [CONV-1, QKV] bf16, "rec": [VH, KD, VD] f32}, attn states {"k", "v": [NKV, S, HD]
    fp8}). T must be a multiple of every row tile; s_max is the KV cache capacity."""
    tl = tiles
    types = c.layer_types
    nt_att = 2 * c.NH + 2 * c.NKV
    n_pad = t * c.TOPK + c.E * (tl.rt - 1)
    tn_e = min(tl.tn, c.E)
    has_gdn, has_att = "gdn" in types, "attn" in types

    # Workspaces: name -> (shape, dtype).
    ws = {"xb": ((t, c.H), BF16), "xq": ((t, c.H), FP8), "xs": ((t, 1), F32),
          "lg": ((t, c.E), BF16), "sgu": ((t, 2 * c.IS), BF16), "hq": ((t, c.IS), FP8),
          "hs": ((t, 1), F32), "sd": ((t, c.H), BF16), "sg": ((t, 1), F32),
          "topi": ((t, c.TOPK), I32), "topw": ((t, c.TOPK), F32), "rank": ((t, c.TOPK), I32),
          "cnt": ((1, c.E), I32), "ys": ((n_pad, c.H), F32), "xf": ((t, c.H), F32),
          "pos": ((t, c.TOPK), I32), "cpos": ((t, c.TOPK), I32)}
    if has_gdn:
        kdim = c.KH * c.KD
        ws.update({"qkvz": ((t, c.QKV + c.Z), BF16), "ba": ((t, 2 * c.VH), BF16),
                   "gq": ((t, kdim), F32), "gk": ((t, kdim), F32), "gv": ((t, c.VH * c.VD), F32),
                   "gb": ((t, 2 * c.VH), F32), "core": ((t, c.Z), BF16),
                   "oq": ((t, c.Z), FP8), "os": ((t, 1), F32)})
    if has_att:
        ws.update({"att": ((nt_att, t, c.HD), BF16), "qn": ((c.NH, t, c.HD), BF16),
                   "ao": ((c.NH, t, c.HD), F32), "aq": ((t, c.NH * c.HD), FP8),
                   "as": ((t, 1), F32)})
    ws_names = list(ws)

    w_names = [(GDN_W if lt == "gdn" else ATT_W) + MOE_W for lt in types]
    s_names = [("conv", "rec") if lt == "gdn" else ("k", "v") for lt in types]
    n_w = sum(len(n) for n in w_names)
    n_s = sum(len(n) for n in s_names)

    def kernel(ctx_ref, h_in, r_in, rope, *refs):
        w_refs, refs = refs[:n_w], refs[n_w:]
        s_in, refs = refs[:n_s], refs[n_s:]
        h_out, r_out = refs[0], refs[1]
        s_out, refs = refs[2:2 + n_s], refs[2 + n_s:]
        W = dict(zip(ws_names, refs))
        ctx = ctx_ref[0]
        wi = si = 0
        for li, lt in enumerate(types):
            lw = dict(zip(w_names[li], w_refs[wi:wi + len(w_names[li])]))
            wi += len(w_names[li])
            ls = dict(zip(s_names[li], s_out[si:si + 2]))
            ls_in = dict(zip(s_names[li], s_in[si:si + 2]))
            si += 2
            hsrc, rsrc = (h_in, r_in) if li == 0 else (h_out, r_out)
            with jax.named_scope(f"L{li}_in_norm"):
                P.norm_add_quant(hsrc, rsrc, lw["in_norm"], r_out, W["xb"], W["xq"], W["xs"],
                                 eps=c.eps, tm=tl.tm_norm)
            if lt == "gdn":
                with jax.named_scope(f"L{li}_gdn_in_proj"):
                    P.qmatmul(W["xq"], W["xs"], lw["qkvz_w"], lw["qkvz_s"], W["qkvz"], tm=tl.tm,
                              tn=tl.tn)
                with jax.named_scope(f"L{li}_gdn_ba"):
                    P.bmatmul(W["xb"], lw["ba_w"], W["ba"], tm=tl.tm)
                with jax.named_scope(f"L{li}_gdn_prep"):
                    P.gdn_prep(W["qkvz"], W["ba"], ls_in["conv"], lw["conv_w"], lw["a_log"],
                               lw["dt_bias"], W["gq"], W["gk"], W["gv"], W["gb"], ls["conv"],
                               kq=c.KH, vh=c.VH, kd=c.KD, vd=c.VD, ks=c.CONV, eps=c.eps,
                               tm=tl.tm_prep)
                with jax.named_scope(f"L{li}_gdn_chunks"):
                    P.gdn_chunks(W["gq"], W["gk"], W["gv"], W["gb"], ls_in["rec"], W["core"],
                                 ls["rec"], kq=c.KH, vh=c.VH, kd=c.KD, vd=c.VD, chunk=tl.chunk)
                with jax.named_scope(f"L{li}_gdn_gated_norm_quant"):
                    P.gdn_gated_norm_quant(W["core"], W["qkvz"], lw["gnorm_w"], W["oq"], W["os"],
                                           vh=c.VH, vd=c.VD, qkv=c.QKV, eps=c.eps, tm=tl.tm_gate)
                with jax.named_scope(f"L{li}_gdn_out_proj"):
                    P.qmatmul(W["oq"], W["os"], lw["out_w"], lw["out_s"], h_out, tm=tl.tm, tn=tl.tn)
            else:
                with jax.named_scope(f"L{li}_att_qkv_proj"):
                    P.qmatmul_heads(W["xq"], W["xs"], lw["qkv_w"], lw["qkv_s"], W["att"], tm=tl.tm,
                                    nh=c.NH, nkv=c.NKV, hd=c.HD)
                with jax.named_scope(f"L{li}_att_qk_norm_rope_kv"):
                    P.att_qk_norm_rope_kv(W["att"], lw["q_norm"], lw["k_norm"], rope, ctx, W["qn"],
                                          ls["k"], ls["v"], nh=c.NH, nkv=c.NKV, hd=c.HD, rot=c.ROT,
                                          eps=c.eps, tm=tl.tm_gate)
                with jax.named_scope(f"L{li}_att_flash"):
                    P.att_flash(W["qn"], ls["k"], ls["v"], ctx, W["ao"], nh=c.NH, nkv=c.NKV,
                                hd=c.HD, bq=tl.bq, bkv=tl.bkv, s_max=s_max, sub=tl.att_sub,
                                fp8=bool(tl.att_fp8), mxu_sum=bool(tl.att_mxu_sum))
                with jax.named_scope(f"L{li}_att_gate_quant"):
                    P.att_gate_quant(W["ao"], W["att"], W["aq"], W["as"], nh=c.NH, hd=c.HD,
                                     tm=tl.tm_gate)
                with jax.named_scope(f"L{li}_att_o_proj"):
                    P.qmatmul(W["aq"], W["as"], lw["o_w"], lw["o_s"], h_out, tm=tl.tm, tn=tl.tn)
            # MoE
            with jax.named_scope(f"L{li}_post_norm"):
                P.norm_add_quant(h_out, r_out, lw["post_norm"], r_out, W["xb"], W["xq"], W["xs"],
                                 eps=c.eps, tm=tl.tm_norm, xf_hbm=W["xf"])
            with jax.named_scope(f"L{li}_router"):
                P.qmatmul(W["xq"], W["xs"], lw["gate_gu_w"], lw["gate_gu_s"], W["lg"], tm=tl.tm,
                          tn=tn_e, n0=0, nn=c.E // tn_e)
            with jax.named_scope(f"L{li}_shared_gate_up"):
                P.qmatmul(W["xq"], W["xs"], lw["gate_gu_w"], lw["gate_gu_s"], W["sgu"], tm=tl.tm,
                          tn=tn_e, n0=c.E // tn_e, nn=2 * c.IS // tn_e)
            with jax.named_scope(f"L{li}_swiglu_quant"):
                P.swiglu_quant(W["sgu"], W["hq"], W["hs"], inter=c.IS, tm=tl.tm_norm)
            with jax.named_scope(f"L{li}_shared_down"):
                P.qmatmul(W["hq"], W["hs"], lw["sd_w"], lw["sd_s"], W["sd"], tm=tl.tm, tn=tl.tn)
            with jax.named_scope(f"L{li}_shared_gate"):
                P.bmatmul(W["xb"], lw["sgate_w"], W["sg"], tm=tl.tm)
            with jax.named_scope(f"L{li}_moe_route"):
                P.moe_route(W["lg"], W["topi"], W["topw"], W["rank"], W["cnt"], e=c.E, topk=c.TOPK,
                            tm=tl.tm_route, simulate_uniform=SIM_UNIFORM,
            salt=li)
            with jax.named_scope(f"L{li}_moe_experts"):
                P.moe_experts(W["xf"], W["topi"], W["rank"], W["cnt"], lw["w13"], lw["w13_s"],
                              lw["w2"], lw["w2_s"], W["ys"], W["pos"], W["cpos"], e=c.E,
                              topk=c.TOPK, inter=c.I, rt=tl.rt, inflight=tl.inflight,
                              tm=tl.tm_route, interleave=tl.moe_interleave, wslots=tl.moe_wslots)
            with jax.named_scope(f"L{li}_moe_combine"):
                P.moe_combine(W["ys"], W["pos"], W["topw"], W["sd"], W["sg"], h_out, topk=c.TOPK,
                              tm=tl.tm_combine, inflight=tl.inflight)
            # Each layer's states were copied into the outputs' aliases before any phase reads
            # them (in place: s_out aliases s_in), so nothing else to do here.

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    n_in = 3 + n_w + n_s                         # h, r, rope, weights, states (after ctx)

    def step(ctx, hidden, residual, weights, states, rope):
        flat_w = [weights[li][n] for li in range(len(types)) for n in w_names[li]]
        flat_s = [states[li][n] for li in range(len(types)) for n in s_names[li]]
        s_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in flat_s]
        out_shape = ([jax.ShapeDtypeStruct((t, c.H), BF16)] * 2 + s_shapes
                     + [jax.ShapeDtypeStruct(*ws[n]) for n in ws_names])
        # input index (counting the scalar-prefetch operand) -> output index
        aliases = {1 + 3 + n_w + i: 2 + i for i in range(n_s)}
        call = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, grid=(), in_specs=[hbm] * n_in,
                out_specs=[hbm] * len(out_shape)),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=tl.vmem_mib << 20),
            interpret=pltpu.InterpretParams() if interpret else False,
            name="qwen3_5_prefill_stage_megakernel",
        )
        outs = call(jnp.asarray(ctx, I32).reshape(1), hidden, residual, rope, *flat_w, *flat_s)
        h, r = outs[0], outs[1]
        new_s, k = [], 2
        for li in range(len(types)):
            new_s.append(dict(zip(s_names[li], outs[k:k + 2])))
            k += 2
        if debug:
            return h, r, new_s, dict(zip(ws_names, outs[k:]))
        return h, r, new_s

    return step


# ------------------------------------------------------------------------------------------------
# The serving form: multi-sequence steps over the unified KV / GDN-state pool
# ------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageLayout:
    """Static facts of a pipeline stage: per layer its type, the index of its region's pool
    tensor (one per full-attention layer and its three GDN layers), and for GDN layers its
    KV-cache group (0..2, which picks the state-index table); the attention layers' K / V scales.
    """
    layer_types: tuple[str, ...]
    layer_region: tuple[int, ...]
    layer_group: tuple[int, ...]
    k_scales: tuple[float, ...] = ()
    v_scales: tuple[float, ...] = ()

    @property
    def n_regions(self):
        return max(self.layer_region) + 1


# vLLM's routing simulation, honored so the megakernel and the stock MoE run the same load.
SIM_UNIFORM = os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "") == "uniform_random"

META = ("tile_seg", "seg_start", "seg_len", "seg_ctx", "bt", "gslot0", "gslot1", "gslot2")


def build_stage(c: StageConfig, layout: StageLayout, t: int, *, max_seqs: int, pps: int,
                pool_shape, plan, stride, tiles: MKTiles = MKTiles(), interpret=False,
                debug=False):
    """Returns step(meta, hidden, residual, weights, pools, rope) -> (hidden, residual, pools):
    the stage's whole prefill step in one pallas_call over a multi-sequence step.

    meta (int32, SMEM): tile_seg [t / 64] (sequence of each 64-row block, -1 = padding),
    seg_start / seg_len / seg_ctx [max_seqs] (padded-layout start, new tokens, cached tokens;
    starts are multiples of 256, ctx multiples of 128), bt [max_seqs * pps] (the attention
    group's page table: kernel page of position p of sequence s at bt[s * pps + p // 128]),
    gslot0..2 [max_seqs] (each GDN group's state block per sequence). hidden / residual
    [t, H] bf16 in the padded layout; weights per layer in kernel_weight_layout; pools: the
    region tensors (pages, 2 kvh, 64, 4, 128) fp8, updated in place (KV of the new tokens, GDN
    states); plan / stride from pool_plan.build_gdn_plan."""
    tl = tiles
    types = layout.layer_types
    assert c.layer_types == types
    nt_att = 2 * c.NH + 2 * c.NKV
    n_pad = t * c.TOPK + c.E * (tl.rt - 1)
    tn_e = min(tl.tn, c.E)
    has_gdn, has_att = "gdn" in types, "attn" in types
    n_reg = layout.n_regions

    ws = {"xb": ((t, c.H), BF16), "xq": ((t, c.H), FP8), "xs": ((t, 1), F32),
          "lg": ((t, c.E), BF16), "sgu": ((t, 2 * c.IS), BF16), "hq": ((t, c.IS), FP8),
          "hs": ((t, 1), F32), "sd": ((t, c.H), BF16), "sg": ((t, 1), F32),
          "topi": ((t, c.TOPK), I32), "topw": ((t, c.TOPK), F32), "rank": ((t, c.TOPK), I32),
          "cnt": ((1, c.E), I32), "ys": ((n_pad, c.H), F32), "xf": ((t, c.H), F32),
          "pos": ((t, c.TOPK), I32), "cpos": ((t, c.TOPK), I32)}
    if has_gdn:
        kdim = c.KH * c.KD
        ws.update({"qkvz": ((t, c.QKV + c.Z), BF16), "ba": ((t, 2 * c.VH), BF16),
                   "gq": ((t, kdim), F32), "gk": ((t, kdim), F32), "gv": ((t, c.VH * c.VD), F32),
                   "gb": ((t, 2 * c.VH), F32), "core": ((t, c.Z), BF16),
                   "oq": ((t, c.Z), FP8), "os": ((t, 1), F32)})
    if has_att:
        ws.update({"att": ((nt_att, t, c.HD), BF16), "qn": ((c.NH, t, c.HD), BF16),
                   "ao": ((c.NH, t, c.HD), F32), "aq": ((t, c.NH * c.HD), FP8),
                   "as": ((t, 1), F32)})
    ws_names = list(ws)
    w_names = [(GDN_W if lt == "gdn" else ATT_W) + MOE_W for lt in types]
    n_w = sum(len(n) for n in w_names)
    att_idx = [li for li, lt in enumerate(types) if lt == "attn"]
    k_scales = layout.k_scales or (1.0,) * len(att_idx)
    v_scales = layout.v_scales or (1.0,) * len(att_idx)

    def kernel(tile_seg, seg_start, seg_len, seg_ctx, bt, gs0, gs1, gs2, h_in, r_in, rope,
               *refs):
        w_refs, refs = refs[:n_w], refs[n_w:]
        pools_in, refs = refs[:n_reg], refs[n_reg:]
        h_out, r_out = refs[0], refs[1]
        pools, refs = refs[2:2 + n_reg], refs[2 + n_reg:]
        W = dict(zip(ws_names, refs))
        del pools_in                            # aliased by `pools`
        gslots = (gs0, gs1, gs2)
        wi = 0
        for li, lt in enumerate(types):
            lw = dict(zip(w_names[li], w_refs[wi:wi + len(w_names[li])]))
            wi += len(w_names[li])
            pool = pools[layout.layer_region[li]]
            segs = P.Segments(tile_seg, seg_start, seg_len, seg_ctx,
                              gslots[layout.layer_group[li]] if lt == "gdn" else gs0)
            hsrc, rsrc = (h_in, r_in) if li == 0 else (h_out, r_out)
            with jax.named_scope(f"L{li}_in_norm"):
                P.norm_add_quant(hsrc, rsrc, lw["in_norm"], r_out, W["xb"], W["xq"], W["xs"],
                                 eps=c.eps, tm=tl.tm_norm)
            if lt == "gdn":
                with jax.named_scope(f"L{li}_gdn_in_proj"):
                    P.qmatmul(W["xq"], W["xs"], lw["qkvz_w"], lw["qkvz_s"], W["qkvz"], tm=tl.tm,
                              tn=tl.tn)
                with jax.named_scope(f"L{li}_gdn_ba"):
                    P.bmatmul(W["xb"], lw["ba_w"], W["ba"], tm=tl.tm)
                with jax.named_scope(f"L{li}_gdn_prep"):
                    P.gdn_prep(W["qkvz"], W["ba"], pool, lw["conv_w"], lw["a_log"], lw["dt_bias"],
                               W["gq"], W["gk"], W["gv"], W["gb"], pool, segs, kq=c.KH, vh=c.VH,
                               kd=c.KD, vd=c.VD, ks=c.CONV, eps=c.eps, tm=tl.tm_prep,
                               plan=plan, stride=stride)
                with jax.named_scope(f"L{li}_gdn_chunks"):
                    P.gdn_chunks(W["gq"], W["gk"], W["gv"], W["gb"], pool, W["core"], pool, segs,
                                 kq=c.KH, vh=c.VH, kd=c.KD, vd=c.VD, chunk=tl.chunk, plan=plan,
                                 stride=stride)
                with jax.named_scope(f"L{li}_gdn_gated_norm_quant"):
                    P.gdn_gated_norm_quant(W["core"], W["qkvz"], lw["gnorm_w"], W["oq"], W["os"],
                                           vh=c.VH, vd=c.VD, qkv=c.QKV, eps=c.eps, tm=tl.tm_gate)
                with jax.named_scope(f"L{li}_gdn_out_proj"):
                    P.qmatmul(W["oq"], W["os"], lw["out_w"], lw["out_s"], h_out, tm=tl.tm,
                              tn=tl.tn)
            else:
                ai = att_idx.index(li)
                with jax.named_scope(f"L{li}_att_qkv_proj"):
                    P.qmatmul_heads(W["xq"], W["xs"], lw["qkv_w"], lw["qkv_s"], W["att"],
                                    tm=tl.tm, nh=c.NH, nkv=c.NKV, hd=c.HD)
                with jax.named_scope(f"L{li}_att_qk_norm_rope_kv"):
                    P.att_qk_norm_rope_kv_paged(
                        W["att"], lw["q_norm"], lw["k_norm"], rope, pool, bt, pps, W["qn"], segs,
                        nh=c.NH, nkv=c.NKV, hd=c.HD, rot=c.ROT, eps=c.eps,
                        k_scale=k_scales[ai], v_scale=v_scales[ai], interpret=interpret)
                with jax.named_scope(f"L{li}_att_flash"):
                    P.att_flash_paged(W["qn"], pool, bt, pps, W["ao"], segs, nh=c.NH, nkv=c.NKV,
                                      hd=c.HD, bq=tl.bq, bkv=tl.bkv, sub=tl.att_sub,
                                      k_scale=k_scales[ai], v_scale=v_scales[ai],
                                      interpret=interpret)
                with jax.named_scope(f"L{li}_att_gate_quant"):
                    P.att_gate_quant(W["ao"], W["att"], W["aq"], W["as"], nh=c.NH, hd=c.HD,
                                     tm=tl.tm_gate)
                with jax.named_scope(f"L{li}_att_o_proj"):
                    P.qmatmul(W["aq"], W["as"], lw["o_w"], lw["o_s"], h_out, tm=tl.tm, tn=tl.tn)
            with jax.named_scope(f"L{li}_post_norm"):
                P.norm_add_quant(h_out, r_out, lw["post_norm"], r_out, W["xb"], W["xq"], W["xs"],
                                 eps=c.eps, tm=tl.tm_norm, xf_hbm=W["xf"])
            with jax.named_scope(f"L{li}_router"):
                P.qmatmul(W["xq"], W["xs"], lw["gate_gu_w"], lw["gate_gu_s"], W["lg"], tm=tl.tm,
                          tn=tn_e, n0=0, nn=c.E // tn_e)
            with jax.named_scope(f"L{li}_shared_gate_up"):
                P.qmatmul(W["xq"], W["xs"], lw["gate_gu_w"], lw["gate_gu_s"], W["sgu"], tm=tl.tm,
                          tn=tn_e, n0=c.E // tn_e, nn=2 * c.IS // tn_e)
            with jax.named_scope(f"L{li}_swiglu_quant"):
                P.swiglu_quant(W["sgu"], W["hq"], W["hs"], inter=c.IS, tm=tl.tm_norm)
            with jax.named_scope(f"L{li}_shared_down"):
                P.qmatmul(W["hq"], W["hs"], lw["sd_w"], lw["sd_s"], W["sd"], tm=tl.tm, tn=tl.tn)
            with jax.named_scope(f"L{li}_shared_gate"):
                P.bmatmul(W["xb"], lw["sgate_w"], W["sg"], tm=tl.tm)
            with jax.named_scope(f"L{li}_moe_route"):
                P.moe_route(W["lg"], W["topi"], W["topw"], W["rank"], W["cnt"], e=c.E,
                            topk=c.TOPK, tm=tl.tm_route, simulate_uniform=SIM_UNIFORM,
            salt=li)
            with jax.named_scope(f"L{li}_moe_experts"):
                P.moe_experts(W["xf"], W["topi"], W["rank"], W["cnt"], lw["w13"], lw["w13_s"],
                              lw["w2"], lw["w2_s"], W["ys"], W["pos"], W["cpos"], e=c.E,
                              topk=c.TOPK, inter=c.I, rt=tl.rt, inflight=tl.inflight,
                              tm=tl.tm_route, interleave=tl.moe_interleave, wslots=tl.moe_wslots)
            with jax.named_scope(f"L{li}_moe_combine"):
                P.moe_combine(W["ys"], W["pos"], W["topw"], W["sd"], W["sg"], h_out, topk=c.TOPK,
                              tm=tl.tm_combine, inflight=tl.inflight)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    n_in = 3 + n_w + n_reg                       # h, r, rope, weights, pools (after the meta)

    def step(meta, hidden, residual, weights, pools, rope):
        flat_w = [weights[li][n] for li in range(len(types)) for n in w_names[li]]
        out_shape = ([jax.ShapeDtypeStruct((t, c.H), BF16)] * 2
                     + [jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools]
                     + [jax.ShapeDtypeStruct(*ws[n]) for n in ws_names])
        aliases = {len(META) + 3 + n_w + i: 2 + i for i in range(n_reg)}
        call = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(META), grid=(), in_specs=[hbm] * n_in,
                out_specs=[hbm] * len(out_shape)),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=tl.vmem_mib << 20),
            interpret=pltpu.InterpretParams() if interpret else False,
            name="qwen3_5_prefill_stage_megakernel",
        )
        outs = call(*[meta[k] for k in META], hidden, residual, rope, *flat_w, *pools)
        h, r = outs[0], outs[1]
        new_pools = list(outs[2:2 + n_reg])
        if debug:
            return h, r, new_pools, dict(zip(ws_names, outs[2 + n_reg:]))
        return h, r, new_pools

    return step
