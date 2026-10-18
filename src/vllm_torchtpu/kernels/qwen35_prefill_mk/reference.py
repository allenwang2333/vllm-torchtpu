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
"""Pure-JAX reference of one Qwen3.5-397B-A17B pipeline stage's prefill step at TP1.

The semantics are vLLM's (models/qwen3_next.py, qwen3_5.py, layers/mamba/gdn/
qwen_gdn_linear_attn.py), the arithmetic is vllm-torchtpu's serving path: fp8 linears are w8a8
with a per-token dynamic activation scale and a per-output-channel weight scale
(kernels/quantized_matmul/util.py:xla_quantized_matmul), the router logits share one fp8 matmul
with the shared expert's gate/up (`_fused_gate_gu_fp8_weight`), and the KV cache is fp8.

One step: one sequence of T new tokens at positions ctx .. ctx+T-1. Per layer:

    hidden, residual = rmsnorm_zc(hidden + residual)          # input_layernorm, fused add
    hidden = GDN(hidden) or GatedAttention(hidden)
    hidden, residual = rmsnorm_zc(hidden + residual)          # post_attention_layernorm
    hidden = MoE(hidden)

The stage takes and returns (hidden, residual); the first stage passes residual = None
(here: zeros, which the fused add makes equivalent).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen35_prefill_mk import gdn_chunked

F32, BF16 = jnp.float32, jnp.bfloat16
FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)


@dataclasses.dataclass(frozen=True)
class StageConfig:
    layer_types: tuple[str, ...] = ("gdn", "gdn", "gdn", "attn", "gdn", "gdn", "gdn", "attn")
    H: int = 4096
    NH: int = 32
    NKV: int = 2
    HD: int = 256
    ROT: int = 64
    theta: float = 1e7
    KH: int = 16
    VH: int = 64
    KD: int = 128
    VD: int = 128
    CONV: int = 4
    E: int = 512
    TOPK: int = 10
    I: int = 1024
    IS: int = 1024
    eps: float = 1e-6

    @property
    def QKV(self):  # GDN conv channels: q, k, v
        return 2 * self.KH * self.KD + self.VH * self.VD

    @property
    def Z(self):
        return self.VH * self.VD

    @property
    def ATT_QKV(self):  # q with its gate, k, v
        return 2 * self.NH * self.HD + 2 * self.NKV * self.HD


def rmsnorm_zc(x, w, eps):
    """GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), in f32."""
    x = x.astype(F32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * (1.0 + w.astype(F32))


def qlinear(x, w_q, w_s):
    """w8a8: per-token dynamic fp8 activation scale x per-output-channel weight scale; the
    result is rounded to bf16, as xla_quantized_matmul returns the activation dtype."""
    x = x.astype(BF16).astype(F32)
    amax = jnp.max(jnp.abs(x), -1, keepdims=True)
    xs = jnp.where(amax > 0, amax * (1.0 / FP8_MAX), 1.0)
    xq = jnp.clip(x / xs, -FP8_MAX, FP8_MAX).astype(FP8)
    out = jax.lax.dot_general(xq, w_q, (((1,), (0,)), ((), ())), preferred_element_type=F32)
    return (out * xs * w_s.astype(F32)[None, :]).astype(BF16).astype(F32)


def rope_neox(x, pos, rot, theta):
    """NeoX RoPE on the first `rot` dims of each head: x [T, heads, D], pos [T]."""
    half = rot // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=F32) * 2.0 / rot))
    ang = pos.astype(F32)[:, None] * inv[None, :]
    cos, sin = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
    x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)


def gdn_layer(c, w, x, conv_state, rec_state):
    """x [T, H] (normed, bf16). Returns out [T, H] f32, new conv_state, new rec_state."""
    t = x.shape[0]
    qkvz = qlinear(x, w["qkvz_w"], w["qkvz_s"]).astype(BF16)
    qkv, z = qkvz[:, :c.QKV], qkvz[:, c.QKV:]
    ba = jax.lax.dot_general(x, w["ba_w"], (((1,), (0,)), ((), ())),
                             preferred_element_type=F32).astype(BF16)
    b, a = ba[:, :c.VH], ba[:, c.VH:]
    (new_conv, new_rec), core = gdn_chunked.gdn_prefill(
        qkv, b, a, conv_state, rec_state, w["conv_w"][:, None, :], None, w["a_log"],
        w["dt_bias"], n_kq=c.KH, n_v=c.VH, d_k=c.KD, d_v=c.VD, kernel_size=c.CONV,
        use_pallas=False)
    core = core.reshape(t, c.VH, c.VD).astype(F32)
    zz = z.reshape(t, c.VH, c.VD).astype(F32)
    normed = core * jax.lax.rsqrt(jnp.mean(core * core, -1, keepdims=True) + c.eps)
    gated = (normed * w["gnorm_w"].astype(F32) * jax.nn.silu(zz)).astype(BF16)
    out = qlinear(gated.reshape(t, c.Z), w["out_w"], w["out_s"])
    return out, new_conv, new_rec


def attn_layer(c, w, x, k_cache, v_cache, ctx, flash_bkv=None):
    """x [T, H]; k_cache, v_cache [ctx + T, NKV, HD] fp8 with the context in [:ctx].
    Returns out [T, H] f32 and the caches with the new tokens' K/V written at ctx .. ctx+T-1.
    flash_bkv: emulate a tiled online softmax over KV tiles of that size (bf16 scores and P per
    tile, f32 state, as the serving RPA), for close comparison with a flash kernel."""
    t = x.shape[0]
    qkv = qlinear(x, w["qkv_w"], w["qkv_s"]).astype(BF16)
    qg = qkv[:, :2 * c.NH * c.HD].reshape(t, c.NH, 2 * c.HD)
    q, gate = qg[..., :c.HD], qg[..., c.HD:].reshape(t, c.NH * c.HD)
    k = qkv[:, 2 * c.NH * c.HD:2 * c.NH * c.HD + c.NKV * c.HD].reshape(t, c.NKV, c.HD)
    v = qkv[:, 2 * c.NH * c.HD + c.NKV * c.HD:].reshape(t, c.NKV, c.HD)
    q = rmsnorm_zc(q, w["q_norm"], c.eps).astype(BF16)
    k = rmsnorm_zc(k, w["k_norm"], c.eps).astype(BF16)
    pos = ctx + jnp.arange(t)
    q = rope_neox(q.astype(F32), pos, c.ROT, c.theta).astype(BF16)
    k = rope_neox(k.astype(F32), pos, c.ROT, c.theta).astype(BF16)
    k_cache = jax.lax.dynamic_update_slice(k_cache, k.astype(FP8), (ctx, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v.astype(FP8), (ctx, 0, 0))
    kk, vv = k_cache.astype(F32), v_cache.astype(F32)                 # [S, NKV, HD]
    qq = q.astype(F32).reshape(t, c.NKV, c.NH // c.NKV, c.HD)
    s = jnp.einsum("tgqd,sgd->gqts", qq, kk) * (c.HD ** -0.5)
    visible = jnp.arange(kk.shape[0])[None, :] <= pos[:, None]
    s = jnp.where(visible[None, None], s, -jnp.inf)
    if flash_bkv is None:
        # The serving kernels feed bf16 probabilities to PV and normalize by the f32 sum.
        e = jnp.exp(s - jnp.max(s, -1, keepdims=True))
        o = jnp.einsum("gqts,sgd->tgqd", e.astype(BF16).astype(F32), vv)
        o = (o / jnp.einsum("gqts->tgq", e)[..., None]).reshape(t, c.NH * c.HD)
    else:
        m = jnp.full(s.shape[:-1] + (1,), -jnp.inf)
        l = jnp.zeros_like(m)
        acc = jnp.zeros(s.shape[:-1] + (c.HD,))
        for j0 in range(0, s.shape[-1], flash_bkv):
            # bf16 scores and exponentials, f32 running max / sum / accumulator (serving RPA).
            st = s[..., j0:j0 + flash_bkv].astype(BF16)
            m_new = jnp.maximum(m, jnp.max(st, -1, keepdims=True).astype(F32))
            alpha = jnp.where(jnp.isfinite(m_new), jnp.exp(m - m_new), 1.0)
            pe = jnp.where(jnp.isfinite(m_new), jnp.exp(st - m_new.astype(BF16)), 0.0)
            pe = pe.astype(BF16).astype(F32)
            l = alpha * l + jnp.sum(pe, -1, keepdims=True)
            acc = alpha * acc + jnp.einsum("gqts,sgd->gqtd", pe.astype(BF16).astype(F32),
                                           vv[j0:j0 + flash_bkv])
            m = m_new
        o = jnp.einsum("gqtd->tgqd", acc / l).reshape(t, c.NH * c.HD)
    o = (o * jax.nn.sigmoid(gate.astype(F32))).astype(BF16)
    return qlinear(o, w["o_w"], w["o_s"]), k_cache, v_cache


def moe(c, w, x):
    """x [T, H] (normed, bf16) -> [T, H] f32: routed top-k experts plus the gated shared expert."""
    t = x.shape[0]
    fused = qlinear(x, w["gate_gu_w"], w["gate_gu_s"])                 # [T, E + 2 IS]
    logits, sgu = fused[:, :c.E], fused[:, c.E:]
    probs = jax.nn.softmax(logits, -1)
    topw, topi = jax.lax.top_k(probs, c.TOPK)
    topw = topw / jnp.sum(topw, -1, keepdims=True)
    # Routed experts, as a dense gather per (token, slot): fine for a reference.
    def expert(xi, ei):
        gu = qlinear(xi[None], w["w13"][ei], w["w13_s"][ei])[0]
        h = (jax.nn.silu(gu[:c.I]) * gu[c.I:]).astype(BF16)
        return qlinear(h[None], w["w2"][ei], w["w2_s"][ei])[0]
    ys = jax.vmap(lambda xi, ids: jax.vmap(lambda e: expert(xi, e))(ids))(x, topi)  # [T,K,H]
    routed = jnp.sum(ys * topw[..., None], 1)
    sh = (jax.nn.silu(sgu[:, :c.IS]) * sgu[:, c.IS:]).astype(BF16)
    shared = qlinear(sh, w["sd_w"], w["sd_s"])
    sgate = jax.nn.sigmoid(jax.lax.dot_general(x, w["sgate_w"], (((1,), (0,)), ((), ())),
                                               preferred_element_type=F32))
    return routed + shared * sgate


def stage_step(c, weights, hidden, residual, states, ctx, flash_bkv=None, moe_fn=None):
    """weights: per-layer dicts; states: per-layer dicts (gdn: conv [CONV-1, QKV] bf16,
    rec [VH, KD, VD] f32; attn: k, v [ctx + T, NKV, HD] fp8). Returns hidden, residual
    (bf16, [T, H]) and the new states."""
    new_states = []
    for lt, w, st in zip(c.layer_types, weights, states):
        res = hidden.astype(F32) + residual.astype(F32)
        residual = res.astype(BF16)
        x = rmsnorm_zc(res, w["in_norm"], c.eps).astype(BF16)
        if lt == "gdn":
            y, conv, rec = gdn_layer(c, w, x, st["conv"], st["rec"])
            new_states.append({"conv": conv, "rec": rec})
        else:
            y, kc, vc = attn_layer(c, w, x, st["k"], st["v"], ctx, flash_bkv)
            new_states.append({"k": kc, "v": vc})
        res = y.astype(BF16).astype(F32) + residual.astype(F32)
        residual = res.astype(BF16)
        x = rmsnorm_zc(res, w["post_norm"], c.eps).astype(BF16)
        hidden = (moe_fn or moe)(c, w, x).astype(BF16)
    return hidden, residual, new_states


def random_stage(c, t, ctx, seed=0, n_layers=None):
    """Random weights and states with the serving layouts, and a random (hidden, residual)."""
    rng = np.random.default_rng(seed)
    types = c.layer_types[:n_layers] if n_layers else c.layer_types

    def fp8w(din, dout, *lead):
        w = rng.standard_normal((*lead, din, dout), np.float32) * 0.5
        s = rng.uniform(0.5, 1.5, (*lead, dout)).astype(np.float32) / np.sqrt(din) / 0.5
        return jnp.asarray(w, FP8), jnp.asarray(s)

    def vec(n, scale=0.1, dtype=BF16):
        return jnp.asarray(rng.standard_normal(n, np.float32) * scale, dtype)

    weights, states = [], []
    for lt in types:
        w = {"in_norm": vec(c.H), "post_norm": vec(c.H)}
        if lt == "gdn":
            w["qkvz_w"], w["qkvz_s"] = fp8w(c.H, c.QKV + c.Z)
            w["ba_w"] = vec((c.H, 2 * c.VH), 0.02)
            w["conv_w"] = vec((c.QKV, c.CONV), 0.3)
            w["a_log"] = jnp.asarray(np.log(rng.uniform(1, 16, c.VH)), F32)
            w["dt_bias"] = vec(c.VH, 1.0, F32)
            w["gnorm_w"] = jnp.asarray(1.0 + 0.1 * rng.standard_normal(c.VD), F32)
            w["out_w"], w["out_s"] = fp8w(c.Z, c.H)
            states.append({"conv": vec((c.CONV - 1, c.QKV), 0.5),
                           "rec": jnp.asarray(rng.standard_normal((c.VH, c.KD, c.VD)) * 0.05, F32)})
        else:
            w["qkv_w"], w["qkv_s"] = fp8w(c.H, c.ATT_QKV)
            w["q_norm"], w["k_norm"] = vec(c.HD), vec(c.HD)
            w["o_w"], w["o_s"] = fp8w(c.NH * c.HD, c.H)
            kv = lambda: jnp.asarray(rng.standard_normal((ctx + t, c.NKV, c.HD)), FP8)
            states.append({"k": kv(), "v": kv()})
        w["gate_gu_w"], w["gate_gu_s"] = fp8w(c.H, c.E + 2 * c.IS)
        w["w13"], w["w13_s"] = fp8w(c.H, 2 * c.I, c.E)
        w["w2"], w["w2_s"] = fp8w(c.I, c.H, c.E)
        w["sd_w"], w["sd_s"] = fp8w(c.IS, c.H)
        w["sgate_w"] = vec((c.H, 1), 0.02)
        weights.append(w)
    hidden = vec((t, c.H), 1.0)
    residual = vec((t, c.H), 1.0)
    return types, weights, states, hidden, residual


def _quant_rows(x):
    x = x.astype(BF16).astype(F32)
    amax = jnp.max(jnp.abs(x), -1, keepdims=True)
    s = jnp.where(amax > 0, amax * (1.0 / FP8_MAX), 1.0)
    return jnp.clip(x / s, -FP8_MAX, FP8_MAX).astype(FP8), s


def moe_ragged(c, w, x):
    """moe() at full scale: slots grouped by expert (stable argsort) and run through
    jax.lax.ragged_dot, w8a8 like the serving grouped matmul."""
    t = x.shape[0]
    fused = qlinear(x, w["gate_gu_w"], w["gate_gu_s"])
    logits, sgu = fused[:, :c.E], fused[:, c.E:]
    topw, topi = jax.lax.top_k(jax.nn.softmax(logits, -1), c.TOPK)
    topw = topw / jnp.sum(topw, -1, keepdims=True)
    flat_e = topi.reshape(-1)
    order = jnp.argsort(flat_e, stable=True)
    e_sorted = flat_e[order]
    sizes = jnp.bincount(flat_e, length=c.E).astype(jnp.int32)
    xq, xs = _quant_rows(x[order // c.TOPK])
    gu = jax.lax.ragged_dot(xq, w["w13"], sizes, preferred_element_type=F32)
    gu = (gu * xs * w["w13_s"][e_sorted]).astype(BF16).astype(F32)
    hq, hs = _quant_rows(jax.nn.silu(gu[:, :c.I]) * gu[:, c.I:])
    y = jax.lax.ragged_dot(hq, w["w2"], sizes, preferred_element_type=F32)
    y = (y * hs * w["w2_s"][e_sorted]).astype(BF16).astype(F32)
    y = jnp.zeros_like(y).at[order].set(y).reshape(t, c.TOPK, c.H)
    routed = jnp.sum(y * topw[..., None], 1)
    sh = (jax.nn.silu(sgu[:, :c.IS]) * sgu[:, c.IS:]).astype(BF16)
    shared = qlinear(sh, w["sd_w"], w["sd_s"])
    sgate = jax.nn.sigmoid(jax.lax.dot_general(x, w["sgate_w"], (((1,), (0,)), ((), ())),
                                               preferred_element_type=F32))
    return routed + shared * sgate


def random_stage_device(c, t, ctx, key, kv_rows=None):
    """random_stage on the device (jax.random), for full-size stages. Same layouts."""
    kv_rows = kv_rows or ctx + t
    keys = iter(jax.random.split(key, 4096))
    nrm = lambda shape, s=1.0, dt=F32: (jax.random.normal(next(keys), shape, F32) * s).astype(dt)

    def fp8w(din, dout, *lead):
        w = nrm((*lead, din, dout), 0.5, FP8)
        s = jax.random.uniform(next(keys), (*lead, dout), F32, 0.5, 1.5) / np.sqrt(din) / 0.5
        return w, s

    weights, states = [], []
    for lt in c.layer_types:
        w = {"in_norm": nrm((c.H,), 0.1, BF16), "post_norm": nrm((c.H,), 0.1, BF16)}
        if lt == "gdn":
            w["qkvz_w"], w["qkvz_s"] = fp8w(c.H, c.QKV + c.Z)
            w["ba_w"] = nrm((c.H, 2 * c.VH), 0.02, BF16)
            w["conv_w"] = nrm((c.QKV, c.CONV), 0.3, BF16)
            w["a_log"] = jnp.log(jax.random.uniform(next(keys), (c.VH,), F32, 1.0, 16.0))
            w["dt_bias"] = nrm((c.VH,))
            w["gnorm_w"] = 1.0 + nrm((c.VD,), 0.1)
            w["out_w"], w["out_s"] = fp8w(c.Z, c.H)
            states.append({"conv": nrm((c.CONV - 1, c.QKV), 0.5, BF16),
                           "rec": nrm((c.VH, c.KD, c.VD), 0.05)})
        else:
            w["qkv_w"], w["qkv_s"] = fp8w(c.H, c.ATT_QKV)
            w["q_norm"], w["k_norm"] = nrm((c.HD,), 0.1, BF16), nrm((c.HD,), 0.1, BF16)
            w["o_w"], w["o_s"] = fp8w(c.NH * c.HD, c.H)
            states.append({"k": nrm((kv_rows, c.NKV, c.HD), 1.0, FP8),
                           "v": nrm((kv_rows, c.NKV, c.HD), 1.0, FP8)})
        w["gate_gu_w"], w["gate_gu_s"] = fp8w(c.H, c.E + 2 * c.IS)
        w["w13"], w["w13_s"] = fp8w(c.H, 2 * c.I, c.E)
        w["w2"], w["w2_s"] = fp8w(c.I, c.H, c.E)
        w["sd_w"], w["sd_s"] = fp8w(c.IS, c.H)
        w["sgate_w"] = nrm((c.H, 1), 0.02, BF16)
        weights.append(w)
    return weights, states, nrm((t, c.H), 1.0, BF16), nrm((t, c.H), 1.0, BF16)
