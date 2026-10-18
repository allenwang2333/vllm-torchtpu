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
"""Phases of the Qwen3.5 prefill megakernel.

Every phase runs inside the one stage kernel on HBM refs (memory space ANY): it is an
`emit_pipeline` over its own tiles, so intermediates live in HBM workspaces between phases and
each phase streams its operands through VMEM with double buffering.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32, BF16 = jnp.float32, jnp.bfloat16
FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)


@dataclasses.dataclass(frozen=True)
class Tiles:
    tm_norm: int = 256     # rows per norm / elementwise tile
    tm: int = 512          # rows per matmul tile
    tn: int = 512          # columns per matmul tile


def quantize_rows(x):
    """Per-row dynamic fp8: returns (x_q fp8, scale [rows, 1] f32), x ~= x_q * scale.
    `x` is quantized from its bf16 value, as the serving path quantizes a bf16 activation."""
    x = x.astype(BF16).astype(F32)
    amax = jnp.max(jnp.abs(x), -1, keepdims=True)
    s = jnp.where(amax > 0, amax * (1.0 / FP8_MAX), 1.0)
    return jnp.clip(x / s, -FP8_MAX, FP8_MAX).astype(FP8), s


def norm_add_quant(h_hbm, r_hbm, w_hbm, r_out_hbm, x_bf_hbm, xq_hbm, xs_hbm, *, eps, tm,
                   xf_hbm=None):
    """residual' = h + r (stored bf16); x = rmsnorm(h + r in f32) * (1 + w) -> bf16 and per-row fp8.

    h, r, r_out, x_bf: [T, H] bf16; w: [1, H] bf16; xq: [T, H] fp8; xs: [T, 1] f32.
    r_out may alias r (each tile is read before it is written). xf: optional [T, H] f32 copy of
    x (32-bit rows, which single-row DMA gathers need)."""
    t, h = h_hbm.shape

    def body(h_ref, r_ref, w_ref, ro_ref, xb_ref, xq_ref, xs_ref, *xf_ref):
        # vLLM's fused_add_rms_norm: the residual is stored in bf16, the norm uses the f32 sum.
        res = h_ref[...].astype(F32) + r_ref[...].astype(F32)
        ro_ref[...] = res.astype(BF16)
        x = res * jax.lax.rsqrt(jnp.mean(res * res, -1, keepdims=True) + eps)
        x = (x * (1.0 + w_ref[...].astype(F32))).astype(BF16)
        xb_ref[...] = x
        if xf_ref:
            xf_ref[0][...] = x.astype(F32)
        xq, xs = quantize_rows(x)
        xq_ref[...] = xq
        xs_ref[...] = xs

    row = pl.BlockSpec((tm, h), lambda i: (i, 0))
    pltpu.emit_pipeline(
        body,
        grid=(t // tm,),
        in_specs=[row, row, pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[row, row, row, pl.BlockSpec((tm, 1), lambda i: (i, 0))]
        + ([row] if xf_hbm is not None else []),
    )(h_hbm, r_hbm, w_hbm, r_out_hbm, x_bf_hbm, xq_hbm, xs_hbm,
      *([xf_hbm] if xf_hbm is not None else []))


def qmatmul(xq_hbm, xs_hbm, w_hbm, ws_hbm, out_hbm, *, tm, tn, n0=0, nn=None, out_n0=0,
            epilogue=None):
    """out[:, out_n0*tn + j] = epilogue((xq @ w)[:, (n0 + j)*tn] * xs * ws), w8a8.

    xq [T, K] fp8, xs [T, 1] f32, w [K, N] fp8, ws [1, N] f32, out [T, N_out]. Full-K blocks:
    no accumulator, one MXU pass per (row tile, column tile). Column tiles are the outer loop,
    so a weight tile stays in VMEM while the row tiles stream past it."""
    t, k = xq_hbm.shape
    n_tiles = w_hbm.shape[1] // tn if nn is None else nn
    epilogue = epilogue or (lambda y, n: y)

    def body(x_ref, s_ref, w_ref, ws_ref, o_ref):
        n = pl.program_id(0)
        y = jax.lax.dot_general(x_ref[...], w_ref[...], (((1,), (0,)), ((), ())),
                                preferred_element_type=F32)
        y = y * s_ref[...] * ws_ref[...]
        o_ref[...] = epilogue(y, n).astype(o_ref.dtype)

    pltpu.emit_pipeline(
        body,
        grid=(n_tiles, t // tm),
        in_specs=[
            pl.BlockSpec((tm, k), lambda n, m: (m, 0)),
            pl.BlockSpec((tm, 1), lambda n, m: (m, 0)),
            pl.BlockSpec((k, tn), lambda n, m: (0, n0 + n)),
            pl.BlockSpec((1, tn), lambda n, m: (0, n0 + n)),
        ],
        out_specs=[pl.BlockSpec((tm, tn), lambda n, m: (m, out_n0 + n))],
    )(xq_hbm, xs_hbm, w_hbm, ws_hbm, out_hbm)


def bmatmul(x_hbm, w_hbm, out_hbm, *, tm, tn=None):
    """out = x @ w in bf16 with f32 accumulation (the small bf16 projections: in_proj_ba,
    shared_expert_gate). x [T, K] bf16, w [K, N] bf16, out [T, N]."""
    t, k = x_hbm.shape
    n = w_hbm.shape[1]
    tn = tn or n

    def body(x_ref, w_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(x_ref[...], w_ref[...], (((1,), (0,)), ((), ())),
                                         preferred_element_type=F32).astype(o_ref.dtype)

    pltpu.emit_pipeline(
        body,
        grid=(n // tn, t // tm),
        in_specs=[pl.BlockSpec((tm, k), lambda j, m: (m, 0)),
                  pl.BlockSpec((k, tn), lambda j, m: (0, j))],
        out_specs=[pl.BlockSpec((tm, tn), lambda j, m: (m, j))],
    )(x_hbm, w_hbm, out_hbm)


def swiglu_quant(gu_hbm, xq_hbm, xs_hbm, *, inter, tm, g0=0):
    """h = silu(gu[:, g0:g0+I]) * gu[:, g0+I:g0+2I] -> bf16 -> per-row fp8 (xq [T, I], xs)."""
    t = gu_hbm.shape[0]
    lanes = 2 * inter

    def body(gu_ref, xq_ref, xs_ref):
        gu = gu_ref[...].astype(F32)
        h = jax.nn.silu(gu[:, :inter]) * gu[:, inter:]
        xq, xs = quantize_rows(h)
        xq_ref[...] = xq
        xs_ref[...] = xs

    assert g0 % lanes == 0
    pltpu.emit_pipeline(
        body,
        grid=(t // tm,),
        in_specs=[pl.BlockSpec((tm, lanes), lambda i: (i, g0 // lanes))],
        out_specs=[pl.BlockSpec((tm, inter), lambda i: (i, 0)),
                   pl.BlockSpec((tm, 1), lambda i: (i, 0))],
    )(gu_hbm, xq_hbm, xs_hbm)


# ------------------------------------------------------------------------------------------------
# GatedDeltaNet
# ------------------------------------------------------------------------------------------------


def gdn_prep(qkvz_hbm, ba_hbm, conv_state_hbm, conv_w_hbm, a_log_hbm, dt_bias_hbm,
             q_hbm, k_hbm, v_hbm, gb_hbm, conv_out_hbm, *, kq, vh, kd, vd, ks, eps, tm):
    """Causal depthwise conv1d over time (carrying the conv state), SiLU, split, L2-norm of q and
    k (q scaled by kd^-0.5), and the gates: g = -exp(A_log) * softplus(a + dt_bias),
    beta = sigmoid(b).

    qkvz [T, QKV + Z] bf16 (conv input = columns :QKV), ba [T, 2 vh] bf16, conv_state
    [ks - 1, QKV] bf16, conv_w [ks, QKV] bf16, a_log / dt_bias [1, vh] f32. Writes q, k
    [T, kq * kd] f32, v [T, vh * vd] f32, gb [T, 2 vh] f32 (g | beta) and the new conv state
    (the sequence's last ks - 1 conv inputs)."""
    t = qkvz_hbm.shape[0]
    qkv = 2 * kq * kd + vh * vd
    halo = 16                                 # >= ks - 1; whole bf16 (16, 128) tiles
    assert tm % halo == 0

    def run(prev_ref, cs_v, sem):
        # prev_ref holds the 16 input rows before the current tile (f32); the conv taps use its
        # last ks - 1. Every DMA and store is whole and tile aligned.
        cp = pltpu.make_async_copy(conv_state_hbm, cs_v, sem.at[0])
        cp.start()
        cp.wait()
        prev_ref[...] = jnp.concatenate(
            [jnp.zeros((halo - (ks - 1), qkv), F32), cs_v[...].astype(F32)], 0)

        def body(x_ref, ba_ref, w_ref, al_ref, dt_ref, q_ref, k_ref, v_ref, gb_ref):
            x = x_ref[...].astype(F32)                                  # [tm, QKV]
            full = jnp.concatenate([prev_ref[...], x], 0)               # [halo + tm, QKV]
            w = w_ref[...].astype(F32)                                  # [ks, QKV]
            acc = jnp.zeros_like(x)
            for j in range(ks):
                acc = acc + full[halo - (ks - 1) + j:halo - (ks - 1) + j + tm] * w[j:j + 1]
            prev_ref[...] = x[tm - halo:tm]
            y = jax.nn.silu(acc.astype(BF16).astype(F32))
            kdim = kq * kd
            def l2(z):
                z = z.reshape(tm, kq, kd)
                z = z * jax.lax.rsqrt(jnp.sum(z * z, -1, keepdims=True) + 1e-6)
                return z.reshape(tm, kdim)
            q_ref[...] = l2(y[:, :kdim]) * (kd ** -0.5)
            k_ref[...] = l2(y[:, kdim:2 * kdim])
            v_ref[...] = y[:, 2 * kdim:]
            ba = ba_ref[...].astype(F32)
            beta = jax.nn.sigmoid(ba[:, :vh])
            g = -jnp.exp(al_ref[...]) * jax.nn.softplus(ba[:, vh:] + dt_ref[...])
            gb_ref[...] = jnp.concatenate([g, beta], -1)

        kdim = kq * kd
        pltpu.emit_pipeline(
            body,
            grid=(t // tm,),
            in_specs=[pl.BlockSpec((tm, qkv), lambda i: (i, 0)),
                      pl.BlockSpec((tm, 2 * vh), lambda i: (i, 0)),
                      pl.BlockSpec((ks, qkv), lambda i: (0, 0)),
                      pl.BlockSpec((1, vh), lambda i: (0, 0)),
                      pl.BlockSpec((1, vh), lambda i: (0, 0))],
            out_specs=[pl.BlockSpec((tm, kdim), lambda i: (i, 0)),
                       pl.BlockSpec((tm, kdim), lambda i: (i, 0)),
                       pl.BlockSpec((tm, vh * vd), lambda i: (i, 0)),
                       pl.BlockSpec((tm, 2 * vh), lambda i: (i, 0))],
        )(qkvz_hbm, ba_hbm, conv_w_hbm, a_log_hbm, dt_bias_hbm, q_hbm, k_hbm, v_hbm, gb_hbm)
        # New conv state: the sequence's last ks - 1 conv inputs, from the halo.
        cs_v[...] = prev_ref[halo - (ks - 1):, :].astype(cs_v.dtype)
        cp = pltpu.make_async_copy(cs_v, conv_out_hbm, sem.at[0])
        cp.start()
        cp.wait()

    pl.run_scoped(run, pltpu.VMEM((halo, qkv), F32), pltpu.VMEM((ks - 1, qkv), BF16),
                  pltpu.SemaphoreType.DMA((1,)))


def gdn_chunks(q_hbm, k_hbm, v_hbm, gb_hbm, s0_hbm, o_hbm, s_out_hbm, *, kq, vh, kd, vd, chunk):
    """The gated delta rule over the sequence, chunk by chunk, all v-heads at once, the
    [vh, kd, vd] f32 state resident in VMEM. Per chunk (see gdn_chunked for the algebra):
    factors (cumulative decay, A, T = (I + A)^-1 by repeated squaring, Ub, W, M) then
    U = Ub - W S, O = Qg S + M U, S = Gamma_C S + Kd^T U.

    q, k [T, kq * kd] f32, v [T, vh * vd] f32, gb [T, 2 vh] f32, s0 / s_out [vh, kd, vd] f32,
    o [T, vh * vd] bf16."""
    t = q_hbm.shape[0]
    c = chunk
    rep = vh // kq
    bmm = (((2,), (1,)), ((0,), (0,)))        # [H,a,b] x [H,b,c]
    bnt = (((2,), (2,)), ((0,), (0,)))        # [H,a,d] x [H,b,d] -> [H,a,b]
    btn = (((1,), (1,)), ((0,), (0,)))        # [H,c,a] x [H,c,b] -> [H,a,b]
    dot = functools.partial(jax.lax.dot_general, preferred_element_type=F32)

    def run(s_ref, sem):
        cp = pltpu.make_async_copy(s0_hbm, s_ref, sem.at[0])
        cp.start()
        cp.wait()
        ti = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
        si = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
        strict, lower = ti > si, ti >= si
        upper_incl = (si >= ti).astype(F32)     # cumsum along t: g @ U, U[s, t] = s <= t
        eye = (ti == si).astype(F32)

        def heads(x, d, per_k):                  # [C, n*d] -> [vh, C, d]
            n = kq if per_k else vh
            x = x.reshape(c, n, d)
            if per_k:
                x = jnp.repeat(x, rep, axis=1)
            return jnp.swapaxes(x, 0, 1)

        def body(q_ref, k_ref, v_ref, gb_ref, o_ref):
            q = heads(q_ref[...], kd, True)                               # [H, C, kd]
            k = heads(k_ref[...], kd, True)
            v = heads(v_ref[...], vd, False)                              # [H, C, vd]
            gb = gb_ref[...]
            g = jnp.swapaxes(gb[:, :vh], 0, 1)                            # [H, C]
            beta = jnp.swapaxes(gb[:, vh:], 0, 1)
            gam = jax.lax.dot_general(g, upper_incl, (((1,), (0,)), ((), ())),
                                      preferred_element_type=F32)         # inclusive cumsum
            diff = gam[:, :, None] - gam[:, None, :]
            dec = jnp.exp(jnp.where(lower[None], diff, -jnp.inf))
            a = jnp.where(strict[None], beta[:, :, None] * dec * dot(k, k, bnt), 0.0)
            tinv = eye[None] - a
            p = a
            n = 2
            while n < c:
                p = dot(p, p, bmm)
                tinv = tinv + dot(tinv, p, bmm)
                n *= 2
            big = jnp.exp(gam)
            ub = dot(tinv, beta[:, :, None] * v, bmm)
            w = dot(tinv, (beta * big)[:, :, None] * k, bmm)
            m = jnp.where(lower[None], dec * dot(q, k, bnt), 0.0)
            qg = big[:, :, None] * q
            kdec = jnp.exp(gam[:, -1:] - gam)[:, :, None] * k
            s = s_ref[...]
            u = ub - dot(w, s, bmm)
            o = dot(qg, s, bmm) + dot(m, u, bmm)                          # [H, C, vd]
            s_ref[...] = big[:, -1][:, None, None] * s + dot(kdec, u, btn)
            o_ref[...] = jnp.swapaxes(o, 0, 1).reshape(c, vh * vd).astype(o_ref.dtype)

        pltpu.emit_pipeline(
            body,
            grid=(t // c,),
            in_specs=[pl.BlockSpec((c, kq * kd), lambda i: (i, 0)),
                      pl.BlockSpec((c, kq * kd), lambda i: (i, 0)),
                      pl.BlockSpec((c, vh * vd), lambda i: (i, 0)),
                      pl.BlockSpec((c, 2 * vh), lambda i: (i, 0))],
            out_specs=[pl.BlockSpec((c, vh * vd), lambda i: (i, 0))],
        )(q_hbm, k_hbm, v_hbm, gb_hbm, o_hbm)
        cp = pltpu.make_async_copy(s_ref, s_out_hbm, sem.at[0])
        cp.start()
        cp.wait()

    pl.run_scoped(run, pltpu.VMEM((vh, kd, vd), F32), pltpu.SemaphoreType.DMA((1,)))


def gdn_gated_norm_quant(o_hbm, qkvz_hbm, w_hbm, xq_hbm, xs_hbm, *, vh, vd, qkv, eps, tm):
    """y = rmsnorm(o per v-head) * w * silu(z) -> bf16 -> per-row fp8 for out_proj.
    o [T, vh * vd] bf16, z = qkvz[:, qkv:qkv + vh * vd], w [1, vd] f32, xq [T, vh * vd] fp8,
    xs [T, 1]. z is read as column blocks of gcd(qkv, vh * vd) so that they start at qkv."""
    t = o_hbm.shape[0]
    z_cols = vh * vd
    zb = math.gcd(qkv, z_cols)
    nzb = z_cols // zb

    def body(o_ref, *refs):
        z_refs, (w_ref, xq_ref, xs_ref) = refs[:nzb], refs[nzb:]
        o = o_ref[...].astype(F32).reshape(tm, vh, vd)
        z = jnp.concatenate([r[...] for r in z_refs], -1).astype(F32).reshape(tm, vh, vd)
        y = o * jax.lax.rsqrt(jnp.mean(o * o, -1, keepdims=True) + eps)
        y = y * w_ref[...].reshape(1, 1, vd) * jax.nn.silu(z)
        xq, xs = quantize_rows(y.reshape(tm, z_cols))
        xq_ref[...] = xq
        xs_ref[...] = xs

    z_specs = [pl.BlockSpec((tm, zb), lambda i, j=j: (i, qkv // zb + j)) for j in range(nzb)]
    pltpu.emit_pipeline(
        body,
        grid=(t // tm,),
        in_specs=[pl.BlockSpec((tm, z_cols), lambda i: (i, 0)), *z_specs,
                  pl.BlockSpec((1, vd), lambda i: (0, 0))],
        out_specs=[pl.BlockSpec((tm, z_cols), lambda i: (i, 0)),
                   pl.BlockSpec((tm, 1), lambda i: (i, 0))],
    )(o_hbm, *([qkvz_hbm] * nzb), w_hbm, xq_hbm, xs_hbm)


# ------------------------------------------------------------------------------------------------
# Gated attention
# ------------------------------------------------------------------------------------------------


def att_tile_dest(n, nh, nkv):
    """qkv_proj column tile n (256 wide) -> head-major buffer slot. The projection's columns are
    [q_0 | gate_0 | q_1 | gate_1 | ... | k_0 .. k_{nkv-1} | v_0 ..]; slots are
    [q_0 .. q_{nh-1} | gate_0 .. | k_0 .. | v_0 ..]."""
    qg = n < 2 * nh
    return jnp.where(qg, (n % 2) * nh + n // 2, n)


def qmatmul_heads(xq_hbm, xs_hbm, w_hbm, ws_hbm, out_hbm, *, tm, nh, nkv, hd):
    """qkv_proj, w8a8, written head-major: out [2 nh + 2 nkv, T, hd] bf16 (att_tile_dest)."""
    t, k = xq_hbm.shape
    n_tiles = w_hbm.shape[1] // hd

    def body(x_ref, s_ref, w_ref, ws_ref, o_ref):
        y = jax.lax.dot_general(x_ref[...], w_ref[...], (((1,), (0,)), ((), ())),
                                preferred_element_type=F32)
        o_ref[...] = (y * s_ref[...] * ws_ref[...]).astype(o_ref.dtype)[None]

    pltpu.emit_pipeline(
        body,
        grid=(n_tiles, t // tm),
        in_specs=[pl.BlockSpec((tm, k), lambda n, m: (m, 0)),
                  pl.BlockSpec((tm, 1), lambda n, m: (m, 0)),
                  pl.BlockSpec((k, hd), lambda n, m: (0, n)),
                  pl.BlockSpec((1, hd), lambda n, m: (0, n))],
        out_specs=[pl.BlockSpec((1, tm, hd), lambda n, m: (att_tile_dest(n, nh, nkv), m, 0))],
    )(xq_hbm, xs_hbm, w_hbm, ws_hbm, out_hbm)


def att_qk_norm_rope_kv(att_hbm, qn_w_hbm, kn_w_hbm, rope_hbm, ctx, qn_hbm, kc_hbm, vc_hbm, *,
                        nh, nkv, hd, rot, eps, tm):
    """Per-head zero-centered RMSNorm of q and k, NeoX RoPE on their first `rot` dims at
    positions ctx + row, then q -> qn [nh, T, hd] bf16 and the new k, v -> fp8 into the caches
    kc, vc [nkv, S, hd] at rows ctx .. ctx + T - 1 (by DMA: ctx is dynamic).

    att [2 nh + 2 nkv, T, hd] bf16 (head-major, see att_tile_dest), qn_w / kn_w [1, hd] bf16,
    rope [S, rot] f32 (cos | sin, rot / 2 each) for absolute positions, ctx: traced scalar.
    Precondition: ctx % 32 == 0 (the fp8 cache tile). It holds for the PP prefill this serves:
    prefix hits land on 2176-token KV blocks and pipeline chunks on 2304-token granules."""
    t = att_hbm.shape[1]
    half = rot // 2

    def run(rope_v, k8, v8, sem):
        def body(q_ref, k_ref, v_ref, qw_ref, kw_ref, qo_ref):
            m = pl.program_id(0)
            r0 = pl.multiple_of(ctx + m * tm, 32)
            cp = pltpu.make_async_copy(rope_hbm.at[pl.ds(r0, tm)], rope_v, sem.at[0])
            cp.start()
            cp.wait()
            cs = rope_v[...]
            cos, sin = cs[:, :half][None], cs[:, half:][None]           # [1, tm, half]

            def norm_rope(x, w):                                         # x [n, tm, hd]
                x = x.astype(F32)
                x = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)
                x = (x * (1.0 + w.astype(F32).reshape(1, 1, hd))).astype(BF16).astype(F32)
                x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
                y = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)
                return y.astype(BF16)

            qo_ref[...] = norm_rope(q_ref[...], qw_ref[...])
            k8[...] = norm_rope(k_ref[...], kw_ref[...]).astype(FP8)
            v8[...] = v_ref[...].astype(FP8)
            ck = pltpu.make_async_copy(k8, kc_hbm.at[:, pl.ds(r0, tm)], sem.at[1])
            cv = pltpu.make_async_copy(v8, vc_hbm.at[:, pl.ds(r0, tm)], sem.at[2])
            ck.start()
            cv.start()
            ck.wait()
            cv.wait()

        kslot, vslot = 2 * nh // nkv, (2 * nh + nkv) // nkv
        pltpu.emit_pipeline(
            body,
            grid=(t // tm,),
            in_specs=[pl.BlockSpec((nh, tm, hd), lambda m: (0, m, 0)),
                      pl.BlockSpec((nkv, tm, hd), lambda m: (kslot, m, 0)),
                      pl.BlockSpec((nkv, tm, hd), lambda m: (vslot, m, 0)),
                      pl.BlockSpec((1, hd), lambda m: (0, 0)),
                      pl.BlockSpec((1, hd), lambda m: (0, 0))],
            out_specs=[pl.BlockSpec((nh, tm, hd), lambda m: (0, m, 0))],
        )(att_hbm, att_hbm, att_hbm, qn_w_hbm, kn_w_hbm, qn_hbm)

    pl.run_scoped(run, pltpu.VMEM((tm, rot), F32), pltpu.VMEM((nkv, tm, hd), FP8),
                  pltpu.VMEM((nkv, tm, hd), FP8), pltpu.SemaphoreType.DMA((3,)))


def att_flash(qn_hbm, kc_hbm, vc_hbm, ctx, o_hbm, *, nh, nkv, hd, bq, bkv, s_max, sub=4):
    """Causal GQA attention of the T new tokens (positions ctx + t) over the cache rows
    0 .. ctx + T - 1, per kv head with its nh / nkv q heads folded into the rows.

    Like the serving RPA: the 1/sqrt(hd) scale is folded into q when it is a power of two (exact;
    hd 256 gives 1/16), scores
    go to bf16 for the max / exp (sum, running max and accumulator in f32), and only KV tiles
    that cross the causal diagonal are masked. The rows are split into `sub` chunks, unrolled, so
    one chunk's QK matmul can overlap another's softmax.
    qn [nh, T, hd] bf16, kc / vc [nkv, S, hd] fp8, o [nh, T, hd] f32."""
    t = qn_hbm.shape[1]
    grp = nh // nkv
    rows = grp * bq
    rc = rows // sub
    n_kv = pl.cdiv(s_max, bkv)
    scale = hd ** -0.5
    fold = math.frexp(scale)[0] == 0.5      # exact in q only for a power of two (hd 256: 1/16)

    def last_kv(i):                         # last kv tile the q tile i needs
        return (ctx + (i + 1) * bq - 1) // bkv

    def run(m_ref, l_ref, acc_ref):
        def step(q_ref, k_ref, v_ref, i, j, masked):
            kk = k_ref[0].astype(BF16)
            vv = v_ref[0].astype(BF16)

            def pv(c, p, alpha):                 # chunk c's PV and accumulator update
                r = slice(c * rc, (c + 1) * rc)
                acc_ref[r, :] = alpha * acc_ref[r, :] + jax.lax.dot_general(
                    p, vv, (((1,), (0,)), ((), ())), preferred_element_type=F32)

            # Software pipelined like the serving RPA: chunk c's QK + softmax, then chunk c-1's
            # PV, so the MXU has independent work while the VPU runs the softmax.
            prev = None
            for c in range(sub):
                r = slice(c * rc, (c + 1) * rc)
                q = q_ref[...].reshape(rows, hd)[r]
                if fold:
                    q = q * jnp.asarray(scale, BF16)
                s = jax.lax.dot_general(q, kk, (((1,), (1,)), ((), ())),
                                        preferred_element_type=F32)
                s = (s if fold else s * scale).astype(BF16)
                if masked:
                    row = jax.lax.broadcasted_iota(jnp.int32, (rc, 1), 0) + c * rc
                    col = jax.lax.broadcasted_iota(jnp.int32, (1, bkv), 1)
                    pos = ctx + i * bq + row % bq        # rows are (head, token), head-major
                    s = jnp.where(j * bkv + col <= pos, s, -jnp.inf)
                m_prev = m_ref[r, :1]
                m_new = jnp.maximum(m_prev, jnp.max(s, -1, keepdims=True).astype(F32))
                alpha = jnp.exp(m_prev - m_new)
                p = jnp.exp(s - m_new.astype(BF16))
                l_ref[r, :] = jnp.broadcast_to(
                    alpha * l_ref[r, :1] + jnp.sum(p, -1, keepdims=True, dtype=F32), (rc, 128))
                m_ref[r, :] = jnp.broadcast_to(m_new, (rc, 128))
                if prev is not None:
                    pv(*prev)
                prev = (c, p, alpha)
            pv(*prev)

        def body(q_ref, k_ref, v_ref, o_ref):
            i, j = pl.program_id(1), pl.program_id(2)

            @pl.when(j == 0)
            def _():
                m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, F32)
                l_ref[...] = jnp.zeros(l_ref.shape, F32)
                acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

            # Fully below the diagonal: every key of the tile precedes the tile's first query.
            full = (j + 1) * bkv - 1 <= ctx + i * bq
            live = j <= last_kv(i)

            @pl.when(live & full)
            def _():
                step(q_ref, k_ref, v_ref, i, j, False)

            @pl.when(live & jnp.logical_not(full))
            def _():
                step(q_ref, k_ref, v_ref, i, j, True)

            @pl.when(j == n_kv - 1)
            def _():
                o_ref[...] = (acc_ref[...] / l_ref[:, :1]).reshape(grp, bq, hd)

        kv_map = lambda g, i, j: (g, jnp.minimum(j, last_kv(i)), 0)
        pltpu.emit_pipeline(
            body,
            grid=(nkv, t // bq, n_kv),
            in_specs=[pl.BlockSpec((grp, bq, hd), lambda g, i, j: (g, i, 0)),
                      pl.BlockSpec((1, bkv, hd), kv_map),
                      pl.BlockSpec((1, bkv, hd), kv_map)],
            out_specs=[pl.BlockSpec((grp, bq, hd), lambda g, i, j: (g, i, 0))],
        )(qn_hbm, kc_hbm, vc_hbm, o_hbm)

    pl.run_scoped(run, pltpu.VMEM((rows, 128), F32), pltpu.VMEM((rows, 128), F32),
                  pltpu.VMEM((rows, hd), F32))


def att_gate_quant(o_hbm, att_hbm, xq_hbm, xs_hbm, *, nh, hd, tm):
    """y = o * sigmoid(gate) -> bf16 -> per-row fp8 over all heads, token-major for o_proj.
    o [nh, T, hd] f32, gate = att[nh:2 nh], xq [T, nh * hd] fp8, xs [T, 1] f32."""
    t = o_hbm.shape[1]

    def body(o_ref, g_ref, xq_ref, xs_ref):
        y = o_ref[...] * jax.nn.sigmoid(g_ref[...].astype(F32))          # [nh, tm, hd]
        y = jnp.swapaxes(y, 0, 1).reshape(tm, nh * hd)
        xq, xs = quantize_rows(y)
        xq_ref[...] = xq
        xs_ref[...] = xs

    pltpu.emit_pipeline(
        body,
        grid=(t // tm,),
        in_specs=[pl.BlockSpec((nh, tm, hd), lambda m: (0, m, 0)),
                  pl.BlockSpec((nh, tm, hd), lambda m: (1, m, 0))],
        out_specs=[pl.BlockSpec((tm, nh * hd), lambda m: (m, 0)),
                   pl.BlockSpec((tm, 1), lambda m: (m, 0))],
    )(o_hbm, att_hbm, xq_hbm, xs_hbm)


# ------------------------------------------------------------------------------------------------
# MoE
# ------------------------------------------------------------------------------------------------


def moe_route(logits_hbm, topi_hbm, topw_hbm, rank_hbm, counts_hbm, *, e, topk, tm):
    """Softmax over the router logits, top-k renormalized, and for every (token, slot) its rank
    among the earlier tokens routed to the same expert, so that grouped rows keep token order.

    logits [T, E] bf16 -> topi [T, K] i32, topw [T, K] f32, rank [T, K] i32 and counts [1, E]
    i32 (tokens per expert). A token's K experts are distinct, so with H the tile's one-hot rows
    summed over slots, rank(t, k) = (strictly-lower-triangular prefix of H)[t, id] plus the
    running counts of earlier tiles."""
    t = logits_hbm.shape[0]

    def run(run_ref, sem):
        run_ref[...] = jnp.zeros(run_ref.shape, F32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (tm, e), 1)
        r_i = jax.lax.broadcasted_iota(jnp.int32, (tm, tm), 0)
        c_i = jax.lax.broadcasted_iota(jnp.int32, (tm, tm), 1)
        strict_lower = (c_i < r_i).astype(BF16)

        def body(lg_ref, ti_ref, tw_ref, rk_ref):
            lg = lg_ref[...].astype(F32)
            p = jnp.exp(lg - jnp.max(lg, -1, keepdims=True))
            p = p / jnp.sum(p, -1, keepdims=True)
            ids, wts = [], []
            for _ in range(topk):
                mx = jnp.max(p, -1, keepdims=True)
                idx = jnp.min(jnp.where(p == mx, lane, e), -1, keepdims=True)  # lowest id on ties
                ids.append(idx)
                wts.append(mx)
                p = jnp.where(lane == idx, -1.0, p)
            hot = jnp.zeros((tm, e), F32)
            for idx in ids:
                hot = hot + (lane == idx).astype(F32)
            prior = jax.lax.dot_general(strict_lower, hot.astype(BF16), (((1,), (0,)), ((), ())),
                                        preferred_element_type=F32) + run_ref[...]
            ranks = [jnp.sum(jnp.where(lane == idx, prior, 0.0), -1, keepdims=True) for idx in ids]
            ti_ref[...] = jnp.concatenate(ids, -1)
            wts = jnp.concatenate(wts, -1)
            tw_ref[...] = wts / jnp.sum(wts, -1, keepdims=True)
            rk_ref[...] = jnp.concatenate(ranks, -1).astype(jnp.int32)
            run_ref[...] = run_ref[...] + jnp.sum(hot, 0, keepdims=True)

        pltpu.emit_pipeline(
            body,
            grid=(t // tm,),
            in_specs=[pl.BlockSpec((tm, e), lambda i: (i, 0))],
            out_specs=[pl.BlockSpec((tm, topk), lambda i: (i, 0))] * 3,
        )(logits_hbm, topi_hbm, topw_hbm, rank_hbm)

        def store(cnt_v):  # f32 counts are exact (< 2^24)
            cnt_v[...] = run_ref[...].astype(jnp.int32)
            cp = pltpu.make_async_copy(cnt_v, counts_hbm, sem.at[0])
            cp.start()
            cp.wait()

        pl.run_scoped(store, pltpu.VMEM((1, e), jnp.int32))

    pl.run_scoped(run, pltpu.VMEM((1, e), F32), pltpu.SemaphoreType.DMA((1,)))


def _gather_issue(src_hbm, idx_fn, dst_vmem, n, sems, inflight, unroll=8):
    """Start dst_vmem[r] = src_hbm[idx_fn(r)] for r < n, one DMA per row; at most `inflight`
    are left outstanding (the oldest are waited as new ones start). Pair with _gather_drain."""
    def one(r):
        @pl.when(r >= inflight)
        def _():
            pltpu.make_async_copy(src_hbm.at[pl.ds(0, 1)], dst_vmem.at[pl.ds(r - inflight, 1)],
                                  sems.at[(r - inflight) % inflight]).wait()
        pltpu.make_async_copy(src_hbm.at[pl.ds(idx_fn(r), 1)], dst_vmem.at[pl.ds(r, 1)],
                              sems.at[r % inflight]).start()

    def group(gi, carry):                    # n is dynamic, so unroll by hand
        for u in range(unroll):
            one(gi * unroll + u)
        return carry

    groups = n // unroll
    jax.lax.fori_loop(0, groups, group, 0)
    jax.lax.fori_loop(groups * unroll, n, lambda r, c: (one(r), c)[1], 0)


def _gather_drain(src_hbm, dst_vmem, n, sems, inflight):
    """Wait for the rows _gather_issue left outstanding."""
    def body(r, c):
        pltpu.make_async_copy(src_hbm.at[pl.ds(0, 1)], dst_vmem.at[pl.ds(r, 1)],
                              sems.at[r % inflight]).wait()
        return c
    jax.lax.fori_loop(jnp.maximum(n - inflight, 0), n, body, 0)


def _gather_rows(src_hbm, idx_fn, dst_vmem, n, sems, inflight):
    """dst_vmem[r] = src_hbm[idx_fn(r)] for r < n (issue, then drain)."""
    _gather_issue(src_hbm, idx_fn, dst_vmem, n, sems, inflight)
    _gather_drain(src_hbm, dst_vmem, n, sems, inflight)


def _expert_tile(x, w13, w13s, w2, w2s, inter):
    """One grouped tile through an expert: per-row fp8, w13, SwiGLU, per-row fp8, w2 (w8a8),
    bf16-rounded like the serving grouped matmul. x [rt, H] f32 -> [rt, H] f32."""
    xq, xs = quantize_rows(x)
    gu = jax.lax.dot_general(xq, w13[...], (((1,), (0,)), ((), ())), preferred_element_type=F32)
    gu = (gu * xs * w13s[...]).astype(BF16).astype(F32)
    hq, hs = quantize_rows(jax.nn.silu(gu[:, :inter]) * gu[:, inter:])
    y = jax.lax.dot_general(hq, w2[...], (((1,), (0,)), ((), ())), preferred_element_type=F32)
    return (y * hs * w2s[...]).astype(BF16).astype(F32)


def _expert_tile_interleaved(x, w13, w13s, w2, w2s, inter, issue_chunks):
    """_expert_tile with the next tile's row-DMA starts (issue_chunks: S callables) interleaved
    between the column chunks of the two matmuls, in one basic block, so the scalar unit issues
    DMAs while the MXU works."""
    n_chunks = len(issue_chunks)
    xq, xs = quantize_rows(x)
    two_i = w13.shape[-1]
    h = w2.shape[-1]
    half = n_chunks // 2
    c1, c2 = two_i // half, h // (n_chunks - half)
    gu = []
    for s in range(half):
        gu.append(jax.lax.dot_general(xq, w13[:, s * c1:(s + 1) * c1], (((1,), (0,)), ((), ())),
                                      preferred_element_type=F32))
        issue_chunks[s]()
    gu = jnp.concatenate(gu, -1)
    gu = (gu * xs * w13s[...]).astype(BF16).astype(F32)
    hq, hs = quantize_rows(jax.nn.silu(gu[:, :inter]) * gu[:, inter:])
    ys = []
    for s in range(n_chunks - half):
        ys.append(jax.lax.dot_general(hq, w2[:, s * c2:(s + 1) * c2], (((1,), (0,)), ((), ())),
                                      preferred_element_type=F32))
        issue_chunks[half + s]()
    y = jnp.concatenate(ys, -1)
    return (y * hs * w2s[...]).astype(BF16).astype(F32)


def moe_experts(xb_hbm, topi_hbm, rank_hbm, counts_hbm, w13_hbm, w13s_hbm, w2_hbm, w2s_hbm,
                y_sorted_hbm, pos_hbm, *, e, topk, inter, rt, inflight=64, tm=256,
                interleave=0):
    """Routed experts. Rows are grouped by expert (expert e's segment starts at off[e], padded to
    a multiple of `rt`), gathered by DMA from the normed activations, quantized per row, run
    through w13 -> SwiGLU -> w2 (w8a8), and written to y_sorted [n_pad, H] f32 (bf16 values) in
    grouped order. Also writes pos [T, K] = off[topi] + rank, the grouped row of every (token,
    slot). Rows are 32-bit: a single-row DMA cannot address one row of a packed bf16 tile.

    The grouped tiles of all experts form one list, software-pipelined: tile g+1's rows are
    gathered while tile g computes, outputs are written asynchronously, and each expert's
    weights are prefetched while the previous expert runs. Experts with no rows cost nothing.
    interleave=S > 0: the next tile's rt row DMAs (pad rows re-read the tile's first token) are
    started in S chunks between the matmul column chunks instead of before the tile, so their
    issue overlaps the MXU work; each row has its own semaphore.

    xb [T, H] f32 (bf16 values), topi / rank [T, K] i32, counts [1, E] i32, w13 [E, H, 2I] fp8,
    w13s [E, 1, 2I] f32, w2 [E, I, H] fp8, w2s [E, 1, H] f32."""
    t, h = xb_hbm.shape
    n_slots = t * topk
    max_tiles = n_slots // rt + e

    def run(cnt_s, off_s, tok_s, stage_s, t_e, t_r0, t_n, t_first, t_next, sems, sem1):
        cp = pltpu.make_async_copy(counts_hbm, cnt_s, sem1.at[0])
        cp.start()
        cp.wait()

        # Offsets (each expert's segment padded to rt rows) and the flat tile list: per tile its
        # expert, first grouped row, live rows, whether it starts its expert, and the next
        # expert with rows (to prefetch, -1 if none).
        def off_body(ei, carry):
            acc, ntile, prev_first = carry
            n = cnt_s[0, ei]
            off_s[0, ei] = acc
            nt = (n + rt - 1) // rt

            @pl.when(nt > 0)
            def _():
                @pl.when(prev_first >= 0)
                def _():
                    t_next[prev_first] = ei

            def tile_body(ti, c):
                g = ntile + ti
                t_e[g] = ei
                t_r0[g] = acc + ti * rt
                t_n[g] = jnp.minimum(rt, n - ti * rt)
                t_first[g] = (ti == 0).astype(jnp.int32)
                t_next[g] = -1
                return c
            jax.lax.fori_loop(0, nt, tile_body, 0)
            return (acc + nt * rt, ntile + nt, jnp.where(nt > 0, ntile, prev_first))

        with jax.named_scope("moe_offsets"):
            _, n_tiles, _ = jax.lax.fori_loop(0, e, off_body,
                                              (jnp.int32(0), jnp.int32(0), jnp.int32(-1)))

        # pos = off[topi] + rank (vector pass), and the inverse map tok_s[pos] = t (scalar).
        def pos_pass(off_v):
            cpo = pltpu.make_async_copy(off_s, off_v, sem1.at[0])  # SMEM -> VMEM
            cpo.start()
            cpo.wait()
            lane = jax.lax.broadcasted_iota(jnp.int32, (tm, e), 1)

            def body(ti_ref, rk_ref, pos_ref):
                ids = ti_ref[...]
                off = off_v[...]
                cols = [jnp.sum(jnp.where(lane == ids[:, kk:kk + 1], off, 0), -1, keepdims=True)
                        for kk in range(topk)]
                pos_ref[...] = jnp.concatenate(cols, -1) + rk_ref[...]

            pltpu.emit_pipeline(
                body, grid=(t // tm,),
                in_specs=[pl.BlockSpec((tm, topk), lambda i: (i, 0))] * 2,
                out_specs=[pl.BlockSpec((tm, topk), lambda i: (i, 0))],
            )(topi_hbm, rank_hbm, pos_hbm)

        with jax.named_scope("moe_pos"):
            pl.run_scoped(pos_pass, pltpu.VMEM((1, e), jnp.int32))

        rows_per_stage = stage_s.shape[0]

        def scatter_stage(si, c):
            cps = pltpu.make_async_copy(pos_hbm.at[pl.ds(si * rows_per_stage, rows_per_stage)],
                                        stage_s, sem1.at[0])
            cps.start()
            cps.wait()
            t0 = si * rows_per_stage

            def tok_body(j, c):
                for kk in range(topk):
                    tok_s[stage_s[j, kk]] = t0 + j
                return c
            return jax.lax.fori_loop(0, rows_per_stage, tok_body, c, unroll=2)
        with jax.named_scope("moe_scatter"):
            jax.lax.fori_loop(0, t // rows_per_stage, scatter_stage, 0)

        def experts(w13_v, w13s_v, w2_v, w2s_v, xin, yout, wsem, ysem, isems):
            wpairs = ((w13_hbm, w13_v, 0), (w13s_hbm, w13s_v, 1), (w2_hbm, w2_v, 2),
                      (w2s_hbm, w2s_v, 3))

            def fetch(ei, slot):
                for src, dst, k in wpairs:
                    pltpu.make_async_copy(src.at[ei], dst.at[slot], wsem.at[slot, k]).start()

            def wait_w(slot):
                for src, dst, k in wpairs:
                    pltpu.make_async_copy(src.at[0], dst.at[slot], wsem.at[slot, k]).wait()

            def gather(g, b):
                r0 = t_r0[g]
                _gather_issue(xb_hbm, lambda r: tok_s[r0 + r], xin.at[b], t_n[g], sems.at[b],
                              inflight)

            def ywrite(g, b):
                return pltpu.make_async_copy(
                    yout.at[b], y_sorted_hbm.at[pl.ds(pl.multiple_of(t_r0[g], rt), rt)],
                    ysem.at[b])

            def row_copy(g, b, r):
                # Rows past the tile's live count re-read its first token (never used).
                src = tok_s[t_r0[g] + jnp.where(r < t_n[g], r, 0)]
                return pltpu.make_async_copy(xb_hbm.at[pl.ds(src, 1)], xin.at[b, pl.ds(r, 1)],
                                             isems.at[b, r])

            def gather_full(g, b, rs):
                for r in rs:
                    row_copy(g, b, r).start()

            def drain_full(g, b):
                for r in range(rt):
                    pltpu.make_async_copy(xb_hbm.at[pl.ds(0, 1)], xin.at[b, pl.ds(r, 1)],
                                          isems.at[b, r]).wait()

            @pl.when(n_tiles > 0)
            def _():
                fetch(t_e[0], 0)
                if interleave:
                    gather_full(0, 0, range(rt))
                else:
                    gather(0, 0)

            def tile(g, wslot):
                b = g % 2
                # The next tile's rows start moving before (or while) this tile computes.
                if not interleave:
                    @pl.when(g + 1 < n_tiles)
                    def _():
                        gather(g + 1, 1 - b)
                with jax.named_scope("moe_gather_wait"):
                    if interleave:
                        drain_full(g, b)
                    else:
                        _gather_drain(xb_hbm, xin.at[b], t_n[g], sems.at[b], inflight)
                first = t_first[g] == 1
                wslot = jnp.where(first & (g > 0), 1 - wslot, wslot)

                @pl.when(first)
                def _():
                    wait_w(wslot)
                    nxt = t_next[g]

                    @pl.when(nxt >= 0)
                    def _():
                        fetch(nxt, 1 - wslot)

                @pl.when(g >= 2)
                def _():
                    ywrite(g - 2, b).wait()
                with jax.named_scope("moe_mm"):
                    if interleave:
                        gn = jnp.minimum(g + 1, n_tiles - 1)    # last tile re-gathers itself
                        per = rt // interleave
                        chunks = [functools.partial(gather_full, gn, 1 - b,
                                                    range(c * per, (c + 1) * per))
                                  for c in range(interleave)]
                        yout[b] = _expert_tile_interleaved(
                            xin[b], w13_v.at[wslot], w13s_v.at[wslot], w2_v.at[wslot],
                            w2s_v.at[wslot], inter, chunks)
                    else:
                        yout[b] = _expert_tile(xin[b], w13_v.at[wslot], w13s_v.at[wslot],
                                               w2_v.at[wslot], w2s_v.at[wslot], inter)
                ywrite(g, b).start()
                return wslot

            jax.lax.fori_loop(0, n_tiles, tile, jnp.int32(0))
            if interleave:                          # the last tile's self re-gather
                @pl.when(n_tiles > 0)
                def _():
                    drain_full(n_tiles - 1, n_tiles % 2)

            def drain_y(g, c):
                ywrite(g, g % 2).wait()
                return c
            jax.lax.fori_loop(jnp.maximum(n_tiles - 2, 0), n_tiles, drain_y, 0)

        pl.run_scoped(experts,
                      pltpu.VMEM((2, h, 2 * inter), FP8), pltpu.VMEM((2, 1, 2 * inter), F32),
                      pltpu.VMEM((2, inter, h), FP8), pltpu.VMEM((2, 1, h), F32),
                      pltpu.VMEM((2, rt, h), F32), pltpu.VMEM((2, rt, h), F32),
                      pltpu.SemaphoreType.DMA((2, 4)), pltpu.SemaphoreType.DMA((2,)),
                      pltpu.SemaphoreType.DMA((2, rt)))

    n_pad = n_slots + e * (rt - 1)
    stage_rows = 64
    pl.run_scoped(run, pltpu.SMEM((1, e), jnp.int32), pltpu.SMEM((1, e), jnp.int32),
                  pltpu.SMEM((n_pad,), jnp.int32), pltpu.SMEM((stage_rows, topk), jnp.int32),
                  pltpu.SMEM((max_tiles,), jnp.int32), pltpu.SMEM((max_tiles,), jnp.int32),
                  pltpu.SMEM((max_tiles,), jnp.int32), pltpu.SMEM((max_tiles,), jnp.int32),
                  pltpu.SMEM((max_tiles,), jnp.int32),
                  pltpu.SemaphoreType.DMA((2, inflight)), pltpu.SemaphoreType.DMA((1,)))


def moe_combine(y_sorted_hbm, pos_hbm, topw_hbm, sd_hbm, sg_hbm, out_hbm, *, topk, tm,
                inflight=64):
    """out[t] = sum_k topw[t, k] * y_sorted[pos[t, k]] + sd[t] * sigmoid(sg[t]) -> bf16.
    y_sorted [n_pad, H] f32, pos / topw [T, K], sd [T, H] (shared expert down, bf16),
    sg [T, 1] f32 (shared expert gate logit), out [T, H] bf16. Slot-major gathers
    (rows[k * tm + t] = y_sorted[pos[t, k]]), pipelined: tile i+1's rows move while tile i
    sums."""
    t, h = out_hbm.shape
    n_rows = tm * topk
    n_t = t // tm

    def run(pos_s, rows, sems, sem1):
        def load_pos(i, b):
            cp = pltpu.make_async_copy(pos_hbm.at[pl.ds(i * tm, tm)], pos_s.at[b], sem1.at[0])
            cp.start()
            cp.wait()

        def issue(i, b):
            load_pos(i, b)
            _gather_issue(y_sorted_hbm, lambda r: pos_s[b, r % tm, r // tm], rows.at[b], n_rows,
                          sems.at[b], inflight)

        issue(0, 0)

        def body(tw_ref, sd_ref, sg_ref, o_ref):
            i = pl.program_id(0)
            b = i % 2

            @pl.when(i + 1 < n_t)
            def _():
                issue(i + 1, 1 - b)
            with jax.named_scope("combine_gather_wait"):
                _gather_drain(y_sorted_hbm, rows.at[b], n_rows, sems.at[b], inflight)
            tw = tw_ref[...]
            routed = jnp.zeros((tm, h), F32)
            for kk in range(topk):
                routed = routed + rows[b, kk * tm:(kk + 1) * tm, :] * tw[:, kk:kk + 1]
            shared = sd_ref[...].astype(F32) * jax.nn.sigmoid(sg_ref[...])
            o_ref[...] = (routed + shared).astype(o_ref.dtype)

        pltpu.emit_pipeline(
            body, grid=(n_t,),
            in_specs=[pl.BlockSpec((tm, topk), lambda i: (i, 0)),
                      pl.BlockSpec((tm, h), lambda i: (i, 0)),
                      pl.BlockSpec((tm, 1), lambda i: (i, 0))],
            out_specs=[pl.BlockSpec((tm, h), lambda i: (i, 0))],
        )(topw_hbm, sd_hbm, sg_hbm, out_hbm)

    pl.run_scoped(run, pltpu.SMEM((2, tm, topk), jnp.int32), pltpu.VMEM((2, n_rows, h), F32),
                  pltpu.SemaphoreType.DMA((2, inflight)), pltpu.SemaphoreType.DMA((1,)))
