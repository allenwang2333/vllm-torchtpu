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


def norm_add_quant(h_hbm, r_hbm, w_hbm, r_out_hbm, x_bf_hbm, xq_hbm, xs_hbm, *, eps, tm):
    """residual' = h + r; x = rmsnorm(residual') * (1 + w) -> bf16 and per-row fp8.

    h, r, r_out, x_bf: [T, H] bf16; w: [1, H] bf16; xq: [T, H] fp8; xs: [T, 1] f32.
    r_out may alias r (each tile is read before it is written)."""
    t, h = h_hbm.shape

    def body(h_ref, r_ref, w_ref, ro_ref, xb_ref, xq_ref, xs_ref):
        res = h_ref[...].astype(F32) + r_ref[...].astype(F32)
        res_b = res.astype(BF16)
        ro_ref[...] = res_b
        res = res_b.astype(F32)
        x = res * jax.lax.rsqrt(jnp.mean(res * res, -1, keepdims=True) + eps)
        x = (x * (1.0 + w_ref[...].astype(F32))).astype(BF16)
        xb_ref[...] = x
        xq, xs = quantize_rows(x)
        xq_ref[...] = xq
        xs_ref[...] = xs

    row = pl.BlockSpec((tm, h), lambda i: (i, 0))
    pltpu.emit_pipeline(
        body,
        grid=(t // tm,),
        in_specs=[row, row, pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[row, row, row, pl.BlockSpec((tm, 1), lambda i: (i, 0))],
    )(h_hbm, r_hbm, w_hbm, r_out_hbm, x_bf_hbm, xq_hbm, xs_hbm)


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
    halo = 8                                  # >= ks - 1, sublane aligned

    def run(prev_ref, sem):
        # prev_ref[halo - (ks - 1):] holds the ks - 1 input rows before the current tile.
        cp = pltpu.make_async_copy(conv_state_hbm, prev_ref.at[pl.ds(halo - (ks - 1), ks - 1)],
                                   sem.at[0])
        cp.start()
        cp.wait()

        def body(x_ref, ba_ref, w_ref, al_ref, dt_ref, q_ref, k_ref, v_ref, gb_ref):
            x = x_ref[...].astype(F32)                                  # [tm, QKV]
            full = jnp.concatenate([prev_ref[...].astype(F32), x], 0)   # [halo + tm, QKV]
            w = w_ref[...].astype(F32)                                  # [ks, QKV]
            acc = jnp.zeros_like(x)
            for j in range(ks):
                acc = acc + full[halo - (ks - 1) + j:halo - (ks - 1) + j + tm] * w[j:j + 1]
            prev_ref[...] = x_ref[tm - halo:tm, :]
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
        # New conv state: the sequence's last ks - 1 conv inputs.
        cp = pltpu.make_async_copy(qkvz_hbm.at[pl.ds(t - (ks - 1), ks - 1), pl.ds(0, qkv)],
                                   conv_out_hbm, sem.at[0])
        cp.start()
        cp.wait()

    pl.run_scoped(run, pltpu.VMEM((halo, qkv), BF16), pltpu.SemaphoreType.DMA((1,)))


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
    rope [S, rot] f32 (cos | sin, rot / 2 each) for absolute positions, ctx: traced scalar."""
    t = att_hbm.shape[1]
    half = rot // 2

    def run(rope_v, k8, v8, sem):
        def body(q_ref, k_ref, v_ref, qw_ref, kw_ref, qo_ref):
            m = pl.program_id(0)
            r0 = ctx + m * tm
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


def att_flash(qn_hbm, kc_hbm, vc_hbm, ctx, o_hbm, *, nh, nkv, hd, bq, bkv, s_max):
    """Causal GQA attention of the T new tokens (positions ctx + t) over the cache rows
    0 .. ctx + T - 1, per kv head with its nh / nkv q heads folded into the rows, f32 online
    softmax. qn [nh, T, hd] bf16, kc / vc [nkv, S, hd] fp8, o [nh, T, hd] f32."""
    t = qn_hbm.shape[1]
    grp = nh // nkv
    rows = grp * bq
    n_kv = pl.cdiv(s_max, bkv)
    scale = hd ** -0.5

    def last_kv(i):                         # last kv tile the q tile i needs
        return (ctx + (i + 1) * bq - 1) // bkv

    def run(m_ref, l_ref, acc_ref):
        def body(q_ref, k_ref, v_ref, o_ref):
            i, j = pl.program_id(1), pl.program_id(2)

            @pl.when(j == 0)
            def _():
                m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, F32)
                l_ref[...] = jnp.zeros(l_ref.shape, F32)
                acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

            @pl.when(j <= last_kv(i))
            def _():
                q = q_ref[...].reshape(rows, hd)
                kk = k_ref[0].astype(BF16)
                vv = v_ref[0].astype(BF16)
                s = jax.lax.dot_general(q, kk, (((1,), (1,)), ((), ())),
                                        preferred_element_type=F32) * scale
                row = jax.lax.broadcasted_iota(jnp.int32, (rows, bkv), 0)
                col = jax.lax.broadcasted_iota(jnp.int32, (rows, bkv), 1)
                pos = ctx + i * bq + row % bq        # rows are (head, token) head-major
                s = jnp.where(j * bkv + col <= pos, s, -jnp.inf)
                m_prev = m_ref[...][:, :1]
                m_new = jnp.maximum(m_prev, jnp.max(s, -1, keepdims=True))
                alpha = jnp.exp(m_prev - m_new)
                p = jnp.exp(s - m_new)
                l_ref[...] = jnp.broadcast_to(alpha * l_ref[...][:, :1] + jnp.sum(p, -1, keepdims=True),
                                              l_ref.shape)
                m_ref[...] = jnp.broadcast_to(m_new, m_ref.shape)
                acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
                    p.astype(BF16), vv, (((1,), (0,)), ((), ())), preferred_element_type=F32)

            @pl.when(j == n_kv - 1)
            def _():
                o_ref[...] = (acc_ref[...] / l_ref[...][:, :1]).reshape(grp, bq, hd)

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
