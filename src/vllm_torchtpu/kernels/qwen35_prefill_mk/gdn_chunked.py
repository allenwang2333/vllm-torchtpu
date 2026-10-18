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
"""Chunk-parallel Gated DeltaNet prefill for one sequence (Qwen3.5 at a TP1 pipeline stage).

Per v-head, with the state S as [d_k, d_v] (the convention of gdn_attention_ref):

    u_t = beta_t (v_t - alpha_t S_{t-1}^T k_t)
    o_t = alpha_t S_{t-1}^T q_t + (q_t . k_t) u_t
    S_t = alpha_t S_{t-1} + k_t u_t^T

Within a chunk of C tokens, with gamma the inclusive cumulative log-decay from the chunk start
and Gamma = exp(gamma):

    (I + A) U = diag(beta) V - diag(beta Gamma) K S0,  A_ts = beta_t e^(g_t - g_s) k_t.k_s, s < t
    O   = diag(Gamma) Q S0 + M U,                      M_ts = e^(g_t - g_s) q_t.k_s,       s <= t
    S_C = Gamma_C S0 + (K * e^(g_C - g))^T U

so with T = (I + A)^-1, Ub = T diag(beta) V and W = T diag(beta Gamma) K, everything but the
S0 terms is independent across chunks (phase 1, batched over every chunk and head). Phase 2 walks
the chunks in order with S resident in VMEM: U = Ub - W S, O = Qg S + M U, S = Gamma_C S + Kd^T U.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


def _l2norm(x, eps=1e-6):
    x = x.astype(F32)
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def prepare(qkv, b, a, conv_state, conv_weight, conv_bias, a_log, dt_bias, *, n_kq, n_v, d_k,
            d_v, kernel_size):
    """Causal depthwise conv1d (with the carried conv state), SiLU, split, L2-norm, gates.

    Returns q, k [T, n_v, d_k] (q scaled by d_k^-0.5), v [T, n_v, d_v], g (log-decay) and beta
    [T, n_v], all f32, and the new conv state.
    """
    t = qkv.shape[0]
    x = jnp.concatenate([conv_state.astype(qkv.dtype), qkv], axis=0)
    acc = jnp.zeros(qkv.shape, F32)
    for j in range(kernel_size):
        acc = acc + x[j:j + t].astype(F32) * conv_weight[:, 0, j].astype(F32)[None, :]
    if conv_bias is not None:
        acc = acc + conv_bias.astype(F32)[None, :]
    y = jax.nn.silu(acc.astype(qkv.dtype).astype(F32))
    new_conv_state = x[-(kernel_size - 1):]
    kd = n_kq * d_k
    rep = n_v // n_kq
    q = jnp.repeat(_l2norm(y[:, :kd].reshape(t, n_kq, d_k)), rep, axis=1) * (d_k ** -0.5)
    k = jnp.repeat(_l2norm(y[:, kd:2 * kd].reshape(t, n_kq, d_k)), rep, axis=1)
    v = y[:, 2 * kd:].reshape(t, n_v, d_v)
    beta = jax.nn.sigmoid(b.astype(F32))
    g = -jnp.exp(a_log.astype(F32))[None, :] * jax.nn.softplus(
        a.astype(F32) + dt_bias.astype(F32)[None, :])
    return q, k, v, g, beta, new_conv_state


def _tri_solve(a_strict, rhs, chunk):
    """(I + A)^-1 rhs for strictly lower-triangular A [..., C, C] by repeated squaring:
    (I + A)^-1 = (I - A)(I + A^2)(I + A^4)... since A^C = 0."""
    y = rhs - jnp.einsum("...ts,...sd->...td", a_strict, rhs, preferred_element_type=F32)
    p = a_strict
    n = 2
    while n < chunk:
        p = jnp.einsum("...ts,...sr->...tr", p, p, preferred_element_type=F32)
        y = y + jnp.einsum("...ts,...sd->...td", p, y, preferred_element_type=F32)
        n *= 2
    return y


def phase1(q, k, v, g, beta, chunk):
    """Per-chunk factors, batched over chunks and heads. Inputs [T, H, D] f32, T % chunk == 0.
    Returns Ub [N,H,C,dv], W, Qg, Kd [N,H,C,dk], M [N,H,C,C], decay_c [N,H]."""
    t, h, dk = k.shape
    n = t // chunk
    r = lambda x: x.reshape(n, chunk, h, -1).transpose(0, 2, 1, 3)  # [N, H, C, D]
    q, k, v = r(q), r(k), r(v)
    g = g.reshape(n, chunk, h).transpose(0, 2, 1)                     # [N, H, C]
    beta = beta.reshape(n, chunk, h).transpose(0, 2, 1)
    gam = jnp.cumsum(g, axis=-1)
    diff = gam[..., :, None] - gam[..., None, :]                      # g_t - g_s
    ti = jnp.arange(chunk)
    strict = ti[:, None] > ti[None, :]
    lower = ti[:, None] >= ti[None, :]
    dec = jnp.exp(jnp.where(lower, diff, -jnp.inf))
    kk = jnp.einsum("nhtd,nhsd->nhts", k, k, preferred_element_type=F32)
    a_strict = jnp.where(strict, beta[..., :, None] * dec * kk, 0.0)
    big_gam = jnp.exp(gam)
    rhs = jnp.concatenate([beta[..., None] * v, (beta * big_gam)[..., None] * k], axis=-1)
    sol = _tri_solve(a_strict, rhs, chunk)
    ub, w = sol[..., :v.shape[-1]], sol[..., v.shape[-1]:]
    m = jnp.where(lower, dec * jnp.einsum("nhtd,nhsd->nhts", q, k, preferred_element_type=F32), 0.0)
    qg = big_gam[..., None] * q
    kd = jnp.exp(gam[..., -1:] - gam)[..., None] * k
    return ub, w, qg, kd, m, big_gam[..., -1]


def _scan_kernel(decay_ref, ub_ref, w_ref, qg_ref, kd_ref, m_ref, s0_ref, o_ref, s_out_ref, s_ref):
    c = pl.program_id(0)

    @pl.when(c == 0)
    def _():
        s_ref[...] = s0_ref[...].astype(F32)

    s = s_ref[...]                                                    # [H, dk, dv]
    dot = functools.partial(jax.lax.dot_general, preferred_element_type=F32)
    bmm = (((2,), (1,)), ((0,), (0,)))                                # [H,a,b] x [H,b,c]
    u = ub_ref[0].astype(F32) - dot(w_ref[0].astype(F32), s, bmm)    # [H, C, dv]
    o = dot(qg_ref[0].astype(F32), s, bmm) + dot(m_ref[0].astype(F32), u, bmm)
    o_ref[0] = o.astype(o_ref.dtype)
    kdt = (((1,), (1,)), ((0,), (0,)))                                # Kd^T U: [H,C,dk],[H,C,dv]
    s = decay_ref[0][:, :, None] * s + dot(kd_ref[0].astype(F32), u, kdt)
    s_ref[...] = s

    @pl.when(c == pl.num_programs(0) - 1)
    def _():
        s_out_ref[...] = s.astype(s_out_ref.dtype)


def phase2_pallas(ub, w, qg, kd, m, decay, s0, out_dtype, interpret=False):
    """Sequential state pass over the N chunks, S [H, dk, dv] f32 resident in VMEM."""
    n, h, c, dv = ub.shape
    dk = w.shape[-1]
    blk = lambda d: pl.BlockSpec((1, h, c, d), lambda i: (i, 0, 0, 0))
    full = pl.BlockSpec((h, dk, dv), lambda i: (0, 0, 0))
    o, s_out = pl.pallas_call(
        _scan_kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1, h, 1), lambda i: (i, 0, 0)), blk(dv), blk(dk), blk(dk), blk(dk),
                  blk(c), full],
        out_specs=[blk(dv), full],
        out_shape=[jax.ShapeDtypeStruct((n, h, c, dv), out_dtype),
                   jax.ShapeDtypeStruct((h, dk, dv), F32)],
        scratch_shapes=[pltpu.VMEM((h, dk, dv), F32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",),
                                             vmem_limit_bytes=60 << 20),
        interpret=interpret,
    )(decay[..., None], ub, w, qg, kd, m, s0)
    return o, s_out


def phase2_scan(ub, w, qg, kd, m, decay, s0, out_dtype):
    """lax.scan version of phase 2, for testing."""
    def step(s, xs):
        ub_c, w_c, qg_c, kd_c, m_c, d_c = xs
        u = ub_c - jnp.einsum("hcd,hde->hce", w_c, s)
        o = jnp.einsum("hcd,hde->hce", qg_c, s) + jnp.einsum("hcs,hse->hce", m_c, u)
        s = d_c[:, None, None] * s + jnp.einsum("hcd,hce->hde", kd_c, u)
        return s, o.astype(out_dtype)
    s, o = jax.lax.scan(step, s0.astype(F32), (ub, w, qg, kd, m, decay))
    return o, s


def gdn_prefill(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias, a_log, dt_bias,
                *, n_kq, n_v, d_k, d_v, kernel_size, chunk=64, factor_dtype=F32,
                use_pallas=True, interpret=False):
    """One sequence of T new tokens (T % chunk == 0) after `conv_state` [K-1, dim] and
    `recurrent_state` [n_v, d_k, d_v]. Returns (new_conv_state, new_recurrent_state), output
    [T, n_v * d_v] in qkv's dtype."""
    t = qkv.shape[0]
    q, k, v, g, beta, new_conv = prepare(qkv, b, a, conv_state, conv_weight, conv_bias, a_log,
                                         dt_bias, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                                         kernel_size=kernel_size)
    ub, w, qg, kd, m, decay = phase1(q, k, v, g, beta, chunk)
    ub, w, qg, kd, m = (x.astype(factor_dtype) for x in (ub, w, qg, kd, m))
    if use_pallas:
        o, s = phase2_pallas(ub, w, qg, kd, m, decay, recurrent_state, qkv.dtype, interpret)
    else:
        o, s = phase2_scan(ub, w, qg, kd, m, decay, recurrent_state, qkv.dtype)
    out = o.transpose(0, 2, 1, 3).reshape(t, n_v * d_v)
    return (new_conv, s.astype(recurrent_state.dtype)), out
