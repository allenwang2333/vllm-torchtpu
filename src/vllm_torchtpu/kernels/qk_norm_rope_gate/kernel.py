# Copyright 2025 Google LLC
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
"""Fused split + per-head RMS norm + partial NeoX RoPE + gate for the
attention layers, as one Pallas call per layer for a few rows.

The served path runs this as about forty small XLA ops per attention layer
(the q|gate split, the per-head views, the two Gemma-style norms, the rope's
cos/sin gather, chunks, rotations and casts). Here one kernel takes

  q_gate  [rows, H * 2 * D] bf16   per head: [q | gate]
  k       [rows, Hk * D]    bf16
  q_w, k_w [D] f32                 the norm weights, already + 1
  cos_sin [rows, R] f32           the rope cache rows for these positions: [cos (R/2) | sin (R/2)]

and returns q [rows, H*D], k [rows, Hk*D] (normed, rotated on the first R
dims, bf16) and gate [rows, H*D] (bf16, pre-sigmoid), matching vLLM's eager
path: norm in f32 (x * rsqrt(mean(x^2) + eps) * w), rope in f32 on the
normed values, NeoX rotation (rotate halves of the first R dims).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_TILE = 8  # bf16 rows per sublane tile (2 per word)


def _norm(x, w, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * w


def _rope(x, cos, sin, r):
    """x [rows, D] f32; rotate the first r dims NeoX-style with cos/sin [rows, r/2]."""
    half = r // 2
    x1, x2 = x[:, :half], x[:, half:r]
    parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    if r < x.shape[1]:  # the pass-through dims, when the rope is partial
        parts.append(x[:, r:])
    return jnp.concatenate(parts, axis=1)


def _kernel(
    qg_ref,
    k_ref,
    qw_ref,
    kw_ref,
    cs_ref,
    q_out,
    k_out,
    g_out,
    *,
    heads,
    kv_heads,
    dim,
    rot,
    eps,
):
    half = rot // 2
    cs = cs_ref[...]
    cos = cs[:, :half]
    sin = cs[:, half:rot]
    qw = qw_ref[...]
    kw = kw_ref[...]
    for h in range(heads):
        q = qg_ref[:, h * 2 * dim : h * 2 * dim + dim].astype(jnp.float32)
        g = qg_ref[:, h * 2 * dim + dim : (h + 1) * 2 * dim]
        q = _rope(_norm(q, qw, eps), cos, sin, rot)
        q_out[:, h * dim : (h + 1) * dim] = q.astype(q_out.dtype)
        g_out[:, h * dim : (h + 1) * dim] = g.astype(g_out.dtype)
    for h in range(kv_heads):
        k = k_ref[:, h * dim : (h + 1) * dim].astype(jnp.float32)
        k = _rope(_norm(k, kw, eps), cos, sin, rot)
        k_out[:, h * dim : (h + 1) * dim] = k.astype(k_out.dtype)


@functools.partial(
    jax.jit, static_argnames=("heads", "kv_heads", "dim", "rot", "eps", "interpret")
)
def qk_norm_rope_gate(
    q_gate,
    k,
    q_w,
    k_w,
    cos_sin_cache,
    positions,
    *,
    heads,
    kv_heads,
    dim,
    rot,
    eps,
    interpret=False,
):
    """The cache entry: gathers the rope rows by position, then the rows entry."""
    cs = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    return qk_norm_rope_gate_rows(
        q_gate,
        k,
        q_w,
        k_w,
        cs,
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        rot=rot,
        eps=eps,
        interpret=interpret,
    )


@functools.partial(
    jax.jit, static_argnames=("heads", "kv_heads", "dim", "rot", "eps", "interpret")
)
def qk_norm_rope_gate_rows(
    q_gate, k, q_w, k_w, cos_sin, *, heads, kv_heads, dim, rot, eps, interpret=False
):
    """q_gate [tokens, heads*2*dim] bf16, k [tokens, kv_heads*dim] bf16, the norm weights [dim] (already + 1),
    cos_sin_cache [max_pos, rot] (any float dtype), positions [tokens] int32. The rope rows are gathered by
    position outside the kernel (one small op); the rows go through the kernel whole, no padding, no un-padding.
    Returns q, k (normed, rotated) and the raw gate, in q_gate's dtype."""
    tokens = q_gate.shape[0]
    cs = cos_sin.astype(jnp.float32)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    kern = functools.partial(
        _kernel, heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=eps
    )
    q_out, k_out, g_out = pl.pallas_call(
        kern,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, heads * dim), q_gate.dtype),
            jax.ShapeDtypeStruct((tokens, kv_heads * dim), k.dtype),
            jax.ShapeDtypeStruct((tokens, heads * dim), q_gate.dtype),
        ),
        in_specs=[vmem] * 5,
        out_specs=(vmem, vmem, vmem),
        name="qk_norm_rope_gate",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(
        q_gate,
        k,
        q_w.reshape(1, dim).astype(jnp.float32),
        k_w.reshape(1, dim).astype(jnp.float32),
        cs,
    )
    return q_out, k_out, g_out


def reference(
    q_gate, k, q_w, k_w, cos_sin_cache, positions, *, heads, kv_heads, dim, rot, eps
):
    """vLLM's eager path in plain JAX: the cos/sin gather, split, Gemma norm per head in f32, NeoX rope on the first rot dims."""
    cos_sin = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    t = q_gate.shape[0]
    qg = q_gate.reshape(t, heads, 2 * dim)
    q, g = qg[:, :, :dim], qg[:, :, dim:]
    kk = k.reshape(t, kv_heads, dim)

    def norm(x, w):
        x = x.astype(jnp.float32)
        return (
            x
            * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
            * w.astype(jnp.float32)
        )

    q = norm(q, q_w)
    kk = norm(kk, k_w)
    half = rot // 2
    cos = cos_sin[:, :half].astype(jnp.float32)[:, None, :]
    sin = cos_sin[:, half:rot].astype(jnp.float32)[:, None, :]

    def rope(x):
        x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
        return jnp.concatenate(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1
        )

    q = rope(q).astype(q_gate.dtype).reshape(t, heads * dim)
    kk = rope(kk).astype(k.dtype).reshape(t, kv_heads * dim)
    return q, kk, g.reshape(t, heads * dim)
