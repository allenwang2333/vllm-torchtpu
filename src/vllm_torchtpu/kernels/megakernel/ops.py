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
"""Shared numeric, normalization, rotary, and routing primitives."""

from __future__ import annotations

import contextlib
import jax
from jax import lax
import jax.numpy as jnp

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
FP8 = jnp.float8_e4m3fn


def rup(n: int, k: int) -> int:
    """Round `n` up to the next multiple of `k`."""
    return (n + k - 1) // k * k


def rnd(x: jax.Array) -> jax.Array:
    """Round float32 tensor through bfloat16 back to float32 (matching HF BF16 intermediates)."""
    return x.astype(BF16).astype(F32)


def bswitch(idx: jax.Array, fns: list):
    """Balanced binary tree of `lax.cond` (O(log N) select depth vs O(N) for `lax.switch`)."""
    if len(fns) == 1:
        return fns[0]()

    def rec(lo: int, hi: int):
        if hi - lo == 1:
            return fns[lo]()
        mid = (lo + hi) // 2
        return lax.cond(idx < mid, lambda: rec(lo, mid), lambda: rec(mid, hi))

    return rec(0, len(fns))


def rms_zc(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Zero-centered RMSNorm (`1.0 + w`) with BF16 rounding matching frozen megakernel."""
    v = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return rnd(v * (1.0 + w))


def rmsnorm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Zero-centered RMSNorm returning BF16."""
    return rms_zc(x, w, eps).astype(BF16)


def rope_partial(x: jax.Array, cos: jax.Array, sin: jax.Array, rot_dim: int) -> jax.Array:
    """Partial rotary positional embedding on the leading `rot_dim` columns."""
    half = rot_dim // 2
    x1, x2, rest = x[:, :half], x[:, half:rot_dim], x[:, rot_dim:]
    return jnp.concatenate(
        (
            rnd(rnd(x1 * cos) - rnd(x2 * sin)),
            rnd(rnd(x2 * cos) + rnd(x1 * sin)),
            rest,
        ),
        axis=-1,
    )


def silu_mul(gate: jax.Array, up: jax.Array) -> jax.Array:
    """SwiGLU activation `rnd(silu(gate)) * up` rounded to BF16."""
    return (rnd(jax.nn.silu(gate)) * up).astype(BF16)


def po2c_pass1(logits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Router softmax + top-1 selection (enables mid-router expert-0 prefetch)."""
    p = jax.nn.softmax(logits, axis=-1)
    k0 = jnp.argmax(p, axis=-1)
    w0 = jnp.max(p, axis=-1)
    return p, k0, w0


def po2c_pass2(
    p: jax.Array,
    k0: jax.Array,
    w0: jax.Array,
    num_experts: int,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Remaining top-k selections (k=1..topk-1) + normalized BF16-rounded weights."""
    ey = lax.broadcasted_iota(I32, (1, num_experts), 1)
    ks, ws = [k0], [w0]
    for _ in range(topk - 1):
        p = jnp.where(ey == ks[-1][:, None], -1.0, p)
        ks.append(jnp.argmax(p, axis=-1))
        ws.append(jnp.max(p, axis=-1))
    ks_arr = jnp.stack(ks, axis=-1)
    ws_arr = jnp.stack(ws, axis=-1)
    ws_arr = rnd(ws_arr / (jnp.sum(ws_arr, axis=-1, keepdims=True) + 1e-20))
    return ks_arr, ws_arr


def scope_fn(enabled: bool):
    """Return a context manager factory for ` jax.named_scope` when `enabled=True`."""
    return jax.named_scope if enabled else lambda _: contextlib.nullcontext()

