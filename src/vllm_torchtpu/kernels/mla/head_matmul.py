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
"""Per-head matmul in token-major layout: `out[t, h] = x[t, h, :p] @ w[h]`.

The MLA absorbs (q_nope @ W_UK, partial @ W_UV) are written as `torch.bmm`,
whose batch dim leads: `[H, T, D] x [H, D, L] -> [H, T, L]`. Their producer and
consumer are token-major `[T, H, *]`, so XLA materialises a relayout on each
side of the bmm (a transpose fusion in, a `copy` out), each an HBM round trip
of the activation. This kernel reads the `[T, H, D]` block as stored, swaps
the token and head axes in VMEM, runs one MXU dot per head and swaps back
before the store, so neither relayout reaches HBM.

Numerics match the bmm path: bf16 operands (an fp8 `w` is converted to the
activation dtype in VMEM, exactly as the `.to()` before `torch.bmm`), f32
accumulation, then the same roundings (`round_before_scale`: round to the
activation dtype, then scale, as `bmm(...) * scale`; otherwise scale in f32).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Heads per grid step. 16 fills a bf16 sublane tile; smaller head counts take
# the whole head axis as one block.
HEADS_PER_BLOCK = 16
# Largest token tile tried; the largest divisor of T in this ladder is used.
_TOKEN_TILES = (256, 128, 64, 32, 16, 8)
_VMEM_LIMIT_BYTES = 56 * 1024 * 1024


def _token_tile(num_tokens: int) -> int | None:
    for t in _TOKEN_TILES:
        if num_tokens % t == 0:
            return t
    return None


def supported(num_tokens: int, num_heads: int) -> bool:
    """Whether the Pallas kernel takes this shape (else the bmm reference)."""
    g = min(HEADS_PER_BLOCK, num_heads)
    return _token_tile(num_tokens) is not None and num_heads % g == 0


def _finish(acc, scale, *, act_dtype, round_before_scale, out_dtype):
    """The bmm path's epilogue on an f32 accumulator, broadcasting `scale`."""
    if scale is None:
        y = acc.astype(act_dtype) if round_before_scale else acc
    elif round_before_scale:
        y = (acc.astype(act_dtype).astype(jnp.float32) * scale.astype(jnp.float32))
        y = y.astype(act_dtype)
    else:
        y = acc * scale.astype(jnp.float32)
    return y.astype(out_dtype)


def _kernel(x_ref, w_ref, *rest, p, has_scale, round_before_scale, out_dtype):
    s_ref = rest[0] if has_scale else None
    o_ref = rest[-1]
    x = x_ref[...]  # (tT, G, Dq), token-major as stored
    if x.shape[-1] != p:
        x = x[..., :p]
    xt = jnp.swapaxes(x, 0, 1)  # (G, tT, p), in VMEM
    w = w_ref[...]
    if w.dtype != x.dtype:
        w = w.astype(x.dtype)
    acc = jnp.einsum("htp,hpl->htl", xt, w, preferred_element_type=jnp.float32)
    scale = None if s_ref is None else s_ref[...][:, None, :]
    o = _finish(
        acc,
        scale,
        act_dtype=x.dtype,
        round_before_scale=round_before_scale,
        out_dtype=out_dtype,
    )
    o_ref[...] = jnp.swapaxes(o, 0, 1)  # (tT, G, L)


def reference(
    x: jax.Array,
    w: jax.Array,
    scale: jax.Array | None = None,
    *,
    round_before_scale: bool = True,
    out_dtype=None,
) -> jax.Array:
    """The bmm formulation (head-major, relayout on both sides)."""
    out_dtype = out_dtype or x.dtype
    p = w.shape[1]
    xt = jnp.swapaxes(x[..., :p], 0, 1)
    acc = jax.lax.dot_general(
        xt,
        w.astype(x.dtype),
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    s = None if scale is None else scale.reshape(w.shape[0], 1, w.shape[2])
    o = _finish(
        acc, s, act_dtype=x.dtype, round_before_scale=round_before_scale,
        out_dtype=out_dtype,
    )
    return jnp.swapaxes(o, 0, 1)


@functools.partial(jax.jit, static_argnames=("round_before_scale", "out_dtype"))
def head_matmul(
    x: jax.Array,
    w: jax.Array,
    scale: jax.Array | None = None,
    *,
    round_before_scale: bool = True,
    out_dtype=None,
) -> jax.Array:
    """`out[t, h, :] = x[t, h, :p] @ w[h]` with `p = w.shape[1]`.

    Args:
      x: `[T, H, Dq]` activations, `Dq >= p`; columns past `p` are ignored
        (so a `[T, H, nope + rope]` q can be passed whole).
      w: `[H, p, L]` weight, activation dtype or fp8.
      scale: optional per-(head, column) scale, `[H, 1, L]` or `[H, L]`.
      round_before_scale: round the f32 product to `x.dtype` before scaling
        (the `bmm(...) * scale` order); else scale in f32.
      out_dtype: output dtype, default `x.dtype`.

    Returns:
      `[T, H, L]` in `out_dtype`, token-major.
    """
    out_dtype = jnp.dtype(out_dtype or x.dtype)
    num_tokens, num_heads, dq = x.shape
    h_w, p, n_out = w.shape
    assert h_w == num_heads and dq >= p, (x.shape, w.shape)
    if not supported(num_tokens, num_heads):
        return reference(
            x, w, scale, round_before_scale=round_before_scale, out_dtype=out_dtype
        )
    g = min(HEADS_PER_BLOCK, num_heads)
    t = _token_tile(num_tokens)
    in_specs = [
        pl.BlockSpec((t, g, dq), lambda hg, i: (i, hg, 0)),
        pl.BlockSpec((g, p, n_out), lambda hg, i: (hg, 0, 0)),
    ]
    args = [x, w]
    if scale is not None:
        in_specs.append(pl.BlockSpec((g, n_out), lambda hg, i: (hg, 0)))
        args.append(scale.reshape(num_heads, n_out))
    kernel = functools.partial(
        _kernel,
        p=p,
        has_scale=scale is not None,
        round_before_scale=round_before_scale,
        out_dtype=out_dtype,
    )
    return pl.pallas_call(
        kernel,
        # Head groups outer: each weight block is fetched once.
        grid=(num_heads // g, num_tokens // t),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((t, g, n_out), lambda hg, i: (i, hg, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads, n_out), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        name="mla_head_matmul_token_major",
    )(*args)
