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
"""Token-major MLA W_UK absorb: ``ql_nope[t, h] = q[t, h, :nope] @ W_UK_T[h]``.

The torch path computes the absorb as ``transpose -> bmm -> transpose``. On a
mixed (prefill) bucket XLA turns the two transposes into full-size relayouts of
q and of ``ql_nope`` (one TensorCore copy and two SparseCore data-format
copies, ~300 us per layer at 4096 tokens), because a batched matmul wants the
head (batch) axis outermost while both the producer (``q_b_proj``) and the
consumer (the sparse MLA kernel, which DMAs ``q[t0:t1]`` token-major) are
token-major.

This kernel reads the q projection output in its native ``[T, H * qk]``
row-major layout and writes ``[T, H, lora]`` in that array's own tiled layout
(heads on sublanes), exactly the buffer the attention kernel DMAs, so no
relayout is needed on either side. (A ``[T, H * lora]`` 2D output would not
do: on TPU the 2D and 3D tiled layouts differ, and XLA relayouts between them
on the SparseCore.) Per head it is the same bf16 x bf16 -> f32 MXU product the bmm
does, rounded once to the output dtype (then scaled, when a per-channel weight
scale exists, the way the torch path scales the bmm output).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HEADS_PER_BLOCK = 16


def pick_blocks(num_tokens: int, num_heads: int) -> tuple[int, int] | None:
    """(token block, heads per block) for the grid, or None if unsupported.

    Heads per block is 16: one bf16 (16, 128) tile of the output's head axis,
    i.e. 8 packed head pairs = one u32 (8, 128) vreg of sublanes."""
    tb = next((b for b in (512, 256, 128, 64, 32, 16) if num_tokens % b == 0), None)
    hb = HEADS_PER_BLOCK if num_heads % HEADS_PER_BLOCK == 0 else None
    if tb is None or hb is None:
        return None
    return tb, hb


def _absorb_kernel(q_ref, w_ref, *rest, hb, nope, qk, lora, has_scale):
    """One (token block, 16-head group): 16 per-head MXU products, written as
    the consumer's [tb, 16, lora] bf16 block.

    A head's product comes out of the MXU token-major ([tb, lora], tokens on
    sublanes) while the output tile wants heads on sublanes (bf16 packs head
    pairs into one 32-bit sublane). So heads 2k / 2k+1 are rounded to bf16 and
    packed into one u32 [tb, lora] word array (even head in the low half, the
    TPU (2, 1) bf16 packing), and the 8 packed pairs of each 128-lane column
    are stacked and swapped (pair, token) -> (token, pair) in vregs, giving
    whole (8, 128) u32 tiles of the output viewed as u32.
    """
    if has_scale:
        s_ref, o_ref = rest
    else:
        (o_ref,) = rest
    o32 = o_ref.bitcast(jnp.uint32)  # [tb, hb // 2, lora]

    def head(j):
        qj = q_ref[:, j * qk : j * qk + nope]
        wj = w_ref[j]
        if wj.dtype != qj.dtype:
            wj = wj.astype(qj.dtype)
        r = jnp.dot(qj, wj, preferred_element_type=jnp.float32)
        r = r.astype(o_ref.dtype).astype(jnp.float32)
        if has_scale:
            # torch: (bmm_out in q dtype) * scale, one more rounding.
            r = (r * s_ref[j].astype(jnp.float32)).astype(o_ref.dtype)
            r = r.astype(jnp.float32)
        return pltpu.bitcast(r, jnp.uint32)  # bf16 bits in the high half

    pairs = []
    for k in range(hb // 2):
        even, odd = head(2 * k), head(2 * k + 1)
        pairs.append((odd & jnp.uint32(0xFFFF0000)) | (even >> 16))
    for c in range(lora // 128):
        cols = jnp.stack([p[:, c * 128 : (c + 1) * 128] for p in pairs], axis=0)
        o32[:, :, c * 128 : (c + 1) * 128] = jnp.swapaxes(cols, 0, 1)


def mla_absorb_token_major(
    q: jax.Array,
    w_uk_t: jax.Array,
    w_scale: jax.Array | None = None,
    *,
    nope_dim: int,
) -> jax.Array:
    """``q [T, H, qk]`` (only ``[..., :nope_dim]`` is read) x ``w_uk_t [H, nope, lora]``
    -> ``[T, H, lora]`` in q's dtype (bf16), laid out as the attention kernel
    reads it. ``w_scale`` (optional) is ``[H, 1, lora]``."""
    num_tokens, num_heads, qk = q.shape
    wh, wn, lora = w_uk_t.shape
    assert wh == num_heads and wn == nope_dim <= qk, (q.shape, w_uk_t.shape)
    assert q.dtype == jnp.bfloat16 and lora % 128 == 0, (q.dtype, lora)
    blocks = pick_blocks(num_tokens, num_heads)
    if blocks is None:
        # Untileable token count (never a served mixed bucket): the reference
        # head-major product, same rounding as the torch bmm path.
        out = jnp.matmul(
            jnp.transpose(q[..., :nope_dim], (1, 0, 2)), w_uk_t.astype(q.dtype)
        )
        if w_scale is not None:
            out = out * w_scale.reshape(num_heads, 1, lora)
        return jnp.transpose(out, (1, 0, 2)).astype(q.dtype)
    tb, hb = blocks
    # q_b's output is [T, H * qk] row-major: read it as such (the 3D view is
    # a different tiled layout, and would cost XLA a relayout).
    q2 = q.reshape(num_tokens, num_heads * qk)
    has_scale = w_scale is not None
    # Heads outer, tokens inner: the weight block is fetched once per head
    # group and stays resident while the token blocks stream.
    in_specs = [
        pl.BlockSpec((tb, hb * qk), lambda g, t: (t, g)),
        pl.BlockSpec((hb, nope_dim, lora), lambda g, t: (g, 0, 0)),
    ]
    operands = [q2, w_uk_t]
    if has_scale:
        in_specs.append(pl.BlockSpec((hb, 1, lora), lambda g, t: (g, 0, 0)))
        operands.append(w_scale.reshape(num_heads, 1, lora))
    return pl.pallas_call(
        functools.partial(
            _absorb_kernel,
            hb=hb,
            nope=nope_dim,
            qk=qk,
            lora=lora,
            has_scale=has_scale,
        ),
        grid=(num_heads // hb, num_tokens // tb),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((tb, hb, lora), lambda g, t: (t, g, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads, lora), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=48 * 1024 * 1024,
        ),
        name="mla_absorb_token_major",
    )(*operands)
