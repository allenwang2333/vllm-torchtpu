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

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16

# Explicit scoped-VMEM budget, repo convention (mla/v1, deepseek_v4 and
# fused_moe use the same constant). Never rely on the backend default —
# it varies by Mosaic version.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
    """Grid token block, and the token count padded to a multiple of it.

    ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
    ``tb`` tokens. When supplied, the block halves until it fits within
    ``vmem_limit_bytes`` — degrading to a smaller block instead of a
    compile-time VMEM OOM — with ``SUBLANE`` as the floor.

    Returns (token_block, padded_tokens).
    """
    tb = min(token_block_size, round_up(num_tokens, SUBLANE))
    if vmem_need is not None:
        while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
            tb //= 2
    return tb, round_up(num_tokens, tb)


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Zero-pad the leading (token) axis of each array to ``padded_tokens``."""
    pad = padded_tokens - arrays[0].shape[0]
    if pad == 0:
        return arrays
    return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Inverse of ``pad_to``: drop the rows the padding added."""
    if arrays[0].shape[0] == num_tokens:
        return arrays
    return tuple(a[:num_tokens] for a in arrays)


class SplitFn(NamedTuple):
    """fn as its three bf16 chunks (see ``split_fn3``), split ahead of time.

    The chunks depend only on the weight, so serving splits each fn once at
    load instead of once per call.
    """

    hi: jax.Array
    mid: jax.Array
    lo: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.hi.shape


def split_fn3(fn: jax.Array | SplitFn) -> SplitFn:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

    The chunks are what stays VMEM-resident. Chunks MUST be built with
    reduce_precision, not dtype round-trips: XLA's excess-precision
    simplification folds f32->bf16->f32 into the identity, which silently
    zeroes the mid/lo chunks. An already split fn is returned as is.
    """
    if isinstance(fn, SplitFn):
        assert all(t.dtype == jnp.bfloat16 for t in fn), [t.dtype for t in fn]
        return fn
    assert fn.dtype == jnp.float32, fn.dtype
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return SplitFn(*(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo)))


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    Shared by the Pallas ``pre_kernel`` and the fused seam op.
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


class GateConsts(NamedTuple):
    """The static constants of ``mhc_pre_gates``' post/comb half."""

    rms_eps: float
    hc_sinkhorn_eps: float
    hc_post_mult_value: float
    sinkhorn_repeat: int


def store_gates_block(
    acc: jax.Array,
    sqr: jax.Array,
    sc_ref,
    hb_ref,
    post_ref,
    comb_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    gates: GateConsts,
) -> None:
    """In-kernel twin of ``mhc_pre_gates``' post/comb outputs for one block.

    Runs on the raw (tb, hc_mult3) mixes and (tb, 1) squared sums a kernel
    already holds, so the tiny gate math costs no extra launch. comb is kept
    as hc_mult row slices (tb, hc_mult) — row i is comb[:, i, :] — so the
    row sums are lane reductions and the column sums elementwise adds.
    Stores post_mix (tb, hc_mult) and comb_mix (tb, hc_mult * hc_mult),
    row-major (i, j), the layout the post/fused kernels read.
    """
    m = hc_mult
    eps = gates.hc_sinkhorn_eps
    mixes = acc * jax.lax.rsqrt(sqr / (m * hidden_size) + gates.rms_eps)

    post_logits = mixes[:, m : 2 * m] * sc_ref[0, 1] + hb_ref[:, m : 2 * m]
    post_ref[...] = jax.nn.sigmoid(post_logits) * gates.hc_post_mult_value

    rows = []
    for i in range(m):
        lo = 2 * m + i * m
        logits = mixes[:, lo : lo + m] * sc_ref[0, 2] + hb_ref[:, lo : lo + m]
        # jax.nn.softmax's own formulation.
        e = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
        rows.append(e / jnp.sum(e, axis=-1, keepdims=True) + eps)

    def col_normalize(rows):
        col = rows[0]
        for r in rows[1:]:
            col = col + r
        return [r / (col + eps) for r in rows]

    rows = col_normalize(rows)
    for _ in range(gates.sinkhorn_repeat - 1):
        rows = [r / (jnp.sum(r, axis=-1, keepdims=True) + eps) for r in rows]
        rows = col_normalize(rows)
    for i, r in enumerate(rows):
        comb_ref[:, i * m : (i + 1) * m] = r
