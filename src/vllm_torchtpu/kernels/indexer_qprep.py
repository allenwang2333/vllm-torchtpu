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
"""``TPU_INDEXER_QPREP_FUSED``: the DSA indexer's query prep as one Pallas pass.

``VllmTPUIndexer.forward`` (rope in half-split order) turns the ``wq_b``
output ``q`` bf16 ``[T, H*D]`` into the top-k kernel's operands with

    q   = rope_half_split_leading(q.view(T, H, D))      # x * C + rot(x) * S
    q8, scale = quantize_tensor(q.view(T*H, D), fp8)    # per (token, head) row

which XLA runs as four passes over a 32 MB (T = 4096) tensor: a relayout of
``[T, H*D]`` into the head-in-sublane ``[T*H, 1, D]`` layout, the rope, an
amax pass and a convert pass. Here every ``(token, head)`` row is done in
VMEM in one read of ``q`` and one write of ``q8``: a ``[tT, H*D]`` block is
cut into its ``H`` lane tiles (a head is exactly one 128-lane tile, so the
cut is free), stacked head-major and transposed to ``[tT, H, D]`` in
registers -- the inverse of what the score kernel's ``transpose_to_head_major``
does to its q block -- then roped, reduced and cast.

Numerics: the step-by-step roundings of the torch path are switches
(``Numerics``) because XLA may keep a bf16 intermediate in f32 inside a
fusion; ``tests/kernels/test_indexer_qprep.py`` picks the combination that
reproduces the compiled torch path bit for bit, and ``DEFAULT_NUMERICS`` is
that combination.
"""

import dataclasses
import functools

import jax
import ml_dtypes
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_TOKENS = 256
# Below this many tokens a block is the whole (padded) array.
MIN_BLOCK_TOKENS = 8


@dataclasses.dataclass(frozen=True)
class Numerics:
    """Where the torch expression rounds to bf16.

    rope_bf16:  round ``x*C`` and ``rot*S`` to bf16 before the add (eager
                torch); else both products are exact in f32 and only the sum
                rounds.
    y_bf16:     the roped ``y`` is rounded to bf16 before the quant (it is
                materialized); else the quant reads the f32 sum.
    scale_mode: "b16r16": scale = bf16(amax * bf16(1/qmax));
                "b16r32": scale = bf16(amax * f32(1/qmax));
                "f32":    scale = f32(amax * f32(1/qmax)).
    inv_bf16:   1/scale rounds to bf16.
    prod_bf16:  y * (1/scale) rounds to bf16 before the clamp and cast.
    """

    rope_bf16: bool = False
    y_bf16: bool = True
    scale_mode: str = "b16r16"
    inv_bf16: bool = True
    prod_bf16: bool = True

    @property
    def tag(self) -> str:
        return (
            f"r{int(self.rope_bf16)}y{int(self.y_bf16)}{self.scale_mode}"
            f"i{int(self.inv_bf16)}p{int(self.prod_bf16)}"
        )


DEFAULT_NUMERICS = Numerics()


def _bf16(x):
    """Round f32 to bf16 and back. In a Mosaic kernel the convert pair is
    honored; plain XLA may drop it (excess precision), so ``reference`` uses
    ``_bf16_xla`` instead."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _bf16_xla(x):
    return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)


def rope_rows(x, rot, c, s, numerics: Numerics, rnd=_bf16):
    """``x * C + rot * S`` in f32 carrying the requested roundings."""
    if numerics.rope_bf16:
        y = rnd(rnd(x * c) + rnd(rot * s))
    else:
        y = x * c + rot * s
    return rnd(y) if numerics.y_bf16 else y


def quant_rows(y, numerics: Numerics, q_max: float, quant_dtype, rnd=_bf16):
    """``quantize_tensor(y, axis=-1)``: per-row amax scale and fp8 cast.

    ``y`` is f32 (holding bf16 values when ``y_bf16``). Returns the fp8 rows
    and the f32 scale (keepdims)."""
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    if numerics.scale_mode == "b16r16":
        recip = float(ml_dtypes.bfloat16(1.0 / q_max))
        scale = rnd(rnd(amax) * jnp.float32(recip))
    elif numerics.scale_mode == "b16r32":
        scale = rnd(rnd(amax) * jnp.float32(1.0 / q_max))
    elif numerics.scale_mode == "f32":
        scale = amax * jnp.float32(1.0 / q_max)
    else:
        raise ValueError(numerics.scale_mode)
    inv = jnp.float32(1.0) / scale
    if numerics.inv_bf16:
        inv = rnd(inv)
    inv = jnp.where(scale == 0, jnp.float32(0.0), inv)
    v = y * inv
    if numerics.prod_bf16:
        v = rnd(v)
    v = jnp.clip(v, -q_max, q_max)
    return v.astype(quant_dtype), scale


def _rot_half(x, rope_dim: int):
    """``[x[h:2h], x[:h], x[2h:]]`` along the 128-lane axis (h = rope_dim/2)
    as two lane rotations and a select: the same values the torch ``cat``
    builds, including the pass-through lanes (multiplied by S = 0 there)."""
    half = rope_dim // 2
    lanes = x.shape[-1]
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    fwd = pltpu.roll(x, lanes - half, x.ndim - 1)  # fwd[i] = x[i + half]
    bwd = pltpu.roll(x, half, x.ndim - 1)  # bwd[i] = x[i - half]
    return jnp.where(lane < half, fwd, jnp.where(lane < rope_dim, bwd, x))


def _kernel(
    x_ref,
    c_ref,
    s_ref,
    q_ref,
    scale_ref,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    numerics: Numerics,
    q_max: float,
):
    """One ``[tT, H*D]`` block -> fp8 ``[tT, H, D]`` + f32 scales ``[tT, H]``."""
    t = x_ref.shape[0]
    # Head h is lane tile h: cut, stack head-major, transpose in registers.
    heads = [
        x_ref[:, h * head_dim : (h + 1) * head_dim].astype(jnp.float32)
        for h in range(n_head)
    ]
    x = jnp.stack(heads, axis=0).transpose(1, 0, 2)  # [tT, H, D]
    c = c_ref[...].astype(jnp.float32).reshape(t, 1, head_dim)
    s = s_ref[...].astype(jnp.float32).reshape(t, 1, head_dim)
    y = rope_rows(x, _rot_half(x, rope_dim), c, s, numerics)
    q, scale = quant_rows(y, numerics, q_max, q_ref.dtype)
    q_ref[...] = q
    scale_ref[...] = scale.reshape(scale_ref.shape)


def _block_tokens(num_tokens: int, block_tokens: int) -> int:
    bt = min(block_tokens, num_tokens)
    while num_tokens % bt:
        bt //= 2
    return bt


@functools.partial(
    jax.jit,
    static_argnames=(
        "n_head",
        "rope_dim",
        "numerics",
        "quant_dtype",
        "block_tokens",
        "interpret",
    ),
)
def indexer_q_prep(
    q: jax.Array,  # bf16 [T, H*D], the wq_b output as it lands
    c: jax.Array,  # bf16 [T, D]: [cos, cos, 1...] per token
    s: jax.Array,  # bf16 [T, D]: [-sin, sin, 0...] per token
    *,
    n_head: int,
    rope_dim: int,
    numerics: Numerics = DEFAULT_NUMERICS,
    quant_dtype=jnp.float8_e4m3fn,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Returns ``(q8 [T, H, D] quant_dtype, scale f32 [T, H])``.

    Token counts below ``MIN_BLOCK_TOKENS`` are padded (statically) to it and
    sliced back, so a decode bucket gets the same kernel."""
    num_tokens, width = q.shape
    head_dim = c.shape[-1]
    assert width == n_head * head_dim, (q.shape, n_head, head_dim)
    assert head_dim % 128 == 0 and rope_dim % 2 == 0 and rope_dim <= head_dim
    pad = (-num_tokens) % MIN_BLOCK_TOKENS if num_tokens < MIN_BLOCK_TOKENS else 0
    if pad:
        q = jnp.pad(q, ((0, pad), (0, 0)))
        c = jnp.pad(c, ((0, pad), (0, 0)))
        s = jnp.pad(s, ((0, pad), (0, 0)))
    rows = num_tokens + pad
    bt = _block_tokens(rows, block_tokens)
    q_max = float(jnp.finfo(quant_dtype).max)
    q8, scale = pl.pallas_call(
        functools.partial(
            _kernel,
            n_head=n_head,
            head_dim=head_dim,
            rope_dim=rope_dim,
            numerics=numerics,
            q_max=q_max,
        ),
        grid=(rows // bt,),
        in_specs=[
            pl.BlockSpec((bt, width), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bt, n_head, head_dim), lambda i: (i, 0, 0)),
            pl.BlockSpec((bt, n_head), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, n_head, head_dim), quant_dtype),
            jax.ShapeDtypeStruct((rows, n_head), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=f"indexer_qprep_h{n_head}_d{head_dim}_r{rope_dim}_b{bt}_{numerics.tag}",
        interpret=interpret,
    )(q, c, s)
    if pad:
        q8, scale = q8[:num_tokens], scale[:num_tokens]
    return q8, scale


def reference(q, c, s, *, n_head: int, rope_dim: int, numerics: Numerics = DEFAULT_NUMERICS,
              quant_dtype=jnp.float8_e4m3fn):
    """The same switches in plain XLA, on the ``[T, H, D]`` view (the torch
    path's layout)."""
    num_tokens = q.shape[0]
    head_dim = c.shape[-1]
    half = rope_dim // 2
    x = q.reshape(num_tokens, n_head, head_dim).astype(jnp.float32)
    rot = jnp.concatenate([x[..., half:rope_dim], x[..., :half], x[..., rope_dim:]], -1)
    y = rope_rows(
        x,
        rot,
        c.astype(jnp.float32)[:, None],
        s.astype(jnp.float32)[:, None],
        numerics,
        rnd=_bf16_xla,
    )
    q8, scale = quant_rows(
        y, numerics, float(jnp.finfo(quant_dtype).max), quant_dtype, rnd=_bf16_xla
    )
    return q8, scale[..., 0]
