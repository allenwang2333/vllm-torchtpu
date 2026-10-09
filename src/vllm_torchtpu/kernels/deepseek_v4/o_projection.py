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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    CosSinRef,
    _cos_sin_lanes,
    _lane_gather,
    _rotate_gptj,
)
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

SUBLANE = 8


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _cos_sin_body(
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    """Expand a ``(tile_n, rotary_dim)`` block to ``(tile_n, 2 * LANE)``."""
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    out_ref[...] = jnp.concatenate([cos, sin], axis=-1).astype(out_dtype)


def _cos_sin_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM input
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM output
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    rotary_dim: int,
    inverse: bool,
    cos_sin_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
) -> None:
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=(tile_n, 2 * LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    # The per-token row gather; one buffer of lookahead hides its DMAs.
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _cos_sin_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[cos_sin_spec],
        out_specs=[out_spec],
    )
    allocations = (
        cos_sin_alloc,
        pltpu.BufferedRef.output(out_spec, out_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "out_dtype", "name"),
)
def gather_cos_sin(
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    out_dtype: jnp.dtype = jnp.float32,
    name: str = "gather_cos_sin",
) -> jax.Array:
    """Builds ``wo_a_projection``'s ``[T, 2 * LANE]`` cos/sin table.

    Gathers ``cos_sin_cache[positions]`` based on the token positions, expands
    each row to ``[2 * LANE]`` in XLA so the gather CSEs across layers.
    """
    del name
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    half = rotary_dim // 2
    cs = cos_sin_cache[positions]
    c = jnp.repeat(cs[:, :half], 2, axis=-1)
    s = jnp.repeat(cs[:, half:], 2, axis=-1)
    if inverse:
        s = -s
    pad_w = LANE - rotary_dim
    if pad_w > 0:
        ones = jnp.ones((num_tokens, pad_w), dtype=cs.dtype)
        zeros = jnp.zeros((num_tokens, pad_w), dtype=cs.dtype)
        return jnp.concatenate([ones, c, zeros, s], axis=-1).astype(out_dtype)
    return jnp.concatenate([c, s], axis=-1).astype(out_dtype)


def _rope_heads(x: jax.Array, cos_sin: jax.Array, *, head_dim: int) -> jax.Array:
    """apply RoPE to the last rope dim of head dim of ``x``."""
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    tail = x[:, :, lo:].astype(jnp.float32)
    roped = tail * cos[:, None, :] + _rotate_gptj(tail) * sin[:, None, :]
    return jnp.concatenate([x[:, :, :lo], roped.astype(x.dtype)], axis=-1)


def _rope_heads_2d(
    x_2d: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    heads_per_group: int,
    head_dim: int,
    lane: jax.Array,
    sign: jax.Array,
    out_dtype: jnp.dtype,
) -> jax.Array:
    """Apply RoPE directly on 2D ``(sub_t, heads_per_group * head_dim)``."""
    lo = head_dim - LANE
    pieces = []
    for h in range(heads_per_group):
        base_col = h * head_dim
        pieces.append(x_2d[:, base_col : base_col + lo])
        tail = x_2d[:, base_col + lo : base_col + head_dim].astype(jnp.float32)
        rot = _lane_gather(tail, lane ^ 1) * sign
        roped = (tail * cos + rot * sin).astype(out_dtype)
        pieces.append(roped)
    return jnp.concatenate(pieces, axis=1)


def _kernel(
    # inputs
    x_ref: jax.Ref,  # VMEM (tile_t, H, head_dim) bf16
    w_ref: jax.Ref,  # VMEM (D, tile_r) fp8
    scale_ref: jax.Ref,  # VMEM (1, tile_r) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32
    # output
    out_ref: jax.Ref,  # VMEM (tile_t, tile_r) out_dtype
    *,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
    head_dim: int,
    inverse: bool = False,
) -> None:
    rhs = w_ref[...]
    scale = scale_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _prep_lhs(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x_2d = x_ref[rows].reshape(sub_t, SUBLANE * head_dim)
        cs = cos_sin[rows]
        sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
        x = _rope_heads_2d(
            x_2d,
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane=lane,
            sign=sign,
            out_dtype=out_dtype,
        )
        inv = None
        if quantize_activations:
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
                jnp.bfloat16
            )
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
        else:
            lhs = x
        return lhs, inv

    lhs_curr, inv_curr = _prep_lhs(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial = jax.lax.dot_general(
            lhs_curr,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if s + 1 < num_sub_t:
            lhs_next, inv_next = _prep_lhs(s + 1)
        else:
            lhs_next, inv_next = None, None
        if quantize_activations:
            assert inv_curr is not None
            partial = partial * (1.0 / inv_curr.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)
        lhs_curr, inv_curr = lhs_next, inv_next


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "inverse",
        "name",
    ),
)
def wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    cos_sin: jax.Array,  # [T, 2 * LANE] float32
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    inverse: bool = False,
    name: str = "wo_a_projection",
) -> jax.Array:
    """Do DSv4 wo_a projection.

    T: num_tokens
    G: num_groups
    H: num_heads_per_group
    D: H * head_dim
    R: o_lora_dim

    Equivalent to
    x = rope(x)
    return ``einsum("tgd,dgr->tgr", x.view(t, g, d), wo_a.view(d, g, r),
    preferred_element_type=f32) * wo_a_scale.view(g, r).to(bf16)``

    ``quantize_activations`` quantizes each activation block's rows to fp8 in
    the kernel so both MXU operands are fp8.
    """
    assert x.ndim == 3
    assert x.dtype == jnp.bfloat16
    assert wo_a.ndim == 2 and wo_a_scale.ndim == 1

    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    assert out_features == wo_a_scale.shape[0]

    assert reduction % head_dim == 0
    heads_per_group = reduction // head_dim

    # DSv4's heads_per_group is 8, exactly equal to SUBLANE
    # The kernel is based on this assumption, which makes it
    # simpler.
    assert heads_per_group == SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups

    assert cos_sin.ndim == 2 and cos_sin.shape == (num_tokens, 2 * LANE)
    assert head_dim % LANE == 0

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = _largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        # 128 is chosen based on large batch size (t) microbenchmarks
        sub_t = _largest_divisor(tile_t, cap=128)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            inverse=inverse,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            # x: one group's heads, [t, g, 0] over [T, G * H, head_dim].
            pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t, r: (t, g, 0)
            ),
            # wo_a: the whole reduction, column block g * R + r * tile_r.
            pl.BlockSpec((reduction, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # wo_a_scale
            pl.BlockSpec((1, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # cos_sin: one token block
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, tile_r), lambda g, t, r: (t, g * num_r_tiles + r)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(x, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
        "inverse",
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def fused_reverse_rope_wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    head_dim: int,
    inverse: bool = True,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """``The RoPE-fused ``wo_a_projection``."""
    assert x.ndim == 3
    assert x.shape[2] == head_dim

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=False,
        name=f"{name}_gather_cos_sin",
    )
    return wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        tile_t=tile_t,
        tile_r=tile_r,
        sub_t=sub_t,
        quantize_activations=quantize_activations,
        inverse=inverse,
        name=name,
    )


def _wq_b_q_rope_kernel_inquant(
    qr_ref: jax.Ref,  # VMEM (tile_t, K) bf16
    w_ref: jax.Ref,  # VMEM (K, heads_per_group * head_dim) fp8
    w_s_ref: jax.Ref,  # VMEM (1, heads_per_group * head_dim) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32 [cos, sin]
    out_ref: jax.Ref,  # VMEM (tile_t, heads_per_group, head_dim) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    heads_per_group: int,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    rhs_s = w_s_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _prep_and_dot(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        qr_f32 = qr_ref[rows].astype(jnp.float32)
        abs_max = jnp.max(jnp.abs(qr_f32), axis=1, keepdims=True)
        lhs_s = abs_max / FP8_E4M3_MAX
        lhs_s_safe = jnp.where(lhs_s == 0, 1.0, lhs_s)
        qr_fp8 = jnp.clip(qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(
            jnp.float8_e4m3fn
        )
        partial = jax.lax.dot_general(
            qr_fp8,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return partial, lhs_s_safe

    partial_curr, lhs_s_curr = _prep_and_dot(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        if s + 1 < num_sub_t:
            partial_next, lhs_s_next = _prep_and_dot(s + 1)
        else:
            partial_next, lhs_s_next = None, None
        cs = cos_sin[rows]
        cos = cs[:, :LANE]
        sin = cs[:, LANE : 2 * LANE]
        q_bf16_2d = ((partial_curr * lhs_s_curr) * rhs_s).astype(jnp.bfloat16)
        roped_2d = _rope_heads_2d(
            q_bf16_2d,
            cos,
            sin,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
            lane=lane,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        out_ref[rows] = roped_2d.reshape(sub_t, heads_per_group, head_dim)
        partial_curr, lhs_s_curr = partial_next, lhs_s_next


def _wq_b_q_rope_kernel(
    qr_fp8_ref: jax.Ref,  # VMEM (tile_t, K) fp8
    w_ref: jax.Ref,  # VMEM (K, heads_per_group * head_dim) fp8
    w_s_ref: jax.Ref,  # VMEM (1, heads_per_group * head_dim) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32 [cos, sin]
    lhs_s_ref: jax.Ref,  # VMEM (tile_t, LANE) f32 [lhs_s]
    out_ref: jax.Ref,  # VMEM (tile_t, heads_per_group, head_dim) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    heads_per_group: int,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    rhs_s = w_s_ref[...]
    cos_sin = cos_sin_ref[...]
    lhs_s_all = lhs_s_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _dot(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        return jax.lax.dot_general(
            qr_fp8_ref[rows],
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    partial_curr = _dot(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial_next = _dot(s + 1) if s + 1 < num_sub_t else None
        cs = cos_sin[rows]
        cos = cs[:, :LANE]
        sin = cs[:, LANE : 2 * LANE]
        lhs_s = lhs_s_all[rows, :1]
        q_bf16_2d = ((partial_curr * lhs_s) * rhs_s).astype(jnp.bfloat16)
        roped_2d = _rope_heads_2d(
            q_bf16_2d,
            cos,
            sin,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
            lane=lane,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        out_ref[rows] = roped_2d.reshape(sub_t, heads_per_group, head_dim)
        partial_curr = partial_next


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "cap_t", "name"),
)
def fused_wq_b_q_rope_projection(
    qr: jax.Array,  # [T, K] bf16
    positions: jax.Array,  # [T] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wq_b: jax.Array,  # [K, H_q * head_dim] fp8
    wq_b_scale: jax.Array,  # [H_q * head_dim] float32
    *,
    head_dim: int,
    cap_t: int = 512,
    name: str = "wq_b_q_rope_projection",
) -> jax.Array:
    """Fused FP8 activation quant + wq_b projection + forward RoPE."""
    assert qr.ndim == 2 and qr.dtype == jnp.bfloat16
    assert wq_b.ndim == 2
    num_tokens, k_dim = qr.shape
    wq_k, out_features = wq_b.shape
    assert k_dim == wq_k
    assert out_features % head_dim == 0
    num_heads = out_features // head_dim
    heads_per_group = SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    cols_per_g = heads_per_group * head_dim

    pad_t = (SUBLANE - (num_tokens % SUBLANE)) % SUBLANE
    if pad_t > 0:
        qr = jnp.pad(qr, ((0, pad_t), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = qr.shape[0]
    eff_cap_t = 1024 if (num_heads >= 64 and t_pad >= 1024 and cap_t == 512) else cap_t
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=eff_cap_t // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    num_t_tiles = t_pad // tile_t

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=False,
        name=f"{name}_gather_cos_sin",
    )
    w_s_2d = wq_b_scale.reshape(1, out_features).astype(jnp.float32)

    if num_groups <= 2:
        out = pl.pallas_call(
            functools.partial(
                _wq_b_q_rope_kernel_inquant,
                tile_t=tile_t,
                num_sub_t=tile_t // sub_t,
                heads_per_group=heads_per_group,
                head_dim=head_dim,
            ),
            out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
            grid=(num_groups, num_t_tiles),
            in_specs=[
                pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
                pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
            ],
            out_specs=pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            name=name,
        )(qr, wq_b, w_s_2d, cos_sin)
    else:
        qr_f32 = qr.astype(jnp.float32)
        abs_max = jnp.max(jnp.abs(qr_f32), axis=-1, keepdims=True)
        lhs_s = abs_max / FP8_E4M3_MAX
        lhs_s_safe = jnp.where(lhs_s == 0, 1.0, lhs_s)
        qr_fp8 = jnp.clip(qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(
            jnp.float8_e4m3fn
        )
        lhs_s_128 = jnp.broadcast_to(lhs_s_safe, (t_pad, LANE))
        out = pl.pallas_call(
            functools.partial(
                _wq_b_q_rope_kernel,
                tile_t=tile_t,
                num_sub_t=tile_t // sub_t,
                heads_per_group=heads_per_group,
                head_dim=head_dim,
            ),
            out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
            grid=(num_groups, num_t_tiles),
            in_specs=[
                pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
                pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
                pl.BlockSpec((tile_t, LANE), lambda g, t: (t, 0)),
            ],
            out_specs=pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            name=name,
        )(qr_fp8, wq_b, w_s_2d, cos_sin, lhs_s_128)
    return out[:num_tokens] if pad_t > 0 else out


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "cap_t", "name"),
)
def fused_wq_b_q_rope_prequant_projection(
    qr_fp8: jax.Array,  # [T, K] fp8_e4m3fn
    qr_s_128: jax.Array,  # [T, LANE] float32
    positions: jax.Array,  # [T] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wq_b: jax.Array,  # [K, H_q * head_dim] fp8
    wq_b_scale: jax.Array,  # [H_q * head_dim] float32
    *,
    head_dim: int,
    cap_t: int = 512,
    name: str = "wq_b_q_rope_prequant_projection",
) -> jax.Array:
    """Fused wq_b projection + forward RoPE consuming pre-quantized (qr_fp8, qr_s_128)."""
    assert qr_fp8.ndim == 2 and qr_fp8.dtype == jnp.float8_e4m3fn
    assert wq_b.ndim == 2
    num_tokens, k_dim = qr_fp8.shape
    wq_k, out_features = wq_b.shape
    assert k_dim == wq_k
    assert out_features % head_dim == 0
    num_heads = out_features // head_dim
    heads_per_group = SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    cols_per_g = heads_per_group * head_dim

    pad_t = (SUBLANE - (num_tokens % SUBLANE)) % SUBLANE
    if pad_t > 0:
        qr_fp8 = jnp.pad(qr_fp8, ((0, pad_t), (0, 0)))
        qr_s_128 = jnp.pad(qr_s_128, ((0, pad_t), (0, 0)), constant_values=1.0)
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = qr_fp8.shape[0]
    eff_cap_t = 1024 if (num_heads >= 64 and t_pad >= 1024 and cap_t == 512) else cap_t
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=eff_cap_t // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    num_t_tiles = t_pad // tile_t

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=False,
        name=f"{name}_gather_cos_sin",
    )
    w_s_2d = wq_b_scale.reshape(1, out_features).astype(jnp.float32)

    out = pl.pallas_call(
        functools.partial(
            _wq_b_q_rope_kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
        grid=(num_groups, num_t_tiles),
        in_specs=[
            pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
            pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
            pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
            pl.BlockSpec((tile_t, LANE), lambda g, t: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(qr_fp8, wq_b, w_s_2d, cos_sin, qr_s_128)
    return out[:num_tokens] if pad_t > 0 else out
