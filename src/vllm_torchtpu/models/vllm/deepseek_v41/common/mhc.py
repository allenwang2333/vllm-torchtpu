"""Single-pass delayed mHC with compact TPU Sinkhorn and upstream residual flow.

Shared by the v1 and v2 DeepSeek V4.1 implementations.
"""

import functools
import math

import jax
import jax.numpy as jnp
import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.normalization import rms_norm


def _pre_gates(
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

    V4.1 keeps Sinkhorn in a compact device loop.
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


_pre_op = None


def _pre_mix_and_gates_jax(
    x: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    mult: int,
    dim: int,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mix projection and Sinkhorn gating on ``x`` without 3D residual collapse."""
    x_f32 = x.astype(jnp.float32)
    if fn.dtype == jnp.bfloat16:
        # Load-time packed weight (see `pack_mix_weight`).
        mixes = _packed_mixes(x, fn, mult * (mult + 2))
    else:
        mixes = jnp.matmul(x_f32, fn.T, precision=jax.lax.Precision.HIGHEST)
    # Upstream's first-layer broadcast optimization folds x/fn to hidden_size.
    # Gate normalization must still use the actual projection input width.
    squares = jnp.mean(jnp.square(x_f32), -1, keepdims=True) * (mult * dim)
    new_pre, post, comb = _pre_gates(
        mixes,
        squares,
        mult,
        dim,
        scale,
        base,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    return post[..., None], comb, new_pre


def delayed_pre_jax(
    residual: jax.Array,
    x: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mult, dim = residual.shape[-2:]
    post, comb, new_pre = _pre_mix_and_gates_jax(
        x,
        fn,
        scale,
        base,
        mult,
        dim,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    # Collapse the flat [T, M * D] rows (as `_delayed_pre_flat_small` does):
    # the [T, M, D] FP32 reduce over axis 1 costs a T(4,128) relayout copy.
    value = _collapse_flat(residual.reshape(residual.shape[0], -1), previous_pre)
    return post, comb, value.astype(residual.dtype), new_pre


def _truncate_bf16(values: jax.Array) -> jax.Array:
    # A mask, not an f32->bf16->f32 round trip: XLA's excess-precision
    # rewrite may drop the round trip and zero the residual parts.
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _pack_mix_weight(fn: jax.Array) -> jax.Array:
    """FP32 ``fn [n, K]`` as BF16 ``[K, 128]`` columns ``hi | mid | lo | 0``.

    ``hi + mid + lo == fn`` exactly, so one BF16 matmul against BF16 inputs
    matches the FP32 projection up to accumulation order.
    """
    high = _truncate_bf16(fn)
    middle = _truncate_bf16(fn - high)
    low = fn - high - middle
    parts = jnp.concatenate([high, middle, low]).astype(jnp.bfloat16)
    return jnp.pad(parts.T, ((0, 0), (0, 128 - parts.shape[0])))


def pack_mix_weight(fn: torch.Tensor) -> torch.Tensor:
    """Torch twin of `_pack_mix_weight`, applied once after weight loading."""
    weight = fn.detach().cpu().float()

    def truncate(values):
        return (values.view(torch.int32) & -65536).view(torch.float32)

    high = truncate(weight)
    middle = truncate(weight - high)
    low = weight - high - middle
    parts = torch.cat([high, middle, low]).to(torch.bfloat16)
    packed = torch.nn.functional.pad(parts.T, (0, 128 - parts.shape[0]))
    return packed.contiguous().to(fn.device)


def pack_mix_weights(layers) -> None:
    """Pack each decoder layer's FP32 mix weights after weight loading.

    Call after upstream ``finalize_mhc_broadcast_weights``: its first-layer
    broadcast weight reads the FP32 ``hc_attn_fn``. Packed weights are BF16,
    so a second call is a no-op.
    """
    for layer in layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name)
            if fn.device.type == "tpu" and fn.dtype == torch.float32:
                packed = torch.nn.Parameter(pack_mix_weight(fn), requires_grad=False)
                setattr(layer, name, packed)


def _packed_mixes(x: jax.Array, weight: jax.Array, mixes: int) -> jax.Array:
    """Mix logits from a packed weight, summed in `_flat_pre_sums` order."""
    parts = jnp.dot(x, weight, preferred_element_type=jnp.float32)
    return (
        parts[:, :mixes] + parts[:, mixes : 2 * mixes] + parts[:, 2 * mixes : 3 * mixes]
    )


def _flat_pre_sums(flat, weight, previous_pre, mixes, tile):
    """One pass over ``flat [T, M*D]``: mix logits, square sums, collapse."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(x_ref, w_ref, pre_ref, mix_ref, square_ref, value_ref):
        pre, x = pre_ref[...], x_ref[:, :dim].astype(jnp.float32)
        val_acc, sq_acc = pre[:, :1] * x, x * x
        for m in range(1, mult):
            x = x_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            val_acc = val_acc + pre[:, m : m + 1] * x
            sq_acc = sq_acc + x * x
        value_ref[...] = val_acc.astype(value_ref.dtype)
        square_ref[...] = jnp.sum(sq_acc, axis=1, keepdims=True)
        parts = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=(56 if tile > 96 else 48) * 1024 * 1024,
        ),
    )(flat, weight, previous_pre)


def _collapse_flat(flat: jax.Array, previous_pre: jax.Array) -> jax.Array:
    """``(previous_pre[..., None] * flat.view(T, M, D).f32).sum(1)`` from slices.

    The ``[T, M, D]`` FP32 view costs XLA a relayout copy of the whole
    residual (``reshape`` to ``T(4,128)``) before the reduce. Summing the
    ``D``-wide column slices avoids it. For ``M == 4`` the sum uses the order
    XLA's reduce applies to that view, ``(s0 + s2) + (s1 + s3)``, so the result
    is bitwise equal. Other ``M`` keep the view.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    if mult != 4:
        residual = flat.reshape(tokens, mult, dim).astype(jnp.float32)
        return (previous_pre[..., None] * residual).sum(1)
    terms = [
        previous_pre[:, m : m + 1]
        * flat[:, m * dim : (m + 1) * dim].astype(jnp.float32)
        for m in range(mult)
    ]
    return (terms[0] + terms[2]) + (terms[1] + terms[3])


def _delayed_pre_flat_small(
    flat: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    *constants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """XLA `delayed_pre_jax` on a flat residual, collapsed by `_collapse_flat`.

    Calling `_pre_mix_and_gates_jax` directly avoids materializing the
    ``[T, M, D]`` view and its discarded 3D reduction.
    """
    mult = previous_pre.shape[1]
    post, comb, new_pre = _pre_mix_and_gates_jax(
        flat,
        fn,
        scale,
        base,
        mult,
        flat.shape[1] // mult,
        *constants,
    )
    value = _collapse_flat(flat, previous_pre).astype(flat.dtype)
    return post, comb, value, new_pre


# Smallest verify-sized batch (beyond the fixed sizes below) for which the
# fused post+pre Pallas pass beats the XLA path. Job 1225 (tpu7x, width 20480,
# packed weight), XLA -> Pallas tile 96, in µs: 288: 41.3 -> 42.9,
# 384: 65.7 -> 56.6, 576: 98.0 -> 71.0, 768: 118.1 -> 88.0. The pre-only pass
# was slower than XLA at every size from 192 to 960, so it gets no new sizes.
_POST_PRE_PALLAS_MIN_TOKENS = 576


def _pallas_tile(tokens: int, mixes: int, fused_post: bool = False) -> int | None:
    """Row tile for the flat Pallas passes, or None for the XLA path.

    Fixed sizes: 64, 96 and 128 (tile = tokens), 256 (tile 128), and
    ``>= 1024`` multiples of 64 (tile 128 when divisible by 128, else 64). With
    ``fused_post``, buckets ``>= _POST_PRE_PALLAS_MIN_TOKENS`` that are
    multiples of 96, 128 (when ``>= 1024``), 64 or 80 (in that order) take
    Pallas: DSpark verify ``C * (K + 1)`` (576, 768, 1152, ...) and draft
    ``C * K`` (640, ...), with tile 128 at 640. Smaller sizes (384, 512) stay
    on the XLA Megacore path. Job 1372 (TD9.3a, tpu7x, width 20480): tile 128
    with 56 MB VMEM at T=8192 cuts 80x call time from 61.86 ms to 55.39 ms
    (-6.47 ms), while verify sizes (576..1152) keep tile 96 with 48 MB VMEM.
    """
    if 3 * mixes > 128:
        return None
    if tokens in (64, 96, 128):
        return tokens
    if tokens == 256:
        return 128
    if fused_post and tokens >= _POST_PRE_PALLAS_MIN_TOKENS:
        if tokens == 640:
            return 128
        for tile in (96, 128, 64, 80) if tokens >= 1024 else (96, 64, 80):
            if tokens % tile == 0:
                return tile
    if tokens >= 1024 and tokens % 64 == 0:
        return 128 if tokens % 128 == 0 else 64
    return None


def delayed_pre_flat_jax(
    flat: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """`delayed_pre_jax` with ``x = residual = flat [T, M*D]``.

    Batches `_pallas_tile` accepts use one Pallas pass; viewing the flat
    residual as ``[T, M, D]`` costs XLA a relayout copy plus an FP32 copy for
    the projection. Other batches keep the XLA path.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    tile = _pallas_tile(tokens, mixes)
    if tile is None:
        return _delayed_pre_flat_small(flat, fn, scale, base, previous_pre, *constants)
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    mixes, square, value = _flat_pre_sums(flat, weight, previous_pre, mixes, tile)
    new_pre, post, comb = _pre_gates(mixes, square, mult, dim, scale, base, *constants)
    return post[..., None], comb, value, new_pre


def _post_streams_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> list[jax.Array]:
    """XLA 2D stream recombine matching `_post_kernel` accumulation order."""
    mult = comb_res_mix.shape[-1]
    dim = flat.shape[-1] // mult
    x_f32 = x.reshape(-1, dim).astype(flat.dtype).astype(jnp.float32)
    post = post_layer_mix.reshape(-1, mult)
    comb = comb_res_mix.reshape(-1, mult * mult)
    streams = [
        flat[:, i * dim : (i + 1) * dim].astype(jnp.float32) for i in range(mult)
    ]
    out_streams = []
    for j in range(mult):
        acc = post[:, j : j + 1] * x_f32
        for i in range(mult):
            acc = acc + comb[:, (i * mult + j) : (i * mult + j + 1)] * streams[i]
        out_streams.append(acc.astype(flat.dtype))
    return out_streams


def _post_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Concatenated `[T, M*D]` residual from `_post_streams_jax`."""
    return jnp.concatenate(
        _post_streams_jax(x, flat, post_layer_mix, comb_res_mix), axis=1
    )


def _post_and_collapse_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    previous_pre: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused stream recombine and collapse before `optimization_barrier`.

    Collapsing directly from the recombined stream slices avoids re-reading
    `flat_new [T, M*D]` from HBM in a separate `_collapse_flat` fusion after
    the barrier while preserving `_collapse_flat`'s `(s0 + s2) + (s1 + s3)`
    accumulation order.
    """
    out_streams = _post_streams_jax(x, flat, post_layer_mix, comb_res_mix)
    flat_new = jnp.concatenate(out_streams, axis=1)
    terms = [
        previous_pre[:, m : m + 1] * s.astype(jnp.float32)
        for m, s in enumerate(out_streams)
    ]
    value = (
        (terms[0] + terms[2]) + (terms[1] + terms[3])
        if len(terms) == 4
        else functools.reduce(jnp.add, terms)
    )
    return flat_new, value.astype(flat.dtype)


def _flat_post_pre_sums(
    sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre, mixes, tile
):
    """One Pallas pass fusing `mhc_post_2d` with `_flat_pre_sums`."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(
        sub_x_ref,
        res_ref,
        post_ref,
        comb_ref,
        w_ref,
        pre_ref,
        out_res_ref,
        mix_ref,
        square_ref,
        value_ref,
    ):
        sub_x_f32 = sub_x_ref[...].astype(jnp.float32)
        post = post_ref[...]
        comb = comb_ref[...]
        old_bf = [res_ref[:, i * dim : (i + 1) * dim] for i in range(mult)]
        for j in range(mult):
            acc = post[:, j : j + 1] * sub_x_f32
            for i in range(mult):
                k = i * mult + j
                acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
            out_res_ref[:, j * dim : (j + 1) * dim] = acc.astype(out_res_ref.dtype)
        pre, x = pre_ref[...], out_res_ref[:, :dim].astype(jnp.float32)
        val_acc, sq_acc = pre[:, :1] * x, x * x
        for m in range(1, mult):
            x = out_res_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            val_acc = val_acc + pre[:, m : m + 1] * x
            sq_acc = sq_acc + x * x
        value_ref[...] = val_acc.astype(value_ref.dtype)
        square_ref[...] = jnp.sum(sq_acc, axis=1, keepdims=True)
        parts = jnp.dot(
            out_res_ref[...], w_ref[...], preferred_element_type=jnp.float32
        )
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, dim), row),
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mult), row),
            pl.BlockSpec((tile, mult * mult), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, width), flat.dtype),
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=(56 if tile > 96 else 48) * 1024 * 1024,
        ),
    )(sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre)


def delayed_post_pre_flat_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused `post` + `delayed_pre_flat_jax` seam for `shifted_post_pre`."""
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    tile = _pallas_tile(tokens, mixes, fused_post=True)
    if tile is None:
        flat_new, value = jax.lax.optimization_barrier(
            _post_and_collapse_jax(
                x,
                flat,
                post_layer_mix,
                comb_res_mix,
                previous_pre,
            )
        )
        post, comb, new_pre = _pre_mix_and_gates_jax(
            flat_new,
            fn,
            scale,
            base,
            mult,
            dim,
            *constants,
        )
        return flat_new, post, comb, value, new_pre
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    flat_new, mixes_out, square, value = _flat_post_pre_sums(
        x.reshape(tokens, dim).astype(flat.dtype),
        flat,
        post_layer_mix.reshape(tokens, mult),
        comb_res_mix.reshape(tokens, mult * mult),
        weight,
        previous_pre,
        mixes,
        tile,
    )
    new_pre, post, comb = _pre_gates(
        mixes_out, square, mult, dim, scale, base, *constants
    )
    return flat_new, post[..., None], comb, value, new_pre


# Partial sums one D-sharded seam rank contributes per token: the mix logits,
# the residual square sum the gates normalize by, and the collapsed value's
# square sum the sublayer RMSNorm normalizes by.
_SHARDED_STATS = ("mixes", "residual_squares", "value_squares")


def _sharded_tile(tokens: int) -> int | None:
    """Row tile for the 1/8-width seam pass, or None for the XLA path.

    A rank's residual is ``[T, M * D / tp]`` (2560 wide at TP8), so a tile
    holds 8x more rows in the same VMEM as the full-width pass. On TPU7x the
    largest dividing tile was fastest: T=1152 runs 13.7 us at tile 384
    against 15.9 us at 96.
    """
    for tile in (384, 256, 128, 96, 64):
        if tokens % tile == 0:
            return tile
    return None


def delayed_post_pre_partial_flat_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    previous_pre: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One rank's slice of `delayed_post_pre_flat_jax` on D-sharded streams.

    ``x [T, D/tp]``, ``flat [T, M * D/tp]`` and the packed ``fn`` rows are this
    rank's columns of every stream. Post and collapse are column-local, so
    ``flat_new`` and ``value`` come out final; the per-token reductions over D
    come out as this rank's partial sums ``[T, mixes + 2]`` (see
    `_SHARDED_STATS`) for the caller to all-reduce.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    mixes = mult * (mult + 2)
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    tile = _sharded_tile(tokens)
    if tile is None:
        flat_new, value = _post_and_collapse_jax(
            x, flat, post_layer_mix, comb_res_mix, previous_pre
        )
        mix = _packed_mixes(flat_new, weight, mixes)
        flat_f32 = flat_new.astype(jnp.float32)
        square = jnp.sum(flat_f32 * flat_f32, axis=-1, keepdims=True)
    else:
        flat_new, mix, square, value = _flat_post_pre_sums(
            x.reshape(tokens, dim).astype(flat.dtype),
            flat,
            post_layer_mix.reshape(tokens, mult),
            comb_res_mix.reshape(tokens, mult * mult),
            weight,
            previous_pre,
            mixes,
            tile,
        )
    value_f32 = value.astype(jnp.float32)
    stats = jnp.concatenate(
        [mix, square, jnp.sum(value_f32 * value_f32, axis=-1, keepdims=True)], -1
    )
    return flat_new, value, stats


def sharded_gates_norm_jax(
    stats: jax.Array,
    value: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
    dim: int,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gates from all-reduced partial sums, and this rank's normalized value.

    ``stats`` is `delayed_post_pre_partial_flat_jax`'s output summed over the
    ranks; ``dim`` is the full hidden size both RMS denominators use.
    ``value`` and ``norm_weight`` are this rank's columns.
    """
    mixes = stats.shape[1] - 2
    mult = math.isqrt(mixes + 1) - 1
    new_pre, post, comb = _pre_gates(
        stats[:, :mixes],
        stats[:, mixes : mixes + 1],
        mult,
        dim,
        scale,
        base,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    value_f32 = value.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(stats[:, mixes + 1 : mixes + 2] / dim + norm_eps)
    normed = value_f32 * inv_rms * norm_weight.astype(jnp.float32)
    return post[..., None], comb, normed.astype(value.dtype), new_pre


def initialize_delayed_mhc():
    global _pre_op
    if _pre_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _pre_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_pre",
        delayed_pre_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            None,
            None,
            None,
            None,
            None,
        ),
    )

    def fake(residual, x, fn, scale, base, previous_pre, *args):
        tokens, mult, dim = residual.shape
        return (
            fn.new_empty((tokens, mult, 1)),
            fn.new_empty((tokens, mult, mult)),
            residual.new_empty((tokens, dim)),
            fn.new_empty((tokens, mult)),
        )

    _pre_op.register_fake(fake)

    global _pre_flat_op
    _pre_flat_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_pre_flat",
        delayed_pre_flat_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 5 + (None,) * 5,
    )

    def fake_flat(flat, fn, scale, base, previous_pre, *args):
        tokens, mult = previous_pre.shape
        return (
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    _pre_flat_op.register_fake(fake_flat)

    global _post_pre_flat_op
    _post_pre_flat_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_post_pre_flat",
        delayed_post_pre_flat_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 8 + (None,) * 5,
    )

    def fake_post_pre_flat(
        x, flat, post_layer_mix, comb_res_mix, fn, scale, base, previous_pre, *args
    ):
        tokens, mult = previous_pre.shape
        return (
            torch.empty_like(flat),
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    _post_pre_flat_op.register_fake(fake_post_pre_flat)

    global _post_pre_partial_flat_op, _sharded_gates_op
    _post_pre_partial_flat_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_post_pre_partial_flat",
        delayed_post_pre_partial_flat_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 6,
    )

    def fake_partial(x, flat, post_layer_mix, comb_res_mix, fn, previous_pre):
        tokens, mult = previous_pre.shape
        return (
            torch.empty_like(flat),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            previous_pre.new_empty((tokens, mult * (mult + 2) + 2)),
        )

    _post_pre_partial_flat_op.register_fake(fake_partial)

    _sharded_gates_op = pallas.jax_op(
        "pallas::deepseek_v41_sharded_mhc_gates_norm",
        sharded_gates_norm_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 5 + (None,) * 7,
    )

    def fake_gates(stats, value, scale, base, norm_weight, *args):
        tokens = stats.shape[0]
        mult = math.isqrt(stats.shape[1] - 1) - 1
        return (
            stats.new_empty((tokens, mult, 1)),
            stats.new_empty((tokens, mult, mult)),
            torch.empty_like(value),
            stats.new_empty((tokens, mult)),
        )

    _sharded_gates_op.register_fake(fake_gates)


_pre_flat_op = None
_post_pre_flat_op = None
_post_pre_partial_flat_op = None
_sharded_gates_op = None


def delayed_pre(residual, x, fn, scale, base, previous_pre, *constants):
    assert _pre_op is not None and _pre_flat_op is not None
    if x is None:
        return _pre_flat_op(
            residual.flatten(1), fn, scale, base, previous_pre, *constants
        )
    return _pre_op(residual, x, fn, scale, base, previous_pre, *constants)


def delayed_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    scale,
    base,
    previous_pre,
    *constants,
):
    assert _post_pre_flat_op is not None
    flat_new, post, comb, value, next_pre = _post_pre_flat_op(
        x,
        residual.flatten(1),
        post_layer_mix,
        comb_res_mix,
        fn,
        scale,
        base,
        previous_pre,
        *constants,
    )
    return flat_new.view_as(residual), post, comb, value, next_pre


_post_op = None


def initialize_mhc_post():
    global _post_op
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (
        get_mhc_post_op,
    )

    _post_op = get_mhc_post_op()


def sharded_mhc_enabled(tp_size: int | None = None) -> bool:
    """Whether seams keep the residual sharded along D across the TP ranks.

    Opt-in (``TPU_DEEPSEEK_V41_SHARDED_MHC``) and only meaningful with TP > 1.
    """
    if not envs.TPU_DEEPSEEK_V41_SHARDED_MHC:
        return False
    size = get_tensor_model_parallel_world_size() if tp_size is None else tp_size
    return size > 1


def _rank_columns(dim: int) -> slice:
    """This rank's contiguous block of the hidden dimension.

    The block a ``reduce_scatter(dim=-1)`` hands this rank, and the one an
    ``all_gather(dim=-1)`` puts back in place.
    """
    width = dim // get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    return slice(rank * width, (rank + 1) * width)


def is_sharded_residual(residual: torch.Tensor, dim: int) -> bool:
    """Whether ``residual [T, M, D_local]`` holds only this rank's columns."""
    return residual.shape[-1] != dim


def full_residual(residual: torch.Tensor, dim: int) -> torch.Tensor:
    """The full ``[T, M, D]`` streams, gathering a D-sharded residual."""
    if not is_sharded_residual(residual, dim):
        return residual
    return tensor_model_parallel_all_gather(residual.contiguous(), dim=-1)


def _local_residual(residual: torch.Tensor, dim: int) -> torch.Tensor:
    """This rank's ``[T, M, D / tp]`` columns of a residual of either width."""
    if is_sharded_residual(residual, dim):
        return residual
    return residual[..., _rank_columns(dim)].contiguous()


def _local_mix_weight(fn: torch.Tensor, dim: int, mult: int) -> torch.Tensor:
    """This rank's rows of the mix projection, stream by stream.

    ``fn`` is either the load-time packed BF16 ``[M * D, 128]`` weight or the
    FP32 ``[mixes, M * D]`` one; both index streams as ``s * D + column``.
    """
    cols = _rank_columns(dim)
    blocks = [slice(s * dim + cols.start, s * dim + cols.stop) for s in range(mult)]
    if fn.dtype == torch.bfloat16:
        return torch.cat([fn[b] for b in blocks], dim=0)
    return torch.cat([fn[:, b] for b in blocks], dim=1)


def _sharded_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    constants,
    pre_mix,
    norm_weight,
    norm_eps,
    capture_aux,
):
    """`shifted_post_pre` with the residual sharded along D.

    ``x`` is the sublayer's unreduced partial sum. Its all-reduce is split
    around the seam: the reduce-scatter hands each rank its columns, the seam
    runs on those 1/tp columns, one ``[T, mixes + 2]`` FP32 all-reduce
    completes the per-token sums the gates and the norm need, and the
    all-gather of the normalized value is the second half of the original
    all-reduce. The returned residual stays sharded.
    """
    assert _post_pre_partial_flat_op is not None and _sharded_gates_op is not None
    dim = x.shape[-1]
    mult = residual.shape[1]
    cols = _rank_columns(dim)
    x_local = tensor_model_parallel_reduce_scatter(x, dim=-1)
    residual = _local_residual(residual, dim)
    flat_new, value, stats = _post_pre_partial_flat_op(
        x_local,
        residual.flatten(1),
        post_layer_mix,
        comb_res_mix,
        _local_mix_weight(fn, dim, mult),
        pre_mix,
    )
    stats = tensor_model_parallel_all_reduce(stats)
    weight = (
        norm_weight[cols]
        if norm_weight is not None
        else torch.ones_like(value[0], dtype=torch.float32)
    )
    post_mix, comb_mix, normed, next_pre = _sharded_gates_op(
        stats, value, hc_scale, hc_base, weight, *constants, dim, norm_eps
    )
    value = tensor_model_parallel_all_gather(
        normed if norm_weight is not None else value, dim=-1
    )
    residual = flat_new.view_as(residual)
    aux = (
        tensor_model_parallel_all_gather(_mean_hc_streams(residual), dim=-1)
        if capture_aux
        else x.new_empty(0, dim)
    )
    return residual, post_mix, comb_mix, value, next_pre, aux


def post(x, residual, post_layer_mix, comb_res_mix):
    """The full ``[T, M, D]`` streams after the post-mix.

    A D-sharded residual is gathered first: the callers (Engram injection and
    the encoder/decoder stage boundary) consume the full streams.
    """
    if envs.TPU_DEEPSEEK_V41_SHARDED_MHC:
        residual = full_residual(residual, x.shape[-1])
    if residual.device.type == "tpu":
        assert _post_op is not None
        return _post_op(x, residual, post_layer_mix, comb_res_mix)
    return mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)


def collapse(x, pre_mix):
    return (x.float() * pre_mix.unsqueeze(-1)).sum(-2).to(x.dtype)


def _default_pre_mix(residual: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # First layer uses residual[:, 0], not the newly computed gate.
    # Avoid `new_zeros(residual.shape[:2])` and slice-scatter so SP
    # profile_run does not pass a SymIntArrayRef `[s0 // tp, 4]` to
    # `aten.new_zeros` in fx_to_mlir.
    first_stream = (
        torch.arange(residual.shape[1], dtype=torch.int32, device=residual.device) == 0
    ).to(dtype=dtype)
    return torch.zeros_like(residual[:, :, 0], dtype=dtype) + first_stream


def pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    x=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    if envs.TPU_DEEPSEEK_V41_SHARDED_MHC and norm_weight is not None:
        residual = full_residual(residual, norm_weight.shape[-1])
    if residual.device.type == "tpu":
        if pre_mix is None:
            pre_mix = _default_pre_mix(residual, hc_scale.dtype)
        post, comb, value, next_pre = delayed_pre(
            residual,
            x,
            fn,
            hc_scale,
            hc_base,
            pre_mix,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        post, comb, value, next_pre = mhc_pre_delayed_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            x=x,
        )
    if norm_weight is not None:
        value = rms_norm(value, norm_weight, norm_eps)
    return post, comb, value, next_pre


def _mean_hc_streams(residual: torch.Tensor) -> torch.Tensor:
    """``residual.mean(1)`` over the ``[T, M, D]`` hyper-connection streams.

    The 3-D view's reduce costs XLA a ``T(4,128)`` relayout copy of the whole
    residual. For ``M == 4`` this sums the ``D``-wide column slices of the
    flat ``[T, M * D]`` rows in FP32 in the order XLA's reduce applies to the
    view, ``(s0 + s2) + (s1 + s3)`` (see `_collapse_flat`), and scales by the
    exact power of two ``0.25`` (= ``/ 4``) before the one BF16 rounding.
    Other ``M`` keep ``mean(1)``.
    """
    tokens, mult, dim = residual.shape
    if mult != 4:
        return residual.mean(1)
    flat = residual.reshape(tokens, mult * dim)
    s = [flat[:, m * dim : (m + 1) * dim].float() for m in range(mult)]
    return (((s[0] + s[2]) + (s[1] + s[3])) * 0.25).to(residual.dtype)


def shifted_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
    capture_aux=False,
    *,
    stream=None,
    reduce_results=False,
):
    if stream is not None:
        raise ValueError("TPU mHC does not use CUDA streams")
    on_tpu = residual.device.type == "tpu" and _post_pre_flat_op is not None
    if on_tpu and pre_mix is None:
        pre_mix = _default_pre_mix(residual, hc_scale.dtype)
    if reduce_results and on_tpu and sharded_mhc_enabled():
        return _sharded_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat),
            pre_mix,
            norm_weight,
            norm_eps,
            capture_aux,
        )
    if reduce_results:
        x = tensor_model_parallel_all_reduce(x)
    if on_tpu:
        residual, post_mix, comb_mix, value, next_pre = delayed_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            pre_mix,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            value = rms_norm(value, norm_weight, norm_eps)
        aux = _mean_hc_streams(residual) if capture_aux else x.new_empty(0, x.shape[-1])
        return residual, post_mix, comb_mix, value, next_pre, aux
    residual = post(x, residual, post_layer_mix, comb_res_mix)
    aux = _mean_hc_streams(residual) if capture_aux else x.new_empty(0, x.shape[-1])
    return (
        residual,
        *pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        ),
        aux,
    )
