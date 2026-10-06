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
"""DeepSeek-V4.1 Single-Pass Delayed mHC Pallas kernels."""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.rope import _lane_gather


def _packed_mixes(x: jax.Array, fn_72_ref, mix_hc: int) -> jax.Array:
    """Single-pass MXU projection on the [3 * mix_hc, hc_dim] bf16 split of hc_fn."""
    parts = jax.lax.dot_general(
        x,
        fn_72_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (
        parts[:, :mix_hc]
        + parts[:, mix_hc : 2 * mix_hc]
        + parts[:, 2 * mix_hc : 3 * mix_hc]
    )


def _rms_norm_to_ref(collapsed_bf16, nw_ref, normed_ref, rms_eps):
    cf32 = collapsed_bf16.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(cf32 * cf32, axis=-1, keepdims=True) + rms_eps)
    normed_ref[...] = (cf32 * rstd).astype(jnp.bfloat16) * nw_ref[...]


def _v41_in_kernel_gates(
    acc: jax.Array,
    sqr: jax.Array,
    scale_ref,
    hb_ref,
    post_out_ref,
    comb_out_ref,
    pre_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """In-kernel mhc_pre_gates (sigmoid + softmax + Sinkhorn-Knopp) for hc_mult=4."""
    mixes = acc * jax.lax.rsqrt(sqr / (hc_mult * hidden_size) + rms_eps)
    logits = mixes * scale_ref[...] + hb_ref[...]

    pre_logits = logits[:, :4]
    pre_out_ref[...] = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    idx4 = jnp.arange(4, dtype=jnp.int32)
    post_logits = _lane_gather(logits, idx4 + 4)
    post_out_ref[...] = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    idx16 = jnp.arange(16, dtype=jnp.int32)
    comb = _lane_gather(logits, idx16 + 8)

    row_swap1 = idx16 ^ 1
    row_swap2 = idx16 ^ 2
    col_swap1 = idx16 ^ 4
    col_swap2 = idx16 ^ 8

    m1 = jnp.maximum(comb, _lane_gather(comb, row_swap1))
    row_max = jnp.maximum(m1, _lane_gather(m1, row_swap2))
    exp_x = jnp.exp(comb - row_max)
    s1 = exp_x + _lane_gather(exp_x, row_swap1)
    row_sum = s1 + _lane_gather(s1, row_swap2)
    comb_mix = exp_x / row_sum + hc_sinkhorn_eps

    def _row_sum(v: jax.Array) -> jax.Array:
        r1 = v + _lane_gather(v, row_swap1)
        return r1 + _lane_gather(r1, row_swap2)

    def _col_sum(v: jax.Array) -> jax.Array:
        c1 = v + _lane_gather(v, col_swap1)
        return c1 + _lane_gather(c1, col_swap2)

    comb_mix = comb_mix / (_col_sum(comb_mix) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (_row_sum(comb_mix) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (_col_sum(comb_mix) + hc_sinkhorn_eps)

    comb_out_ref[...] = comb_mix


def _v41_pre_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
):
    """mhc_pre: 1-pass packed mix GEMM + sqrsum + in-kernel Sinkhorn gates + RMSNorm."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    acc = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    sqr = None
    collapsed_f32 = None
    for j in range(hc_mult):
        xj_f32 = x[:, j * hidden_size : (j + 1) * hidden_size].astype(jnp.float32)
        sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
        sqr = sj if sqr is None else sqr + sj
        if not use_stream0:
            assert pre_in is not None
            term = pre_in[:, j : j + 1] * xj_f32
            collapsed_f32 = term if collapsed_f32 is None else collapsed_f32 + term

    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    if use_stream0:
        collapsed_bf16 = x[:, :hidden_size]
    else:
        assert collapsed_f32 is not None
        collapsed_bf16 = collapsed_f32.astype(jnp.bfloat16)
    _rms_norm_to_ref(collapsed_bf16, nw_ref, normed_ref, rms_eps)


def _v41_pre_mixes_kernel(
    res_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    use_stream0: bool,
):
    """mhc_pre (multi-block): 1-pass packed mix GEMM + sqrsum + RMSNorm."""
    mix_hc = hc_mult * (hc_mult + 2)
    x = res_ref[...]
    mixes_ref[...] = _packed_mixes(x, fn_72_ref, mix_hc)
    pre_in = None if use_stream0 else pre_in_ref[...]
    sqr = None
    collapsed_f32 = None
    for j in range(hc_mult):
        xj_f32 = x[:, j * hidden_size : (j + 1) * hidden_size].astype(jnp.float32)
        sj = jnp.sum(xj_f32 * xj_f32, axis=-1, keepdims=True)
        sqr = sj if sqr is None else sqr + sj
        if not use_stream0:
            assert pre_in is not None
            term = pre_in[:, j : j + 1] * xj_f32
            collapsed_f32 = term if collapsed_f32 is None else collapsed_f32 + term

    assert sqr is not None
    sqr_ref[...] = sqr
    if use_stream0:
        collapsed_bf16 = x[:, :hidden_size]
    else:
        assert collapsed_f32 is not None
        collapsed_bf16 = collapsed_f32.astype(jnp.bfloat16)
    _rms_norm_to_ref(collapsed_bf16, nw_ref, normed_ref, rms_eps)


def _v41_fused_post_pre_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    scale_ref,
    hb_ref,
    newres_ref,
    post_out_ref,
    comb_out_ref,
    normed_ref,
    pre_out_ref,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Seam kernel: post recombine + 1-pass packed GEMM + in-kernel Sinkhorn."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    x_bf = x_ref[...]
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    streams_bf = []
    collapsed_f32 = None
    sqr = None
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)
        nf = new_bf.astype(jnp.float32)
        sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
        sqr = sj if sqr is None else sqr + sj
        term = pre_in[:, j : j + 1] * nf
        collapsed_f32 = term if collapsed_f32 is None else collapsed_f32 + term

    g_bf = jnp.concatenate(streams_bf, axis=1)
    acc = _packed_mixes(g_bf, fn_72_ref, mix_hc)
    assert sqr is not None
    _v41_in_kernel_gates(
        acc,
        sqr,
        scale_ref,
        hb_ref,
        post_out_ref,
        comb_out_ref,
        pre_out_ref,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    assert collapsed_f32 is not None
    _rms_norm_to_ref(collapsed_f32.astype(jnp.bfloat16), nw_ref, normed_ref, rms_eps)


def _v41_fused_post_pre_mixes_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    pre_in_ref,
    nw_ref,
    fn_72_ref,
    newres_ref,
    mixes_ref,
    sqr_ref,
    normed_ref,
    *,
    rms_eps: float,
):
    """Seam kernel (multi-block): post recombine + 1-pass packed mix GEMM + RMSNorm."""
    post = post_ref[...]
    comb = comb_ref[...]
    pre_in = pre_in_ref[...]
    hidden_size = x_ref.shape[1]
    hc_mult = post.shape[1]
    mix_hc = hc_mult * (hc_mult + 2)
    x_bf = x_ref[...]
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    streams_bf = []
    collapsed_f32 = None
    sqr = None
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)
        nf = new_bf.astype(jnp.float32)
        sj = jnp.sum(nf * nf, axis=-1, keepdims=True)
        sqr = sj if sqr is None else sqr + sj
        term = pre_in[:, j : j + 1] * nf
        collapsed_f32 = term if collapsed_f32 is None else collapsed_f32 + term

    g_bf = jnp.concatenate(streams_bf, axis=1)
    mixes_ref[...] = _packed_mixes(g_bf, fn_72_ref, mix_hc)
    assert sqr is not None
    sqr_ref[...] = sqr
    assert collapsed_f32 is not None
    _rms_norm_to_ref(collapsed_f32.astype(jnp.bfloat16), nw_ref, normed_ref, rms_eps)


def _v41_mhc_pallas_call(
    kernel,
    token_inputs: tuple[jax.Array, ...],
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array | None,
    hc_base: jax.Array | None,
    hc_mult: int,
    out_widths_dtypes: tuple[tuple[int, Any], ...],
    *,
    max_token_block: int = 128,
) -> tuple[jax.Array, ...]:
    """Run a V4.1 mHC kernel over token blocks with resident weights in VMEM."""
    num_tokens = token_inputs[0].shape[0]
    max_tb = (
        max(utils.SUBLANE, math.gcd(num_tokens, max_token_block))
        if num_tokens % utils.SUBLANE == 0
        else max_token_block
    )
    tb, padded_tokens = utils.select_token_block(num_tokens, max_tb)
    token_inputs = utils.pad_to(padded_tokens, *token_inputs)
    nw_2d = norm_weight.reshape(1, -1).astype(jnp.bfloat16)

    mix_hc = hc_mult * (hc_mult + 2)
    if hc_fn.ndim == 3 and hc_fn.dtype == jnp.bfloat16:
        hc_dim = hc_fn.shape[2]
        fn_72 = hc_fn.reshape(3 * mix_hc, hc_dim)
    elif (
        hc_fn.ndim == 2 and hc_fn.dtype == jnp.bfloat16 and hc_fn.shape[0] == 3 * mix_hc
    ):
        hc_dim = hc_fn.shape[1]
        fn_72 = hc_fn
    else:
        hc_dim = hc_fn.shape[1]
        fn_72 = jnp.concatenate(utils.split_fn3(hc_fn), axis=0)
    fn_spec = pl.BlockSpec((3 * mix_hc, hc_dim), lambda i: (0, 0))

    extra_inputs: tuple[jax.Array, ...] = ()
    extra_specs: tuple[pl.BlockSpec, ...] = ()
    if hc_scale is not None and hc_base is not None:
        if hc_scale.size == mix_hc:
            scale_vec = hc_scale.reshape(1, mix_hc).astype(jnp.float32)
        else:
            scale_vec = (
                jnp.repeat(
                    hc_scale.reshape(3),
                    jnp.array([hc_mult, hc_mult, hc_mult * hc_mult]),
                    total_repeat_length=mix_hc,
                )
                .reshape(1, mix_hc)
                .astype(jnp.float32)
            )
        hb_vec = hc_base.reshape(1, mix_hc).astype(jnp.float32)
        extra_inputs = (scale_vec, hb_vec)
        extra_specs = (
            pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
            pl.BlockSpec((1, mix_hc), lambda i: (0, 0)),
        )

    outs = pl.pallas_call(
        kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            *(pl.BlockSpec((tb, a.shape[1]), lambda i: (i, 0)) for a in token_inputs),
            pl.BlockSpec(nw_2d.shape, lambda i: (0, 0)),
            fn_spec,
            *extra_specs,
        ],
        out_specs=[
            pl.BlockSpec((tb, w), lambda i: (i, 0)) for w, _ in out_widths_dtypes
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, w), dt) for w, dt in out_widths_dtypes
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(*token_inputs, nw_2d, fn_72, *extra_inputs)
    return utils.trim_to(num_tokens, *outs)


def mhc_pre_v41(
    residual: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    num_tokens = res2d.shape[0]
    if num_tokens <= 16:
        post_mix, comb_mix, normed, next_pre_mix = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
            ),
            max_token_block=128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        mixes, sqr, normed = _v41_mhc_pallas_call(
            functools.partial(
                _v41_pre_mixes_kernel,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                use_stream0=use_stream0,
            ),
            (res2d, pre_mix_in.reshape(num_tokens, hc_mult)),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (hidden_size, jnp.bfloat16),
            ),
            max_token_block=128,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )


def mhc_fused_post_pre_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    if num_tokens <= 16:
        new_res2d, post_mix, comb_mix, normed, next_pre_mix = _v41_mhc_pallas_call(
            functools.partial(
                _v41_fused_post_pre_kernel,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
                (hc_mult * hc_mult, jnp.float32),
                (hidden_size, jnp.bfloat16),
                (hc_mult, jnp.float32),
            ),
            max_token_block=128,
        )
    else:
        mix_hc = hc_mult * (hc_mult + 2)
        new_res2d, mixes, sqr, normed = _v41_mhc_pallas_call(
            functools.partial(
                _v41_fused_post_pre_mixes_kernel,
                rms_eps=rms_eps,
            ),
            (
                x.reshape(num_tokens, hidden_size).astype(jnp.bfloat16),
                residual.reshape(num_tokens, hc_mult * hidden_size),
                post_layer_mix.reshape(num_tokens, hc_mult),
                comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
                pre_mix_in.reshape(num_tokens, hc_mult),
            ),
            norm_weight,
            hc_fn,
            None,
            None,
            hc_mult,
            (
                (hc_mult * hidden_size, jnp.bfloat16),
                (mix_hc, jnp.float32),
                (1, jnp.float32),
                (hidden_size, jnp.bfloat16),
            ),
            max_token_block=64,
        )
        next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
            mixes,
            sqr,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    return (
        new_res2d.reshape(residual.shape),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )
