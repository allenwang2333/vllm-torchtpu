# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics tests for the fused K3 AttentionResidual mix kernel."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.kimi_k3.attn_res_mix import attn_res_mix

HIDDEN = 7168
EPS = 1e-6


def _reference_mix(prefix_sum: torch.Tensor, block_residuals: torch.Tensor,
                   w_norm: torch.Tensor, w_proj: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """AttentionResidual.forward with vLLM RMSNorm/ReplicatedLinear rounding.

    RMSNorm: fp32 mean-square, round to bf16, then bf16 multiply by weight.
    ReplicatedLinear: bf16 inputs, fp32 accumulation, bf16 output.
    """
    values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
    v32 = values.float()
    inv_rms = torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + eps)
    normed = (v32 * inv_rms).to(values.dtype) * w_norm
    scores = (normed.float() @ w_proj.float().unsqueeze(-1)).to(values.dtype)
    probabilities = scores.float().softmax(dim=-2)
    return (probabilities * v32).sum(dim=-2).to(values.dtype)


def _make_case(n: int, num_tokens: int, seed: int):
    torch.manual_seed(seed)
    prefix = torch.randn(num_tokens, HIDDEN).bfloat16()
    blocks = torch.randn(num_tokens, n, HIDDEN).bfloat16()
    w_norm = (torch.rand(HIDDEN) + 0.5).bfloat16()
    w_proj = (torch.randn(HIDDEN) * 0.02).bfloat16()
    return prefix, blocks, w_norm, w_proj


def _to_jnp_bf16(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16)


def _run_kernel(prefix, blocks, w_norm, w_proj, n, exact_emulation):
    w_score = (w_norm.float() * w_proj.float()).numpy()
    out = attn_res_mix(
        _to_jnp_bf16(prefix),
        _to_jnp_bf16(blocks),
        jnp.asarray(w_score.reshape(1, HIDDEN), dtype=jnp.float32),
        _to_jnp_bf16(w_norm.reshape(1, HIDDEN)),
        _to_jnp_bf16(w_proj.reshape(1, HIDDEN)),
        n=n,
        eps=EPS,
        exact_emulation=exact_emulation,
    )
    return torch.from_numpy(np.asarray(out.astype(jnp.float32)))


@pytest.mark.parametrize("n", list(range(1, 9)))
@pytest.mark.parametrize("num_tokens", [2, 4, 8])
def test_default_mode_matches_torch_reference(n: int, num_tokens: int) -> None:
    prefix, blocks, w_norm, w_proj = _make_case(n, num_tokens, seed=n)
    actual = _run_kernel(prefix,
                         blocks,
                         w_norm,
                         w_proj,
                         n,
                         exact_emulation=False)
    expected = _reference_mix(prefix, blocks, w_norm, w_proj, EPS).float()
    # The fused score keeps fp32 where the reference rounds to bf16, so
    # agreement is bf16-grade, not bitwise.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_default_mode_prefill_block_boundary() -> None:
    # T=512 spans 32 grid steps; T=13 exercises a ragged final block; T=16 is
    # the serving decode bucket that hit Mosaic E2002 before the block-sizing
    # fix; T=27 exercises a ragged tail above one full block.
    for num_tokens in (13, 16, 27, 512):
        prefix, blocks, w_norm, w_proj = _make_case(4, num_tokens, seed=99)
        actual = _run_kernel(prefix,
                             blocks,
                             w_norm,
                             w_proj,
                             4,
                             exact_emulation=False)
        expected = _reference_mix(prefix, blocks, w_norm, w_proj, EPS).float()
        # At T=512 (3.7M elements) the tail of the designed fp32-vs-bf16
        # scoring difference lands a handful of elements just past 2e-2; the
        # exact_emulation test is the strict gate for real kernel drift.
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def _assert_near_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bitwise up to reduce-order noise: <=0.1% of elements off by <=1 ulp."""
    if torch.equal(actual, expected):
        return
    mismatch_fraction = (actual != expected).float().mean().item()
    max_abs = (actual - expected).abs().max().item()
    assert mismatch_fraction <= 1e-3 and max_abs <= 8e-3, (
        f"exact_emulation drifted: {mismatch_fraction=} {max_abs=}")


@pytest.mark.parametrize("n", [1, 4, 8])
def test_exact_emulation_is_near_bitwise(n: int) -> None:
    prefix, blocks, w_norm, w_proj = _make_case(n, 8, seed=n + 100)
    actual = _run_kernel(prefix,
                         blocks,
                         w_norm,
                         w_proj,
                         n,
                         exact_emulation=True)
    expected = _reference_mix(prefix, blocks, w_norm, w_proj, EPS).float()
    _assert_near_bitwise(actual, expected)


@pytest.mark.parametrize("n", list(range(1, 9)))
def test_kernel_abi_shapes(n: int) -> None:
    """Cheap eval_shape guard over all n variants (no device time)."""
    fn = functools.partial(attn_res_mix, n=n, eps=EPS, exact_emulation=False)
    out = jax.eval_shape(
        fn,
        jax.ShapeDtypeStruct((16, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((16, n, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, HIDDEN), jnp.float32),
        jax.ShapeDtypeStruct((1, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, HIDDEN), jnp.bfloat16),
    )
    assert out.shape == (16, HIDDEN) and out.dtype == jnp.bfloat16


@pytest.mark.parametrize("n", [1, 3, 8])
def test_torch_op_bridge_matches_reference(n: int) -> None:
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.vllm.custom_ops.k3_attn_res_op import \
        build_k3_attn_res_mix_op

    prefix, blocks, w_norm, w_proj = _make_case(n, 8, seed=n + 200)
    op = build_k3_attn_res_mix_op(n, EPS, False)
    assert op is build_k3_attn_res_mix_op(n, EPS, False)  # memoized

    device = torch.device("tpu")
    w_score = (w_norm.float() * w_proj.float()).reshape(1, HIDDEN)
    out = op(
        prefix.to(device),
        blocks.to(device),
        w_score.to(device),
        w_norm.reshape(1, HIDDEN).to(device),
        w_proj.reshape(1, HIDDEN).to(device),
    )
    expected = _reference_mix(prefix, blocks, w_norm, w_proj, EPS)
    torch.testing.assert_close(out.cpu().float(),
                               expected.float(),
                               rtol=2e-2,
                               atol=2e-2)
