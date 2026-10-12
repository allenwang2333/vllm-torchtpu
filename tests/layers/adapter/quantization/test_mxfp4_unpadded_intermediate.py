# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MXFP4 MoE: unpadded intermediate with block-tied w2 scales."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.quantization import mxfp4 as mxfp4_mod
from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor

EXPERTS = 2


def _moe_config(use_ep=True):
    return SimpleNamespace(
        experts_per_token=2,
        has_bias=True,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ep_size=2 if use_ep else 1,
        ep_rank=0,
        is_sequence_parallel=False,
        intermediate_size_per_partition=128,
    )


def _checkpoint_layer(inter=128, hidden=256, use_ep=True):
    """MXFP4 checkpoint tensors: packed nibbles plus e8m0 block-32 scales."""
    g = torch.Generator().manual_seed(0)

    def packed(rows, cols):
        return torch.randint(0, 256, (EXPERTS, rows, cols // 2), generator=g).to(
            torch.uint8
        )

    def scales(rows, cols):
        return torch.full((EXPERTS, rows, cols // 32), 127, dtype=torch.uint8)

    return SimpleNamespace(
        moe_config=_moe_config(use_ep),
        w13_weight=packed(2 * inter, hidden),
        w13_weight_scale=scales(2 * inter, hidden),
        w2_weight=packed(hidden, inter),
        w2_weight_scale=scales(hidden, inter),
        w13_bias=torch.randn(EXPERTS, 2 * inter, generator=g).to(torch.bfloat16),
        w2_bias=torch.randn(EXPERTS, hidden, generator=g).to(torch.bfloat16),
    )


def _dequant(quantized, scale, rows_per_scale):
    """Unpacked [E, K, N] fp4 values times their [E, K / rows, 1, N] scales."""
    values = np.asarray(quantized.astype(jnp.float32))
    scales = np.repeat(np.asarray(scale)[:, :, 0, :], rows_per_scale, axis=1)
    return values * scales


@pytest.mark.parametrize("pack", [False, True])
def test_tied_quantization_matches_padded_block(pack):
    """2304 rows tied at 512 -> 256 equal the 2560-row padded block-512 result."""
    w = jax.random.normal(jax.random.PRNGKey(0), (2, 2304, 256), jnp.float32)
    padded = jnp.pad(w, ((0, 0), (0, 256), (0, 0)))
    ref_q, ref_s = quantize_to_native_fp4_kmajor(padded, 512, pack)
    q, s = mxfp4_mod._quantize_fp4_kmajor_tied(w, block=512, sub=256, pack=pack)

    rows = 1152 if pack else 2304
    assert q.shape == (2, rows, 256) and s.shape == (2, 9, 1, 256)
    as_bits = (
        (lambda a: np.asarray(a))
        if pack
        else (lambda a: np.asarray(a.astype(jnp.float32)))
    )
    np.testing.assert_array_equal(as_bits(q), as_bits(ref_q[:, :rows]))
    np.testing.assert_array_equal(np.asarray(s[:, 0::2]), np.asarray(ref_s))
    np.testing.assert_array_equal(np.asarray(s[:, 1::2]), np.asarray(ref_s[:, :4]))
    if not pack:
        np.testing.assert_array_equal(
            _dequant(q, s, 256), _dequant(ref_q, ref_s, 512)[:, :2304]
        )


def test_tied_quantization_without_tail_only_repeats_scales():
    """K a whole number of blocks: no padding, every scale repeated."""
    w = jax.random.normal(jax.random.PRNGKey(1), (1, 1024, 128), jnp.float32)
    ref_q, ref_s = quantize_to_native_fp4_kmajor(w, 512, False)
    q, s = mxfp4_mod._quantize_fp4_kmajor_tied(w, block=512, sub=128, pack=False)
    assert s.shape == (1, 8, 1, 128)
    np.testing.assert_array_equal(
        np.asarray(q.astype(jnp.float32)), np.asarray(ref_q.astype(jnp.float32))
    )
    np.testing.assert_array_equal(
        np.asarray(s), np.repeat(np.asarray(ref_s), 4, axis=1)
    )


def test_tied_op_is_cached_per_closure(monkeypatch):
    built = []

    def fake_jax_op(name, fn):
        built.append((name, fn.keywords))
        return fn

    monkeypatch.setattr(mxfp4_mod.pallas, "jax_op", fake_jax_op)
    monkeypatch.setattr(mxfp4_mod, "_quantize_fp4_kmajor_tied_ops", {})
    first = mxfp4_mod._quantize_fp4_kmajor_tied_op(512, 256, True)
    assert mxfp4_mod._quantize_fp4_kmajor_tied_op(512, 256, True) is first
    mxfp4_mod._quantize_fp4_kmajor_tied_op(512, 256, False)
    assert built == [
        (
            "pallas::mxfp4_quantize_tied_kmajor_b512_s256",
            {"block": 512, "sub": 256, "pack": True},
        ),
        (
            "pallas::mxfp4_quantize_tied_kmajor_b512_s256_unpacked",
            {"block": 512, "sub": 256, "pack": False},
        ),
    ]


@pytest.mark.parametrize(
    "fused_kernel,w4a8,use_ep,expected",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_fused_ep_requested(monkeypatch, fused_kernel, w4a8, use_ep, expected):
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", fused_kernel)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", w4a8)
    layer = _checkpoint_layer(use_ep=use_ep)
    method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod
    assert method._fused_ep_requested(layer) is expected


@pytest.mark.parametrize(
    "block,inter,fused_ep,expected_inter",
    [
        # gcd(256, 384) = 128: the GMM layout keeps 384 unpadded.
        (256, 384, False, 384),
        # The fused EP kernel takes one block for both matmuls: pad to 512.
        (256, 384, True, 512),
        # gcd(256, 320) = 64 is off the 128 tile: pad as before.
        (256, 320, False, 512),
        # Already whole blocks.
        (128, 256, False, 256),
    ],
)
def test_dequantize_and_pad_intermediate(
    monkeypatch, block, inter, fused_ep, expected_inter
):
    monkeypatch.setattr(mxfp4_mod, "REQUANTIZED_BLOCK_SIZE", block)
    monkeypatch.setattr(envs, "VLLM_TPU_MXFP4_DEVICE_DEQUANT", False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", fused_ep)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", fused_ep)
    method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(_moe_config())
    method.intermediate_size = inter
    layer = _checkpoint_layer(inter=inter)
    w13, w2, b13, _, orig = method._dequantize_and_pad(layer)
    assert orig == inter
    assert w13.shape == (2, 256, 2 * expected_inter)
    assert w2.shape == (2, expected_inter, 256)
    assert b13.shape == (2, 1, 2 * expected_inter)
    pad = expected_inter - inter
    if pad:
        assert not w2[:, inter:].any()
        assert not w13[:, :, inter:expected_inter].any()


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize(
    "w2_rows,orig,expect_tied",
    [(2304, 2304, (512, 256)), (2560, 2304, None), (1024, 1024, None)],
)
def test_requantize_dispatches_tied_w2(monkeypatch, pack, w2_rows, orig, expect_tied):
    monkeypatch.setattr(mxfp4_mod, "REQUANTIZED_BLOCK_SIZE", 512)
    plain, tied = [], []

    def fake_plain(t, block, pack=True):
        plain.append((tuple(t.shape), block, pack))
        return t, torch.ones(1)

    def fake_tied_op(block, sub, pack):
        def run(t):
            tied.append((tuple(t.shape), block, sub, pack))
            return t, torch.zeros(1)

        return run

    monkeypatch.setattr(mxfp4_mod, "quantize_native_fp4_kmajor", fake_plain)
    monkeypatch.setattr(mxfp4_mod, "_quantize_fp4_kmajor_tied_op", fake_tied_op)
    method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(_moe_config())
    w13 = torch.zeros(1, 512, 2 * w2_rows)
    w2 = torch.zeros(1, w2_rows, 512)
    out = method._requantize_native_fp4(w13, w2, orig, torch.device("cpu"), pack=pack)

    assert out[0] is not None and out[1] is not None
    assert plain[0] == ((1, 512, 2 * w2_rows), 512, pack)
    if expect_tied is None:
        assert plain[1:] == [((1, w2_rows, 512), 512, pack)] and tied == []
    else:
        assert len(plain) == 1
        assert tied == [((1, w2_rows, 512), *expect_tied, pack)]
