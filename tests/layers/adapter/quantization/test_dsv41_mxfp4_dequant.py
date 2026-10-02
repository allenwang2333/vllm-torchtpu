"""The V4.1 on-device MXFP4 dequantize must match the V4 CPU path bit for bit."""

import types

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as v41
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    VllmDeepseekV4Mxfp4MoEMethod,
)


def _jax_dequantize_pad(w_packed, scale_u8, split, pad_k, pad_n):
    out = v41.dequantize_pad_mxfp4_kmajor(
        jnp.asarray(w_packed.numpy()),
        jnp.asarray(scale_u8.numpy()),
        split=split,
        pad_k=pad_k,
        pad_n=pad_n,
    )
    return torch.from_numpy(np.asarray(out.astype(jnp.float32))).to(torch.bfloat16)


@pytest.mark.parametrize("hidden,inter", [(1024, 384), (768, 640)])
def test_dequantize_and_pad_matches_cpu_path(monkeypatch, hidden, inter):
    generator = torch.Generator().manual_seed(hidden + inter)
    experts = 3

    def packed(*shape):
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)

    def scales(*shape):
        return torch.randint(100, 150, shape, generator=generator, dtype=torch.uint8)

    layer = types.SimpleNamespace(
        w13_weight=packed(experts, 2 * inter, hidden // 2),
        w13_weight_scale=scales(experts, 2 * inter, hidden // 32),
        w2_weight=packed(experts, hidden, inter // 2),
        w2_weight_scale=scales(experts, hidden, inter // 32),
    )
    v4 = object.__new__(VllmDeepseekV4Mxfp4MoEMethod)
    v4.intermediate_size = inter
    expected = v4._dequantize_and_pad(layer)

    monkeypatch.setattr(v41, "_dequantize_pad", _jax_dequantize_pad)
    method = object.__new__(v41.VllmDeepseekV41Mxfp4MoEMethod)
    method.intermediate_size = inter
    got = method._dequantize_and_pad(layer)

    for g, e in zip(got[:2], expected[:2]):
        assert g.shape == e.shape
        assert torch.equal(g.view(torch.int16), e.view(torch.int16))
    assert got[2:] == (None, None, expected[4])
