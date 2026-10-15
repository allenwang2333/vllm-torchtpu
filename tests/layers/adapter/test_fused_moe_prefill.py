# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Token-count dispatch of the fused-MoE op's per-rank body with the prefill
kernel on: from FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS the prefill kernel serves
it; below, the decode kernel up to its cap when on, else the GMM MoE fallback.
No TPU needed: every path is a stub."""

import numpy as np
import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge

H = 8


def test_prefill_envs(monkeypatch):
    monkeypatch.delenv("FUSED_MOE_PREFILL_KERNEL", raising=False)
    monkeypatch.delenv("FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS", raising=False)
    assert envs.FUSED_MOE_PREFILL_KERNEL is False
    assert envs.FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS == 256
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS", "1024")
    assert envs.FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS == 1024


@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize(
    "decode_max,tokens,path",
    [
        (128, 1, "decode"),
        (128, 128, "decode"),
        (128, 129, "gmm"),
        (128, 256, "prefill"),
        (512, 255, "decode"),
        (512, 256, "prefill"),  # the prefill kernel takes precedence
        (0, 255, "gmm"),  # decode kernel off
        (0, 4096, "prefill"),
    ],
)
def test_token_dispatch(monkeypatch, decode_max, tokens, path, expert_parallel):
    calls = []

    def decode(rank, x, *args, expert_start, **kwargs):
        calls.append(("decode", expert_start))
        return x

    def gmm(*args, topk):
        calls.append(("gmm", args[12]))
        return args[0]

    def prefill(*args):
        calls.append(("prefill", args[12]))
        return args[0]

    monkeypatch.setattr(bridge, "_kernel_body", decode)
    monkeypatch.setattr(bridge, "_gmm_layer", gmm)
    monkeypatch.setattr(bridge, "prefill_layer", prefill)
    rank = np.array([[3, 192]], np.int32)  # [[mesh index, first expert]]
    w = np.zeros((1, 1), np.float32)
    bridge._layer(
        rank,
        np.zeros((tokens, H), np.float32),
        *([w] * 9),
        np.zeros((tokens, 2), np.float32),
        np.zeros((tokens, 2), np.int32),
        n_ranks=8,
        topk=2,
        expert_parallel=expert_parallel,
        max_tokens=decode_max,
        prefill_min_tokens=256,
    )
    assert calls == [(path, 192 if expert_parallel else None)]
