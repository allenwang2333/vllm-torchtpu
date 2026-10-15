# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Which buckets the V-space DCP merge (TPU_DCP_MERGE_VSPACE) takes: exactly
the backbone's chunked reduce-scatter buckets, never decode, never the MTP
drafter, never under RS_BF16=2; MIN_TOKENS above the chunk gate narrows it."""

import types

import pytest

from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

ENV = {
    "TPU_DCP_MERGE_VSPACE": True,
    "TPU_DCP_MERGE_VSPACE_MIN_TOKENS": 1024,
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_MERGE_CHUNKED": 8,
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 1024,
    "TPU_DCP_MERGE_RS_BF16": 1,
}


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setattr(cp_mla.envs, k, v, raising=False)
    monkeypatch.setattr(
        cp_mla, "get_dcp_group", lambda: types.SimpleNamespace(world_size=8)
    )
    return monkeypatch


@pytest.mark.parametrize(
    "tokens,backbone,want",
    [
        (4, True, False),
        (16, True, False),
        (64, True, False),
        (512, True, False),
        (1024, True, True),
        (2048, True, True),
        (4096, True, True),
        (1032, True, False),  # not a multiple of 8 * chunks: unchunked
        (2048, False, False),
        (4096, False, False),  # the drafter
    ],
)
def test_buckets(env, tokens, backbone, want):
    assert cp_mla.vspace_merge_applies(tokens, backbone) is want


def test_off_rs_a2a_and_no_group(env):
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_VSPACE", False)
    assert not cp_mla.vspace_merge_applies(4096, True)
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_VSPACE", True)
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_RS_BF16", 2)
    assert not cp_mla.vspace_merge_applies(4096, True)
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_RS_BF16", 0)
    assert cp_mla.vspace_merge_applies(4096, True)
    env.setattr(cp_mla, "get_dcp_group", lambda: None)
    assert not cp_mla.vspace_merge_applies(4096, True)


def test_min_tokens_above_chunk_gate(env):
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_VSPACE_MIN_TOKENS", 2048)
    assert not cp_mla.vspace_merge_applies(1024, True)
    assert cp_mla.vspace_merge_applies(2048, True)
    # At or below the chunk gate it is implied (no comparison of its own).
    env.setattr(cp_mla.envs, "TPU_DCP_MERGE_VSPACE_MIN_TOKENS", 0)
    assert cp_mla.vspace_merge_applies(1024, True)


def test_drafter_never_compares_tokens(env):
    class Sym(int):
        def __lt__(self, o):
            raise AssertionError("compared")

        __le__ = __gt__ = __ge__ = __mod__ = __lt__

    assert not cp_mla.vspace_merge_applies(Sym(4096), False)
