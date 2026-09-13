# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Block count of a KV tensor shared by attention and Mamba layers.

Numbers are Kimi-Linear-48B on tpu7x-8 with fp8 KV and 256-token pages: an
MLA page is 256 x 640 B = 163,840 B and each KDA layer keeps 280,576 B of
state per request. vLLM pads every layer's page to
163,840 + 3 x 280,576 = 1,005,568 B and sizes the shared tensor as
num_blocks x that page.
"""
import dataclasses

import pytest
import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, MambaSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend)
from vllm_torchtpu.runner.tpu_runner import shared_kv_tensor_num_blocks

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 256
NUM_BLOCKS = 2167


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )


def _kda_specs() -> list[MambaSpec]:
    # Three KDA layers share the tensor with one attention layer; their
    # per-request state (conv window + recurrent state) is what vLLM pads
    # into every block.
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 3, 32, 128), (32, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=None,
    )
    return [spec, spec, spec]


def _mamba_bytes(spec: MambaSpec) -> int:
    return dataclasses.replace(spec, page_size_padded=None).page_size_bytes


def test_mla_shared_tensor_holds_every_block_vllm_hands_out():
    mla_page = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, torch.float8_e4m3fn)
    assert mla_page == 256 * 640
    padded_page = mla_page + sum(_mamba_bytes(s) for s in _kda_specs())
    tensor_size = NUM_BLOCKS * padded_page

    num_blocks = shared_kv_tensor_num_blocks(tensor_size,
                                             [_mla_spec(), *_kda_specs()],
                                             PallasMLAttentionBackend,
                                             NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS


def test_rpa_page_formula_undersizes_an_mla_tensor_and_is_rejected():
    # The RPA layout of an fp8 MLA spec (two packed heads x four values per
    # word) is four times the MLA page. Sizing an MLA-served tensor with it
    # leaves about a third of vLLM's block ids without a page, which the
    # helper must refuse.
    rpa_page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, torch.float8_e4m3fn)
    mla_page = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, torch.float8_e4m3fn)
    assert rpa_page == 4 * mla_page
    padded_page = mla_page + sum(_mamba_bytes(s) for s in _kda_specs())
    tensor_size = NUM_BLOCKS * padded_page

    with pytest.raises(ValueError, match="out of bounds"):
        shared_kv_tensor_num_blocks(tensor_size, [_mla_spec(), *_kda_specs()],
                                    PallasAttentionBackend, NUM_BLOCKS)


def test_rpa_shared_tensor_keeps_the_packed_page():
    # A non-MLA fp8 model served by the RPA backend keeps its packed page,
    # so a tensor sized with that page holds exactly vLLM's count.
    attn = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    rpa_page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 8, 128, torch.float8_e4m3fn)
    padded_page = rpa_page + sum(_mamba_bytes(s) for s in _kda_specs())
    tensor_size = NUM_BLOCKS * padded_page

    num_blocks = shared_kv_tensor_num_blocks(tensor_size,
                                             [attn, *_kda_specs()],
                                             PallasAttentionBackend,
                                             NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS
