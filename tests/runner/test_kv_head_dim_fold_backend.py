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
"""`TPU_RPA_FOLD_KV_HEAD_DIM` has to reach the vLLM-side page accounting too.

A half-applied flag is the failure mode that matters: the kernel would read a
512 B/token cache with a 1024 B/token stride, or the pool would be sized for a
page the backend never allocates. Both are silent.
"""

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
    wrapper as longctx_wrapper
from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.vllm import attention as vllm_attention
from vllm_torchtpu.layers.vllm.attention import (
    PallasAttentionBackend, PallasBatchedRPAAttentionBackend)


class _FakeTpuInfo:
    num_lanes = 128
    num_sublanes = 8


@pytest.fixture(autouse=True)
def _deviceless_tpu_info(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _FakeTpuInfo())
    # `attention` picks the wrapper module at import time from
    # USE_BATCHED_RPA_LONGCTX, so setting the env afterwards is not enough.
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True)
    monkeypatch.setattr(vllm_attention, "rpa_batched_wrapper", longctx_wrapper)


@pytest.fixture
def longctx_fold(monkeypatch):
    monkeypatch.setattr(envs, "TPU_RPA_FOLD_KV_HEAD_DIM", True)


def test_default_is_the_current_layout(monkeypatch):
    monkeypatch.setattr(envs, "TPU_RPA_FOLD_KV_HEAD_DIM", False)
    assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 1, 256, torch.float8_e4m3fn) == (4, 256, 1, 4, 256)


def test_backend_shape_and_page_bytes_move_together(longctx_fold):
    shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 1, 256, torch.float8_e4m3fn)
    assert shape == (4, 256, 1, 4, 128)

    page_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 256, torch.float8_e4m3fn)
    # This is what `fa_bytes_per_token` reports, and what the GDN pool sizes
    # its state regions against.
    assert page_bytes == 256 * 512
    assert page_bytes * 2 == PallasAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 256, torch.float8_e4m3fn)


def test_dp_rank_page_is_unchanged(longctx_fold):
    """Two KV heads already fill the packed word; the flag must not move them."""
    assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 2, 256, torch.float8_e4m3fn) == (4, 256, 1, 4, 256)


def test_spec_normalizer_follows_the_selected_backend(longctx_fold):
    """The pool is padded to the page the *selected* backend allocates.

    `allocate_raw_kv_cache_tensors` asserts `fa_page_bytes * split ==
    pool_page_bytes`; padding the spec with `PallasAttentionBackend` while the
    batched backend allocates a folded page breaks that assert (and, before it
    existed, silently over-allocated).
    """
    spec = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    normalized = normalize_kv_cache_specs_for_tpu(
        {"attn": spec},
        torch.float8_e4m3fn,
        attn_backend_cls=PallasBatchedRPAAttentionBackend,
    )
    assert normalized["attn"].page_size_bytes == (
        PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 1, 256, torch.float8_e4m3fn))
