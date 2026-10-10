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
"""DeepSeek-V4 indexer K cache layout policy: one decision, every consumer."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.deepseek_v4 import indexer_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import config
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (
    VllmDeepseekV4IndexerCache,
)

SAL = KVLayout.SEQ_ALONG_LANE
HAS = KVLayout.HEAD_ALONG_SUBLANE


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "VLLM_TPU_INDEXER_KV_CACHE_MODE",
        "TPU_USE_RAIDEN_KV_CACHE_MANAGER",
        "TPU_RAIDEN_GLM_ADMISSION",
        "TPU_RAIDEN_DSV4_ADMISSION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "mode, page_size, expected",
    [
        (None, 256, SAL),  # default mode, DSv4's 1024-token block
        ("sal", 256, SAL),
        ("has", 256, HAS),
        ("xla", 256, HAS),
        # The compressor state rows need a 256-lane page.
        (None, 128, HAS),
        ("sal", 512, HAS),
        (None, 64, HAS),
    ],
)
def test_select_layout(monkeypatch, mode, page_size, expected):
    if mode is not None:
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
    assert indexer_cache_layout.select_layout(page_size) == expected


def test_raiden_dsv4_admission_falls_back(monkeypatch):
    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_DSV4_ADMISSION", "1")
    assert indexer_cache_layout.select_layout(256) == HAS
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    with pytest.raises(ValueError, match="Raiden DSv4 admission"):
        indexer_cache_layout.select_layout(256)


@pytest.mark.parametrize(
    "layout, expected",
    [(SAL, (5, 33, 4, 256)), (HAS, (5, 64, 4, 256))],
)
def test_cache_shape(layout, expected):
    assert indexer_cache_layout.cache_shape(5, 256, layout) == expected


@pytest.mark.parametrize("mode", ["sal", "has"])
def test_state_cache_pages_match_the_allocation(monkeypatch, mode):
    """The state cache spec and the runner must page the indexer array alike;
    the SEQ_ALONG_LANE page still holds the CSA peer's 16 token states."""
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
    block_size, ratio = 1024, 4
    layout = indexer_cache_layout.select_layout(block_size // ratio)
    rows = indexer_cache_layout.cache_shape(1, block_size // ratio, layout)[1]
    assert config.physical_page_size(config.Mode.CSA_INDEXER, block_size, ratio) == (
        rows
    )
    assert config.state_block_size(config.Mode.CSA_INDEXER, block_size, ratio) == 16
    assert config.state_block_size(config.Mode.CSA, block_size, ratio) == 16


@pytest.mark.parametrize(
    "mode, block_size, expected",
    [("sal", 1024, SAL), ("has", 1024, HAS), ("sal", 512, HAS)],
)
def test_indexer_cache_layer_layout(monkeypatch, mode, block_size, expected):
    """The layer the compressor and indexer ops read the layout from."""
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
    cache = VllmDeepseekV4IndexerCache.__new__(VllmDeepseekV4IndexerCache)
    cache.cache_config = SimpleNamespace(block_size=block_size)
    cache.compress_ratio = 4
    assert cache.kv_layout == expected
