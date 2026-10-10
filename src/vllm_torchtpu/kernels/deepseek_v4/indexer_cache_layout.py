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
"""DeepSeek-V4 lightning-indexer K cache layout.

A record is 128 fp8 values and one UE8M0 scale byte. For a page of ``T``
compressed tokens the cache is one of (see `streamindex_topk.config.KVLayout`):

  HEAD_ALONG_SUBLANE  ``uint8[num_pages, T // 4, 4, 256]``: 4 tokens per row,
                      each record padded to 256 bytes.
  SEQ_ALONG_LANE      ``uint8[num_pages, 33, 4, T]``: token ``t`` is lane
                      ``t``; byte ``4 g + k`` of its record is ``[g, k, t]``,
                      the scale is byte 128, so 132 bytes per token.

The same array also hosts the indexer compressor's f32 state on pages of the
state cache group. The compress-and-store kernels address that state as
``(4, 256)`` byte rows, two per token state, so SEQ_ALONG_LANE is only taken
when ``T == 256``: its trailing dims are then exactly that row, and a page's 33
rows hold the 16 token states the CSA peer group is floored to.

`select_layout` is the single decision: the runner allocates with it, the state
cache spec pages with it, and the compressor and indexer ops trace for it.
"""

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    get_kv_cache_shape,
    indexer_kv_layout,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

HEAD_DIM = 128
# Compressed tokens per page SEQ_ALONG_LANE needs (see the module docstring).
SEQ_ALONG_LANE_PAGE_SIZE = 256

_FALLBACK_LOGGED: set[tuple[str, int]] = set()


def _warn_fallback_once(reason: str, page_size: int, message: str) -> None:
    key = (reason, page_size)
    if key not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add(key)
        logger.warning(message)


def select_layout(page_size: int) -> KVLayout:
    """Layout of a DeepSeek-V4 indexer K cache with ``page_size`` tokens/page.

    Follows ``VLLM_TPU_INDEXER_KV_CACHE_MODE`` (default ``sal``) through
    `indexer_kv_layout`, then falls back to HEAD_ALONG_SUBLANE where the DSv4
    cache cannot take SEQ_ALONG_LANE: a page size other than
    `SEQ_ALONG_LANE_PAGE_SIZE`, or Raiden DSv4 admission, whose pool manifest
    expects the packed-row array.

    Args:
      page_size: Compressed tokens per page (``block_size // compress_ratio``).
    """
    layout = indexer_kv_layout(page_size)
    if layout != KVLayout.SEQ_ALONG_LANE:
        return layout

    from vllm_torchtpu import envs

    if envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER and envs.TPU_RAIDEN_DSV4_ADMISSION:
        if envs.is_indexer_kv_cache_mode_explicit():
            raise ValueError(
                "VLLM_TPU_INDEXER_KV_CACHE_MODE='sal' is incompatible with "
                "Raiden DSv4 admission (TPU_RAIDEN_DSV4_ADMISSION=1), whose "
                "pool manifest expects the HEAD_ALONG_SUBLANE indexer array. "
                "Set VLLM_TPU_INDEXER_KV_CACHE_MODE=has."
            )
        _warn_fallback_once(
            "raiden_dsv4_admission",
            page_size,
            "Raiden DSv4 admission expects the HEAD_ALONG_SUBLANE indexer "
            "array; using it instead of the default SEQ_ALONG_LANE. Set "
            "VLLM_TPU_INDEXER_KV_CACHE_MODE=has to silence this warning.",
        )
        return KVLayout.HEAD_ALONG_SUBLANE
    if page_size != SEQ_ALONG_LANE_PAGE_SIZE:
        _warn_fallback_once(
            "page_size",
            page_size,
            f"DeepSeek-V4 indexer K cache: SEQ_ALONG_LANE needs "
            f"{SEQ_ALONG_LANE_PAGE_SIZE} compressed tokens per page to host the "
            f"compressor state, got {page_size}; using HEAD_ALONG_SUBLANE.",
        )
        return KVLayout.HEAD_ALONG_SUBLANE
    return KVLayout.SEQ_ALONG_LANE


def cache_shape(num_pages: int, page_size: int, layout: KVLayout) -> tuple[int, ...]:
    """The uint8 array shape of an indexer K cache in ``layout``."""
    return get_kv_cache_shape(num_pages, page_size, HEAD_DIM, layout)


def page_rows(page_size: int, layout: KVLayout) -> int:
    """Rows per page (``shape[1]``), the unit the compressor state is paged in."""
    return cache_shape(1, page_size, layout)[1]
