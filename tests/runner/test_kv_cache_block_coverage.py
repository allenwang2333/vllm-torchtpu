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
"""CPU tests for the KV-cache block-id coverage check.

Three parties decide how big a paged KV pool is, and until now none of them
compared notes:

  1. the scheduler treats the pool as `kv_cache_config.num_blocks` pages and
     hands the layers sharing it *disjoint* ids drawn from that whole range;
  2. the allocator in `initialize_kv_cache` decides `shape[0]`;
  3. the Pallas kernel computes `base + block_id * page_elems`.

Step 3 does no bounds check, so when 1 and 2 disagree the over-range id is
DMA'd into whatever allocation happens to sit next in HBM. There is no
`IndexError`; the symptom is garbage output or an SC/TC halt, and only once
occupancy climbs past the allocated fraction, which is why #845 and #986 both
shipped and why #838 looked like a hardware fault.

These tests live apart from `test_kv_cache_manager.py` on purpose. That module
builds a real `VllmConfig`, which needs `torch.device("tpu")`, which
`torch_tpu._loader.load()` only registers when `get_tpu_device_count()` is
non-zero -- so it cannot run on the CPU lane. The coverage check takes plain
tensors and specs, so it can.
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        KVCacheTensor, MLAAttentionSpec)

from vllm_torchtpu.runner.kv_cache_manager import \
    check_kv_caches_cover_block_ids

PAGE_SIZE_BYTES = 16384
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 128


def _attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=BLOCK_SIZE,
                             num_kv_heads=NUM_KV_HEADS,
                             head_size=HEAD_SIZE,
                             dtype=torch.bfloat16,
                             page_size_padded=PAGE_SIZE_BYTES)


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(block_size=BLOCK_SIZE,
                            num_kv_heads=1,
                            head_size=HEAD_SIZE,
                            dtype=torch.bfloat16)


def _cache(num_blocks: int) -> torch.Tensor:
    """A stand-in for the TPU 5D packed layout; only `shape[0]` is read."""
    return torch.empty((num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1, HEAD_SIZE),
                       dtype=torch.bfloat16)


def _spec_lookup(specs: dict[str, KVCacheSpec]):
    return lambda layer_name: specs[layer_name]


@pytest.mark.cpu_test
def test_pool_smaller_than_the_block_id_range_is_rejected():
    """The exact shape of #986: L layers share a pool, each gets 1/L of it."""
    specs = {"attn.0": _attention_spec(), "attn.1": _attention_spec()}
    kv_cache_tensors = [
        KVCacheTensor(size=8 * PAGE_SIZE_BYTES, shared_by=["attn.0",
                                                           "attn.1"]),
    ]
    # The scheduler will issue ids 0..7 to both layers, but each layer was
    # given only 4 pages.
    kv_caches = {"attn.0": _cache(4), "attn.1": _cache(4)}

    with pytest.raises(ValueError) as excinfo:
        check_kv_caches_cover_block_ids(kv_caches, kv_cache_tensors, 8,
                                        _spec_lookup(specs))

    message = str(excinfo.value)
    assert "attn.0" in message
    assert "4" in message and "7" in message


@pytest.mark.cpu_test
def test_pool_covering_every_block_id_is_accepted():
    """A guard that rejected everything would pass the test above."""
    specs = {"attn.0": _attention_spec(), "attn.1": _attention_spec()}
    kv_cache_tensors = [
        KVCacheTensor(size=8 * PAGE_SIZE_BYTES, shared_by=["attn.0",
                                                           "attn.1"]),
    ]
    kv_caches = {"attn.0": _cache(8), "attn.1": _cache(8)}

    check_kv_caches_cover_block_ids(kv_caches, kv_cache_tensors, 8,
                                    _spec_lookup(specs))


@pytest.mark.cpu_test
def test_an_oversized_pool_is_also_rejected():
    """Not a safety problem, but it means someone's arithmetic is wrong and
    the extra pages are unreachable memory. Fail loudly rather than leak."""
    specs = {"attn.0": _attention_spec()}
    kv_cache_tensors = [
        KVCacheTensor(size=8 * PAGE_SIZE_BYTES, shared_by=["attn.0"]),
    ]
    kv_caches = {"attn.0": _cache(16)}

    with pytest.raises(ValueError):
        check_kv_caches_cover_block_ids(kv_caches, kv_cache_tensors, 8,
                                        _spec_lookup(specs))


@pytest.mark.cpu_test
def test_mla_layers_are_out_of_scope():
    """MLA owns its own layouts (and sparse MLA binds a tuple), so the check
    deliberately skips it. `MLAAttentionSpec` subclasses `AttentionSpec`, so
    this exclusion has to be explicit -- hence the test."""
    specs = {"mla.0": _mla_spec()}
    kv_cache_tensors = [
        KVCacheTensor(size=8 * PAGE_SIZE_BYTES, shared_by=["mla.0"]),
    ]
    kv_caches = {"mla.0": _cache(4)}

    check_kv_caches_cover_block_ids(kv_caches, kv_cache_tensors, 8,
                                    _spec_lookup(specs))


@pytest.mark.cpu_test
def test_layers_with_no_allocation_are_skipped():
    """Mamba layers bind a tuple of state tensors under a different block
    count, and are filtered out by the spec check before this point. A layer
    that never made it into `kv_caches` must not crash the check itself."""
    specs = {"attn.0": _attention_spec()}
    kv_cache_tensors = [
        KVCacheTensor(size=8 * PAGE_SIZE_BYTES, shared_by=["attn.0"]),
    ]

    check_kv_caches_cover_block_ids({}, kv_cache_tensors, 8,
                                    _spec_lookup(specs))
