# SPDX-License-Identifier: Apache-2.0
"""The block-ID invariant DeepSeek-V4's Raiden admission would rest on.

DSv4 overlays several logical caches onto one physical array: every SWA cache
onto a CSA NoPE array, every compressor state onto its own compressed-KV
layer's array (see ``TPUModelRunner._allocate_ds_v4_kv_caches``). Those
overlays sit at the *same* byte offset in the same array, so nothing about
their geometry keeps them apart. What keeps them apart is block ownership:
the docstring's claim is that "state and full-attention groups never own the
same block ID, so the rows never collide".

A pool manifest for DSv4 would therefore declare several pools over one
storage, all with ``base_offset_bytes=0`` and overlapping block ranges, and
rely on this invariant for correctness. That is worth an explicit test,
because a violation is silent: Raiden would copy real bytes to real
addresses, just the wrong ones, and the failure would surface as bad tokens
rather than as an error.

Deviceless: vLLM's block allocator is plain Python and never touches a TPU.
"""

from __future__ import annotations

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec,
                                        MLAAttentionSpec, SlidingWindowMLASpec)

# Mirrors the DSv4 geometry in _allocate_ds_v4_kv_caches: a CSA layer pages
# 1024 tokens at compress_ratio 4 (256 compressed tokens per page), and
# VllmDeepseekV4SWACache._swa_block_size gives 1024 // 4 // 2 = 128 so the SWA
# page lands on the same byte size as the CSA NoPE page it overlays.
_CSA_BLOCK_TOKENS = 1024
_CSA_COMPRESS_RATIO = 4
_SWA_BLOCK_TOKENS = _CSA_BLOCK_TOKENS // _CSA_COMPRESS_RATIO // 2
# Sized for the multi-request case: the SWA group takes 8 blocks to the CSA
# group's 1, so four 2-page requests need 4 * (2 + 2 + 16) blocks. Running
# short raises out of the *shared* pool, which is the invariant showing
# through: one group's allocation drains the blocks another group could use.
_NUM_BLOCKS = 256


def _csa_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=_CSA_BLOCK_TOKENS,
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.uint8,
        compress_ratio=_CSA_COMPRESS_RATIO,
        model_version="deepseek_v4",
    )


def _indexer_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=_CSA_BLOCK_TOKENS,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        compress_ratio=_CSA_COMPRESS_RATIO,
        model_version="deepseek_v4",
    )


def _swa_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=_SWA_BLOCK_TOKENS,
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.uint8,
        sliding_window=512,
        model_version="deepseek_v4",
    )


def _coordinator():
    """A DSv4-shaped coordinator: CSA, indexer and SWA in separate groups.

    Prefix caching is off so the probe measures plain allocation rather than
    cache-hit block sharing, which is a different (and deliberate) form of
    reuse.
    """
    config = KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["csa.0", "csa.1"],
                             kv_cache_spec=_csa_spec()),
            KVCacheGroupSpec(layer_names=["indexer.0"],
                             kv_cache_spec=_indexer_spec()),
            KVCacheGroupSpec(layer_names=["swa.0", "swa.1", "swa.2"],
                             kv_cache_spec=_swa_spec()),
        ],
    )
    return get_kv_cache_coordinator(
        config,
        max_model_len=_CSA_BLOCK_TOKENS * 8,
        max_in_flight_tokens=_CSA_BLOCK_TOKENS,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=_CSA_BLOCK_TOKENS,
        hash_block_size=_CSA_BLOCK_TOKENS,
    )


def _group_block_ids(coordinator, request_id: str) -> list[set[int]]:
    return [{block.block_id
             for block in blocks if block.block_id != 0}
            for blocks in coordinator.get_blocks(request_id)]


def test_all_groups_draw_from_one_shared_block_pool():
    """Every single-type manager is handed the *same* BlockPool instance.

    This is the structural reason block IDs are globally unique: there is one
    free-block queue, so a block that is popped for one group cannot also be
    popped for another.
    """
    coordinator = _coordinator()
    pools = {id(m.block_pool) for m in coordinator.single_type_managers}
    assert len(pools) == 1
    assert id(coordinator.block_pool) == pools.pop()


@pytest.mark.parametrize("num_tokens",
                         [1, _CSA_BLOCK_TOKENS, 3 * _CSA_BLOCK_TOKENS])
def test_cache_groups_never_share_a_block_id(num_tokens):
    """The invariant the DSv4 overlay depends on, for one request."""
    coordinator = _coordinator()
    coordinator.allocate_new_blocks("req-0",
                                    num_tokens=num_tokens,
                                    num_tokens_main_model=num_tokens)

    per_group = _group_block_ids(coordinator, "req-0")
    assert all(per_group), "every group should have been allocated blocks"

    union: set[int] = set()
    for group_index, block_ids in enumerate(per_group):
        collision = union & block_ids
        assert not collision, (
            f"group {group_index} shares block ids {sorted(collision)} with "
            "an earlier group; the DSv4 overlay would corrupt those pages")
        union |= block_ids
    assert len(union) == sum(len(ids) for ids in per_group)


def test_groups_stay_disjoint_across_concurrent_requests():
    """Disjointness must hold across requests too, not just within one.

    Two pools over one storage are safe only if no *live* block id is claimed
    twice anywhere, so the interesting check spans every request in flight.
    """
    coordinator = _coordinator()
    request_ids = [f"req-{index}" for index in range(4)]
    for request_id in request_ids:
        coordinator.allocate_new_blocks(request_id,
                                        num_tokens=2 * _CSA_BLOCK_TOKENS,
                                        num_tokens_main_model=2 *
                                        _CSA_BLOCK_TOKENS)

    seen: dict[int, tuple[str, int]] = {}
    for request_id in request_ids:
        for group_index, block_ids in enumerate(
                _group_block_ids(coordinator, request_id)):
            for block_id in block_ids:
                owner = seen.get(block_id)
                assert owner is None, (
                    f"block {block_id} is held by both {owner} and "
                    f"({request_id}, group {group_index})")
                seen[block_id] = (request_id, group_index)


def test_swa_group_pages_more_blocks_than_csa_for_the_same_tokens():
    """SWA's smaller logical block is a *count* difference, not a byte one.

    _swa_block_size divides the CSA block by 8, so the SWA group takes 8x the
    blocks for the same token count while each block covers the same 131072
    bytes. This is why the manifest can give both pools the same
    block_stride_bytes and the same num_blocks, and why the difference shows
    up in region granularity instead.
    """
    coordinator = _coordinator()
    coordinator.allocate_new_blocks("req-0",
                                    num_tokens=_CSA_BLOCK_TOKENS,
                                    num_tokens_main_model=_CSA_BLOCK_TOKENS)
    csa_blocks, _, swa_blocks = _group_block_ids(coordinator, "req-0")

    assert _CSA_BLOCK_TOKENS // _SWA_BLOCK_TOKENS == 8
    assert len(swa_blocks) == 8 * len(csa_blocks)

    csa_page_bytes = (_CSA_BLOCK_TOKENS // _CSA_COMPRESS_RATIO) * 512
    swa_page_bytes = _SWA_BLOCK_TOKENS * 1024
    assert csa_page_bytes == swa_page_bytes == 131072
