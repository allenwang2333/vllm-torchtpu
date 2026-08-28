from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec,
                                        SlidingWindowSpec)

from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes


def _config(*,
            pcp: int,
            enable_prefix_caching: bool = False,
            connector_enabled: bool = False,
            prefix_match_unit: int | None = None):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=512,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=object() if connector_enabled else None,
    )


def _full_spec(block_size: int = 512):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _mamba_spec(block_size: int = 4096):
    return MambaSpec(
        block_size=block_size,
        shapes=((3, 8192), ),
        dtypes=(torch.bfloat16, ),
        mamba_cache_mode="align",
    )


def _resolve_outcome(resolver, kv_cache_config, vllm_config):
    try:
        return "return", resolver(kv_cache_config, vllm_config)
    except ValueError as error:
        return "value_error", str(error)


def test_hybrid_full_attention_mamba_pcp_resolves_effective_block_size():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.engine import core as engine_core

    assert engine_core.resolve_kv_cache_block_sizes is resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )

    assert resolve_kv_cache_block_sizes(kv_cache_config,
                                        _config(pcp=8)) == (4096, 4096)
    assert resolve_kv_cache_block_sizes(
        kv_cache_config, _config(pcp=8,
                                 connector_enabled=True)) == (4096, 4096)
    assert resolve_kv_cache_block_sizes(
        kv_cache_config, _config(pcp=8,
                                 enable_prefix_caching=True)) == (4096, 512)


def test_hybrid_pcp_prefix_match_unit_uses_physical_block_multiples():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )

    explicit = _config(pcp=8,
                       enable_prefix_caching=True,
                       prefix_match_unit=1024)
    assert resolve_kv_cache_block_sizes(kv_cache_config,
                                        explicit) == (4096, 1024)

    sub_block = _config(pcp=8,
                        enable_prefix_caching=True,
                        prefix_match_unit=256)
    with pytest.raises(ValueError, match="multiple of physical block_size"):
        resolve_kv_cache_block_sizes(kv_cache_config, sub_block)

    not_group_aligned = _config(pcp=8,
                                enable_prefix_caching=True,
                                prefix_match_unit=1536)
    with pytest.raises(ValueError, match="PCP KV cache coordinator group"):
        resolve_kv_cache_block_sizes(kv_cache_config, not_group_aligned)


def test_hybrid_full_attention_mamba_pcp_builds_prefix_cache_coordinator():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                                   get_kv_cache_coordinator)
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=4096)),
            KVCacheGroupSpec(["gdn"], _mamba_spec()),
        ],
    )
    vllm_config = _config(pcp=8,
                          enable_prefix_caching=True,
                          connector_enabled=True)
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config)

    assert (scheduler_block_size, hash_block_size) == (32768, 512)
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=65536,
        max_in_flight_tokens=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=8,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )

    assert isinstance(coordinator, HybridKVCacheCoordinator)
    assert coordinator.enable_partial_hash_hits is True
    assert coordinator._cache_hit_alignment_tokens == 512
    assert [
        group.kv_cache_spec.block_size
        for group in coordinator.kv_cache_config.kv_cache_groups
    ] == [32768, 4096]
    assert [
        manager.block_size for manager in coordinator.single_type_managers
    ] == [32768, 4096]
    # PCP is folded into attention group specs only. Mamba recurrent state is
    # not token-sharded across PCP ranks, so its manager keeps the physical
    # block size. The caller's config remains untouched and rank-local.
    assert [
        group.kv_cache_spec.block_size
        for group in kv_cache_config.kv_cache_groups
    ] == [4096, 4096]


def test_hybrid_pcp_coordinator_recovers_pcp_folded_by_vllm_scheduler():
    """vLLM 0.26 passes PCP=1 after folding PCP into scheduler blocks."""
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec(block_size=2304)),
            KVCacheGroupSpec(["gdn"], _mamba_spec(block_size=2304)),
        ],
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        max_in_flight_tokens=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        # This is intentionally 1: vLLM 0.26's Scheduler hard-codes it.
        pcp_world_size=1,
        scheduler_block_size=9216,
        hash_block_size=2304,
    )

    assert [
        manager.block_size for manager in coordinator.single_type_managers
    ] == [9216, 2304]


def test_hybrid_pcp_patch_keeps_non_mamba_scope():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core import kv_cache_utils

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(
                ["sliding"],
                SlidingWindowSpec(
                    block_size=512,
                    num_kv_heads=4,
                    head_size=128,
                    sliding_window=1024,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )

    patched_resolver = kv_cache_utils.resolve_kv_cache_block_sizes
    original_resolver = (
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes)
    vllm_config = _config(pcp=8)

    assert _resolve_outcome(
        patched_resolver,
        kv_cache_config,
        vllm_config,
    ) == _resolve_outcome(
        original_resolver,
        kv_cache_config,
        vllm_config,
    )


def test_hybrid_pcp_coordinator_patch_keeps_non_mamba_scope():
    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], _full_spec()),
            KVCacheGroupSpec(
                ["sliding"],
                SlidingWindowSpec(
                    block_size=512,
                    num_kv_heads=4,
                    head_size=128,
                    sliding_window=1024,
                    dtype=torch.bfloat16,
                ),
            ),
        ],
    )

    with pytest.raises(AssertionError, match="PCP not support hybrid"):
        get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=4096,
            max_in_flight_tokens=4096,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=8,
            scheduler_block_size=512,
            hash_block_size=512,
        )
