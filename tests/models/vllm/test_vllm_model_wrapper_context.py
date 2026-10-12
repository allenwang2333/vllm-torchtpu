# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VllmModelWrapperContext nesting and LRU cache bounds."""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    _context_cache,
    get_vllm_model_wrapper_context,
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def clean_context():
    _context_cache.clear()
    import vllm_torchtpu.models.vllm.vllm_model_wrapper_context as mod

    mod._vllm_model_wrapper_context = None
    yield
    _context_cache.clear()
    mod._vllm_model_wrapper_context = None


def test_get_context_uninitialized():
    with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
        get_vllm_model_wrapper_context()


def test_context_manager_lifecycle_and_nesting():
    mesh1 = object()
    mesh2 = object()

    ctx1 = set_vllm_model_wrapper_context(mesh=mesh1)
    with ctx1 as active1:
        assert active1 is ctx1
        assert get_vllm_model_wrapper_context() is ctx1

        ctx2 = set_vllm_model_wrapper_context(mesh=mesh2)
        with ctx2 as active2:
            assert active2 is ctx2
            assert get_vllm_model_wrapper_context() is ctx2

        # Outer restored
        assert get_vllm_model_wrapper_context() is ctx1

    with pytest.raises(AssertionError):
        get_vllm_model_wrapper_context()


def test_context_cache_hit_and_attribute_mismatch():
    mesh = object()
    cfg = object()
    k1 = [torch.zeros(2)]
    idx1 = {"layer0": 0}
    b1 = torch.zeros(4)

    ctx1 = set_vllm_model_wrapper_context(
        mesh=mesh,
        kv_caches=k1,
        layer_name_to_kvcache_index=idx1,
        vllm_config=cfg,
        kv_cache_bundle=b1,
    )
    # Exact same attributes -> cache hit
    ctx2 = set_vllm_model_wrapper_context(
        mesh=mesh,
        kv_caches=k1,
        layer_name_to_kvcache_index=idx1,
        vllm_config=cfg,
        kv_cache_bundle=b1,
    )
    assert ctx2 is ctx1

    # Mismatch kv_caches
    k2 = [torch.ones(2)]
    ctx3 = set_vllm_model_wrapper_context(
        mesh=mesh,
        kv_caches=k2,
        layer_name_to_kvcache_index=idx1,
        vllm_config=cfg,
        kv_cache_bundle=b1,
    )
    assert ctx3 is not ctx1
    assert ctx3.kv_caches is k2

    # Mismatch layer_name_to_kvcache_index
    idx2 = {"layer1": 1}
    ctx4 = set_vllm_model_wrapper_context(
        mesh=mesh,
        kv_caches=k2,
        layer_name_to_kvcache_index=idx2,
        vllm_config=cfg,
        kv_cache_bundle=b1,
    )
    assert ctx4 is not ctx3
    assert ctx4.layer_name_to_kvcache_index is idx2

    # Mismatch kv_cache_bundle
    b2 = torch.ones(4)
    ctx5 = set_vllm_model_wrapper_context(
        mesh=mesh,
        kv_caches=k2,
        layer_name_to_kvcache_index=idx2,
        vllm_config=cfg,
        kv_cache_bundle=b2,
    )
    assert ctx5 is not ctx4
    assert ctx5.kv_cache_bundle is b2


def test_context_cache_active_context_safeguard():
    mesh = object()
    cfg = object()
    k1 = [torch.zeros(2)]
    k2 = [torch.ones(2)]

    ctx = set_vllm_model_wrapper_context(mesh=mesh, kv_caches=k1, vllm_config=cfg)
    key = (id(mesh), id(cfg))
    assert _context_cache[key] is ctx

    # When ctx is active in a with-statement:
    with ctx:
        # Requesting a different context with the same key while cached is active
        # creates a new ctx but does NOT overwrite _context_cache[key]
        ctx_temp = set_vllm_model_wrapper_context(
            mesh=mesh, kv_caches=k2, vllm_config=cfg
        )
        assert ctx_temp is not ctx
        assert _context_cache[key] is ctx

    # Nested active context where cached._prev is not None
    ctx_outer = set_vllm_model_wrapper_context(mesh=object())
    with ctx_outer:
        ctx_inner = set_vllm_model_wrapper_context(mesh=mesh, vllm_config=cfg)
        with ctx_inner:
            # Now ctx_inner is active and has _prev == ctx_outer
            assert ctx_inner._prev is not None
            # Mutating ctx_inner's key while _prev is not None
            ctx_nested_diff = set_vllm_model_wrapper_context(
                mesh=mesh, kv_caches=k2, vllm_config=cfg
            )
            assert ctx_nested_diff is not ctx_inner
            # Cache still holds ctx_inner
            assert _context_cache[key] is ctx_inner


def test_context_cache_lru_eviction():
    keys = []
    meshes = [object() for _ in range(5)]
    configs = [object() for _ in range(5)]

    # Fill cache up to capacity 4
    for i in range(4):
        set_vllm_model_wrapper_context(mesh=meshes[i], vllm_config=configs[i])
        keys.append((id(meshes[i]), id(configs[i])))
    assert len(_context_cache) == 4
    assert keys[0] in _context_cache

    # 5th item triggers LRU eviction of the oldest key (keys[0])
    key4 = (id(meshes[4]), id(configs[4]))
    set_vllm_model_wrapper_context(mesh=meshes[4], vllm_config=configs[4])
    assert len(_context_cache) == 4
    assert keys[0] not in _context_cache
    assert key4 in _context_cache

    # Updating an existing key in a full cache (key1 in _context_cache) does not evict
    set_vllm_model_wrapper_context(
        mesh=meshes[1], vllm_config=configs[1], kv_caches=[torch.zeros(1)]
    )
    assert len(_context_cache) == 4
    assert keys[1] in _context_cache
    assert keys[2] in _context_cache
    assert keys[3] in _context_cache
    assert key4 in _context_cache
