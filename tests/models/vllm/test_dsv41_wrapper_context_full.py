# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for models/vllm/vllm_model_wrapper_context.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.models", ("vllm_torchtpu", "models")),
    ("vllm_torchtpu.models.vllm", ("vllm_torchtpu", "models", "vllm")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg

from vllm_torchtpu.models.vllm import (  # noqa: E402
    vllm_model_wrapper_context as ctx_mod,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (  # noqa: E402
    get_vllm_model_wrapper_context,
    set_vllm_model_wrapper_context,
)


def test_vllm_model_wrapper_context_all_branches() -> None:
    ctx_mod._vllm_model_wrapper_context = None
    ctx_mod._last_context = None
    ctx_mod._context_cache.clear()
    ctx_mod._reentrant_cache.clear()

    # 1. get_vllm_model_wrapper_context raises AssertionError when unset
    with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
        get_vllm_model_wrapper_context()

    mesh_a: Any = object()
    mesh_b: Any = object()
    kvc: Any = []
    lmap = {"layer.0": 0}
    cfg = object()
    bundle: Any = object()

    # 2. Cold miss creates context and populates _context_cache + _last_context
    ctx1 = set_vllm_model_wrapper_context(
        mesh=mesh_a,
        kv_caches=kvc,
        layer_name_to_kvcache_index=lmap,
        vllm_config=cfg,
        kv_cache_bundle=bundle,
    )

    # 3. _last_context fast path hit when inactive
    ctx1_again = set_vllm_model_wrapper_context(
        mesh=mesh_a,
        kv_caches=kvc,
        layer_name_to_kvcache_index=lmap,
        vllm_config=cfg,
        kv_cache_bundle=bundle,
    )
    assert ctx1_again is ctx1

    # 4. Second distinct config (misses _last_context, misses _context_cache)
    ctx2 = set_vllm_model_wrapper_context(
        mesh=mesh_b,
        kv_caches=kvc,
        layer_name_to_kvcache_index=lmap,
        vllm_config=cfg,
        kv_cache_bundle=bundle,
    )
    assert ctx2 is not ctx1

    # 5. Switching back to mesh_a misses _last_context, hits _context_cache
    ctx1_from_dict = set_vllm_model_wrapper_context(
        mesh=mesh_a,
        kv_caches=kvc,
        layer_name_to_kvcache_index=lmap,
        vllm_config=cfg,
        kv_cache_bundle=bundle,
    )
    assert ctx1_from_dict is ctx1

    # 6. Enter context, test get_vllm_model_wrapper_context, and test re-entrant cache
    with ctx1 as entered1:
        assert get_vllm_model_wrapper_context() is entered1
        # First re-entrant call while ctx1 is active populates _reentrant_cache
        with set_vllm_model_wrapper_context(
            mesh=mesh_a,
            kv_caches=kvc,
            layer_name_to_kvcache_index=lmap,
            vllm_config=cfg,
            kv_cache_bundle=bundle,
        ) as entered_re1:
            assert entered_re1 is not entered1
            assert get_vllm_model_wrapper_context() is entered_re1

        # Second re-entrant call while ctx1 is active hits _reentrant_cache
        with set_vllm_model_wrapper_context(
            mesh=mesh_a,
            kv_caches=kvc,
            layer_name_to_kvcache_index=lmap,
            vllm_config=cfg,
            kv_cache_bundle=bundle,
        ) as entered_re2:
            assert entered_re2 is entered_re1

    assert ctx_mod._vllm_model_wrapper_context is None
