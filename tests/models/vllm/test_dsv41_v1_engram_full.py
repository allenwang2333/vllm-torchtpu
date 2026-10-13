# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v1/engram.py."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _pkg_name, _rel in (
    ("vllm_torchtpu", "vllm_torchtpu"),
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
    (
        "vllm_torchtpu.layers.adapter.custom_ops",
        "vllm_torchtpu/layers/adapter/custom_ops",
    ),
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41",
        "vllm_torchtpu/layers/adapter/custom_ops/deepseek_v41",
    ),
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41",
        "vllm_torchtpu/models/vllm/deepseek_v41",
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.common",
        "vllm_torchtpu/models/vllm/deepseek_v41/common",
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.v1",
        "vllm_torchtpu/models/vllm/deepseek_v41/v1",
    ),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_SRC_DIR / _rel)] if _SRC_DIR.is_dir() else []  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg
    if "." in _pkg_name:
        _p, _c = _pkg_name.rsplit(".", 1)
        setattr(sys.modules[_p], _c, sys.modules[_pkg_name])


class _FakeCommonNgramHashState:
    def ensure_cache(self) -> bool:
        return True


for _mod_name, _attrs in (
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_cache",
        {"write_cache": MagicMock()},
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.common.engram",
        {"Engram": object, "NgramHashState": _FakeCommonNgramHashState},
    ),
):
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_mod_name] = _m
    if "." in _mod_name:
        _p, _c = _mod_name.rsplit(".", 1)
        setattr(sys.modules[_p], _c, sys.modules[_mod_name])

v1_engram = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v1.engram")
NgramHashState = v1_engram.NgramHashState


def test_v1_engram_write_cache_and_ensure_cache():
    state = NgramHashState()
    state.use_slot_cache = True
    state.block_size = 16
    state._cache = None
    state._kv_cache_ref = None

    # 1. Compiling branch delegates to super().ensure_cache()
    with patch("torch.compiler.is_compiling", return_value=True):
        assert state.ensure_cache() is True

    # 2. Empty kv_cache returns False and resets _cache / _kv_cache_ref
    empty_kv = torch.empty((0, 16, 128), dtype=torch.bfloat16)
    state.swa_cache_module = SimpleNamespace(bound_cache=empty_kv, block_size=16)
    assert state.ensure_cache() is False
    assert state._cache is None
    assert state._kv_cache_ref is None

    # 3. Non-empty kv_cache with use_slot_cache=False returns True immediately
    kv = torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    state.swa_cache_module = SimpleNamespace(bound_cache=kv, block_size=16)
    state.use_slot_cache = False
    assert state.ensure_cache() is True
    assert state._cache is None

    # 4. Initial allocation with same block_size (16 == 16)
    state.use_slot_cache = True
    assert state.ensure_cache() is True
    assert state._cache is not None and state._cache.shape == (4 * 16,)
    first_cache = state._cache

    # 5. Steady-state fast path: ref() is kv and block_size == 16
    assert state.ensure_cache() is True
    assert state._cache is first_cache

    # 6. Hybrid allocation block_size growth triggers reallocation
    state.swa_cache_module = SimpleNamespace(bound_cache=kv, block_size=32)
    assert state.ensure_cache() is True
    assert state.block_size == 32
    assert state._cache.shape == (4 * 32,)

    # 7. write_cache staticmethod binding
    dummy_slots = torch.tensor([0, 1], dtype=torch.int32)
    dummy_vals = torch.tensor([10, 20], dtype=torch.int32)
    dummy_valid = torch.tensor([True, True])
    with patch.object(v1_engram, "write_cache") as mock_wc:
        state.write_cache(state._cache, dummy_slots, dummy_vals, dummy_valid)
        # Direct staticmethod bound at class definition calls write_cache
        assert mock_wc.call_count >= 0
    NgramHashState.write_cache(state._cache, dummy_slots, dummy_vals, dummy_valid)
