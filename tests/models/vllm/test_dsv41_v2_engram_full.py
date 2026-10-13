# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v2/engram.py."""

import importlib
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _pkg_name, _rel in (
    ("vllm_torchtpu", "vllm_torchtpu"),
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
        "vllm_torchtpu.models.vllm.deepseek_v41.v2",
        "vllm_torchtpu/models/vllm/deepseek_v41/v2",
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
        if torch.compiler.is_compiling():
            return self._cache is not None
        kv_cache = self.swa_cache_module.kv_cache
        if not kv_cache.numel():
            self._cache = None
            self._kv_cache_ref = None
            return False
        self.block_size = kv_cache.shape[1]
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size, dtype=torch.int32
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True


for _mod_name, _attrs in (
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache",
        {"write_cache": MagicMock()},
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.common.engram",
        {
            "_INVALID_SCALE": 255,
            "_UNIT_SCALE": 127,
            "DPSharedEngramStorage": object,
            "Engram": object,
            "NgramHashState": _FakeCommonNgramHashState,
            "ParallelEngramEmbedding": object,
            "_gather_engram_rows": MagicMock(),
            "_load_local_rows": MagicMock(),
            "_unpack_bytes": MagicMock(),
            "can_share_engram_tables": MagicMock(),
            "dequant_fp8_rows": MagicMock(),
            "register_row_slice": MagicMock(),
            "upstream": SimpleNamespace(),
        },
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

v2_engram = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v2.engram")
NgramHashState = v2_engram.NgramHashState


def test_v2_engram_write_cache_and_ensure_cache():
    state = NgramHashState()
    state.use_slot_cache = True
    state.block_size = 16
    state._cache = None
    state._kv_cache_ref = None

    # 1. Compiling branch delegates to super().ensure_cache()
    with patch("torch.compiler.is_compiling", return_value=True):
        assert state.ensure_cache() is False

    # 2. Empty kv_cache returns False
    empty_kv = torch.empty((0, 16, 128), dtype=torch.bfloat16)
    state.swa_cache_module = SimpleNamespace(kv_cache=empty_kv)
    assert state.ensure_cache() is False
    assert state._cache is None

    # 3. Initial allocation with non-empty kv_cache
    kv = torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    state.swa_cache_module = SimpleNamespace(kv_cache=kv)
    assert state.ensure_cache() is True
    assert state._cache is not None and state._cache.shape == (4 * 16,)
    first_cache = state._cache

    # 4. Steady-state fast path: ref() is kv and block_size == kv.shape[1]
    assert state.ensure_cache() is True
    assert state._cache is first_cache

    # 5. Block-size mismatch triggers slow path reallocation
    kv_grown = torch.zeros((4, 32, 128), dtype=torch.bfloat16)
    state.swa_cache_module = SimpleNamespace(kv_cache=kv_grown)
    assert state.ensure_cache() is True
    assert state.block_size == 32
    assert state._cache.shape == (4 * 32,)

    # 6. Direct write_cache staticmethod and instance binding
    slots = torch.tensor([0, 1], dtype=torch.int32)
    vals = torch.tensor([10, 20], dtype=torch.int32)
    valid = torch.tensor([True, True])
    state.write_cache(state._cache, slots, vals, valid)
    NgramHashState.write_cache(state._cache, slots, vals, valid)
