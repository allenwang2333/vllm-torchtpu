# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for executors/kv_block_override.py."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.executors", ("vllm_torchtpu", "executors")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg

if "vllm.logger" not in sys.modules:
    _vllm_pkg = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    _vlog = types.ModuleType("vllm.logger")
    _vlog.init_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
    _vlog._print_warning_once = types.SimpleNamespace(  # type: ignore[attr-defined]
        cache_clear=lambda: None
    )
    _vllm_pkg.logger = _vlog  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = _vlog

from vllm_torchtpu.executors import kv_block_override as kbo_mod  # noqa: E402
from vllm_torchtpu.executors.kv_block_override import (  # noqa: E402
    _NUM_BLOCKS_OVERRIDE_REL_TOL,
    _NUM_BLOCKS_OVERRIDE_TOL,
    reconcile_num_gpu_blocks_override,
)


def test_kv_block_override_all_branches() -> None:
    kbo_mod._RECONCILE_CACHE.clear()
    assert _NUM_BLOCKS_OVERRIDE_TOL == 4
    assert _NUM_BLOCKS_OVERRIDE_REL_TOL == 0.005

    # 1. Empty list and all-None lists (single-stage and multi-stage)
    assert reconcile_num_gpu_blocks_override([]) is None
    assert reconcile_num_gpu_blocks_override([None, None, None]) is None
    assert reconcile_num_gpu_blocks_override([None, None, None, None], 2) is None

    # 2. Single-stage with no Nones (cold miss + warm cache hit)
    assert reconcile_num_gpu_blocks_override([5428, 5430, 5429]) == 5428
    assert reconcile_num_gpu_blocks_override([5428, 5430, 5429]) == 5428

    # 3. Single-stage with partial Nones (hits n_none != 0 list comprehension branch)
    assert reconcile_num_gpu_blocks_override([None, 5428, 5430, None]) == 5428

    # 4. Single-stage spread > tol raises ValueError with sorted values
    with pytest.raises(ValueError, match="workers disagree"):
        reconcile_num_gpu_blocks_override([6456, 8000, 6457, None])

    # 5. Multi-stage with missing stage raises ValueError
    with pytest.raises(ValueError, match="pipeline stage"):
        reconcile_num_gpu_blocks_override([1000, 1002, None, None], 2)

    # 6. Multi-stage valid (cold miss + warm cache hit)
    assert reconcile_num_gpu_blocks_override([1000, 1002, 900, 901], 2) == 900
    assert reconcile_num_gpu_blocks_override([1000, 1002, 900, 901], 2) == 900
