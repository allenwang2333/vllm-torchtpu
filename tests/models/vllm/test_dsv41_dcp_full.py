# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for distributed/dcp.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.distributed", ("vllm_torchtpu", "distributed")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg

if "vllm_torchtpu.distributed.mesh_utils" not in sys.modules:
    _mesh_utils_stub = types.ModuleType("vllm_torchtpu.distributed.mesh_utils")
    _mesh_utils_stub.CpGroupLayout = object  # type: ignore[attr-defined]
    _mesh_utils_stub.get_cp_group_layout = lambda grp: ("layout", grp)  # type: ignore[attr-defined]
    _mesh_utils_stub.get_or_create_cp_mesh = (  # type: ignore[attr-defined]
        lambda axis_name, group: (axis_name, group)
    )
    sys.modules["vllm_torchtpu.distributed.mesh_utils"] = _mesh_utils_stub

from vllm_torchtpu.distributed import dcp  # noqa: E402


def test_dcp_group_resolution_and_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Resolution when vllm.distributed.parallel_state is in sys.modules
    grp_a = SimpleNamespace(rank_in_group=2, world_size=4)
    fake_ps = types.ModuleType("vllm.distributed.parallel_state")
    fake_ps.get_dcp_group = lambda: grp_a  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"vllm.distributed.parallel_state": fake_ps}):
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", False)
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group", None)
        assert dcp.get_dcp_group() is grp_a
        # Second call with _vllm_get_dcp_group_resolved == True
        assert dcp.get_dcp_group() is grp_a

    # 2. Resolution when vllm.distributed.parallel_state is NOT in sys.modules
    #    and import succeeds via parent vllm.distributed package
    grp_b = SimpleNamespace(rank_in_group=1, world_size=8)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_dist = types.ModuleType("vllm.distributed")
    fake_ps_import = types.ModuleType("vllm.distributed.parallel_state")
    fake_ps_import.get_dcp_group = lambda: grp_b  # type: ignore[attr-defined]
    fake_vllm_dist.parallel_state = fake_ps_import  # type: ignore[attr-defined]

    orig_get = dcp._SYS_MODULES.get
    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.distributed": fake_vllm_dist,
            "vllm.distributed.parallel_state": fake_ps_import,
        },
    ):
        monkeypatch.setattr(dcp, "_SYS_MODULES", {})
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", False)
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group", None)
        assert dcp.get_dcp_group() is grp_b
    assert orig_get is not None

    # 3. Resolution when import raises ImportError
    with patch.dict(
        sys.modules,
        {
            "vllm": None,
            "vllm.distributed": None,
            "vllm.distributed.parallel_state": None,
        },
    ):
        monkeypatch.setattr(dcp, "_SYS_MODULES", {})
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", False)
        monkeypatch.setattr(dcp, "_vllm_get_dcp_group", None)
        assert dcp.get_dcp_group() is None

    # 4. Getter raises AssertionError -> clears _vllm_get_dcp_group to None
    def _raise_uninit() -> None:
        raise AssertionError("DCP not initialized")

    monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", True)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", _raise_uninit)
    assert dcp.get_dcp_group() is None
    assert dcp._vllm_get_dcp_group is None
    # Subsequent call takes the `if getter is not None:` False branch directly
    assert dcp.get_dcp_group() is None


def test_dcp_rank_world_size_layout_and_mesh_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Uninitialized (group is None)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1

    # Cache miss then cache hit starting from get_dcp_rank()
    grp1 = SimpleNamespace(rank_in_group=3, world_size=4)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: grp1)
    assert dcp.get_dcp_rank() == 3
    assert dcp.get_dcp_rank() == 3
    assert dcp.get_dcp_world_size() == 4

    # Cache miss then cache hit starting from get_dcp_world_size()
    grp2 = SimpleNamespace(rank_in_group=5, world_size=8)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: grp2)
    assert dcp.get_dcp_world_size() == 8
    assert dcp.get_dcp_world_size() == 8
    assert dcp.get_dcp_rank() == 5

    # get_dcp_group_layout cache miss and cache hit
    layout_calls = 0

    def _fake_layout(g: object) -> tuple[str, object]:
        nonlocal layout_calls
        layout_calls += 1
        return ("layout", g)

    monkeypatch.setattr(dcp, "get_cp_group_layout", _fake_layout)
    assert dcp.get_dcp_group_layout() == ("layout", grp2)
    assert dcp.get_dcp_group_layout() == ("layout", grp2)
    assert layout_calls == 1

    # get_or_create_dcp_mesh cache miss and cache hit for default, None, and custom axis
    mesh_calls = 0

    def _fake_mesh(axis_name: str, group: object) -> tuple[str, object]:
        nonlocal mesh_calls
        mesh_calls += 1
        return (axis_name, group)

    monkeypatch.setattr(dcp, "get_or_create_cp_mesh", _fake_mesh)
    assert dcp.get_or_create_dcp_mesh() == ("dcp", grp2)
    assert dcp.get_or_create_dcp_mesh(None) == ("dcp", grp2)
    assert mesh_calls == 1
    assert dcp.get_or_create_dcp_mesh("custom_dcp") == ("custom_dcp", grp2)
    assert dcp.get_or_create_dcp_mesh("custom_dcp") == ("custom_dcp", grp2)
    assert mesh_calls == 2
