# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for distributed/intra_chip.py."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_vllm_pkg = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
if "vllm.logger" not in sys.modules:
    _vlog = types.ModuleType("vllm.logger")
    _vlog.init_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
    _vlog._print_warning_once = SimpleNamespace(cache_clear=lambda: None)  # type: ignore[attr-defined]
    _vllm_pkg.logger = _vlog  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = _vlog

from vllm_torchtpu.distributed import intra_chip as ic_mod  # noqa: E402
from vllm_torchtpu.distributed.intra_chip import (  # noqa: E402
    _chip_rank_groups,
    get_intra_chip_group,
)


def _make_topology(
    cores_per_chip: int,
    device_ids: tuple[int, ...],
    chip_map: dict[int, tuple[int, int]],
) -> SimpleNamespace:
    return SimpleNamespace(
        cores_per_chip=cores_per_chip,
        device_ids=device_ids,
        chip_of=lambda r: chip_map[r],
    )


def test_intra_chip_all_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    importlib.reload(ic_mod)

    # 1. topology is None
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: None)
    assert _chip_rank_groups() is None

    # 2. cores_per_chip < 2
    topo_1core = _make_topology(1, (0, 1), {0: (0, 0), 1: (1, 0)})
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: topo_1core)
    assert _chip_rank_groups() is None

    # 3. dist not available or not initialized
    topo_2core = _make_topology(
        2,
        (0, 1, 2, 3),
        {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)},
    )
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: topo_2core)
    monkeypatch.setattr(ic_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ic_mod.dist, "is_initialized", lambda: False)
    assert _chip_rank_groups() is None

    # 4. world_size != num_devices
    monkeypatch.setattr(ic_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ic_mod.dist, "get_world_size", lambda: 2)
    assert _chip_rank_groups() is None

    # 5. Monotonic chip order (happy path)
    monkeypatch.setattr(ic_mod.dist, "get_world_size", lambda: 4)
    assert _chip_rank_groups() == [[0, 1], [2, 3]]

    # 6. Non-monotonic chip order (triggers sorting branch)
    topo_non_monotonic = _make_topology(
        2,
        (0, 1, 2, 3),
        {0: (1, 0), 1: (0, 0), 2: (1, 1), 3: (0, 1)},
    )
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: topo_non_monotonic)
    assert _chip_rank_groups() == [[1, 3], [0, 2]]

    # 7. Ragged chip groups (len(g) != cores_per_chip)
    topo_ragged = _make_topology(
        2,
        (0, 1, 2, 3),
        {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0)},
    )
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: topo_ragged)
    assert _chip_rank_groups() is None

    # 8. get_intra_chip_group when _chip_rank_groups() is None + cached _initialized
    ic_mod._initialized = False
    ic_mod._group = None
    assert get_intra_chip_group() is None
    assert get_intra_chip_group() is None

    # 9. get_intra_chip_group with warm sys.modules parallel_state
    monkeypatch.setattr(ic_mod, "get_chip_topology", lambda: topo_2core)
    monkeypatch.setattr(ic_mod.dist, "get_backend", lambda grp: "gloo")
    fake_coord = SimpleNamespace(rank_in_group=1, world_size=2)
    ps_stub = types.ModuleType("vllm.distributed.parallel_state")
    ps_stub.get_world_group = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        device_group="dev_grp", local_rank=1
    )
    ps_stub.init_model_parallel_group = (  # type: ignore[attr-defined]
        lambda groups, local_rank, backend, group_name: fake_coord
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", ps_stub)
    ic_mod._initialized = False
    ic_mod._group = None
    assert get_intra_chip_group() is fake_coord
    assert get_intra_chip_group() is fake_coord

    # 10. get_intra_chip_group with cold import of vllm.distributed.parallel_state
    vllm_dist_dir = tmp_path / "vllm" / "distributed"
    vllm_dist_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    (vllm_dist_dir / "__init__.py").write_text("", encoding="utf-8")
    (vllm_dist_dir / "parallel_state.py").write_text(
        "from types import SimpleNamespace\n"
        "def get_world_group():\n"
        "    return SimpleNamespace(device_group='dg', local_rank=0)\n"
        "def init_model_parallel_group(groups, lr, backend, group_name='intra_chip'):\n"
        "    return SimpleNamespace(rank_in_group=lr, world_size=len(groups[0]))\n",
        encoding="utf-8",
    )
    vllm_mod = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(vllm_mod, "__path__", [str(tmp_path / "vllm")], raising=False)
    vdist_mod = sys.modules.setdefault(
        "vllm.distributed", types.ModuleType("vllm.distributed")
    )
    monkeypatch.setattr(vdist_mod, "__path__", [str(vllm_dist_dir)], raising=False)
    monkeypatch.delitem(sys.modules, "vllm.distributed.parallel_state", raising=False)
    ic_mod._initialized = False
    ic_mod._group = None
    cold_grp = get_intra_chip_group()
    assert cold_grp is not None
    assert cold_grp.world_size == 2
