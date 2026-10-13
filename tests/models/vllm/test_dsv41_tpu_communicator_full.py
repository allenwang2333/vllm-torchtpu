# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for distributed.tpu_communicator."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.distributed", ("vllm_torchtpu", "distributed")),
):
    _pkg = sys.modules.get(_pkg_name)
    if _pkg is None:
        _pkg = types.ModuleType(_pkg_name)
        sys.modules[_pkg_name] = _pkg
    _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
sys.modules.pop("vllm_torchtpu.distributed.tpu_communicator", None)


class _StubAll2AllManagerBase:
    def __init__(self, cpu_group: Any = None) -> None:
        self.cpu_group = cpu_group


class _StubDeviceCommunicatorBase:
    def __init__(
        self,
        cpu_group: Any = None,
        use_all2all: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.cpu_group = cpu_group
        self.use_all2all = use_all2all


if "vllm.distributed.device_communicators.base_device_communicator" not in sys.modules:
    for _mod_name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        "vllm.distributed.device_communicators.base_device_communicator",
        "vllm.distributed.parallel_state",
    ):
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)
    _base_comm = sys.modules[
        "vllm.distributed.device_communicators.base_device_communicator"
    ]
    _base_comm.All2AllManagerBase = _StubAll2AllManagerBase  # type: ignore[attr-defined]
    _base_comm.DeviceCommunicatorBase = _StubDeviceCommunicatorBase  # type: ignore[attr-defined]
    _pstate = sys.modules["vllm.distributed.parallel_state"]
    _pstate.get_dp_group = lambda: None  # type: ignore[attr-defined]
    _pstate.get_ep_group = lambda: None  # type: ignore[attr-defined]

import importlib  # noqa: E402

from vllm_torchtpu.distributed import tpu_communicator  # noqa: E402

pytestmark = pytest.mark.cpu_test


class _FakeGroup:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.all_gather_calls: list[tuple[torch.Tensor, int]] = []
        self.reduce_scatter_calls: list[tuple[torch.Tensor, int]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.all_gather_calls.append((tensor, dim))
        return torch.cat([tensor, tensor + 10], dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.reduce_scatter_calls.append((tensor, dim))
        half = tensor.shape[dim] // 2
        return tensor[:half]


def test_tpu_communicator_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    comm = importlib.reload(tpu_communicator)

    def _fake_init(
        instance: Any,
        cpu_group: Any = None,
        use_all2all: bool = False,
    ) -> None:
        instance.cpu_group = cpu_group
        instance.use_all2all = use_all2all

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", _fake_init)

    # 1. __init__ with use_all2all=False and use_all2all=True
    cpu_grp = object()
    c_no_a2a = comm.TpuDeviceCommunicator(cpu_grp, use_all2all=False)
    assert not hasattr(c_no_a2a, "all2all_manager")

    c_a2a = comm.TpuDeviceCommunicator(cpu_grp, use_all2all=True)
    assert isinstance(c_a2a.all2all_manager, comm.TpuNullAll2AllManager)

    # 2. _token_group helper coverage for both False and True
    dp1 = _FakeGroup(world_size=1)
    ep2 = _FakeGroup(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp1)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep2)
    assert comm.TpuDeviceCommunicator._token_group(False) is dp1
    assert comm.TpuDeviceCommunicator._token_group(True) is ep2

    h = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    r_full = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    r_sliced = torch.tensor([[0.5, 0.6]])
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    # 3. extra_tensors error paths
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        c_no_a2a.dispatch_router_logits(h, r_full, extra_tensors=[h])
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        c_no_a2a.dispatch(h, w, ids, extra_tensors=[h])

    # 4. world_size == 1 early returns across all 3 methods
    h_out, r_out = c_no_a2a.dispatch_router_logits(
        h, r_full, is_sequence_parallel=False
    )
    assert h_out is h and r_out is r_full
    dh, dw, dids = c_no_a2a.dispatch(h, w, ids, is_sequence_parallel=False)
    assert dh is h and dw is w and dids is ids
    ch = c_no_a2a.combine(h, is_sequence_parallel=False)
    assert ch is h

    # 5. world_size > 1 paths on dispatch_router_logits
    dp2 = _FakeGroup(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp2)

    # 5a. router_logits is hidden_states alias path
    h_alias, r_alias = c_no_a2a.dispatch_router_logits(h, h, is_sequence_parallel=True)
    assert h_alias is r_alias
    assert len(ep2.all_gather_calls) == 1

    # 5b. is_sequence_parallel=True distinct router_logits path
    h_sp, r_sp = c_no_a2a.dispatch_router_logits(h, r_full, is_sequence_parallel=True)
    assert h_sp.shape == (4, 2) and r_sp.shape == (4, 2)
    assert len(ep2.all_gather_calls) == 3

    # 5c. is_sequence_parallel=False and router_logits.shape[0] >= hidden.shape[0]
    h_dp_full, r_dp_full = c_no_a2a.dispatch_router_logits(
        h, r_full, is_sequence_parallel=False
    )
    assert h_dp_full.shape == (4, 2) and r_dp_full.shape == (4, 2)
    assert len(dp2.all_gather_calls) == 2

    # 5d. is_sequence_parallel=False and router_logits.shape[0] < hidden.shape[0]
    h_dp_sl, r_dp_sl = c_no_a2a.dispatch_router_logits(
        h, r_sliced, is_sequence_parallel=False
    )
    assert h_dp_sl.shape == (4, 2) and r_dp_sl is r_sliced
    assert len(dp2.all_gather_calls) == 3

    # 6. world_size > 1 paths on dispatch and combine
    dh2, dw2, dids2 = c_no_a2a.dispatch(h, w, ids, is_sequence_parallel=True)
    assert dh2.shape == (4, 2) and dw2.shape == (4, 2) and dids2.shape == (4, 2)

    comb_out = c_no_a2a.combine(dh2, is_sequence_parallel=True)
    assert comb_out.shape == (2, 2)
    assert len(ep2.reduce_scatter_calls) == 1
