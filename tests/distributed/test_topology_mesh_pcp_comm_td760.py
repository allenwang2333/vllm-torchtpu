# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.60 topology, mesh, PCP & communicator optimizations."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed import (
    chip_topology,
    mesh_utils,
    pcp,
    pcp_rank_order,
    tpu_communicator,
)


def test_chip_topology_td760_full_coverage(monkeypatch):
    """Verify O(1) _device_to_chip lookup and 100% coverage of chip_topology.py."""
    orig_get_chip_topology = chip_topology.get_chip_topology
    orig_get_chip_topology.cache_clear()
    try:
        topo = chip_topology.ChipTopology([[4, 5], [8, 9]], [9, 5, 8, 4])
        assert topo._device_to_chip == {4: (0, 0), 5: (0, 1), 8: (1, 0), 9: (1, 1)}
        assert [topo.chip_of(r) for r in range(4)] == [(1, 1), (0, 1), (1, 0), (0, 0)]
        assert "2 chip(s) x 2 core(s)" in topo.describe()

        with pytest.raises(ValueError, match="outside"):
            topo.chip_of(-1)
        with pytest.raises(RuntimeError, match="is on no chip"):
            chip_topology.ChipTopology([[4, 5]], [9]).chip_of(0)

        # get_chip_topology: exception, empty, missing coords, mismatched sizes, valid
        monkeypatch.setattr(jax, "devices", Mock(side_effect=RuntimeError("boom")))
        assert chip_topology.get_chip_topology() is None
        orig_get_chip_topology.cache_clear()

        monkeypatch.setattr(jax, "devices", lambda: [])
        assert chip_topology.get_chip_topology() is None
        orig_get_chip_topology.cache_clear()

        monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(id=0)])
        assert chip_topology.get_chip_topology() is None
        orig_get_chip_topology.cache_clear()

        monkeypatch.setattr(
            jax,
            "devices",
            lambda: [
                SimpleNamespace(id=0, coords=(0,)),
                SimpleNamespace(id=1, coords=(0,)),
                SimpleNamespace(id=2, coords=(1,)),
            ],
        )
        assert chip_topology.get_chip_topology() is None
        assert chip_topology.hierarchical_moe_split(4, 0) is None
        orig_get_chip_topology.cache_clear()

        monkeypatch.setattr(
            jax,
            "devices",
            lambda: [
                SimpleNamespace(id=9, coords=(1, 0)),
                SimpleNamespace(id=5, coords=(0, 0)),
                SimpleNamespace(id=8, coords=(1, 0)),
                SimpleNamespace(id=4, coords=(0, 0)),
            ],
        )
        valid_topo = chip_topology.get_chip_topology()
        assert valid_topo is not None
        assert chip_topology.get_chip_topology() is valid_topo
        assert chip_topology.hierarchical_moe_split(4, 0) == (2, 1, 2, 1)
        assert chip_topology.hierarchical_moe_split(2, 0) is None

        monkeypatch.setattr(
            chip_topology,
            "get_chip_topology",
            lambda: chip_topology.ChipTopology([[0], [1]], [0, 1]),
        )
        assert chip_topology.hierarchical_moe_split(2, 0) is None
    finally:
        orig_get_chip_topology.cache_clear()


def test_mesh_utils_td760_full_coverage(monkeypatch):
    """Verify 100% statement and branch coverage of mesh_utils.py."""
    from torch_tpu._internal.distributed import tpu_distributed

    monkeypatch.setattr(mesh_utils, "_LAYOUT_CACHE", {})
    monkeypatch.setattr(mesh_utils, "_MESH_CACHE", {})

    # _get_current_global_rank
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    assert mesh_utils._get_current_global_rank() == 0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    assert mesh_utils._get_current_global_rank() == 3

    # _get_tpu_global_device_id
    monkeypatch.setattr(tpu_distributed, "global_device_id", lambda: 5)
    assert mesh_utils._get_tpu_global_device_id() == 5
    monkeypatch.setattr(
        tpu_distributed, "global_device_id", Mock(side_effect=RuntimeError)
    )
    monkeypatch.setattr(jax, "local_devices", lambda: [SimpleNamespace(id=11)])
    assert mesh_utils._get_tpu_global_device_id() == 11
    monkeypatch.setattr(jax, "local_devices", lambda: [])
    assert mesh_utils._get_tpu_global_device_id() == 0
    monkeypatch.setattr(jax, "local_devices", Mock(side_effect=RuntimeError))
    assert mesh_utils._get_tpu_global_device_id() == 0

    # _collect_rank_to_device_id
    grp = SimpleNamespace(
        world_size=2, cpu_group=object(), ranks=[0, 1], rank_in_group=1
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10), None]),
    )
    with pytest.raises(RuntimeError, match="empty entry"):
        mesh_utils._collect_rank_to_device_id(grp, 1, 20)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10)]),
    )
    with pytest.raises(RuntimeError, match="missing="):
        mesh_utils.get_cp_group_layout(grp)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10), (1, 20)]),
    )
    assert mesh_utils.get_cp_group_layout(None).world_size == 1
    layout = mesh_utils.get_cp_group_layout(grp)
    assert mesh_utils.get_cp_group_layout(grp) is layout
    assert (layout.device_id, layout.prev_rank_in_group, layout.next_rank_in_group) == (
        20,
        0,
        0,
    )

    # get_or_create_cp_mesh: missing device error, build mesh, and cache hit
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(id=10)])
    with pytest.raises(RuntimeError, match="Cannot build the 'dcp' mesh"):
        mesh_utils.get_or_create_cp_mesh("dcp", grp)

    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [
            SimpleNamespace(id=10, coords=(0, 0), core_on_chip=0),
            SimpleNamespace(id=20, coords=(0, 0), core_on_chip=1),
        ],
    )
    monkeypatch.setattr(
        jax.sharding,
        "Mesh",
        lambda devs, axis_names: SimpleNamespace(devices=devs, axis_names=axis_names),
    )
    mesh = mesh_utils.get_or_create_cp_mesh("dcp", grp)
    assert mesh.axis_names == ("dcp",)
    assert mesh_utils.get_or_create_cp_mesh("dcp", grp) is mesh


def test_pcp_td760_full_coverage(monkeypatch):
    """Verify single get_pcp_group call in get_or_create_pcp_mesh and 100% coverage."""
    from torch_tpu._internal import pallas as pallas_mod

    monkeypatch.setattr(pcp, "_MESH_CACHE", {})
    monkeypatch.setattr(mesh_utils, "_LAYOUT_CACHE", {})

    # get_pcp_group, get_pcp_rank, get_pcp_world_size
    monkeypatch.setattr(
        parallel_state, "get_pcp_group", Mock(side_effect=AssertionError)
    )
    assert pcp.get_pcp_group() is None
    assert pcp.get_pcp_rank() == 0
    assert pcp.get_pcp_world_size() == 1

    grp2 = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        ranks=[0, 1],
        all_gather=lambda t, dim=0: [t, t + 1]
        if dim == 1
        else torch.cat([t, t], dim=dim),
        all_reduce=lambda t: t * 2,
    )
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: None)
    t_noncontig = torch.ones(2, 3).t()
    assert pcp.all_gather_equal_tokens(t_noncontig) is t_noncontig
    assert pcp.all_reduce_sum(t_noncontig) is t_noncontig

    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: grp2)
    assert pcp.get_pcp_rank() == 1
    assert pcp.get_pcp_world_size() == 2
    assert pcp.all_gather_equal_tokens(t_noncontig, dim=0).shape == (6, 2)
    assert pcp.all_gather_equal_tokens(torch.ones(2, 3), dim=1).shape == (2, 6)
    assert pcp.all_reduce_sum(t_noncontig).shape == (3, 2)
    assert pcp.all_reduce_sum(torch.ones(2, 3)).shape == (2, 3)

    monkeypatch.setattr(
        pcp, "get_cp_group_layout", lambda g: pcp.PcpGroupLayout((0,), (0,), 0, 1)
    )
    assert pcp.get_pcp_group_layout().world_size == 1

    # Duplicate axis_name error
    with pytest.raises(ValueError, match="must be distinct"):
        pcp.get_or_create_pcp_mesh(axis_name="pcp", tp_axis_name="pcp")

    # group is None branch in get_or_create_pcp_mesh
    get_group_calls = 0

    def _count_none_group():
        nonlocal get_group_calls
        get_group_calls += 1
        return None

    monkeypatch.setattr(pcp, "get_pcp_group", _count_none_group)
    monkeypatch.setattr(jax, "local_devices", lambda: [SimpleNamespace(id=0)])
    monkeypatch.setattr(
        jax.sharding,
        "Mesh",
        lambda devs, axis_names: SimpleNamespace(devices=devs, axis_names=axis_names),
    )
    m_none = pcp.get_or_create_pcp_mesh()
    # Verify get_pcp_group() was called only once!
    assert get_group_calls == 1
    assert m_none.axis_names == ("pcp",)
    assert pcp.get_or_create_pcp_mesh() is m_none

    # group is not None branches in get_or_create_pcp_mesh
    pcp._MESH_CACHE.clear()
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: grp2)
    monkeypatch.setattr(
        pcp, "get_cp_group_layout", lambda g: pcp.PcpGroupLayout((0, 1), (10, 20), 0, 2)
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    with pytest.raises(NotImplementedError, match="world_size=2"):
        pcp.get_or_create_pcp_mesh()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        pcp, "get_cp_group_layout", lambda g: pcp.PcpGroupLayout((2, 3), (10, 20), 0, 2)
    )
    with pytest.raises(RuntimeError, match=r"ranks 0\.\.N-1"):
        pcp.get_or_create_pcp_mesh()

    monkeypatch.setattr(
        pcp, "get_cp_group_layout", lambda g: pcp.PcpGroupLayout((0, 1), (10, 20), 0, 2)
    )
    monkeypatch.setattr(pallas_mod, "get_pallas_mesh", lambda **kw: ("pallas_mesh", kw))
    m_pcp = pcp.get_or_create_pcp_mesh()
    assert m_pcp == ("pallas_mesh", {"axis_names": ("pcp",)})

    # With tp_axis_name
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        pcp, "get_cp_group_layout", lambda g: pcp.PcpGroupLayout((0, 2), (10, 20), 0, 2)
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, ranks=[1, 0]),
    )
    with pytest.raises(RuntimeError, match="TP group coordinates"):
        pcp.get_or_create_pcp_mesh(tp_axis_name="tp")

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, ranks=[0, 1]),
    )
    m_pcp_tp = pcp.get_or_create_pcp_mesh(tp_axis_name="tp")
    assert m_pcp_tp == (
        "pallas_mesh",
        {"axis_names": ("pcp", "tp"), "mesh_shape": (2, 2)},
    )


def test_pcp_rank_order_td760_full_coverage(monkeypatch):
    """Verify single orig_lists construction and 100% coverage of pcp_rank_order.py."""

    def _cfg(pcp_sz=4, tp_sz=2, dp_sz=1, pp_sz=1, nnodes=1):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_sz,
                tensor_parallel_size=tp_sz,
                data_parallel_size=dp_sz,
                pipeline_parallel_size=pp_sz,
                nnodes=nnodes,
            )
        )

    assert pcp_rank_order.resolve_pcp_topology_order(_cfg(pcp_sz=1)) is None
    assert pcp_rank_order.resolve_pcp_topology_order(_cfg(nnodes=2)) is None
    with pytest.raises(RuntimeError, match="pipeline_parallel_size"):
        pcp_rank_order.resolve_pcp_topology_order(_cfg(pp_sz=2))
    with pytest.raises(RuntimeError, match="data_parallel_size"):
        pcp_rank_order.resolve_pcp_topology_order(_cfg(dp_sz=2))

    monkeypatch.setitem(vars(tpu_envs), "TPU_PCP_TOPOLOGY_AWARE_MESH", False)
    assert pcp_rank_order.resolve_pcp_topology_order(_cfg()) is None

    monkeypatch.setitem(vars(tpu_envs), "TPU_PCP_TOPOLOGY_AWARE_MESH", True)
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="not available"):
        pcp_rank_order.resolve_pcp_topology_order(_cfg())

    bad_mesh = np.array([0, 1, 2, 3, 4, 5, 6, 6]).reshape(1, 4, 2)
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(topology_aware_mesh=lambda _s: bad_mesh),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="not a permutation"):
        pcp_rank_order.resolve_pcp_topology_order(_cfg())

    good_mesh = np.array([[[0, 1], [6, 7], [4, 5], [2, 3]]])
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(topology_aware_mesh=lambda _s: good_mesh),
        raising=False,
    )
    orders = pcp_rank_order.resolve_pcp_topology_order(_cfg())
    assert orders == {
        "pcp": [[0, 6, 4, 2], [1, 7, 5, 3]],
        "tp": [[0, 1], [6, 7], [4, 5], [2, 3]],
    }

    # pcp_topology_order context manager
    orig_new_group = torch.distributed.new_group
    monkeypatch.setattr(torch.distributed, "new_group", lambda ranks=None: None)
    with (
        pytest.raises(RuntimeError, match="no sort_ranks parameter"),
        pcp_rank_order.pcp_topology_order(orders),
    ):
        pass

    sort_flags = []

    def _fake_new_group(ranks=None, sort_ranks=True, **kw):
        sort_flags.append(sort_ranks)

    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group)
    monkeypatch.setattr(
        parallel_state,
        "init_model_parallel_group",
        lambda group_ranks, local_rank, backend, **kw: [
            torch.distributed.new_group(r) for r in group_ranks
        ],
    )
    with pcp_rank_order.pcp_topology_order(orders, preserve_dcp_order=True):
        parallel_state.init_model_parallel_group(
            [[0, 2, 4, 6], [1, 3, 5, 7]], 0, "gloo", group_name="pcp"
        )
        parallel_state.init_model_parallel_group(
            [[0, 8, 1, 9]], 0, "gloo", group_name="dcp"
        )
        parallel_state.init_model_parallel_group(
            [[0, 1, 8, 9]], 0, "gloo", group_name="dcp"
        )
        parallel_state.init_model_parallel_group([[0]], 0, "gloo", group_name="dp")
    assert sort_flags == [False, False, False, True, True]

    with pcp_rank_order.pcp_topology_order(None, preserve_dcp_order=False):
        parallel_state.init_model_parallel_group([[0]], 0, "gloo", group_name="dp")

    monkeypatch.setattr(torch.distributed, "new_group", orig_new_group)

    # verify_pcp_topology_order
    pcp_rank_order.verify_pcp_topology_order(None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        parallel_state,
        "get_pcp_group",
        lambda: SimpleNamespace(ranks=[0, 6, 4, 2]),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1]),
    )
    pcp_rank_order.verify_pcp_topology_order(orders)

    with pytest.raises(RuntimeError, match="No group accessor"):
        pcp_rank_order.verify_pcp_topology_order({"ep": [[0, 1]]})
    with pytest.raises(RuntimeError, match="does not appear"):
        pcp_rank_order.verify_pcp_topology_order({"pcp": [[1, 7, 5, 3]]})
    with pytest.raises(RuntimeError, match="did not adopt"):
        pcp_rank_order.verify_pcp_topology_order({"pcp": [[0, 2, 4, 6]]})


def test_tpu_communicator_td760_full_coverage(monkeypatch):
    """Verify 100% statement and branch coverage of tpu_communicator.py."""
    cpu_group = object()

    def _init_false(self, *args, **kwargs):
        self.use_all2all = False
        self.cpu_group = cpu_group

    def _init_true(self, *args, **kwargs):
        self.use_all2all = True
        self.cpu_group = cpu_group

    monkeypatch.setattr(tpu_communicator.DeviceCommunicatorBase, "__init__", _init_true)
    monkeypatch.setattr(
        tpu_communicator, "TpuNullAll2AllManager", lambda cg: ("mgr", cg)
    )
    c_true = tpu_communicator.TpuDeviceCommunicator()
    assert c_true.all2all_manager == ("mgr", cpu_group)

    monkeypatch.setattr(
        tpu_communicator.DeviceCommunicatorBase, "__init__", _init_false
    )
    comm = tpu_communicator.TpuDeviceCommunicator()

    with pytest.raises(NotImplementedError, match="extra_tensors"):
        comm.dispatch_router_logits(
            torch.ones(2, 4), torch.ones(2, 4), extra_tensors=[]
        )
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        comm.dispatch(
            torch.ones(2, 4), torch.ones(2, 2), torch.ones(2, 2), extra_tensors=[]
        )

    grp1 = SimpleNamespace(world_size=1)
    grp2 = SimpleNamespace(
        world_size=2,
        all_gather=lambda x, dim=0: torch.cat([x, x + 1], dim=dim),
        reduce_scatter=lambda x, dim=0: x[: x.shape[0] // 2],
    )
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: grp1)
    monkeypatch.setattr(tpu_communicator, "get_ep_group", lambda: grp2)

    h = torch.ones(4, 4)
    r_full = torch.ones(4, 8)
    r_sliced = torch.ones(1, 8)
    w = torch.ones(4, 2)
    idx = torch.ones(4, 2, dtype=torch.int32)

    # world_size == 1 (non-SP uses dp_group=grp1)
    assert comm.dispatch_router_logits(h, r_full) == (h, r_full)
    assert comm.dispatch(h, w, idx) == (h, w, idx)
    assert comm.combine(h) is h

    # world_size == 2 (SP uses ep_group=grp2): same=True and same=False
    h_sp, r_sp_same = comm.dispatch_router_logits(h, h, is_sequence_parallel=True)
    assert r_sp_same is h_sp and h_sp.shape == (8, 4)
    h_sp2, r_sp_diff = comm.dispatch_router_logits(h, r_full, is_sequence_parallel=True)
    assert h_sp2.shape == (8, 4) and r_sp_diff.shape == (8, 8)

    # non-SP with dp_group=grp2 and tp_sliced_logits=True (exercises 76->78 branch!)
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: grp2)
    h_dp, r_dp_sliced = comm.dispatch_router_logits(
        h, r_sliced, is_sequence_parallel=False
    )
    assert h_dp.shape == (8, 4) and r_dp_sliced is r_sliced

    # dispatch and combine with world_size == 2
    d_h, d_w, d_i = comm.dispatch(h, w, idx, is_sequence_parallel=False)
    assert d_h.shape == (8, 4) and d_w.shape == (8, 2) and d_i.shape == (8, 2)
    assert comm.combine(d_h, is_sequence_parallel=False).shape == (4, 4)
