# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DCP-side accessors over `mesh_utils.py`'s shared layout/mesh helpers.

Both helpers are one-liners, and both are one-liners whose failure mode is
silent: a mesh built over the wrong group, or over the right group under the
wrong axis name, still builds. It just attends the wrong shard. So what is
worth pinning is not the arithmetic but the plumbing -- which group is asked
for, and which axis name comes out.

No hardware and no JAX: `get_or_create_cp_mesh` is faked out, because the
question here is what it gets called with.
"""

import pytest

from vllm_torchtpu.distributed import dcp, mesh_utils, pcp


class _FakeGroup:
    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group


def test_mesh_helper_uses_the_dcp_group_and_axis(monkeypatch):
    """`get_or_create_dcp_mesh` must not silently build a PCP mesh.

    Forwarding the wrong group or the wrong axis name is the failure this
    guards: under `tp=N, dcp=N` the two CP groups cover the same devices, so
    the mesh builds either way, and the only symptom is that the DCP partition
    specs name an axis the mesh does not have.
    """
    seen = {}

    def _fake_mesh(axis_name, group):
        seen["axis_name"] = axis_name
        seen["group"] = group
        return "mesh"

    group = _FakeGroup([4, 5, 6, 7], rank_in_group=1)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(dcp, "get_or_create_cp_mesh", _fake_mesh)

    assert dcp.get_or_create_dcp_mesh() == "mesh"
    assert seen["axis_name"] == dcp.DCP_AXIS_NAME
    assert seen["group"] is group


def test_an_explicit_axis_name_wins_over_the_default(monkeypatch):
    """The default is a default, not a constant folded into the call."""
    seen = {}
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: _FakeGroup([0, 1]))
    monkeypatch.setattr(
        dcp,
        "get_or_create_cp_mesh",
        lambda axis_name, group: seen.setdefault("axis_name", axis_name),
    )

    dcp.get_or_create_dcp_mesh(axis_name="something_else")
    assert seen["axis_name"] == "something_else"


def test_layout_describes_the_group_it_is_given(monkeypatch):
    """The shared layout helper must follow its argument, not the PCP group.

    Without this, a DCP-only run -- where the PCP group is trivial -- would get
    a world_size=1 layout and build a single-device mesh.
    """
    pcp_group = _FakeGroup([0])
    dcp_group = _FakeGroup([0, 1, 2, 3], rank_in_group=2)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(
        mesh_utils,
        "_collect_rank_to_device_id",
        lambda group, *_: {r: 10 + r for r in group.ranks},
    )
    monkeypatch.setattr(mesh_utils, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 12)

    # The no-argument call is the pre-existing PCP contract, and it still
    # holds -- extracting the shared helper did not change pcp.py's surface.
    assert pcp.get_pcp_group_layout().world_size == 1

    layout = dcp.get_dcp_group_layout()
    assert layout.world_size == 4
    assert layout.rank_in_group == 2
    assert layout.device_ids == (10, 11, 12, 13)
    assert layout.device_id == 12


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_the_plain_accessors_survive_an_uninitialized_group(monkeypatch, world_size):
    """Rank and world size have to answer before distributed init, too.

    `get_dcp_group` returns None then, and callers read these two without
    checking -- so the defaults are load-bearing, not defensive.
    """
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1

    group = _FakeGroup(list(range(world_size)), rank_in_group=world_size - 1)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    assert dcp.get_dcp_rank() == world_size - 1
    assert dcp.get_dcp_world_size() == world_size


class _FakeDevice:
    def __init__(self, device_id):
        self.id = device_id


def _world_mesh_setup(monkeypatch, my_ids, all_rows, world_size):
    """Fake one rank of a world whose DCP groups are `all_rows` (device ids)."""
    import types

    monkeypatch.setattr(dcp, "get_dcp_group", lambda: _FakeGroup(range(len(my_ids))))
    monkeypatch.setattr(dcp, "_torch_world_size", lambda: world_size)
    monkeypatch.setattr(
        dcp,
        "get_dcp_group_layout",
        lambda: mesh_utils.CpGroupLayout(
            ranks=tuple(range(len(my_ids))),
            device_ids=tuple(my_ids),
            rank_in_group=0,
            world_size=len(my_ids),
        ),
    )

    def _gather(out, obj, group=None):
        for i in range(len(out)):
            out[i] = all_rows[i // len(my_ids)]

    import torch

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(
        ps, "get_world_group", lambda: types.SimpleNamespace(cpu_group=None)
    )
    import jax

    devices = [_FakeDevice(i) for i in range(world_size)]
    monkeypatch.setattr(jax, "devices", lambda: devices)
    created = {}

    class _Mesh:
        def __init__(self, devs, axis_names):
            created["ids"] = [[d.id for d in row] for row in devs]
            created["axis_names"] = axis_names

    monkeypatch.setattr(jax.sharding, "Mesh", _Mesh)
    dcp._WORLD_MESH_CACHE.clear()
    return created


def test_a_dcp_group_smaller_than_the_world_gets_a_world_wide_mesh(monkeypatch):
    """TorchTPU compiles op-local programs over the whole world, so a mesh over
    only the DCP group fails at compile time ("tile_assignment should have 8
    devices but has 4"). Each row must be one DCP group, in its rank order, and
    the DCP axis must be the minor one so P("dcp") shards within a group."""
    rows = [(0, 1, 4, 5), (6, 7, 2, 3)]
    created = _world_mesh_setup(monkeypatch, rows[1], rows, world_size=8)
    dcp.get_or_create_dcp_mesh()
    assert created["axis_names"] == (dcp.DCP_REPLICA_AXIS_NAME, dcp.DCP_AXIS_NAME)
    assert created["ids"] == [list(r) for r in sorted(rows)]


def test_a_dcp_group_equal_to_the_world_keeps_the_one_axis_mesh(monkeypatch):
    seen = {}
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: _FakeGroup(range(8)))
    monkeypatch.setattr(dcp, "_torch_world_size", lambda: 8)
    monkeypatch.setattr(
        dcp, "get_or_create_cp_mesh", lambda axis_name, group: seen.setdefault("g", group)
    )
    dcp.get_or_create_dcp_mesh()
    assert seen["g"].world_size == 8


def test_dcp_groups_that_do_not_partition_the_world_are_refused(monkeypatch):
    rows = [(0, 1, 2, 3), (0, 1, 2, 3)]
    _world_mesh_setup(monkeypatch, rows[0], rows, world_size=8)
    with pytest.raises(RuntimeError, match="do not partition"):
        dcp.get_or_create_dcp_mesh()
