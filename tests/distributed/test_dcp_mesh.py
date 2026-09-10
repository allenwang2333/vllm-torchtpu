# SPDX-License-Identifier: Apache-2.0
"""The DCP-side accessors over `pcp.py`'s shared layout and mesh helpers.

Both helpers are one-liners, and both are one-liners whose failure mode is
silent: a mesh built over the wrong group, or over the right group under the
wrong axis name, still builds. It just attends the wrong shard. So what is
worth pinning is not the arithmetic but the plumbing -- which group is asked
for, and which axis name comes out.

No hardware and no JAX: `get_or_create_pcp_mesh` is faked out, because the
question here is what it gets called with.
"""

import pytest

from vllm_torchtpu.distributed import dcp, pcp


class _FakeGroup:

    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group


def test_mesh_helper_uses_the_dcp_group_and_axis(monkeypatch):
    """`get_or_create_dcp_mesh` must not silently build a PCP mesh.

    The shared implementation defaults to the PCP group and the `pcp` axis, so
    forwarding neither is the failure this guards: under `tp=N, dcp=N` the two
    groups cover the same devices, the mesh builds, and the only symptom is
    that the DCP partition specs name an axis the mesh does not have.
    """
    seen = {}

    def _fake_mesh(axis_name, group):
        seen["axis_name"] = axis_name
        seen["group"] = group
        return "mesh"

    group = _FakeGroup([4, 5, 6, 7], rank_in_group=1)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(dcp, "get_or_create_pcp_mesh", _fake_mesh)

    assert dcp.get_or_create_dcp_mesh() == "mesh"
    assert seen["axis_name"] == dcp.DCP_AXIS_NAME
    assert seen["group"] is group


def test_an_explicit_axis_name_wins_over_the_default(monkeypatch):
    """The default is a default, not a constant folded into the call."""
    seen = {}
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: _FakeGroup([0, 1]))
    monkeypatch.setattr(
        dcp, "get_or_create_pcp_mesh",
        lambda axis_name, group: seen.setdefault("axis_name", axis_name))

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
    monkeypatch.setattr(pcp, "_collect_rank_to_device_id",
                        lambda group, *_: {r: 10 + r
                                           for r in group.ranks})
    monkeypatch.setattr(pcp, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(pcp, "_get_tpu_global_device_id", lambda: 12)

    # The no-argument call is the pre-existing contract, and it still holds.
    assert pcp.get_pcp_group_layout().world_size == 1

    layout = dcp.get_dcp_group_layout()
    assert layout.world_size == 4
    assert layout.rank_in_group == 2
    assert layout.device_ids == (10, 11, 12, 13)
    assert layout.device_id == 12


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_the_plain_accessors_survive_an_uninitialized_group(
        monkeypatch, world_size):
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
