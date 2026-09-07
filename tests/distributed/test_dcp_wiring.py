# SPDX-License-Identifier: Apache-2.0
"""The DCP wiring that sits between the kernels and vLLM.

The kernels themselves are covered on hardware
(`tests/kernels/test_dsv4_streamindex_topk_dcp.py`,
`tests/layers/common/test_sparse_mla_attention.py::SparseMlaDcpTest`) and the
merge arithmetic in `test_dcp_merge.py`. What is left, and what this file
covers, is the glue: the capacity rule that three separate call sites have to
agree on, and the mesh helper that DCP shares with PCP.
"""

import pytest

from vllm_torchtpu.distributed import dcp, pcp
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME, dcp_topk_capacity)


class _FakeGroup:

    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_capacity_leaves_headroom_over_the_mean(dcp_size):
    """Mean occupancy is `k / dcp_size`; the width must exceed it."""
    k = 2048
    capacity = dcp_topk_capacity(k, dcp_size)
    assert capacity > k // dcp_size
    assert capacity <= k


def test_capacity_scales_with_the_shard_count():
    """A fixed width would be a hard truncation at small `dcp_size`.

    This is the bug the multichip run actually hit: `CAPACITY = 512` was
    generous at dcp_size=8 and dropped 131072 winners at dcp_size=2.
    """
    k = 2048
    widths = [dcp_topk_capacity(k, d) for d in (2, 4, 8, 16)]
    assert widths == sorted(widths, reverse=True)
    # At dcp_size=2 a rank can hold at most all of k, and half of k on average,
    # so nothing below k would be safe -- the rule must saturate, not scale.
    assert dcp_topk_capacity(k, 2) == k


def test_capacity_is_the_identity_without_dcp():
    """`dcp_size=1` must leave the non-DCP buffer geometry untouched."""
    assert dcp_topk_capacity(2048, 1) == 2048
    assert dcp_topk_capacity(2048, 0) == 2048


def test_capacity_matches_the_swept_value_at_the_target_shape():
    """GLM-5.2 at dcp=16: the sweep's worst observed occupancy was 199."""
    assert dcp_topk_capacity(2048, 16) == 256


def test_mesh_helper_uses_the_dcp_group_and_axis(monkeypatch):
    """`get_or_create_dcp_mesh` must not silently build a PCP mesh."""
    seen = {}

    def _fake_mesh(axis_name, group):
        seen["axis_name"] = axis_name
        seen["group"] = group
        return "mesh"

    group = _FakeGroup([4, 5, 6, 7], rank_in_group=1)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(dcp, "get_or_create_pcp_mesh", _fake_mesh)

    assert dcp.get_or_create_dcp_mesh() == "mesh"
    assert seen["axis_name"] == DCP_AXIS_NAME
    assert seen["group"] is group


def test_layout_describes_the_group_it_is_given(monkeypatch):
    """The shared layout helper must follow its argument, not the PCP group.

    Without this, a DCP-only run (where the PCP group is trivial) would get a
    world_size=1 layout and build a single-device mesh.
    """
    pcp_group = _FakeGroup([0])
    dcp_group = _FakeGroup([0, 1, 2, 3], rank_in_group=2)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(pcp, "_collect_rank_to_device_id",
                        lambda group, *_: {r: 10 + r for r in group.ranks})
    monkeypatch.setattr(pcp, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(pcp, "_get_tpu_global_device_id", lambda: 12)

    assert pcp.get_pcp_group_layout().world_size == 1

    layout = dcp.get_dcp_group_layout()
    assert layout.world_size == 4
    assert layout.rank_in_group == 2
    assert layout.device_ids == (10, 11, 12, 13)
    assert layout.device_id == 12
