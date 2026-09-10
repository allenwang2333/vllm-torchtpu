# SPDX-License-Identifier: Apache-2.0
"""The DCP wiring that sits between the kernels and vLLM.

The kernels themselves are covered on hardware
(`tests/kernels/test_dsv4_streamindex_topk_dcp.py`,
`tests/layers/common/test_sparse_mla_attention.py::SparseMlaDcpTest`) and the
merge arithmetic in `test_dcp_merge.py`. What is left, and what this file
covers, is the glue: the ownership rule that the indexer's return leg and the
KV write have to agree on, and the mesh helper that DCP shares with PCP.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.distributed import dcp, pcp
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME, cp_global_to_local, cp_owner_rank)


class _FakeGroup:

    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group


# The indexer's return leg is an all-gather plus a per-rank owner filter, which
# is only exact because `cp_owner_rank` is a *partition*: total, single-valued,
# and onto every rank. These pin that, because it is what replaced the
# capacity-sized all-to-all -- an all-to-all had to bet a static per-rank width
# and silently shed winners when the bet lost.


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_ownership_is_a_partition_of_the_positions(dcp_size, interleave_c):
    """Every position has exactly one owner, and every rank is an owner."""
    n = 5 * dcp_size * interleave_c + 3
    owners = np.asarray(
        cp_owner_rank(jnp.arange(n, dtype=jnp.int32), dcp_size, interleave_c))
    assert owners.min() >= 0 and owners.max() < dcp_size
    assert set(owners.tolist()) == set(range(dcp_size))


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_owner_counts_are_balanced_over_a_whole_cycle(dcp_size, interleave_c):
    """Over a whole interleave cycle the shards are exactly equal.

    Which is why mean per-rank occupancy of a `k`-wide top-k is `k / dcp_size`
    as an identity, not an estimate -- the spread around it comes only from
    where the winners cluster inside the cycle.
    """
    cycle = dcp_size * interleave_c
    owners = np.asarray(
        cp_owner_rank(jnp.arange(3 * cycle, dtype=jnp.int32), dcp_size,
                      interleave_c))
    counts = np.bincount(owners, minlength=dcp_size)
    assert (counts == 3 * interleave_c).all()


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_local_indices_stay_inside_the_shard(dcp_size, interleave_c):
    """A rank is only ever handed an index into its own cache shard."""
    n = 5 * dcp_size * interleave_c
    gs = jnp.arange(n, dtype=jnp.int32)
    locals_ = np.asarray(cp_global_to_local(gs, dcp_size, interleave_c))
    assert locals_.min() >= 0
    assert locals_.max() < n // dcp_size


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
                        lambda group, *_: {r: 10 + r
                                           for r in group.ranks})
    monkeypatch.setattr(pcp, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(pcp, "_get_tpu_global_device_id", lambda: 12)

    assert pcp.get_pcp_group_layout().world_size == 1

    layout = dcp.get_dcp_group_layout()
    assert layout.world_size == 4
    assert layout.rank_in_group == 2
    assert layout.device_ids == (10, 11, 12, 13)
    assert layout.device_id == 12
