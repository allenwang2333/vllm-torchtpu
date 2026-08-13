# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu.distributed import pcp


class _FakeGroup:

    def __init__(self, ranks=(0, 1, 2, 3), rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group
        self.cpu_group = MagicMock()


def test_get_topology_aware_pcp_ranks_single_process():
    group = _FakeGroup(ranks=[0], rank_in_group=0)
    ranks = pcp._get_topology_aware_pcp_ranks(group, global_rank=0)
    assert ranks == (0, )


def test_get_topology_aware_pcp_ranks_tp1_preserves_native_order():
    group = _FakeGroup(ranks=[0, 1, 2, 3], rank_in_group=1)
    with patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1):
        ranks = pcp._get_topology_aware_pcp_ranks(group, global_rank=1)
    assert ranks == (0, 1, 2, 3)


def test_get_topology_aware_pcp_ranks_tp2_extracts_topo_lane():
    # 4 PCP x 2 TP mesh tensor:
    # TP lane 0: [0, 2, 4, 6]
    # TP lane 1: [1, 3, 5, 7]
    topo_tensor = torch.tensor([
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ])
    group = _FakeGroup(ranks=[0, 2, 4, 6], rank_in_group=0)

    with patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            return_value=2):
        if not hasattr(torch, "tpu"):
            torch.tpu = MagicMock()
        with patch("torch.tpu.topology_aware_mesh", return_value=topo_tensor):
            ranks = pcp._get_topology_aware_pcp_ranks(group, global_rank=2)

    assert ranks == (0, 2, 4, 6)


def test_get_pcp_group_layout_builds_layout(monkeypatch):
    group = _FakeGroup(ranks=[0, 1], rank_in_group=0)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    monkeypatch.setattr(pcp, "_get_current_global_rank", lambda: 0)
    monkeypatch.setattr(pcp, "_get_tpu_global_device_id", lambda: 10)
    monkeypatch.setattr(pcp, "_collect_rank_to_device_id",
                        lambda g, r, d: {
                            0: 10,
                            1: 11
                        })

    pcp._LAYOUT_CACHE.clear()
    layout = pcp.get_pcp_group_layout()

    assert layout.ranks == (0, 1)
    assert layout.device_ids == (10, 11)
    assert layout.rank_in_group == 0
    assert layout.world_size == 2
