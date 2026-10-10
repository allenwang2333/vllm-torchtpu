# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import stage_groups


def test_make_stage_groups_standalone_rebinds_duplicate_rank_coordinators(
    monkeypatch: pytest.MonkeyPatch,
):
    tp_pg = object()
    ep_pg = object()
    tp_comm = SimpleNamespace(device_group=tp_pg)
    ep_comm = SimpleNamespace(device_group=ep_pg)

    tp_group = SimpleNamespace(
        unique_name="tp:0",
        ranks=[2, 3],
        world_size=2,
        device_group=tp_pg,
        device_communicator=tp_comm,
    )
    ep_group = SimpleNamespace(
        unique_name="ep:0",
        ranks=[2, 3],
        world_size=2,
        device_group=ep_pg,
        device_communicator=ep_comm,
    )
    dp_group = SimpleNamespace(
        unique_name="dp:0",
        ranks=[2],
        world_size=1,
        device_group=object(),
        device_communicator=None,
    )
    pp_group = SimpleNamespace(rank_in_group=1, world_size=4)

    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)

    called_pgs: list[object] = []
    monkeypatch.setattr(
        stage_groups, "_load_make_standalone", lambda: called_pgs.append
    )
    monkeypatch.setattr(stage_groups, "_STANDALONE_PG_BY_RANKS", {})

    parallel_config = SimpleNamespace(
        pipeline_parallel_size=4,
        tensor_parallel_size=2,
        prefill_context_parallel_size=1,
    )
    stage_groups.make_stage_groups_standalone(parallel_config)

    assert called_pgs == [tp_pg]
    assert ep_group.device_group is tp_pg
    assert ep_comm.device_group is tp_pg
    assert stage_groups.standalone_device_group(ep_group) is tp_pg
