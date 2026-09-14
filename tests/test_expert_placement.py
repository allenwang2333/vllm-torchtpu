"""The static expert placement (expert_placement.py) on a synthetic table:
the table checks, the slot maps, the router-row permutation, and the
identity without a table."""
import json

import pytest
import torch

from vllm_torchtpu import expert_placement as ep


def _table(tmp_path, dup=False):
    order = [3, 3, 2, 1] if dup else [3, 0, 2, 1]  # rank 0: slots 0, 1; rank 1
    table = {
        "shards": 2,
        "experts": 4,
        "slots_per_core": 2,
        "layers": {
            str(n): {
                "cores": [order[:2], order[2:]]
            }
            for n in (0, 1)
        },
    }
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(table))
    return str(path), order


def test_maps_and_router_rows(tmp_path, monkeypatch):
    path, order = _table(tmp_path)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    assert ep.enabled()
    assert (ep.shards(), ep.slots(), ep.global_count()) == (2, 2, 4)
    assert ep.virtual_order(0) == order
    for rank in range(2):
        m = ep.expert_map(0, rank)
        for j, e in enumerate(ep.cores(0)[rank]):
            assert m[e] == j
            assert ep.virtual_order(0)[rank * 2 + j] == e
        assert ep.local_ids(0, rank) == frozenset(ep.cores(0)[rank])
        assert m.count(-1) == 2
    w = torch.arange(4.0).reshape(4, 1)
    out = ep.permute_router_rows("model.layers.0.mlp.gate.weight", w)
    assert out[:, 0].tolist() == [3.0, 0.0, 2.0, 1.0]
    assert ep.permute_router_rows("model.layers.0.mlp.up_proj.weight", w) is w
    with pytest.raises(ValueError):
        ep.permute_router_rows("model.layers.1.mlp.gate.weight",
                               torch.zeros(5, 1))


def test_a_duplicated_expert_is_refused(tmp_path, monkeypatch):
    path, _ = _table(tmp_path, dup=True)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    with pytest.raises(ValueError):
        ep.data()


def test_unset_is_the_identity(monkeypatch):
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
    assert not ep.enabled()
    w = torch.zeros(4, 1)
    assert ep.permute_router_rows("model.layers.0.mlp.gate.weight", w) is w
