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
"""Tests for env-gated local top-k routing and packed metadata AllGather."""

import torch

from vllm_torchtpu.layers.adapter.moe_routing import route
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    _gather_routing_meta,
)


class _FakeCollectiveGroup:
    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.calls: list[torch.Tensor] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.calls.append(tensor)
        return torch.cat([tensor] * self.world_size, dim=dim)


def test_gather_routing_meta_default_disabled(monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOCAL_ROUTING", "0")
    group = _FakeCollectiveGroup(world_size=4)
    weights = torch.randn((8, 8), dtype=torch.bfloat16)
    ids = torch.randint(0, 384, (8, 8), dtype=torch.int32)

    ag_tw, ag_ti = _gather_routing_meta(group, weights, ids)

    assert len(group.calls) == 2
    assert torch.equal(ag_tw, torch.cat([weights] * 4, dim=0))
    assert torch.equal(ag_ti, torch.cat([ids] * 4, dim=0))


def test_gather_routing_meta_packed_roundtrip(monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOCAL_ROUTING", "1")
    group = _FakeCollectiveGroup(world_size=4)
    weights = torch.randn((8, 8), dtype=torch.bfloat16)
    ids = torch.randint(0, 384, (8, 8), dtype=torch.int32)

    ag_tw, ag_ti = _gather_routing_meta(group, weights, ids)

    assert len(group.calls) == 1
    assert group.calls[0].dtype == torch.int32
    assert group.calls[0].shape == (8, 16)
    assert torch.equal(ag_tw, torch.cat([weights] * 4, dim=0))
    assert torch.equal(ag_ti, torch.cat([ids] * 4, dim=0))


def test_route_passthrough_prerouted_tuple():
    weights = torch.ones((4, 8), dtype=torch.bfloat16)
    ids = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    out_w, out_ids = route(None, torch.zeros((4, 16)), (weights, ids))
    assert out_w is weights
    assert out_ids is ids
