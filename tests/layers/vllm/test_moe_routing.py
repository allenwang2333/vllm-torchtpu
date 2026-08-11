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
"""Unit tests for vllm_torchtpu.layers.vllm.moe_routing helpers."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.vllm import moe_routing


def _make_layer(*, use_ep, ep_size=1, ep_rank=0, global_num_experts=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ep_size=ep_size,
            ep_rank=ep_rank),
        global_num_experts=global_num_experts,
    )


# (ep_size, global_num_experts) -- mix of divisible and non-divisible cases,
# plus the production sizes we actually ship (Qwen3-30B: 128, Qwen3-480B: 160).
@pytest.mark.parametrize("ep_size,global_num_experts", [
    (2, 8),
    (2, 9),
    (4, 18),
    (4, 33),
    (8, 128),
    (8, 160),
])
def test_get_experts_start_matches_determine_expert_map(
        ep_size, global_num_experts):
    """Parity against vLLM's linear-placement contract for every rank."""
    determine_expert_map = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.expert_map_manager"
    ).determine_expert_map
    for ep_rank in range(ep_size):
        _, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy="linear",
        )
        owned = (expert_map >= 0).nonzero(as_tuple=True)[0]
        expected = int(owned[0].item()) if owned.numel() else 0
        actual = moe_routing.get_experts_start(
            _make_layer(use_ep=True,
                        ep_size=ep_size,
                        ep_rank=ep_rank,
                        global_num_experts=global_num_experts))
        assert actual == expected, (
            f"ep_size={ep_size} ep_rank={ep_rank} "
            f"global_num_experts={global_num_experts}: "
            f"expected experts_start={expected}, got {actual}")


def test_get_experts_start_returns_none_without_ep():
    assert moe_routing.get_experts_start(_make_layer(use_ep=False)) is None


def test_validate_linear_ep_placement_accepts_linear():
    moe_routing.validate_linear_ep_placement(
        SimpleNamespace(expert_placement_strategy="linear"))


def test_validate_linear_ep_placement_requires_declared_strategy():
    # An object that never declares its placement must fail loudly, not be
    # assumed linear.
    with pytest.raises(AttributeError):
        moe_routing.validate_linear_ep_placement(SimpleNamespace())


def test_validate_linear_ep_placement_rejects_round_robin():
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        moe_routing.validate_linear_ep_placement(
            SimpleNamespace(expert_placement_strategy="round_robin"))


def test_capacity_aware_routing_bounds_rank_load(monkeypatch):
    import torch
    scores = torch.randn(8, 64)
    # Make rank 0 (experts 0..7) artificially dominant
    scores[:, :8] += 20.0

    layer = _make_layer(use_ep=True, ep_size=8, ep_rank=0, global_num_experts=64)
    layer.scoring_func = "softmax"
    layer.renormalize = True

    # With capacity factor 1.25, max capacity per rank should be bounded
    monkeypatch.setenv("MOE_ROUTING_CAPACITY_FACTOR", "1.25")
    hidden_dummy = torch.randn(8, 128)
    topk_weights, topk_ids = moe_routing.select_experts(
        hidden_states=hidden_dummy,
        router_logits=scores,
        topk=8,
        renormalize=True,
        scoring_fn="softmax",
        layer=layer,
    )

    assert topk_ids.shape == (8, 8)
    assert topk_weights.shape == (8, 8)

    # Check rank counts for rank 0: under 8*8=64 total routes, ideal cap is 8.
    # With capacity_factor 1.25, max_cap is 8*1.25 + 1 = 11.
    rank0_count = (topk_ids < 8).sum().item()
    assert rank0_count <= 11, f"Rank 0 exceeded capacity bound: {rank0_count} > 11"


def test_local_expert_affinity_epsilon(monkeypatch):
    import torch
    scores = torch.zeros(4, 64)
    layer = _make_layer(use_ep=True, ep_size=8, ep_rank=2, global_num_experts=64)
    layer.scoring_func = "softmax"
    layer.renormalize = True

    # Rank 2 owns experts 16..23
    monkeypatch.setenv("MOE_LOCAL_EXPERT_AFFINITY_EPSILON", "0.5")
    hidden_dummy = torch.randn(4, 128)
    _, topk_ids = moe_routing.select_experts(
        hidden_states=hidden_dummy,
        router_logits=scores,
        topk=8,
        renormalize=True,
        scoring_fn="softmax",
        layer=layer,
    )

    # Since base scores were all 0 and local experts got +0.5 bias, topk_ids should all select experts 16..23
    assert set(topk_ids[0].tolist()) == set(range(16, 24))
