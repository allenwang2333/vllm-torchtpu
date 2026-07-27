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

import torch

def test_sepp_router_profile_generation():
    router = moe_routing.SortedExpertProfileRouter(num_experts=4, total_tokens=100, top_k=2, alignment=16)
    profile = router.static_profile
    assert len(profile) == 4
    # Profile should be sorted in descending order
    assert profile == sorted(profile, reverse=True)
    # All capacities must be aligned to 16
    for cap in profile:
        assert cap % 16 == 0

def test_sepp_router_route_and_generate_indices():
    router = moe_routing.SortedExpertProfileRouter(num_experts=4, total_tokens=4, top_k=2, alignment=16)
    
    # Simulate topk_ids where expert 2 is hottest, 0 is second, 3 is third, 1 is coldest
    topk_ids = torch.tensor([
        [2, 0],
        [2, 0],
        [2, 3],
        [3, 1]
    ])
    
    actual_counts, static_profile, phys_to_profile_map = router.route_and_generate_indices(topk_ids)
    
    # Check actual counts:
    # Expert 0: 2 tokens
    # Expert 1: 1 token
    # Expert 2: 3 tokens
    # Expert 3: 2 tokens
    # Sorted actual counts should be [3, 2, 2, 1]
    assert actual_counts.tolist() == [3, 2, 2, 1]
    
    # Hottest expert 2 -> profile index 0
    # Next experts 0 and 3 -> profile index 1 and 2 (order can vary depending on sorting stability, but 2 should be index 0)
    assert phys_to_profile_map[2] == 0
    assert phys_to_profile_map[1] == 3

def test_sepp_router_with_out_of_bound_ids():
    router = moe_routing.SortedExpertProfileRouter(num_experts=4, total_tokens=4, top_k=2, alignment=16)
    
    # Simulate padded IDs (-1) and out-of-bound IDs (99)
    topk_ids = torch.tensor([
        [2, -1],
        [99, 0],
        [2, 3],
        [3, -1]
    ])
    
    actual_counts, static_profile, phys_to_profile_map = router.route_and_generate_indices(topk_ids)
    
    # Valid tokens: [2, 0, 2, 3, 3] -> Expert 2: 2, Expert 3: 2, Expert 0: 1, Expert 1: 0
    assert actual_counts.tolist() == [2, 2, 1, 0]
    
    # It shouldn't crash and the returned arrays should correctly reflect valid distributions
    assert len(phys_to_profile_map) == 4
    for phys_id in phys_to_profile_map.keys():
        assert 0 <= phys_id < 4
