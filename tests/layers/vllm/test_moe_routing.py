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


def test_route_on_gathered_tokens_with_gate_module():
    import torch

    num_tokens = 4
    hidden_size = 16
    num_experts = 8
    topk = 2

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(experts_per_token=topk),
        renormalize=True,
        scoring_func="softmax",
        gate=torch.nn.Linear(hidden_size, num_experts, bias=False),
    )
    gathered_hidden_states = torch.randn(num_tokens, hidden_size)

    topk_weights, topk_ids = moe_routing.route_on_gathered_tokens(
        layer, gathered_hidden_states)

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)
    assert topk_weights.dtype == gathered_hidden_states.dtype
    assert topk_ids.dtype == torch.int32
    assert (topk_ids >= 0).all() and (topk_ids < num_experts).all()


def test_route_on_gathered_tokens_with_gate_weight():
    import torch

    num_tokens = 4
    hidden_size = 16
    num_experts = 8
    topk = 2

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(experts_per_token=topk),
        renormalize=True,
        scoring_func="softmax",
        gate_weight=torch.randn(hidden_size, num_experts),
    )
    gathered_hidden_states = torch.randn(num_tokens, hidden_size)

    topk_weights, topk_ids = moe_routing.route_on_gathered_tokens(
        layer, gathered_hidden_states)

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)
    assert (topk_ids >= 0).all() and (topk_ids < num_experts).all()
