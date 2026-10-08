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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as mod


def _moe(*, sp, hash_table, naive):
    moe = object.__new__(mod.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    object.__setattr__(moe, "gate", SimpleNamespace(tid2eid=hash_table))
    object.__setattr__(moe, "experts", SimpleNamespace(do_naive_dispatch_combine=naive))
    return moe


@pytest.mark.parametrize(
    "sp,hash_table,naive,gathered",
    [
        (True, torch.zeros(4, 2), True, True),
        (False, torch.zeros(4, 2), True, False),
        (True, None, True, False),
        # An armed fused-EP layer routes this rank's own rows.
        (True, torch.zeros(4, 2), False, False),
    ],
)
def test_hash_ids_follow_sp_dispatch(monkeypatch, sp, hash_table, naive, gathered):
    ep = Mock()
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    seen = {}

    def upstream_forward(self, hidden_states, input_ids=None):
        seen["ids"] = input_ids
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", upstream_forward)
    ids = torch.tensor([1, 2])
    hidden = torch.ones(2, 3)
    assert _moe(sp=sp, hash_table=hash_table, naive=naive).forward(hidden, ids) is (
        hidden
    )
    if gathered:
        ep.all_gather.assert_called_once_with(ids, dim=0)
        torch.testing.assert_close(seen["ids"], torch.tensor([1, 2, 11, 12]))
    else:
        ep.all_gather.assert_not_called()
        assert seen["ids"] is ids


def _make_non_sp_ep_moe(
    *,
    sp: bool = False,
    dp_size: int = 1,
    pcp_size: int = 1,
    use_ep: bool = True,
    is_monolithic: bool = True,
    supports_internal_mk: bool = False,
    custom_routing_function=None,
    gate_fn=None,
    experts_fn=None,
):
    moe = object.__new__(mod.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    if gate_fn is None:

        def _default_gate_fn(x):
            return x @ torch.ones(x.shape[-1], 8, dtype=x.dtype), None

        gate_fn = _default_gate_fn
    object.__setattr__(moe, "gate", gate_fn)
    moe_config = SimpleNamespace(
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=sp,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
    )
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            is_monolithic=is_monolithic,
            supports_internal_mk=supports_internal_mk,
        ),
        custom_routing_function=custom_routing_function,
    )
    if experts_fn is None:
        experts = SimpleNamespace(
            moe_config=moe_config,
            routed_experts=routed_experts,
            do_naive_dispatch_combine=False,
        )
    else:
        experts = experts_fn
        experts.moe_config = moe_config
        experts.routed_experts = routed_experts
    object.__setattr__(moe, "experts", experts)
    return moe


@pytest.mark.parametrize(
    "override_kwargs,ep_size,ep_rank,num_tokens,use_sim,expect_sliced",
    [
        ({}, 4, 2, 16, False, True),
        ({"sp": True}, 4, 2, 16, False, False),
        ({"dp_size": 2}, 4, 2, 16, False, False),
        ({"pcp_size": 2}, 4, 2, 16, False, False),
        ({"use_ep": False}, 4, 2, 16, False, False),
        ({"is_monolithic": False}, 4, 2, 16, False, False),
        ({"supports_internal_mk": True}, 4, 2, 16, False, False),
        ({"custom_routing_function": lambda **kw: None}, 4, 2, 16, False, False),
        ({}, 4, 2, 16, True, False),
        ({}, 1, 0, 16, False, False),
        ({}, 4, 1, 15, False, False),
    ],
)
def test_maybe_slice_gate_input_guards(
    monkeypatch, override_kwargs, ep_size, ep_rank, num_tokens, use_sim, expect_sliced
):
    ep = SimpleNamespace(world_size=ep_size, rank_in_group=ep_rank)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    monkeypatch.setattr(
        mod.moe_routing,
        "_SIMULATION_STRATEGY",
        object() if use_sim else None,
    )
    moe = _make_non_sp_ep_moe(**override_kwargs)
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    sliced = moe._maybe_slice_gate_input(hidden)
    if expect_sliced:
        chunk = num_tokens // ep_size
        expected = hidden[ep_rank * chunk : (ep_rank + 1) * chunk]
        assert sliced.shape == (chunk, 4)
        assert torch.equal(sliced, expected)
    else:
        assert sliced is hidden


def test_forward_fused_moe_slices_gate_and_preserves_hidden_shape(monkeypatch):
    ep = SimpleNamespace(world_size=4, rank_in_group=1)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    monkeypatch.setattr(mod.moe_routing, "_SIMULATION_STRATEGY", None)

    captured = {}

    def gate_fn(x):
        captured["gate_in"] = x
        return x[:, :2] * 3.0, None

    class DummyExperts:
        def __call__(self, *, hidden_states, router_logits, input_ids=None):
            captured["experts_hidden"] = hidden_states
            captured["experts_logits"] = router_logits
            captured["experts_ids"] = input_ids
            return hidden_states + 1.0

    moe = _make_non_sp_ep_moe(gate_fn=gate_fn, experts_fn=DummyExperts())
    hidden = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
    ids = torch.arange(16, dtype=torch.int64)
    out = moe._forward_fused_moe(hidden, input_ids=ids)

    assert torch.equal(captured["gate_in"], hidden[4:8])
    assert captured["experts_hidden"] is hidden
    assert torch.equal(captured["experts_logits"], hidden[4:8, :2] * 3.0)
    assert captured["experts_ids"] is ids
    assert torch.equal(out, hidden + 1.0)


@pytest.mark.parametrize(
    "num_experts,use_hash", [(96, False), (128, False), (96, True)]
)
def test_non_sp_ep8_partitioned_routing_bit_exact_parity(
    monkeypatch, num_experts, use_hash
):
    """Verify 8-rank EP non-SP gate slicing + moe_routing.route matches full routing."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(mod.moe_routing, "_SIMULATION_STRATEGY", None)
    monkeypatch.setattr(mod.moe_routing, "_ROUTER_TOPK", "rowmax")

    ep_size = 8
    total_tokens = 288  # C=48 decode with K=5 (48 * 6 = 288)
    hidden_dim = 32
    topk = 8

    torch.manual_seed(42)
    hidden = torch.randn(total_tokens, hidden_dim, dtype=torch.bfloat16)
    gate_weight = torch.randn(num_experts, hidden_dim, dtype=torch.float32)
    bias = torch.randn(num_experts, dtype=torch.float32)
    tid2eid = (
        torch.randint(0, num_experts, (512, topk), dtype=torch.int32)
        if use_hash
        else None
    )
    input_ids = torch.randint(0, 512, (total_tokens,), dtype=torch.int64)

    def gate_fn(x):
        return torch.nn.functional.linear(x.float(), gate_weight), None

    def make_routing_layer(use_ep_flag: bool):
        return SimpleNamespace(
            moe_config=SimpleNamespace(
                experts_per_token=topk,
                is_sequence_parallel=False,
                dp_size=1,
                pcp_size=1,
                moe_parallel_config=SimpleNamespace(use_ep=use_ep_flag),
            ),
            custom_routing_function=None,
            renormalize=True,
            scoring_func="sigmoid",
            e_score_correction_bias=None if use_hash else bias,
            routed_scaling_factor=1.0,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            router=SimpleNamespace(
                hash_indices_table=tid2eid,
                bias_vl=None,
                image_sentinel_lo=0,
            ),
        )

    # 1. Unpartitioned reference
    ref_logits, _ = gate_fn(hidden)
    ref_layer = make_routing_layer(use_ep_flag=False)
    ref_weights, ref_ids = mod.moe_routing.route(
        ref_layer, hidden, ref_logits, input_ids=input_ids
    )

    # 2. Simulate all 8 EP ranks running _maybe_slice_gate_input + gate_fn + route
    rank_sliced_logits = []
    for r in range(ep_size):
        ep_stub = SimpleNamespace(world_size=ep_size, rank_in_group=r)
        monkeypatch.setattr(mod, "get_ep_group", lambda ep_stub=ep_stub: ep_stub)
        moe = _make_non_sp_ep_moe(gate_fn=gate_fn)
        gate_in = moe._maybe_slice_gate_input(hidden)
        assert gate_in.shape == (total_tokens // ep_size, hidden_dim)
        local_logits, _ = moe.gate(gate_in)
        rank_sliced_logits.append(local_logits)

    # Simulate the collective all_gather inside _gather_partitioned_topk across 8 ranks
    ep_layer = make_routing_layer(use_ep_flag=True)
    local_packed_shards: list[torch.Tensor] = []

    class _CaptureGroup:
        def __init__(self, rank: int):
            self.world_size = ep_size
            self.rank_in_group = rank

        def all_gather(self, tensor, dim=0):
            local_packed_shards.append(tensor)
            return tensor.repeat(ep_size, 1)

    for r in range(ep_size):
        monkeypatch.setattr(ps, "get_ep_group", lambda r=r: _CaptureGroup(r))
        mod.moe_routing.route(
            ep_layer, hidden, rank_sliced_logits[r], input_ids=input_ids
        )

    full_packed = torch.cat(local_packed_shards, dim=0)

    class _ReplayAllRanksGroup:
        world_size = ep_size
        rank_in_group = 0

        def all_gather(self, tensor, dim=0):
            return full_packed

    monkeypatch.setattr(ps, "get_ep_group", _ReplayAllRanksGroup)
    ep_weights, ep_ids = mod.moe_routing.route(
        ep_layer, hidden, rank_sliced_logits[0], input_ids=input_ids
    )

    assert torch.equal(ep_weights, ref_weights)
    assert torch.equal(ep_ids, ref_ids)
