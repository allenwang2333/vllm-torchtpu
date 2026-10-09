# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TD7.31: int32 routing ids, cached gate-slice plan, hierarchical MoE split."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_hierarchical, moe_routing
from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as v41

# ---------------------------------------------------------------------------
# moe_routing: ids stay in their own integer dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts", [8, 96, 256])
def test_gather_expert_scores_int32_matches_int64_and_gather(num_experts) -> None:
    torch.manual_seed(num_experts)
    scores = torch.rand(5, 7, num_experts, dtype=torch.float32)
    ids64 = torch.randint(0, num_experts, (5, 7, 6), dtype=torch.int64)
    ids32 = ids64.to(torch.int32)

    out32 = moe_routing._gather_expert_scores(scores, ids32)
    out64 = moe_routing._gather_expert_scores(scores, ids64)

    assert torch.equal(out32, out64)
    assert torch.equal(out64, scores.gather(-1, ids64))
    assert out32.dtype == scores.dtype


def test_gather_expert_scores_keeps_non_finite_out_of_unselected() -> None:
    scores = torch.tensor([[0.25, float("inf"), 0.5, float("nan")]])
    ids = torch.tensor([[2, 0]], dtype=torch.int32)
    assert torch.equal(
        moe_routing._gather_expert_scores(scores, ids), torch.tensor([[0.5, 0.25]])
    )


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_hash_select_keeps_table_dtype_and_matches_int64(renormalize, scale) -> None:
    torch.manual_seed(0)
    num_experts, topk, vocab = 96, 6, 64
    scores = torch.rand(9, num_experts, dtype=torch.float32)
    table32 = torch.randint(0, num_experts, (vocab, topk), dtype=torch.int32)
    ids = torch.randint(0, vocab, (9,), dtype=torch.int64)

    w32, i32 = moe_routing._hash_moe_select(scores, table32, ids, renormalize, scale)
    w64, i64 = moe_routing._hash_moe_select(
        scores, table32.long(), ids, renormalize, scale
    )

    # The production table is int32: no int64 widening on the hot path.
    assert i32.dtype == torch.int32
    assert i64.dtype == torch.int64
    assert torch.equal(i32.long(), i64)
    assert torch.equal(w32, w64)
    assert torch.equal(i32, table32[ids])


# ---------------------------------------------------------------------------
# deepseek_v41: gate-slice plan cached per layer, dynamic checks per call
# ---------------------------------------------------------------------------


def _moe(*, sp=False, dp_size=1, use_ep=True):
    moe = object.__new__(v41.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    object.__setattr__(
        moe,
        "experts",
        SimpleNamespace(
            moe_config=SimpleNamespace(
                dp_size=dp_size,
                pcp_size=1,
                moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ),
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    is_monolithic=True, supports_internal_mk=False
                ),
                custom_routing_function=None,
            ),
        ),
    )
    return moe


def _counting_group(monkeypatch, world_size, rank):
    calls = {"ep": 0, "tp": 0}
    group = SimpleNamespace(world_size=world_size, rank_in_group=rank)

    def ep():
        calls["ep"] += 1
        return group

    def tp():
        calls["tp"] += 1
        return group

    monkeypatch.setattr(v41, "get_ep_group", ep)
    monkeypatch.setattr(v41, "get_tp_group", tp)
    return calls


@pytest.mark.parametrize("dp_size,key", [(1, "ep"), (2, "tp")])
def test_gate_slice_plan_resolved_once(monkeypatch, dp_size, key) -> None:
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    calls = _counting_group(monkeypatch, 4, 3)
    moe = _moe(dp_size=dp_size)
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    for _ in range(3):
        assert torch.equal(moe._maybe_slice_gate_input(hidden), hidden[6:8])
    # Odd token count: plan is reused, divisibility still checked per call.
    odd = hidden[:7]
    assert moe._maybe_slice_gate_input(odd) is odd

    assert calls[key] == 1
    assert sum(calls.values()) == 1
    assert moe.__dict__["_gate_slice_plan"] == (4, 3)
    # Stored outside nn.Module's registries.
    assert "_gate_slice_plan" not in moe.__dict__.get("_buffers", {})


@pytest.mark.parametrize(
    "build,world_size",
    [(lambda: _moe(use_ep=False), 4), (lambda: _moe(), 1)],
)
def test_gate_slice_plan_caches_none(monkeypatch, build, world_size) -> None:
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    calls = _counting_group(monkeypatch, world_size, 0)
    moe = build()
    hidden = torch.ones(8, 4)
    assert moe._maybe_slice_gate_input(hidden) is hidden
    assert moe._maybe_slice_gate_input(hidden) is hidden
    assert moe.__dict__["_gate_slice_plan"] is None
    assert sum(calls.values()) <= 1


def test_sp_and_simulation_stay_dynamic(monkeypatch) -> None:
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    _counting_group(monkeypatch, 2, 1)
    moe = _moe()
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(moe._maybe_slice_gate_input(hidden), hidden[2:4])

    # Toggling either after the plan is cached still takes effect.
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    assert moe._maybe_slice_gate_input(hidden) is hidden
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    object.__setattr__(moe, "use_sequence_parallel", True)
    assert moe._maybe_slice_gate_input(hidden) is hidden
    object.__setattr__(moe, "use_sequence_parallel", False)
    assert torch.equal(moe._maybe_slice_gate_input(hidden), hidden[2:4])


def test_simulation_does_not_resolve_plan(monkeypatch) -> None:
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    calls = _counting_group(monkeypatch, 2, 0)
    moe = _moe()
    hidden = torch.ones(4, 4)
    assert moe._maybe_slice_gate_input(hidden) is hidden
    assert "_gate_slice_plan" not in moe.__dict__
    assert sum(calls.values()) == 0


# ---------------------------------------------------------------------------
# moe_hierarchical: split resolution and weight-filter guard branches
# ---------------------------------------------------------------------------


def _tp(monkeypatch, world_size, rank):
    import vllm.distributed as vd

    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: world_size)
    monkeypatch.setattr(vd, "get_tensor_model_parallel_rank", lambda: rank)


def test_split_none_when_single_rank(monkeypatch) -> None:
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    _tp(monkeypatch, 1, 0)
    monkeypatch.setattr(
        moe_hierarchical,
        "hierarchical_moe_split",
        lambda *a: pytest.fail("must not be called"),
    )
    assert moe_hierarchical.hierarchical_split_or_none() is None


def test_split_falls_back_when_topology_unusable(monkeypatch) -> None:
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    _tp(monkeypatch, 8, 3)
    seen = []
    monkeypatch.setattr(
        moe_hierarchical,
        "hierarchical_moe_split",
        lambda ws, r: seen.append((ws, r)),
    )
    assert moe_hierarchical.hierarchical_split_or_none() is None
    assert seen == [(8, 3)]


def test_split_returned_when_topology_usable(monkeypatch) -> None:
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    _tp(monkeypatch, 8, 5)
    monkeypatch.setattr(
        moe_hierarchical, "hierarchical_moe_split", lambda ws, r: (4, r // 2, 2, r % 2)
    )
    assert moe_hierarchical.hierarchical_split_or_none() == (4, 2, 2, 1)


def test_align_filter_noop_without_split(monkeypatch) -> None:
    from vllm.model_executor.model_loader import default_loader

    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    monkeypatch.setattr(moe_hierarchical, "hierarchical_split_or_none", lambda: None)
    before = default_loader.compute_local_expert_ids
    moe_hierarchical.align_ep_weight_filter()
    assert default_loader.compute_local_expert_ids is before
    assert moe_hierarchical._filter_patched is False


def test_align_filter_runs_once(monkeypatch) -> None:
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", True)
    monkeypatch.setattr(
        moe_hierarchical,
        "hierarchical_split_or_none",
        lambda: pytest.fail("must not be called"),
    )
    moe_hierarchical.align_ep_weight_filter()
