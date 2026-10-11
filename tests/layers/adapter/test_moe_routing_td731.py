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
    monkeypatch.setattr(
        moe_hierarchical, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(
        moe_hierarchical, "get_tensor_model_parallel_rank", lambda: rank
    )


def test_hash_select_3d_scores_and_int32_input_ids() -> None:
    scores = torch.rand(2, 3, 16, dtype=torch.float32)
    table = torch.randint(0, 16, (32, 4), dtype=torch.int32)
    ids32 = torch.randint(0, 32, (2, 3), dtype=torch.int32)
    w, idx = moe_routing._hash_moe_select(
        scores, table, ids32, renormalize=False, routed_scaling_factor=1.0
    )
    assert w.shape == (2, 3, 4)
    assert idx.shape == (2, 3, 4)
    assert idx.dtype == torch.int32


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


def test_td8291_hash_moe_select_and_routed_topk_ids_int32_index_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD8.291: Verify _hash_moe_select and _hash_routed_topk_ids use native
    torch.int32 1D .index_select(0, ...) for both int32 and int64 input_ids."""
    from types import SimpleNamespace

    from vllm_torchtpu.layers.adapter import fused_moe_ep

    seen_dtypes: list[torch.dtype] = []
    orig_index_select = torch.Tensor.index_select

    def _spy_index_select(self, dim, index):
        seen_dtypes.append(index.dtype)
        return orig_index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", _spy_index_select)

    scores = torch.rand(6, 16, dtype=torch.float32)
    table = torch.randint(0, 16, (32, 4), dtype=torch.int32)
    ids32 = torch.randint(0, 32, (2, 3), dtype=torch.int32)
    ids64 = ids32.to(torch.int64)

    w32, idx32 = moe_routing._hash_moe_select(
        scores, table, ids32, renormalize=False, routed_scaling_factor=1.0
    )
    w64, idx64 = moe_routing._hash_moe_select(
        scores, table, ids64, renormalize=False, routed_scaling_factor=1.0
    )
    assert torch.equal(w32, w64)
    assert torch.equal(idx32, idx64)

    layer = SimpleNamespace(hash_indices_table=table)
    ep32 = fused_moe_ep.hash_topk_ids_operand(layer, ids32)
    ep64 = fused_moe_ep.hash_topk_ids_operand(layer, ids64)
    assert ep32 is not None and ep64 is not None
    assert torch.equal(ep32, ep64)
    assert seen_dtypes == [torch.int32, torch.int32, torch.int32, torch.int32]


def test_td8310_moe_routing_cast_guards_and_v2_host_int32_cumsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD8.310: Verify redundant .float() / .to() casts are bypassed in
    moe_routing.py when inputs/outputs already match target dtypes, and all
    jnp.cumsum calls in fused_moe/v2/host.py pass explicit dtype=jnp.int32."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host

    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    float_calls: list[torch.dtype] = []
    to_calls: list[object] = []
    orig_float = torch.Tensor.float
    orig_to = torch.Tensor.to

    def _spy_float(self: torch.Tensor) -> torch.Tensor:
        float_calls.append(self.dtype)
        return orig_float(self)

    def _spy_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        to_calls.append(args[0] if args else kwargs.get("dtype"))
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", _spy_float)
    monkeypatch.setattr(torch.Tensor, "to", _spy_to)

    # 1. _topk with float32 vs bfloat16 bias
    scores = torch.rand(4, 8, dtype=torch.float32)
    bias_f32 = torch.zeros(8, dtype=torch.float32)
    bias_bf16 = torch.zeros(8, dtype=torch.bfloat16)
    float_calls.clear()
    moe_routing._topk(scores, 2, bias=bias_f32)
    assert float_calls == []
    moe_routing._topk(scores, 2, bias=bias_bf16)
    assert float_calls == [torch.bfloat16]

    # 2. _apply_scoring_fn with float32 vs bfloat16 router_logits
    float_calls.clear()
    moe_routing._apply_scoring_fn("softmax", scores)
    assert float_calls == []
    moe_routing._apply_scoring_fn("softmax", scores.bfloat16())
    assert float_calls == [torch.bfloat16]

    # 3. _uniform_random_routing with float32/bfloat16 logits and int32/int64 topk
    hidden_bf16 = torch.ones(4, 16, dtype=torch.bfloat16)
    float_calls.clear()
    to_calls.clear()
    w_ur, i_ur = moe_routing._uniform_random_routing(hidden_bf16, scores, topk=2)
    assert w_ur.dtype == torch.bfloat16 and i_ur.dtype == torch.int32
    assert float_calls == []
    assert to_calls == [torch.int32]

    orig_topk = torch.topk
    monkeypatch.setattr(
        torch,
        "topk",
        lambda x, k, dim=-1: (
            orig_topk(x, k=k, dim=dim)[0],
            orig_topk(x, k=k, dim=dim)[1].to(torch.int32),
        ),
    )
    float_calls.clear()
    to_calls.clear()
    w_ur2, i_ur2 = moe_routing._uniform_random_routing(
        hidden_bf16, scores.bfloat16(), topk=2
    )
    assert w_ur2.dtype == torch.bfloat16 and i_ur2.dtype == torch.int32
    assert float_calls == [torch.bfloat16]
    # Only the helper's internal .to(torch.int32) ran.
    assert to_calls == [torch.int32]
    monkeypatch.setattr(torch, "topk", orig_topk)

    # 4. maybe_simulate_routing with matching vs non-matching dtypes
    monkeypatch.setattr(moe_routing, "_strategy_name", "custom_sim")
    monkeypatch.setattr(
        moe_routing,
        "_SIMULATION_STRATEGY",
        SimpleNamespace(
            route_tokens=lambda **kw: (
                torch.ones(4, 2, dtype=torch.bfloat16),
                torch.zeros(4, 2, dtype=torch.int32),
            )
        ),
    )
    to_calls.clear()
    sim_out = moe_routing.maybe_simulate_routing(hidden_bf16, scores, topk=2)
    assert sim_out is not None
    assert to_calls == []

    monkeypatch.setattr(
        moe_routing,
        "_SIMULATION_STRATEGY",
        SimpleNamespace(
            route_tokens=lambda **kw: (
                torch.ones(4, 2, dtype=torch.float32),
                torch.zeros(4, 2, dtype=torch.int64),
            )
        ),
    )
    to_calls.clear()
    sim_out2 = moe_routing.maybe_simulate_routing(hidden_bf16, scores, topk=2)
    assert sim_out2 is not None
    assert to_calls == [torch.bfloat16, torch.int32]
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    # 5. select_experts: hash routing, grouped topk, and classic topk fast paths
    hidden_f32 = torch.ones(4, 16, dtype=torch.float32)
    table_i32 = torch.zeros(16, 2, dtype=torch.int32)
    input_ids_i32 = torch.zeros(4, dtype=torch.int32)
    layer_hash = SimpleNamespace(
        hash_indices_table=table_i32,
        routed_scaling_factor=1.0,
    )
    to_calls.clear()
    w_h, i_h = moe_routing.select_experts(
        hidden_f32,
        scores,
        topk=2,
        renormalize=False,
        scoring_fn="softmax",
        layer=layer_hash,
        input_ids=input_ids_i32,
    )
    assert w_h.dtype == torch.float32 and i_h.dtype == torch.int32
    assert to_calls == []

    layer_grouped = SimpleNamespace(
        hash_indices_table=None,
        use_grouped_topk=True,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )
    monkeypatch.setattr(
        moe_routing._grouped_topk_router,
        "grouped_topk",
        lambda **kw: (
            torch.ones(4, 2, dtype=torch.float32),
            torch.zeros(4, 2, dtype=torch.int32),
        ),
    )
    to_calls.clear()
    w_g, i_g = moe_routing.select_experts(
        hidden_f32,
        scores,
        topk=2,
        renormalize=False,
        scoring_fn="softmax",
        layer=layer_grouped,
    )
    assert w_g.dtype == torch.float32 and i_g.dtype == torch.int32
    assert to_calls == []

    monkeypatch.setattr(
        moe_routing,
        "_topk",
        lambda *a, **kw: (
            torch.ones(4, 2, dtype=torch.float32),
            torch.zeros(4, 2, dtype=torch.int32),
        ),
    )
    to_calls.clear()
    w_c, i_c = moe_routing.select_experts(
        hidden_f32,
        scores,
        topk=2,
        renormalize=False,
        scoring_fn="softmax",
        layer=None,
    )
    assert w_c.dtype == torch.float32 and i_c.dtype == torch.int32
    assert to_calls == []

    # 6. fused_moe/v2/host.py explicit dtype=jnp.int32 on jnp.cumsum
    cumsum_dtypes: list[object] = []
    orig_cumsum = jnp.cumsum

    def _spy_cumsum(a, axis=None, dtype=None, out=None):
        cumsum_dtypes.append(dtype)
        return orig_cumsum(a, axis=axis, dtype=dtype, out=out)

    monkeypatch.setattr(v2_host.jnp, "cumsum", _spy_cumsum)
    topk_idx = jnp.zeros((16, 2), dtype=jnp.int32)
    rt = v2_host.build_routing_tables(
        topk_idx,
        e_total=4,
        ep=2,
        t_local=8,
        block=16,
        tile_m=8,
        shard_stride=16,
    )
    v2_host.build_routing_tables_sharded(
        topk_idx,
        jnp.int32(0),
        e_total=4,
        ep=2,
        t_local=8,
        block=16,
        tile_m=8,
        shard_stride=16,
        all_gather_rows=lambda x: jnp.repeat(x, 2, axis=0),
    )
    v2_host.shard_transport_tables_in_blocks(rt, jnp.int32(0), e_total=4, ep=2)
    assert len(cumsum_dtypes) == 12
    assert all(dt == jnp.int32 for dt in cumsum_dtypes)
