# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness contract for the experimental Kimi-K3 Phase 1/2 SP4 layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.kimi_k3.model import KimiDecoderLayer, KimiModel
from vllm_torchtpu.models.vllm.kimi_k3.prefill_sp import (
    K3PrefillSPLayout, build_k3_prefill_sp_group_plan)


@pytest.mark.parametrize(
    ("world_size", "expected_lane_size", "expected_lane_groups",
     "expected_sp_groups"),
    [
        (
            8,
            2,
            ((0, 4), (1, 5), (2, 6), (3, 7)),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
        ),
        (
            32,
            8,
            tuple(tuple(range(sp_rank, 32, 4)) for sp_rank in range(4)),
            tuple(tuple(range(lane * 4, lane * 4 + 4)) for lane in range(8)),
        ),
    ],
)
def test_sp4_group_plan_scales_from_phase1_to_phase2(
    world_size: int,
    expected_lane_size: int,
    expected_lane_groups: tuple[tuple[int, ...], ...],
    expected_sp_groups: tuple[tuple[int, ...], ...],
) -> None:
    plan = build_k3_prefill_sp_group_plan(tuple(range(world_size)), 4)

    assert plan.lane_size == expected_lane_size
    assert plan.lane_groups == expected_lane_groups
    assert plan.sp_groups == expected_sp_groups


def test_native_sp_trace_is_preserved_by_fullgraph_dynamo() -> None:
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    old_capture = torch._dynamo.config.capture_profiler_record_function
    torch._dynamo.reset()
    try:
        torch._dynamo.config.capture_profiler_record_function = True

        @torch.compile(backend=backend, fullgraph=True)
        def compiled(tensor: torch.Tensor) -> torch.Tensor:
            with torch.profiler.record_function("attention_lane_ar"):
                return tensor + 1

        torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
        targets = [str(node.target) for node in graphs[0].graph.nodes]
        assert any("_record_function_enter" in target for target in targets)
        assert any("_record_function_exit" in target for target in targets)
    finally:
        torch._dynamo.config.capture_profiler_record_function = old_capture
        torch._dynamo.reset()


@pytest.mark.parametrize("world_size", [8, 32])
def test_lane_ar_then_sp_rs_matches_tp_sum_and_token_order(
        world_size: int) -> None:
    plan = build_k3_prefill_sp_group_plan(tuple(range(world_size)), 4)
    num_tokens = 8
    partials = {
        rank:
        torch.arange(num_tokens, dtype=torch.float32).unsqueeze(-1) +
        rank * 100
        for rank in plan.tp_ranks
    }

    lane_reduced: dict[int, torch.Tensor] = {}
    for group in plan.lane_groups:
        reduced = sum((partials[rank] for rank in group),
                      torch.zeros_like(partials[group[0]]))
        for rank in group:
            lane_reduced[rank] = reduced

    outputs: dict[int, torch.Tensor] = {}
    for group in plan.sp_groups:
        reduced = sum((lane_reduced[rank] for rank in group),
                      torch.zeros_like(lane_reduced[group[0]]))
        for sp_rank, rank in enumerate(group):
            outputs[rank] = reduced.chunk(4, dim=0)[sp_rank]

    tp_sum = sum((partials[rank] for rank in plan.tp_ranks),
                 torch.zeros_like(partials[0]))
    for lane in range(plan.lane_size):
        for sp_rank in range(plan.sp_size):
            rank = lane * plan.sp_size + sp_rank
            torch.testing.assert_close(outputs[rank],
                                       tp_sum.chunk(4, dim=0)[sp_rank])


def test_only_fixed_long_prefill_buckets_enter_sp() -> None:
    plan = build_k3_prefill_sp_group_plan(tuple(range(8)), 4)
    layout = K3PrefillSPLayout(plan=plan, sp_rank=2)

    for num_tokens in (1024, 2048, 4096):
        hidden_states = torch.arange(num_tokens * 2).view(num_tokens, 2)
        assert layout.should_shard(hidden_states)
        shard = layout.shard_embeddings(hidden_states)
        expected = hidden_states.chunk(4, dim=0)[2]
        torch.testing.assert_close(shard, expected)
        assert shard.data_ptr() != hidden_states.data_ptr()

    for num_tokens in (4, 256, 512, 4095):
        assert not layout.should_shard(torch.empty(num_tokens, 2))


@pytest.mark.parametrize(("num_tokens", "local_tokens"), [
    (1024, 256),
    (2048, 512),
    (4096, 1024),
])
def test_sp_shard_has_concrete_local_token_dim_under_fullgraph_dynamo(
        num_tokens: int, local_tokens: int) -> None:
    plan = build_k3_prefill_sp_group_plan(tuple(range(8)), 4)
    layout = K3PrefillSPLayout(plan=plan, sp_rank=2)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    try:

        @torch.compile(backend=backend, fullgraph=True, dynamic=False)
        def compiled(tensor: torch.Tensor) -> torch.Tensor:
            local = layout._local_shard(tensor)
            # Mirror the first downstream projection that previously emitted
            # an _unsafe_view with a symbolic T/4 dimension.
            projected = local @ torch.ones(4, 4)
            return projected.view(projected.shape[0], 2, 2)

        tensor = torch.arange(num_tokens * 4,
                              dtype=torch.float32).view(num_tokens, 4)
        torch._dynamo.mark_dynamic(tensor, 0)
        output = compiled(tensor)

        assert tuple(output.shape) == (local_tokens, 2, 2)
        view = next(node for node in graphs[0].graph.nodes
                    if node.op == "call_method" and node.target == "view")
        assert view.args[1:] == (local_tokens, 2, 2)
        assert not isinstance(view.meta["example_value"].shape[0],
                              torch.SymInt)
    finally:
        torch._dynamo.reset()


def test_sp_collective_buffers_are_derived_from_input_shapes(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class Group:
        world_size = 4
        device_group = object()

    group = Group()
    calls = []

    def all_gather_into_tensor(output, input_, *, group):
        calls.append(("ag", tuple(output.shape), group))
        output.copy_(torch.cat([input_] * 4, dim=0))

    def reduce_scatter_tensor(output, input_, *, group):
        calls.append(("rs", tuple(output.shape), group))
        output.copy_(input_.chunk(4, dim=0)[0])

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor",
                        all_gather_into_tensor)
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor",
                        reduce_scatter_tensor)

    local = torch.arange(6, dtype=torch.float32).view(2, 3)
    gathered = K3PrefillSPLayout._all_gather_dim0(group, local)
    scattered = K3PrefillSPLayout._reduce_scatter_dim0(group, gathered)

    torch.testing.assert_close(gathered, torch.cat([local] * 4, dim=0))
    torch.testing.assert_close(scattered, local)
    assert calls == [
        ("ag", (8, 3), group.device_group),
        ("rs", (2, 3), group.device_group),
    ]


def test_sp_initialization_is_explicit_and_rejects_unsupported_meshes(
) -> None:
    config = SimpleNamespace(parallel_config=SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=8,
    ))
    assert not K3PrefillSPLayout.from_vllm_config(config, sp_size=1).enabled
    with pytest.raises(ValueError, match="only supports sp_size=4"):
        K3PrefillSPLayout.from_vllm_config(config, sp_size=2)

    config.parallel_config.data_parallel_size = 4
    with pytest.raises(ValueError, match="requires DP1/PP1"):
        K3PrefillSPLayout.from_vllm_config(config, sp_size=4)


class _FakeSPLayout:

    enabled = True

    def __init__(self, sp_rank: int = 1) -> None:
        self.sp_rank = sp_rank
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        self.calls.append((name, tuple(tensor.shape)))

    def gather_next_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("next_attention_sp_ag", tensor)
        return torch.cat([tensor] * 4, dim=0)

    def reduce_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("attention_lane_ar+attention_sp_rs", tensor)
        return tensor.chunk(4, dim=0)[self.sp_rank].clone()

    def gather_moe_compat(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("moe_compat_ag", tensor)
        return torch.cat([tensor] * 4, dim=0)

    def reshard_moe_compat(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("moe_compat_reshard", tensor)
        return tensor.chunk(4, dim=0)[self.sp_rank].clone()

    def finish_standard_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        self._record("attention_tp_ar_fallback", tensor)
        return tensor * 2


class _ShapeRecorder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(tensor.shape))
        return tensor


class _AttentionRecorder(_ShapeRecorder):

    def forward(self, positions: torch.Tensor,
                tensor: torch.Tensor) -> torch.Tensor:
        assert tuple(positions.shape) == (8, )
        return super().forward(tensor)


class _ResidualRecorder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, prefix_sum: torch.Tensor,
                block_residuals: torch.Tensor) -> torch.Tensor:
        self.shapes.append(
            (tuple(prefix_sum.shape), tuple(block_residuals.shape)))
        return prefix_sum


@pytest.mark.parametrize("is_moe", [False, True], ids=["dense", "moe"])
def test_decoder_sp_contract_keeps_block_local_except_compatibility(
        is_moe: bool) -> None:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layout = _FakeSPLayout()
    layer.prefill_sp = layout
    layer.attn_res_block_size = None
    layer.input_layernorm = _ShapeRecorder()
    layer.self_attn = _AttentionRecorder()
    layer.post_attention_layernorm = _ShapeRecorder()
    layer.is_moe = is_moe
    feed_forward = _ShapeRecorder()
    if is_moe:
        layer.block_sparse_moe = feed_forward
    else:
        layer.mlp = feed_forward

    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    output, block_residuals = layer.forward_sp(torch.arange(8), hidden_states,
                                               None)

    assert tuple(output.shape) == (2, 4)
    assert block_residuals is None
    assert layer.input_layernorm.shapes == [(2, 4)]
    assert layer.self_attn.shapes == [(8, 4)]
    assert layer.post_attention_layernorm.shapes == [(2, 4)]
    assert feed_forward.shapes == [(8, 4)]
    assert layout.calls == [
        ("next_attention_sp_ag", (2, 4)),
        ("attention_lane_ar+attention_sp_rs", (8, 4)),
        ("moe_compat_ag", (2, 4)),
        ("moe_compat_reshard", (8, 4)),
    ]


class _FlexibleAttention(_ShapeRecorder):

    def forward(self, positions: torch.Tensor,
                tensor: torch.Tensor) -> torch.Tensor:
        assert positions.shape[0] == tensor.shape[0]
        return super().forward(tensor)


def test_sp_opt_in_keeps_standard_tp_reduction_for_decode_shape() -> None:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layout = _FakeSPLayout()
    layer.prefill_sp = layout
    layer.attn_res_block_size = None
    layer.input_layernorm = _ShapeRecorder()
    layer.self_attn = _FlexibleAttention()
    layer.post_attention_layernorm = _ShapeRecorder()
    layer.is_moe = True
    layer.block_sparse_moe = _ShapeRecorder()

    hidden_states = torch.ones(4, 4)
    output, block_residuals = layer(torch.arange(4), hidden_states, None)

    assert tuple(output.shape) == (4, 4)
    assert block_residuals is None
    assert layer.self_attn.shapes == [(4, 4)]
    assert layer.block_sparse_moe.shapes == [(4, 4)]
    assert layout.calls == [("attention_tp_ar_fallback", (4, 4))]


def test_attnres_stack_and_prefix_sum_remain_local() -> None:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layout = _FakeSPLayout()
    layer.prefill_sp = layout
    layer.attn_res_block_size = 2
    layer.layer_idx = 2
    layer.self_attention_res = _ResidualRecorder()
    layer.mlp_res = _ResidualRecorder()
    layer.input_layernorm = _ShapeRecorder()
    layer.self_attn = _AttentionRecorder()
    layer.post_attention_layernorm = _ShapeRecorder()
    layer.is_moe = True
    layer.block_sparse_moe = _ShapeRecorder()

    hidden_states = torch.ones(2, 4)
    block_residuals = torch.zeros(2, 1, 4)
    output, updated_residuals = layer.forward_sp(torch.arange(8),
                                                 hidden_states,
                                                 block_residuals)

    assert tuple(output.shape) == (2, 4)
    assert updated_residuals is not None
    assert tuple(updated_residuals.shape) == (2, 2, 4)
    assert layer.self_attention_res.shapes == [((2, 4), (2, 1, 4))]
    assert layer.mlp_res.shapes == [((2, 4), (2, 2, 4))]
    assert layer.input_layernorm.shapes == [(2, 4)]
    assert layer.post_attention_layernorm.shapes == [(2, 4)]


class _ModelSPLayout:

    enabled = True

    def __init__(self) -> None:
        self.shard_shape: tuple[int, ...] | None = None
        self.final_shape: tuple[int, ...] | None = None

    def should_shard(self, hidden_states: torch.Tensor) -> bool:
        return hidden_states.shape[0] == 8

    def shard_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.shard_shape = tuple(hidden_states.shape)
        return hidden_states.chunk(4, dim=0)[0].clone()

    def gather_final(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.final_shape = tuple(hidden_states.shape)
        return torch.cat([hidden_states] * 4, dim=0)


class _Embedding(nn.Module):

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([input_ids.float()] * 4, dim=-1)


class _SPDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.sp_shapes: list[tuple[int, ...]] = []
        self.block_residual_shapes: list[tuple[int, ...] | None] = []

    def forward(self, *args, **kwargs):
        raise AssertionError("SP bucket must not use the standard layer path")

    def forward_sp(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                   block_residuals: torch.Tensor | None):
        assert tuple(positions.shape) == (8, )
        self.sp_shapes.append(tuple(hidden_states.shape))
        self.block_residual_shapes.append(
            None if block_residuals is None else tuple(block_residuals.shape))
        return hidden_states + 1, block_residuals


def test_embedding_first_layer_and_final_norm_boundaries() -> None:
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    layout = _ModelSPLayout()
    decoder = _SPDecoder()
    final_norm = _ShapeRecorder()
    model.prefill_sp = layout
    model.attn_res_block_size = None
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([decoder])
    model.norm = final_norm

    # The instance intentionally bypasses KimiModel.__init__, so call the
    # implementation directly instead of the vLLM compile-decorator wrapper.
    output = model.forward(
        input_ids=torch.arange(8),
        positions=torch.arange(8),
    )

    assert layout.shard_shape == (8, 4)
    assert decoder.sp_shapes == [(2, 4)]
    assert decoder.block_residual_shapes == [None]
    assert final_norm.shapes == [(2, 4)]
    assert layout.final_shape == (2, 4)
    assert tuple(output.shape) == (8, 4)


def test_sp_attnres_empty_stack_uses_local_token_shape() -> None:
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    layout = _ModelSPLayout()
    decoder = _SPDecoder()
    output_attn_res = _ResidualRecorder()
    model.prefill_sp = layout
    model.attn_res_block_size = 2
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([decoder])
    model.output_attn_res = output_attn_res
    model.norm = _ShapeRecorder()

    output = model.forward(
        input_ids=torch.arange(8),
        positions=torch.arange(8),
    )

    assert decoder.block_residual_shapes == [(2, 0, 4)]
    assert output_attn_res.shapes == [((2, 4), (2, 0, 4))]
    assert tuple(output.shape) == (8, 4)
