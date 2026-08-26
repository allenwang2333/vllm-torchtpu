# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed.sequence_parallel import (
    SequenceParallelMesh, build_sequence_parallel_group_plan,
    resolve_sequence_parallel_size)


def _config(*, native=None, additional=None):
    parallel_config = SimpleNamespace()
    if native is not None:
        parallel_config.sequence_parallel_size = native
    return SimpleNamespace(
        parallel_config=parallel_config,
        additional_config={} if additional is None else additional,
    )


@pytest.mark.parametrize(
    ("world_size", "expected_lane_size", "expected_lane_groups",
     "expected_sequence_groups"),
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
def test_group_plan_scales_across_tp_sizes(
    world_size,
    expected_lane_size,
    expected_lane_groups,
    expected_sequence_groups,
):
    plan = build_sequence_parallel_group_plan(tuple(range(world_size)), 4)

    assert plan.sequence_parallel_size == 4
    assert plan.lane_size == expected_lane_size
    assert plan.lane_groups == expected_lane_groups
    assert plan.sequence_groups == expected_sequence_groups


@pytest.mark.parametrize("tp_ranks,sp_size,match", [
    ((), 4, "must not be empty"),
    ((0, 0, 1, 2), 4, "must be unique"),
    ((0, 1, 2), 2, "not divisible"),
    ((0, 1), 1, "greater than 1"),
])
def test_group_plan_rejects_invalid_meshes(tp_ranks, sp_size, match):
    with pytest.raises(ValueError, match=match):
        build_sequence_parallel_group_plan(tp_ranks, sp_size)


@pytest.mark.parametrize("world_size", [8, 32])
def test_lane_reduce_then_sequence_scatter_matches_tp_sum(
        world_size: int) -> None:
    plan = build_sequence_parallel_group_plan(tuple(range(world_size)), 4)
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
    for group in plan.sequence_groups:
        reduced = sum((lane_reduced[rank] for rank in group),
                      torch.zeros_like(lane_reduced[group[0]]))
        for sequence_rank, rank in enumerate(group):
            outputs[rank] = reduced.chunk(4, dim=0)[sequence_rank]

    tp_sum = sum((partials[rank] for rank in plan.tp_ranks),
                 torch.zeros_like(partials[0]))
    for lane_rank in range(plan.lane_size):
        for sequence_rank in range(plan.sequence_parallel_size):
            rank = lane_rank * plan.sequence_parallel_size + sequence_rank
            torch.testing.assert_close(
                outputs[rank],
                tp_sum.chunk(4, dim=0)[sequence_rank],
            )


@pytest.mark.parametrize(("num_tokens", "local_tokens"), [
    (4, 1),
    (1024, 256),
    (2048, 512),
    (4096, 1024),
])
@pytest.mark.parametrize("sequence_rank", range(4))
def test_shard_has_concrete_local_token_dim_under_fullgraph_dynamo(
        num_tokens: int, local_tokens: int, sequence_rank: int) -> None:
    plan = build_sequence_parallel_group_plan(tuple(range(8)), 4)
    mesh = SequenceParallelMesh(
        plan=plan,
        tp_group=None,
        lane_group=None,
        sequence_group=None,
        lane_rank=0,
        sequence_rank=sequence_rank,
    )
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    try:

        @torch.compile(backend=backend, fullgraph=True, dynamic=False)
        def compiled(tensor: torch.Tensor) -> torch.Tensor:
            local = mesh.shard_tokens(
                tensor,
                full_token_sizes=(4, 1024, 2048, 4096),
            )
            projected = local @ torch.ones(4, 4)
            return projected.view(projected.shape[0], 2, 2)

        tensor = torch.arange(num_tokens * 4,
                              dtype=torch.float32).view(num_tokens, 4)
        torch._dynamo.mark_dynamic(tensor, 0)
        output = compiled(tensor)

        assert tuple(output.shape) == (local_tokens, 2, 2)
        expected = tensor.chunk(4, dim=0)[sequence_rank] @ torch.ones(4, 4)
        torch.testing.assert_close(output.view(local_tokens, 4), expected)
        view = next(node for node in graphs[0].graph.nodes
                    if node.op == "call_method" and node.target == "view")
        assert view.args[1:] == (local_tokens, 2, 2)
        assert not isinstance(view.meta["example_value"].shape[0],
                              torch.SymInt)
    finally:
        torch._dynamo.reset()


def test_collective_buffers_are_derived_from_input_shapes(
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
    gathered = SequenceParallelMesh._all_gather_dim0(group, local)
    scattered = SequenceParallelMesh._reduce_scatter_dim0(group, gathered)

    torch.testing.assert_close(gathered, torch.cat([local] * 4, dim=0))
    torch.testing.assert_close(scattered, local)
    assert calls == [
        ("ag", (8, 3), group.device_group),
        ("rs", (2, 3), group.device_group),
    ]


def test_size_defaults_to_disabled():
    assert resolve_sequence_parallel_size(_config()) == 1


def test_size_uses_standard_additional_config_path():
    config = _config(additional={"sequence_parallel_size": 4})
    assert resolve_sequence_parallel_size(config) == 4


def test_future_native_field_takes_precedence_over_disabled_default():
    native = _config(native=4)
    compatible_fallback = _config(
        native=1,
        additional={"sequence_parallel_size": 4},
    )

    assert resolve_sequence_parallel_size(native) == 4
    assert resolve_sequence_parallel_size(compatible_fallback) == 4


def test_conflicting_standard_sources_fail():
    config = _config(native=4, additional={"sequence_parallel_size": 2})
    with pytest.raises(ValueError, match="Conflicting sequence parallel"):
        resolve_sequence_parallel_size(config)


@pytest.mark.parametrize("value", [0, -1, True, "4", 4.0])
def test_size_must_be_a_positive_json_integer(value):
    config = _config(additional={"sequence_parallel_size": value})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_sequence_parallel_size(config)
