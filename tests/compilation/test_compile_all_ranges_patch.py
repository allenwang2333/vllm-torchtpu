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
"""How TPU_PARALLEL_PRECOMPILE splits a subgraph's ranges among a host's ranks.

Each range has one owner on the host, which compiles it; the other ranks
compile their own ranges first, then wait for and replay the owners' artifacts.
The biggest range takes far longer than the others, so startup is bounded by
when it starts: local rank 0 must start on it, not reach it after the smallest.
"""

from types import SimpleNamespace

import pytest
from vllm.compilation import piecewise_backend
from vllm.config.utils import Range
from vllm.distributed import parallel_state

from vllm_torchtpu import _patch_vllm_compile_all_ranges
from vllm_torchtpu.compilation import host_split, tpu_compiler

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# (range end, pieces, this rank's piece) of each bucket the last _compile
# built inside host_split.split_across_host.
splits = []


@pytest.fixture
def compile_all_ranges(monkeypatch):
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setattr(
        piecewise_backend, "create_concrete_args", lambda graph, size: [size]
    )
    # Restored on teardown, so the patch does not leak into other tests.
    monkeypatch.setattr(
        piecewise_backend.PiecewiseBackend,
        "compile_all_ranges",
        piecewise_backend.PiecewiseBackend.compile_all_ranges,
    )
    _patch_vllm_compile_all_ranges()
    return piecewise_backend.PiecewiseBackend.compile_all_ranges


def _host(monkeypatch, ranks_per_host):
    """A world of two hosts with `ranks_per_host` ranks each, or an unknown
    one for None."""
    if ranks_per_host is None:

        def get_world_group():
            raise AssertionError("world group is not initialized")

    else:

        def get_world_group():
            return SimpleNamespace(world_size=2 * ranks_per_host)

    monkeypatch.setattr(parallel_state, "get_world_group", get_world_group)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 2)


def _compile(compile_all_ranges, monkeypatch, local_rank, ranks_per_host=8, graph=None):
    """(range end, peer wait in seconds) per compile, in order."""
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    calls = []
    splits.clear()

    def compile(graph, args, *rest, compile_range, **kwargs):
        calls.append(
            (compile_range.end, getattr(tpu_compiler._peer_wait, "timeout_s", 0.0))
        )
        plan = host_split.current_plan()
        if plan is not None:
            splits.append((compile_range.end, plan.pieces, plan.piece))
        return lambda *a: a

    # In vLLM's order: smallest range first.
    backend = SimpleNamespace(
        graph=object() if graph is None else graph,
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(compile=compile),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    compile_all_ranges(backend)
    return calls


def _owned(calls):
    return [end for end, wait in calls if wait == 0.0]


def test_local_rank_0_starts_on_the_biggest_range(compile_all_ranges, monkeypatch):
    assert _compile(compile_all_ranges, monkeypatch, 0)[0][0] == 4096


def test_each_range_has_one_owner_per_host(compile_all_ranges, monkeypatch):
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # Dealt in a snake: the rank with the smallest first range gets the 9th.
    assert owned == [[4096], [2048], [1024], [512], [256], [128], [64], [32, 16]]


def test_other_ranges_wait_for_their_owner(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 0)
    assert calls == [(4096, 0.0)] + [
        (end, tpu_compiler.PEER_WAIT_S)
        for end in [32, 64, 128, 256, 512, 1024, 2048, 16]
    ]


def test_every_rank_compiles_every_range(compile_all_ranges, monkeypatch):
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


def test_a_smaller_host_splits_the_ranges_among_its_ranks(
    compile_all_ranges, monkeypatch
):
    owned = [
        _owned(_compile(compile_all_ranges, monkeypatch, r, ranks_per_host=4))
        for r in range(4)
    ]
    assert owned == [[4096, 32, 16], [2048, 64], [1024, 128], [512, 256]]


def test_no_wait_when_the_host_size_is_unknown(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 1, ranks_per_host=None)
    assert all(wait == 0.0 for _, wait in calls)
    assert sorted(end for end, _ in calls) == SIZES


def test_biggest_first_off_deals_in_vllm_order(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "0")
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # vLLM's order, smallest first: local rank 0 starts on the smallest range.
    assert owned == [[16], [32], [64], [128], [256], [512], [1024], [2048, 4096]]
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


@pytest.mark.parametrize(
    "biggest_first, firsts",
    [
        ("1", [4096, 2048, 1024, 512, 256, 128, 64, 32]),
        ("0", SIZES[:8]),
    ],
)
def test_peer_replay_off_compiles_every_range_without_waiting(
    compile_all_ranges, monkeypatch, biggest_first, firsts
):
    """TPU_PRECOMPILE_PEER_REPLAY=0: no owners and no waits; each local rank
    starts on a different range and compiles every range itself."""
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "0")
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", biggest_first)
    runs = [_compile(compile_all_ranges, monkeypatch, r) for r in range(8)]
    assert [calls[0][0] for calls in runs] == firsts
    for calls in runs:
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES


def _decoder_graph(layers=8):
    """A traced graph with `layers` decoder layers, as Dynamo records them."""
    import torch

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    for layer in range(layers):
        x = graph.call_function(torch.relu, (x,))
        x.meta["nn_module_stack"] = {"k": (f"L['self'].layers.{layer}", None)}
    graph.output(x)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_big_buckets_are_compiled_by_every_rank_together(
    compile_all_ranges, monkeypatch
):
    """TPU_PRECOMPILE_SPLIT_MIN_TOKENS=1024: every rank starts on 4096, 2048
    and 1024, one piece each, and only then deals out the rest."""
    monkeypatch.setenv("TPU_PRECOMPILE_SPLIT_MIN_TOKENS", "1024")
    owned = []
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank, graph=_decoder_graph())
        assert [end for end, _ in calls[:3]] == [4096, 2048, 1024]
        assert splits == [(4096, 8, rank), (2048, 8, rank), (1024, 8, rank)]
        assert sorted(end for end, _ in calls) == SIZES
        owned.append(_owned(calls[3:]))
    assert owned == [[512], [256], [128], [64], [32], [16], [], []]


def test_a_graph_without_layers_is_dealt_out_whole(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_SPLIT_MIN_TOKENS", "1024")
    owned = [
        _owned(_compile(compile_all_ranges, monkeypatch, r, graph=_decoder_graph(1)))
        for r in range(8)
    ]
    assert splits == []
    assert owned == [[4096], [2048], [1024], [512], [256], [128], [64], [32, 16]]


def test_no_split_when_the_host_size_is_unknown(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_SPLIT_MIN_TOKENS", "1024")
    calls = _compile(
        compile_all_ranges,
        monkeypatch,
        0,
        ranks_per_host=None,
        graph=_decoder_graph(),
    )
    assert splits == []
    assert sorted(end for end, _ in calls) == SIZES


def test_split_off_by_default(compile_all_ranges, monkeypatch):
    owned = [
        _owned(_compile(compile_all_ranges, monkeypatch, r, graph=_decoder_graph()))
        for r in range(8)
    ]
    assert splits == []
    assert owned == [[4096], [2048], [1024], [512], [256], [128], [64], [32, 16]]
