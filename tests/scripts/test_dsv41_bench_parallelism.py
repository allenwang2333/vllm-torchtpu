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
"""CPU checks for torchrun rank handling in the component benchmark harness."""

import argparse
import contextlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.dsv41_bench import common

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def runtime(monkeypatch):
    active_config = None
    world_configs = []
    model_configs = []

    @contextlib.contextmanager
    def config_context(config):
        nonlocal active_config
        active_config = config
        try:
            yield config
        finally:
            active_config = None

    dist = ModuleType("torch.distributed")
    dist.init_process_group = Mock()
    dist.new_group = Mock()
    torch = ModuleType("torch")
    torch.distributed = dist
    torch.empty = Mock()
    torch.manual_seed = Mock()
    config = ModuleType("vllm.config")
    config.ParallelConfig = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    config.VllmConfig = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    config.set_current_vllm_config = config_context
    vllm_dist = ModuleType("vllm.distributed")
    vllm_dist.init_distributed_environment = Mock(
        side_effect=lambda **_: world_configs.append(active_config)
    )
    vllm_dist.ensure_model_parallel_initialized = Mock(
        side_effect=lambda *_: model_configs.append(active_config)
    )
    for name, module in {
        "torch": torch,
        "torch.distributed": dist,
        "torch_tpu": ModuleType("torch_tpu"),
        "vllm": ModuleType("vllm"),
        "vllm.config": config,
        "vllm.distributed": vllm_dist,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    return SimpleNamespace(
        dist=dist,
        config=config,
        vllm_dist=vllm_dist,
        world_configs=world_configs,
        model_configs=model_configs,
    )


def test_dp_cli_default_and_override():
    parser = argparse.ArgumentParser()
    common.add_common_args(parser, ("moe_only",))
    assert parser.parse_args([]).dp_size == 1
    args = parser.parse_args(["--tp-size", "1", "--dp-size", "8"])
    assert (args.tp_size, args.dp_size) == (1, 8)


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "rank"), [(8, 1, 0), (1, 8, 0), (1, 8, 7), (2, 4, 3)]
)
def test_worker_keeps_global_world_and_configures_dp(
    runtime, monkeypatch, tmp_path, tp_size, dp_size, rank
):
    monkeypatch.setenv("RANK", str(rank))
    args = argparse.Namespace(
        tp_size=tp_size, dp_size=dp_size, seed=0, result_dir=tmp_path
    )
    ctx = common.init_worker(args)
    assert (ctx.rank, ctx.world_size, ctx.tp_size, ctx.dp_size) == (
        rank,
        8,
        tp_size,
        dp_size,
    )
    runtime.dist.init_process_group.assert_called_once_with(backend="tpu_dist")
    runtime.vllm_dist.init_distributed_environment.assert_called_once_with(
        world_size=8,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    assert runtime.world_configs == [None]
    runtime.vllm_dist.ensure_model_parallel_initialized.assert_called_once_with(
        tp_size, 1
    )
    assert len(runtime.model_configs) == 1
    if dp_size == 1:
        runtime.config.VllmConfig.assert_called_once_with()
        runtime.config.ParallelConfig.assert_not_called()
    else:
        parallel = runtime.model_configs[0].parallel_config
        assert parallel.tensor_parallel_size == tp_size
        assert parallel.data_parallel_size == dp_size
        assert parallel.data_parallel_size_local == dp_size
        assert parallel.data_parallel_rank == rank // tp_size
        assert parallel.data_parallel_rank_local == rank // tp_size
        assert parallel.enable_expert_parallel
    runtime.dist.new_group.assert_called_once_with(ranks=list(range(8)), backend="gloo")


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "message"),
    [
        (0, 1, "must be positive"),
        (1, 0, "must be positive"),
        (1, -1, "must be positive"),
        (5, 1, "not divisible"),
        (2, 8, "not divisible"),
    ],
)
def test_invalid_parallelism_fails_before_collectives(
    runtime, tp_size, dp_size, message
):
    with pytest.raises(ValueError, match=message):
        common.init_worker(argparse.Namespace(tp_size=tp_size, dp_size=dp_size))
    runtime.dist.init_process_group.assert_not_called()


@pytest.mark.parametrize("rank", [0, 7])
def test_profile_waits_for_all_ranks_before_stopping(monkeypatch, tmp_path, rank):
    events = []
    cpu_group = object()

    @contextlib.contextmanager
    def capture(**kwargs):
        events.append("start")
        yield
        events.append("stop")

    def barrier(*, group):
        assert group is cpu_group
        events.append("barrier")

    dist = ModuleType("torch.distributed")
    dist.barrier = barrier
    torch = ModuleType("torch")
    torch.distributed = dist
    torch.profiler = SimpleNamespace(
        profile=capture,
        ProfilerActivity=SimpleNamespace(CPU="cpu", PrivateUse1="tpu"),
        tensorboard_trace_handler=lambda **kwargs: None,
        record_function=lambda _: contextlib.nullcontext(),
    )
    tpu_profiler = ModuleType("torch_tpu._internal.profiler")
    tpu_profiler.TpuProfilerConfig = lambda **kwargs: kwargs
    plugin = ModuleType("vllm_torchtpu")
    plugin.profiler_trace = SimpleNamespace(
        rank_capture_dir=lambda root, rank: str(tmp_path / f"rank{rank}"),
        resolve_canonical_dst_ts=lambda *args, **kwargs: "timestamp",
        merge_rank_capture=lambda *args, **kwargs: events.append("merge"),
        clear_canonical_ts_marker=lambda *args: events.append("clear"),
    )
    for name, module in {
        "torch": torch,
        "torch.distributed": dist,
        "torch_tpu": ModuleType("torch_tpu"),
        "torch_tpu._internal": ModuleType("torch_tpu._internal"),
        "torch_tpu._internal.profiler": tpu_profiler,
        "vllm_torchtpu": plugin,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(common._PROFILE_SESSION_ENV, "test_session")
    monkeypatch.setattr(common, "synchronize", lambda _: events.append("synchronize"))
    ctx = common.WorkerContext(rank, 8, 1, cpu_group, tmp_path, dp_size=8)
    args = argparse.Namespace(
        profile=tmp_path / "profile", trace_mode="TRACE_COMPUTE", profile_steps=2
    )
    common.profile_steps(ctx, args, lambda: events.append("step"), label="test")

    stop = events.index("stop")
    final_sync = max(i for i, event in enumerate(events) if event == "synchronize")
    assert events.count("step") == events.count("synchronize") == 2
    assert events[final_sync + 1 : stop] == ["barrier"]
    assert events[stop + 1 : events.index("merge")] == ["barrier"]
