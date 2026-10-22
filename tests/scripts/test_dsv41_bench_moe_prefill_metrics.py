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
"""CPU regressions for fused MoE prefill device timing."""

import argparse
import contextlib
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.dsv41_bench import bench_prefill_moe as bench

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("mismatch", [None, "jax", "torch-tpu", "commit"])
def test_runtime_requires_formal_packages_and_vllm_source(
    mismatch: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    versions = {**bench.RUNTIME_PINS, "vllm": "0.1.dev1+ge37e51dd2.empty"}
    if mismatch in ("jax", "torch-tpu"):
        versions[mismatch] = "old-version"
    monkeypatch.setattr(bench.md, "version", versions.__getitem__)
    monkeypatch.setattr(
        bench.md,
        "distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps({"url": "file:///workspace/vllm"})
        ),
    )
    monkeypatch.setattr(
        bench.subprocess,
        "check_output",
        lambda *args, **kwargs: "wrong-sha"
        if mismatch == "commit"
        else bench.VLLM_COMMIT,
    )
    check_diff = MagicMock()
    monkeypatch.setattr(bench.subprocess, "run", check_diff)
    if mismatch:
        with pytest.raises(RuntimeError, match="expected"):
            bench.check_runtime()
    else:
        result = bench.check_runtime()
        assert result["vllm_git_sha"] == bench.VLLM_COMMIT
        assert result["jax"] == "0.11.2"
        check_diff.assert_called_once_with(
            ["git", "-C", "/workspace/vllm", "diff", "--exit-code", "HEAD"], check=True
        )


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("group_sizes", [(1, 8, 8), (8, 1, 8), (1, 8, 1)])
def test_worker_requires_tp1_dp8_ep8_and_uses_distinct_token_shards(
    rank: int,
    group_sizes: tuple[int, int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = MagicMock()
    torch.isfinite.return_value.all.return_value.item.return_value = True
    mesh = SimpleNamespace(shape={"d": 8})
    pallas = MagicMock()
    pallas.jax_op.return_value.return_value.shape = (2048, 5120)
    kernel = MagicMock()
    modules = {
        "jax": SimpleNamespace(Array=object, numpy=SimpleNamespace(float32=object())),
        "jax.sharding": SimpleNamespace(PartitionSpec=lambda *axes: axes),
        "torch": torch,
        "torch_tpu._internal": SimpleNamespace(pallas=pallas),
        "vllm.distributed": SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(world_size=group_sizes[0]),
            get_dp_group=lambda: SimpleNamespace(world_size=group_sizes[1]),
            get_ep_group=lambda: SimpleNamespace(world_size=group_sizes[2]),
        ),
        "vllm_torchtpu.distributed.ep_mesh": SimpleNamespace(
            EP_AXIS_NAME="d", build_ep_mesh=lambda: mesh
        ),
        "vllm_torchtpu.kernels.fused_moe.v2": SimpleNamespace(
            WeightFormat=SimpleNamespace(FP4=object()), fused_ep_moe_v2=kernel
        ),
    }
    modules["jax.numpy"] = modules["jax"].numpy
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(bench.common, "model_wrapper_context", contextlib.nullcontext)
    monkeypatch.setattr(bench.common, "synchronize", lambda _: None)
    monkeypatch.setattr(bench.common, "check_against_golden", lambda *args: {})
    monkeypatch.setattr(bench.common, "profile_steps", lambda *args, **kwargs: None)
    monkeypatch.setattr(bench.common, "hbm_stats", lambda: {})
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    ctx = bench.common.WorkerContext(rank, 8, 1, object(), tmp_path, dp_size=8)
    args = argparse.Namespace(
        seed=0, warmup=1, profile_steps=1, trace_mode="TRACE_ONLY_XLA", dump_llo=None
    )
    if group_sizes != (1, 8, 8):
        with pytest.raises(ValueError, match="requires TP=1, DP=8 and EP=8"):
            bench.worker(ctx, args)
        torch.randint.assert_not_called()
        pallas.jax_op.assert_not_called()
        return
    result = bench.worker(ctx, args)
    assert (result["tp_size"], result["dp_size"], result["ep_size"]) == (1, 8, 8)
    assert [call.args[0] for call in torch.randn.call_args_list] == [
        (2048, 5120),
        (2048, 384),
    ]
    assert [call.args[2][0] for call in torch.randint.call_args_list] == [48, 48]
    assert result["local_tokens"] * result["ep_size"] == 16384
    assert result["token_layout"] == "sharded_no_replicas"
    invoke = pallas.jax_op.call_args.args[1]
    invoke(*pallas.jax_op.return_value.call_args.args)
    assert kernel.call_args.kwargs["mesh"] is mesh
    assert kernel.call_args.kwargs["token_replica_groups"] == tuple(
        (r,) for r in range(8)
    )


def _planes(*, timestamps: bool = True) -> list[dict[str, Any]]:
    planes = []
    names = (
        "tt_jit_call_custom_kernel_custom_kernel(123)",
        "fused_ep_moe_v2_fp4w_silu_and_mul_with_clamp_g48_c128_nb2.1",
        "moe_v2_combine_t2048_k6_b64.1",
    )
    rank_durations = (
        ((10, 1), (0.2, 0.7), (0.05, 0.1)),
        ((1, 10), (0.8, 0.3), (0.02, 0.04)),
        ((1, 1), (0.1, 0.1), (0.01, 0.01)),
    )
    for rank in range(8):
        durations = rank_durations[min(rank, 2)]
        module_events = []
        op_events = []
        for step in range(2):
            for scope, name in enumerate(names):
                event = {"name": name, "duration_ns": durations[scope][step] * 1e6}
                if timestamps:
                    event["start_ns"] = step * 20e6 + scope * 1e3
                (module_events if scope == 0 else op_events).append(event)
        planes.append(
            {
                "file": f"rank{rank}.xplane.pb",
                "plane": f"/device:TPU:{rank}",
                "lines": [
                    {"line": "XLA Modules", "events": module_events},
                    {"line": "XLA Ops", "events": op_events},
                ],
            }
        )
    return planes


@pytest.mark.parametrize("timestamps", [True, False])
def test_device_time_uses_slowest_core_per_step(timestamps: bool) -> None:
    result = bench.summarize_device_events(_planes(timestamps=timestamps), 2)
    assert result["primary_metric"] == "module"
    assert result["timing_source"] == "XProf device events only"
    assert result["metrics"]["module"] == {
        "device_median_ms": 10.0,
        "slowest_core_samples_ms": [10.0, 10.0],
    }
    assert result["metrics"]["pallas_main"]["slowest_core_samples_ms"] == [0.8, 0.7]
    assert result["metrics"]["pallas_main"]["device_median_ms"] == pytest.approx(0.75)
    assert result["metrics"]["combine"]["device_median_ms"] == pytest.approx(0.075)


def test_full_hlo_names_match_short_names_without_matching_operands() -> None:
    planes = _planes()
    expected = bench.summarize_device_events(planes, 2)
    for plane in planes:
        events = plane["lines"][1]["events"]
        for event in events:
            event["name"] = f"%{event['name']} = f32[1] custom-call(%input)"
        events.append(
            {
                "name": "%broadcast_select_fusion = f32[1] fusion("
                "%moe_v2_combine_t2048_k6_b64.1, "
                "%fused_ep_moe_v2_fp4w_silu_and_mul_with_clamp_g48_c128_nb2.1)",
                "start_ns": 2e3,
                "duration_ns": 100e6,
            }
        )
    assert bench.summarize_device_events(planes, 2) == expected


def test_timestamps_align_steps_after_event_order_changes() -> None:
    planes = _planes()
    expected = bench.summarize_device_events(planes, 2)
    for plane in planes[::2]:
        for line in plane["lines"]:
            line["events"].reverse()
    assert bench.summarize_device_events(planes, 2) == expected


def test_mixed_timestamp_presence_is_rejected() -> None:
    planes = _planes()
    del planes[0]["lines"][1]["events"][0]["start_ns"]
    with pytest.raises(ValueError, match="mixed missing start_ns"):
        bench.summarize_device_events(planes, 2)


@pytest.mark.parametrize("layout", ["missing", "extra", "duplicate"])
def test_incomplete_or_duplicated_core_capture_is_rejected(layout: str) -> None:
    planes = _planes()
    if layout == "missing":
        planes.pop()
    elif layout == "extra":
        planes.append(copy.deepcopy(planes[0]))
    else:
        planes[-1] = copy.deepcopy(planes[0])
    with pytest.raises(ValueError, match="unique TensorCore planes"):
        bench.summarize_device_events(planes, 2)


@pytest.mark.parametrize("line,event", [(0, 0), (1, 0), (1, 1)])
def test_missing_step_is_rejected(line: int, event: int) -> None:
    planes = _planes()
    planes[0]["lines"][line]["events"].pop(event)
    with pytest.raises(ValueError, match="expected 2 events, got 1"):
        bench.summarize_device_events(planes, 2)


@pytest.mark.parametrize("duration_ns", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("line,event", [(0, 0), (1, 0), (1, 1)])
def test_invalid_device_duration_is_rejected(
    duration_ns: float, line: int, event: int
) -> None:
    planes = _planes()
    planes[0]["lines"][line]["events"][event]["duration_ns"] = duration_ns
    with pytest.raises(ValueError, match="Invalid .* duration"):
        bench.summarize_device_events(planes, 2)


def test_non_tensorcore_planes_are_ignored() -> None:
    planes = _planes()
    expected = bench.summarize_device_events(planes, 2)
    planes.extend(
        {"plane": name} for name in ("/host:CPU", "/device:TPU:0 SparseCore 0")
    )
    assert bench.summarize_device_events(planes, 2) == expected


@pytest.mark.parametrize("profile_steps", [0, -1])
def test_nonpositive_profile_steps_are_rejected(profile_steps: int) -> None:
    with pytest.raises(ValueError, match="profile_steps must be positive"):
        bench.summarize_device_events(_planes(), profile_steps)


def test_collect_preserves_device_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xplane = tmp_path / "rank0.xplane.pb"
    xplane.touch()
    event = SimpleNamespace(
        name="moe_v2_combine_t2048_k6_b64.1", start_ns=123, duration_ns=45
    )
    line = SimpleNamespace(name="XLA Ops", events=[event])
    plane = SimpleNamespace(name="/device:TPU:0", lines=[line])
    profiler = ModuleType("jax.profiler")
    monkeypatch.setattr(
        profiler,
        "ProfileData",
        SimpleNamespace(from_file=lambda _: SimpleNamespace(planes=[plane])),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "jax", ModuleType("jax"))
    monkeypatch.setitem(sys.modules, "jax.profiler", profiler)
    collected = bench.collect_device_events(tmp_path)
    assert collected[0]["lines"][0]["events"] == [
        {"name": event.name, "start_ns": 123, "duration_ns": 45}
    ]


@pytest.mark.parametrize("existing", [False, True])
def test_runtime_config_overrides_p_state_and_tracing_idempotently(
    existing: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_flags = ["--xla_jf_dump_to=/tmp/moe llo", "--xla_jf_dump_llo_text=true"]
    if existing:
        monkeypatch.setenv(
            "LIBTPU_INIT_ARGS",
            shlex.join(
                other_flags
                + [
                    "--xla_tpu_dvfs_p_state=3",
                    "--xla_tpu_dvfs_p_state=5",
                    "--xla_xprof_enable_custom_call_tracing=false",
                    "--xla_xprof_register_llo_debug_info=false",
                ]
            ),
        )
    else:
        monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    bench.configure_runtime_env()
    first = os.environ["LIBTPU_INIT_ARGS"]
    bench.configure_runtime_env()
    assert os.environ["LIBTPU_INIT_ARGS"] == first
    flags = shlex.split(first)
    assert flags == (other_flags if existing else []) + [
        "--xla_tpu_dvfs_p_state=7",
        "--xla_xprof_enable_custom_call_tracing=true",
        "--xla_xprof_register_llo_debug_info=true",
    ]


@pytest.mark.parametrize("trace_mode", [None, "TRACE_COMPUTE"])
def test_main_defaults_to_tp1_dp8_ep8_with_sharded_tokens(
    trace_mode: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    profile = tmp_path / "profile"
    launched: dict[str, argparse.Namespace] = {}

    def run_main(name: str, parser: argparse.ArgumentParser, worker: Any) -> None:
        assert "--xla_tpu_dvfs_p_state=7" in shlex.split(os.environ["LIBTPU_INIT_ARGS"])
        launched["args"] = parser.parse_args()

    monkeypatch.setattr(bench.common, "run_main", run_main)
    monkeypatch.setattr(bench, "check_runtime", lambda: {})
    monkeypatch.setattr(bench.common, "git_sha", lambda: "test-head")
    monkeypatch.setattr(bench, "collect_device_events", lambda _: _planes())
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_prefill_moe",
            "--no-verify",
            "--profile",
            str(profile),
            "--result-dir",
            str(results),
            "--profile-steps",
            "2",
        ]
        + (["--trace-mode", trace_mode] if trace_mode else []),
    )
    bench.main()
    assert (launched["args"].tp_size, launched["args"].dp_size) == (1, 8)
    expected_trace_mode = trace_mode or "TRACE_ONLY_XLA"
    assert launched["args"].trace_mode == expected_trace_mode
    metrics = json.loads((results / "device-metrics.json").read_text())
    assert (metrics["tp_size"], metrics["dp_size"], metrics["ep_size"]) == (1, 8, 8)
    assert (metrics["global_tokens"], metrics["local_tokens"]) == (16384, 2048)
    assert metrics["token_layout"] == "sharded_no_replicas"
    assert metrics["primary_metric"] == "module"
    assert metrics["replay_only"] is False
    assert metrics["trace_mode"] == expected_trace_mode
    assert "BENCHMARK_OK" in capsys.readouterr().out


def test_replay_does_not_infer_capture_settings_from_current_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replay = tmp_path / "events.json"
    replay.write_text(json.dumps(_planes()))
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_prefill_moe",
            "--replay-device-events",
            str(replay),
            "--result-dir",
            str(results),
            "--profile-steps",
            "2",
        ],
    )
    bench.main()
    metrics = json.loads((results / "device-metrics.json").read_text())
    assert "dp_size" not in metrics
    assert "tp_size" not in metrics
    assert "trace_mode" not in metrics
    assert metrics["primary_metric"] == "module"
    assert metrics["replay_only"] is True
    assert "REPLAY_OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "override", [("--tp-size", "8"), ("--dp-size", "1"), ("--global-tokens", "2048")]
)
def test_main_rejects_other_workloads(
    override: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_prefill_moe", *override])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    assert (
        "requires --global-tokens 16384 --tp-size 1 --dp-size 8"
        in capsys.readouterr().err
    )


@pytest.fixture
def captured_run(tmp_path: Path) -> Path:
    run = tmp_path / "capture"
    results = run / "baseline" / "results"
    results.mkdir(parents=True)
    planes = _planes()
    for rank, plane in enumerate(planes):
        plane["file"] = f"rank{rank}_host.xplane.pb"
        trace = run / "baseline/xprof/plugins/profile/session" / plane["file"]
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(f"trace-{rank}".encode())
        compiler = run / "baseline/compiler" / f"rank{rank}"
        compiler.mkdir(parents=True)
        (compiler / "kernel.llo.txt").write_text(f"llo-{rank}")
        (results / f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "tp_size": 1,
                    "dp_size": 8,
                    "ep_size": 8,
                    "libtpu_init_args": "--xla_tpu_dvfs_p_state=7",
                }
            )
        )
    metrics = bench.summarize_device_events(planes, 2)
    metrics.update(replay_only=False, tp_size=1, dp_size=8)
    (results / "device-events.json").write_text(json.dumps(planes))
    (results / "device-metrics.json").write_text(json.dumps(metrics))
    (run / "smoke/results").mkdir(parents=True)
    (run / "smoke/results/summary.json").write_text("{}")
    (run / "golden").mkdir()
    (run / "golden/reference.pt").write_bytes(b"golden")
    (run / "smoke/ignored.xplane.pb").write_bytes(b"smoke")
    for phase in ("smoke", "baseline"):
        (run / phase / "bench.log").write_text(phase)
    for name in ("runtime-versions.json", "source-files.json"):
        (run / name).write_text("{}")
    (run / "python-packages.txt").write_text("jax==test\n")
    (run / "exit-status.txt").write_text("0\n")
    return run


def test_export_has_root_contract_and_distinct_rank_measurements(
    captured_run: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    bench.export_falcon_artifact(captured_run, artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["workflow"] == "operator-optimization"
    assert manifest["operator_name"] == "fused_ep_moe_v2"
    assert manifest["dimensions"] == {
        "global_tokens": 16384,
        "local_tokens": 2048,
        "tp_size": 1,
        "dp_size": 8,
        "ep_size": 8,
    }
    assert manifest["rank_coverage"] == list(range(8))
    assert len(list(artifact.rglob("*.xplane.pb"))) == 8
    assert len(list(artifact.rglob("metrics.jsonl"))) == 8
    for rank in range(8):
        root = artifact / f"rank-{rank}"
        rows = [
            json.loads(line)
            for line in (root / "benchmark/metrics.jsonl").read_text().splitlines()
        ]
        expected = ([10, 1], [1, 10], [1, 1])[min(rank, 2)]
        assert [row["latency_ms"] for row in rows] == expected
        assert all(row["rank"] == rank and row["tokens"] == 2048 for row in rows)
        assert (root / "compiler/llo/kernel.llo.txt").read_text() == f"llo-{rank}"
        trace = next((root / "profiling/xprof").rglob("*.xplane.pb"))
        assert trace.read_bytes() == f"trace-{rank}".encode()
    assert (artifact / "profiling/python-packages.txt").read_text() == "jax==test\n"
    assert "rank-7 LIBTPU_INIT_ARGS=" in (artifact / "profiling/env.txt").read_text()


@pytest.mark.parametrize("missing", ["llo", "trace", "rank", "steps", "replay"])
def test_export_rejects_incomplete_or_replayed_capture(
    captured_run: Path, tmp_path: Path, missing: str
) -> None:
    baseline = captured_run / "baseline"
    if missing == "llo":
        (baseline / "compiler/rank7/kernel.llo.txt").write_text("")
    elif missing == "trace":
        next((baseline / "xprof").rglob("rank7_*.xplane.pb")).unlink()
    elif missing in ("rank", "steps"):
        path = baseline / "results/device-events.json"
        planes = json.loads(path.read_text())
        if missing == "rank":
            planes[-1]["file"] = planes[0]["file"]
        else:
            planes[-1]["lines"][0]["events"].pop()
        path.write_text(json.dumps(planes))
    else:
        path = baseline / "results/device-metrics.json"
        metrics = json.loads(path.read_text())
        metrics["replay_only"] = True
        path.write_text(json.dumps(metrics))
    artifact = tmp_path / "artifact"
    with pytest.raises(ValueError):
        bench.export_falcon_artifact(captured_run, artifact)
    assert not (artifact / "manifest.json").exists()


@pytest.mark.parametrize(
    "exit_code,failure", [(0, None), (17, None), (0, "llo"), (0, "copy")]
)
def test_shell_publishes_contract_or_preserves_failed_run(
    captured_run: Path, tmp_path: Path, exit_code: int, failure: str | None
) -> None:
    artifact = tmp_path / "mounted-artifact"
    artifact.mkdir()
    if failure == "llo":
        (captured_run / "baseline/compiler/rank7/kernel.llo.txt").unlink()
    elif failure == "copy":
        (artifact / "rank-7").write_text("blocked destination")
    script = Path(bench.__file__).with_name("run_prefill_moe.sh").read_text()
    publisher = script[script.index("publish() {") : script.index("trap publish EXIT")]
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail\n{publisher}\ntrap publish EXIT\nexit {exit_code}",
        ],
        env={
            **os.environ,
            "run_local": str(captured_run),
            "run_artifact": str(artifact),
            "MOE_RUN_NAME": captured_run.name,
            "PYTHONPATH": str(Path(bench.__file__).resolve().parents[2]),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (1 if failure else exit_code), result.stderr
    if result.returncode == 0:
        assert (artifact / "manifest.json").is_file()
        assert len(list(artifact.rglob("metrics.jsonl"))) == 8
    else:
        assert not (artifact / "manifest.json").exists()
        failed = artifact / "failed" / captured_run.name
        assert (failed / "baseline/bench.log").read_text() == "baseline"
        assert (failed / "exit-status.txt").read_text() == f"{result.returncode}\n"
