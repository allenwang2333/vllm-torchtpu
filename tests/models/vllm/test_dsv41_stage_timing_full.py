# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for stage_timing.py."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.runner", ("vllm_torchtpu", "runner")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg


class _FakeTpuEvent:
    def __init__(self) -> None:
        self.recorded = 0
        self.synced = 0

    def record(self) -> None:
        self.recorded += 1

    def synchronize(self) -> None:
        self.synced += 1


if "torch_tpu._internal.stream.streams" not in sys.modules:
    _streams_mod = types.ModuleType("torch_tpu._internal.stream.streams")
    _streams_mod.TpuEvent = _FakeTpuEvent  # type: ignore[attr-defined]
    sys.modules.setdefault("torch_tpu", types.ModuleType("torch_tpu"))
    sys.modules.setdefault(
        "torch_tpu._internal", types.ModuleType("torch_tpu._internal")
    )
    sys.modules.setdefault(
        "torch_tpu._internal.stream", types.ModuleType("torch_tpu._internal.stream")
    )
    sys.modules["torch_tpu._internal.stream.streams"] = _streams_mod

from vllm_torchtpu.runner import stage_timing  # noqa: E402


def test_pending_timing_gate_and_cached_stage_contexts() -> None:
    with patch.object(stage_timing, "TpuEvent", _FakeTpuEvent):
        timer = stage_timing.RunnerStageTiming(rank=2)

        # Repeated host_stage calls reuse the cached _HostStageContext instance
        ctx_h1 = timer.host_stage("prepare_inputs")
        ctx_h2 = timer.host_stage("prepare_inputs")
        assert ctx_h1 is ctx_h2
        with ctx_h1:
            pass
        with timer.host_stage("prepare_inputs"):
            pass

        # Repeated stage calls reuse the cached _StageContext instance
        ctx_s1 = timer.stage("model_forward")
        ctx_s2 = timer.stage("model_forward")
        assert ctx_s1 is ctx_s2
        with ctx_s1:
            pass
        with timer.stage("model_forward"):
            pass

        # Exception inside stage body still closes pending timing and records CPU sample
        with (
            pytest.raises(RuntimeError, match="boom"),
            timer.stage("failing_forward"),
        ):
            raise RuntimeError("boom")

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with timer._samples_lock:
                if len(timer._device_samples) >= 3:
                    break
            time.sleep(0.002)

        # Fallback path when _slots.acquire returns False
        with (
            patch.object(timer._slots, "acquire", return_value=False),
            timer.stage("model_forward"),
        ):
            pass

        cpu_samples, dev_samples = timer.take_samples()
        assert [s[1] for s in cpu_samples] == [
            "prepare_inputs",
            "prepare_inputs",
            "model_forward",
            "model_forward",
            "failing_forward",
            "model_forward",
        ]
        assert all(s[0] == "2" for s in cpu_samples)
        assert [s[1] for s in dev_samples] == [
            "model_forward",
            "model_forward",
            "failing_forward",
        ]
        assert all(s[0] == "2" for s in dev_samples)
        assert timer.take_samples() == ([], [])


def test_collect_device_time_direct_main_thread() -> None:
    with patch.object(stage_timing, "TpuEvent", _FakeTpuEvent):
        timer = stage_timing.RunnerStageTiming(rank=7)
        timer._slots.acquire(False)
        start_ev = _FakeTpuEvent()
        end_ev = _FakeTpuEvent()
        item = stage_timing._PendingTiming("direct_collect", start_ev, end_ev)  # type: ignore[arg-type]
        assert item.end_recorded is item
        item.end_recorded.set()

        timer._pending = SimpleNamespace(  # type: ignore[assignment]
            get=MagicMock(side_effect=[item, StopIteration("collector_done")])
        )
        with pytest.raises(StopIteration, match="collector_done"):
            timer._collect_device_time()

        assert start_ev.synced == 1
        assert end_ev.synced == 1
        cpu_samples, dev_samples = timer.take_samples()
        assert cpu_samples == []
        assert len(dev_samples) == 1
        assert dev_samples[0][0] == "7"
        assert dev_samples[0][1] == "direct_collect"
        assert dev_samples[0][2] >= 0.0
