# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host-path flags of the decode loop: each is off by default, and on it must
change only host work, never a value the device computes."""

import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.runner.stage_timing import NullStageTiming


def test_disable_runner_stage_timing_flag(monkeypatch):
    monkeypatch.delenv("TPU_DISABLE_RUNNER_STAGE_TIMING", raising=False)
    assert envs.TPU_DISABLE_RUNNER_STAGE_TIMING is False
    monkeypatch.setenv("TPU_DISABLE_RUNNER_STAGE_TIMING", "1")
    assert envs.TPU_DISABLE_RUNNER_STAGE_TIMING is True


def test_null_stage_timing_runs_the_body_and_records_nothing():
    timing = NullStageTiming()
    ran = []
    with timing.stage("main_forward"):
        ran.append("stage")
    with timing.host_stage("output_assembly"):
        ran.append("host_stage")
    assert ran == ["stage", "host_stage"]
    assert timing.take_samples() == ([], [])
    with pytest.raises(RuntimeError), timing.stage("sample"):
        raise RuntimeError("stage body error propagates")


@pytest.mark.parametrize("every", [0, 3])
def test_gc_step_aligned_collect(monkeypatch, every):
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    monkeypatch.setenv("TPU_GC_STEP_ALIGNED_COLLECT", str(every))
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            capture_model=lambda: None, execute_model=lambda so: ("out", so)
        ),
        compilation_config=SimpleNamespace(compilation_time=0.0),
        _gc_collect_every=0,
        _gc_steps_since_collect=0,
    )
    was_enabled = gc.isenabled()
    try:
        with (
            patch("vllm_torchtpu.worker.tpu_worker.freeze_gc_heap"),
            patch("vllm_torchtpu.worker.tpu_worker.maybe_attach_gc_debug_callback"),
            patch("vllm_torchtpu.worker.tpu_worker.set_torch_threads_for_runtime"),
        ):
            TPUWorker.compile_or_warm_up_model(worker)
        assert gc.isenabled() == (every == 0)
        with patch("vllm_torchtpu.worker.tpu_worker.gc.collect") as collect:
            outs = [TPUWorker.execute_model(worker, i) for i in range(7)]
        assert outs == [("out", i) for i in range(7)]
        assert collect.call_count == (7 // every if every else 0)
    finally:
        if was_enabled:
            gc.enable()
