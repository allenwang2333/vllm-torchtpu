# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host-path flags of the decode loop: each is off by default, and on it must
change only host work, never a value the device computes."""

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
