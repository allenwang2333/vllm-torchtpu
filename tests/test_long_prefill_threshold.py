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
"""VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY: vLLM 0.32's per-step long-prefill cap
on vLLM 0.29's scheduler."""

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from tests.test_srpt_prefill_scheduler import (
    _build_real_scheduler,
    _make_real_request,
    shared_model_config,  # noqa: F401  (fixture)
)
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import long_prefill_threshold
from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import SrptPrefillScheduler
from vllm_torchtpu.core.long_prefill_threshold import effective_long_prefill_threshold


@pytest.mark.parametrize(
    "configured, num_requests, policy, expected",
    [
        (16, 1, "always", 16),
        (16, 3, "always", 16),
        (0, 3, "multi", 0),
        (16, 1, "multi", 0),
        (16, 0, "multi", 0),
        (16, 2, "multi", 16),
        (16, 1, "adaptive", 0),
        (16, 2, "adaptive", 32),
        (16, 8, "adaptive", 16),
        (0, 2, "adaptive", 0),
    ],
)
def test_effective_threshold(configured, num_requests, policy, expected):
    assert (
        effective_long_prefill_threshold(configured, num_requests, 64, policy)
        == expected
    )


@pytest.fixture
def threshold_policy(monkeypatch):
    """Install the patch for a policy, and take it off Scheduler afterwards."""
    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    monkeypatch.setattr(Scheduler, "_tpu_long_prefill_threshold", False, raising=False)

    def install(policy: str) -> None:
        monkeypatch.setenv("VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY", policy)
        long_prefill_threshold.patch_scheduler_long_prefill_threshold()

    return install


def _schedule_first_step(model_config, num_requests: int, scheduler_cls):
    scheduler, hasher = _build_real_scheduler(
        model_config,
        scheduler_cls=scheduler_cls,
        num_blocks=64,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
    )
    for i in range(num_requests):
        scheduler.add_request(_make_real_request(f"r{i}", 128, hasher, seed=i))
    out = scheduler.schedule()
    assert scheduler.scheduler_config.long_prefill_token_threshold == 16
    return out.num_scheduled_tokens


@pytest.mark.parametrize("scheduler_cls", [AsyncScheduler, SrptPrefillScheduler])
@pytest.mark.parametrize(
    "policy, num_requests, expected",
    [
        # vLLM 0.29: every request is capped, alone or not.
        ("always", 1, {"r0": 16}),
        ("always", 2, {"r0": 16, "r1": 16}),
        # vLLM 0.32: a lone request takes the whole budget.
        ("multi", 1, {"r0": 64}),
        ("multi", 2, {"r0": 16, "r1": 16}),
        # Adaptive: the cap is floored at budget / requests (64 / 2).
        ("adaptive", 2, {"r0": 32, "r1": 32}),
        ("adaptive", 8, {f"r{i}": 16 for i in range(4)}),
    ],
)
def test_first_step_chunks(
    shared_model_config,  # noqa: F811
    threshold_policy,
    scheduler_cls,
    policy,
    num_requests,
    expected,
):
    threshold_policy(policy)
    got = _schedule_first_step(shared_model_config, num_requests, scheduler_cls)
    assert got == expected


def test_always_leaves_scheduler_unpatched(threshold_policy):
    orig = Scheduler.schedule
    threshold_policy("always")
    assert Scheduler.schedule is orig


def test_srpt_plans_with_the_step_cap(
    shared_model_config,  # noqa: F811
    threshold_policy,
):
    threshold_policy("multi")
    scheduler, _ = _build_real_scheduler(
        shared_model_config, long_prefill_token_threshold=16
    )
    assert scheduler._long_prefill_threshold(1) == 0
    assert scheduler._long_prefill_threshold(2) == 16


def test_policy_is_not_a_compile_factor(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY", "adaptive")
    assert (
        "VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY"
        in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    )
    assert (
        "VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY"
        not in tpu_compiler._tpu_compile_env_factors()
    )
