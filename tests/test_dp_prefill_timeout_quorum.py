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

from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import dp_prefill_coordinator
from vllm_torchtpu.core.dp_prefill_coordinator import (
    PrefillCohortPolicy,
    PrefillCohortSettings,
    _get_policy,
    _log_prefill_interval_stats,
    get_prefill_cohort_settings,
)


def test_exact_all_rank_cohort_takes_precedence() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([5, 2, 3, 4], [0, 0, 0, 0])

    assert plan is not None
    assert plan.reason == "all-exact"
    assert plan.budgets == (2, 2, 2, 2)


def test_dp4_immediate_three_rank_quorum() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([5, 2, 3, 0], [0, 0, 0, 0])

    assert plan is not None
    assert plan.reason == "quorum-exact"
    assert plan.budgets == (2, 2, 2, 0)


def test_dp8_quorum_requires_six_ready_ranks() -> None:
    policy = PrefillCohortPolicy(8, quorum_fraction=0.75)

    assert policy.plan([2, 2, 2, 2, 2, 0, 0, 0], [0] * 8) is None
    plan = policy.plan([2, 2, 2, 2, 2, 2, 0, 0], [0] * 8)

    assert plan is not None
    assert plan.budgets == (2, 2, 2, 2, 2, 2, 0, 0)



def test_timeout_precedes_quorum_and_uses_all_rank_ready_floor() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([5, 4, 3, 0], [2, 0, 0, 0])

    assert plan is not None
    assert plan.reason == "timeout"
    assert plan.budgets == (2, 2, 2, 0)


def test_timeout_tops_up_to_the_all_rank_ready_floor() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([27, 30, 35, 28], [1, 1, 1, 1])

    assert plan is not None
    assert plan.reason == "timeout"
    assert plan.budgets == (27, 27, 27, 27)


def test_timeout_does_not_top_up_a_sparse_minority() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([5, 4, 0, 0], [1, 0, 0, 0])

    assert plan is not None
    assert plan.reason == "timeout"
    assert plan.budgets == (1, 1, 0, 0)


def test_timeout_always_services_the_full_expired_floor() -> None:
    policy = PrefillCohortPolicy(4, quorum_fraction=0.75)

    plan = policy.plan([6, 3, 5, 0], [4, 1, 0, 0])

    assert plan is not None
    assert plan.reason == "timeout"
    assert plan.budgets == (4, 3, 4, 0)


def test_settings_are_read_from_serialized_additional_config() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(prefill_schedule_interval=16),
        additional_config={
            "tpu_dp_prefill_flush_timeout_ms": 1500,
            "tpu_dp_prefill_quorum_fraction": 0.75,
        })

    assert get_prefill_cohort_settings(config) == PrefillCohortSettings(
        state_sync_steps=16,
        flush_timeout_ms=1500,
        quorum_fraction=0.75,
    )


def test_runtime_policy_uses_serialized_quorum_settings() -> None:
    settings = PrefillCohortSettings(
        flush_timeout_ms=1500,
        quorum_fraction=0.75,
    )
    engine = SimpleNamespace(dp_size=4)

    policy = _get_policy(engine, settings)

    assert policy.quorum_rank_count == 3


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_invalid_quorum_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="quorum_fraction"):
        PrefillCohortPolicy(4, quorum_fraction=fraction)


def test_log_prefill_interval_stats_reports_totals_and_per_rank(
        monkeypatch) -> None:
    records: list[str] = []
    monkeypatch.setattr(
        dp_prefill_coordinator.logger, "info",
        lambda msg, *args: records.append(msg % args if args else msg))

    # rows: [unfinished, pending_pause, active, ready, expired, remaining,
    #        wave, running, backlog] per DP rank.
    rows = [
        [1, 0, 1, 3, 1, 2048, 5, 7, 10],
        [1, 0, 1, 2, 0, 1536, 5, 6, 8],
        [0, 0, 1, 0, 0, 512, 5, 8, 4],
        [1, 0, 1, 4, 2, 3072, 5, 5, 12],
    ]
    _log_prefill_interval_stats(984, 123, 2000, rows)

    assert len(records) == 1
    text = records[0]
    # Totals aggregate each column across ranks.
    assert "step=984 interval=123 flush_timeout_ms=2000" in text
    assert "pending=9" in text
    assert "backlog=34" in text
    assert "expired=3" in text
    assert "running=26" in text
    assert "inflight_prefill_tokens=7168" in text
    assert "cohorts_active=4/4" in text
    # Per-rank line preserves the column-index mapping.
    assert ("rank3: pending=4 backlog=12 expired=2 running=5 "
            "inflight_prefill_tokens=3072 cohort_active=1 unfinished=1 wave=5"
            in text)
