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

import math
from dataclasses import dataclass
from typing import Any, Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

@dataclass(frozen=True)
class PrefillCohortPlan:
    """Per-rank request budgets for one synchronized cohort."""

    budgets: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class PrefillCohortSettings:
    """Runtime timeout settings serialized with ``VllmConfig``."""

    state_sync_steps: int = 1
    flush_timeout_ms: int = 0
    quorum_fraction: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.flush_timeout_ms > 0

    def validate(self) -> None:
        if self.flush_timeout_ms < 0:
            raise ValueError(
                "TPU_DP_PREFILL_FLUSH_TIMEOUT_MS must be non-negative")
        if self.enabled and self.state_sync_steps <= 0:
            raise ValueError("prefill_schedule_interval must be positive")
        if not 0.0 < self.quorum_fraction <= 1.0:
            raise ValueError(
                "TPU_DP_PREFILL_QUORUM_FRACTION must be in (0, 1]")


def get_prefill_cohort_settings(vllm_config: Any) -> PrefillCohortSettings:
    """Read settings captured before vLLM spawns the DP engine processes."""
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return PrefillCohortSettings()
    settings = PrefillCohortSettings(
        state_sync_steps=int(
            vllm_config.scheduler_config.prefill_schedule_interval),
        flush_timeout_ms=int(
            additional_config.get(
                envs.TPU_DP_PREFILL_FLUSH_TIMEOUT_MS_CONFIG_KEY, 0)),
        quorum_fraction=float(
            additional_config.get(
                envs.TPU_DP_PREFILL_QUORUM_FRACTION_CONFIG_KEY, 1.0)),
    )
    settings.validate()
    return settings


def _log_prefill_interval_stats(
    step: int,
    interval: int,
    flush_timeout_ms: int,
    rows: Sequence[Sequence[int]],
) -> None:
    """Log per-rank DP prefill coordination stats for one sync interval.

    ``rows`` is the reduced per-rank state matrix widened with the two debug
    columns, so each row is ``[unfinished, pending_pause, active, ready,
    expired, remaining, wave, running, backlog]``. ``ready`` is the
    capacity-bounded pending count used for cohort planning; ``backlog`` is the
    full unadmitted-prefill queue depth; ``running`` is in-flight requests; and
    ``remaining`` is the active cohort's unfinished prompt tokens.
    """
    total_pending = sum(row[3] for row in rows)
    total_expired = sum(row[4] for row in rows)
    total_remaining = sum(row[5] for row in rows)
    total_running = sum(row[7] for row in rows)
    total_backlog = sum(row[8] for row in rows)
    active_ranks = sum(row[2] for row in rows)
    lines = [
        "[dp-prefill-stats] step=%d interval=%d flush_timeout_ms=%d | totals: "
        "pending=%d backlog=%d expired=%d running=%d inflight_prefill_tokens=%d "
        "cohorts_active=%d/%d" %
        (step, interval, flush_timeout_ms, total_pending, total_backlog,
         total_expired, total_running, total_remaining, active_ranks, len(rows))
    ]
    for rank, row in enumerate(rows):
        lines.append(
            "  rank%d: pending=%d backlog=%d expired=%d running=%d "
            "inflight_prefill_tokens=%d cohort_active=%d unfinished=%d wave=%d" %
            (rank, row[3], row[8], row[4], row[7], row[5], row[2], row[0],
             row[6]))
    logger.info("\n".join(lines))


class PrefillCohortPolicy:
    """Choose aligned dense gangs and bounded timeout flushes for fairness."""

    def __init__(
        self,
        dp_size: int,
        quorum_fraction: float = 1.0,
    ) -> None:
        if dp_size <= 1:
            raise ValueError("DP prefill cohorts require dp_size > 1")
        if not 0.0 < quorum_fraction <= 1.0:
            raise ValueError("quorum_fraction must be in (0, 1]")
        self.dp_size = dp_size
        self.quorum_rank_count = math.ceil(dp_size * quorum_fraction)
        self.sync_index = 0

    def tick(self) -> None:
        """Advance once per completed DP state exchange."""
        self.sync_index += 1

    def _record_open(self, plan: PrefillCohortPlan) -> PrefillCohortPlan:
        return plan

    def plan(
        self,
        readiness: Sequence[int],
        expired: Sequence[int],
    ) -> PrefillCohortPlan | None:
        """Return a timeout, exact, or sufficiently dense quorum cohort."""
        if len(readiness) != self.dp_size:
            raise ValueError(f"expected {self.dp_size} readiness values, got "
                             f"{len(readiness)}")
        if len(expired) != self.dp_size:
            raise ValueError(f"expected {self.dp_size} expiration values, got "
                             f"{len(expired)}")
        if any(count < 0 for count in readiness):
            raise ValueError("readiness values must be non-negative")
        if any(count < 0 for count in expired):
            raise ValueError("expiration values must be non-negative")
        if any(old > ready for old, ready in zip(expired, readiness)):
            raise ValueError("expired count cannot exceed readiness")

        self.tick()
        if any(expired):
            # Flush every expired request while filling ranks that have fewer
            # expirations from their fresh ready queues. Using the all-rank
            # ready floor preserves alignment when every rank has work; a zero
            # ready rank contributes no fresh top-up but cannot prevent another
            # rank from servicing its expired requests.
            target = max(max(expired), min(readiness))
            budgets = tuple(min(ready, target) for ready in readiness)
            return self._record_open(PrefillCohortPlan(budgets, "timeout"))

        if all(ready > 0 for ready in readiness):
            target = min(readiness)
            budgets = (target, ) * self.dp_size
            return self._record_open(
                PrefillCohortPlan(budgets, "all-exact"))

        ready_ranks = sum(ready > 0 for ready in readiness)
        if ready_ranks < self.quorum_rank_count:
            return None

        target = min(ready for ready in readiness if ready > 0)
        budgets = tuple(target if ready > 0 else 0 for ready in readiness)
        return self._record_open(
            PrefillCohortPlan(budgets, "quorum-exact"))


def _get_policy(
    engine: Any,
    settings: PrefillCohortSettings,
) -> PrefillCohortPolicy:
    policy = getattr(engine, "_tpu_prefill_cohort_policy", None)
    if policy is None:
        policy = PrefillCohortPolicy(
            dp_size=engine.dp_size,
            quorum_fraction=settings.quorum_fraction,
        )
        engine._tpu_prefill_cohort_policy = policy
    return policy


def install_dp_prefill_coordinator(vllm_config: Any | None = None) -> None:
    """Coordinate finite prefill snapshots through vLLM's DP state sync.

    The admission gate cannot safely run a collective because an idle peer
    may not reach it. Instead, replace the configured finish-state
    exchange with one rank-indexed exchange carrying the same finish/pause
    state plus cohort readiness. The collective order is unchanged and cohort
    decisions use a counter advanced only after that collective completes.
    """
    import torch
    import torch.distributed as dist
    from vllm.v1.engine.core import DPEngineCoreProc

    settings = (get_prefill_cohort_settings(vllm_config)
                if vllm_config is not None else PrefillCohortSettings())
    if getattr(DPEngineCoreProc, "_tpu_prefill_coordinator_patch", False):
        logger.info(
            "Using inherited TPU DP prefill timeout hook "
            "(serialized_timeout_ms=%d).", settings.flush_timeout_ms)
        return

    required_scheduler_methods = (
        "_tpu_prefill_cohort_active",
        "_tpu_prefill_ready_state",
        "_tpu_prefill_cohort_remaining",
        "_tpu_open_prefill_cohort",
        "_tpu_close_prefill_cohort",
    )
    original_dp_state = DPEngineCoreProc._has_global_unfinished_reqs

    def coordinated_dp_state(self, local_unfinished: bool) -> bool:
        settings = getattr(self, "_tpu_prefill_cohort_settings", None)
        if settings is None:
            settings = get_prefill_cohort_settings(self.vllm_config)
            self._tpu_prefill_cohort_settings = settings
        if not settings.enabled:
            return original_dp_state(self, local_unfinished)

        self.step_counter += 1
        if self.step_counter % settings.state_sync_steps != 0:
            return True

        scheduler = self.scheduler
        missing = [
            name for name in required_scheduler_methods
            if not hasattr(scheduler, name)
        ]
        if missing:
            raise RuntimeError(
                "coordinated DP prefill requires TpuDpScheduler methods: " +
                ", ".join(missing))

        active = scheduler._tpu_prefill_cohort_active()
        # Even an active cohort reports readiness excluding its own reserved
        # IDs. If it is fully drained globally, every rank can close it and
        # use this same collective result to open the next cohort.
        ready, expired = scheduler._tpu_prefill_ready_state(
            settings.flush_timeout_ms)
        remaining = (scheduler._tpu_prefill_cohort_remaining()
                     if active else 0)

        # One row per DP rank prevents SUM from losing the readiness vector.
        # Columns are unfinished, pending pause, active, ready, expired,
        # remaining, and current wave. Detailed stats append two debug columns
        # (running, backlog) so the per-interval log reports them per rank; the
        # width is identical across ranks because the env flag is process-wide.
        stats_enabled = envs.TPU_DP_PREFILL_STATS
        row_values = [
            int(local_unfinished),
            int(self.pending_pause),
            int(active),
            ready,
            expired,
            remaining,
            self.current_wave,
        ]
        if stats_enabled:
            debug_fn = getattr(scheduler, "_tpu_prefill_debug_stats", None)
            running, backlog = debug_fn() if debug_fn else (0, 0)
            row_values.extend([running, backlog])
        state = torch.zeros((self.dp_size, len(row_values)),
                            dtype=torch.int64,
                            device="cpu")
        state[self.dp_rank] = torch.tensor(row_values,
                                           dtype=torch.int64,
                                           device="cpu")
        dist.all_reduce(state, op=dist.ReduceOp.SUM, group=self.dp_group)

        rows = state.tolist()
        if stats_enabled and self.dp_rank == 0:
            _log_prefill_interval_stats(
                self.step_counter,
                self.step_counter // settings.state_sync_steps,
                settings.flush_timeout_ms,
                rows,
            )
        pause_count = sum(row[1] for row in rows)
        active_count = sum(row[2] for row in rows)
        has_unfinished = (any(row[0] for row in rows)
                          or pause_count % self.dp_size != 0)

        if active_count not in (0, self.dp_size):
            raise RuntimeError(
                "DP prefill cohort state diverged across ranks: "
                f"active_count={active_count}, dp_size={self.dp_size}")

        policy = _get_policy(self, settings)
        readiness = [row[3] for row in rows]
        expirations = [row[4] for row in rows]

        def open_or_extend_cohort() -> None:
            plan = policy.plan(readiness, expirations)
            if plan is not None:
                extending = scheduler._tpu_prefill_cohort_active()
                admitted = scheduler._tpu_open_prefill_cohort(
                    plan.budgets[self.dp_rank],
                    prefer_expired=plan.reason == "timeout")
                if admitted != plan.budgets[self.dp_rank]:
                    raise RuntimeError(
                        "prefill readiness changed before cohort snapshot: "
                        f"planned={plan.budgets[self.dp_rank]}, "
                        f"snapshotted={admitted}")
                if self.dp_rank == 0:
                    logger.info(
                        "%s %s DP prefill cohort at sync %d: "
                        "readiness=%s expired=%s budgets=%s.",
                        "Extended" if extending else "Opened", plan.reason,
                        policy.sync_index, readiness, expirations,
                        plan.budgets)

        if active_count == self.dp_size:
            if any(row[5] for row in rows):
                if pause_count == 0:
                    open_or_extend_cohort()
                else:
                    policy.tick()
            else:
                scheduler._tpu_close_prefill_cohort()
                logger.debug("Closed coordinated DP prefill cohort.")
                if pause_count == 0:
                    open_or_extend_cohort()
                else:
                    policy.tick()
        elif pause_count == 0:
            open_or_extend_cohort()
        else:
            policy.tick()

        if pause_count == self.dp_size:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

        return has_unfinished

    DPEngineCoreProc._tpu_original_has_global_unfinished_reqs = (
        original_dp_state)
    DPEngineCoreProc._has_global_unfinished_reqs = coordinated_dp_state
    DPEngineCoreProc._tpu_prefill_coordinator_patch = True
    logger.info(
        "Applied TPU patch: coordinated DP prefill timeout hook "
        "(serialized_timeout_ms=%d, state_sync_steps=%d, "
        "quorum_fraction=%.3f).", settings.flush_timeout_ms,
        settings.state_sync_steps, settings.quorum_fraction)
