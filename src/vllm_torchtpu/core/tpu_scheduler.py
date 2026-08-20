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
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core.dp_prefill_coordinator import \
    get_prefill_cohort_settings
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks.

    vLLM passes a DP-synchronized scheduling signal through ``throttle_prefills``.
    The legacy path drains locally after that signal. When coordinated cohorts
    are enabled, the engine-core DP sync snapshots a finite set of request IDs
    and this scheduler holds every later arrival until the next cohort.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        cohort_settings = get_prefill_cohort_settings(self.vllm_config)
        self._tpu_coordinated_prefill = cohort_settings.enabled
        self._tpu_prefill_cohort_ids: frozenset[str] | None = None
        self._tpu_prefill_eligible_since_ns: dict[str, int] = {}
        self._tpu_prefill_expired_ids: frozenset[str] = frozenset()
        logger.info("TpuDpScheduler active (coordinated_prefill=%s, "
                    "flush_timeout_ms=%d)", self._tpu_coordinated_prefill,
                    cohort_settings.flush_timeout_ms)

    def _schedule_paused_new(self) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(False)
        finally:
            self._pause_state = prev_pause_state

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        if self._tpu_coordinated_prefill:
            # Local monotonic eligibility tracking never initiates a
            # collective; the cohort state advances only at engine-core sync.
            self._tpu_refresh_prefill_eligibility()
            return self._schedule_prefill_cohort()

        if not throttle_prefills:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
        if self._drain_prefill and not (self.waiting or self.skipped_waiting):
            # In drain mode until all accumulated prefills are scheduled.
            self._drain_prefill = False
        if self._drain_prefill:
            return super().schedule(False)
        return self._schedule_paused_new()

    @staticmethod
    def _tpu_is_unadmitted_prefill(request) -> bool:
        return (request.num_preemptions == 0 and request.status in (
            RequestStatus.WAITING,
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
        ))

    def _tpu_prefill_cohort_active(self) -> bool:
        return self._tpu_prefill_cohort_ids is not None

    def _tpu_prefill_ready_requests(self):
        """Return ungranted eligible requests bounded by free slots."""
        cohort_ids = self._tpu_prefill_cohort_ids or frozenset()
        reserved_slots = self._tpu_prefill_cohort_unadmitted_count()
        free_slots = max(
            0,
            self.max_num_running_reqs - len(self.running) -
            self.num_waiting_for_streaming_input - reserved_slots,
        )
        if free_slots == 0:
            return []
        ready = []
        for request in self.waiting:
            if (request.num_preemptions == 0
                    and request.status == RequestStatus.WAITING
                    and request.request_id not in cohort_ids):
                ready.append(request)
                if len(ready) == free_slots:
                    break
        return ready

    def _tpu_refresh_prefill_eligibility(
        self,
        now_ns: int | None = None,
    ):
        """Track when requests first enter the capacity-bounded ready set."""
        if now_ns is None:
            now_ns = time.monotonic_ns()
        ready = self._tpu_prefill_ready_requests()
        ready_ids = {request.request_id for request in ready}
        self._tpu_prefill_eligible_since_ns = {
            request_id: since
            for request_id, since in self._tpu_prefill_eligible_since_ns.items()
            if request_id in ready_ids
        }
        for request in ready:
            self._tpu_prefill_eligible_since_ns.setdefault(
                request.request_id, now_ns)
        self._tpu_prefill_expired_ids = frozenset(
            request_id for request_id in self._tpu_prefill_expired_ids
            if request_id in ready_ids)
        return ready

    def _tpu_prefill_ready_state(
        self,
        flush_timeout_ms: int,
    ) -> tuple[int, int]:
        """Return capacity-bounded ready and wall-clock-expired counts."""
        if flush_timeout_ms <= 0:
            raise ValueError("prefill flush timeout must be positive")
        now_ns = time.monotonic_ns()
        ready = self._tpu_refresh_prefill_eligibility(now_ns)
        timeout_ns = flush_timeout_ms * 1_000_000
        self._tpu_prefill_expired_ids = frozenset(
            request.request_id for request in ready
            if now_ns - self._tpu_prefill_eligible_since_ns[
                request.request_id] >= timeout_ns)
        return len(ready), len(self._tpu_prefill_expired_ids)

    def _tpu_open_prefill_cohort(
        self,
        budget: int,
        prefer_expired: bool = False,
    ) -> int:
        """Add eligible IDs to the cohort, prioritizing expired then oldest."""
        if budget < 0:
            raise ValueError("prefill cohort budget must be non-negative")

        ready = self._tpu_refresh_prefill_eligibility()
        expired_ids = self._tpu_prefill_expired_ids
        ready.sort(key=lambda request: (
            prefer_expired and request.request_id not in expired_ids,
            self._tpu_prefill_eligible_since_ns[request.request_id],
            request.arrival_time,
            request.request_id,
        ))
        request_ids = [request.request_id for request in ready[:budget]]
        cohort_ids = self._tpu_prefill_cohort_ids or frozenset()
        self._tpu_prefill_cohort_ids = cohort_ids | frozenset(request_ids)
        for request_id in request_ids:
            self._tpu_prefill_eligible_since_ns.pop(request_id, None)
        self._tpu_prefill_expired_ids = frozenset(
            request_id for request_id in self._tpu_prefill_expired_ids
            if request_id not in self._tpu_prefill_cohort_ids)
        return len(request_ids)

    def _tpu_prefill_cohort_remaining(self) -> int:
        """Return unfinished prompt tokens owned by the active cohort."""
        cohort_ids = self._tpu_prefill_cohort_ids
        if cohort_ids is None:
            return 0
        return sum(
            max(request.num_prompt_tokens - request.num_computed_tokens, 0)
            for queue in (self.waiting, self.skipped_waiting, self.running)
            for request in queue
            if request.request_id in cohort_ids)

    def _tpu_prefill_debug_stats(self) -> tuple[int, int]:
        """Per-rank counters for detailed interval logging.

        Returns ``(running, backlog)`` where ``running`` is the number of
        in-flight requests (admitted and executing this step) and ``backlog``
        is every unadmitted prefill still waiting for a cohort slot -- the raw
        pending-queue depth, a superset of the capacity-bounded ``ready`` count
        that cohort planning exchanges.
        """
        running = len(self.running)
        backlog = sum(
            self._tpu_is_unadmitted_prefill(request)
            for queue in (self.waiting, self.skipped_waiting)
            for request in queue)
        return running, backlog

    def _tpu_prefill_cohort_unadmitted_count(self) -> int:
        """Count reserved IDs that still consume future running slots."""
        cohort_ids = self._tpu_prefill_cohort_ids
        if cohort_ids is None:
            return 0
        return sum(request.request_id in cohort_ids
                   and self._tpu_is_unadmitted_prefill(request)
                   for queue in (self.waiting, self.skipped_waiting)
                   for request in queue)

    def _tpu_close_prefill_cohort(self) -> None:
        if not self._tpu_prefill_cohort_active():
            raise RuntimeError("cannot close an inactive prefill cohort")
        self._tpu_prefill_cohort_ids = None

    @staticmethod
    def _tpu_restore_queue(queue, original, held) -> None:
        """Restore held requests and the original relative queue order."""
        held_ids = {id(request) for request in held}
        current = list(queue)
        current_ids = {id(request) for request in current}
        restore_ids = held_ids | {
            id(request)
            for request in original if id(request) in current_ids
        }
        to_remove = [
            request for request in current if id(request) in restore_ids
        ]
        if to_remove:
            queue.remove_requests(to_remove)
        for request in original:
            if id(request) in restore_ids:
                queue.add_request(request)

    def _schedule_prefill_cohort(self) -> SchedulerOutput:
        """Run vLLM scheduling while exposing only the active snapshot."""
        cohort_ids = self._tpu_prefill_cohort_ids or frozenset()
        original_waiting = list(self.waiting)
        original_skipped = list(self.skipped_waiting)
        held_waiting = [
            request for request in original_waiting
            if self._tpu_is_unadmitted_prefill(request)
            and request.request_id not in cohort_ids
        ]
        held_skipped = [
            request for request in original_skipped
            if self._tpu_is_unadmitted_prefill(request)
            and request.request_id not in cohort_ids
        ]
        if held_waiting:
            self.waiting.remove_requests(held_waiting)
        if held_skipped:
            self.skipped_waiting.remove_requests(held_skipped)
        try:
            # Running prefills, decodes, and PREEMPTED requests remain visible.
            return super().schedule(False)
        finally:
            self._tpu_restore_queue(self.waiting, original_waiting,
                                    held_waiting)
            self._tpu_restore_queue(self.skipped_waiting, original_skipped,
                                    held_skipped)
