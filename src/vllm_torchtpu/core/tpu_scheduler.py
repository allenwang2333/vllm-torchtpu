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
from vllm_torchtpu.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs

logger = init_logger(__name__)


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks.

    vLLM passes a DP-synchronized scheduling signal through ``throttle_prefills``.
    This scheduler uses that signal only to decide when to admit new prefill
    requests until all pending prefills are drained.

    Why the cadence alone is not enough. With EP the DP ranks run in
    lockstep: ``_dp_coordinated_step`` all-reduces the local token count with
    ``ReduceOp.MAX``, so a single rank that is prefilling drags all eight onto
    the largest token bucket. Keeping the ranks in the *same regime* -- all
    prefilling or all decoding -- is therefore worth far more than any
    per-rank scheduling win.

    The historical behaviour only buffers brand-new requests, and only while
    this rank's queues are non-empty. Under a saturated benchmark the waiting
    queue never empties, so the drain window never closes and the buffering
    never engages; and an in-progress prefill *chunk* is never deferrable at
    all, because ``prefill_capacity_bound`` is pinned True. ``envs``:

    ``TPU_DP_PREFILL_DEFER``
        Forward the cadence to upstream so it can defer in-progress prefill
        chunks (``Scheduler.schedule``'s ``defer_prefills``).
    ``TPU_DP_PREFILL_HARD``
        Pin ``prefill_capacity_bound`` False so a backlog cannot re-open
        admission mid-window.
    ``TPU_DP_PREFILL_DRAIN_STEPS``
        Bound the drain window in steps so it closes under saturation.

    All three default off, which reproduces the historical behaviour exactly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        self._steps_since_window = 0
        logger.info(
            "TpuDpScheduler active (defer=%s, hard=%s, drain_steps=%s)",
            envs.TPU_DP_PREFILL_DEFER,
            envs.TPU_DP_PREFILL_HARD,
            envs.TPU_DP_PREFILL_DRAIN_STEPS,
        )

    def _schedule_paused_new(self,
                             throttle_prefills: bool = False) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(throttle_prefills)
        finally:
            self._pause_state = prev_pause_state

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        defer = envs.TPU_DP_PREFILL_DEFER
        if not defer:
            # Upstream throttling is neutralized so the cadence is the only
            # clock.
            self.prefill_capacity_bound = True
        elif envs.TPU_DP_PREFILL_HARD:
            # A non-empty waiting queue must not re-open prefill admission.
            self.prefill_capacity_bound = False

        if not throttle_prefills:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
            self._steps_since_window = 0
        elif defer:
            self._steps_since_window += 1

        drain_steps = envs.TPU_DP_PREFILL_DRAIN_STEPS
        window_expired = (defer and drain_steps > 0
                          and self._steps_since_window >= drain_steps)
        if self._drain_prefill and (window_expired or not (
                self.waiting or self.skipped_waiting)):
            # In drain mode until all accumulated prefills are scheduled, or
            # until the bounded window closes.
            self._drain_prefill = False
        if self._drain_prefill:
            # Forwarding the cadence lets upstream defer in-progress prefill
            # chunks too; passing False (the historical value) makes its
            # ``defer_prefills`` unconditionally False.
            return super().schedule(defer and throttle_prefills)
        return self._schedule_paused_new(defer and throttle_prefills)
