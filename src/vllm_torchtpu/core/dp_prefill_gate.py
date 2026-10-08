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
"""Scheduler that measures, and optionally tightens, vLLM's DP prefill hold-back.

``DPEngineCoreProc`` passes ``throttle_prefills=True`` on every step that is not a
multiple of ``--prefill-schedule-interval``, and ``Scheduler.schedule`` then holds
back prefill compute, except on a rank that:

* is ``prefill_capacity_bound``: its last step that was not held back left
  requests waiting, for any reason; or
* has no decoding request, so holding back would leave it nothing to run.

Under expert parallelism the DP ranks run every forward in lockstep and pad to
the largest rank's token count, so one rank that skips the hold-back costs every
rank a prefill-sized forward.

This scheduler counts, per rank, why throttled steps carried prefill tokens and
why cadence-aligned steps left requests waiting, and logs the counts every
``TPU_PREFILL_GATE_LOG_STEPS`` steps. With ``TPU_PREFILL_GATE_MIN_CADENCES=N``
(N >= 1) it also replaces upstream's capacity-bound test: a rank counts as
capacity-bound only after N consecutive cadence-aligned steps that used the whole
token budget and still left requests waiting. That is the only case in which
prefilling more often lets the rank admit more work; a queue held back by
``max_num_seqs`` or KV capacity does not drain faster. N=0 leaves upstream's flag
untouched.

Select with
``--scheduler-cls vllm_torchtpu.core.dp_prefill_gate.DpPrefillGateScheduler``.
It extends ``AsyncScheduler``, so it expects ``--async-scheduling``.
"""

from collections import Counter

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs

logger = init_logger("vllm." + __name__)


class DpPrefillGateScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_min_cadences = envs.TPU_PREFILL_GATE_MIN_CADENCES
        self._gate_log_steps = envs.TPU_PREFILL_GATE_LOG_STEPS
        self._gate_streak = 0
        self._gate_stats: Counter[str] = Counter()
        logger.info(
            "DpPrefillGateScheduler active (min_cadences=%d, log_steps=%d)",
            self._gate_min_cadences,
            self._gate_log_steps,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Reason upstream will use to skip the hold-back on this step, read
        # before super() rewrites prefill_capacity_bound.
        bound_before = self.prefill_capacity_bound
        has_decoder = any(not r.is_prefill_chunk for r in self.running)

        output = super().schedule(throttle_prefills)

        # Without spec decode a decode schedules exactly one token, so any
        # request given more than one token this step is prefilling.
        prefill_tokens = sum(n for n in output.num_scheduled_tokens.values() if n > 1)
        waiting = bool(self.waiting)
        budget_full = output.total_num_scheduled_tokens >= self.max_num_scheduled_tokens
        seqs_full = len(self.running) >= self.max_num_running_reqs
        self._record(
            throttle_prefills,
            bound_before,
            has_decoder,
            prefill_tokens,
            waiting,
            budget_full,
            seqs_full,
        )

        if self._gate_min_cadences > 0:
            starved = waiting and budget_full
            if not throttle_prefills:
                self._gate_streak = self._gate_streak + 1 if starved else 0
            elif bound_before and not starved:
                # An escape step that no longer exhausts the budget (queue
                # drained, or now held back by seqs/KV): stop escaping.
                self._gate_streak = 0
            self.prefill_capacity_bound = self._gate_streak >= self._gate_min_cadences
        return output

    def _record(
        self,
        throttled: bool,
        bound_before: bool,
        has_decoder: bool,
        prefill_tokens: int,
        waiting: bool,
        budget_full: bool,
        seqs_full: bool,
    ) -> None:
        s = self._gate_stats
        s["steps"] += 1
        s["bound_steps"] += self.prefill_capacity_bound
        if throttled:
            s["throttled"] += 1
            if prefill_tokens:
                if bound_before:
                    reason = "bound"
                elif not has_decoder:
                    reason = "no_decoder"
                else:
                    reason = "other"
                s[f"throttled_prefill_{reason}"] += 1
                s["throttled_prefill_tokens"] += prefill_tokens
        else:
            s["cadence"] += 1
            if prefill_tokens:
                s["cadence_prefill"] += 1
                s["cadence_prefill_tokens"] += prefill_tokens
            if waiting:
                if budget_full:
                    s["cadence_waiting_budget"] += 1
                elif seqs_full:
                    s["cadence_waiting_seqs"] += 1
                else:
                    s["cadence_waiting_other"] += 1
        if self._gate_log_steps > 0 and s["steps"] >= self._gate_log_steps:
            logger.info(
                "DP prefill gate: steps=%d throttled=%d "
                "throttled_prefill[bound=%d no_decoder=%d other=%d tokens=%d] "
                "cadence=%d cadence_prefill=%d tokens=%d "
                "cadence_waiting[budget=%d seqs=%d other=%d] bound_steps=%d",
                s["steps"],
                s["throttled"],
                s["throttled_prefill_bound"],
                s["throttled_prefill_no_decoder"],
                s["throttled_prefill_other"],
                s["throttled_prefill_tokens"],
                s["cadence"],
                s["cadence_prefill"],
                s["cadence_prefill_tokens"],
                s["cadence_waiting_budget"],
                s["cadence_waiting_seqs"],
                s["cadence_waiting_other"],
                s["bound_steps"],
            )
            s.clear()
