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
"""Shortest-remaining-prefill-first scheduling of a step's prefill budget.

vLLM schedules the RUNNING queue in admission order and the WAITING queue after
it, so an in-progress long prefill chunk takes a small step budget every step and
a short request that arrives behind it waits for the whole long prefill. On a
multi-turn agent trace, where most prompts are cached and a few have tens of
thousands of new tokens, that queueing is most of the time to first token.

Each step this scheduler

* keeps every decoding request (decode and speculative verify) ahead of all
  prefill chunks, in the order vLLM had them;
* orders RUNNING prefill chunks by remaining prompt tokens, fewest first;
* orders WAITING requests by remaining prompt tokens net of their local prefix
  cache hit, fewest first (requests resumed after preemption first); with
  ``TPU_SRPT_COUNT_EXTERNAL_HIT=1`` also net of the KV connector's (host KV
  store's) hit, so a prompt whose prefix was offloaded does not sort as cold;
* holds back, for this step, running prefill chunks that have more remaining
  tokens than an admissible waiting request, so the budget reaches the waiting
  request instead of the long chunk (vLLM admits waiting requests only after the
  running queue has had the budget).

Held-back chunks keep their KV blocks and run again on a later step. If the
shorter requests cannot use the whole budget, the rest of that step's budget is
left unused: one partial step for the long chunk against the whole long prefill
for the short request. The running limit is lowered by the number held back so
the step never admits past ``max_num_seqs``.

A hold can admit nothing: the waiting request may not fit the KV cache (its
whole prompt must, and the held chunks keep their blocks), so vLLM stops at it,
and the next step holds the same chunks again. With no decode on the rank that
step does no work at all, every step, until the request is gone. With
``TPU_SRPT_HOLD_PROGRESS_GUARD=1`` a held step that admits no waiting request
(neither runs it nor starts its KV load) is followed by one step without
holding, so the held chunks advance at least every other step. (Pausing until
the next admission was tried and is worse: while paused, the long chunks run
first and take the whole budget, so nothing is admitted until they finish.)

With ``TPU_SCHED_UNCHARGED_DRAFT_SLOTS=1`` the step's input budget no longer
charges each request the drafter's extra slots (K-1 per request for DSpark): the
TPU drafter runs its own forward sized from the request count, so on TPU the
charge only takes prefill tokens out of the target forward. The scheduled-token
cap, and with it the largest token bucket, stays ``max_num_batched_tokens``.

With ``--prefill-schedule-interval N`` vLLM defers prefill to every N-th step,
except on a rank whose last prefill step left requests waiting
(``prefill_capacity_bound``); under sustained load that is nearly every rank,
so the deferral rarely holds. ``TPU_SRPT_STRICT_PREFILL_CADENCE=1`` clears that
flag every step, so a rank with a decoding request prefills only on cadence
steps.

Select with
``--scheduler-cls vllm_torchtpu.core.srpt_prefill_scheduler.SrptPrefillScheduler``.
It extends ``AsyncScheduler``, so it expects ``--async-scheduling``.
"""

import copy

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs

logger = init_logger("vllm." + __name__)

_LOG_EVERY_STEPS = 5000
_ADMITTABLE = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


class SrptPrefillScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError(
                "SrptPrefillScheduler reorders an FCFS waiting queue; got "
                f"scheduling policy {self.policy}."
            )
        self._srpt_steps = 0
        self._srpt_held_steps = 0
        self._srpt_held_chunks = 0
        self._srpt_progress_guard = envs.TPU_SRPT_HOLD_PROGRESS_GUARD
        self._srpt_strict_cadence = envs.TPU_SRPT_STRICT_PREFILL_CADENCE
        self._srpt_hold_paused = False
        self._srpt_paused_steps = 0
        self._srpt_count_external = envs.TPU_SRPT_COUNT_EXTERNAL_HIT
        self._srpt_external_end: dict[str, int] = {}
        if envs.TPU_SCHED_UNCHARGED_DRAFT_SLOTS:
            self._uncharge_draft_slots()
        logger.info("SrptPrefillScheduler active")

    def _uncharge_draft_slots(self) -> None:
        """Widen this scheduler's input budget by every running request's
        drafting slots, so only the scheduled-token cap binds."""
        spec = self.vllm_config.speculative_config
        slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        if slots == 0:
            return
        config = copy.copy(self.scheduler_config)
        input_budget = config.max_num_batched_tokens + slots * self.max_num_running_reqs
        # A private copy: the engine's own config, and every other reader of
        # max_num_batched_tokens, keep the served value.
        object.__setattr__(config, "max_num_batched_tokens", input_budget)
        self.scheduler_config = config
        logger.info(
            "SrptPrefillScheduler: %d drafting slots/request uncharged; input "
            "budget %d, scheduled-token cap %d",
            slots,
            input_budget,
            self.max_num_scheduled_tokens,
        )

    def _waiting_remaining(self, request: Request) -> int:
        """Prompt tokens a waiting request still has to prefill, net of its
        local prefix-cache hit and, with TPU_SRPT_COUNT_EXTERNAL_HIT, of its
        KV connector hit."""
        computed = request.num_computed_tokens
        kvm = self.kv_cache_manager
        if (
            computed == 0
            and kvm.enable_caching
            and not request.skip_reading_prefix_cache
        ):
            _, computed, _ = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
            if self._srpt_count_external:
                computed = max(computed, self._external_hit_end(request, computed))
        return max(request.num_tokens - computed, 1)

    def _external_hit_end(self, request: Request, local: int) -> int:
        """Prompt position the connector can load up to, asked once per waiting
        request with the block-aligned local hit, as admission asks it. An
        undetermined answer (None) is asked again on a later step."""
        end = self._srpt_external_end.get(request.request_id)
        if end is None and self.connector is not None:
            aligned = local - local % self.block_size
            matched, _ = self.connector.get_num_new_matched_tokens(request, aligned)
            if matched is not None:
                end = aligned + matched
                self._srpt_external_end[request.request_id] = end
        return end or 0

    @staticmethod
    def _running_remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _order_queues(self) -> list[tuple[int, Request]]:
        """Sort RUNNING (decodes first, then prefill chunks by remaining) and
        WAITING (preempted first, then by remaining); return the waiting
        requests' (remaining, request) in that order."""
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=self._running_remaining,
        )
        self.running[:] = decoding + prefilling

        if self._srpt_external_end:
            waiting_ids = {r.request_id for r in self.waiting}
            for rid in [k for k in self._srpt_external_end if k not in waiting_ids]:
                del self._srpt_external_end[rid]
        scored = sorted(
            ((self._waiting_remaining(r), r) for r in self.waiting),
            key=lambda t: (t[1].num_preemptions == 0, t[0], t[1].arrival_time),
        )
        waiting = self.waiting
        assert isinstance(waiting, FCFSRequestQueue)
        waiting.clear()
        waiting.extend(r for _, r in scored)
        return scored

    def _chunks_to_hold(self, waiting: list[tuple[int, Request]]) -> list[Request]:
        """Running prefill chunks to skip this step: those with more remaining
        tokens than some admissible waiting request."""
        if len(self.running) >= self.max_num_running_reqs:
            return []
        admissible = [
            n for n, r in waiting if not self._is_blocked_waiting_status(r.status)
        ]
        if not admissible:
            return []
        shortest = min(admissible)
        return [
            r
            for r in self.running
            if r.is_prefill_chunk and self._running_remaining(r) > shortest
        ]

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        waiting = self._order_queues()
        if self._srpt_strict_cadence:
            self.prefill_capacity_bound = False
        if self._srpt_strict_cadence and throttle_prefills:
            held = []  # upstream defers every prefill this step anyway
        elif self._srpt_hold_paused:
            held = []
            self._srpt_paused_steps += 1
        else:
            held = self._chunks_to_hold(waiting)
        self._srpt_steps += 1
        held_ids = {id(r) for r in held}
        self.running[:] = [r for r in self.running if id(r) not in held_ids]
        # Admission either runs a waiting request or starts its KV load
        # (WAITING_FOR_REMOTE_KVS); both take it out of these two states.
        candidates = [
            r
            for r in (*(r for _, r in waiting), *self.skipped_waiting)
            if r.status in _ADMITTABLE
        ]
        if held:
            self._srpt_held_steps += 1
            self._srpt_held_chunks += len(held)
        self.max_num_running_reqs -= len(held)
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs += len(held)
            self.running.extend(held)
        admitted = any(r.status not in _ADMITTABLE for r in candidates)
        if self._srpt_progress_guard:
            self._srpt_hold_paused = bool(held) and not admitted
        if self._srpt_steps % _LOG_EVERY_STEPS == 0:
            logger.info(
                "SRPT prefill: steps=%d held_steps=%d held_chunks=%d paused_steps=%d",
                self._srpt_steps,
                self._srpt_held_steps,
                self._srpt_held_chunks,
                self._srpt_paused_steps,
            )
        return output
