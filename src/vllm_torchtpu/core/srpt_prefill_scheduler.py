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
  cache hit, fewest first (requests resumed after preemption first);
* holds back, for this step, running prefill chunks that have more remaining
  tokens than an admissible waiting request, so the budget reaches the waiting
  request instead of the long chunk (vLLM admits waiting requests only after the
  running queue has had the budget).

Held-back chunks keep their KV blocks and run again on a later step. If the
shorter requests cannot use the whole budget, the rest of that step's budget is
left unused: one partial step for the long chunk against the whole long prefill
for the short request. The running limit is lowered by the number held back so
the step never admits past ``max_num_seqs``.

Select with
``--scheduler-cls vllm_torchtpu.core.srpt_prefill_scheduler.SrptPrefillScheduler``.
It extends ``AsyncScheduler``, so it expects ``--async-scheduling``.
"""

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request

logger = init_logger("vllm." + __name__)

_LOG_EVERY_STEPS = 5000


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
        logger.info("SrptPrefillScheduler active")

    def _waiting_remaining(self, request: Request) -> int:
        """Prompt tokens a waiting request still has to prefill, net of its
        local prefix-cache hit (an external KV connector hit is not known
        until admission, so it is not counted)."""
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
        return max(request.num_tokens - computed, 1)

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
        held = self._chunks_to_hold(waiting)
        self._srpt_steps += 1
        if not held:
            output = super().schedule(throttle_prefills)
        else:
            self._srpt_held_steps += 1
            self._srpt_held_chunks += len(held)
            held_ids = {id(r) for r in held}
            self.running[:] = [r for r in self.running if id(r) not in held_ids]
            self.max_num_running_reqs -= len(held)
            try:
                output = super().schedule(throttle_prefills)
            finally:
                self.max_num_running_reqs += len(held)
                self.running.extend(held)
        if self._srpt_steps % _LOG_EVERY_STEPS == 0:
            logger.info(
                "SRPT prefill: steps=%d held_steps=%d held_chunks=%d",
                self._srpt_steps,
                self._srpt_held_steps,
                self._srpt_held_chunks,
            )
        return output
