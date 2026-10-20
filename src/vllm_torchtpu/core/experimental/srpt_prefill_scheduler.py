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
"""Shortest-remaining-first ordering and hold-back of running prefill chunks.

vLLM schedules RUNNING in admission order before WAITING, so an in-progress
long prefill chunk can take the step's token budget before a short waiting
request is reached. Each step (FCFS policy, prefills not throttled) this
scheduler:

* orders RUNNING as decodes first, in their existing order, then prefill
  chunks by remaining prompt tokens, fewest first, with stalled chunks (below)
  ahead of the rest;
* with ``VLLM_TPU_SRPT_COUNT_EXTERNAL_HIT=1`` (or ``TPU_SRPT_COUNT_EXTERNAL_HIT=1``),
  orders ``kv_holding_waiting`` and ``waiting`` requests by remaining prompt
  tokens net of local prefix-cache and ``KVConnector`` (host KV store) hits
  (ready requests ahead of in-flight async KV loads), and when the next
  admissible waiting request has an unstarted async external KV load
  (``load_kv_async=True``), reserves a 1-token step margin (plus a 2,048-token
  final-tail prefetch) instead of holding back running prefill chunks so the
  running prefill chunk and the async KV DMA launch overlap in the same step;
* holds back, for this step, the longest running prefill chunks needed for the
  request upstream admits next (the first admissible request in
  ``kv_holding_waiting``, then ``waiting``) to fit in the remaining budget,
  when that request has fewer uncached tokens and free KV blocks cover the
  kept requests plus it, so a hold does not cause preemption;
* with ``VLLM_TPU_SRPT_ATTN_WORK_CAP=W`` (or ``TPU_SRPT_ATTN_WORK_CAP=W``),
  clips prefill chunks to keep per-step attention work ``sum(n * (c + n / 2))``
  under ``W``, snapping work-capped chunks to 32-token Pallas query blocks
  (``bq_sz=32`` in ``streamindex_topk`` and ``mla_ragged_paged_attention`` at
  ``T<=128``) so clipped steps never pay for a 32-token hardware query block
  while advancing only a partial block.

Held chunks keep their KV blocks; ``max_num_running_reqs`` and
``max_num_active_reqs`` are lowered by the number held during the step so
admission never exceeds ``max_num_seqs``. A hold whose target is not
admitted backs off exponentially for that target.

Anti-starvation: a running prefill chunk that goes
``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS`` steps in a row without getting at least
one KV block of tokens in a step, whether it was held or crowded out by shorter
prefills, is stalled: it moves ahead of the other prefill chunks and is not
held until it gets a block. Decodes still go first, so a stalled chunk can
still wait while decodes fill the budget.

Under KV pressure upstream preempts ``running[-1]``, which is the longest
non-stalled prefill chunk, so a long prompt can be preempted repeatedly.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.experimental.srpt_prefill_scheduler.SrptPrefillScheduler``.
"""

from itertools import chain
import math
import os

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MAX_BACKOFF_SHIFT = 6
_ATTN_CAP_MIN_TOKENS = 32
_ATTN_SNAP_BIAS = 24
_ASYNC_LOAD_PREFETCH_REMAINING = 2048
_ADMITTABLE = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


class SrptPrefillScheduler(AsyncScheduler):
    """Order RUNNING prefill chunks by fewest remaining tokens and hold back
    longer ones for a shorter waiting request."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_consecutive_holds = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        self.count_external_hit = (
            envs.VLLM_TPU_SRPT_COUNT_EXTERNAL_HIT
            or os.environ.get("TPU_SRPT_COUNT_EXTERNAL_HIT") == "1"
        )
        attn_cap_env = envs.VLLM_TPU_SRPT_ATTN_WORK_CAP or int(
            os.environ.get("TPU_SRPT_ATTN_WORK_CAP", "0") or 0
        )
        self.attn_work_cap = float(attn_cap_env)
        # Running prefill request id -> consecutive steps with less than one
        # block of progress.
        self._stalled_steps: dict[str, int] = {}
        # Target request id -> (failed holds, first step a hold may retry).
        self._hold_backoff: dict[str, tuple[int, int]] = {}
        # request_id -> (aligned_local, external_end, load_kv_async)
        self._srpt_external_end: dict[str, tuple[int, int, bool] | int] = {}
        self._reserve_for_async_load = False
        self._capped_for_async_load = False
        self._step_scheduled_tokens = 0
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_rejected_waiting = False
        self._attn_clipped_chunks = 0
        self._srpt_async_kickoffs = 0
        self._in_super_schedule = False
        logger.info(
            "SrptPrefillScheduler active (max_consecutive_holds=%d, "
            "count_external_hit=%s, attn_work_cap=%g)",
            self.max_consecutive_holds,
            self.count_external_hit,
            self.attn_work_cap,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_rejected_waiting = False
        self._reserve_for_async_load = False
        self._capped_for_async_load = False
        self._step_scheduled_tokens = 0

        held: list[Request] = []
        target = None
        prefilling: list[Request] = []
        if not throttle_prefills and self.policy == SchedulingPolicy.FCFS:
            self._prune_hold_state()
            self._order_running()
            if self.count_external_hit:
                self._order_waiting()
            else:
                self._srpt_external_end.clear()
            prefilling = [r for r in self.running if r.is_prefill_chunk]
            held, target = self._select_chunks_to_hold()
        else:
            self._srpt_external_end.clear()

        if self._reserve_for_async_load:
            self._srpt_async_kickoffs += 1

        held_ids = {r.request_id for r in held}
        self.running[:] = [r for r in self.running if r.request_id not in held_ids]
        self.max_num_running_reqs -= len(held)
        self.max_num_active_reqs -= len(held)
        self._in_super_schedule = True
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._in_super_schedule = False
            self.max_num_running_reqs += len(held)
            self.max_num_active_reqs += len(held)
            self.running.extend(held)

        # One KV cache block of the main attention group; not self.block_size,
        # which for hybrid models is the LCM of all group block sizes and can
        # exceed a whole chunk.
        progress_tokens = self.cache_config.block_size
        for r in prefilling:
            if (
                output.num_scheduled_tokens.get(r.request_id, 0) >= progress_tokens
                or not r.is_prefill_chunk
            ):
                self._stalled_steps.pop(r.request_id, None)
            else:
                self._stalled_steps[r.request_id] = (
                    self._stalled_steps.get(r.request_id, 0) + 1
                )
        if target is not None and not self._attn_rejected_waiting:
            self._record_hold_outcome(target)
        return output

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _is_stalled(self, request: Request) -> bool:
        return (
            self._stalled_steps.get(request.request_id, 0) >= self.max_consecutive_holds
        )

    def _order_running(self) -> None:
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=lambda r: (not self._is_stalled(r), self._remaining(r)),
        )
        self.running[:] = decoding + prefilling

    def _order_waiting_queue(self, queue: RequestQueue) -> None:
        if not queue:
            return
        scored = sorted(
            ((self._waiting_remaining(r), r) for r in queue),
            key=lambda t: (
                self._is_blocked_waiting_status(t[1].status),
                t[1].num_preemptions == 0,
                t[0],
                t[1].arrival_time,
            ),
        )
        queue.clear()
        queue.extend(r for _, r in scored)

    def _order_waiting(self) -> None:
        if self._srpt_external_end:
            waiting_ids = {
                r.request_id
                for r in chain(self.kv_holding_waiting, self.waiting)
            }
            for rid in [k for k in self._srpt_external_end if k not in waiting_ids]:
                del self._srpt_external_end[rid]
        self._order_waiting_queue(self.kv_holding_waiting)
        self._order_waiting_queue(self.waiting)

    def _waiting_remaining(self, request: Request) -> int:
        """Prompt tokens still to prefill for ``request`` after local and
        (when ``count_external_hit`` is enabled) external KV cache hits."""
        if self._is_blocked_waiting_status(request.status):
            self._srpt_external_end.pop(request.request_id, None)
            return request.num_tokens
        if request.num_computed_tokens > 0:
            # Already admitted on an earlier step (e.g. preempted or resumed
            # after WAITING_FOR_REMOTE_KVS); its blocks are in KVCacheManager.
            self._srpt_external_end.pop(request.request_id, None)
            computed = request.num_computed_tokens
        elif self.cache_config.enable_prefix_caching:
            _, computed, _, _ = super()._get_local_prefix_cache_hit(request)
            if self.count_external_hit and self.connector is not None:
                aligned = (computed // self.block_size) * self.block_size
                cached = self._srpt_external_end.get(request.request_id)
                if isinstance(cached, tuple) and cached[0] == aligned:
                    _, ext_end, load_async = cached
                else:
                    ext_end, load_async = self._external_hit_info(request, aligned)
                    self._srpt_external_end[request.request_id] = (
                        aligned,
                        ext_end,
                        load_async,
                    )
                if ext_end > computed:
                    computed = ext_end
        else:
            self._srpt_external_end.pop(request.request_id, None)
            computed = 0
        return max(request.num_tokens - computed, 1)

    def _external_hit_info(self, request: Request, local: int) -> tuple[int, bool]:
        """Prompt position the connector can load up to and whether that load is
        asynchronous, or ``(0, False)`` if none."""
        ext, load_kv_async = self.connector.get_num_new_matched_tokens(request, local)
        if not ext or ext <= 0:
            return 0, False
        return local + ext, bool(load_kv_async)

    def _external_hit_tokens(self, request: Request, local: int) -> int:
        """Prompt position the connector can load up to, or 0 if none."""
        ext_end, _ = self._external_hit_info(request, local)
        return ext_end

    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        blocks, num_hit, aligned, full_hit = super()._get_local_prefix_cache_hit(
            request
        )
        cached = self._srpt_external_end.get(request.request_id)
        ext_end = cached[1] if isinstance(cached, tuple) else (cached or 0)
        if ext_end <= aligned:
            return blocks, num_hit, aligned, full_hit
        # Under fine prefix hits (TPU_FINE_PREFIX_HIT_TOKENS), `num_hit` can be
        # sub-block-aligned (even the whole prompt -> `full_hit=True`), which
        # makes `Scheduler.schedule()` skip the connector and recompute from
        # `aligned`. When the connector can load strictly past `aligned`,
        # present the block-aligned local hit so `Scheduler.schedule()` asks
        # the connector.
        n_blocks = aligned // self.cache_config.block_size
        truncated = tuple(g[:n_blocks] for g in blocks.blocks)
        return KVCacheBlocks(truncated), aligned, aligned, False

    def _is_unstarted_async_load(self, request: Request) -> bool:
        """Whether ``request`` is waiting to kick off an asynchronous external
        KV load (which consumes 0 tokens of step budget when admitted)."""
        if not self.count_external_hit or request.num_computed_tokens > 0:
            return False
        cached = self._srpt_external_end.get(request.request_id)
        if not isinstance(cached, tuple):
            return False
        aligned_local, ext_end, load_async = cached
        return bool(load_async and ext_end > aligned_local)

    def _prune_hold_state(self) -> None:
        prefilling = {r.request_id for r in self.running if r.is_prefill_chunk}
        self._stalled_steps = {
            k: v for k, v in self._stalled_steps.items() if k in prefilling
        }
        if self._hold_backoff:
            waiting = {
                r.request_id for r in chain(self.kv_holding_waiting, self.waiting)
            }
            self._hold_backoff = {
                k: v for k, v in self._hold_backoff.items() if k in waiting
            }

    def _record_hold_outcome(self, target: Request) -> None:
        # Admission either runs the target or starts its KV load; both move it
        # out of WAITING / PREEMPTED.
        if target.status not in _ADMITTABLE:
            self._hold_backoff.pop(target.request_id, None)
            return
        failures = self._hold_backoff.get(target.request_id, (0, 0))[0] + 1
        delay = 1 << min(failures - 1, _MAX_BACKOFF_SHIFT)
        self._hold_backoff[target.request_id] = (
            failures,
            self.current_step + 1 + delay,
        )

    def _long_prefill_threshold(self, num_eligible_reqs: int) -> int:
        # Mirrors the long_prefill_token_threshold computation inline in
        # Scheduler.schedule() (vLLM v0.31.0, scheduler.py ~L611-626). Re-sync
        # on vLLM upgrades.
        lptt = (
            self.scheduler_config.long_prefill_token_threshold
            if num_eligible_reqs > 1
            else 0
        )
        if lptt > 0 and self.adaptive_long_prefill_threshold:
            lptt = max(
                lptt,
                self.scheduler_config.max_num_batched_tokens // num_eligible_reqs,
            )
        return lptt

    def _attn_fit(self, context: int, attn_work: float | None = None) -> int:
        """Most tokens ``n`` of a chunk starting at ``context`` with
        ``n * (context + n / 2)`` within the step's remaining attention work."""
        used = self._attn_work if attn_work is None else attn_work
        room = self.attn_work_cap - used
        if room <= 0:
            return 0
        return int(math.sqrt(context * context + 2 * room) - context)

    def _clip_prefill_tokens(
        self,
        context: int,
        num_new_tokens: int,
        attn_work: float,
        prefill_scheduled: bool,
    ) -> tuple[int, bool]:
        """Clip ``num_new_tokens`` starting at ``context`` to fit the remaining
        attention work budget and snap to 32-token Pallas query blocks.

        Returns ``(clipped_tokens, was_clipped)``.
        """
        if self.attn_work_cap <= 0 or num_new_tokens <= 0:
            return num_new_tokens, False
        fit = self._attn_fit(context, attn_work)
        if fit >= num_new_tokens:
            return num_new_tokens, False
        snapped = (
            (fit + _ATTN_CAP_MIN_TOKENS - _ATTN_SNAP_BIAS) // _ATTN_CAP_MIN_TOKENS
        ) * _ATTN_CAP_MIN_TOKENS
        if snapped >= num_new_tokens:
            return num_new_tokens, False
        if snapped >= _ATTN_CAP_MIN_TOKENS:
            return snapped, True
        if not prefill_scheduled:
            # First prefill chunk on the step always gets at least
            # min(num_new_tokens, _ATTN_CAP_MIN_TOKENS) so a prompt whose
            # context alone exceeds the cap still advances by a full 32-token
            # Pallas query block.
            clipped = min(num_new_tokens, _ATTN_CAP_MIN_TOKENS)
            return clipped, clipped < num_new_tokens
        return 0, True

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if not self._in_super_schedule:
            return super()._reserve_prefill_lookahead(
                request, num_computed_tokens, num_new_tokens
            )
        prefill = (
            request.is_prefill_chunk or request.status != RequestStatus.RUNNING
        )
        if self._reserve_for_async_load and num_new_tokens > 0:
            # Leave at least 1 token of step budget (`token_budget > 0`) so
            # `Scheduler.schedule()` does not break out of its WAITING loop
            # before reaching the unstarted async KV load at the head of
            # `self.waiting`. Starting an async KV load (`load_kv_async=True`)
            # consumes 0 tokens from `token_budget`, so both the running
            # prefill chunk and the async KV DMA launch happen in the same step.
            max_allowed = max(
                self.max_num_scheduled_tokens - 1 - self._step_scheduled_tokens,
                0,
            )
            if num_new_tokens > max_allowed:
                num_new_tokens = max_allowed
                self._capped_for_async_load = True
            elif (
                request.status == RequestStatus.RUNNING
                and request.is_prefill_chunk
                and not self._capped_for_async_load
            ):
                # When a running prefill chunk is on its final tail
                # (`<= _ASYNC_LOAD_PREFETCH_REMAINING` tokens left) and an
                # unstarted async KV load is waiting behind it, completing the
                # prefill this step would allocate lookahead/spec slots and
                # prevent `Scheduler.schedule()` from reaching `self.waiting`
                # if tokens run tight. Leaving 1 token for the next step (when
                # the chunk would otherwise use the whole budget or leave 0)
                # overlaps the async KV DMA launch one step earlier.
                rem = self._remaining(request)
                if 1 < rem <= _ASYNC_LOAD_PREFETCH_REMAINING and num_new_tokens >= rem:
                    num_new_tokens = rem - 1
                    self._capped_for_async_load = True

        if self.attn_work_cap > 0 and prefill and num_new_tokens > 0:
            clipped, was_clipped = self._clip_prefill_tokens(
                num_computed_tokens,
                num_new_tokens,
                self._attn_work,
                self._attn_prefill_scheduled,
            )
            if was_clipped:
                self._attn_clipped_chunks += 1
                num_new_tokens = clipped
                self._capped_for_async_load = False
                if num_new_tokens == 0 and request.status != RequestStatus.RUNNING:
                    self._attn_rejected_waiting = True

        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        self._step_scheduled_tokens += num_new_tokens
        if self.attn_work_cap > 0:
            self._attn_prefill_scheduled |= prefill and num_new_tokens > 0
            self._attn_work += num_new_tokens * (
                num_computed_tokens + num_new_tokens / 2
            )
        return num_new_tokens

    def _plan_running(
        self, reqs: list[Request], lptt: int
    ) -> tuple[dict[str, int], int, float, bool]:
        """Tokens the RUNNING pass of Scheduler.schedule() would give ``reqs``
        and the budget + attention work state it leaves for WAITING.

        Mirrors the per-request ``num_new_tokens`` math inline in that pass
        (vLLM v0.31.0, scheduler.py ~L628-722); re-sync on vLLM upgrades.
        Encoder, Mamba-alignment and KV-allocation limits are not modelled, so
        a misestimate only makes a hold miss, which the backoff absorbs.
        """
        next_step = self.current_step + 1
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.scheduler_config.max_num_batched_tokens
        alloc: dict[str, int] = {}
        plan_attn_work = 0.0
        plan_prefill_scheduled = False
        for r in reqs:
            if token_budget <= 0 or input_budget <= draft_slots:
                break
            if (
                r.num_output_placeholders > 0
                and r.num_computed_tokens + 2 - r.num_output_placeholders
                >= r.num_prompt_tokens + r.max_tokens
            ) or next_step < r.next_decode_eligible_step:
                continue
            n = (
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens
            )
            if 0 < lptt < n:
                n = lptt
            n = min(
                n,
                token_budget,
                input_budget - draft_slots,
                self.max_model_len
                - r.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if self.attn_work_cap > 0 and r.is_prefill_chunk and n > 0:
                n, _ = self._clip_prefill_tokens(
                    r.num_computed_tokens,
                    n,
                    plan_attn_work,
                    plan_prefill_scheduled,
                )
            n = self._reserve_prefill_lookahead(r, r.num_computed_tokens, n)
            if n <= 0:
                continue
            alloc[r.request_id] = n
            token_budget -= n
            input_budget -= n + draft_slots
            if self.attn_work_cap > 0:
                plan_prefill_scheduled |= r.is_prefill_chunk
                plan_attn_work += n * (r.num_computed_tokens + n / 2)
        return (
            alloc,
            min(token_budget, input_budget - draft_slots),
            plan_attn_work,
            plan_prefill_scheduled,
        )

    def _next_waiting(self) -> Request | None:
        """The request upstream's WAITING pass admits next."""
        return next(
            (
                r
                for r in chain(self.kv_holding_waiting, self.waiting)
                if not self._is_blocked_waiting_status(r.status)
            ),
            None,
        )

    def _prefix_hit(self, request: Request) -> tuple[KVCacheBlocks, int]:
        if request.num_computed_tokens > 0:
            return (
                self.kv_cache_manager.empty_kv_cache_blocks,
                request.num_computed_tokens,
            )
        blocks, num_hit_tokens, _, _ = super()._get_local_prefix_cache_hit(request)
        return blocks, num_hit_tokens

    def _select_chunks_to_hold(self) -> tuple[list[Request], Request | None]:
        """Return ``(chunks_to_hold, target_waiting_request)``."""
        if self.max_consecutive_holds <= 0:
            return [], None
        if not any(r.is_prefill_chunk for r in self.running):
            return [], None
        if (
            len(self.running) + self.num_waiting_for_streaming_input
            >= self.max_num_active_reqs
        ):
            return [], None
        target = self._next_waiting()
        if target is None:
            return [], None
        _, retry_step = self._hold_backoff.get(target.request_id, (0, 0))
        if self.current_step + 1 < retry_step:
            return [], None

        if self._is_unstarted_async_load(target):
            if (
                len(self.running) + self.num_waiting_for_streaming_input
                < self.max_num_active_reqs
            ):
                self._reserve_for_async_load = True
                return [], target
            return [], None

        hit_blocks, local_hit_tokens = self._prefix_hit(target)
        if self.count_external_hit:
            target_remaining = self._waiting_remaining(target)
            effective_hit_tokens = max(target.num_tokens - target_remaining, 0)
        else:
            target_remaining = max(target.num_tokens - local_hit_tokens, 1)
            effective_hit_tokens = local_hit_tokens

        _, num_waiting = self.get_request_counts()
        kept = list(self.running)
        held: list[Request] = []
        while True:
            lptt = self._long_prefill_threshold(len(kept) + num_waiting)
            alloc, left, plan_attn_work, plan_prefill_scheduled = (
                self._plan_running(kept, lptt)
            )
            need = min(target_remaining, lptt) if lptt > 0 else target_remaining
            if self.attn_work_cap > 0 and left > 0:
                clipped_need, _ = self._clip_prefill_tokens(
                    effective_hit_tokens,
                    min(need, left),
                    plan_attn_work,
                    plan_prefill_scheduled,
                )
            else:
                clipped_need = min(need, left)
            if left >= need and (self.attn_work_cap <= 0 or clipped_need > 0):
                break
            # Non-stalled prefill chunks sit last in `kept`, ordered by
            # remaining tokens, so this is the longest one.
            candidate = next(
                (
                    r
                    for r in reversed(kept)
                    if r.request_id in alloc
                    and r.is_prefill_chunk
                    and self._remaining(r) > target_remaining
                    and not self._is_stalled(r)
                ),
                None,
            )
            if candidate is None:
                break
            kept.remove(candidate)
            held.append(candidate)
        if not held or left <= 0:
            return [], None
        if self.attn_work_cap > 0 and clipped_need <= 0:
            return [], None
        admit_tokens = clipped_need if self.attn_work_cap > 0 else min(need, left)
        if not self._kv_fits_hold(
            kept, alloc, target, hit_blocks, local_hit_tokens, admit_tokens
        ):
            return [], None
        return held, target

    def _kv_fits_hold(
        self,
        kept: list[Request],
        alloc: dict[str, int],
        target: Request,
        hit_blocks: KVCacheBlocks,
        hit_tokens: int,
        target_tokens: int,
    ) -> bool:
        """Whether free KV blocks cover ``kept`` running requests plus
        ``target``, counted as KVCacheManager.allocate_slots() counts them,
        so a hold never forces a preemption."""
        coordinator = self.kv_cache_manager.coordinator
        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks

        def blocks_for(
            request: Request,
            computed_tokens: int,
            num_new_tokens: int,
            computed_blocks: tuple,
        ) -> int:
            num_tokens_main = computed_tokens + num_new_tokens
            return coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(
                    num_tokens_main + self.num_lookahead_tokens, self.max_model_len
                ),
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed_tokens,
                num_local_computed_tokens=computed_tokens,
                num_tokens_main_model=num_tokens_main,
                prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
            )

        needed = sum(
            blocks_for(r, r.num_computed_tokens, alloc[r.request_id], empty_blocks)
            for r in kept
            # In-flight prefills are covered by the reservation below.
            if r.request_id in alloc and r not in self._inflight_prefills
        )
        target_blocks = blocks_for(target, hit_tokens, target_tokens, hit_blocks.blocks)
        if self.scheduler_reserve_full_isl:
            full_num_tokens = min(target.num_tokens, self.max_model_len)
            target_blocks = max(
                target_blocks,
                coordinator.get_num_blocks_to_allocate(
                    request_id=target.request_id,
                    num_tokens=full_num_tokens,
                    new_computed_blocks=hit_blocks.blocks,
                    num_encoder_tokens=0,
                    total_computed_tokens=hit_tokens,
                    num_local_computed_tokens=hit_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                    prefill_end=max(target.num_prompt_tokens, target.num_tokens - 1),
                ),
            )
        needed += target_blocks
        if kept:
            # allocate_slots() adds the watermark when others are scheduled.
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks()
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed
