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
import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs

logger = init_logger(__name__)


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks and optimizes

    compilation bucket alignment.

    Features:
      1. Compilation Bucket Budgeting (Scheme 3): Caps prefill admission
         token budget to fit precisely within static TPU compilation bucket
         thresholds (compile_sizes), preventing minor token overshoots from
         forcing 2x larger compilation buckets and wasting padding compute.
      2. Synchronized Dynamic Drain (Scheme 1 Complete): Performs a lightweight
         cross-rank CPU All-Reduce sync to trigger prefill drain when queued
         requests or waiting time exceed thresholds across all DP ranks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        self._enable_bucket_budgeting = envs.DP_SCHED_ENABLE_BUCKET_BUDGETING
        self._enable_dynamic_drain = envs.DP_SCHED_DYNAMIC_DRAIN
        self._dynamic_drain_min_reqs = envs.DP_SCHED_DYNAMIC_DRAIN_MIN_REQS
        self._dynamic_drain_timeout_ms = envs.DP_SCHED_DYNAMIC_DRAIN_TIMEOUT_MS
        logger.info(
            "TpuDpScheduler active: bucket_budgeting=%s dynamic_drain=%s "
            "dynamic_drain_min_reqs=%d dynamic_drain_timeout_ms=%d",
            self._enable_bucket_budgeting, self._enable_dynamic_drain,
            self._dynamic_drain_min_reqs, self._dynamic_drain_timeout_ms)

    def _get_target_compile_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest static compilation bucket size >= num_tokens."""
        compile_sizes = getattr(self.compilation_config, "compile_sizes", None)
        if not compile_sizes:
            return None
        for size in sorted(compile_sizes):
            if size >= num_tokens:
                return size
        return compile_sizes[-1]

    def _oldest_waiting_age_ms(self) -> float:
        oldest = None
        for queue in (self.waiting, self.skipped_waiting):
            if queue:
                arrival = queue.peek_request().arrival_time
                if oldest is None or arrival < oldest:
                    oldest = arrival
        if oldest is None:
            return 0.0
        return (time.time() - oldest) * 1000.0

    def _check_synchronized_dynamic_drain(self) -> bool:
        """Lightweight cross-rank CPU sync to check if all DP ranks are ready to drain."""
        if not self._enable_dynamic_drain:
            return False
        num_waiting = len(self.waiting) + len(self.skipped_waiting)
        age_ms = self._oldest_waiting_age_ms()
        if self.parallel_config.data_parallel_size <= 1:
            return (num_waiting >= self._dynamic_drain_min_reqs or
                    age_ms >= self._dynamic_drain_timeout_ms)

        try:
            from vllm.distributed.parallel_state import get_dp_group
            dp_group = get_dp_group()
            if dp_group is None or dp_group.cpu_group is None:
                return False
            t = torch.tensor([int(num_waiting), int(age_ms)],
                             dtype=torch.int64,
                             device="cpu")
            torch.distributed.all_reduce(t,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=dp_group.cpu_group)
            max_waiting, max_age = int(t[0].item()), int(t[1].item())
            return (max_waiting >= self._dynamic_drain_min_reqs or
                    max_age >= self._dynamic_drain_timeout_ms)
        except Exception:
            return False

    def _schedule_paused_new(self) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(False)
        finally:
            self._pause_state = prev_pause_state

    def _schedule_with_bucket_budgeting(self) -> SchedulerOutput:
        """Schedule prefill requests with compilation bucket budgeting."""
        if not self._enable_bucket_budgeting:
            return super().schedule(False)

        compile_sizes = getattr(self.compilation_config, "compile_sizes", None)
        if not compile_sizes:
            return super().schedule(False)

        orig_max_batched = self.scheduler_config.max_num_batched_tokens
        try:
            # Estimate pending token demand to pick optimal bucket cap
            pending_tokens = 0
            for queue in (self.waiting, self.skipped_waiting):
                for req in queue:
                    pending_tokens += req.num_computed_tokens or req.num_tokens
            if pending_tokens > 0:
                target_bucket = self._get_target_compile_bucket(pending_tokens)
                if target_bucket is not None and target_bucket < orig_max_batched:
                    self.scheduler_config.max_num_batched_tokens = target_bucket
            return super().schedule(False)
        finally:
            self.scheduler_config.max_num_batched_tokens = orig_max_batched

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        dynamic_trigger = self._check_synchronized_dynamic_drain()
        if not throttle_prefills or dynamic_trigger:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
        if self._drain_prefill and not (self.waiting or self.skipped_waiting):
            # In drain mode until all accumulated prefills are scheduled.
            self._drain_prefill = False
        if self._drain_prefill:
            return self._schedule_with_bucket_budgeting()
        return self._schedule_paused_new()
