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
"""Scheduler that concentrates DP prefill into cadence steps and keeps the rest
decode-only.

Under expert parallelism the DP ranks run every forward in lockstep, padded to
the largest rank's token bucket. One rank that prefills therefore puts every rank
in a prefill-sized forward, and the slowest rank's prefill attention (which grows
with chunk tokens x context) gates the step for all of them.

``DPEngineCoreProc`` already aligns ``--prefill-schedule-interval`` across ranks:
it passes ``throttle_prefills=True`` on every step that is not a multiple of the
interval. Upstream ``Scheduler.schedule`` then holds back prefill, except on a
rank that is ``prefill_capacity_bound`` (anything left waiting) or has no
decoding request. Either escape makes the step prefill-sized for every rank.
This scheduler:

* **Throttled steps stay decode-only.** It clears ``prefill_capacity_bound``
  before each throttled step (``TPU_DP_CADENCE_BOUND_ESCAPE=1`` keeps upstream's
  escape). A rank with no decoding request may prefill at most
  ``TPU_DP_CADENCE_FILLER_TOKENS`` tokens in total, small enough that the step's
  DP bucket stays the decode-only one; 0 makes it idle.
* **Cadence steps cap prefill by attention cost, not tokens.** With
  ``TPU_DP_CADENCE_ATTN_BUDGET=M`` (millions of token x context), each request's
  chunk is at most ``M * 1e6 / ctx``, where ctx is the longest context this rank
  is about to prefill (a running chunk's computed tokens, or a waiting request's
  prompt length, an upper bound until its prefix-cache hit is known), and at
  least ``TPU_DP_CADENCE_MIN_CHUNK``. A long-context prompt then takes small
  chunks and cannot stretch the cadence step for all ranks, while short-context
  prompts fill the bucket. 0 disables the cap. An explicit
  ``--long-prefill-token-threshold`` still applies as an upper bound.
* **It logs per-rank step costs.** Every ``TPU_DP_CADENCE_LOG_STEPS`` steps it
  logs, for cadence and throttled steps separately, the wall time between
  consecutive ``update_from_output`` calls (the step period with async
  scheduling) and the local decode / prefill / filler tokens. Throttled steps
  without filler measure the decode-only forward directly.

With ``--prefill-schedule-interval=1`` every step is a cadence step and only the
attention cap applies.

Select with
``--scheduler-cls vllm_torchtpu.core.dp_cadence_scheduler.DpCadenceScheduler``.
It extends ``AsyncScheduler``, so it expects ``--async-scheduling``.
"""

import time
from collections import Counter
from itertools import islice

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs

logger = init_logger("vllm." + __name__)

# Step periods longer than this span an idle gap, not a forward.
_MAX_STEP_SECONDS = 5.0


class DpCadenceScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cad_filler_tokens = envs.TPU_DP_CADENCE_FILLER_TOKENS
        self._cad_attn_budget = envs.TPU_DP_CADENCE_ATTN_BUDGET * 1e6
        self._cad_min_chunk = envs.TPU_DP_CADENCE_MIN_CHUNK
        self._cad_bound_escape = envs.TPU_DP_CADENCE_BOUND_ESCAPE
        self._cad_log_steps = envs.TPU_DP_CADENCE_LOG_STEPS
        self._cad_init_stats()
        logger.info(
            "DpCadenceScheduler active (filler_tokens=%d, attn_budget=%gM, "
            "min_chunk=%d, bound_escape=%s, log_steps=%d)",
            self._cad_filler_tokens,
            self._cad_attn_budget / 1e6,
            self._cad_min_chunk,
            self._cad_bound_escape,
            self._cad_log_steps,
        )

    def _cad_init_stats(self) -> None:
        self._cad_stats: Counter[str] = Counter()
        self._cad_periods: dict[str, list[float]] = {}
        # id(SchedulerOutput) -> step kind, consumed by update_from_output.
        self._cad_pending: dict[int, str] = {}
        self._cad_last_update: float | None = None

    # ------------------------------------------------------------------ schedule

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        decoders = {r.request_id for r in self.running if not r.is_prefill_chunk}
        saved_budget = self.max_num_scheduled_tokens
        saved_lpt = self.scheduler_config.long_prefill_token_threshold
        filler = False
        if throttle_prefills:
            if not self._cad_bound_escape:
                self.prefill_capacity_bound = False
            if not decoders:
                # Upstream does not hold back a rank with no decoder; cap the
                # whole step instead so its bucket stays the decode-only one.
                filler = True
                self.max_num_scheduled_tokens = min(
                    saved_budget, self._cad_filler_tokens
                )
        else:
            lpt = self._cadence_chunk_limit(saved_lpt)
            if lpt != saved_lpt:
                self.scheduler_config.long_prefill_token_threshold = lpt
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = saved_budget
            if self.scheduler_config.long_prefill_token_threshold != saved_lpt:
                self.scheduler_config.long_prefill_token_threshold = saved_lpt
        self._record_schedule(output, throttle_prefills, filler, decoders)
        return output

    def _cadence_chunk_limit(self, configured: int) -> int:
        """Per-request prefill chunk limit for this cadence step."""
        if self._cad_attn_budget <= 0:
            return configured
        ctx = 0
        for r in self.running:
            if r.is_prefill_chunk:
                ctx = max(ctx, r.num_computed_tokens)
        free_slots = max(0, self.max_num_running_reqs - len(self.running))
        for r in islice(iter(self.waiting), free_slots):
            ctx = max(ctx, r.num_prompt_tokens)
        if ctx <= 0:
            return configured
        limit = max(self._cad_min_chunk, int(self._cad_attn_budget / ctx))
        if 0 < configured < limit:
            return configured
        # Never exceed the budget: a limit at or above it is no limit.
        return limit if limit < self.max_num_scheduled_tokens else configured

    # ------------------------------------------------------------------- stats

    def _record_schedule(
        self,
        output: SchedulerOutput,
        throttled: bool,
        filler: bool,
        decoders: set[str],
    ) -> None:
        decode_tokens = prefill_tokens = 0
        for req_id, n in output.num_scheduled_tokens.items():
            if req_id in decoders:
                decode_tokens += n
            else:
                prefill_tokens += n
        if not throttled:
            kind = "cadence"
        elif prefill_tokens:
            kind = "filler"
        else:
            kind = "decode"
        s = self._cad_stats
        s["steps"] += 1
        s[f"{kind}_steps"] += 1
        s[f"{kind}_decode_tokens"] += decode_tokens
        s[f"{kind}_prefill_tokens"] += prefill_tokens
        s[f"{kind}_max_tokens"] = max(
            s[f"{kind}_max_tokens"], output.total_num_scheduled_tokens
        )
        if throttled and not filler and prefill_tokens:
            # Should not happen: an escape this scheduler failed to close.
            s["throttled_escape"] += 1
        if not throttled and self.waiting:
            s["cadence_left_waiting"] += 1
        self._cad_pending[id(output)] = kind

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        now = time.monotonic()
        kind = self._cad_pending.pop(id(scheduler_output), None)
        if kind is not None and self._cad_last_update is not None:
            dt = now - self._cad_last_update
            if dt < _MAX_STEP_SECONDS:
                self._cad_periods.setdefault(kind, []).append(dt)
        self._cad_last_update = now
        if self._cad_log_steps > 0 and self._cad_stats["steps"] >= self._cad_log_steps:
            self._log_stats()
        return result

    def _log_stats(self) -> None:
        s = self._cad_stats
        parts = []
        for kind in ("cadence", "decode", "filler"):
            n = s[f"{kind}_steps"]
            periods = sorted(self._cad_periods.get(kind, []))
            if periods:
                p50 = periods[len(periods) // 2] * 1e3
                p90 = periods[min(len(periods) - 1, int(0.9 * len(periods)))] * 1e3
                mean = sum(periods) / len(periods) * 1e3
            else:
                p50 = p90 = mean = 0.0
            parts.append(
                f"{kind}[steps={n} ms_mean={mean:.1f} ms_p50={p50:.1f} "
                f"ms_p90={p90:.1f} decode_tok={s[f'{kind}_decode_tokens']} "
                f"prefill_tok={s[f'{kind}_prefill_tokens']} "
                f"max_tok={s[f'{kind}_max_tokens']}]"
            )
        logger.info(
            "DP cadence: steps=%d %s cadence_left_waiting=%d throttled_escape=%d",
            s["steps"],
            " ".join(parts),
            s["cadence_left_waiting"],
            s["throttled_escape"],
        )
        pending = self._cad_pending
        last = self._cad_last_update
        self._cad_init_stats()
        # Outputs scheduled before the reset are still in flight.
        self._cad_pending = pending
        self._cad_last_update = last
