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
"""Prefix-aware scheduler that keeps prefill chunks from stalling decodes.

With one long prefill chunk per step, every request that is decoding waits for
the whole chunk, so on agentic traces (tens of thousands of new prompt tokens
per turn) the slowest decodes are the short ones that overlap someone else's
prefill. Two knobs bound that stall while leaving a lone request's TTFT alone:

* ``QX_DECODE_CHUNK=N``: while at least ``QX_DECODE_MIN`` requests are
  decoding, the step's token budget (decode + prefill tokens) is ``N``
  instead of ``max_num_batched_tokens``. With no decodes in flight the full
  budget is used.
* ``QX_PREFILL_EVERY=K``: while requests are decoding, prefill work (new
  admissions and in-progress chunks) runs only on every ``K``-th step; the
  other steps are decode-only.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

try:
    from vllm.v1.core.sched.interface import PauseState
except ImportError:  # pragma: no cover - older vLLM layouts
    from vllm.v1.core.sched.scheduler import PauseState  # type: ignore

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 2000


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value.strip() else default


class DecodeProtectScheduler(PrefixAwareScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._qx_full_budget = self.max_num_scheduled_tokens
        self._qx_decode_chunk = _env_int("QX_DECODE_CHUNK", 0)
        self._qx_prefill_every = max(_env_int("QX_PREFILL_EVERY", 1), 1)
        self._qx_decode_min = max(_env_int("QX_DECODE_MIN", 1), 1)
        self._qx_decoding_steps = 0
        self._qx_steps = 0
        self._qx_throttled = 0
        self._qx_capped = 0
        logger.info(
            "DecodeProtectScheduler ACTIVE (full_budget=%d decode_chunk=%d "
            "prefill_every=%d decode_min=%d)",
            self._qx_full_budget,
            self._qx_decode_chunk,
            self._qx_prefill_every,
            self._qx_decode_min,
        )

    def _qx_num_decoding(self) -> int:
        return sum(
            1
            for r in self.running
            if not r.is_prefill_chunk and r.num_computed_tokens >= r.num_prompt_tokens
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._qx_steps += 1
        protect = self._qx_num_decoding() >= self._qx_decode_min

        budget = self._qx_full_budget
        if protect and 0 < self._qx_decode_chunk < budget:
            budget = self._qx_decode_chunk
            self._qx_capped += 1
        self.max_num_scheduled_tokens = budget

        throttle = False
        if protect and self._qx_prefill_every > 1:
            self._qx_decoding_steps += 1
            throttle = self._qx_decoding_steps % self._qx_prefill_every != 0

        if self._qx_steps % _LOG_EVERY == 0:
            logger.info(
                "decode-protect: steps=%d capped=%d throttled=%d",
                self._qx_steps,
                self._qx_capped,
                self._qx_throttled,
            )

        if not throttle:
            return super().schedule(throttle_prefills)

        # Decode-only step: defer in-progress prefill chunks and keep new
        # requests waiting.
        self._qx_throttled += 1
        self.prefill_capacity_bound = False
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(True)
        finally:
            self._pause_state = prev_pause_state
