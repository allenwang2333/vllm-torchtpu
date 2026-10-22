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
"""vLLM 0.32's ``long_prefill_token_threshold`` policy on vLLM 0.29
(``VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY``).

vLLM 0.29 caps every request's chunk at ``--long-prefill-token-threshold`` on
every step, so a long prompt that is the only request still prefills in
threshold-sized chunks and leaves the rest of the step budget unused. vLLM 0.32
(scheduler.py ~L640-655) computes the cap once per step from the number of
running plus waiting requests:

* ``multi``: the cap applies only when more than one request is running or
  waiting; a lone request has nobody to starve and gets the whole budget;
* ``adaptive``: ``multi``, and the cap is floored at ``max_num_batched_tokens
  // num_requests`` (vLLM 0.32's ``long_prefill_token_threshold_adaptive``),
  so it never cuts a request below a fair share of the budget.

The default ``always`` keeps vLLM 0.29's behaviour and leaves the scheduler
unpatched. With a policy set and a threshold configured, ``Scheduler.schedule``
runs with the step's cap in ``scheduler_config.long_prefill_token_threshold``
and restores the configured value afterwards, so both the RUNNING and the
WAITING pass read it. vLLM 0.32 keeps the configured value for the Mamba
block-alignment split; here that split reads the step's cap as well, which
matters only for hybrid Mamba models. The step budget, token buckets and
compile shapes are unchanged.
"""

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

POLICIES = ("always", "multi", "adaptive")


def effective_long_prefill_threshold(
    configured: int, num_requests: int, max_num_batched_tokens: int, policy: str
) -> int:
    """The per-request chunk cap for a step with ``num_requests`` running plus
    waiting requests (0 = no cap)."""
    if configured <= 0 or policy == "always":
        return configured
    if num_requests <= 1:
        return 0
    if policy == "adaptive":
        return max(configured, max_num_batched_tokens // num_requests)
    return configured


def scheduler_long_prefill_threshold(scheduler: Any, num_requests: int) -> int:
    """``effective_long_prefill_threshold`` for ``scheduler`` under the
    configured policy."""
    from vllm_torchtpu import envs

    cfg = scheduler.scheduler_config
    return effective_long_prefill_threshold(
        cfg.long_prefill_token_threshold,
        num_requests,
        cfg.max_num_batched_tokens,
        envs.VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY,
    )


def patch_scheduler_long_prefill_threshold() -> None:
    """Wrap ``Scheduler.schedule`` when ``VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY``
    is not ``always``. Must run in the engine-core process."""
    from vllm_torchtpu import envs

    policy = envs.VLLM_TPU_LONG_PREFILL_THRESHOLD_POLICY
    if policy == "always":
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_long_prefill_threshold", False):
        return
    orig = Scheduler.schedule

    def schedule(self, throttle_prefills: bool = False):
        cfg = self.scheduler_config
        configured = cfg.long_prefill_token_threshold
        if configured <= 0:
            return orig(self, throttle_prefills)
        if not self.__dict__.get("_tpu_lptt_logged"):
            self._tpu_lptt_logged = True
            logger.info(
                "Long-prefill threshold policy %s on %s: cap %d, budget %d",
                policy,
                type(self).__name__,
                configured,
                cfg.max_num_batched_tokens,
            )
        num_running, num_waiting = self.get_request_counts()
        cap = effective_long_prefill_threshold(
            configured,
            num_running + num_waiting,
            cfg.max_num_batched_tokens,
            policy,
        )
        if cap == configured:
            return orig(self, throttle_prefills)
        cfg.long_prefill_token_threshold = cap
        try:
            return orig(self, throttle_prefills)
        finally:
            cfg.long_prefill_token_threshold = configured

    Scheduler.schedule = schedule
    Scheduler._tpu_long_prefill_threshold = True
