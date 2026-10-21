# SPDX-License-Identifier: Apache-2.0
"""One log line per finished request with its time-to-first-token split
(``TPU_REQUEST_TIMING_LOG``).

vLLM's per-request stats already carry, for every request, the frontend's
arrival-to-first-token time and the engine's queued and prefill intervals; they
only reach Prometheus as histograms, which cannot be joined to prompt length.
This logs them per request in the API server, so TTFT can be regressed on ISL
and cached tokens term by term:

- ``ttft``: frontend arrival (before the chat template and tokenization) to
  the first token reaching the frontend.
- ``queued``: engine QUEUED (added to the scheduler, after the request's block
  hashes) to first SCHEDULED.
- ``prefill``: first SCHEDULED to the engine emitting the first token.
- ``front``: ``ttft - queued - prefill``: rendering, tokenization, request
  transfer, engine-side request creation and the first output's way back.
"""

from __future__ import annotations

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_request_timing_log() -> None:
    from vllm_torchtpu import envs

    if not envs.TPU_REQUEST_TIMING_LOG:
        return
    from vllm.v1.metrics.stats import IterationStats

    if getattr(IterationStats, "_tpu_request_timing_log", False):
        return
    orig = IterationStats.update_from_finished_request

    def update_from_finished_request(
        self,
        finish_reason,
        request_id,
        num_prompt_tokens,
        max_tokens_param,
        req_stats,
        num_cached_tokens=0,
    ):
        orig(
            self,
            finish_reason,
            request_id,
            num_prompt_tokens,
            max_tokens_param,
            req_stats,
            num_cached_tokens,
        )
        done = self.finished_requests[-1]
        stats = req_stats
        ttft = stats.first_token_latency
        logger.info(
            "request timing: id=%s isl=%d cached=%d out=%d arrival=%.3f "
            "ttft=%.4f front=%.4f queued=%.4f prefill=%.4f e2e=%.4f",
            done.request_id,
            done.num_prompt_tokens,
            done.num_cached_tokens,
            done.num_generation_tokens,
            stats.arrival_time,
            ttft,
            ttft - done.queued_time - done.prefill_time,
            done.queued_time,
            done.prefill_time,
            done.e2e_latency,
        )

    IterationStats.update_from_finished_request = update_from_finished_request
    IterationStats._tpu_request_timing_log = True
    logger.info("TPU_REQUEST_TIMING_LOG: one 'request timing' line per request.")
