# SPDX-License-Identifier: Apache-2.0
"""TPU_REQUEST_TIMING_LOG logs each finished request's TTFT split."""

import logging

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats, RequestStateStats

from vllm_torchtpu import request_timing_log


def test_logs_ttft_split(monkeypatch, caplog):
    monkeypatch.setenv("TPU_REQUEST_TIMING_LOG", "1")
    saved = IterationStats.__dict__["update_from_finished_request"]
    try:
        request_timing_log.patch_request_timing_log()
        stats = RequestStateStats(
            num_generation_tokens=5,
            arrival_time=100.0,
            queued_ts=10.0,
            scheduled_ts=10.25,
            first_token_ts=10.75,
            last_token_ts=11.0,
            first_token_latency=1.5,
        )
        it = IterationStats()
        with caplog.at_level(logging.INFO):
            it.update_from_finished_request(
                finish_reason=FinishReason.LENGTH,
                request_id="r1",
                num_prompt_tokens=1000,
                max_tokens_param=5,
                req_stats=stats,
                num_cached_tokens=900,
            )
        assert it.finished_requests[-1].queued_time == 0.25
        line = next(
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("request timing")
        )
        assert "id=r1 isl=1000 cached=900 out=5" in line
        assert "ttft=1.5000 front=0.7500 queued=0.2500 prefill=0.5000" in line
    finally:
        IterationStats.update_from_finished_request = saved
        IterationStats._tpu_request_timing_log = False
