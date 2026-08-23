# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vllm_torchtpu.runner.dp_batch_stats import (
    DPBatchStatsLogger, analyze_records, build_perfetto_trace, load_records)


def _record(step_id, rank, prefill, decode, *, active_decode=0, duration=47.0):
    return {
        "step_id": step_id,
        "rank": rank,
        "real_prefill_tokens": prefill,
        "real_decode_tokens": decode,
        "padded_tokens": 256 - prefill - decode,
        "compiled_bucket": 256,
        "active_decode_requests": active_decode,
        "queued_prefill_tokens": 0,
        "step_duration": duration,
        "num_chunks": 1,
    }


def test_logger_uses_next_step_start_for_cadence_duration(tmp_path):
    logger = DPBatchStatsLogger(str(tmp_path), rank=3, worker_rank=3)
    logger.begin_step(monotonic_ns=1_000_000_000, unix_ns=10_000_000_000)
    logger.stage_step(_record(1, 3, 128, 32))
    logger.begin_step(monotonic_ns=1_047_000_000, unix_ns=10_047_000_000)
    logger.stage_step(_record(2, 3, 0, 32))
    logger.close()

    rows = [json.loads(line) for line in (tmp_path / "rank3.jsonl").read_text(
    ).splitlines()]
    assert rows[0]["step_duration"] == pytest.approx(47.0)
    assert rows[1]["step_duration"] is None
    assert all(field in rows[0] for field in (
        "step_id",
        "rank",
        "real_prefill_tokens",
        "real_decode_tokens",
        "padded_tokens",
        "compiled_bucket",
        "active_decode_requests",
        "queued_prefill_tokens",
        "step_duration",
    ))


def test_analysis_identifies_sparse_prefill_and_padding():
    rows = []
    # A sparse mixed step: only two of eight ranks have prefill.
    for rank in range(8):
        rows.append(
            _record(1,
                    rank,
                    128 if rank < 2 else 0,
                    32,
                    active_decode=4))
    # A dense prefill cohort.
    for rank in range(8):
        rows.append(_record(2, rank, 256, 0))

    analysis = analyze_records(rows, expected_dp_size=8)

    assert analysis["fraction_steps_containing_prefill"] == 1.0
    assert analysis["fraction_steps_mixed_prefill_decode"] == 0.5
    assert analysis["sparse_prefill"]["steps"] == 1
    assert analysis["sparse_prefill"]["real_prefill_tokens"] == 256
    assert analysis["decode_requests_exposed_per_prefill_step_mean"] == 16
    assert analysis["missing_rank_steps"] == {}
    assert analysis["bucket_utilization"]["utilization"] == pytest.approx(
        (256 + 6 * 32 + 2 * 32 + 8 * 256) / (16 * 256))


def test_loader_rejects_duplicate_step_rank(tmp_path):
    row = _record(1, 0, 0, 32)
    (tmp_path / "rank0.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate step/rank"):
        load_records(tmp_path)


def test_perfetto_contains_global_and_per_rank_step_events():
    rows = [_record(1, rank, 224 if rank == 0 else 0, 32)
            for rank in range(8)]
    trace = build_perfetto_trace(rows)
    complete_events = [event for event in trace["traceEvents"]
                       if event["ph"] == "X"]
    assert len(complete_events) == 9
    assert complete_events[0]["args"]["active_prefill_ranks"] == 1
