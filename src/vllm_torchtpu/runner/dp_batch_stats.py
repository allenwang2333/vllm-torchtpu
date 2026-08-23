# SPDX-License-Identifier: Apache-2.0
"""Low-overhead, full-run DP batch-composition telemetry.

The writer intentionally measures serving cadence (the interval between model
step starts) rather than forcing a device synchronization.  That makes the
diagnostic suitable for throughput comparisons while still capturing the time
for which decode requests are exposed to a padded DP step.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "step_id",
    "rank",
    "real_prefill_tokens",
    "real_decode_tokens",
    "padded_tokens",
    "compiled_bucket",
    "active_decode_requests",
    "queued_prefill_tokens",
    "step_duration",
)


class DPBatchStatsLogger:
    """Write one compact JSONL stream for one DP rank.

    A record is staged as soon as its compiled DP bucket is known.  It is
    finalized at the next model-step start, at which point its start-to-start
    duration is known.  The final record is emitted with a null duration at
    shutdown rather than including arbitrary shutdown latency.
    """

    def __init__(self, output_dir: str, rank: int, worker_rank: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.worker_rank = int(worker_rank)
        self.path = self.output_dir / f"rank{self.rank}.jsonl"
        self._file = self.path.open("w", buffering=1024 * 1024)
        self._step_id = 0
        self._current: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._idle_queued_prefill_tokens = 0
        self._closed = False
        atexit.register(self.close)

    def begin_step(
        self,
        *,
        monotonic_ns: int | None = None,
        unix_ns: int | None = None,
    ) -> int:
        """Finalize the previous cadence interval and begin a new step."""
        if self._closed:
            raise RuntimeError("DP batch stats logger is closed")
        if self._current is not None:
            raise RuntimeError("previous DP batch stats step was not staged")
        monotonic_ns = (time.monotonic_ns()
                        if monotonic_ns is None else monotonic_ns)
        unix_ns = time.time_ns() if unix_ns is None else unix_ns
        self._finalize_pending(monotonic_ns)
        self._step_id += 1
        self._current = {
            "step_id": self._step_id,
            "rank": self.rank,
            "worker_rank": self.worker_rank,
            "step_start_monotonic_ns": monotonic_ns,
            "step_start_unix_ns": unix_ns,
        }
        return self._step_id

    def stage_step(self, stats: dict[str, Any]) -> None:
        if self._current is None:
            raise RuntimeError("begin_step() must precede stage_step()")
        if self._pending is not None:
            raise RuntimeError("pending DP batch stats record was not finalized")
        record = dict(self._current)
        record.update(stats)
        record.setdefault("step_duration", None)
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"DP batch stats record missing fields: {missing}")
        self._pending = record
        self._current = None

    def note_idle_scheduler_output(self, scheduler_output: Any) -> None:
        """Save queue depth from a zero-token schedule for its dummy step."""
        self._idle_queued_prefill_tokens = int(
            getattr(scheduler_output, "_tpu_queued_prefill_tokens", 0))

    def take_idle_queued_prefill_tokens(self) -> int:
        value = self._idle_queued_prefill_tokens
        self._idle_queued_prefill_tokens = 0
        return value

    def _finalize_pending(self, next_start_monotonic_ns: int) -> None:
        if self._pending is None:
            return
        start_ns = int(self._pending["step_start_monotonic_ns"])
        self._pending["step_duration"] = (
            next_start_monotonic_ns - start_ns) / 1_000_000
        self._write(self._pending)
        self._pending = None

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Worker shutdown can be signal-driven, in which case Python atexit
        # handlers are not guaranteed to run. One short line per ~47 ms model
        # step is cheap enough to flush eagerly and keeps captures complete.
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        # Preserve the last composition sample but do not pretend shutdown
        # latency is serving-step latency.
        if self._pending is not None:
            self._pending["step_duration"] = None
            self._write(self._pending)
            self._pending = None
        self._file.flush()
        self._file.close()
        self._closed = True


def load_records(stats_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load and validate every rank JSONL stream in ``stats_dir``."""
    paths = sorted(Path(stats_dir).glob("rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rank*.jsonl files under {stats_dir}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with path.open() as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields {missing}")
                key = (int(record["step_id"]), int(record["rank"]))
                if key in seen:
                    raise ValueError(f"duplicate step/rank record {key}")
                seen.add(key)
                records.append(record)
    return sorted(records, key=lambda row: (row["step_id"], row["rank"]))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_dp_size: int | None = None,
    target_bucket: int = 256,
) -> dict[str, Any]:
    """Calculate the DP packing metrics requested by the experiment."""
    rows = list(records)
    if not rows:
        raise ValueError("cannot analyze an empty DP batch stats stream")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ranks = sorted({int(row["rank"]) for row in rows})
    for row in rows:
        grouped[int(row["step_id"])].append(row)
    dp_size = expected_dp_size or len(ranks)
    expected_ranks = set(range(dp_size)) if expected_dp_size else set(ranks)
    missing_rank_steps = {
        str(step_id): sorted(expected_ranks -
                             {int(row["rank"]) for row in step_rows})
        for step_id, step_rows in grouped.items()
        if len(step_rows) != dp_size
    }

    prefill_steps = mixed_steps = sparse_steps = 0
    sparse_prefill_tokens = 0
    bucket_real = bucket_capacity = 0
    all_real = all_capacity = 0
    decode_exposure: list[int] = []
    active_prefill_rank_counts: list[int] = []
    per_step_prefill_imbalance: list[int] = []
    analysis_ranks = sorted(expected_ranks)
    rank_prefill_totals = {rank: 0 for rank in analysis_ranks}
    durations = [
        float(row["step_duration"])
        for row in rows
        if row.get("step_duration") is not None
    ]

    for step_rows in grouped.values():
        prefill_by_rank = {
            int(row["rank"]): int(row["real_prefill_tokens"])
            for row in step_rows
        }
        for rank, tokens in prefill_by_rank.items():
            rank_prefill_totals[rank] += tokens
        active_prefill_ranks = sum(tokens > 0 for tokens in prefill_by_rank.values())
        total_prefill = sum(prefill_by_rank.values())
        total_decode = sum(int(row["real_decode_tokens"]) for row in step_rows)
        if total_prefill > 0:
            prefill_steps += 1
            active_prefill_rank_counts.append(active_prefill_ranks)
            values = [prefill_by_rank.get(rank, 0) for rank in analysis_ranks]
            per_step_prefill_imbalance.append(max(values) - min(values))
            decode_exposure.append(
                sum(int(row["active_decode_requests"]) for row in step_rows))
            if total_decode > 0:
                mixed_steps += 1
            if 1 <= active_prefill_ranks <= min(4, dp_size):
                sparse_steps += 1
                sparse_prefill_tokens += total_prefill

        for row in step_rows:
            chunks = max(1, int(row.get("num_chunks", 1)))
            capacity = int(row["compiled_bucket"]) * chunks
            real = int(row["real_prefill_tokens"]) + int(
                row["real_decode_tokens"])
            all_real += real
            all_capacity += capacity
            if int(row["compiled_bucket"]) == target_bucket:
                bucket_real += real
                bucket_capacity += capacity

    step_count = len(grouped)
    rank_values = list(rank_prefill_totals.values())
    rank_mean = statistics.fmean(rank_values) if rank_values else 0.0
    rank_stddev = statistics.pstdev(rank_values) if len(rank_values) > 1 else 0.0
    theoretical_dense_cohorts = (
        math.ceil(sparse_prefill_tokens / (target_bucket * dp_size))
        if sparse_prefill_tokens else 0)

    return {
        "steps": step_count,
        "records": len(rows),
        "ranks": analysis_ranks,
        "expected_dp_size": dp_size,
        "missing_rank_steps": missing_rank_steps,
        "fraction_steps_containing_prefill": prefill_steps / step_count,
        "fraction_steps_mixed_prefill_decode": mixed_steps / step_count,
        "prefill_steps": prefill_steps,
        "mixed_prefill_decode_steps": mixed_steps,
        "active_prefill_ranks_mean": (
            statistics.fmean(active_prefill_rank_counts)
            if active_prefill_rank_counts else 0.0),
        "prefill_tokens_per_rank": rank_prefill_totals,
        "prefill_rank_imbalance": {
            "min_tokens": min(rank_values) if rank_values else 0,
            "max_tokens": max(rank_values) if rank_values else 0,
            "mean_tokens": rank_mean,
            "stddev_tokens": rank_stddev,
            "coefficient_of_variation": rank_stddev / rank_mean
            if rank_mean else 0.0,
            "mean_per_step_max_minus_min_tokens": (
                statistics.fmean(per_step_prefill_imbalance)
                if per_step_prefill_imbalance else 0.0),
        },
        "bucket_utilization": {
            "target_bucket": target_bucket,
            "real_tokens": bucket_real,
            "capacity_tokens": bucket_capacity,
            "utilization": bucket_real / bucket_capacity
            if bucket_capacity else 0.0,
        },
        "overall_compiled_capacity_utilization": all_real / all_capacity
        if all_capacity else 0.0,
        "decode_requests_exposed_per_prefill_step_mean": (
            statistics.fmean(decode_exposure) if decode_exposure else 0.0),
        "sparse_prefill": {
            "active_rank_range": [1, min(4, dp_size)],
            "steps": sparse_steps,
            "fraction_of_all_steps": sparse_steps / step_count,
            "fraction_of_prefill_steps": sparse_steps / prefill_steps
            if prefill_steps else 0.0,
            "real_prefill_tokens": sparse_prefill_tokens,
            "theoretical_dense_cohorts": theoretical_dense_cohorts,
            "theoretical_steps_avoided": max(
                0, sparse_steps - theoretical_dense_cohorts),
        },
        "step_duration_ms": {
            "count": len(durations),
            "mean": statistics.fmean(durations) if durations else None,
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
    }


def build_perfetto_trace(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a step-aligned Chrome/Perfetto JSON trace."""
    rows = list(records)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ranks = sorted({int(row["rank"]) for row in rows})
    for row in rows:
        grouped[int(row["step_id"])].append(row)

    events: list[dict[str, Any]] = []
    for rank in ranks:
        pid = rank + 1
        events.extend(({
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "args": {"name": f"DP rank {rank}"},
        }, {
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": 1,
            "args": {"name": "model steps"},
        }))

    timestamp_us = 0.0
    for step_id in sorted(grouped):
        step_rows = grouped[step_id]
        duration_values = [
            float(row["step_duration"])
            for row in step_rows
            if row.get("step_duration") is not None
        ]
        step_duration_ms = max(duration_values, default=0.001)
        total_prefill = sum(int(row["real_prefill_tokens"])
                            for row in step_rows)
        total_decode = sum(int(row["real_decode_tokens"])
                           for row in step_rows)
        active_prefill_ranks = sum(
            int(row["real_prefill_tokens"]) > 0 for row in step_rows)
        events.append({
            "name": "DP global step",
            "cat": "dp_summary",
            "ph": "X",
            "pid": 0,
            "tid": 0,
            "ts": timestamp_us,
            "dur": step_duration_ms * 1000,
            "args": {
                "step_id": step_id,
                "active_prefill_ranks": active_prefill_ranks,
                "real_prefill_tokens": total_prefill,
                "real_decode_tokens": total_decode,
            },
        })
        for row in step_rows:
            prefill = int(row["real_prefill_tokens"])
            decode = int(row["real_decode_tokens"])
            if prefill and decode:
                phase = "mixed"
            elif prefill:
                phase = "prefill"
            elif decode:
                phase = "decode"
            else:
                phase = "dummy"
            row_duration_ms = row.get("step_duration")
            duration_ms = (step_duration_ms if row_duration_ms is None else
                           float(row_duration_ms))
            args = {field: row[field] for field in REQUIRED_FIELDS
                    if field != "step_duration"}
            args.update({
                "step_duration_ms": row_duration_ms,
                "num_chunks": int(row.get("num_chunks", 1)),
                "phase": phase,
            })
            events.append({
                "name": f"{phase}: {prefill}P + {decode}D",
                "cat": "dp_batch",
                "ph": "X",
                "pid": int(row["rank"]) + 1,
                "tid": 1,
                "ts": timestamp_us,
                "dur": max(duration_ms * 1000, 1.0),
                "args": args,
            })
        timestamp_us += step_duration_ms * 1000

    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
    }


def write_converted_outputs(
    stats_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_dp_size: int | None = None,
    target_bucket: int = 256,
) -> dict[str, Path]:
    records = load_records(stats_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "dp-batch-stats.jsonl"
    analysis_path = output / "dp-batch-analysis.json"
    trace_path = output / "dp-batch-perfetto.json"
    with merged_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    analysis_path.write_text(json.dumps(
        analyze_records(records,
                        expected_dp_size=expected_dp_size,
                        target_bucket=target_bucket),
        indent=2,
        sort_keys=True,
    ) + "\n")
    trace_path.write_text(json.dumps(build_perfetto_trace(records),
                                     separators=(",", ":")) + "\n")
    return {
        "merged": merged_path,
        "analysis": analysis_path,
        "perfetto": trace_path,
    }
