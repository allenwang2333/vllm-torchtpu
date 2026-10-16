# SPDX-License-Identifier: Apache-2.0
"""Lightweight, zero-device-sync host step telemetry for TPU runner & scheduler.

Gated by ``VLLM_TPU_LIGHTWEIGHT_STEP_STATS=1`` (default ``0``). Records pure
host ``time.perf_counter_ns()`` durations in fixed-capacity ring buffers with
zero ``torch_xla.sync()`` or device waits, summarizing p50/p90/p99 every
``N=500`` steps on rank 0 (and on the scheduler process):
  (a) scheduler ``schedule()`` duration,
  (b) KV offload H2D transfer poll/wait duration,
  (c) ``_prepare_inputs`` duration,
  (d) ``mk_backbone`` vs ``stock_mixed`` dispatch counts & token bucket
      distributions (``B=4..64`` vs ``T=384,512,1024,2048``),
  (e) ``eagle3`` draft preparation duration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_SUMMARY_INTERVAL = 500


def is_enabled() -> bool:
    """Return True when lightweight host step telemetry is enabled."""
    return bool(envs.VLLM_TPU_LIGHTWEIGHT_STEP_STATS)


def get_summary_interval() -> int:
    """Return the step interval between summary logs (default 500)."""
    try:
        return max(1, int(envs.VLLM_TPU_LIGHTWEIGHT_STEP_STATS_INTERVAL))
    except (TypeError, ValueError):
        return _DEFAULT_SUMMARY_INTERVAL


class _NsRingBuffer:
    """Fixed-size ring buffer storing nanosecond durations without allocations."""

    __slots__ = ("_buf", "_cap", "_count", "_idx")

    def __init__(self, capacity: int = _DEFAULT_SUMMARY_INTERVAL) -> None:
        self._cap = max(1, int(capacity))
        self._buf = np.zeros(self._cap, dtype=np.int64)
        self._idx = 0
        self._count = 0

    def append(self, duration_ns: int) -> None:
        self._buf[self._idx] = duration_ns
        self._idx = (self._idx + 1) % self._cap
        if self._count < self._cap:
            self._count += 1

    def clear(self) -> None:
        self._idx = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def percentiles_us(self) -> tuple[float, float, float]:
        """Return (p50_us, p90_us, p99_us) over current window."""
        if self._count == 0:
            return (0.0, 0.0, 0.0)
        view = self._buf[: self._count]
        p50, p90, p99 = np.percentile(view, [50, 90, 99])
        return (float(p50) * 1e-3, float(p90) * 1e-3, float(p99) * 1e-3)


class LightweightStepStatsRecorder:
    """Process-local ring-buffer telemetry for scheduler and worker rank 0."""

    def __init__(self, interval: int = _DEFAULT_SUMMARY_INTERVAL) -> None:
        self.reset(interval=interval)

    def reset(self, interval: int | None = None) -> None:
        cap = interval if interval is not None else get_summary_interval()
        self.interval = max(1, int(cap))
        # Scheduler-side ring buffers & counters
        self.sched_steps = 0
        self.schedule_ns = _NsRingBuffer(self.interval)
        self.kv_offload_wait_ns = _NsRingBuffer(self.interval)
        self.kv_offload_Wait_nonzero_ns = _NsRingBuffer(self.interval)
        self.kv_offload_early_completed_reqs = 0
        self.kv_offload_waited_completed_reqs = 0
        self.kv_offload_wait_calls = 0

        # Worker-side ring buffers & counters
        self.runner_steps = 0
        self.prepare_inputs_ns = _NsRingBuffer(self.interval)
        self.eagle3_draft_prep_ns = _NsRingBuffer(self.interval)
        self.worker_kv_setup_ns = _NsRingBuffer(self.interval)
        self.mk_backbone_count = 0
        self.stock_mixed_count = 0
        self.mk_bucket_counts: Counter[int] = Counter()
        self.stock_mixed_bucket_counts: Counter[int] = Counter()

    def record_scheduler_step(
        self,
        schedule_ns: int,
        kv_wait_ns: int = 0,
        early_completed_reqs: int = 0,
        waited_completed_reqs: int = 0,
        waited: bool = False,
    ) -> str | None:
        self.sched_steps += 1
        self.schedule_ns.append(schedule_ns)
        self.kv_offload_wait_ns.append(kv_wait_ns)
        if waited or kv_wait_ns > 0:
            self.kv_offload_Wait_nonzero_ns.append(kv_wait_ns)
        if waited:
            self.kv_offload_wait_calls += 1
        self.kv_offload_early_completed_reqs += early_completed_reqs
        self.kv_offload_waited_completed_reqs += waited_completed_reqs

        if self.sched_steps % self.interval == 0:
            msg = self.format_scheduler_summary()
            logger.info("%s", msg)
            self._clear_scheduler_window()
            return msg
        return None

    def _clear_scheduler_window(self) -> None:
        self.schedule_ns.clear()
        self.kv_offload_wait_ns.clear()
        self.kv_offload_Wait_nonzero_ns.clear()
        self.kv_offload_early_completed_reqs = 0
        self.kv_offload_waited_completed_reqs = 0
        self.kv_offload_wait_calls = 0

    def format_scheduler_summary(self) -> str:
        s50, s90, s99 = self.schedule_ns.percentiles_us()
        w50, w90, w99 = self.kv_offload_wait_ns.percentiles_us()
        wn50, wn90, wn99 = self.kv_offload_Wait_nonzero_ns.percentiles_us()
        return (
            f"[LightweightStepStats][Scheduler step={self.sched_steps}] "
            f"schedule_us(p50/p90/p99)={s50:.1f}/{s90:.1f}/{s99:.1f} | "
            f"kv_h2d_poll_wait_us(p50/p90/p99)={w50:.1f}/{w90:.1f}/{w99:.1f} "
            f"(active_wait_us={wn50:.1f}/{wn90:.1f}/{wn99:.1f}, "
            f"waits={self.kv_offload_wait_calls}, "
            f"early_promoted={self.kv_offload_early_completed_reqs}, "
            f"wait_promoted={self.kv_offload_waited_completed_reqs})"
        )

    def record_worker_step(
        self,
        prepare_inputs_ns: int,
        *,
        is_mk_backbone: bool | None = None,
        padded_tokens: int = 0,
        eagle3_draft_prep_ns: int | None = None,
        kv_setup_ns: int = 0,
        emit_log: bool = True,
    ) -> str | None:
        self.runner_steps += 1
        self.prepare_inputs_ns.append(prepare_inputs_ns)
        if kv_setup_ns > 0:
            self.worker_kv_setup_ns.append(kv_setup_ns)
        if eagle3_draft_prep_ns is not None:
            self.eagle3_draft_prep_ns.append(eagle3_draft_prep_ns)
        if is_mk_backbone is True:
            self.mk_backbone_count += 1
            if padded_tokens > 0:
                self.mk_bucket_counts[int(padded_tokens)] += 1
        elif is_mk_backbone is False:
            self.stock_mixed_count += 1
            if padded_tokens > 0:
                self.stock_mixed_bucket_counts[int(padded_tokens)] += 1

        if self.runner_steps % self.interval == 0:
            msg = self.format_worker_summary()
            if emit_log:
                logger.info("%s", msg)
            self._clear_worker_window()
            return msg
        return None

    def record_eagle3_draft_prep(self, duration_ns: int) -> None:
        self.eagle3_draft_prep_ns.append(duration_ns)

    def _clear_worker_window(self) -> None:
        self.prepare_inputs_ns.clear()
        self.eagle3_draft_prep_ns.clear()
        self.worker_kv_setup_ns.clear()
        self.mk_backbone_count = 0
        self.stock_mixed_count = 0
        self.mk_bucket_counts.clear()
        self.stock_mixed_bucket_counts.clear()

    def format_worker_summary(self) -> str:
        p50, p90, p99 = self.prepare_inputs_ns.percentiles_us()
        e50, e90, e99 = self.eagle3_draft_prep_ns.percentiles_us()
        k50, k90, k99 = self.worker_kv_setup_ns.percentiles_us()
        mk_buckets_str = (
            ",".join(f"B{b}:{c}" for b, c in sorted(self.mk_bucket_counts.items()))
            or "none"
        )
        stock_buckets_str = (
            ",".join(
                f"T{t}:{c}" for t, c in sorted(self.stock_mixed_bucket_counts.items())
            )
            or "none"
        )
        return (
            f"[LightweightStepStats][Worker step={self.runner_steps}] "
            f"prepare_inputs_us(p50/p90/p99)={p50:.1f}/{p90:.1f}/{p99:.1f} | "
            f"eagle3_draft_prep_us(p50/p90/p99)={e50:.1f}/{e90:.1f}/{e99:.1f} | "
            f"kv_setup_us(p50/p90/p99)={k50:.1f}/{k90:.1f}/{k99:.1f} | "
            f"dispatch: mk_backbone={self.mk_backbone_count} [{mk_buckets_str}] "
            f"stock_mixed={self.stock_mixed_count} [{stock_buckets_str}]"
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a structured snapshot of current window statistics."""
        return {
            "sched_steps": self.sched_steps,
            "schedule_us": self.schedule_ns.percentiles_us(),
            "kv_offload_wait_us": self.kv_offload_wait_ns.percentiles_us(),
            "kv_offload_early_completed_reqs": self.kv_offload_early_completed_reqs,
            "kv_offload_waited_completed_reqs": self.kv_offload_waited_completed_reqs,
            "runner_steps": self.runner_steps,
            "prepare_inputs_us": self.prepare_inputs_ns.percentiles_us(),
            "eagle3_draft_prep_us": self.eagle3_draft_prep_ns.percentiles_us(),
            "worker_kv_setup_us": self.worker_kv_setup_ns.percentiles_us(),
            "mk_backbone_count": self.mk_backbone_count,
            "stock_mixed_count": self.stock_mixed_count,
            "mk_bucket_counts": dict(self.mk_bucket_counts),
            "stock_mixed_bucket_counts": dict(self.stock_mixed_bucket_counts),
        }


_GLOBAL_RECORDER: LightweightStepStatsRecorder | None = None


def get_step_stats_recorder() -> LightweightStepStatsRecorder:
    global _GLOBAL_RECORDER
    if _GLOBAL_RECORDER is None:
        _GLOBAL_RECORDER = LightweightStepStatsRecorder()
    return _GLOBAL_RECORDER
