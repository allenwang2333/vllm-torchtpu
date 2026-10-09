# SPDX-License-Identifier: Apache-2.0
"""Periodic log of the runner's step period, grouped by DP token bucket.

The period between two consecutive ``execute_model`` starts is what a serving
step costs end to end (device time plus any host time on the critical path),
measured without synchronizing the device. It is attributed to the bucket of
the step that opened the period.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A period longer than this is an idle gap between waves, not a step.
_MAX_STEP_S = 5.0


class StepTimingLog:
    def __init__(self, rank: int, log_every_steps: int):
        self._rank = rank
        self._every = log_every_steps
        self._prev_start: float | None = None
        self._prev_bucket: int | None = None
        self._bucket: int | None = None
        self._periods: dict[int, list[float]] = defaultdict(list)
        self._sync_s: list[float] = []
        self._steps = 0

    def step_start(self) -> None:
        now = time.perf_counter()
        if self._prev_start is not None and self._prev_bucket is not None:
            period = now - self._prev_start
            if period < _MAX_STEP_S:
                self._periods[self._prev_bucket].append(period)
        self._prev_start = now
        self._prev_bucket = None

    def dp_sync(self, bucket: int, seconds: float) -> None:
        self._prev_bucket = bucket
        self._sync_s.append(seconds)
        self._steps += 1
        if self._steps % self._every == 0:
            self._flush()

    def _flush(self) -> None:
        total = sum(len(v) for v in self._periods.values())
        if total == 0:
            return
        all_periods = np.concatenate([np.asarray(v) for v in self._periods.values()])
        parts = []
        for bucket in sorted(self._periods):
            p = np.asarray(self._periods[bucket]) * 1e3
            parts.append(
                f"T{bucket}: n={p.size} ({100.0 * p.size / total:.0f}%) "
                f"mean={p.mean():.1f} p50={np.median(p):.1f} "
                f"p90={np.percentile(p, 90):.1f}"
            )
        sync = np.asarray(self._sync_s) * 1e3
        logger.info(
            "step timing rank=%d steps=%d period_ms mean=%.1f p50=%.1f | "
            "dp_sync_ms mean=%.2f p90=%.2f | %s",
            self._rank,
            total,
            all_periods.mean() * 1e3,
            np.median(all_periods) * 1e3,
            sync.mean(),
            np.percentile(sync, 90),
            " | ".join(parts),
        )
        self._periods.clear()
        self._sync_s.clear()
