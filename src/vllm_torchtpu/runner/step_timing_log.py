# SPDX-License-Identifier: Apache-2.0
"""Periodic log of the runner's step period, grouped by DP token bucket.

The period between two consecutive ``execute_model`` starts is what a serving
step costs end to end (device time plus any host time on the critical path),
measured without synchronizing the device. It is attributed to the bucket of
the step that opened the period.

Each flush also logs the DP step mix: a step is pure decode only when no DP
rank scheduled a prefill token, since one prefilling rank sets the bucket that
every rank runs.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

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
        self._prev_any_prefill = False
        self._periods: dict[int, list[float]] = defaultdict(list)
        # (any rank prefilled, bucket) -> periods.
        self._mix_periods: dict[tuple[bool, int], list[float]] = defaultdict(list)
        # This rank's (tokens, requests, summed context) on pure decode steps.
        self._decode_local: list[tuple[int, int, int]] = []
        # Every rank's (requests, tokens, prefill) per step, rows in that order.
        self._decode_spread: list[np.ndarray] = []
        self._mixed_spread: list[np.ndarray] = []
        self._sync_s: list[float] = []
        self._result_wait_s: list[float] = []
        self._steps = 0

    def step_start(self) -> None:
        now = time.perf_counter()
        if self._prev_start is not None and self._prev_bucket is not None:
            period = now - self._prev_start
            if period < _MAX_STEP_S:
                self._periods[self._prev_bucket].append(period)
                key = (self._prev_any_prefill, self._prev_bucket)
                self._mix_periods[key].append(period)
        self._prev_start = now
        self._prev_bucket = None

    def dp_sync(
        self,
        bucket: int,
        seconds: float,
        any_prefill: bool = False,
        local: tuple[int, int, int] = (0, 0, 0),
        per_rank: np.ndarray | None = None,
    ) -> None:
        self._prev_bucket = bucket
        self._prev_any_prefill = any_prefill
        if not any_prefill:
            self._decode_local.append(local)
        if per_rank is not None and per_rank.size:
            (self._mixed_spread if any_prefill else self._decode_spread).append(per_rank)
        self._sync_s.append(seconds)
        self._steps += 1
        if self._steps % self._every == 0:
            self._flush()

    def result_wait(self, seconds: float) -> None:
        # Host time blocked on the previous step's sampled-token copy: near
        # zero means the host, not the device, sets the step period.
        self._result_wait_s.append(seconds)

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
        wait = np.asarray(self._result_wait_s or [0.0]) * 1e3
        logger.info(
            "step timing rank=%d steps=%d period_ms mean=%.1f p50=%.1f | "
            "dp_sync_ms mean=%.2f p90=%.2f | result_wait_ms mean=%.2f p10=%.2f | %s",
            self._rank,
            total,
            all_periods.mean() * 1e3,
            np.median(all_periods) * 1e3,
            sync.mean(),
            np.percentile(sync, 90),
            wait.mean(),
            np.percentile(wait, 10),
            " | ".join(parts),
        )
        self._flush_mix(total, float(all_periods.sum()))
        self._flush_spread()
        self._periods.clear()
        self._mix_periods.clear()
        self._decode_local.clear()
        self._decode_spread.clear()
        self._mixed_spread.clear()
        self._sync_s.clear()
        self._result_wait_s.clear()

    def _flush_mix(self, total: int, total_s: float) -> None:
        parts = []
        for any_prefill, bucket in sorted(self._mix_periods):
            p = np.asarray(self._mix_periods[(any_prefill, bucket)]) * 1e3
            parts.append(
                f"{'mixed' if any_prefill else 'decode'}/T{bucket}: n={p.size} "
                f"sum_ms={p.sum():.0f} p50={np.median(p):.1f}"
            )
        decode = [
            np.asarray(v)
            for (any_prefill, _), v in self._mix_periods.items()
            if not any_prefill
        ]
        decode_n = sum(v.size for v in decode)
        decode_s = sum(float(v.sum()) for v in decode)
        local = np.asarray(self._decode_local or [(0, 0, 0)], dtype=np.float64)
        logger.info(
            "step mix rank=%d steps=%d pure_decode n=%d (%.1f%%) time_share=%.1f%% | "
            "decode local tokens mean=%.2f reqs mean=%.2f ctx_sum mean=%.0f | %s",
            self._rank,
            total,
            decode_n,
            100.0 * decode_n / total,
            100.0 * decode_s / total_s if total_s > 0 else 0.0,
            local[:, 0].mean(),
            local[:, 1].mean(),
            local[:, 2].mean(),
            " | ".join(parts),
        )

    def _flush_spread(self) -> None:
        def hist(values) -> str:
            return ",".join(f"{k}:{n}" for k, n in sorted(Counter(values).items()))

        parts = []
        if self._decode_spread:
            d = np.stack(self._decode_spread)  # [steps, 3, ranks]
            reqs, tokens = d[:, 0], d[:, 1]
            parts.append(
                f"decode: active_ranks {hist((reqs > 0).sum(1).tolist())} "
                f"max_reqs {hist(reqs.max(1).tolist())} "
                f"max_tokens {hist(tokens.max(1).tolist())} "
                f"sum_reqs mean={reqs.sum(1).mean():.2f} "
                f"sum_tokens mean={tokens.sum(1).mean():.1f} "
                f"rank_reqs {hist(reqs.flatten().tolist())}"
            )
        if self._mixed_spread:
            m = np.stack(self._mixed_spread)
            parts.append(
                f"mixed: prefill_ranks {hist(m[:, 2].sum(1).tolist())} "
                f"active_ranks {hist((m[:, 0] > 0).sum(1).tolist())} "
                f"max_tokens {hist(m[:, 1].max(1).tolist())}"
            )
        if parts:
            logger.info("step spread rank=%d | %s", self._rank, " | ".join(parts))
