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
"""The clock for standalone JAX/Pallas kernel microbenchmarks.

Never call ``jax.block_until_ready`` inside a timing loop: it charges a host
round trip to every sample. The cost is additive, so it inflates fast kernels
proportionally more and drags every A/B ratio toward 1. Dispatch a burst and
block once instead.

Timing one kernel:

    pipelined_ms   the default clock
    slope_ms       more accurate at ~3x the calls: (t(2n) - t(n)) / n
    blocking_ms    a sync per call; diagnostic only, never rank on it
    both_ms        (pipelined, blocking), to show the dispatch gap
    burst_ms       one raw burst, no warmup or median; the primitive

Comparing two kernels:

    paired_ms      A against B, alternating which runs first so that drift
                   hits both arms equally. Each arm uses the burst above.

MIN_ITERS is a floor, not a guarantee of convergence: how many calls it takes to
amortize the fixed cost scales with how fast the kernel is.
"""

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax

MIN_ITERS = 20

DEFAULT_ITERS = 50
DEFAULT_BURSTS = 3
DEFAULT_WARMUP = 3

# Even, so each arm leads the same number of bursts.
PAIRED_BURSTS = 4


def _check_iters(iters: int) -> None:
    if iters < MIN_ITERS:
        raise ValueError(
            f"iters={iters} is below MIN_ITERS={MIN_ITERS}; a shorter burst "
            "has not amortized the fixed cost and reads high on fast kernels. "
            f"Pass iters>={MIN_ITERS} (default {DEFAULT_ITERS})."
        )


def burst_ms(fn: Callable[..., Any], *args: Any, iters: int) -> float:
    """Per-call milliseconds of one pipelined burst: dispatch N, block once."""
    out = None
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) * 1e3 / iters


def pipelined_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    bursts: int = DEFAULT_BURSTS,
) -> float:
    """Median per-call ms for one kernel, over `bursts`."""
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    return statistics.median(burst_ms(fn, *args, iters=iters) for _ in range(bursts))


def slope_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    bursts: int = DEFAULT_BURSTS,
) -> float:
    """Marginal per-call ms: (t(2n) - t(n)) / n, so fixed cost cancels exactly.

    Closer to device time than `pipelined_ms` at ~3x the calls. Use it for a
    roofline, where the number is a denominator rather than a ranking.
    """
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    out = []
    for _ in range(bursts):
        one = burst_ms(fn, *args, iters=iters) * iters
        two = burst_ms(fn, *args, iters=2 * iters) * 2 * iters
        out.append((two - one) / iters)
    return statistics.median(out)


def blocking_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
) -> float:
    """Per-call time with a sync after each call, so it includes host dispatch.

    Diagnostic only -- never rank on it. Unlike `pipelined_ms`, a warmup of 0
    is respected, so `both_ms` warms once rather than twice.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def both_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    bursts: int = DEFAULT_BURSTS,
) -> tuple[float, float]:
    """One kernel on both clocks, as (pipelined, blocking).

    The gap between them is this host's dispatch cost. Quote the pipelined
    value; a gap that has drifted means the environment moved.
    """
    pipe = pipelined_ms(fn, *args, iters=iters, warmup=warmup, bursts=bursts)
    wall = blocking_ms(fn, *args, iters=iters, warmup=0)
    return pipe, wall


def paired_ms(
    first: Callable[..., Any],
    second: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    bursts: int = PAIRED_BURSTS,
) -> tuple[float, float]:
    """Per-call ms for two kernels, A against B, as (first, second).

    Alternates which arm leads each burst, so drift penalizes both equally.
    """
    _check_iters(iters)
    if bursts % 2:
        raise ValueError(
            f"bursts={bursts} is odd, so one arm leads once more than the "
            "other and identical arms differ by the position penalty. Pass "
            f"an even count (default {PAIRED_BURSTS})."
        )
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(first(*args))
        jax.block_until_ready(second(*args))
    samples: tuple[list[float], list[float]] = ([], [])
    for burst in range(bursts):
        arms = ((first, samples[0]), (second, samples[1]))
        if burst % 2:
            arms = arms[::-1]
        for fn, values in arms:
            values.append(burst_ms(fn, *args, iters=iters))
    return statistics.median(samples[0]), statistics.median(samples[1])
