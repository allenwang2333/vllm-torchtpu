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

Never call ``jax.block_until_ready`` inside a timing loop: it charges a ~150 us
host round trip to every sample, which is additive and so inflates fast kernels
most (1.8x at 0.22 ms, 1.07x at 2.35 ms) and drags A/B ratios toward 1.
Dispatch a burst and block once instead.

Timing one kernel:

    pipelined_ms   the default clock
    slope_ms       more accurate for ~3x the calls: (t(2n) - t(n)) / n
    blocking_ms    a sync per call; diagnostic only, never rank on it
    both_ms        (pipelined, blocking), to show the dispatch gap
    burst_ms       one raw burst, no warmup or median; the primitive

Comparing two kernels:

    paired_ms      A against B, alternating which runs first so that drift
                   hits both arms equally. Each arm uses the burst above.

MIN_ITERS is measured: per-call time is flat past 20 calls, and 10 reads ~8% high.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax

MIN_ITERS = 20  # see module docstring

DEFAULT_ITERS = 50
DEFAULT_BURSTS = 3
DEFAULT_WARMUP = 3


def _check_iters(iters: int) -> None:
    if iters < MIN_ITERS:
        raise ValueError(
            f"iters={iters} is below MIN_ITERS={MIN_ITERS}; a shorter burst "
            "has not converged and reads high on fast kernels. Pass "
            f"iters>={MIN_ITERS} (default {DEFAULT_ITERS})."
        )


def burst_ms(fn: Callable[..., Any], args: Sequence[Any], iters: int) -> float:
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
    """Median per-call ms for one kernel, over `bursts`. The clock to use."""
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    return statistics.median(burst_ms(fn, args, iters) for _ in range(bursts))


def slope_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    bursts: int = DEFAULT_BURSTS,
) -> float:
    """Marginal per-call ms: (t(2n) - t(n)) / n, so fixed cost cancels exactly.

    Closer to device time than pipelined_ms for ~3x the calls; use it for rooflines.
    """
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    out = []
    for _ in range(bursts):
        one = burst_ms(fn, args, iters) * iters
        two = burst_ms(fn, args, 2 * iters) * 2 * iters
        out.append((two - one) / iters)
    return statistics.median(out)


def blocking_ms(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
) -> float:
    """Per-call time with a sync every call. NOT a kernel time -- diagnostic only.

    Honors warmup=0, unlike `pipelined_ms`, so `both_ms` does not re-warm.
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
    """(pipelined, blocking); the gap is this host's dispatch cost, ~0.15 ms.

    Quote the pipelined value; a drifted gap means the environment moved.
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
    bursts: int = DEFAULT_BURSTS,
) -> tuple[float, float]:
    """Per-call ms for two kernels, A against B, as (first, second).

    Alternates which arm leads each burst, so drift penalizes both equally.
    """
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(first(*args))
        jax.block_until_ready(second(*args))
    samples: tuple[list[float], list[float]] = ([], [])
    for burst in range(bursts):
        arms = ((first, samples[0]), (second, samples[1]))
        if burst % 2:
            arms = arms[::-1]
        for fn, values in arms:
            values.append(burst_ms(fn, args, iters))
    return statistics.median(samples[0]), statistics.median(samples[1])
