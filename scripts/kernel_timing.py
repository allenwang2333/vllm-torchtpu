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

Never call ``jax.block_until_ready`` inside a timing loop. It charges a host
round trip -- ~150 us on tpu7x -- to every sample. The cost is additive, so it
inflates fast kernels most (1.8x on a 0.22 ms kernel, 1.07x on a 2.35 ms one),
drags every A/B ratio toward 1, and is ~8x noisier than the alternative.

Instead dispatch a burst and block once, so JAX's async dispatch overlaps call
i+1's host cost with call i's device execution:

    pipelined_ms   the default
    slope_ms       (t(2n)-t(n))/n; cancels the fixed cost exactly, ~2x closer
                   to true device time, for ~3x the calls
    both_ms        (pipelined, blocking) to expose the gap as a diagnostic
    paired_ms      an A/B with alternating arm order

MIN_ITERS is measured, not chosen: per-call time is flat past 20 calls and a
burst of 10 still reads ~8% high.
"""

import statistics
import time
from typing import Any, Callable, Sequence

import jax

MIN_ITERS = 20      # see module docstring

DEFAULT_ITERS = 50
DEFAULT_BURSTS = 3
DEFAULT_WARMUP = 3


def _check_iters(iters: int) -> None:
    if iters < MIN_ITERS:
        raise ValueError(
            f"iters={iters} is below MIN_ITERS={MIN_ITERS}; a shorter burst "
            "has not converged and reads high on fast kernels. Pass "
            f"iters>={MIN_ITERS} (default {DEFAULT_ITERS}).")


def burst_ms(fn: Callable[..., Any], args: Sequence[Any], iters: int) -> float:
    """Per-call milliseconds of one pipelined burst: dispatch N, block once."""
    out = None
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) * 1e3 / iters


def pipelined_ms(fn: Callable[..., Any],
                 *args: Any,
                 iters: int = DEFAULT_ITERS,
                 warmup: int = DEFAULT_WARMUP,
                 bursts: int = DEFAULT_BURSTS) -> float:
    """Median per-call ms over `bursts`. The clock to use."""
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    return statistics.median(burst_ms(fn, args, iters) for _ in range(bursts))


def slope_ms(fn: Callable[..., Any],
             *args: Any,
             iters: int = DEFAULT_ITERS,
             warmup: int = DEFAULT_WARMUP,
             bursts: int = DEFAULT_BURSTS) -> float:
    """Marginal per-call ms: (t(2n) - t(n)) / n. Fixed cost cancels exactly.

    Measurably closer to true device time than pipelined_ms, for ~3x the calls.
    Use it when the number feeds a roofline or a close A/B.
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


def blocking_ms(fn: Callable[..., Any],
                *args: Any,
                iters: int = DEFAULT_ITERS,
                warmup: int = DEFAULT_WARMUP) -> float:
    """Per-call time with a sync every call. NOT a kernel time -- diagnostic only.

    Kept so callers can show the gap beside `pipelined_ms`. Never rank on it.
    Honours warmup=0, unlike `pipelined_ms`, so `both_ms` does not re-warm.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def both_ms(fn: Callable[..., Any],
            *args: Any,
            iters: int = DEFAULT_ITERS,
            warmup: int = DEFAULT_WARMUP,
            bursts: int = DEFAULT_BURSTS) -> tuple[float, float]:
    """(pipelined, blocking); the gap is this host's dispatch cost, ~0.15 ms.

    Costs ~1% extra. Quote the pipelined value; a drifted gap means the
    environment changed, not the kernel.
    """
    pipe = pipelined_ms(fn, *args, iters=iters, warmup=warmup, bursts=bursts)
    wall = blocking_ms(fn, *args, iters=iters, warmup=0)
    return pipe, wall


def paired_ms(first: Callable[..., Any],
              second: Callable[..., Any],
              *args: Any,
              iters: int = DEFAULT_ITERS,
              warmup: int = DEFAULT_WARMUP,
              bursts: int = DEFAULT_BURSTS) -> tuple[float, float]:
    """A/B with alternating arm order, so drift penalises both arms equally."""
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
