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

**Never call ``jax.block_until_ready`` inside a timing loop.** Doing so charges
a full host round trip to every sample. Measured on this tpu7x host
(2026-09-17, jax 0.10.2 / libtpu 0.0.44.1) the round trip is **146-190 us**:

===========================  =========  =========  ==========
arm                          blocking   pipelined  inflation
===========================  =========  =========  ==========
decode gather (0.22 ms)        403.8       224.2      1.80x
prefill bitmap (0.70 ms)       873.3       700.6      1.25x
prefill gather (2.35 ms)      2510.7      2349.4      1.07x
===========================  =========  =========  ==========

Three consequences, all of which have bitten this repo's benchmarks:

1. Anything under ~1 ms is materially wrong; decode reads 80% high.
2. The offset is **additive**, so it inflates the faster arm proportionally
   more. Every A/B ratio is biased toward 1 -- the winner is reported as
   winning by less than it does, and a block-size tuner comparing candidates
   will call real differences a tie.
3. It is ~8x noisier: 10.8% run-to-run on decode versus 1.4% pipelined.

Dispatching a burst and blocking once at the end lets JAX's async dispatch
overlap call i+1's host cost with call i's device execution, so per-call time
converges on device time. Sweeping burst length on the decode gather gave
425/304/250/235/224/219/218 us at iters=1/2/5/10/20/50/100 -- flat past 20,
hence ``MIN_ITERS``. Cross-checked against the xplane device clock: pipelined
runs 10-40 us (0.6-5%) above true per-call kernel time, which is the residual
dispatch gap and the error bar to quote.

The device clock itself needs jax 0.11.0; jax 0.10.2 SIGSEGVs in
``PluginTracer::Start()`` because jaxlib expects an 80-byte
``PLUGIN_Profiler_Api`` and libtpu 0.0.44.1 supplies 104. See docs/profiling.md
for the serving-path profiler, which is a different mechanism.
"""

import statistics
import time
from typing import Any, Callable, Sequence

import jax

# Below 20 the burst has not converged: iters=10 reads 7.6% above the plateau
# on the fastest arm here.
MIN_ITERS = 20
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
    """Median per-call time of ``fn(*args)`` in ms. The clock to use.

    Median over ``bursts`` so one scheduling hiccup does not set the number.
    """
    _check_iters(iters)
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))
    return statistics.median(burst_ms(fn, args, iters) for _ in range(bursts))


def blocking_ms(fn: Callable[..., Any],
                *args: Any,
                iters: int = DEFAULT_ITERS,
                warmup: int = DEFAULT_WARMUP) -> float:
    """Median per-call time with a sync after every call. **Not a kernel time.**

    This is the clock this module exists to replace, kept only so callers can
    report it *alongside* `pipelined_ms` and expose the gap. Never rank or gate
    on it: it adds a full host round trip to every sample, and that offset is
    additive, so it biases ratios between arms of different speed toward 1.

    Unlike `pipelined_ms` this honours `warmup=0` rather than forcing one call,
    so `both_ms` does not re-warm a kernel the pipelined arm just ran.
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
    """`(pipelined_ms, blocking_ms)`. Their difference is this host's dispatch cost.

    Costs ~1% more than `pipelined_ms` alone and makes every recorded result
    carry the evidence of its own clock health: the gap should sit near 0.15 ms
    on tpu7x, and a result where it has drifted is telling you the measurement
    environment changed, not the kernel.

    Report the pipelined value; keep the blocking one as a diagnostic.
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
    """Two arms timed with alternating order, to cancel drift between them.

    Use for an A/B: the arms interleave, so a device that warms or throttles
    part-way through penalizes both equally instead of whichever ran second.
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
