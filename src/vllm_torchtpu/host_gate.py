# SPDX-License-Identifier: Apache-2.0
"""Whether the model runner is mid-dispatch, for work that can wait for a gap.

A prefill step is 175 ms on the local 35B pair, and for about 168 of those the
host has nothing to do but wait for the chip. The exception is the 6.7 ms at
the start where it packages the step and hands it over. Work landing in that
window delays the handover, and because the PCP ranks enqueue collectively,
delaying one rank delays the step.

That is what stage 5 does. The producer's transfer submit fires from
start_load_kv, which vLLM calls from inside execute_model, so it lands in the
dispatch window on every single request. Measured on the local pair, that costs
1.29 ms of step period and 0.60% of saturated throughput, and holding the
submit back by 30 ms gives all of it back. Eighty gives back no more, so it is
the collision and not the call.

A fixed delay is the wrong fix because it cannot tell a busy host from an idle
one. At concurrency 32 the 30 ms vanished into a five-second queue and TTFT
improved; at concurrency 1 there is no queue and it cost the full 30 ms, which
is 11.6% of TTFT and precisely the case stage 5 exists to improve.

So gate on the thing itself. The flag is clear while the host is packaging a
step and set once the step is on the chip. A submit arriving mid-dispatch waits
out the remainder of that dispatch and no longer, because the step it is
sitting inside is already on its way to the chip; there is no waiting for some
future step that may never come. At concurrency 1 that is a few milliseconds
instead of thirty.

The runner and the connector's submit workers are in the same worker process,
so this is a plain in-process flag. It carries no request state and is never
awaited by the runner itself, so a caller that forgets to clear it can only
cost latency, never correctness.

The one way it can cost a lot of latency is if a step clears the flag and then
returns without setting it, because the next thing to set it is the next step,
and at low load there may not be one for a long time. That is not hypothetical:
the first version cost 50.4 ms of TTFT at concurrency 1, which was the timeout
to the decimal, because execute_model's zero-token path returns early and that
is precisely the path the producer submit fires from when prefill has nothing
else queued. Every exit from a step now sets the flag, and the counters below
exist so that a regression of that shape shows up as a timeout count rather
than as a mysterious latency.
"""

import threading
import time

# vllm_torchtpu's own init_logger, not vllm's. The vllm one hands back a logger
# whose effective level is WARNING in this package, so the first version of
# these counters logged nothing at all and the run that was supposed to prove
# the timeouts were gone proved only that grep found no lines. This module
# imports nothing else from the package, so it stays a leaf.
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Set, not clear, at import. A process that never runs a step has no dispatch
# window to avoid, and anything waiting here should pass straight through
# rather than sit out a timeout.
_gap = threading.Event()
_gap.set()

# A waiter that times out has not found a gap, it has given up looking for one,
# and the two are indistinguishable from the outside. Count them separately and
# say so out loud, because a gate that always times out is a fixed delay
# wearing a costume and it will read as a clean result if nobody is counting.
_stats_lock = threading.Lock()
_stats_every = 200
_waits = 0
_timeouts = 0
_wait_ms = 0.0
_max_wait_ms = 0.0


def mark_dispatching() -> None:
    """The host has started packaging a step. Called at the top of the step."""
    _gap.clear()


def mark_dispatched() -> None:
    """The step is on the chip, so the host is free until it comes back."""
    _gap.set()


def wait_for_gap(timeout_s: float) -> bool:
    """Blocks until the host is out of the dispatch window.

    Returns True if a gap was reached and False if the timeout won, which the
    caller should treat as permission to proceed anyway. The timeout is a
    backstop against a runner that stops stepping, not a tuning knob: a submit
    that waits forever is a transfer that never happens.
    """
    started = time.perf_counter()
    reached = _gap.wait(timeout_s)
    waited_ms = (time.perf_counter() - started) * 1000.0

    global _waits, _timeouts, _wait_ms, _max_wait_ms
    with _stats_lock:
        _waits += 1
        _timeouts += 0 if reached else 1
        _wait_ms += waited_ms
        _max_wait_ms = max(_max_wait_ms, waited_ms)
        if _waits < _stats_every:
            return reached
        waits, timeouts, total_ms, peak_ms = (_waits, _timeouts, _wait_ms,
                                              _max_wait_ms)
        _waits = _timeouts = 0
        _wait_ms = _max_wait_ms = 0.0

    logger.info(
        '{"event": "kv_gate_wait", "waits": %d, "timeouts": %d, '
        '"mean_ms": %.3f, "max_ms": %.3f}', waits, timeouts,
        total_ms / waits, peak_ms)
    return reached
