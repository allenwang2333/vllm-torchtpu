#!/usr/bin/env python3
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
"""Per-call device-time measurement for standalone JAX/Pallas microbenchmarks.

``device_time`` is the number to trust: it traces ``num_iters`` calls with
``jax.profiler``, reads the ``/device:TPU:0`` xplane, and returns the *median*
per-call device duration of the XLA module. That excludes host dispatch,
Python overhead, and the gaps between calls, so it stays meaningful for kernels
whose device time is smaller than one dispatch.

Ported from tpu-inference's
``scripts/benchmarking/kernels/collective_bench_lib.py`` (``device_time`` and
its xplane helpers). The barrier-cores subtraction from that version is
dropped: it cancels the cross-core wait of an in-kernel collective, and nothing
benchmarked here contains one. Only single-device kernel bodies are shared
between the two repos; nothing here touches a mesh or a collective.

``wall_time`` is a host-clock fallback (median of blocking wall-clock samples)
for hosts where the profiler is unavailable. It measures dispatch + device +
sync, so it is an upper bound and is *not* comparable to a device number.
``measure`` prefers ``device_time`` and reports which one produced the value;
every caller is expected to print that label next to the number.
"""

import argparse
import dataclasses
import glob
import os
import shutil
import statistics
import tempfile
import time
from typing import Any, Callable

import jax

# Device plane and per-module timeline line written by the TPU profiler.
_TPU_PLANE_NAME = "/device:TPU:0"
_MODULE_LINE_NAME = "XLA Modules"

DEVICE = "device"
WALL = "wall"
PIPELINED = "pipe"

# How many back-to-back bursts pipelined_time averages over. Three is enough
# to reject a single scheduling hiccup without tripling every sweep.
_PIPELINED_BURSTS = 3


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One timing result and how it was obtained.

    A dataclass (not a tuple) so callers can read the number by field name or
    through ``dataclasses.asdict``; ``us`` is the median per call.
    """

    label: str
    us: float
    method: str  # DEVICE or WALL

    def __str__(self) -> str:
        return f"{self.label}: {self.us:.2f} us [{self.method}]"


def _event_duration_ps(event) -> float:
    for name, value in event.stats:
        if name == "device_duration_ps":
            return value
    return 0.0


def _module_durations_ps(plane) -> list[float]:
    """Per-execution XLA-module device durations (ps), in start-time order."""
    events = [(e.start_ns, _event_duration_ps(e)) for line in plane.lines
              if line.name == _MODULE_LINE_NAME for e in line.events]
    return [duration for _, duration in sorted(events)]


def _line_names(plane) -> list[str]:
    return sorted({line.name for line in plane.lines})


def _call_n_times(fn: Callable[..., Any], args: tuple, num_iters: int) -> None:
    """Dispatch ``num_iters`` calls, then block on the last output only.

    Blocking once at the end keeps the calls pipelined the way a serving step
    issues them; every call still shows up as its own module execution in the
    trace, which is what the median is taken over.
    """
    out = None
    for _ in range(num_iters):
        out = fn(*args)
    jax.block_until_ready(out)


def _warmup(fn: Callable[..., Any], args: tuple, warmup: int) -> None:
    """Compile and settle before tracing (blocking, so nothing leaks in)."""
    for _ in range(max(warmup, 1)):
        jax.block_until_ready(fn(*args))


def device_time(
    fn: Callable[..., Any],
    *args: Any,
    num_iters: int = 20,
    warmup: int = 3,
    trace_dir: str | None = None,
) -> float:
    """Median per-call device time of ``fn(*args)``, in microseconds.

    Nothing is donated, so ``args`` stay valid across calls and can be reused
    by the caller afterwards. When ``trace_dir`` is given the xprof trace is
    kept there for inspection; otherwise it is written to a temp dir that is
    removed before returning.

    Raises RuntimeError when the profile has no usable device time (no trace
    file, no TPU plane, no module line, or a module count that is not a whole
    multiple of ``num_iters``).
    """
    _warmup(fn, args, warmup)
    keep = trace_dir is not None
    path = trace_dir or tempfile.mkdtemp(prefix="device_time-")
    if keep:
        os.makedirs(path, exist_ok=True)
    try:
        with jax.profiler.trace(path):
            _call_n_times(fn, args, num_iters)

        traces = glob.glob(os.path.join(path, "**", "*.xplane.pb"),
                           recursive=True)
        if not traces:
            raise RuntimeError(f"profiler wrote no xplane trace under {path}")
        # Newest, so a reused --trace-dir reports this run and not an old one.
        newest = max(traces, key=os.path.getmtime)
        plane = jax.profiler.ProfileData.from_file(
            newest).find_plane_with_name(_TPU_PLANE_NAME)
        if plane is None:
            raise RuntimeError(
                f"no {_TPU_PLANE_NAME} plane in {newest}; is this a TPU host?")

        durations = _module_durations_ps(plane)
        if not durations:
            raise RuntimeError(
                f"no {_MODULE_LINE_NAME!r} device time in {newest}; "
                f"lines present: {_line_names(plane)}")
        if len(durations) % num_iters != 0:
            raise RuntimeError(
                f"{len(durations)} module executions is not a multiple of "
                f"{num_iters} calls; the traced function is not deterministic "
                "in its module count")
        # A single call can lower to more than one module (e.g. a Pallas
        # kernel plus a reshape program). Sum each call's modules first, then
        # take the median over calls, so the result is still "time per call".
        per_call = len(durations) // num_iters
        calls = [
            sum(durations[i * per_call:(i + 1) * per_call])
            for i in range(num_iters)
        ]
        median = statistics.median(calls)
        if median <= 0:
            raise RuntimeError("non-positive per-call device time")
        return median / 1e6  # ps -> us
    finally:
        if not keep:
            shutil.rmtree(path, ignore_errors=True)


def wall_time(
    fn: Callable[..., Any],
    *args: Any,
    num_iters: int = 20,
    warmup: int = 3,
) -> float:
    """Median host wall-clock time per blocking call, in microseconds.

    Fallback only: this includes host dispatch and the block-until-ready sync,
    so it overstates small kernels. Callers must label it as such.
    """
    _warmup(fn, args, warmup)
    samples = []
    for _ in range(num_iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def pipelined_time(
    fn: Callable[..., Any],
    *args: Any,
    num_iters: int = 20,
    warmup: int = 3,
) -> float:
    """Median per-call time of a pipelined burst of ``num_iters`` calls, in us.

    Dispatches every call before blocking once at the end, so JAX's async
    dispatch overlaps the host cost of call ``i+1`` with the device execution
    of call ``i``. For any kernel slower than a dispatch (tens of us) this
    converges on device time; ``wall_time``, which blocks per call, instead
    adds a full host round trip (~150 us on this host) to every measurement
    and so reports a floor rather than the kernel.

    Repeated over several bursts, median of the bursts, so one scheduling
    hiccup does not set the number.
    """
    _warmup(fn, args, warmup)
    bursts = []
    for _ in range(max(_PIPELINED_BURSTS, 1)):
        start = time.perf_counter()
        _call_n_times(fn, args, num_iters)
        bursts.append((time.perf_counter() - start) * 1e6 / num_iters)
    return statistics.median(bursts)


def measure(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    num_iters: int = 20,
    warmup: int = 3,
    trace_dir: str | None = None,
    prefer_device: bool = False,
) -> Measurement:
    """Time ``fn(*args)``, preferring device time and reporting which was used.

    ``prefer_device`` is opt-in, not the default: on a host whose jaxlib and
    libtpu disagree on the profiler plugin ABI, ``jax.profiler.start_trace``
    does not raise, it SIGSEGVs the interpreter ("Unexpected
    PLUGIN_Profiler_Api size: expected 80, got 104"), which no ``except`` can
    catch. ``pipelined_time`` needs no profiler and is accurate for anything
    slower than a dispatch, so it is the default clock. A failure in ``fn`` is
    not caught here and propagates to the caller.
    """
    if prefer_device:
        try:
            micros = device_time(fn,
                                 *args,
                                 num_iters=num_iters,
                                 warmup=warmup,
                                 trace_dir=trace_dir)
            return Measurement(label, micros, DEVICE)
        except Exception as err:  # profiler crashes are not all RuntimeError
            print(f"  [warn] device timing unavailable for {label}: {err}; "
                  "falling back to pipelined host clock")
    micros = pipelined_time(fn, *args, num_iters=num_iters, warmup=warmup)
    return Measurement(label, micros, PIPELINED)


def bench(
    label: str | Callable[..., Any],
    fn: Callable[..., Any] | None = None,
    *args: Any,
    num_iters: int = 20,
    warmup: int = 3,
    trace_dir: str | None = None,
    prefer_device: bool = False,
    iters: int | None = None,
    n: int | None = None,
) -> Measurement:
    """``measure`` plus a formatted one-line report. Returns the measurement.

    Called either as ``bench("label", fn, *args)`` or as ``bench(fn, *args)``,
    where the label then defaults to the callable's name; ``iters``/``n`` are
    accepted as aliases for ``num_iters`` so a caller can pass whichever name
    it already has.
    """
    if callable(label):
        if fn is not None:
            args = (fn, ) + args
        fn = label
        label = getattr(fn, "__name__", None) or type(fn).__name__
    if fn is None:
        raise TypeError("bench() requires a callable to time")
    result = measure(label,
                     fn,
                     *args,
                     num_iters=iters or n or num_iters,
                     warmup=warmup,
                     trace_dir=trace_dir,
                     prefer_device=prefer_device)
    print(f"{result.label:<44} {result.us:>12.2f} us  [{result.method}]")
    return result


def _self_test(args: argparse.Namespace) -> None:
    """Square matmul with a known FLOP count: proves the timer is wired up."""
    import jax.numpy as jnp

    dtype = jnp.dtype(args.dtype)
    n = args.n
    key = jax.random.key(0)
    lhs = jax.random.normal(key, (n, n), dtype=dtype)
    rhs = jax.random.normal(jax.random.key(1), (n, n), dtype=dtype)
    matmul = jax.jit(jnp.matmul)

    print(f"device={jax.devices()[0].device_kind}, {n}x{n}x{n} "
          f"{dtype.name} matmul, iters={args.iters}, warmup={args.warmup}")
    dev = bench("device_time",
                matmul,
                lhs,
                rhs,
                num_iters=args.iters,
                warmup=args.warmup,
                trace_dir=args.trace_dir)
    wall = bench("wall_time",
                 matmul,
                 lhs,
                 rhs,
                 num_iters=args.iters,
                 warmup=args.warmup,
                 prefer_device=False)
    flops = 2.0 * n * n * n
    print(f"TFLOP/s ({dev.method}): "
          f"{flops / (dev.us * 1e-6) / 1e12:.2f}")
    if dev.method == DEVICE:
        print(f"host overhead:  {wall.us - dev.us:.2f} us/call")
    else:
        print("host overhead:  n/a (device timing unavailable, both rows are "
              "the host clock)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-test for the device-time harness: times a square "
        "matmul both ways and prints the host-dispatch gap.")
    parser.add_argument("--n", type=int, default=4096, help="matmul size")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-dir",
                        default=None,
                        help="keep the xprof trace here instead of a temp dir")
    _self_test(parser.parse_args())


if __name__ == "__main__":
    main()
