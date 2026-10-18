# SPDX-License-Identifier: Apache-2.0
"""Per-worker host CPU scheduling diagnostics (TPU_HOST_SCHED_DIAG_STEPS).

Every N runner steps, log one line with the deltas since the previous line:
the serving thread's CPU time and run-queue wait (/proc/thread-self/schedstat:
time spent runnable but not running is CPU starvation), the same summed over
every thread of the process plus its busiest threads, and the container's CFS
throttling counters (cgroup cpu.stat). Off (N = 0) by default; reading /proc
costs tens of microseconds per line, nothing on the other steps.
"""

from __future__ import annotations

import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_CGROUP_STATS = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat")

# Forward-call host anatomy (ns, calls), filled by the wrappers that
# install_fwd_timers() puts around the compiled-forward call chain:
#   aot    AOTCompiledFunction.__call__ (guard check + Dynamo pre-graph bytecode
#          + graph call + post-graph bytecode)
#   guard  AOTCompiledFunction.guard_check
#   graph  VllmSerializableFunction.__call__ (AOTAutograd runtime wrapper +
#          compiled executable + input-mutation epilogue)
#   exe    TorchTpuCompiledExecutable.__call__ (PJRT execute incl. inputs)
FWD_TIMERS: dict[str, list[int]] = {}


def _timed(name: str, fn):
    import functools

    acc = FWD_TIMERS.setdefault(name, [0, 0])
    perf = time.perf_counter_ns

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = perf()
        try:
            return fn(*args, **kwargs)
        finally:
            acc[0] += perf() - t0
            acc[1] += 1

    wrapper._h5_timed = True  # type: ignore[attr-defined]
    return wrapper


def install_fwd_timers() -> list[str]:
    """Wrap the compiled-forward call chain with timers; returns what was
    wrapped. Idempotent. Diagnostic only (TPU_HOST_SCHED_DIAG_STEPS > 0)."""
    targets = []
    try:
        from torch._dynamo.aot_compile import AOTCompiledFunction

        targets += [(AOTCompiledFunction, "__call__", "aot"),
                    (AOTCompiledFunction, "guard_check", "guard")]
    except Exception:
        pass
    try:
        from vllm.compilation.caching import VllmSerializableFunction

        targets.append((VllmSerializableFunction, "__call__", "graph"))
    except Exception:
        pass
    try:
        from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
            TorchTpuCompiledExecutable,
        )

        targets.append((TorchTpuCompiledExecutable, "__call__", "exe"))
    except Exception:
        pass
    done = []
    for cls, attr, name in targets:
        fn = cls.__dict__.get(attr)
        if fn is None or getattr(fn, "_h5_timed", False):
            continue
        setattr(cls, attr, _timed(name, fn))
        done.append(name)
    return done


def _thread_cpu_ctx() -> tuple[int, int]:
    """(CPU the calling thread last ran on, its involuntary context switches)."""
    with open("/proc/thread-self/stat") as f:
        cpu = int(f.read().rsplit(")", 1)[1].split()[36])
    nvcsw = 0
    with open("/proc/thread-self/status") as f:
        for line in f:
            if line.startswith("nonvoluntary_ctxt_switches"):
                nvcsw = int(line.split()[1])
    return cpu, nvcsw


def _schedstat(path: str) -> tuple[int, int]:
    """(ns on CPU, ns runnable but waiting) from a schedstat file."""
    with open(path) as f:
        run, wait, _ = f.read().split()
    return int(run), int(wait)


def _threads() -> dict[int, tuple[str, int, int]]:
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            run, wait = _schedstat(f"/proc/self/task/{tid}/schedstat")
            with open(f"/proc/self/task/{tid}/comm") as f:
                name = f.read().strip()
        except OSError:
            continue  # the thread exited
        out[int(tid)] = (name, run, wait)
    return out


def _cgroup() -> dict[str, int]:
    for path in _CGROUP_STATS:
        try:
            with open(path) as f:
                return {k: int(v) for k, v in (line.split() for line in f if line.strip())}
        except (OSError, ValueError):
            continue
    return {}


class HostSchedDiag:
    def __init__(self, rank: int, every: int, fwd_timers: bool = True):
        self.rank = rank
        self.every = every
        self.steps = 0
        self._snap = self._take() if every > 0 else None
        self._fwd0: dict[str, tuple[int, int]] = {}
        self._cpus_seen: set[int] = set()
        self._nvcsw0 = 0
        if every > 0 and fwd_timers:
            logger.warning("HOSTDIAG rank=%d fwd timers: %s", rank, install_fwd_timers())

    def _take(self):
        return (time.monotonic_ns(), _schedstat("/proc/thread-self/schedstat"), _threads(), _cgroup())

    def tick(self, suffix: str = "") -> str | None:
        """Count one runner step; every `every` steps log and return the line."""
        if self._snap is None:
            return None
        self.steps += 1
        if self.steps % self.every:
            return None
        try:
            snap = self._take()
        except OSError:
            self._snap = None  # no /proc: stop quietly
            return None
        (t0, (mr0, mw0), th0, cg0), (t1, (mr1, mw1), th1, cg1) = self._snap, snap
        self._snap = snap
        n = self.every
        deltas = []
        for tid, (name, run, wait) in th1.items():
            r0, w0 = th0.get(tid, (name, 0, 0))[1:]
            deltas.append((run - r0, wait - w0, name, tid))
        prun = sum(d[0] for d in deltas)
        pwait = sum(d[1] for d in deltas)
        top = sorted(deltas, reverse=True)[:4]
        thr = cg1.get("nr_throttled", 0) - cg0.get("nr_throttled", 0)
        per = cg1.get("nr_periods", 0) - cg0.get("nr_periods", 0)
        thr_us = (cg1.get("throttled_usec", cg1.get("throttled_time", 0) // 1000)
                  - cg0.get("throttled_usec", cg0.get("throttled_time", 0) // 1000))
        line = (
            f"HOSTDIAG rank={self.rank} steps={n} wall_ms/step={(t1 - t0) / 1e6 / n:.3f} "
            f"main_run_ms/step={(mr1 - mr0) / 1e6 / n:.3f} main_wait_ms/step={(mw1 - mw0) / 1e6 / n:.3f} "
            f"proc_run_ms/step={prun / 1e6 / n:.3f} proc_wait_ms/step={pwait / 1e6 / n:.3f} "
            f"threads={len(th1)} cg_throttled={thr}/{per} cg_throttled_ms={thr_us / 1e3:.1f} "
            f"cpus={len(os.sched_getaffinity(0))} top="
            + ",".join(f"{name}:{run / 1e6 / n:.2f}/{wait / 1e6 / n:.2f}" for run, wait, name, _ in top)
        )
        fwd = {k: (v[0], v[1]) for k, v in FWD_TIMERS.items()}
        parts = []
        for k in ("aot", "guard", "graph", "exe"):
            if k in fwd:
                ns0, c0 = self._fwd0.get(k, (0, 0))
                parts.append(f"{k}:{(fwd[k][0] - ns0) / 1e6 / n:.3f}/{(fwd[k][1] - c0) / n:.2f}")
        self._fwd0 = fwd
        try:
            cpu, nvcsw = _thread_cpu_ctx()
        except (OSError, ValueError, IndexError):
            cpu, nvcsw = -1, self._nvcsw0
        line += (
            f" fwd_ms/step(calls)={','.join(parts) or '-'} cpu={cpu}"
            f" nvcsw/step={(nvcsw - self._nvcsw0) / n:.2f}"
        )
        self._nvcsw0 = nvcsw
        line += suffix
        logger.warning(line)
        return line
