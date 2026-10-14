# SPDX-License-Identifier: Apache-2.0
"""Per-step host timestamps for the serving loop (TPU_HOST_LOOP_TRACE_DIR).

Each traced process appends one JSON line per step to
``<dir>/<role>_<pid>.jsonl``: ``perf_counter_ns`` marks (one clock for every
process on a host, so the EngineCore, worker and output-thread files join on
time) plus a few counters. It answers where the host spends the time between a
step's result and the next step's forward, without a profiler's overhead.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm_torchtpu import envs

_FLUSH_EVERY = 256


class HostLoopTrace:
    def __init__(self, directory: str, role: str):
        os.makedirs(directory, exist_ok=True)
        self._path = os.path.join(directory, f"{role}_{os.getpid()}.jsonl")
        self._rec: dict[str, Any] = {}
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def mark(self, name: str) -> None:
        """Timestamp the first occurrence of `name` in the current step."""
        self._rec.setdefault(name, time.perf_counter_ns())

    def mark_last(self, name: str) -> None:
        self._rec[name] = time.perf_counter_ns()

    def note(self, name: str, value: Any) -> None:
        self._rec[name] = value

    def add_ns(self, name: str, start_ns: int) -> None:
        """Accumulate the time since `start_ns` under `name` (repeated spans)."""
        self._rec[name] = self._rec.get(name, 0) + time.perf_counter_ns() - start_ns

    def end_step(self) -> None:
        if not self._rec:
            return
        rec, self._rec = self._rec, {}
        with self._lock:
            self._lines.append(json.dumps(rec, separators=(",", ":")))
            if len(self._lines) >= _FLUSH_EVERY:
                self._flush_locked()

    def _flush_locked(self) -> None:
        with open(self._path, "a") as f:
            f.write("\n".join(self._lines) + "\n")
        self._lines.clear()


def make_trace(role: str) -> HostLoopTrace | None:
    directory = envs.TPU_HOST_LOOP_TRACE_DIR
    return HostLoopTrace(directory, role) if directory else None


def _timed(trace: HostLoopTrace, name: str, fn):
    def wrapper(*args, **kwargs):
        trace.mark_last(name + "_s")
        try:
            return fn(*args, **kwargs)
        finally:
            trace.mark_last(name + "_e")

    return wrapper


def install_engine_core_trace() -> None:
    """EngineCore: time schedule, dispatch, result wait and output update."""
    if not envs.TPU_HOST_LOOP_TRACE_DIR:
        return
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    if getattr(EngineCore, "_tpu_host_loop_trace", False):
        return
    original_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trace = make_trace("engine")
        assert trace is not None
        sched = self.scheduler
        sched.schedule = _timed(trace, "sched", sched.schedule)
        sched.update_from_output = _timed(trace, "upd", sched.update_from_output)
        ex = self.model_executor
        ex.execute_model = _timed(trace, "exec", ex.execute_model)
        ex.sample_tokens = _timed(trace, "samp", ex.sample_tokens)
        if isinstance(self, EngineCoreProc):
            self._process_input_queue = _timed(trace, "inq", self._process_input_queue)
        step_fn = self.step_fn

        def traced_step():
            trace.mark_last("step_s")
            try:
                outputs, executed = step_fn()
            finally:
                trace.mark_last("step_e")
            trace.note("ran", int(bool(executed)))
            trace.note("out", int(outputs is not None and len(outputs) > 0))
            trace.end_step()
            return outputs, executed

        self.step_fn = traced_step

    EngineCore.__init__ = __init__
    EngineCore._tpu_host_loop_trace = True


def install_worker_output_trace() -> None:
    """Worker output thread: time the result copy wait and the reply send."""
    if not envs.TPU_HOST_LOOP_TRACE_DIR:
        return
    from vllm.v1.executor.multiproc_executor import WorkerProc

    if getattr(WorkerProc, "_tpu_host_loop_trace", False):
        return
    original = WorkerProc.enqueue_output
    traces: dict[int, HostLoopTrace] = {}

    def enqueue_output(self, output):
        trace = traces.get(id(self))
        if trace is None:
            trace = traces.setdefault(id(self), make_trace("wout"))
        trace.mark_last("out_s")
        try:
            return original(self, output)
        finally:
            trace.mark_last("out_e")
            trace.note("kind", type(output).__name__)
            trace.end_step()

    WorkerProc.enqueue_output = enqueue_output
    WorkerProc._tpu_host_loop_trace = True
