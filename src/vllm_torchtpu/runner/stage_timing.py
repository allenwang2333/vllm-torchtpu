# SPDX-License-Identifier: Apache-2.0
"""Runner stage timings without synchronizing the serving thread."""

from __future__ import annotations

import queue
import threading
import time

from torch_tpu._internal.stream.streams import TpuEvent

_allocate_lock = threading.Lock
_perf_counter = time.perf_counter


class _PendingTiming:
    __slots__ = ("_gate", "end", "end_recorded", "stage", "start")

    def __init__(self, stage: str, start: TpuEvent, end: TpuEvent) -> None:
        self.stage = stage
        self.start = start
        self.end = end
        gate = _allocate_lock()
        gate.acquire()
        self._gate = gate
        self.end_recorded = self

    def set(self) -> None:
        self._gate.release()

    def wait(self) -> None:
        self._gate.acquire()


class RunnerStageTiming:
    MAX_IN_FLIGHT_DEVICE_TIMINGS = 1024
    __slots__ = (
        "_cpu_samples",
        "_device_samples",
        "_host_ctxs",
        "_pending",
        "_rank",
        "_samples_lock",
        "_slots",
        "_stage_ctxs",
    )

    def __init__(self, rank: int):
        self._rank = str(rank)
        self._cpu_samples: list[tuple[str, str, float]] = []
        self._device_samples: list[tuple[str, str, float]] = []
        self._host_ctxs: dict[str, _HostStageContext] = {}
        self._stage_ctxs: dict[str, _StageContext] = {}
        # Check Event support before serving; synchronization stays off the
        # serving thread after startup.
        event = TpuEvent()
        event.record()
        event.synchronize()
        self._pending: queue.Queue[_PendingTiming] = queue.Queue()
        # Bound in-flight device timings; a full collector keeps CPU samples only.
        self._slots = threading.BoundedSemaphore(self.MAX_IN_FLIGHT_DEVICE_TIMINGS)
        self._samples_lock = _allocate_lock()
        threading.Thread(target=self._collect_device_time, daemon=True).start()

    def _collect_device_time(self) -> None:
        perf_counter = _perf_counter
        pending_get = self._pending.get
        samples_lock = self._samples_lock
        release_slot = self._slots.release
        rank = self._rank
        while True:
            pending = pending_get()
            pending.start.synchronize()
            start_time = perf_counter()
            pending.end_recorded.wait()
            pending.end.synchronize()
            seconds = perf_counter() - start_time
            with samples_lock:
                self._device_samples.append((rank, pending.stage, seconds))
            release_slot()

    def host_stage(self, stage: str) -> _HostStageContext:
        ctx = self._host_ctxs.get(stage)
        if ctx is None:
            ctx = _HostStageContext(self, self._rank, stage)
            self._host_ctxs[stage] = ctx
        return ctx

    def stage(self, stage: str) -> _StageContext:
        ctx = self._stage_ctxs.get(stage)
        # Close the queued timing even when the stage body raises.
        if ctx is None:
            ctx = _StageContext(self, self._rank, stage)
            self._stage_ctxs[stage] = ctx
        return ctx

    def take_samples(
        self,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        cpu, self._cpu_samples = self._cpu_samples, []
        with self._samples_lock:
            device, self._device_samples = self._device_samples, []
        return cpu, device


class _HostStageContext:
    __slots__ = ("_cpu_start", "_owner", "_rank", "_stage")

    def __init__(self, owner: RunnerStageTiming, rank: str, stage: str) -> None:
        self._owner = owner
        self._rank = rank
        self._stage = stage
        self._cpu_start = 0.0

    def __enter__(self) -> None:
        self._cpu_start = _perf_counter()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._owner._cpu_samples.append(
            (self._rank, self._stage, _perf_counter() - self._cpu_start)
        )


class _StageContext:
    __slots__ = ("_cpu_start", "_end", "_owner", "_pending", "_rank", "_stage")

    def __init__(self, owner: RunnerStageTiming, rank: str, stage: str) -> None:
        self._owner = owner
        self._rank = rank
        self._stage = stage
        self._cpu_start = 0.0
        self._end: TpuEvent | None = None
        self._pending: _PendingTiming | None = None

    def __enter__(self) -> None:
        owner = self._owner
        if not owner._slots.acquire(False):
            self._pending = None
            self._cpu_start = _perf_counter()
            return
        event_cls = TpuEvent
        start = event_cls()
        end = event_cls()
        start.record()
        pending = _PendingTiming(self._stage, start, end)
        owner._pending.put_nowait(pending)
        self._end = end
        self._pending = pending
        self._cpu_start = _perf_counter()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        cpu_elapsed = _perf_counter() - self._cpu_start
        pending = self._pending
        if pending is not None:
            self._end.record()  # type: ignore[union-attr]
            pending.set()
            self._pending = None
            self._end = None
        self._owner._cpu_samples.append((self._rank, self._stage, cpu_elapsed))
