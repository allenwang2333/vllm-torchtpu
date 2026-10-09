# SPDX-License-Identifier: Apache-2.0
"""Runner stage timings without synchronizing the serving thread."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from torch_tpu._internal.stream.streams import TpuEvent


@dataclass(slots=True)
class _PendingTiming:
    stage: str
    start: TpuEvent
    end: TpuEvent
    end_recorded: threading.Event = field(default_factory=threading.Event)


class RunnerStageTiming:
    MAX_IN_FLIGHT_DEVICE_TIMINGS = 1024
    __slots__ = (
        "_cpu_samples",
        "_device_samples",
        "_pending",
        "_rank",
        "_samples_lock",
        "_slots",
    )

    def __init__(self, rank: int):
        self._rank = str(rank)
        self._cpu_samples: list[tuple[str, str, float]] = []
        self._device_samples: list[tuple[str, str, float]] = []
        # Check Event support before serving; synchronization stays off the
        # serving thread after startup.
        event = TpuEvent()
        event.record()
        event.synchronize()
        self._pending: queue.Queue[_PendingTiming] = queue.Queue()
        # Bound in-flight device timings; a full collector keeps CPU samples only.
        self._slots = threading.BoundedSemaphore(self.MAX_IN_FLIGHT_DEVICE_TIMINGS)
        self._samples_lock = threading.Lock()
        threading.Thread(target=self._collect_device_time, daemon=True).start()

    def _collect_device_time(self) -> None:
        perf_counter = time.perf_counter
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

    @contextlib.contextmanager
    def host_stage(self, stage: str) -> Iterator[None]:
        perf_counter = time.perf_counter
        cpu_start = perf_counter()
        yield
        self._cpu_samples.append((self._rank, stage, perf_counter() - cpu_start))

    @contextlib.contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        if not self._slots.acquire(blocking=False):
            with self.host_stage(stage):
                yield
            return

        perf_counter = time.perf_counter
        start = TpuEvent()
        end = TpuEvent()
        start.record()
        pending = _PendingTiming(stage, start, end)
        self._pending.put_nowait(pending)
        cpu_start = perf_counter()
        # Close the queued timing even when the stage body raises.
        try:
            yield
        finally:
            cpu_elapsed = perf_counter() - cpu_start
            end.record()
            pending.end_recorded.set()
            self._cpu_samples.append((self._rank, stage, cpu_elapsed))

    def take_samples(
        self,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        cpu, self._cpu_samples = self._cpu_samples, []
        with self._samples_lock:
            device, self._device_samples = self._device_samples, []
        return cpu, device
