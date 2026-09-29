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
"""Release router for data-parallel prompt admission (DP_PLACEMENT=release).

vLLM's DP load balancer places each request on arrival, so the engines'
prefill steps drift apart. The release router holds arrivals in the API
server and releases them as a group, one per engine where it can, so the
engines run their prefill steps together.
"""

import asyncio
import time
from collections import Counter

import msgspec
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Release when the oldest held request has waited this long.
DEADLINE_S = 5.0
# Release when no request has arrived for this long.
QUIET_S = 2.0
_POLL_S = 0.005


def install_release(
    client_cls,
    threshold: int = 0,
    deadline_s: float = DEADLINE_S,
    quiet_s: float = QUIET_S,
) -> None:
    """Install the release rules on vLLM's DP load-balancing client class.

    Held requests are released together when `threshold` are pending (the
    engine count when 0), when an engine is idle, when the oldest has waited
    `deadline_s`, or when none has arrived for `quiet_s`. Each released
    request goes to the engine with the fewest assignments in this release,
    then the fewest in flight, ties cyclic; a request pinned to a DP rank
    keeps its engine.
    """
    if client_cls.__dict__.get("_tpu_dp_release_patch", False):
        return
    orig_abort = client_cls.abort_requests_async
    orig_shutdown = client_cls.shutdown

    def _buf(self):
        return self.__dict__.setdefault("_release_buf", [])

    def _any_idle(self):
        if not self.engines_running:
            return True
        return any(c[0] == 0 and c[1] == 0 for c in self.lb_engines)

    def _settle(fut, exc=None):
        if fut.done():
            return
        if exc is None:
            fut.set_result(None)
        else:
            fut.set_exception(exc)

    def _fail_held(self, exc):
        buf = _buf(self)
        for _, _, fut in buf:
            _settle(fut, exc)
        buf.clear()

    async def _send_one(self, request, idx):
        chosen = self.core_engines[idx]
        request.current_wave = self.current_wave
        request.client_index = self.client_index
        self.reqs_in_flight[request.request_id] = chosen
        self.engine_inflight[chosen] += 1
        self.lb_engines[idx][0] += self.client_count
        try:
            to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen)
            if not self.engines_running:
                await self.first_req_send_socket.send(
                    msgspec.msgpack.encode(("FIRST_REQ", chosen))
                )
            await to_await
        except BaseException:
            # The counts must not claim a request no engine holds.
            self.reqs_in_flight.pop(request.request_id, None)
            self.engine_inflight[chosen] -= 1
            self.lb_engines[idx][0] -= self.client_count
            raise
        self._ensure_output_queue_task()

    async def _release_all(self):
        # Runs in its own task, so a cancelled add_request_async cancels no
        # other request's send.
        buf = _buf(self)
        num = len(self.core_engines)
        assigned = Counter()
        start = self.__dict__.get("_rr_next", 0) % num
        while buf:
            # Popped before the send: an abort during the send reaches the
            # engine through the send's registration instead.
            _, request, fut = buf.pop(0)
            if fut.done():
                continue
            if request.data_parallel_rank is not None:
                idx = request.data_parallel_rank
            else:
                order = [(i + start) % num for i in range(num)]
                idx = min(
                    order,
                    key=lambda i: (
                        assigned[i],
                        self.engine_inflight[self.core_engines[i]],
                    ),
                )
                assigned[idx] += 1
                start = (idx + 1) % num
            try:
                await _send_one(self, request, idx)
            except asyncio.CancelledError:
                fut.cancel()
                raise
            except Exception as exc:
                logger.exception(
                    "DP release router: request %s could not be sent to engine %s",
                    request.request_id,
                    self.core_engines[idx],
                )
                _settle(fut, exc)
                continue
            _settle(fut)
        self._rr_next = start

    def _job_done(job):
        if not job.cancelled() and job.exception() is not None:
            logger.error(
                "DP release router: release task failed", exc_info=job.exception()
            )

    def _start_release(self):
        job = self.__dict__.get("_release_job")
        if job is None or job.done():
            self._release_job = asyncio.get_running_loop().create_task(
                _release_all(self)
            )
            self._release_job.add_done_callback(_job_done)

    def _maybe_release(self):
        buf = _buf(self)
        if not buf:
            return
        if self.resources.engine_dead:
            _fail_held(self, EngineDeadError())
            return
        now = time.monotonic()
        if (
            len(buf) >= (threshold or len(self.core_engines))
            or now - buf[0][0] >= deadline_s
            or now - buf[-1][0] >= quiet_s
            or _any_idle(self)
        ):
            _start_release(self)

    async def _poll(self):
        while True:
            await asyncio.sleep(_POLL_S)
            try:
                _maybe_release(self)
            except Exception:
                logger.exception("DP release router: release poller")

    async def add_request_async(self, request):
        self._ensure_stats_update_task()
        loop = asyncio.get_running_loop()
        if self.__dict__.get("_release_task") is None:
            self._release_task = loop.create_task(_poll(self))
        fut = loop.create_future()
        _buf(self).append((time.monotonic(), request, fut))
        _maybe_release(self)
        # Returns once this request has been sent, or raises its send's error.
        await fut

    async def abort_requests_async(self, request_ids):
        ids = set(request_ids)
        buf = _buf(self)
        kept = []
        for entry in buf:
            if entry[1].request_id in ids:
                entry[2].cancel()
            else:
                kept.append(entry)
        # In place: a release in progress holds this same list.
        buf[:] = kept
        await orig_abort(self, request_ids)

    def shutdown(self, *args, **kwargs):
        try:
            for name in ("_release_task", "_release_job"):
                task = self.__dict__.get(name)
                if task is not None:
                    task.cancel()
            _fail_held(self, EngineDeadError())
        except RuntimeError:
            pass  # The event loop is closed: nobody is waiting.
        return orig_shutdown(self, *args, **kwargs)

    client_cls.add_request_async = add_request_async
    client_cls.abort_requests_async = abort_requests_async
    client_cls.shutdown = shutdown
    client_cls._tpu_dp_release_patch = True
    logger.info(
        "DP release router: threshold %s, deadline %.1f s, quiet %.1f s",
        threshold or "engine count",
        deadline_s,
        quiet_s,
    )


def install_if_enabled() -> None:
    """Install the release router when DP_PLACEMENT=release."""
    if envs.DP_PLACEMENT is None:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    install_release(DPLBAsyncMPClient)
