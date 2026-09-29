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
import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_torchtpu.core import dp_placement

pytestmark = pytest.mark.cpu_test


def _client_cls(n=4, counts=None, fail_engine=None):
    """A stand-in DP client with what the release router touches."""

    class Sock:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

    class C:
        def __init__(self):
            self.core_engines = [f"e{i}" for i in range(n)]
            self.reqs_in_flight = {}
            self.engine_inflight = Counter({f"e{i}": 8 for i in range(n)})
            self.engine_inflight["e2"] = 7
            self.lb_engines = (
                counts if counts is not None else [[1, 8, 0.5] for _ in range(n)]
            )
            self.client_count = 1
            self.engines_running = True
            self.current_wave = 0
            self.client_index = 0
            self.first_req_send_socket = Sock()
            self.resources = SimpleNamespace(engine_dead=False)
            self.sent = []

        def _ensure_stats_update_task(self):
            pass

        def _ensure_output_queue_task(self):
            pass

        def _send_input(self, kind, request, engine):
            if engine == fail_engine:
                raise RuntimeError(f"{engine} is down")
            self.sent.append((request.request_id, engine))

            async def done():
                pass

            return done()

        async def abort_requests_async(self, ids):
            self.aborted = list(ids)

        def shutdown(self, timeout=None):
            pass

    return C


class R:
    def __init__(self, i, rank=None):
        self.request_id = str(i)
        self.current_wave = None
        self.client_index = None
        self.data_parallel_rank = rank


def _add(c, r):
    """An add returns once its request is sent, so held adds run as tasks."""
    return asyncio.get_running_loop().create_task(c.add_request_async(r))


def _install(C, threshold=4, deadline_s=30.0, quiet_s=30.0):
    dp_placement.install_release(
        C, threshold=threshold, deadline_s=deadline_s, quiet_s=quiet_s
    )


def test_holds_until_threshold_then_one_per_engine():
    C = _client_cls()
    _install(C)
    c = C()

    async def go():
        held = [_add(c, R(i)) for i in range(3)]
        await asyncio.sleep(0.01)
        assert c.sent == [] and len(c._release_buf) == 3
        assert not any(t.done() for t in held)
        await c.add_request_async(R(3))
        await asyncio.gather(*held)
        return c.sent

    sent = asyncio.run(go())
    # The fewest in flight first, then cyclic, each engine once.
    assert [e for _, e in sent] == ["e2", "e3", "e0", "e1"]
    assert c.engine_inflight["e2"] == 8 and c.reqs_in_flight["0"] == "e2"
    assert c._release_buf == []


def test_idle_engine_and_deadline_and_abort():
    C = _client_cls(counts=[[0, 0, 0.0], [1, 8, 0.5], [1, 8, 0.5], [1, 8, 0.5]])
    _install(C, deadline_s=0.05)
    c = C()

    async def go():
        # Engine 0 is idle: released at once, to the least in flight.
        await c.add_request_async(R(0))
        first = list(c.sent)
        c.lb_engines[0] = [1, 8, 0.5]
        one = _add(c, R(1))
        await asyncio.sleep(0.01)
        held = len(c._release_buf)
        await c.abort_requests_async(["1"])
        dropped = len(c._release_buf)
        with pytest.raises(asyncio.CancelledError):
            await one
        # The poller releases this one at the deadline.
        await c.add_request_async(R(2))
        return first, held, dropped, c.sent

    first, held, dropped, sent = asyncio.run(go())
    assert first == [("0", "e2")] and held == 1 and dropped == 0
    assert ("2", "e3") in sent


def test_quiet_release():
    C = _client_cls()
    _install(C, quiet_s=0.05)
    c = C()

    async def go():
        adds = [_add(c, R(0)), _add(c, R(1))]
        await asyncio.sleep(0.01)
        held = len(c._release_buf)
        await asyncio.wait_for(asyncio.gather(*adds), 2)
        return held, c.sent

    held, sent = asyncio.run(go())
    assert held == 2 and [r for r, _ in sent] == ["0", "1"]


def test_pinned_request_keeps_its_engine():
    C = _client_cls()
    _install(C, threshold=2)
    c = C()

    async def go():
        pinned = _add(c, R(0, rank=1))
        await asyncio.sleep(0.01)
        await c.add_request_async(R(1))
        await pinned
        return c.sent

    assert asyncio.run(go())[0] == ("0", "e1")


def test_failed_send_reaches_its_own_request():
    """The failed request's add raises, the rest go out, and the counts
    claim nothing for the failed one."""
    C = _client_cls(fail_engine="e2")
    _install(C, threshold=2)
    c = C()

    async def go():
        failing = _add(c, R(0))
        await asyncio.sleep(0.01)
        await c.add_request_async(R(1))
        with pytest.raises(RuntimeError, match="e2 is down"):
            await failing
        return c.sent, len(c._release_buf)

    sent, held = asyncio.run(go())
    assert sent == [("1", "e3")] and held == 0
    assert c.engine_inflight["e2"] == 7 and "0" not in c.reqs_in_flight
    assert c.reqs_in_flight["1"] == "e3"


def test_dead_engine_fails_every_held_request():
    C = _client_cls()
    _install(C)
    c = C()

    async def go():
        held = [_add(c, R(i)) for i in range(2)]
        await asyncio.sleep(0.01)
        c.resources.engine_dead = True
        with pytest.raises(EngineDeadError):
            await c.add_request_async(R(2))
        for t in held:
            with pytest.raises(EngineDeadError):
                await t
        return c.sent, c._release_buf

    assert asyncio.run(go()) == ([], [])


def test_install_if_enabled_checks_the_mode(monkeypatch):
    monkeypatch.setenv("DP_PLACEMENT", "relase")
    with pytest.raises(ValueError, match="release"):
        dp_placement.install_if_enabled()
    monkeypatch.delenv("DP_PLACEMENT")
    dp_placement.install_if_enabled()


def _block_first_send(C):
    """Make the client's first send block until `finish` is set."""
    started, finish = asyncio.Event(), asyncio.Event()

    def send_input(self, kind, request, engine):
        self.sent.append((request.request_id, engine))
        first = len(self.sent) == 1

        async def done():
            if first:
                started.set()
                await finish.wait()

        return done()

    C._send_input = send_input
    return started, finish


def test_abort_during_a_release_is_honoured_once():
    """A request aborted while another is being sent is never sent, and the
    rest are sent once."""
    C = _client_cls()
    started, finish = _block_first_send(C)
    _install(C, threshold=2, deadline_s=100.0)
    c = C()

    async def go():
        first = _add(c, R(0))
        await asyncio.sleep(0.01)
        releasing = _add(c, R(1))
        await asyncio.wait_for(started.wait(), 5)
        await c.abort_requests_async(["1"])
        with pytest.raises(asyncio.CancelledError):
            await releasing
        finish.set()
        await first
        held = list(c._release_buf)
        await asyncio.gather(_add(c, R(2)), _add(c, R(3)))
        return held, [rid for rid, _ in c.sent]

    held, sent = asyncio.run(go())
    assert held == [] and sent == ["0", "2", "3"]


def test_cancelled_add_leaves_the_other_sends_alone():
    """A client that leaves while another request's send is in flight: that
    send completes, and the leaver is never sent."""
    C = _client_cls()
    started, finish = _block_first_send(C)
    _install(C, threshold=2, deadline_s=100.0)
    c = C()

    async def go():
        first = _add(c, R(0))
        await asyncio.sleep(0.01)
        leaver = _add(c, R(1))
        await asyncio.wait_for(started.wait(), 5)
        leaver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaver
        finish.set()
        await asyncio.wait_for(first, 5)
        await asyncio.sleep(0.01)
        return [rid for rid, _ in c.sent], list(c._release_buf)

    sent, held = asyncio.run(go())
    assert sent == ["0"] and held == []
    assert c.reqs_in_flight["0"] == "e2" and "1" not in c.reqs_in_flight
