from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core.srpt_prefill_scheduler import SrptPrefillScheduler


def _running(name, num_tokens, computed, prefilling=True):
    return SimpleNamespace(
        name=name,
        is_prefill_chunk=prefilling,
        num_tokens=num_tokens,
        num_computed_tokens=computed,
    )


def _waiting(name, num_tokens, cached=0, status=RequestStatus.WAITING, preempted=0):
    return SimpleNamespace(
        name=name,
        num_tokens=num_tokens,
        num_computed_tokens=0,
        block_hashes=(name, cached),
        skip_reading_prefix_cache=False,
        status=status,
        num_preemptions=preempted,
        arrival_time=0.0,
    )


def _make_scheduler(running=(), waiting=(), max_running=8):
    s = SrptPrefillScheduler.__new__(SrptPrefillScheduler)
    s.running = list(running)
    s.waiting = FCFSRequestQueue(waiting)
    s.max_num_running_reqs = max_running
    # Fake prefix cache: block_hashes carries the cached token count.
    s.kv_cache_manager = SimpleNamespace(
        enable_caching=True,
        coordinator=SimpleNamespace(
            find_longest_cache_hit=lambda hashes, max_len: ((), hashes[1], 0)
        ),
    )
    s._srpt_steps = 0
    s._srpt_held_steps = 0
    s._srpt_held_chunks = 0
    return s


def _step(s):
    """Run schedule() with upstream faked; return what upstream saw."""
    seen = {}

    def fake(*args, **kwargs):
        seen["running"] = [r.name for r in s.running]
        seen["waiting"] = [r.name for r in s.waiting]
        seen["max_running"] = s.max_num_running_reqs
        return "output"

    with patch.object(AsyncScheduler, "schedule", side_effect=fake):
        assert s.schedule() == "output"
    return seen


def test_decodes_first_then_prefill_chunks_by_remaining():
    s = _make_scheduler(
        running=[
            _running("long", 40000, 1000),
            _running("dec", 500, 499, prefilling=False),
            _running("short", 3000, 2000),
        ]
    )
    seen = _step(s)
    assert seen["running"] == ["dec", "short", "long"]
    assert seen["max_running"] == 8


def test_waiting_sorted_by_uncached_tokens_preempted_first():
    s = _make_scheduler(
        waiting=[
            _waiting("cold", 9000, cached=0),
            _waiting("warm", 90000, cached=89000),
            _waiting("resumed", 50000, preempted=1),
        ]
    )
    seen = _step(s)
    assert seen["waiting"] == ["resumed", "warm", "cold"]


def test_long_chunk_held_for_shorter_waiting_request_and_restored():
    long_chunk = _running("long", 40000, 1000)
    s = _make_scheduler(
        running=[long_chunk, _running("dec", 500, 499, prefilling=False)],
        waiting=[_waiting("new", 80000, cached=79500)],
    )
    seen = _step(s)
    assert seen["running"] == ["dec"]
    assert seen["max_running"] == 7
    # Restored after the step, with the running limit.
    assert s.running[-1] is long_chunk
    assert s.max_num_running_reqs == 8
    assert (s._srpt_held_steps, s._srpt_held_chunks) == (1, 1)


def test_no_hold_when_waiting_is_longer_or_blocked_or_full():
    chunk = _running("chunk", 3000, 2000)
    for waiting, max_running in (
        ([_waiting("longer", 5000)], 8),
        ([_waiting("blocked", 10, status=RequestStatus.WAITING_FOR_REMOTE_KVS)], 8),
        ([_waiting("short", 10)], 1),
    ):
        s = _make_scheduler(running=[chunk], waiting=waiting, max_running=max_running)
        seen = _step(s)
        assert seen["running"] == ["chunk"], waiting[0].name
        assert s._srpt_held_steps == 0


def test_running_restored_when_upstream_raises():
    chunk = _running("long", 40000, 0)
    s = _make_scheduler(running=[chunk], waiting=[_waiting("new", 100)])
    with patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            s.schedule()
    assert s.running == [chunk]
    assert s.max_num_running_reqs == 8
