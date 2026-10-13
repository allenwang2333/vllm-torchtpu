from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
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
        request_id=name,
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
    s._srpt_count_external = False
    s._srpt_external_end = {}
    s.connector = None
    s.block_size = 1024
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


def _with_budget(max_num_batched_tokens=128, max_running=64, draft_slots=5):
    s = _make_scheduler(max_running=max_running)
    s.scheduler_config = SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    s.max_num_scheduled_tokens = max_num_batched_tokens
    spec = (
        None
        if draft_slots is None
        else SimpleNamespace(max_num_new_slots_for_drafting=draft_slots)
    )
    s.vllm_config = SimpleNamespace(speculative_config=spec)
    return s


def test_uncharged_draft_slots_widen_only_the_private_input_budget():
    s = _with_budget()
    served = s.scheduler_config
    s._uncharge_draft_slots()
    # Every running request's drafting slots fit, so only the scheduled-token
    # cap can bind: 64 requests x 5 slots + 128 tokens.
    assert s.scheduler_config.max_num_batched_tokens == 128 + 64 * 5
    assert s.max_num_scheduled_tokens == 128
    assert served.max_num_batched_tokens == 128


def test_uncharged_draft_slots_noop_without_drafting_slots():
    for slots in (None, 0):
        s = _with_budget(draft_slots=slots)
        served = s.scheduler_config
        s._uncharge_draft_slots()
        assert s.scheduler_config is served


@pytest.mark.parametrize("value, widened", [("1", True), ("0", False), ("", False)])
def test_uncharged_draft_slots_follows_env(monkeypatch, value, widened):
    monkeypatch.setenv("TPU_SCHED_UNCHARGED_DRAFT_SLOTS", value)
    s = _with_budget()

    def fake_init(self, *args, **kwargs):
        self.policy = SchedulingPolicy.FCFS

    with patch.object(AsyncScheduler, "__init__", fake_init):
        SrptPrefillScheduler.__init__(s)
    assert (s.scheduler_config.max_num_batched_tokens > 128) is widened


class _FakeConnector:
    """Host-store hit lengths by request name; records each lookup."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append((request.name, num_computed_tokens))
        hit = self.hits[request.name]
        return (None, False) if hit is None else (hit, hit > 0)


def _with_connector(waiting, hits):
    s = _make_scheduler(waiting=waiting)
    s._srpt_count_external = True
    s.connector = _FakeConnector(hits)
    return s


def test_external_hit_counts_as_cached_when_enabled():
    # "offloaded": 2500 cached locally (block-aligned to 2048), 88000 more in
    # the host store -> 9500 left; "cold": 5000 left.
    waiting = [_waiting("cold", 5000), _waiting("offloaded", 100000, cached=2500)]
    s = _with_connector(waiting, {"cold": 0, "offloaded": 88000})
    assert _step(s)["waiting"] == ["cold", "offloaded"]
    s.connector.hits["offloaded"] = 96000  # 100000 - 98048 = 1952 left
    s._srpt_external_end.clear()
    assert _step(s)["waiting"] == ["offloaded", "cold"]
    assert ("offloaded", 2048) in s.connector.calls


def test_external_hit_asked_once_and_undetermined_asked_again():
    waiting = [_waiting("a", 9000), _waiting("b", 9000)]
    s = _with_connector(waiting, {"a": 4096, "b": None})
    _step(s)
    _step(s)
    names = [n for n, _ in s.connector.calls]
    assert names.count("a") == 1
    assert names.count("b") == 2


def test_external_hit_cache_dropped_when_request_leaves_waiting():
    waiting = [_waiting("a", 9000)]
    s = _with_connector(waiting, {"a": 4096})
    _step(s)
    assert "a" in s._srpt_external_end
    s.waiting.clear()
    _step(s)
    assert s._srpt_external_end == {}


def test_external_hit_ignored_when_disabled():
    s = _make_scheduler(waiting=[_waiting("cold", 5000), _waiting("offloaded", 100000)])
    s.connector = _FakeConnector({"cold": 0, "offloaded": 99000})
    assert _step(s)["waiting"] == ["cold", "offloaded"]
    assert s.connector.calls == []
