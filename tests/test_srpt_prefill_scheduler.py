import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
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
    s.skipped_waiting = FCFSRequestQueue()
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
    s._srpt_progress_guard = False
    s._srpt_strict_cadence = False
    s._offcadence_tokens = 0
    s._offcadence_steps = 0
    s._offcadence_max_context = None
    s._offcadence_active = False
    s._deep_chunk_align = 0
    s._deep_chunk_context = 0
    s._deep_chunks_aligned = 0
    s.max_num_scheduled_tokens = 128
    s.prefill_capacity_bound = False
    s._srpt_hold_paused = False
    s._srpt_paused_steps = 0
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


def _step_admitting(s, admit):
    """schedule() with upstream faked: with admit="run" it admits the head
    waiting request to RUNNING, with admit="load" it starts the head's KV load
    (upstream parks it in skipped_waiting as WAITING_FOR_REMOTE_KVS), with
    admit=False it admits nothing. Returns the names upstream saw running."""
    seen = {}

    def fake(*args, **kwargs):
        seen["running"] = [r.name for r in s.running]
        if admit and len(s.waiting):
            r = s.waiting.pop_request()
            if admit == "load":
                r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                s.skipped_waiting.prepend_request(r)
            else:
                r.status = RequestStatus.RUNNING
                r.is_prefill_chunk = True
                r.num_computed_tokens = r.num_tokens - 1
                s.running.append(r)
        return "output"

    with patch.object(AsyncScheduler, "schedule", side_effect=fake):
        assert s.schedule() == "output"
    return seen["running"]


def _livelock_scheduler(guard):
    long_chunk = _running("long", 40000, 1000)
    s = _make_scheduler(
        running=[long_chunk], waiting=[_waiting("new", 80000, cached=79500)]
    )
    s._srpt_progress_guard = guard
    return s


def test_hold_that_admits_nothing_skips_holding_for_one_step():
    s = _livelock_scheduler(guard=True)
    assert _step_admitting(s, admit=False) == []  # held; "new" did not fit
    assert _step_admitting(s, admit=False) == ["long"]  # one step without holding
    assert _step_admitting(s, admit=False) == []  # holding again
    assert _step_admitting(s, admit=False) == ["long"]
    assert (s._srpt_held_steps, s._srpt_paused_steps) == (2, 2)


def test_hold_that_admits_keeps_holding():
    s = _livelock_scheduler(guard=True)
    assert _step_admitting(s, admit="run") == []  # held; "new" admitted
    s.waiting.extend([_waiting("next", 90000, cached=89900)])
    assert _step_admitting(s, admit=False) == ["new"]  # holding again
    assert s._srpt_paused_steps == 0


def test_hold_repeats_without_progress_guard():
    s = _livelock_scheduler(guard=False)
    for _ in range(3):
        assert _step_admitting(s, admit=False) == []
    assert (s._srpt_held_steps, s._srpt_paused_steps) == (3, 0)


def test_hold_that_starts_a_kv_load_counts_as_progress():
    s = _livelock_scheduler(guard=True)
    assert _step_admitting(s, admit="load") == []  # held; "new" starts its host load
    assert not s._srpt_hold_paused
    s.waiting.extend([_waiting("next", 80000, cached=79500)])
    assert _step_admitting(s, admit=False) == []  # still holding for "next"
    assert s._srpt_paused_steps == 0


def _step_throttled(s, throttle):
    seen = {}

    def fake(throttle_prefills):
        seen["running"] = [r.name for r in s.running]
        seen["bound"] = s.prefill_capacity_bound
        seen["throttle"] = throttle_prefills
        # upstream: the release step left requests waiting
        s.prefill_capacity_bound = True
        return "output"

    with patch.object(AsyncScheduler, "schedule", side_effect=fake):
        assert s.schedule(throttle) == "output"
    return seen


def test_strict_cadence_clears_capacity_bound_and_skips_holds_off_cadence():
    s = _livelock_scheduler(guard=False)
    s._srpt_strict_cadence = True
    seen = _step_throttled(s, throttle=False)  # cadence step: SRPT holds as usual
    assert (seen["running"], seen["bound"], seen["throttle"]) == ([], False, False)
    seen = _step_throttled(s, throttle=True)  # off-cadence: upstream defers prefill
    assert (seen["running"], seen["bound"], seen["throttle"]) == (["long"], False, True)


def _decoding(name, spec=6):
    """A decoding request upstream schedules 1 + spec tokens for."""
    return SimpleNamespace(
        name=name,
        is_prefill_chunk=False,
        num_tokens=1000,
        num_tokens_with_spec=1001 + spec,
        num_output_placeholders=0,
        num_computed_tokens=1000,
    )


def _step_capped(s, throttle):
    seen = {}

    def fake(throttle_prefills):
        seen["running"] = [r.name for r in s.running]
        seen["throttle"] = throttle_prefills
        seen["cap"] = s.max_num_scheduled_tokens
        return "output"

    with patch.object(AsyncScheduler, "schedule", side_effect=fake):
        assert s.schedule(throttle) == "output"
    assert s.max_num_scheduled_tokens == 128
    return seen


def _offcadence_scheduler(running, tokens=32):
    s = _make_scheduler(running=running)
    s._srpt_strict_cadence = True
    s._offcadence_tokens = tokens
    return s


def test_offcadence_step_prefills_within_the_cap():
    s = _offcadence_scheduler([_decoding("d"), _running("p", 5000, 1000)])
    seen = _step_capped(s, throttle=True)
    assert seen == {"running": ["d", "p"], "throttle": False, "cap": 32}
    assert s._offcadence_steps == 1


def test_offcadence_cap_never_starves_decodes():
    decodes = [_decoding(f"d{i}") for i in range(6)]  # 6 x 7 = 42 tokens
    s = _offcadence_scheduler(decodes + [_running("p", 5000, 1000)])
    assert _step_capped(s, throttle=True)["cap"] == 42


def test_offcadence_cap_counts_placeholders_and_short_drafts():
    d = _decoding("d", spec=2)
    d.num_output_placeholders = 3
    s = _offcadence_scheduler([d], tokens=1)
    assert _step_capped(s, throttle=True)["cap"] == 6


def test_cadence_step_keeps_the_full_budget():
    s = _offcadence_scheduler([_decoding("d"), _running("p", 5000, 1000)])
    seen = _step_capped(s, throttle=False)
    assert (seen["throttle"], seen["cap"]) == (False, 128)
    assert s._offcadence_steps == 0


def test_offcadence_cap_restored_when_upstream_raises():
    s = _offcadence_scheduler([_decoding("d")])
    with patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            s.schedule(True)
    assert s.max_num_scheduled_tokens == 128


def test_capacity_bound_kept_without_strict_cadence():
    s = _livelock_scheduler(guard=False)
    _step_throttled(s, throttle=False)
    assert _step_throttled(s, throttle=True)["bound"] is True


def _capped(cap):
    s = _make_scheduler()
    s._attn_work_cap = cap
    s._attn_work = 0.0
    s._attn_prefill_scheduled = False
    s._attn_clipped_chunks = 0
    s.num_prefill_lookahead = 0  # upstream's MTP lookahead reserve: a no-op here
    return s


def _req(prompt, status=RequestStatus.RUNNING, num_tokens=None):
    return SimpleNamespace(
        num_prompt_tokens=prompt, num_tokens=num_tokens or prompt, status=status
    )


def _offcadence_context(limit, active=True):
    s = _capped(0)
    s._offcadence_max_context = limit
    s._offcadence_active = active
    return s


def test_offcadence_context_limit_clips_and_defers_chunks():
    s = _offcadence_context(8192)
    assert s._reserve_prefill_lookahead(_req(20_000), 0, 32) == 32  # shallow: whole
    assert s._reserve_prefill_lookahead(_req(20_000), 8180, 32) == 12  # stops at 8192
    assert s._reserve_prefill_lookahead(_req(200_000), 150_000, 18) == 0  # deep: waits


def test_offcadence_context_zero_leaves_only_decodes():
    s = _offcadence_context(0)
    assert s._reserve_prefill_lookahead(_req(20_000), 0, 32) == 0
    decode = _req(200_000, num_tokens=200_010)
    assert s._reserve_prefill_lookahead(decode, 200_010, 7) == 7


def test_offcadence_context_limit_ignored_on_cadence_steps():
    s = _offcadence_context(0, active=False)
    assert s._reserve_prefill_lookahead(_req(200_000), 150_000, 128) == 128


def test_offcadence_step_flag_set_only_during_capped_schedule():
    s = _offcadence_scheduler([_decoding("d")])
    seen = []
    with patch.object(
        AsyncScheduler, "schedule", side_effect=lambda t: seen.append(s._offcadence_active)
    ):
        s.schedule(True)
        s.schedule(False)
    assert seen == [True, False] and not s._offcadence_active


def _deep_aligned(align=32, context=16384):
    s = _capped(0)
    s._deep_chunk_align = align
    s._deep_chunk_context = context
    return s


def test_deep_chunk_rounded_down_to_whole_query_blocks():
    s = _deep_aligned()
    assert s._reserve_prefill_lookahead(_req(200_000), 84_000, 49) == 32
    assert s._reserve_prefill_lookahead(_req(200_000), 84_000, 113) == 96
    assert s._reserve_prefill_lookahead(_req(200_000), 84_000, 64) == 64
    assert s._reserve_prefill_lookahead(_req(200_000), 84_000, 17) == 0
    assert s._deep_chunks_aligned == 3


def test_deep_chunk_alignment_spares_shallow_and_final_chunks():
    s = _deep_aligned()
    assert s._reserve_prefill_lookahead(_req(200_000), 1000, 49) == 49  # shallow
    assert s._reserve_prefill_lookahead(_req(84_049), 84_000, 49) == 49  # ends prompt
    decode = _req(200_000, num_tokens=200_010)
    assert s._reserve_prefill_lookahead(decode, 200_010, 7) == 7
    assert s._deep_chunks_aligned == 0


def test_deep_chunk_alignment_off_by_default():
    s = _capped(0)
    assert s._reserve_prefill_lookahead(_req(200_000), 84_000, 49) == 49


def test_attn_cap_off_leaves_chunks_alone():
    s = _capped(0)
    assert s._reserve_prefill_lookahead(_req(400_000), 300_000, 128) == 128
    assert s._attn_work == 0.0


def test_attn_cap_clips_a_deep_chunk_to_the_remaining_work():
    s = _capped(10_000_000)
    n = s._reserve_prefill_lookahead(_req(400_000), 100_000, 128)
    assert n == 99  # 99 * (100k + 49.5) <= 10M < 100 * (100k + 50)
    assert n * (100_000 + n / 2) <= 10_000_000
    assert s._attn_clipped_chunks == 1


def test_attn_cap_counts_decodes_first_and_never_clips_them():
    s = _capped(1_000_000)
    decode = _req(200_000, num_tokens=200_010)
    assert s._reserve_prefill_lookahead(decode, 200_010, 7) == 7  # over the cap alone
    assert s._attn_work == pytest.approx(7 * (200_010 + 3.5))
    # no room left: the step's first chunk still advances, a second one waits
    assert s._reserve_prefill_lookahead(_req(400_000), 100_000, 128) == 16
    assert s._reserve_prefill_lookahead(_req(400_000), 100_000, 128) == 0


def test_attn_cap_still_advances_the_steps_first_chunk():
    s = _capped(1_000_000)  # less than 16 tokens of a 300k-context chunk
    assert s._reserve_prefill_lookahead(_req(400_000), 300_000, 128) == 16
    # a second deep chunk in the same step waits
    assert s._reserve_prefill_lookahead(_req(500_000), 300_000, 128) == 0


def test_attn_cap_admits_a_waiting_request_with_one_token_rather_than_none():
    s = _capped(1_000_000)
    s._reserve_prefill_lookahead(_req(50_000), 9_000, 100)  # most of the cap
    waiting = _req(400_000, status=RequestStatus.WAITING)
    assert s._reserve_prefill_lookahead(waiting, 300_000, 128) == 1


def test_attn_cap_lets_shallow_chunks_through_whole():
    s = _capped(25_000_000)
    assert s._reserve_prefill_lookahead(_req(60_000), 40_000, 121) == 121
    assert s._attn_clipped_chunks == 0


def test_schedule_resets_the_attention_work():
    s = _capped(10_000_000)
    s._reserve_prefill_lookahead(_req(400_000), 100_000, 128)
    _step(s)
    assert s._attn_work == 0.0


def test_upstream_routes_both_queues_through_the_chunk_hook():
    # The cap is inert unless upstream calls the hook for running chunks and for
    # waiting requests alike; fail loudly if a vLLM update drops either call.
    assert (
        inspect.getsource(Scheduler.schedule).count("self._reserve_prefill_lookahead(")
        == 2
    )
