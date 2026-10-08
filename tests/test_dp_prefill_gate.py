from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core.dp_prefill_gate import DpPrefillGateScheduler

BUDGET = 2048


def _req(prefilling):
    return SimpleNamespace(is_prefill_chunk=prefilling)


def _make_scheduler(min_cadences=0, log_steps=0, running=(), waiting=()):
    scheduler = DpPrefillGateScheduler.__new__(DpPrefillGateScheduler)
    scheduler.running = list(running)
    scheduler.waiting = list(waiting)
    scheduler.prefill_capacity_bound = False
    scheduler.max_num_scheduled_tokens = BUDGET
    scheduler.max_num_running_reqs = 8
    scheduler._gate_min_cadences = min_cadences
    scheduler._gate_log_steps = log_steps
    scheduler._gate_streak = 0
    scheduler._gate_stats = Counter()
    return scheduler


def _step(scheduler, throttle, tokens, upstream_bound=None):
    """Run one schedule() with upstream replaced by a fake that schedules ``tokens``
    (request id -> token count) and, like upstream on a step it did not hold
    back, may rewrite prefill_capacity_bound."""
    output = SimpleNamespace(
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
    )

    def fake(*args, **kwargs):
        if upstream_bound is not None:
            scheduler.prefill_capacity_bound = upstream_bound
        return output

    with patch.object(AsyncScheduler, "schedule", side_effect=fake) as schedule:
        result = scheduler.schedule(throttle_prefills=throttle)
    schedule.assert_called_once_with(throttle)
    assert result is output
    return result


def _starved():
    # Four 512-token chunks use the whole budget.
    return {f"p{i}": 512 for i in range(4)}


def test_stats_only_leaves_upstream_flag_alone():
    scheduler = _make_scheduler(
        min_cadences=0, running=[_req(True)], waiting=[object()]
    )

    _step(scheduler, throttle=False, tokens={"p0": 512}, upstream_bound=True)

    assert scheduler.prefill_capacity_bound is True
    assert scheduler._gate_stats["cadence_prefill"] == 1
    assert scheduler._gate_stats["cadence_waiting_other"] == 1


def test_throttled_prefill_attributed_to_escape_reason():
    scheduler = _make_scheduler(running=[_req(True), _req(False)])

    scheduler.prefill_capacity_bound = True
    _step(scheduler, throttle=True, tokens={"p0": 512, "d0": 1})
    scheduler.prefill_capacity_bound = False
    scheduler.running = [_req(True)]
    _step(scheduler, throttle=True, tokens={"p0": 512})
    # Held back: only the decode ran.
    scheduler.running = [_req(True), _req(False)]
    _step(scheduler, throttle=True, tokens={"d0": 1})

    stats = scheduler._gate_stats
    assert stats["throttled"] == 3
    assert stats["throttled_prefill_bound"] == 1
    assert stats["throttled_prefill_no_decoder"] == 1
    assert stats["throttled_prefill_other"] == 0
    assert stats["throttled_prefill_tokens"] == 1024


def test_bound_needs_consecutive_starved_cadences():
    scheduler = _make_scheduler(
        min_cadences=2, running=[_req(True)] * 4, waiting=[object()]
    )

    _step(scheduler, throttle=False, tokens=_starved(), upstream_bound=True)
    assert scheduler.prefill_capacity_bound is False
    # A held-back step in between neither advances nor resets the streak.
    _step(scheduler, throttle=True, tokens={})
    assert scheduler.prefill_capacity_bound is False
    _step(scheduler, throttle=False, tokens=_starved(), upstream_bound=True)
    assert scheduler.prefill_capacity_bound is True
    assert scheduler._gate_stats["cadence_waiting_budget"] == 2


def test_waiting_without_exhausted_budget_is_not_bound():
    # Held back by max_num_seqs: prefilling more often admits nobody.
    scheduler = _make_scheduler(
        min_cadences=1, running=[_req(False)] * 8, waiting=[object()]
    )

    _step(
        scheduler,
        throttle=False,
        tokens={f"d{i}": 1 for i in range(8)},
        upstream_bound=True,
    )

    assert scheduler.prefill_capacity_bound is False
    assert scheduler._gate_stats["cadence_waiting_seqs"] == 1


def test_escape_stops_once_budget_no_longer_exhausted():
    scheduler = _make_scheduler(
        min_cadences=1, running=[_req(True)] * 4, waiting=[object()]
    )
    _step(scheduler, throttle=False, tokens=_starved(), upstream_bound=True)
    assert scheduler.prefill_capacity_bound is True

    _step(scheduler, throttle=True, tokens=_starved(), upstream_bound=True)
    assert scheduler.prefill_capacity_bound is True

    scheduler.waiting = []
    _step(scheduler, throttle=True, tokens={"p0": 512}, upstream_bound=False)
    assert scheduler.prefill_capacity_bound is False


def test_log_resets_counters():
    scheduler = _make_scheduler(log_steps=2, running=[_req(False)])

    _step(scheduler, throttle=True, tokens={"d0": 1})
    assert scheduler._gate_stats["steps"] == 1
    _step(scheduler, throttle=False, tokens={"d0": 1})
    assert scheduler._gate_stats["steps"] == 0
