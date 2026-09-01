import os
from unittest.mock import patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler


def _make_scheduler(waiting=(), drain_prefill=False):
    scheduler = TpuDpScheduler.__new__(TpuDpScheduler)
    scheduler.waiting = list(waiting)
    scheduler.skipped_waiting = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler._drain_prefill = drain_prefill
    scheduler._steps_since_window = 0
    scheduler.prefill_capacity_bound = False
    return scheduler


def _defer_env(hard=False, drain_steps=0):
    """Env patch enabling the DP prefill deferral path."""
    return patch.dict(
        os.environ, {
            "TPU_DP_PREFILL_DEFER": "1",
            "TPU_DP_PREFILL_HARD": "1" if hard else "0",
            "TPU_DP_PREFILL_DRAIN_STEPS": str(drain_steps),
        })


def test_unthrottled_step_opens_drain_window():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()

    with patch.object(AsyncScheduler, "schedule",
                      return_value=expected) as schedule:
        result = scheduler.schedule(throttle_prefills=False)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert scheduler._drain_prefill is True
    # Upstream throttling is neutralized so the cadence is the only clock.
    assert scheduler.prefill_capacity_bound is True


def test_throttled_step_pauses_new_requests():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return expected

    with patch.object(AsyncScheduler, "schedule",
                      side_effect=capture) as schedule:
        result = scheduler.schedule(throttle_prefills=True)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._pause_state is PauseState.UNPAUSED


def test_drain_spans_throttled_steps_until_queue_empties():
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return object()

    # A throttled step keeps draining while requests are still waiting.
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.UNPAUSED
    assert scheduler._drain_prefill is True

    # Once the queue drains, the window closes and buffering resumes.
    scheduler.waiting = []
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._drain_prefill is False


def test_pause_state_restored_when_schedule_raises():
    scheduler = _make_scheduler(waiting=[object()])

    with patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError):
        try:
            scheduler.schedule(throttle_prefills=True)
        except RuntimeError:
            pass

    assert scheduler._pause_state is PauseState.UNPAUSED


def test_defer_forwards_cadence_to_upstream():
    """With deferral on, the cadence reaches upstream's ``defer_prefills``."""
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)

    with _defer_env(), patch.object(AsyncScheduler, "schedule") as schedule:
        scheduler.schedule(throttle_prefills=True)

    schedule.assert_called_once_with(True)


def test_defer_hard_unpins_capacity_bound():
    """Hard mode keeps a backlog from re-opening prefill admission."""
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)
    scheduler.prefill_capacity_bound = True

    with _defer_env(hard=True), patch.object(AsyncScheduler, "schedule"):
        scheduler.schedule(throttle_prefills=True)

    assert scheduler.prefill_capacity_bound is False


def test_defer_soft_leaves_capacity_bound_to_upstream():
    """Without hard mode the scheduler must not touch upstream's flag."""
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)
    scheduler.prefill_capacity_bound = True

    with _defer_env(hard=False), patch.object(AsyncScheduler, "schedule"):
        scheduler.schedule(throttle_prefills=True)

    assert scheduler.prefill_capacity_bound is True


def test_bounded_window_closes_while_queue_stays_backlogged():
    """The saturation case: ``waiting`` never empties, so only the step
    budget can close the drain window."""
    scheduler = _make_scheduler(waiting=[object()])
    states = []

    def capture(*args, **kwargs):
        states.append(scheduler._pause_state)
        return object()

    with _defer_env(drain_steps=2), patch.object(AsyncScheduler,
                                                 "schedule",
                                                 side_effect=capture):
        # Cadence-aligned step opens the window.
        scheduler.schedule(throttle_prefills=False)
        # Two throttled steps: the first is still inside the window, the
        # second hits the budget and closes it.
        scheduler.schedule(throttle_prefills=True)
        scheduler.schedule(throttle_prefills=True)
        scheduler.schedule(throttle_prefills=True)

    assert scheduler.waiting, "queue must stay backlogged for this test"
    assert states == [
        PauseState.UNPAUSED,
        PauseState.UNPAUSED,
        PauseState.PAUSED_NEW,
        PauseState.PAUSED_NEW,
    ]
    assert scheduler._drain_prefill is False


def test_unbounded_window_never_closes_under_backlog():
    """Regression guard for the default: drain_steps=0 keeps the historical
    behaviour, where a backlogged rank never leaves drain mode."""
    scheduler = _make_scheduler(waiting=[object()])

    with _defer_env(drain_steps=0), patch.object(AsyncScheduler, "schedule"):
        scheduler.schedule(throttle_prefills=False)
        for _ in range(16):
            scheduler.schedule(throttle_prefills=True)

    assert scheduler._drain_prefill is True


def test_window_is_dp_identical_regardless_of_queue_depth():
    """The window is driven only by the DP-identical prefill cadence, so two
    ranks with different backlogs must agree step for step -- that agreement
    is what keeps the ``ReduceOp.MAX`` token bucket from being inflated."""
    rank_a = _make_scheduler(waiting=[object()] * 32)
    rank_b = _make_scheduler(waiting=[object()] * 2)
    cadence = [False] + [True] * 7 + [False] + [True] * 7
    seen_a, seen_b = [], []

    with _defer_env(hard=True, drain_steps=3):
        for throttle in cadence:
            for scheduler, seen in ((rank_a, seen_a), (rank_b, seen_b)):
                with patch.object(AsyncScheduler, "schedule") as schedule:
                    scheduler.schedule(throttle_prefills=throttle)
                seen.append(
                    (scheduler._drain_prefill, schedule.call_args.args))

    assert seen_a == seen_b
