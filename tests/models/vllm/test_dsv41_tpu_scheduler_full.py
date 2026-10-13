# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.core.tpu_scheduler."""

from __future__ import annotations

import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "core").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.core", _SRC_ROOT / "core"),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

if "vllm.v1.core.sched.async_scheduler" not in sys.modules:
    _vllm = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    _vllm_logger = sys.modules.setdefault(
        "vllm.logger", types.ModuleType("vllm.logger")
    )
    _vllm_logger.init_logger = lambda name: MagicMock()  # type: ignore[attr-defined]
    _vllm_logger._print_warning_once = MagicMock()  # type: ignore[attr-defined]
    _vllm.logger = _vllm_logger  # type: ignore[attr-defined]
    _vllm_v1 = sys.modules.setdefault("vllm.v1", types.ModuleType("vllm.v1"))
    _vllm_core = sys.modules.setdefault(
        "vllm.v1.core", types.ModuleType("vllm.v1.core")
    )
    _vllm_sched = sys.modules.setdefault(
        "vllm.v1.core.sched", types.ModuleType("vllm.v1.core.sched")
    )
    _vllm_async = sys.modules.setdefault(
        "vllm.v1.core.sched.async_scheduler",
        types.ModuleType("vllm.v1.core.sched.async_scheduler"),
    )
    _vllm_iface = sys.modules.setdefault(
        "vllm.v1.core.sched.interface",
        types.ModuleType("vllm.v1.core.sched.interface"),
    )
    _vllm_out = sys.modules.setdefault(
        "vllm.v1.core.sched.output",
        types.ModuleType("vllm.v1.core.sched.output"),
    )

    class _PauseState(Enum):
        UNPAUSED = 0
        PAUSED_NEW = 1
        PAUSED_ALL = 2

    class _SchedulerOutput:
        pass

    class _AsyncScheduler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def schedule(self, throttle_prefills: bool = False) -> Any:
            return _SchedulerOutput()

    _vllm_iface.PauseState = _PauseState  # type: ignore[attr-defined]
    _vllm_out.SchedulerOutput = _SchedulerOutput  # type: ignore[attr-defined]
    _vllm_async.AsyncScheduler = _AsyncScheduler  # type: ignore[attr-defined]

from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.interface import PauseState  # noqa: E402

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler  # noqa: E402


def _make_scheduler(
    waiting: list[Any] | None = None,
    kv_holding: list[Any] | None = None,
    skipped: list[Any] | None = None,
    drain_prefill: bool = False,
    pause_state: PauseState = PauseState.UNPAUSED,
) -> TpuDpScheduler:
    sched = TpuDpScheduler.__new__(TpuDpScheduler)
    sched.waiting = [] if waiting is None else list(waiting)
    if kv_holding is not None:
        sched.kv_holding_waiting = list(kv_holding)
    if skipped is not None:
        sched.skipped_waiting = list(skipped)
    sched._drain_prefill = drain_prefill
    sched._pause_state = pause_state
    sched.prefill_capacity_bound = False
    return sched


def test_init_and_direct_schedule_paused_new() -> None:
    """Verify __init__ and _schedule_paused_new for UNPAUSED and non-UNPAUSED states."""
    with patch.object(AsyncScheduler, "__init__", return_value=None):
        sched = TpuDpScheduler()
    assert sched._drain_prefill is False

    sched._pause_state = PauseState.UNPAUSED
    seen: list[PauseState] = []

    def _capture(self_obj: Any, flag: bool) -> str:
        assert flag is False
        seen.append(self_obj._pause_state)
        return "ok"

    with patch.object(AsyncScheduler, "schedule", _capture):
        assert sched._schedule_paused_new() == "ok"
        assert seen == [PauseState.PAUSED_NEW]
        assert sched._pause_state is PauseState.UNPAUSED

        sched._pause_state = PauseState.PAUSED_ALL
        assert sched._schedule_paused_new() == "ok"
        assert seen[-1] is PauseState.PAUSED_ALL
        assert sched._pause_state is PauseState.PAUSED_ALL


def test_schedule_throttled_and_unthrottled_branches() -> None:
    """Verify all branches of TpuDpScheduler.schedule."""
    seen: list[PauseState] = []

    def _capture(self_obj: Any, flag: bool) -> str:
        assert flag is False
        seen.append(self_obj._pause_state)
        return "step"

    with patch.object(AsyncScheduler, "schedule", _capture):
        # 1. Throttled, not draining, UNPAUSED -> pauses new and restores
        s1 = _make_scheduler(waiting=[object()], drain_prefill=False)
        assert s1.schedule(throttle_prefills=True) == "step"
        assert seen[-1] is PauseState.PAUSED_NEW
        assert s1._pause_state is PauseState.UNPAUSED
        assert s1.prefill_capacity_bound is True

        # 2. Throttled, not draining, already PAUSED_ALL -> keeps PAUSED_ALL
        s2 = _make_scheduler(
            waiting=[object()],
            drain_prefill=False,
            pause_state=PauseState.PAUSED_ALL,
        )
        assert s2.schedule(throttle_prefills=True) == "step"
        assert seen[-1] is PauseState.PAUSED_ALL

        # 3. Unthrottled with waiting -> opens drain window and runs unpaused
        s3 = _make_scheduler(waiting=[object()], drain_prefill=False)
        assert s3.schedule(throttle_prefills=False) == "step"
        assert s3._drain_prefill is True
        assert seen[-1] is PauseState.UNPAUSED

        # 4. Throttled while draining via kv_holding_waiting -> stays draining
        s4 = _make_scheduler(waiting=[], kv_holding=[object()], drain_prefill=True)
        assert s4.schedule(throttle_prefills=True) == "step"
        assert s4._drain_prefill is True
        assert seen[-1] is PauseState.UNPAUSED

        # 5. Throttled while draining via skipped_waiting -> stays draining
        s5 = _make_scheduler(waiting=[], skipped=[object()], drain_prefill=True)
        assert s5.schedule(throttle_prefills=True) == "step"
        assert s5._drain_prefill is True
        assert seen[-1] is PauseState.UNPAUSED

        # 6. Draining with empty queues, UNPAUSED -> closes drain window & pauses new
        s6 = _make_scheduler(waiting=[], kv_holding=[], drain_prefill=True)
        assert s6.schedule(throttle_prefills=True) == "step"
        assert s6._drain_prefill is False
        assert seen[-1] is PauseState.PAUSED_NEW
        assert s6._pause_state is PauseState.UNPAUSED

        # 7. Draining with empty queues & PAUSED_ALL -> closes drain window
        s7 = _make_scheduler(
            waiting=[],
            drain_prefill=True,
            pause_state=PauseState.PAUSED_ALL,
        )
        assert s7.schedule(throttle_prefills=False) == "step"
        assert s7._drain_prefill is False
        assert seen[-1] is PauseState.PAUSED_ALL

    # 8. Exception inside super().schedule restores _pause_state
    s_err = _make_scheduler(waiting=[object()], drain_prefill=False)
    with (
        patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        s_err.schedule(throttle_prefills=True)
    assert s_err._pause_state is PauseState.UNPAUSED
