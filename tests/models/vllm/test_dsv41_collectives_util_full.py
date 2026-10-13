# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.kernels.collectives.util."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "kernels" / "collectives").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.kernels", _SRC_ROOT / "kernels"),
        ("vllm_torchtpu.kernels.collectives", _SRC_ROOT / "kernels" / "collectives"),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

if "jax.experimental.pallas" not in sys.modules:
    _jax_mod = sys.modules.setdefault("jax", types.ModuleType("jax"))
    _jax_src_mod = sys.modules.setdefault("jax._src", types.ModuleType("jax._src"))
    _xla_bridge_mod = sys.modules.setdefault(
        "jax._src.xla_bridge", types.ModuleType("jax._src.xla_bridge")
    )
    _xla_bridge_mod.backends_are_initialized = lambda: False  # type: ignore[attr-defined]
    _jax_exp_mod = sys.modules.setdefault(
        "jax.experimental", types.ModuleType("jax.experimental")
    )
    _pl_mod = sys.modules.setdefault(
        "jax.experimental.pallas", types.ModuleType("jax.experimental.pallas")
    )
    _pltpu_mod = sys.modules.setdefault(
        "jax.experimental.pallas.tpu", types.ModuleType("jax.experimental.pallas.tpu")
    )
    _jax_mod._src = _jax_src_mod  # type: ignore[attr-defined]
    _jax_src_mod.xla_bridge = _xla_bridge_mod  # type: ignore[attr-defined]
    _jax_mod.experimental = _jax_exp_mod  # type: ignore[attr-defined]
    _jax_exp_mod.pallas = _pl_mod  # type: ignore[attr-defined]
    _pl_mod.tpu = _pltpu_mod  # type: ignore[attr-defined]

from vllm_torchtpu.kernels.collectives import util as coll_util  # noqa: E402


def _make_fake_pallas_and_tpu() -> tuple[Any, Any, list[tuple[Any, ...]]]:
    events: list[tuple[Any, ...]] = []

    def _run_scoped(fn: Any, **kwargs: Any) -> Any:
        scoped_sems = {k: f"scoped:{k}:{v}" for k, v in kwargs.items()}
        events.append(("run_scoped", tuple(sorted(kwargs.items()))))
        return fn(**scoped_sems)

    def _get_barrier_semaphore() -> str:
        events.append(("get_barrier_semaphore",))
        return "global_barrier_sem"

    def _semaphore_signal(
        sem: Any,
        inc: int = 1,
        device_id: tuple[Any, ...] | None = None,
        device_id_type: Any = None,
    ) -> None:
        events.append(("signal", sem, inc, device_id, device_id_type))

    def _semaphore_wait(sem: Any, value: int) -> None:
        events.append(("wait", sem, value))

    fake_pl = SimpleNamespace(run_scoped=_run_scoped)
    fake_pltpu = SimpleNamespace(
        get_barrier_semaphore=_get_barrier_semaphore,
        semaphore_signal=_semaphore_signal,
        semaphore_wait=_semaphore_wait,
        DeviceIdType=SimpleNamespace(MESH="MESH_DEVICE_ID"),
        SemaphoreType=SimpleNamespace(REGULAR="REGULAR_SEM"),
    )
    return fake_pl, fake_pltpu, events


@pytest.mark.parametrize("double_barrier", [False, True])
def test_local_barrier_single_and_double(double_barrier: bool) -> None:
    """Verify local_barrier signals left and right neighbors and waits on semaphores."""
    fake_pl, fake_pltpu, events = _make_fake_pallas_and_tpu()
    with (
        patch.object(coll_util, "pl", fake_pl),
        patch.object(coll_util, "pltpu", fake_pltpu),
    ):
        coll_util.local_barrier(7, 1, double_barrier=double_barrier)

    expected: list[tuple[Any, ...]] = [
        ("get_barrier_semaphore",),
        ("signal", "global_barrier_sem", 1, (7,), "MESH_DEVICE_ID"),
        ("signal", "global_barrier_sem", 1, (1,), "MESH_DEVICE_ID"),
        ("wait", "global_barrier_sem", 2),
    ]
    if double_barrier:
        scoped_sem = "scoped:second_barrier:REGULAR_SEM"
        expected.extend(
            [
                ("run_scoped", (("second_barrier", "REGULAR_SEM"),)),
                ("signal", scoped_sem, 1, (7,), "MESH_DEVICE_ID"),
                ("signal", scoped_sem, 1, (1,), "MESH_DEVICE_ID"),
                ("wait", scoped_sem, 2),
            ]
        )
    assert events == expected


def test_local_barrier_reuses_device_id_tuples_across_double_barrier() -> None:
    """Verify left_dev and right_dev 1-tuples are allocated once and reused."""
    fake_pl, fake_pltpu, events = _make_fake_pallas_and_tpu()
    with (
        patch.object(coll_util, "pl", fake_pl),
        patch.object(coll_util, "pltpu", fake_pltpu),
    ):
        coll_util.local_barrier(2, 4, double_barrier=True)

    signal_events = [e for e in events if e[0] == "signal"]
    assert len(signal_events) == 4
    assert signal_events[0][3] is signal_events[2][3]
    assert signal_events[1][3] is signal_events[3][3]
