# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v2/async_fence.py."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _pkg_name, _rel in (
    ("vllm_torchtpu", "vllm_torchtpu"),
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41",
        "vllm_torchtpu/models/vllm/deepseek_v41",
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v41.v2",
        "vllm_torchtpu/models/vllm/deepseek_v41/v2",
    ),
    ("vllm_torchtpu.runner", "vllm_torchtpu/runner"),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_SRC_DIR / _rel)] if _SRC_DIR.is_dir() else []  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg
    if "." in _pkg_name:
        _p, _c = _pkg_name.rsplit(".", 1)
        setattr(sys.modules[_p], _c, sys.modules[_pkg_name])


class _FakeTPUModelRunner:
    def _prepare_lookback_token_ids(self, num_reqs: int) -> torch.Tensor:
        return torch.arange(num_reqs, dtype=torch.int32)


if "vllm_torchtpu.runner.tpu_runner" not in sys.modules:
    _tr = types.ModuleType("vllm_torchtpu.runner.tpu_runner")
    _tr.TPUModelRunner = _FakeTPUModelRunner  # type: ignore[attr-defined]
    sys.modules["vllm_torchtpu.runner.tpu_runner"] = _tr
    sys.modules["vllm_torchtpu.runner"].tpu_runner = _tr

v2_fence = importlib.import_module(
    "vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence"
)
tpu_runner_mod = importlib.import_module("vllm_torchtpu.runner.tpu_runner")


def test_install_lookback_fence_all_branches(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def dummy_original(runner_self, num_reqs: int) -> torch.Tensor:
        calls.append(num_reqs)
        return torch.tensor([num_reqs], dtype=torch.int32)

    monkeypatch.setattr(
        tpu_runner_mod.TPUModelRunner,
        "_prepare_lookback_token_ids",
        dummy_original,
        raising=False,
    )

    # 1. First installation wraps _prepare_lookback_token_ids
    v2_fence.install_lookback_fence()
    wrapped = tpu_runner_mod.TPUModelRunner._prepare_lookback_token_ids
    assert getattr(wrapped, "_deepseek_v41_fenced", False) is True

    # 2. Second installation is idempotent
    v2_fence.install_lookback_fence()
    assert tpu_runner_mod.TPUModelRunner._prepare_lookback_token_ids is wrapped

    # 3. Cold execution when _deepseek_v41_lookback_fence is None
    runner = tpu_runner_mod.TPUModelRunner()
    ev1 = MagicMock()
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=lambda: ev1), raising=False)
    out1 = runner._prepare_lookback_token_ids(4)
    assert torch.equal(out1, torch.tensor([4], dtype=torch.int32))
    assert calls == [4]
    ev1.synchronize.assert_not_called()
    ev1.record.assert_called_once()
    assert runner._deepseek_v41_lookback_fence is ev1

    # 4. Steady-state fast_prepare closure bound on instance
    ev1.reset_mock()
    out2 = runner._prepare_lookback_token_ids(8)
    assert torch.equal(out2, torch.tensor([8], dtype=torch.int32))
    assert calls == [4, 8]
    ev1.synchronize.assert_called_once()
    ev1.record.assert_called_once()

    # 5. Pre-existing _deepseek_v41_lookback_fence before first call
    runner_pre = tpu_runner_mod.TPUModelRunner()
    runner_pre._deepseek_v41_lookback_fence = ev1
    ev1.reset_mock()
    out3 = runner_pre._prepare_lookback_token_ids(2)
    assert torch.equal(out3, torch.tensor([2], dtype=torch.int32))
    ev1.synchronize.assert_called_once()
    ev1.record.assert_called_once()
