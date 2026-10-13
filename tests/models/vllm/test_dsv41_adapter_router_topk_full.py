# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for layers.adapter.router_topk."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.layers", ("vllm_torchtpu", "layers")),
    ("vllm_torchtpu.layers.adapter", ("vllm_torchtpu", "layers", "adapter")),
    ("vllm_torchtpu.layers.core", ("vllm_torchtpu", "layers", "core")),
    ("vllm_torchtpu.kernels", ("vllm_torchtpu", "kernels")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg


class _DummyJaxOp:
    def __init__(self, name: str, fn: Any) -> None:
        self.name = name
        self.fn = fn
        self.fake_fn: Any = None

    def register_fake(self, fake_fn: Any) -> None:
        self.fake_fn = fake_fn


if "torch_tpu" not in sys.modules:
    _tt_mod = types.ModuleType("torch_tpu")
    _tt_int = types.ModuleType("torch_tpu._internal")
    _tt_pallas = types.ModuleType("torch_tpu._internal.pallas")
    _tt_pallas.jax_op = _DummyJaxOp  # type: ignore[attr-defined]
    _tt_int.pallas = _tt_pallas  # type: ignore[attr-defined]
    _tt_mod._internal = _tt_int  # type: ignore[attr-defined]
    sys.modules["torch_tpu"] = _tt_mod
    sys.modules["torch_tpu._internal"] = _tt_int
    sys.modules["torch_tpu._internal.pallas"] = _tt_pallas

from vllm_torchtpu.layers.adapter import router_topk as adapter_rtk  # noqa: E402
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk  # noqa: E402

pytestmark = pytest.mark.cpu_test


class _FakeTpuTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._t = tensor
        self.device = SimpleNamespace(type="tpu")
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.float_calls = 0
        self.view_calls = 0

    def float(self) -> _FakeTpuTensor:
        self.float_calls += 1
        return _FakeTpuTensor(self._t.float())

    def view(self, *shape: int) -> _FakeTpuTensor:
        self.view_calls += 1
        return _FakeTpuTensor(self._t.view(*shape))


def test_adapter_router_topk_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 5.0, 2.0]], dtype=torch.float32
    )
    bias = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float32)

    # 1. _fake_router_topk and _fake_router_topk_biased
    w_f, i_f = adapter_rtk._fake_router_topk(scores, 2, False)
    assert w_f.shape == (2, 2) and w_f.dtype == torch.float32
    assert i_f.shape == (2, 2) and i_f.dtype == torch.int32

    wb_f, ib_f = adapter_rtk._fake_router_topk_biased(scores, bias.view(1, 4), 2, True)
    assert wb_f.shape == (2, 2) and wb_f.dtype == torch.float32
    assert ib_f.shape == (2, 2) and ib_f.dtype == torch.int32

    # 2. CPU fallback paths
    w_cpu, i_cpu = adapter_rtk.rowmax_select(scores, 2, renormalize=True)
    w_ref, i_ref = rowmax_topk(scores, 2, renormalize=True)
    assert torch.allclose(w_cpu, w_ref) and torch.equal(i_cpu, i_ref)

    # 3. TPU non-2D error path
    bad_tpu = _FakeTpuTensor(torch.zeros((2, 3, 4), dtype=torch.float32))
    with pytest.raises(ValueError, match="expects \\[rows, experts\\]"):
        adapter_rtk.rowmax_select(bad_tpu, 2)  # type: ignore[arg-type]

    captured: list[Any] = []

    def _mock_op(s: Any, topk: int, renorm: bool):
        captured.append(("unbiased", s, topk, renorm))
        return rowmax_topk(s._t, topk, renormalize=renorm)

    def _mock_biased_op(s: Any, b: Any, topk: int, renorm: bool):
        captured.append(("biased", s, b, topk, renorm))
        return rowmax_topk(s._t, topk, bias=b._t, renormalize=renorm)

    monkeypatch.setattr(adapter_rtk, "_op", _mock_op)
    monkeypatch.setattr(adapter_rtk, "_biased_op", _mock_biased_op)

    # 4. TPU unbiased (float32 and bfloat16)
    s_f32 = _FakeTpuTensor(scores)
    adapter_rtk.rowmax_select(s_f32, 2, renormalize=False)  # type: ignore[arg-type]
    assert s_f32.float_calls == 0

    s_bf16 = _FakeTpuTensor(scores.to(torch.bfloat16))
    adapter_rtk.rowmax_select(s_bf16, 2, renormalize=True)  # type: ignore[arg-type]
    assert s_bf16.float_calls == 1

    # 5. TPU biased (float32 2D bias and bfloat16 1D bias)
    b_2d = _FakeTpuTensor(bias.view(1, 4))
    adapter_rtk.rowmax_select(
        s_f32,
        2,
        bias=b_2d,
        renormalize=True,  # type: ignore[arg-type]
    )
    assert b_2d.float_calls == 0 and b_2d.view_calls == 0

    b_1d_bf16 = _FakeTpuTensor(bias.to(torch.bfloat16))
    adapter_rtk.rowmax_select(
        s_bf16,
        2,
        bias=b_1d_bf16,
        renormalize=False,  # type: ignore[arg-type]
    )
    assert b_1d_bf16.float_calls == 1
    assert captured[-1][2].shape == (1, 4)
    assert captured[-1][2].dtype == torch.float32
