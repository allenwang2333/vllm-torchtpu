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
"""100% statement & branch coverage unit tests for layers/adapter/linear.py."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel in (
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR) / _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg


class _StubRowParallelLinear(nn.Module):
    @classmethod
    def register_oot(cls, subcls):
        return subcls

    def forward(self, input_: torch.Tensor):
        return input_ + 100.0


def _stub_split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int
) -> tuple[torch.Tensor, ...]:
    part = tensor.shape[-1] // num_partitions
    return torch.split(tensor, part, dim=-1)


def _stub_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0


class _StubModule(types.ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "RowParallelLinear":
            return _StubRowParallelLinear
        if name == "split_tensor_along_last_dim":
            return _stub_split_tensor_along_last_dim
        if name == "tensor_model_parallel_all_reduce":
            return _stub_all_reduce
        if name[:1].isupper():
            return type(name, (), {"__init__": lambda self, *a, **k: None})
        return MagicMock()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root == "vllm":
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        return None


try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    sys.meta_path.insert(0, _StubFinder())

from vllm_torchtpu.layers.adapter import linear as mod  # noqa: E402


class _DeviceViewTensor(torch.Tensor):
    _fake_device_type = "tpu"

    @property
    def device(self):  # type: ignore[override]
        return types.SimpleNamespace(type=self._fake_device_type)


def _as_tpu_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_DeviceViewTensor)


def _mock_compile(fn, **kw):
    def _wrapped(*args):
        clean_args = [
            a.as_subclass(torch.Tensor) if isinstance(a, _DeviceViewTensor) else a
            for a in args
        ]
        return fn(*clean_args)

    return _wrapped


def _build_layer(
    *,
    input_is_parallel: bool = True,
    tp_rank: int = 0,
    tp_size: int = 2,
    reduce_results: bool = True,
    skip_bias_add: bool = False,
    bias: torch.Tensor | None = None,
    return_bias: bool = False,
    weight: torch.Tensor | None = None,
) -> mod.TpuRowParallelLinear:
    layer = mod.TpuRowParallelLinear.__new__(mod.TpuRowParallelLinear)
    nn.Module.__init__(layer)
    layer.input_is_parallel = input_is_parallel
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.reduce_results = reduce_results
    layer.skip_bias_add = skip_bias_add
    layer.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
    layer.return_bias = return_bias
    w = weight if weight is not None else torch.eye(4, dtype=torch.float32)
    layer.quant_method = SimpleNamespace(
        apply=lambda _lyr, inp, _b: torch.mm(inp, w.t())
    )
    return layer


def test_tpu_row_parallel_linear_cpu_and_tpu(monkeypatch):
    mod._TPU_OP_CACHE.clear()
    monkeypatch.setattr(mod, "tensor_model_parallel_all_reduce", lambda t: t * 2.0)
    monkeypatch.setattr(mod.RowParallelLinear, "forward", lambda self, x: x + 999.0)

    x_par = torch.ones((3, 4), dtype=torch.float32)
    x_full = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    w = torch.eye(4, dtype=torch.float32)

    # 1. defer_bias=False -> delegates to super().forward
    layer_no_defer = _build_layer(tp_size=1, bias=bias, weight=w)
    assert torch.equal(layer_no_defer(x_par), x_par + 999.0)

    # 2. defer_bias=True, input_is_parallel=True, return_bias=False
    layer_par = _build_layer(
        input_is_parallel=True,
        tp_rank=0,
        tp_size=2,
        bias=bias,
        return_bias=False,
        weight=w,
    )
    out_cpu_par = layer_par(x_par)
    assert torch.equal(out_cpu_par, x_par * 2.0 + bias)

    with patch.object(torch, "compile", side_effect=_mock_compile):
        out_tpu_par1 = layer_par(_as_tpu_tensor(x_par))
        out_tpu_par2 = layer_par(_as_tpu_tensor(x_par))
        assert torch.equal(out_cpu_par, out_tpu_par1)
        assert torch.equal(out_cpu_par, out_tpu_par2)

    # 3. defer_bias=True, input_is_parallel=False, return_bias=True (CPU + TPU)
    layer_nonpar = _build_layer(
        input_is_parallel=False,
        tp_rank=1,
        tp_size=2,
        bias=bias,
        return_bias=True,
        weight=w,
    )
    out_cpu_np, ret_bias_cpu = layer_nonpar(x_full)
    assert ret_bias_cpu is None
    assert torch.equal(out_cpu_np, x_full[:, 4:] * 2.0 + bias)

    with patch.object(torch, "compile", side_effect=_mock_compile):
        out_tpu_np, ret_bias_tpu = layer_nonpar(_as_tpu_tensor(x_full))
        assert ret_bias_tpu is None
        assert torch.equal(out_cpu_np, out_tpu_np)
