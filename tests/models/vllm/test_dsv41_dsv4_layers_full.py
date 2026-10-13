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
"""100% statement & branch coverage unit tests for deepseek_v4/layers.py."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel in (
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    (
        "vllm_torchtpu.models.vllm.deepseek_v4",
        "vllm_torchtpu/models/vllm/deepseek_v4",
    ),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR) / _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg


class _StubSiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


class _StubSiluAndMulWithClamp(nn.Module):
    def __init__(self, limit: float) -> None:
        super().__init__()
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].clamp(max=self.limit)
        up = x[..., d:].clamp(min=-self.limit, max=self.limit)
        return torch.nn.functional.silu(gate) * up


class _StubMergedLinear(nn.Module):
    def __init__(self, in_features: int, out_sizes: list[int], **kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, sum(out_sizes), bias=False)
        self._return_tpu = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = self.proj(x)
        if self._return_tpu:
            out = _as_tpu_tensor(out)
        return out, None


class _StubRowLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if isinstance(x, _DeviceViewTensor):
            x = x.as_subclass(torch.Tensor)
        return self.proj(x), None


class _StubModule(types.ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "SiluAndMul":
            return _StubSiluAndMul
        if name == "SiluAndMulWithClamp":
            return _StubSiluAndMulWithClamp
        if name == "MergedColumnParallelLinear":
            return _StubMergedLinear
        if name == "RowParallelLinear":
            return _StubRowLinear
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

from vllm_torchtpu.models.vllm.deepseek_v4 import layers as mod  # noqa: E402


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


def test_mhc_collapse_head_cpu_and_tpu():
    mod._TPU_OP_CACHE.clear()
    torch.manual_seed(0)
    hs_bf16 = torch.randn((3, 4, 8), dtype=torch.bfloat16)
    hc_fn = torch.randn((4, 32), dtype=torch.float32)
    hc_scale = torch.randn((4,), dtype=torch.float32)
    hc_base = torch.randn((4,), dtype=torch.float32)

    out_bf16 = mod.mhc_collapse_head(hs_bf16, hc_fn, hc_scale, hc_base, 1e-6, 1e-4)
    assert out_bf16.shape == (3, 8) and out_bf16.dtype == torch.bfloat16

    out_f32 = mod.mhc_collapse_head(
        hs_bf16.float(), hc_fn, hc_scale, hc_base, 1e-6, 1e-4
    )
    assert out_f32.shape == (3, 8) and out_f32.dtype == torch.float32

    with patch.object(torch, "compile", side_effect=_mock_compile):
        out_tpu1 = mod.mhc_collapse_head(
            _as_tpu_tensor(hs_bf16), hc_fn, hc_scale, hc_base, 1e-6, 1e-4
        )
        # Warm cache hit
        out_tpu2 = mod.mhc_collapse_head(
            _as_tpu_tensor(hs_bf16), hc_fn, hc_scale, hc_base, 1e-6, 1e-4
        )
        assert torch.equal(out_bf16, out_tpu1) and torch.equal(out_bf16, out_tpu2)


def test_deepseek_v4_mlp_init_and_forward():
    with pytest.raises(ValueError, match="Unsupported activation"):
        mod.DeepseekV4MLP(8, 16, "gelu")

    mlp_plain = mod.DeepseekV4MLP(8, 16, "silu", swiglu_limit=None)
    mlp_clamped = mod.DeepseekV4MLP(8, 16, "silu", swiglu_limit=2.0)

    x = torch.randn((4, 8), dtype=torch.float32)
    out_plain = mlp_plain(x)
    out_clamped = mlp_clamped(x)
    assert out_plain.shape == (4, 8) and out_clamped.shape == (4, 8)

    with patch.object(torch, "compile", side_effect=_mock_compile):
        mlp_clamped.gate_up_proj._return_tpu = True  # type: ignore[attr-defined]
        out_tpu = mlp_clamped(x)
        out_tpu_warm = mlp_clamped(x)
        assert torch.allclose(out_clamped, out_tpu)
        assert torch.allclose(out_clamped, out_tpu_warm)
