# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_TPU_ACTIVATION_FP32: the override must be the method vLLM really runs.

vllm-torchtpu is an out-of-tree platform, so ``CustomOp.dispatch_forward`` never
returns ``forward_tpu``: it returns ``forward_oot`` when the op is enabled and
``forward_native`` when it is not. An override that lands anywhere else is a
silent no-op that still logs as active, which is exactly what this file pins.
"""

import pytest
import torch
import torch.nn.functional as F
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.model_executor import custom_op
from vllm.model_executor.layers import activation

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.activation import (
    _FAMILIES,
    _register_fp32_activations,
)

SWIGLU = set(_FAMILIES["silu"])


def _activation_custom_ops() -> set[str]:
    return {
        name
        for name, cls in vars(activation).items()
        if isinstance(cls, type)
        and cls.__module__ == activation.__name__
        and issubclass(cls, custom_op.CustomOp)
    }


@pytest.fixture
def oot_registry():
    """Leave the process-wide out-of-tree registry as it was found."""
    saved = dict(custom_op.op_registry_oot)
    yield custom_op.op_registry_oot
    custom_op.op_registry_oot.clear()
    custom_op.op_registry_oot.update(saved)


class _OutOfTreePlatform:
    """What vllm_torchtpu's TpuPlatform looks like to CustomOp.dispatch_forward."""

    simple_compile_backend = "eager"

    def is_rocm(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False

    def is_tpu(self) -> bool:
        return False

    def is_xpu(self) -> bool:
        return False

    def is_out_of_tree(self) -> bool:
        return True


def _silu_and_mul_fp32(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return (F.silu(x[..., :d].float()) * x[..., d:].float()).to(x.dtype)


def _build(op_cls, custom_ops: str):
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=[custom_ops]),
    )
    with set_current_vllm_config(config):
        return op_cls()


def test_unset_registers_nothing(monkeypatch, oot_registry):
    monkeypatch.setattr(envs, "VLLM_TPU_ACTIVATION_FP32", "")
    before = set(oot_registry)
    _register_fp32_activations()
    assert set(oot_registry) == before


@pytest.mark.parametrize(
    ("value", "expected"),
    [("silu", SWIGLU), ("all", None)],
)
def test_selected_families_are_registered(monkeypatch, oot_registry, value, expected):
    monkeypatch.setattr(envs, "VLLM_TPU_ACTIVATION_FP32", value)
    before = set(oot_registry)
    _register_fp32_activations()
    added = set(oot_registry) - before
    assert added == (_activation_custom_ops() if expected is None else expected)
    for name in added:
        assert issubclass(oot_registry[name], getattr(activation, name))


@pytest.mark.parametrize("custom_ops", ["all", "none"])
def test_dispatched_forward_runs_in_fp32(monkeypatch, oot_registry, custom_ops):
    monkeypatch.setattr(envs, "VLLM_TPU_ACTIVATION_FP32", "silu")
    monkeypatch.setattr(custom_op, "current_platform", _OutOfTreePlatform())
    _register_fp32_activations()
    op = _build(activation.SiluAndMul, custom_ops)
    assert isinstance(op, oot_registry["SiluAndMul"])
    x = torch.randn(4, 64, dtype=torch.bfloat16) * 40
    assert torch.equal(op(x), _silu_and_mul_fp32(x))


def test_registration_is_idempotent(monkeypatch, oot_registry):
    monkeypatch.setattr(envs, "VLLM_TPU_ACTIVATION_FP32", "silu")
    _register_fp32_activations()
    first = oot_registry["SiluAndMul"]
    _register_fp32_activations()
    assert oot_registry["SiluAndMul"] is first


def test_existing_override_is_left_alone(monkeypatch, oot_registry):
    class SomeoneElsesGELU(activation.GELU):
        pass

    oot_registry["GELU"] = SomeoneElsesGELU
    monkeypatch.setattr(envs, "VLLM_TPU_ACTIVATION_FP32", "all")
    _register_fp32_activations()
    assert oot_registry["GELU"] is SomeoneElsesGELU
