"""Check that prepared expert groups can serve different EP placements."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


def _selection():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/vllm_torchtpu/layers/adapter/quantization/static_dsv41_moe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "static_dsv41_moe_groups", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    executor = ModuleType("vllm.model_executor")
    executor.__path__ = []
    layers = ModuleType("vllm.model_executor.layers")
    layers.__path__ = []
    fused_moe = ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe.RoutedExperts = type("RoutedExperts", (), {})
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm,
            "vllm.model_executor": executor,
            "vllm.model_executor.layers": layers,
            "vllm.model_executor.layers.fused_moe": fused_moe,
        },
    ):
        spec.loader.exec_module(module)
    return module._select_grouped_experts


def test_grouped_storage_selects_runtime_experts_in_local_order() -> None:
    select = _selection()
    groups = {
        group: {
            name: torch.arange(group * 48, (group + 1) * 48)
            for name in ("w13", "w2", "s13", "s2")
        }
        for group in (0, 1)
    }
    mixed = select(groups, [49, 0, 48, 2])
    for tensor in mixed.values():
        assert tensor.tolist() == [49, 0, 48, 2]
    contiguous = select({0: groups[0]}, list(range(48)))
    assert contiguous is groups[0]
