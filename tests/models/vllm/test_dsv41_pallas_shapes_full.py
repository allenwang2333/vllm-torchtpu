# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.distributed.pallas_shapes."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "distributed").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.distributed", _SRC_ROOT / "distributed"),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

from vllm_torchtpu.distributed import pallas_shapes  # noqa: E402


def test_get_global_and_local_shape_passthrough_and_single_axis() -> None:
    """Verify None mesh, empty partition_spec, and single-axis conversion."""
    mesh = SimpleNamespace(shape={"tp": 8, "pcp": 4})
    assert pallas_shapes.get_global_shape((4, 16), None, ("tp",)) == (4, 16)
    assert pallas_shapes.get_local_shape([4, 16], None, ("tp",)) == (4, 16)
    assert pallas_shapes.get_global_shape([4, 16], mesh, None) == (4, 16)
    assert pallas_shapes.get_local_shape((4, 16), mesh, ()) == (4, 16)

    assert pallas_shapes.get_global_shape((2, 16, 64), mesh, (None, "tp")) == (
        2,
        128,
        64,
    )
    assert pallas_shapes.get_local_shape((2, 128, 64), mesh, (None, "tp")) == (
        2,
        16,
        64,
    )


def test_compound_axis_and_longer_spec_and_divisibility_error() -> None:
    """Verify multi-axis tuple factors, spec longer than shape, and error branch."""
    mesh = SimpleNamespace(shape={"tp": 8, "pcp": 4})
    assert pallas_shapes.get_global_shape(
        (3, 8, 16), mesh, (None, ("tp", "pcp"), None, "tp")
    ) == (3, 256, 16)
    assert pallas_shapes.get_local_shape(
        (3, 256, 16), mesh, (None, ("tp", "pcp"), None, "tp")
    ) == (3, 8, 16)
    assert pallas_shapes.get_global_shape((3, 8), mesh, ("pcp", "tp", "pcp")) == (
        12,
        64,
    )

    with pytest.raises(ValueError, match="not divisible by mesh factor 32"):
        pallas_shapes.get_local_shape((3, 63, 16), mesh, (None, ("tp", "pcp")))


def test_install_pallas_shape_conversion() -> None:
    """Verify install_pallas_shape_conversion patches torch_tpu pallas module."""
    fake_pallas = SimpleNamespace(get_global_shape=None, get_local_shape=None)
    fake_pkg = types.ModuleType("torch_tpu._internal.pallas")
    fake_pkg.pallas = fake_pallas  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "torch_tpu": types.ModuleType("torch_tpu"),
            "torch_tpu._internal": types.ModuleType("torch_tpu._internal"),
            "torch_tpu._internal.pallas": fake_pkg,
        },
    ):
        pallas_shapes.install_pallas_shape_conversion()
    assert fake_pallas.get_global_shape is pallas_shapes.get_global_shape
    assert fake_pallas.get_local_shape is pallas_shapes.get_local_shape
