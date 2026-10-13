# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for upstream_bindings.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.layers", ("vllm_torchtpu", "layers")),
    ("vllm_torchtpu.layers.adapter", ("vllm_torchtpu", "layers", "adapter")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg

from vllm_torchtpu.layers.adapter import upstream_bindings  # noqa: E402


def _upstream_op(x: int) -> int:
    return x + 1


def _upstream_fn(x: int, scale: int = 2, *, bias: int = 3) -> int:
    return _upstream_op(x) * scale + bias


def test_upstream_bindings_all_branches() -> None:
    """Exercise 100% of statements and branches in upstream_bindings.py."""
    target = SimpleNamespace(a=10, b=20, c=30)

    # 1. Empty replacements branch
    with upstream_bindings.device_hooks(target):
        assert target.a == 10
        assert target.b == 20
    assert target.a == 10

    # 2. Non-empty replacements + normal exit + nested re-entrant lock
    with upstream_bindings.device_hooks(target, a=100, b=200):
        assert target.a == 100
        assert target.b == 200
        with upstream_bindings.device_hooks(target, a=300):
            assert target.a == 300
        assert target.a == 100
    assert target.a == 10
    assert target.b == 20

    # 3. Exception inside context restores previous attributes
    with (
        pytest.raises(RuntimeError, match="hook_error"),
        upstream_bindings.device_hooks(target, a=999, c=888),
    ):
        assert target.a == 999
        assert target.c == 888
        raise RuntimeError("hook_error")
    assert target.a == 10
    assert target.c == 30

    # 4. bind_device_ops creates isolated module namespace and preserves metadata
    bound = upstream_bindings.bind_device_ops(
        _upstream_fn,
        _upstream_op=lambda x: x + 10,
    )
    assert _upstream_fn(4) == (4 + 1) * 2 + 3
    assert bound(4) == (4 + 10) * 2 + 3
    assert bound(4, 5, bias=7) == (4 + 10) * 5 + 7
    assert bound.__name__ == _upstream_fn.__name__
    assert bound.__qualname__ == _upstream_fn.__qualname__
    assert bound.__kwdefaults__ == {"bias": 3}
    assert bound.__annotations__ == _upstream_fn.__annotations__
    expected_mod = f"{upstream_bindings.__name__}.{_upstream_fn.__qualname__}"
    assert sys.modules[expected_mod]._upstream_fn is bound
