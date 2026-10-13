# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for fused_moe.v2.token_parallel."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.kernels", ("vllm_torchtpu", "kernels")),
    ("vllm_torchtpu.kernels.fused_moe", ("vllm_torchtpu", "kernels", "fused_moe")),
    (
        "vllm_torchtpu.kernels.fused_moe.v2",
        ("vllm_torchtpu", "kernels", "fused_moe", "v2"),
    ),
):
    _pkg = sys.modules.get(_pkg_name)
    if _pkg is None:
        _pkg = types.ModuleType(_pkg_name)
        sys.modules[_pkg_name] = _pkg
    _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]

sys.modules.pop("vllm_torchtpu.kernels.fused_moe.v2.token_parallel", None)

from vllm_torchtpu.kernels.fused_moe.v2 import token_parallel  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_fused_moe_token_parallel_all_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.reload(token_parallel)
    cls = mod.TokenReplicaLayout

    # 1. Validation errors on ep, input_rows, row_alignment
    with pytest.raises(ValueError, match="must be positive"):
        cls.create(ep=0, input_rows=8)
    with pytest.raises(ValueError, match="must be positive"):
        cls.create(ep=4, input_rows=0)
    with pytest.raises(ValueError, match="power of two"):
        cls.create(ep=4, input_rows=8, row_alignment=0)
    with pytest.raises(ValueError, match="power of two"):
        cls.create(ep=4, input_rows=8, row_alignment=3)

    # 2. Validation errors on groups structure
    for bad_groups in (
        (),
        ((),),
        ((0, 1),),
        ((0, 1), (2,)),
        ((0, 0), (2, 3)),
        ((-1, 1), (2, 3)),
        ((0, 4), (2, 3)),
    ):
        with pytest.raises(ValueError, match="token replica groups must partition"):
            cls.create(ep=4, input_rows=8, groups=bad_groups)

    # 3. Default groups=None singleton path and explicit singleton groups
    l_default = cls.create(ep=4, input_rows=7, groups=None, row_alignment=8)
    assert l_default.groups == ((0,), (1,), (2,), (3,))
    assert l_default.slots == (0, 0, 0, 0)
    assert l_default.local_rows == 7

    l_explicit_single = cls.create(
        ep=4, input_rows=7, groups=((0,), (1,), (2,), (3,)), row_alignment=8
    )
    assert l_explicit_single.local_rows == 7

    x7 = jnp.arange(14, dtype=jnp.float32).reshape(7, 2)
    assert l_default.split(x7, jnp.int32(0)) is x7
    assert l_default.restore(x7, "d") is x7

    # 4. Non-singleton groups: aligned (no padding) and unaligned (with padding)
    groups = ((3, 1, 0, 2), (7, 5, 4, 6))
    l_aligned = cls.create(ep=8, input_rows=16, groups=groups, row_alignment=4)
    assert l_aligned.local_rows == 4
    x16 = jnp.arange(32, dtype=jnp.float32).reshape(16, 2)
    s_aligned = l_aligned.split(x16, jnp.int32(1))
    np.testing.assert_array_equal(s_aligned, x16[4:8])

    l_padded = cls.create(ep=8, input_rows=13, groups=groups, row_alignment=4)
    assert l_padded.local_rows == 4
    x13 = jnp.arange(26, dtype=jnp.float32).reshape(13, 2)
    s_padded = l_padded.split(x13, jnp.int32(2), fill_value=-1.0)
    expected_last = np.array([[24.0, 25.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]])
    np.testing.assert_array_equal(s_padded, expected_last)

    # 5. restore with exact input_rows and with padded gathered output
    monkeypatch.setattr(mod.lax, "all_gather", lambda rows, *a, **kw: x16)
    r_exact = l_aligned.restore(s_aligned, "d")
    assert r_exact is x16

    r_sliced = l_padded.restore(s_padded, "d")
    assert r_sliced.shape == (13, 2)
    np.testing.assert_array_equal(r_sliced, x16[:13])
