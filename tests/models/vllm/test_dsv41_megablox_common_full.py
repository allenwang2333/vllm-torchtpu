"""100% coverage test suite for vllm_torchtpu.kernels.megablox.common."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.megablox import common as mb_common


def test_is_tpu_tpu_kind_and_tpu_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify is_tpu, tpu_kind, and tpu_generation across TPU and non-TPU kinds."""
    mb_common._parse_tpu_generation.cache_clear()

    def _set_kind(kind: str) -> None:
        monkeypatch.setattr(
            mb_common.jax,
            "devices",
            lambda: [SimpleNamespace(device_kind=kind)],
        )

    _set_kind("TPU v5p")
    assert mb_common.is_tpu() is True
    assert mb_common.tpu_kind() == "TPU v5p"
    assert mb_common.tpu_generation() == 5

    _set_kind("TPU v6e")
    assert mb_common.is_tpu() is True
    assert mb_common.tpu_kind() == "TPU v6e"
    assert mb_common.tpu_generation() == 6

    _set_kind("TPU7x")
    assert mb_common.is_tpu() is True
    assert mb_common.tpu_kind() == "TPU7x"
    assert mb_common.tpu_generation() == 7
    assert mb_common.tpu_generation() == 7

    _set_kind("cpu")
    assert mb_common.is_tpu() is False
    assert mb_common.tpu_kind() == "cpu"
    with pytest.raises(NotImplementedError, match="only TPU devices are supported"):
        mb_common.tpu_generation()


def test_assert_is_supported_dtype() -> None:
    """Verify assert_is_supported_dtype accepts supported dtypes and rejects others."""
    for dt in (
        jnp.bfloat16,
        jnp.float32,
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
        jnp.int8,
        jnp.int4,
        jnp.float4_e2m1fn,
        jnp.uint4,
    ):
        mb_common.assert_is_supported_dtype(dt)
        mb_common.assert_is_supported_dtype(jnp.dtype(dt))

    with pytest.raises(ValueError, match="No support for"):
        mb_common.assert_is_supported_dtype(jnp.dtype(jnp.float64))
