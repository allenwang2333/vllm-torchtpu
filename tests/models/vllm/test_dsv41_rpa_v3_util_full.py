"""100% statement and branch coverage test suite for ragged_paged_attention v3 util."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import util as rpa_util


def test_cdiv_and_align_to() -> None:
    """Verify cdiv and align_to arithmetic and zero-divisor assertions."""
    assert rpa_util.cdiv(0, 8) == 0
    assert rpa_util.cdiv(1, 8) == 1
    assert rpa_util.cdiv(8, 8) == 1
    assert rpa_util.cdiv(9, 8) == 2

    assert rpa_util.align_to(0, 128) == 0
    assert rpa_util.align_to(1, 128) == 128
    assert rpa_util.align_to(128, 128) == 128
    assert rpa_util.align_to(129, 128) == 256

    with pytest.raises(AssertionError):
        rpa_util.cdiv(4, 0)

    with pytest.raises(AssertionError):
        rpa_util.align_to(4, 0)


def test_dtype_bitwidth_and_packing() -> None:
    """Verify get_dtype_bitwidth and get_dtype_packing across dtypes."""
    rpa_util.get_dtype_bitwidth.cache_clear()
    rpa_util.get_dtype_packing.cache_clear()

    assert rpa_util.get_dtype_bitwidth(jnp.float32) == 32
    assert rpa_util.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert rpa_util.get_dtype_bitwidth(jnp.int8) == 8

    assert rpa_util.get_dtype_packing(jnp.float32) == 1
    assert rpa_util.get_dtype_packing(jnp.bfloat16) == 2
    assert rpa_util.get_dtype_packing(jnp.int8) == 4


def test_next_power_of_2() -> None:
    """Verify next_power_of_2 for 1, powers of 2, non-powers of 2, and invalid <= 0."""
    assert rpa_util.next_power_of_2(1) == 1
    assert rpa_util.next_power_of_2(2) == 2
    assert rpa_util.next_power_of_2(3) == 4
    assert rpa_util.next_power_of_2(4) == 4
    assert rpa_util.next_power_of_2(5) == 8
    assert rpa_util.next_power_of_2(127) == 128
    assert rpa_util.next_power_of_2(128) == 128
    assert rpa_util.next_power_of_2(129) == 256

    with pytest.raises(AssertionError):
        rpa_util.next_power_of_2(0)

    with pytest.raises(AssertionError):
        rpa_util.next_power_of_2(-3)


def test_get_tpu_version_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_tpu_version across CPU, TPU lite, v5e/v5p, v6e, TPU7x, and invalid."""
    rpa_util._parse_tpu_version.cache_clear()

    def _set_kind(kind: str) -> None:
        monkeypatch.setattr(
            rpa_util.jax,
            "devices",
            lambda: [SimpleNamespace(device_kind=kind)],
        )

    _set_kind("cpu")
    assert rpa_util.get_tpu_version() == -1

    _set_kind("TPU v5 lite")
    assert rpa_util.get_tpu_version() == 5

    _set_kind("TPU v5e")
    assert rpa_util.get_tpu_version() == 5

    _set_kind("TPU v5p")
    assert rpa_util.get_tpu_version() == 5

    _set_kind("TPU v6e")
    assert rpa_util.get_tpu_version() == 6

    _set_kind("TPU7x")
    assert rpa_util.get_tpu_version() == 7
    assert rpa_util.get_tpu_version() == 7

    _set_kind("TPU unknown9")
    with pytest.raises(AssertionError):
        rpa_util.get_tpu_version()
