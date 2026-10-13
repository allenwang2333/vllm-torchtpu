"""100% coverage test suite for all_gather_matmul_tuned_block_sizes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.collectives import (
    all_gather_matmul_tuned_block_sizes as agm_tbs,
)


def test_get_tpu_version_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_tpu_version across CPU, TPU lite, v5e/v5p, v6e, TPU7x, and invalid."""

    def _check_kind(kind: str) -> int:
        agm_tbs.get_tpu_version.cache_clear()
        monkeypatch.setattr(
            agm_tbs.jax,
            "devices",
            lambda: [SimpleNamespace(device_kind=kind)],
        )
        return agm_tbs.get_tpu_version()

    assert _check_kind("cpu") == -1
    assert _check_kind("TPU v5 lite") == 5
    assert _check_kind("TPU v5e") == 5
    assert _check_kind("TPU v5p") == 5
    assert _check_kind("TPU v6e") == 6
    assert _check_kind("TPU7x") == 7
    assert agm_tbs.get_tpu_version() == 7

    agm_tbs.get_tpu_version.cache_clear()
    monkeypatch.setattr(
        agm_tbs.jax,
        "devices",
        lambda: [SimpleNamespace(device_kind="TPU invalid9")],
    )
    with pytest.raises(AssertionError):
        agm_tbs.get_tpu_version()
    agm_tbs.get_tpu_version.cache_clear()


def test_get_key_and_get_tuned_block_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_key and get_tuned_block_sizes for hit and miss shapes."""
    agm_tbs.get_tpu_version.cache_clear()
    monkeypatch.setattr(
        agm_tbs.jax,
        "devices",
        lambda: [SimpleNamespace(device_kind="TPU v6e")],
    )

    assert agm_tbs.get_key(1024, 51200, 5120, "bfloat16", 8) == (
        6,
        1024,
        51200,
        5120,
        "bfloat16",
        8,
    )
    assert agm_tbs.get_tuned_block_sizes(1024, 51200, 5120, "bfloat16", 8) == (
        6400,
        2560,
    )
    assert agm_tbs.get_tuned_block_sizes(1024, 57344, 8192, "bfloat16", 8) == (
        7168,
        8192,
    )
    assert agm_tbs.get_tuned_block_sizes(128, 128, 128, "bfloat16", 8) == (
        None,
        None,
    )
    agm_tbs.get_tpu_version.cache_clear()
