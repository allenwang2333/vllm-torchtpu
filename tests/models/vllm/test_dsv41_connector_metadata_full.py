# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.distributed.kv_transfer.connector_metadata."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "distributed" / "kv_transfer").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.distributed", _SRC_ROOT / "distributed"),
        (
            "vllm_torchtpu.distributed.kv_transfer",
            _SRC_ROOT / "distributed" / "kv_transfer",
        ),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

if "vllm.distributed.kv_transfer.kv_connector.v1.base" not in sys.modules:
    _vllm_mod = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    _vllm_dist = sys.modules.setdefault(
        "vllm.distributed", types.ModuleType("vllm.distributed")
    )
    _vllm_kvt = sys.modules.setdefault(
        "vllm.distributed.kv_transfer",
        types.ModuleType("vllm.distributed.kv_transfer"),
    )
    _vllm_kvc = sys.modules.setdefault(
        "vllm.distributed.kv_transfer.kv_connector",
        types.ModuleType("vllm.distributed.kv_transfer.kv_connector"),
    )
    _vllm_v1 = sys.modules.setdefault(
        "vllm.distributed.kv_transfer.kv_connector.v1",
        types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1"),
    )
    _vllm_base = sys.modules.setdefault(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base"),
    )

    class _KVConnectorMetadata:
        pass

    _vllm_base.KVConnectorMetadata = _KVConnectorMetadata  # type: ignore[attr-defined]

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (  # noqa: E402
    LoadMeta,
    ReqId,
    SendMeta,
    TPUConnectorMetadata,
)


def test_send_meta_defaults_and_slots() -> None:
    """Verify SendMeta defaults, explicit fields, and slotted storage."""
    assert ReqId is str
    default_send = SendMeta(
        uuid=101,
        local_block_ids=[0, 1, 2],
        expiration_time=123.5,
    )
    assert default_send.uuid == 101
    assert default_send.local_block_ids == [0, 1, 2]
    assert default_send.expiration_time == 123.5
    assert default_send.num_tokens is None
    assert default_send.mamba_state_block_ids is None
    assert default_send.group_block_ids is None
    assert hasattr(SendMeta, "__slots__")
    with pytest.raises(AttributeError):
        default_send.unexpected_attr = 1  # type: ignore[attr-defined]

    full_send = SendMeta(
        uuid=202,
        local_block_ids=[3, 4],
        expiration_time=456.0,
        num_tokens=64,
        mamba_state_block_ids=[9],
        group_block_ids=[[3, 4], [7, 8]],
    )
    assert full_send.num_tokens == 64
    assert full_send.mamba_state_block_ids == [9]
    assert full_send.group_block_ids == [[3, 4], [7, 8]]


def test_load_meta_defaults_and_slots() -> None:
    """Verify LoadMeta defaults, explicit fields, and slotted storage."""
    default_load = LoadMeta(
        uuid=303,
        local_block_ids=[5, 6],
        remote_block_ids=[1, 2],
        remote_host="10.0.0.1",
        remote_port=5555,
    )
    assert default_load.uuid == 303
    assert default_load.local_block_ids == [5, 6]
    assert default_load.remote_block_ids == [1, 2]
    assert default_load.remote_host == "10.0.0.1"
    assert default_load.remote_port == 5555
    assert default_load.remote_side_channel_port is None
    assert default_load.report_completion is True
    assert default_load.fail_only is False
    assert hasattr(LoadMeta, "__slots__")
    with pytest.raises(AttributeError):
        default_load.unexpected_attr = 1  # type: ignore[attr-defined]

    multihost_load = LoadMeta(
        uuid=404,
        local_block_ids=None,
        remote_block_ids=None,
        remote_host=["10.0.0.1", "10.0.0.2"],
        remote_port=[5555, 5556],
        remote_side_channel_port=6000,
        report_completion=False,
        fail_only=True,
    )
    assert multihost_load.local_block_ids is None
    assert multihost_load.remote_block_ids is None
    assert multihost_load.remote_host == ["10.0.0.1", "10.0.0.2"]
    assert multihost_load.remote_port == [5555, 5556]
    assert multihost_load.remote_side_channel_port == 6000
    assert multihost_load.report_completion is False
    assert multihost_load.fail_only is True


def test_tpu_connector_metadata_init_and_isolation() -> None:
    """Verify TPUConnectorMetadata default dicts and explicit dict passing."""
    meta_a = TPUConnectorMetadata()
    meta_b = TPUConnectorMetadata()
    assert meta_a.reqs_to_send == {}
    assert meta_a.reqs_to_load == {}
    assert meta_a.reqs_to_send is not meta_b.reqs_to_send
    assert meta_a.reqs_to_load is not meta_b.reqs_to_load
    assert hasattr(TPUConnectorMetadata, "__slots__")

    send_entry = SendMeta(uuid=1, local_block_ids=[0], expiration_time=1.0)
    load_entry = LoadMeta(
        uuid=2,
        local_block_ids=[1],
        remote_block_ids=[0],
        remote_host="127.0.0.1",
        remote_port=5555,
    )
    send_map = {"req-send": send_entry}
    load_map = {"req-load": load_entry}
    meta_custom = TPUConnectorMetadata(
        reqs_to_send=send_map,
        reqs_to_load=load_map,
    )
    assert meta_custom.reqs_to_send is send_map
    assert meta_custom.reqs_to_load is load_map
    assert meta_custom.reqs_to_send["req-send"] is send_entry
    assert meta_custom.reqs_to_load["req-load"] is load_entry
