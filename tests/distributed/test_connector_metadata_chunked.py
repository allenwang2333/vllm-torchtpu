# SPDX-License-Identifier: Apache-2.0
"""Unit tests for chunked transfer metadata schema extensions."""

import dataclasses
import json
import pickle

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta,
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    _Stage3RegisteredSend,
)


def test_send_meta_defaults() -> None:
    """Verifies SendMeta defaults for legacy compatibility."""
    meta = SendMeta(
        uuid=12345,
        local_block_ids=[10, 11, 12],
        expiration_time=100.0,
    )
    assert meta.uuid == 12345
    assert meta.local_block_ids == [10, 11, 12]
    assert meta.expiration_time == 100.0
    assert meta.num_tokens is None
    assert meta.mamba_state_block_ids is None
    assert meta.token_base == 0
    assert meta.is_final is True


def test_send_meta_subchunk_fields() -> None:
    """Verifies SendMeta instantiation with sub-chunk coordinates."""
    meta = SendMeta(
        uuid=67890,
        local_block_ids=[20, 21],
        expiration_time=200.0,
        num_tokens=1024,
        mamba_state_block_ids=None,
        token_base=1024,
        is_final=False,
    )
    assert meta.uuid == 67890
    assert meta.local_block_ids == [20, 21]
    assert meta.expiration_time == 200.0
    assert meta.num_tokens == 1024
    assert meta.mamba_state_block_ids is None
    assert meta.token_base == 1024
    assert meta.is_final is False


def test_load_meta_defaults_and_subchunk_fields() -> None:
    """Verifies LoadMeta defaults and sub-chunk coordinates."""
    # Default legacy instantiation
    meta_default = LoadMeta(
        uuid=11111,
        local_block_ids=[1, 2],
        remote_block_ids=[3, 4],
        remote_host="10.0.0.1",
        remote_port=5000,
    )
    assert meta_default.uuid == 11111
    assert meta_default.local_block_ids == [1, 2]
    assert meta_default.remote_block_ids == [3, 4]
    assert meta_default.remote_host == "10.0.0.1"
    assert meta_default.remote_port == 5000
    assert meta_default.remote_side_channel_port is None
    assert meta_default.report_completion is True
    assert meta_default.token_base == 0
    assert meta_default.is_final is True

    # Subchunk instantiation
    meta_chunked = LoadMeta(
        uuid=22222,
        local_block_ids=[5, 6],
        remote_block_ids=[7, 8],
        remote_host="10.0.0.2",
        remote_port=5001,
        remote_side_channel_port=6000,
        report_completion=False,
        token_base=2048,
        is_final=False,
    )
    assert meta_chunked.token_base == 2048
    assert meta_chunked.is_final is False
    assert meta_chunked.report_completion is False


def test_stage3_registered_send_schema() -> None:
    """Verifies _Stage3RegisteredSend chunk coordinates and defaults."""
    # Default legacy registration
    reg_default = _Stage3RegisteredSend(
        uuid=33333,
        local_block_ids=(100, 101),
        num_tokens=4096,
        expiration_time=500.0,
    )
    assert reg_default.uuid == 33333
    assert reg_default.local_block_ids == (100, 101)
    assert reg_default.num_tokens == 4096
    assert reg_default.expiration_time == 500.0
    assert reg_default.token_base == 0
    assert reg_default.is_final is True

    # Chunked registration
    reg_chunked = _Stage3RegisteredSend(
        uuid=44444,
        local_block_ids=(102, 103),
        num_tokens=1024,
        expiration_time=600.0,
        token_base=1024,
        is_final=False,
    )
    assert reg_chunked.token_base == 1024
    assert reg_chunked.is_final is False


def test_metadata_serialization_roundtrip() -> None:
    """Verifies serialization roundtrip for TPUConnectorMetadata."""
    send_meta_1 = SendMeta(
        uuid=101,
        local_block_ids=[1, 2],
        expiration_time=150.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )
    send_meta_2 = SendMeta(
        uuid=102,
        local_block_ids=[3, 4],
        expiration_time=160.0,
        num_tokens=1024,
        token_base=1024,
        is_final=True,
    )
    load_meta_1 = LoadMeta(
        uuid=201,
        local_block_ids=[10, 11],
        remote_block_ids=[1, 2],
        remote_host="10.0.0.5",
        remote_port=8000,
        token_base=0,
        is_final=False,
    )
    load_meta_2 = LoadMeta(
        uuid=202,
        local_block_ids=[12, 13],
        remote_block_ids=[3, 4],
        remote_host="10.0.0.5",
        remote_port=8000,
        token_base=1024,
        is_final=True,
    )

    tpu_meta = TPUConnectorMetadata(
        reqs_to_send={
            "req-1-chunk-0": send_meta_1,
            "req-1-chunk-1": send_meta_2,
        },
        reqs_to_load={
            "req-2-chunk-0": load_meta_1,
            "req-2-chunk-1": load_meta_2,
        },
    )

    # 1. Dataclass asdict & JSON serialization
    dict_data = dataclasses.asdict(tpu_meta)
    json_str = json.dumps(dict_data)
    recovered_dict = json.loads(json_str)

    send_dict = recovered_dict["reqs_to_send"]
    load_dict = recovered_dict["reqs_to_load"]
    assert send_dict["req-1-chunk-0"]["token_base"] == 0
    assert send_dict["req-1-chunk-0"]["is_final"] is False
    assert send_dict["req-1-chunk-1"]["token_base"] == 1024
    assert send_dict["req-1-chunk-1"]["is_final"] is True

    assert load_dict["req-2-chunk-0"]["token_base"] == 0
    assert load_dict["req-2-chunk-0"]["is_final"] is False
    assert load_dict["req-2-chunk-1"]["token_base"] == 1024
    assert load_dict["req-2-chunk-1"]["is_final"] is True

    # Reconstruct objects from dictionary
    reconstructed_sends = {
        k: SendMeta(**v) for k, v in send_dict.items()
    }
    reconstructed_loads = {
        k: LoadMeta(**v) for k, v in load_dict.items()
    }
    reconstructed_meta = TPUConnectorMetadata(
        reqs_to_send=reconstructed_sends,
        reqs_to_load=reconstructed_loads,
    )

    assert reconstructed_meta.reqs_to_send["req-1-chunk-0"].token_base == 0
    assert reconstructed_meta.reqs_to_send["req-1-chunk-0"].is_final is False
    assert reconstructed_meta.reqs_to_send["req-1-chunk-1"].token_base == 1024
    assert reconstructed_meta.reqs_to_send["req-1-chunk-1"].is_final is True
    assert reconstructed_meta.reqs_to_load["req-2-chunk-0"].token_base == 0
    assert reconstructed_meta.reqs_to_load["req-2-chunk-0"].is_final is False
    assert reconstructed_meta.reqs_to_load["req-2-chunk-1"].token_base == 1024
    assert reconstructed_meta.reqs_to_load["req-2-chunk-1"].is_final is True

    # 2. Pickle roundtrip
    pickled = pickle.dumps(tpu_meta)
    unpickled_meta = pickle.loads(pickled)
    assert unpickled_meta.reqs_to_send["req-1-chunk-0"].token_base == 0
    assert unpickled_meta.reqs_to_send["req-1-chunk-0"].is_final is False
    assert unpickled_meta.reqs_to_send["req-1-chunk-1"].token_base == 1024
    assert unpickled_meta.reqs_to_send["req-1-chunk-1"].is_final is True
    assert unpickled_meta.reqs_to_load["req-2-chunk-0"].token_base == 0
    assert unpickled_meta.reqs_to_load["req-2-chunk-0"].is_final is False
    assert unpickled_meta.reqs_to_load["req-2-chunk-1"].token_base == 1024
    assert unpickled_meta.reqs_to_load["req-2-chunk-1"].is_final is True
