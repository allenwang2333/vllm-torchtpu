# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scheduler wrapper pass-through and stride chunk emission."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnector,
    TPURaidenConnector,
    TPURaidenConnectorScheduler,
    _Stage3SendState,
    chunk_uuid,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


def _make_raiden_scheduler(
    *,
    is_producer: bool = True,
    block_size: int = 16,
    pcp_size: int = 4,
    interleave_size: int = 256,
) -> TPURaidenConnectorScheduler:
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 8192
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"), \
         patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100), \
         patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"):
        scheduler = TPURaidenConnectorScheduler(cfg)
    return scheduler


def test_chunk_uuid_determinism_and_properties() -> None:
    """Verifies chunk_uuid determinism, collision resistance, and range."""
    base_uuid = 987654321
    uuid_0 = chunk_uuid(base_uuid, 0)
    uuid_0_dup = chunk_uuid(base_uuid, 0)
    uuid_1 = chunk_uuid(base_uuid, 1)
    uuid_2 = chunk_uuid(base_uuid, 2)
    other_base_uuid = 987654322
    other_uuid_0 = chunk_uuid(other_base_uuid, 0)

    # 1. Determinism
    assert uuid_0 == uuid_0_dup

    # 2. Distinct chunk indices produce distinct UUIDs
    assert uuid_0 != uuid_1
    assert uuid_1 != uuid_2
    assert uuid_0 != other_uuid_0

    # 3. Fits in 50-bit positive integer space (safe for JSON and 64-bit int)
    assert 0 <= uuid_0 < (1 << 50)
    assert 0 <= uuid_1 < (1 << 50)
    assert 0 <= uuid_2 < (1 << 50)


def test_stage3_send_state_dataclass_defaults() -> None:
    """Verifies default values and field assignment in _Stage3SendState."""
    state = _Stage3SendState(
        base_uuid=12345,
        block_ids=[1, 2, 3],
    )
    assert state.base_uuid == 12345
    assert state.block_ids == [1, 2, 3]
    assert state.num_computed_tokens == 0
    assert state.tokens_sent == 0
    assert state.next_chunk_idx == 0
    assert state.total_prompt_tokens == 0


def test_scheduler_step_chunk_emission() -> None:
    """Simulate a 4096-token prefill executed in 1024-token steps on PCP=4.

    Verify that build_connector_meta() emits SendMeta([0, 1024), is_final=False)
    on step 1, SendMeta([1024, 2048), is_final=False) on step 2, etc.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-4096"
    base_uuid = 123456789
    block_ids = [100, 101, 102, 103, 104, 105, 106, 107]

    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=base_uuid,
        block_ids=block_ids,
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=4096,
    )

    # Step 1: 1024 tokens computed -> emits chunk [0, 1024)
    sched_out_1 = MagicMock(spec=SchedulerOutput)
    sched_out_1.num_scheduled_tokens = {req_id: 1024}

    meta_1 = scheduler.build_connector_meta(sched_out_1)
    assert req_id in meta_1.reqs_to_send
    send_meta_1 = meta_1.reqs_to_send[req_id]
    assert send_meta_1.uuid == chunk_uuid(base_uuid, 0)
    assert send_meta_1.token_base == 0
    assert send_meta_1.num_tokens == 1024
    assert send_meta_1.is_final is False
    assert send_meta_1.local_block_ids == block_ids
    assert send_meta_1.expiration_time > 0

    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 1024
    assert req_state.num_computed_tokens == 1024
    assert req_state.next_chunk_idx == 1

    # Step 2: 1024 tokens computed -> emits chunk [1024, 2048)
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_id: 1024}

    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_id in meta_2.reqs_to_send
    send_meta_2 = meta_2.reqs_to_send[req_id]
    assert send_meta_2.uuid == chunk_uuid(base_uuid, 1)
    assert send_meta_2.token_base == 1024
    assert send_meta_2.num_tokens == 1024
    assert send_meta_2.is_final is False
    assert send_meta_2.local_block_ids == block_ids

    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2048
    assert req_state.next_chunk_idx == 2

    # Step 3: 1024 tokens computed -> emits chunk [2048, 3072)
    sched_out_3 = MagicMock(spec=SchedulerOutput)
    sched_out_3.num_scheduled_tokens = {req_id: 1024}

    meta_3 = scheduler.build_connector_meta(sched_out_3)
    assert req_id in meta_3.reqs_to_send
    send_meta_3 = meta_3.reqs_to_send[req_id]
    assert send_meta_3.uuid == chunk_uuid(base_uuid, 2)
    assert send_meta_3.token_base == 2048
    assert send_meta_3.num_tokens == 1024
    assert send_meta_3.is_final is False

    assert req_state.tokens_sent == 3072
    assert req_state.num_computed_tokens == 3072
    assert req_state.next_chunk_idx == 3

    # Step 4: 1024 tokens computed -> emits chunk [3072, 4096)
    sched_out_4 = MagicMock(spec=SchedulerOutput)
    sched_out_4.num_scheduled_tokens = {req_id: 1024}

    meta_4 = scheduler.build_connector_meta(sched_out_4)
    assert req_id in meta_4.reqs_to_send
    send_meta_4 = meta_4.reqs_to_send[req_id]
    assert send_meta_4.uuid == chunk_uuid(base_uuid, 3)
    assert send_meta_4.token_base == 3072
    assert send_meta_4.num_tokens == 1024
    assert send_meta_4.is_final is False

    assert req_state.tokens_sent == 4096
    assert req_state.num_computed_tokens == 4096
    assert req_state.next_chunk_idx == 4

    # Step 5: No new tokens scheduled -> no chunks emitted
    sched_out_5 = MagicMock(spec=SchedulerOutput)
    sched_out_5.num_scheduled_tokens = {req_id: 0}
    meta_5 = scheduler.build_connector_meta(sched_out_5)
    assert len(meta_5.reqs_to_send) == 0


def test_scheduler_unaligned_remainder_carry() -> None:
    """Simulate prefill steps of 768 tokens.

    Verify that step 1 emits nothing (carries 768 tokens) and step 2 emits
    SendMeta([0, 1024)) once 1536 tokens are computed.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-768"
    base_uuid = 54321
    block_ids = [200, 201, 202, 203]

    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=base_uuid,
        block_ids=block_ids,
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=3072,
    )

    # Step 1: 768 tokens (< 1024 min) -> emits nothing, carries 768.
    sched_out_1 = MagicMock(spec=SchedulerOutput)
    sched_out_1.num_scheduled_tokens = {req_id: 768}

    meta_1 = scheduler.build_connector_meta(sched_out_1)
    assert len(meta_1.reqs_to_send) == 0

    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 0
    assert req_state.num_computed_tokens == 768
    assert req_state.next_chunk_idx == 0

    # Step 2: 1536 tokens computed -> emits [0, 1024), carries 512 remainder.
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_id: 768}

    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_id in meta_2.reqs_to_send
    send_meta_2 = meta_2.reqs_to_send[req_id]
    assert send_meta_2.uuid == chunk_uuid(base_uuid, 0)
    assert send_meta_2.token_base == 0
    assert send_meta_2.num_tokens == 1024
    assert send_meta_2.is_final is False
    assert send_meta_2.local_block_ids == block_ids

    assert req_state.tokens_sent == 1024
    assert req_state.num_computed_tokens == 1536
    assert req_state.next_chunk_idx == 1

    # Step 3: 2304 tokens computed -> emits [1024, 2048), carries 256 rem.
    sched_out_3 = MagicMock(spec=SchedulerOutput)
    sched_out_3.num_scheduled_tokens = {req_id: 768}

    meta_3 = scheduler.build_connector_meta(sched_out_3)
    assert req_id in meta_3.reqs_to_send
    send_meta_3 = meta_3.reqs_to_send[req_id]
    assert send_meta_3.uuid == chunk_uuid(base_uuid, 1)
    assert send_meta_3.token_base == 1024
    assert send_meta_3.num_tokens == 1024
    assert send_meta_3.is_final is False

    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2304
    assert req_state.next_chunk_idx == 2


def test_update_state_after_alloc_stage3_producer() -> None:
    """Verifies update_state_after_alloc initializes producer send state."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    req = MagicMock()
    req.request_id = "req-alloc-1"
    req.num_computed_tokens = 0
    req.num_prompt_tokens = 2048
    req.prompt_token_ids = [0] * 2048
    req.kv_transfer_params = None

    blocks = MagicMock()
    blocks.get_block_ids.return_value = [[50, 51, 52, 53]]

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True):
        scheduler.update_state_after_alloc(req, blocks, 0)

    assert "req-alloc-1" in scheduler._stage3_send_requests
    state = scheduler._stage3_send_requests["req-alloc-1"]
    assert state.block_ids == [50, 51, 52, 53]
    assert state.tokens_sent == 0
    assert state.next_chunk_idx == 0
    assert state.total_prompt_tokens == 2048


def test_update_state_after_alloc_existing_request_preserves_progress() -> None:
    """Verifies updating block_ids on existing request preserves progress."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-existing-1"

    # Pre-existing state with 1024 tokens already sent and chunk_idx=1
    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=99999,
        block_ids=[10, 11],
        num_computed_tokens=1024,
        tokens_sent=1024,
        next_chunk_idx=1,
        total_prompt_tokens=4096,
    )

    req = MagicMock()
    req.request_id = req_id
    req.num_computed_tokens = 1024
    req.num_prompt_tokens = 4096
    req.kv_transfer_params = None

    new_blocks = MagicMock()
    new_blocks.get_block_ids.return_value = [[10, 11, 12, 13]]

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True):
        scheduler.update_state_after_alloc(req, new_blocks, 0)

    state = scheduler._stage3_send_requests[req_id]
    assert state.block_ids == [10, 11, 12, 13]
    assert state.tokens_sent == 1024
    assert state.num_computed_tokens == 1024
    assert state.next_chunk_idx == 1
    assert state.base_uuid == 99999
    assert state.total_prompt_tokens == 4096


def test_update_state_after_alloc_block_format_variants() -> None:
    """Verifies flash-attention block IDs parse from various formats."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    # Variant A: Flat list of ints
    req_a = MagicMock()
    req_a.request_id = "req-flat-list"
    req_a.num_computed_tokens = 0
    req_a.num_prompt_tokens = 1024
    req_a.kv_transfer_params = None

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True):
        scheduler.update_state_after_alloc(req_a, [100, 101, 102], 0)

    state_a = scheduler._stage3_send_requests["req-flat-list"]
    assert state_a.block_ids == [100, 101, 102]

    # Variant B: Nested tuple of tuples (multi-group)
    req_b = MagicMock()
    req_b.request_id = "req-nested-tuple"
    req_b.num_computed_tokens = 0
    req_b.num_prompt_tokens = 1024
    req_b.kv_transfer_params = None

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True):
        scheduler.update_state_after_alloc(req_b, ((200, 201), (300, 301)), 0)

    state_b = scheduler._stage3_send_requests["req-nested-tuple"]
    assert state_b.block_ids == [200, 201]

    # Variant C: Prompt token fallback when num_prompt_tokens is None
    req_c = MagicMock()
    req_c.request_id = "req-token-fallback"
    req_c.num_computed_tokens = 0
    req_c.num_prompt_tokens = None
    req_c.prompt_token_ids = [1, 2, 3, 4, 5]
    req_c.kv_transfer_params = None

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True):
        scheduler.update_state_after_alloc(req_c, [300], 0)

    state_c = scheduler._stage3_send_requests["req-token-fallback"]
    assert state_c.total_prompt_tokens == 5


def test_update_state_after_alloc_non_stage3_or_non_producer() -> None:
    """Verifies update_state_after_alloc ignores calls when Stage-3 disabled."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    req = MagicMock()
    req.request_id = "req-no-stage3"
    req.num_computed_tokens = 0
    req.num_prompt_tokens = 1024
    req.kv_transfer_params = None

    with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=False):
        scheduler.update_state_after_alloc(req, [1, 2], 0)

    assert "req-no-stage3" not in scheduler._stage3_send_requests


def test_multi_request_interleaved_chunk_emission() -> None:
    """Verifies that multiple concurrent requests emit chunks independently."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    req_a = "req-A"
    req_b = "req-B"
    base_uuid_a = 111111
    base_uuid_b = 222222

    scheduler._stage3_send_requests[req_a] = _Stage3SendState(
        base_uuid=base_uuid_a,
        block_ids=[1, 2],
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=2048,
    )
    scheduler._stage3_send_requests[req_b] = _Stage3SendState(
        base_uuid=base_uuid_b,
        block_ids=[3, 4],
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=2048,
    )

    # Step 1: Req A computes 1024 (emits), Req B computes 512 (carries)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_a: 1024, req_b: 512}

    meta = scheduler.build_connector_meta(sched_out)
    assert req_a in meta.reqs_to_send
    assert req_b not in meta.reqs_to_send
    assert meta.reqs_to_send[req_a].num_tokens == 1024
    assert meta.reqs_to_send[req_a].token_base == 0

    # Step 2: Req A carries (512), Req B reaches 1024 and emits.
    sched_out_2 = MagicMock(spec=SchedulerOutput)
    sched_out_2.num_scheduled_tokens = {req_a: 512, req_b: 512}

    meta_2 = scheduler.build_connector_meta(sched_out_2)
    assert req_a not in meta_2.reqs_to_send
    assert req_b in meta_2.reqs_to_send
    assert meta_2.reqs_to_send[req_b].num_tokens == 1024
    assert meta_2.reqs_to_send[req_b].token_base == 0


def test_build_connector_meta_no_scheduled_tokens_or_empty() -> None:
    """Verifies build_connector_meta handles empty or None output safely."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    # 1. scheduler_output is None
    meta_none = scheduler.build_connector_meta(None)
    assert isinstance(meta_none, TPUConnectorMetadata)
    assert len(meta_none.reqs_to_send) == 0

    # 2. scheduler_output.num_scheduled_tokens is None
    sched_out_none_tokens = MagicMock(spec=SchedulerOutput)
    sched_out_none_tokens.num_scheduled_tokens = None
    meta_none_tokens = scheduler.build_connector_meta(sched_out_none_tokens)
    assert len(meta_none_tokens.reqs_to_send) == 0

    # 3. scheduler_output.num_scheduled_tokens is empty
    sched_out_empty = MagicMock(spec=SchedulerOutput)
    sched_out_empty.num_scheduled_tokens = {}
    meta_empty = scheduler.build_connector_meta(sched_out_empty)
    assert len(meta_empty.reqs_to_send) == 0


def test_build_connector_meta_untracked_request_ignored() -> None:
    """Verifies requests not in _stage3_send_requests are ignored."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )

    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {"unknown-req-id": 1024}

    meta = scheduler.build_connector_meta(sched_out)
    assert len(meta.reqs_to_send) == 0


def test_build_connector_meta_multi_stride_single_step() -> None:
    """Verifies multi-stride step emits a chunk covering full extent."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-multi-stride"
    base_uuid = 777777

    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=base_uuid,
        block_ids=[1, 2, 3, 4],
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=4096,
    )

    # 2048 tokens scheduled at once (2 strides of 1024)
    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 2048}

    meta = scheduler.build_connector_meta(sched_out)
    assert req_id in meta.reqs_to_send
    send_meta = meta.reqs_to_send[req_id]
    assert send_meta.token_base == 0
    assert send_meta.num_tokens == 2048
    assert send_meta.is_final is False
    assert send_meta.uuid == chunk_uuid(base_uuid, 0)

    req_state = scheduler._stage3_send_requests[req_id]
    assert req_state.tokens_sent == 2048
    assert req_state.num_computed_tokens == 2048
    assert req_state.next_chunk_idx == 1


def test_build_connector_meta_zero_min_transfer_tokens() -> None:
    """Verifies zero min_transfer_tokens emits computed tokens immediately."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    scheduler.pcp_size = 0
    scheduler.interleave_tokens = 0
    req_id = "req-zero-min"
    base_uuid = 888888

    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=base_uuid,
        block_ids=[10, 11],
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=500,
    )

    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 128}

    meta = scheduler.build_connector_meta(sched_out)
    assert req_id in meta.reqs_to_send
    send_meta = meta.reqs_to_send[req_id]
    assert send_meta.token_base == 0
    assert send_meta.num_tokens == 128
    assert send_meta.is_final is False


def test_build_connector_meta_consumer_mode_no_chunk_emission() -> None:
    """Verifies consumer mode does not emit chunks in build_connector_meta."""
    scheduler = _make_raiden_scheduler(
        is_producer=False, pcp_size=4, interleave_size=256
    )
    req_id = "req-consumer"

    scheduler._stage3_send_requests[req_id] = _Stage3SendState(
        base_uuid=999,
        block_ids=[1, 2],
        num_computed_tokens=0,
        tokens_sent=0,
        next_chunk_idx=0,
        total_prompt_tokens=2048,
    )

    sched_out = MagicMock(spec=SchedulerOutput)
    sched_out.num_scheduled_tokens = {req_id: 1024}

    meta = scheduler.build_connector_meta(sched_out)
    assert len(meta.reqs_to_send) == 0
    assert scheduler._stage3_send_requests[req_id].tokens_sent == 0


def test_connector_wrapper_pass_through() -> None:
    """Verifies that TPUConnector passes scheduler_output downstream."""
    mock_sched = MagicMock()
    mock_sched.build_connector_meta.return_value = TPUConnectorMetadata()

    connector = TPUConnector.__new__(TPUConnector)
    connector.connector_scheduler = mock_sched

    sched_out = MagicMock(spec=SchedulerOutput)
    result = connector.build_connector_meta(sched_out)

    mock_sched.build_connector_meta.assert_called_once_with(sched_out)
    assert isinstance(result, TPUConnectorMetadata)


def test_raiden_connector_wrapper_pass_through() -> None:
    """Verifies TPURaidenConnector passes scheduler_output to scheduler."""
    mock_sched = MagicMock()
    mock_sched.build_connector_meta.return_value = TPUConnectorMetadata()

    connector = TPURaidenConnector.__new__(TPURaidenConnector)
    connector.connector_scheduler = mock_sched

    sched_out = MagicMock(spec=SchedulerOutput)
    result = connector.build_connector_meta(sched_out)

    mock_sched.build_connector_meta.assert_called_once_with(sched_out)
    assert isinstance(result, TPUConnectorMetadata)
