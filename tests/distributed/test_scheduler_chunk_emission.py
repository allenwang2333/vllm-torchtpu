# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scheduler wrapper pass-through and stride chunk emission."""

from collections.abc import Iterator
import contextlib
from unittest.mock import MagicMock, patch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
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
    chunk_transfer_window: int | None = None,
) -> TPURaidenConnectorScheduler:
    """Creates a mock-configured TPURaidenConnectorScheduler for testing."""
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 8192
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
        patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"),
    ):
        scheduler = TPURaidenConnectorScheduler(cfg)
    return scheduler


@contextlib.contextmanager
def _patch_raiden_stage3_env(
    *,
    base_uuid: int = 12345,
    parallelism: int = 4,
    timeout_s: float = 30.0,
    controller_address: str = "127.0.0.1:27000",
    job_name: str = "prefill",
    engine_id: str = "producer-0",
) -> Iterator[None]:
    """Patches Stage-3 Raiden connector environment variables and utilities."""
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
            controller_address,
        ),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME", job_name),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", engine_id),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
        ),
        patch(
            f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
            return_value=timeout_s,
        ),
        patch(f"{_MOD}.get_uuid", return_value=base_uuid),
    ):
        yield


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
        is_producer=True,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=4,
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
        is_producer=True,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=8,
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


# ===========================================================================
# Step 2.3 — Short-Prompt Bypass & Terminal Remainder Emission Tests
# ===========================================================================


def test_short_prompt_bypass() -> None:
    """Verifies 512-token prompt bypasses intermediate chunking.

    Under PCP4 with stride 1024, a 512-token prompt is shorter than the minimum
    transfer threshold. build_connector_meta() must emit zero intermediate
    chunks, and request_finished() must emit a single terminal SendMeta
    descriptor covering [0, 512) with is_final=True.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-short-512"
    base_uuid = 555555
    # PCP4 with block_size=16 -> scheduler block = 64 tokens.
    # 512 tokens -> (512 + 63) // 64 = 8 scheduler blocks.
    block_ids = list(range(100, 108))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 512
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # 1. build_connector_meta during execution step: emits 0 chunks
        sched_out = MagicMock(spec=SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: 512}

        meta_step = scheduler.build_connector_meta(sched_out)
        assert req_id not in meta_step.reqs_to_send
        assert len(meta_step.reqs_to_send) == 0

        # Verify state cursor untouched
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0

        # 2. request_finished emits single terminal chunk
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        send_meta = meta_fin.reqs_to_send[req_id]
        assert send_meta.token_base == 0
        assert send_meta.num_tokens == 512
        assert send_meta.is_final is True
        assert send_meta.uuid == chunk_uuid(base_uuid, 0)
        assert send_meta.local_block_ids == block_ids

        # 3. Verify cleanup of request tracking state
        assert req_id not in scheduler._stage3_send_requests


def test_terminal_remainder_emission() -> None:
    """Verifies intermediate chunks and terminal remainder on 3500-token prompt.

    Intermediate execution steps emit chunks for [0, 1024), [1024, 2048),
    and [2048, 3072). Upon completion, request_finished() emits the terminal
    remainder [3072, 3500) marked with is_final=True.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=4,
    )
    req_id = "req-long-3500"
    base_uuid = 999111
    # 3500 tokens on PCP4 (block_size=16): (3500 + 63) // 64 = 55 blocks.
    block_ids = list(range(100, 155))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Step 1: 1024 tokens -> [0, 1024), chunk_idx 0
        sched_out_1 = MagicMock(spec=SchedulerOutput)
        sched_out_1.num_scheduled_tokens = {req_id: 1024}
        meta_1 = scheduler.build_connector_meta(sched_out_1)
        assert req_id in meta_1.reqs_to_send
        s1 = meta_1.reqs_to_send[req_id]
        assert s1.token_base == 0
        assert s1.num_tokens == 1024
        assert s1.is_final is False
        assert s1.uuid == chunk_uuid(base_uuid, 0)

        # Step 2: 1024 tokens -> [1024, 2048), chunk_idx 1
        sched_out_2 = MagicMock(spec=SchedulerOutput)
        sched_out_2.num_scheduled_tokens = {req_id: 1024}
        meta_2 = scheduler.build_connector_meta(sched_out_2)
        assert req_id in meta_2.reqs_to_send
        s2 = meta_2.reqs_to_send[req_id]
        assert s2.token_base == 1024
        assert s2.num_tokens == 1024
        assert s2.is_final is False
        assert s2.uuid == chunk_uuid(base_uuid, 1)

        # Step 3: 1024 tokens -> [2048, 3072), chunk_idx 2
        sched_out_3 = MagicMock(spec=SchedulerOutput)
        sched_out_3.num_scheduled_tokens = {req_id: 1024}
        meta_3 = scheduler.build_connector_meta(sched_out_3)
        assert req_id in meta_3.reqs_to_send
        s3 = meta_3.reqs_to_send[req_id]
        assert s3.token_base == 2048
        assert s3.num_tokens == 1024
        assert s3.is_final is False
        assert s3.uuid == chunk_uuid(base_uuid, 2)

        # Step 4: 428 tokens scheduled (3500 total) -> carries, no chunk emitted
        sched_out_4 = MagicMock(spec=SchedulerOutput)
        sched_out_4.num_scheduled_tokens = {req_id: 428}
        meta_4 = scheduler.build_connector_meta(sched_out_4)
        assert req_id not in meta_4.reqs_to_send

        # Prefill finished: request_finished emits remainder [3072, 3500)
        req.num_computed_tokens = 3500
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 3072
        assert s_fin.num_tokens == 428
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 3)
        assert s_fin.local_block_ids == block_ids

        # Verify cleanup of request tracking state
        assert req_id not in scheduler._stage3_send_requests


def test_fail_safe_fallback_degradation() -> None:
    """Verifies monolithic fallback when scheduler emits no intermediate chunks.

    If intermediate chunk emission is skipped (e.g. window capacity limit),
    request_finished() must emit the full range [0, 3500) with is_final=True,
    matching baseline monolithic behavior.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-fallback-3500"
    base_uuid = 888222
    block_ids = list(range(100, 155))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 3500
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Scheduler fails to emit intermediate chunks; tokens_sent remains 0.
        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0

        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 3500
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 0)
        assert s_fin.local_block_ids == block_ids

        # Verify cleanup of request tracking state
        assert req_id not in scheduler._stage3_send_requests


def test_request_finished_duplicate_idempotency_after_chunking() -> None:
    """Verifies duplicate request_finished calls return cached parameters."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-dedup-test"
    base_uuid = 333444
    block_ids = list(range(100, 108))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 512
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        delay1, params1 = scheduler.request_finished(req, block_ids)
        meta1 = scheduler.build_connector_meta()
        delay2, params2 = scheduler.request_finished(req, block_ids)
        meta2 = scheduler.build_connector_meta()

        assert delay1 is True and delay2 is True
        assert params1 == params2
        assert req_id in meta1.reqs_to_send
        assert len(meta2.reqs_to_send) == 0


def test_terminal_emission_exact_stride_aligned_all_tokens_sent() -> None:
    """Verifies request_finished() when all prompt tokens were already emitted.

    For an exact stride multiple (e.g. 4096 tokens on PCP=4), all tokens are
    emitted across intermediate steps. request_finished() must clean up
    the send state, return (True, params), and NOT emit an empty remainder into
    reqs_to_send.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=4,
    )
    req_id = "req-exact-4096"
    base_uuid = 777111
    block_ids = list(range(100, 164))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 4096
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Emit 4 chunks covering all 4096 tokens
        for step in range(4):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: 1024}
            meta = scheduler.build_connector_meta(sched_out)
            assert req_id in meta.reqs_to_send
            assert meta.reqs_to_send[req_id].token_base == step * 1024
            assert meta.reqs_to_send[req_id].num_tokens == 1024
            assert meta.reqs_to_send[req_id].is_final is False
            assert meta.reqs_to_send[req_id].uuid == chunk_uuid(
                base_uuid, step
            )

        # Finish request: all 4096 tokens already sent
        req.num_computed_tokens = 4096
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert params["req_id"] == req_id
        assert params["uuid"] == chunk_uuid(base_uuid, 4)
        assert params["num_tokens"] == 4096

        # No redundant/zero-token SendMeta emitted in build_connector_meta
        assert req_id not in meta_fin.reqs_to_send
        assert len(meta_fin.reqs_to_send) == 0

        # Verify state is cleaned up
        assert req_id not in scheduler._stage3_send_requests


def test_untracked_request_terminal_fallback() -> None:
    """Verifies request_finished() fallback when request was not tracked.

    If update_state_after_alloc was not called or request is not in
    _stage3_send_requests (req_state is None), request_finished() must
    gracefully emit a single monolithic SendMeta([0, num_tokens), is_final=True)
    with a freshly generated UUID.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-untracked-2048"
    base_uuid = 666777
    block_ids = list(range(100, 132))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 2048
    req.num_computed_tokens = 2048
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    # Ensure req_id is NOT in _stage3_send_requests
    assert req_id not in scheduler._stage3_send_requests

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert params["req_id"] == req_id
        assert params["uuid"] == base_uuid
        assert params["num_tokens"] == 2048

        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 2048
        assert s_fin.is_final is True
        assert s_fin.uuid == base_uuid
        assert s_fin.local_block_ids == block_ids


def test_short_prompt_multi_step_bypass() -> None:
    """Verifies that multi-step execution for short prompts stays bypassed.

    A 512-token prompt executed in two 256-token steps must emit zero
    intermediate chunks on both steps, and emit a single terminal chunk on
    request_finished().
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-short-multistep-512"
    base_uuid = 444333
    block_ids = list(range(100, 108))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 512
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Step 1: 256 tokens scheduled -> bypassed, 0 chunks emitted
        sched_out_1 = MagicMock(spec=SchedulerOutput)
        sched_out_1.num_scheduled_tokens = {req_id: 256}
        meta_1 = scheduler.build_connector_meta(sched_out_1)
        assert len(meta_1.reqs_to_send) == 0

        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 0
        assert req_state.next_chunk_idx == 0

        # Step 2: another 256 tokens scheduled (512 total) -> still
        # bypassed (< 1024 stride)
        sched_out_2 = MagicMock(spec=SchedulerOutput)
        sched_out_2.num_scheduled_tokens = {req_id: 256}
        meta_2 = scheduler.build_connector_meta(sched_out_2)
        assert len(meta_2.reqs_to_send) == 0
        assert req_state.tokens_sent == 0

        # Prefill completed: request_finished emits single terminal [0, 512)
        req.num_computed_tokens = 512
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert req_id in meta_fin.reqs_to_send
        s_fin = meta_fin.reqs_to_send[req_id]
        assert s_fin.token_base == 0
        assert s_fin.num_tokens == 512
        assert s_fin.is_final is True
        assert s_fin.uuid == chunk_uuid(base_uuid, 0)
        assert s_fin.local_block_ids == block_ids

        assert req_id not in scheduler._stage3_send_requests


def test_request_finished_duplicate_idempotency_multi_chunk() -> None:
    """Verifies duplicate request_finished calls on multi-chunk requests.

    A multi-chunk request with intermediate emissions and a terminal remainder
    must return identical cached parameters and avoid duplicate emissions on
    subsequent request_finished() calls.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=4,
    )
    req_id = "req-dedup-multichunk"
    base_uuid = 123987
    block_ids = list(range(100, 155))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 3500
    req.num_computed_tokens = 0
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Emit 3 chunks (3072 tokens)
        for _ in range(3):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: 1024}
            scheduler.build_connector_meta(sched_out)

        req.num_computed_tokens = 3500

        # First request_finished call
        delay1, params1 = scheduler.request_finished(req, block_ids)
        meta1 = scheduler.build_connector_meta()
        assert delay1 is True
        assert req_id in meta1.reqs_to_send
        assert meta1.reqs_to_send[req_id].token_base == 3072
        assert meta1.reqs_to_send[req_id].num_tokens == 428
        assert meta1.reqs_to_send[req_id].is_final is True

        # Second request_finished call (duplicate)
        delay2, params2 = scheduler.request_finished(req, block_ids)
        meta2 = scheduler.build_connector_meta()
        assert delay2 is True
        assert params1 == params2
        assert len(meta2.reqs_to_send) == 0

