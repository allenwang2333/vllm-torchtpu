# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prefix-aware suffix transfers and hybrid recurrent states.

Tests verify:
1. Prefill transfers skip already-cached prompt prefix chunks.
2. Boundary chunks configure dst_skip_bytes for uncached suffix bytes.
3. Hybrid models isolate recurrent state transfers to terminal chunks.
4. FA-only models omit recurrent state transfers entirely.
5. Workers forward dst_skip_bytes and validate duplicate/conflicting metadata.
6. Helper functions compute exact byte offsets and handle edge cases.
"""

from collections.abc import Iterator
import contextlib
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorScheduler,
    TPURaidenConnectorWorker,
    _compute_boundary_skip_bytes,
    _compute_cached_prefix_progress,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class _FakeRaidenControllerFacade:
    """Mock controller client recording RPC invocations for testing."""

    def __init__(self) -> None:
        """Initializes recorded call lists and default returns."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit call."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks call."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks call."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed call."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result


def _make_raiden_scheduler(
    *,
    is_producer: bool = True,
    block_size: int = 16,
    pcp_size: int = 4,
    interleave_size: int = 256,
    chunk_transfer_window: int = 2,
) -> TPURaidenConnectorScheduler:
    """Creates a mock-configured TPURaidenConnectorScheduler for testing."""
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 8192
    cfg.model_config.num_hidden_layers = 32
    cfg.model_config.get_num_layers = MagicMock(return_value=32)
    cfg.model_config.get_num_kv_heads = MagicMock(return_value=8)
    cfg.model_config.get_head_size = MagicMock(return_value=128)
    cfg.model_config.dtype = torch.bfloat16
    cfg.parallel_config = MagicMock()
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


def _make_test_worker(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    is_producer: bool = True,
    pcp_size: int = 4,
    interleave_size: int = 256,
    block_size: int = 1024,
    token_bytes: int = 64,
    transfer_rank: int = 0,
) -> tuple[TPURaidenConnectorWorker, _FakeRaidenControllerFacade]:
    """Creates a TPURaidenConnectorWorker wired with fake controllers."""
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 16384
    cfg.model_config.num_hidden_layers = 32
    cfg.model_config.get_num_layers = MagicMock(return_value=32)
    cfg.model_config.get_num_kv_heads = MagicMock(return_value=8)
    cfg.model_config.get_head_size = MagicMock(return_value=128)
    cfg.model_config.dtype = torch.bfloat16
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
        patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"),
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=tp_rank),
        patch(
            f"{_MOD}.get_tensor_model_parallel_world_size", return_value=tp_size
        ),
    ):
        worker = TPURaidenConnectorWorker(cfg)

    facade = _FakeRaidenControllerFacade()
    unit = SimpleNamespace(
        job_name="prefill",
        job_replica_id="producer-0",
        data_name="kv.fa",
        data_replica_idx=0,
    )
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "127.0.0.1:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=transfer_rank)
    worker._raiden_transfer_parallelism = MagicMock(return_value=pcp_size)
    worker._stage3_fa_token_bytes = MagicMock(return_value=token_bytes)
    worker._flattened_kv_cache_tensors = MagicMock(return_value=[])

    return worker, facade


def test_prefix_cache_chunk_skipping() -> None:
    """Verifies that cached prefix chunks are skipped when prefix is cached."""
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        block_size=16,
        pcp_size=4,
        interleave_size=256,  # min_transfer_tokens = 4 * 256 = 1024
    )

    req_id = "req-prefix-skip-test"
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2048  # Cached prefix length M = 2048
    mock_req.num_prompt_tokens = 4096
    mock_req.prompt_token_ids = list(range(4096))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    base_uuid = 555123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        # Initialize tracking: 2048 tokens cached -> 2 chunks skipped
        scheduler._update_stage3_producer_state_after_alloc(
            mock_req, blocks=list(range(64))
        )

        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.cached_prefix_tokens == 2048
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2

        # Step 1: Schedule 1024 tokens (computed progresses from 2048 to 3072)
        sched_output = MagicMock(spec=SchedulerOutput)
        sched_output.num_scheduled_tokens = {req_id: 1024}

        meta = scheduler.build_connector_meta(sched_output)

        # Verify that a single chunk is emitted starting at token_base=2048
        assert req_id in meta.reqs_to_send
        chunk = meta.reqs_to_send[req_id]
        assert chunk.token_base == 2048
        assert chunk.num_tokens == 1024
        assert chunk.is_final is False
        assert chunk.dst_skip_bytes is None
        assert req_state.tokens_sent == 3072
        assert req_state.next_chunk_idx == 3

        # Step 2: Request finishes at 4096 tokens
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 4096
        scheduler.request_finished(mock_req, block_ids=list(range(64)))
        assert req_id in scheduler.reqs_to_send
        final_chunk = scheduler.reqs_to_send[req_id]
        assert final_chunk.token_base == 3072
        assert final_chunk.num_tokens == 1024
        assert final_chunk.is_final is True
        assert final_chunk.dst_skip_bytes is None


def test_prefix_cache_boundary_chunk_dst_skip_bytes() -> None:
    """Verifies unaligned cached prefixes produce correct dst_skip_bytes."""
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        block_size=16,
        pcp_size=4,
        interleave_size=256,  # min_transfer_tokens = 1024
    )
    # Configure 64 bytes per token for test
    scheduler._stage3_fa_token_bytes = MagicMock(return_value=64)

    req_id = "req-prefix-boundary-test"
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2560  # Unaligned prefix: 2048 + 512 cached
    mock_req.num_prompt_tokens = 4096
    mock_req.prompt_token_ids = list(range(4096))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    base_uuid = 777123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        # 2560 // 1024 = 2 cached chunks skipped (tokens_sent=2048)
        scheduler._update_stage3_producer_state_after_alloc(
            mock_req, blocks=list(range(64))
        )

        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.cached_prefix_tokens == 2560
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2

        # Step 1: Schedule 512 tokens (computed reaches 3072)
        sched_output = MagicMock(spec=SchedulerOutput)
        sched_output.num_scheduled_tokens = {req_id: 512}

        meta = scheduler.build_connector_meta(sched_output)

        # Verify boundary chunk carries dst_skip_bytes
        # (2560 - 2048) = 512 tokens * 64 bytes = 32768 bytes
        assert req_id in meta.reqs_to_send
        boundary_chunk = meta.reqs_to_send[req_id]
        assert boundary_chunk.token_base == 2048
        assert boundary_chunk.num_tokens == 1024
        assert boundary_chunk.is_final is False
        assert boundary_chunk.dst_skip_bytes == 32768
        assert req_state.tokens_sent == 3072
        assert req_state.next_chunk_idx == 3

        # Step 2: Subsequent chunk [3072, 4096) has token_base >= cached_prefix
        # dst_skip_bytes should be None
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 4096
        scheduler.request_finished(mock_req, block_ids=list(range(64)))
        assert req_id in scheduler.reqs_to_send
        final_chunk = scheduler.reqs_to_send[req_id]
        assert final_chunk.token_base == 3072
        assert final_chunk.num_tokens == 1024
        assert final_chunk.is_final is True
        assert final_chunk.dst_skip_bytes is None


def test_prefix_cache_terminal_remainder_dst_skip_bytes() -> None:
    """Verifies terminal remainder chunks carry dst_skip_bytes if boundary."""
    scheduler = _make_raiden_scheduler(
        is_producer=True,
        block_size=16,
        pcp_size=4,
        interleave_size=256,  # min_transfer_tokens = 1024
    )
    scheduler._stage3_fa_token_bytes = MagicMock(return_value=64)

    req_id = "req-prefix-terminal-boundary-test"
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_computed_tokens = 2560  # Cached prefix M = 2560
    mock_req.num_prompt_tokens = 3000  # Total prompt < 3072
    mock_req.prompt_token_ids = list(range(3000))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    base_uuid = 888123
    with _patch_raiden_stage3_env(base_uuid=base_uuid):
        # 2560 // 1024 = 2 chunks skipped (tokens_sent=2048)
        scheduler._update_stage3_producer_state_after_alloc(
            mock_req, blocks=list(range(64))
        )

        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 2048

        # Request finishes directly at 3000 tokens
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_req.num_computed_tokens = 3000
        scheduler.request_finished(mock_req, block_ids=list(range(64)))

        assert req_id in scheduler.reqs_to_send
        terminal_chunk = scheduler.reqs_to_send[req_id]
        assert terminal_chunk.token_base == 2048
        assert terminal_chunk.num_tokens == 952  # 3000 - 2048 = 952
        assert terminal_chunk.is_final is True
        # (2560 - 2048) * 64 = 32768
        assert terminal_chunk.dst_skip_bytes == 32768


def test_recurrent_states_on_terminal_chunk_only() -> None:
    """Verifies that recurrent states are only sent on the terminal chunk."""
    worker, facade = _make_test_worker(pcp_size=4, interleave_size=256)
    worker._stage3_state_group_count = 2  # 2 GDN state groups
    worker._stage3_state_pool_spans = MagicMock(
        return_value=[
            SimpleNamespace(tag="gdn.conv.g0", spans=[]),
            SimpleNamespace(tag="gdn.ssm.g0", spans=[]),
        ]
    )

    # 1. Non-final chunk (intermediate)
    intermediate_req_id = "req-hybrid-intermediate"
    intermediate_meta = TPUConnectorMetadata()
    intermediate_meta.reqs_to_send[intermediate_req_id] = SendMeta(
        uuid=1001,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,  # Intermediate chunk
        mamba_state_block_ids=[10, 11],
    )

    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(intermediate_meta)

    assert len(facade.register_request_blocks_calls) == 1
    call1 = facade.register_request_blocks_calls[0]
    assert call1["req_id"] == intermediate_req_id
    assert call1["uuid"] == 1001
    pool_spans1 = call1["pool_spans"]

    # Verify intermediate registration contains ONLY FlashAttention spans
    tags1 = [span.tag for span in pool_spans1]
    assert "gdn.conv.g0" not in tags1
    assert "gdn.ssm.g0" not in tags1

    # 2. Final chunk (terminal)
    terminal_req_id = "req-hybrid-terminal"
    terminal_meta = TPUConnectorMetadata()
    terminal_meta.reqs_to_send[terminal_req_id] = SendMeta(
        uuid=1002,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=1024,
        is_final=True,  # Terminal chunk
        mamba_state_block_ids=[10, 11],
    )

    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(terminal_meta)

    assert len(facade.register_request_blocks_calls) == 2
    call2 = facade.register_request_blocks_calls[1]
    assert call2["req_id"] == terminal_req_id
    assert call2["uuid"] == 1002
    pool_spans2 = call2["pool_spans"]

    # Verify terminal registration contains FA and GDN recurrent state spans
    tags2 = [span.tag for span in pool_spans2]
    assert "gdn.conv.g0" in tags2
    assert "gdn.ssm.g0" in tags2


def test_worker_forwards_dst_skip_bytes() -> None:
    """Verifies that TPURaidenConnectorWorker forwards dst_skip_bytes."""
    worker, facade = _make_test_worker(pcp_size=4, interleave_size=256)
    req_id = "req-worker-skip-bytes"

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=2001,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
        dst_skip_bytes=32768,
    )

    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(meta)

    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    assert call["dst_skip_bytes"] == [32768]
    assert worker._active_tensor_refs[req_id].dst_skip_bytes == 32768


def test_worker_rejects_negative_dst_skip_bytes() -> None:
    """Verifies TPURaidenConnectorWorker raises on negative dst_skip_bytes."""
    worker, facade = _make_test_worker(pcp_size=4, interleave_size=256)
    req_id = "req-negative-skip"

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=2002,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
        dst_skip_bytes=-100,
    )

    with (
        patch.object(
            worker,
            "_require_stage3_controller",
            return_value=(facade, "127.0.0.1:27000"),
        ),
        pytest.raises(
            ValueError,
            match="Stage-3 producer send metadata requires a non-negative "
                  "dst_skip_bytes",
        ),
    ):
        worker._register_stage3_request_blocks(meta)


def test_worker_duplicate_and_replay_dst_skip_bytes_validation() -> None:
    """Verifies duplicate and replay dst_skip_bytes validation."""
    worker, facade = _make_test_worker(pcp_size=4, interleave_size=256)
    req_id = "req-dup-test"

    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(
        uuid=2003,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
        dst_skip_bytes=1024,
    )

    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(meta1)

    assert len(facade.register_request_blocks_calls) == 1

    # 1. Matching duplicate registration -> idempotent, no additional call
    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(meta1)
    assert len(facade.register_request_blocks_calls) == 1

    # 2. Conflicting duplicate registration -> ValueError
    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(
        uuid=2003,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
        dst_skip_bytes=2048,  # Conflict: 2048 != 1024
    )

    with (
        patch.object(
            worker,
            "_require_stage3_controller",
            return_value=(facade, "127.0.0.1:27000"),
        ),
        pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ),
    ):
        worker._register_stage3_request_blocks(meta_conflict)

    # 3. Conflicting replay after terminal registration -> ValueError
    terminal_reg = worker._stage3_registered_sends.pop(req_id)
    worker._stage3_terminal_sends[req_id] = terminal_reg
    worker._active_tensor_refs.pop(req_id, None)

    with (
        patch.object(
            worker,
            "_require_stage3_controller",
            return_value=(facade, "127.0.0.1:27000"),
        ),
        pytest.raises(
            ValueError,
            match="Conflicting replay after terminal Stage-3 producer "
                  "registration",
        ),
    ):
        worker._register_stage3_request_blocks(meta_conflict)


def test_fa_only_model_terminal_no_mamba_spans() -> None:
    """Verifies that FA-only models register no recurrent state spans."""
    worker, facade = _make_test_worker(pcp_size=4, interleave_size=256)
    worker._stage3_state_group_count = 0  # FA-only model

    req_id = "req-fa-only-terminal"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=3001,
        local_block_ids=[1, 2, 3, 4],
        expiration_time=time.perf_counter() + 60.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )

    with patch.object(
        worker,
        "_require_stage3_controller",
        return_value=(facade, "127.0.0.1:27000"),
    ):
        worker._register_stage3_request_blocks(meta)

    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    tags = [span.tag for span in call["pool_spans"]]
    assert "gdn.conv.g0" not in tags
    assert "gdn.ssm.g0" not in tags


@pytest.mark.parametrize(
    "cached_prefix,token_base,token_bytes,expected_skip",
    [
        (2560, 2048, 64, 32768),
        (2048, 2048, 64, None),
        (1024, 2048, 64, None),
        (2560, 2048, 0, None),
        (2560, 2048, -10, None),
        (0, 0, 64, None),
    ],
)
def test_compute_boundary_skip_bytes_contract(
    cached_prefix: int,
    token_base: int,
    token_bytes: int,
    expected_skip: int | None,
) -> None:
    """Verifies calculation and edge-case contracts of boundary skip bytes."""
    skip = _compute_boundary_skip_bytes(cached_prefix, token_base, token_bytes)
    assert skip == expected_skip


@pytest.mark.parametrize(
    "cached_prefix,total_prompt,min_transfer,expected_progress",
    [
        (2560, 4096, 1024, (2048, 2)),
        (4096, 4096, 1024, (0, 0)),
        (5000, 4096, 1024, (0, 0)),
        (0, 4096, 1024, (0, 0)),
        (2560, 4096, 0, (0, 0)),
    ],
)
def test_compute_cached_prefix_progress_contract(
    cached_prefix: int,
    total_prompt: int,
    min_transfer: int,
    expected_progress: tuple[int, int],
) -> None:
    """Verifies calculation and edge-case contracts of prefix progress."""
    progress = _compute_cached_prefix_progress(
        cached_prefix, total_prompt, min_transfer
    )
    assert progress == expected_progress


def test_stage3_fa_token_bytes_calculation_and_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies _stage3_fa_token_bytes computes bytes or returns fallback."""
    scheduler = _make_raiden_scheduler()

    # Exact calculation:
    # 2 * layers(32) * heads(8) * dim(128) * dtype_bytes(2) / tp(1) = 131072
    token_bytes = scheduler._stage3_fa_token_bytes()
    assert token_bytes == 131072

    # Broken model_config triggering fallback and warning
    scheduler.vllm_config.model_config.num_hidden_layers = None
    scheduler.vllm_config.model_config.get_num_layers = MagicMock(
        side_effect=AttributeError("broken")
    )
    with caplog.at_level(logging.WARNING):
        fallback_bytes = scheduler._stage3_fa_token_bytes()
    assert fallback_bytes == 64
    assert any(
        "failed to compute FA token bytes" in record.message
        for record in caplog.records
    )
