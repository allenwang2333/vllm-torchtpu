# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sliding window slot admission and memory scaling.

Validates per-request in-flight chunk limiting (W=2 by default) during
pipelined prefill chunk emission, sub-chunk completion signals, and linear
host slot scaling with configured window sizes.
"""

from collections.abc import Iterator
import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    CHUNK_TRANSFER_WINDOW,
    TPURaidenConnectorScheduler,
    TPURaidenConnectorWorker,
    _get_chunk_transfer_window,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


def _make_raiden_scheduler(
    *,
    is_producer: bool = True,
    block_size: int = 16,
    pcp_size: int = 4,
    interleave_size: int = 256,
    chunk_transfer_window: int | None = None,
    extra_config_window: int | None = None,
) -> TPURaidenConnectorScheduler:
    """Creates a mock-configured TPURaidenConnectorScheduler for testing.

    Args:
        is_producer: Whether the scheduler operates in producer role.
        block_size: Number of tokens per physical KV cache block.
        pcp_size: Prefill context parallelism size.
        interleave_size: Token chunk stride per context parallel rank.
        chunk_transfer_window: Direct window size attribute on transfer config.
        extra_config_window: Window size entry in extra config dictionary.

    Returns:
        A configured instance of TPURaidenConnectorScheduler.
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    extra_cfg: dict[str, Any] = {}
    if extra_config_window is not None:
        extra_cfg["chunk_transfer_window"] = extra_config_window
    cfg.kv_transfer_config.kv_connector_extra_config = extra_cfg
    if chunk_transfer_window is not None:
        cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    else:
        cfg.kv_transfer_config.chunk_transfer_window = None

    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 8192
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


def _make_raiden_worker(
    *,
    is_producer: bool = True,
    tp_rank: int = 0,
    tp_size: int = 4,
    dp_rank: int = 0,
    dp_size: int = 1,
    pcp_size: int = 4,
    interleave_size: int = 256,
    block_size: int = 16,
    chunk_transfer_window: int | None = None,
    extra_config_window: int | None = None,
) -> TPURaidenConnectorWorker:
    """Creates a mock-configured TPURaidenConnectorWorker for testing.

    Args:
        is_producer: Whether the worker operates in producer role.
        tp_rank: Tensor model parallel rank.
        tp_size: Tensor model parallel world size.
        dp_rank: Data parallel rank.
        dp_size: Data parallel world size.
        pcp_size: Prefill context parallelism size.
        interleave_size: Token chunk stride per context parallel rank.
        block_size: Number of tokens per physical KV cache block.
        chunk_transfer_window: Direct window size attribute on transfer config.
        extra_config_window: Window size entry in extra config dictionary.

    Returns:
        A configured instance of TPURaidenConnectorWorker.
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    extra_cfg: dict[str, Any] = {}
    if extra_config_window is not None:
        extra_cfg["chunk_transfer_window"] = extra_config_window
    cfg.kv_transfer_config.kv_connector_extra_config = extra_cfg
    if chunk_transfer_window is not None:
        cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    else:
        cfg.kv_transfer_config.chunk_transfer_window = None

    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 8192
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=tp_rank),
        patch(
            f"{_MOD}.get_tensor_model_parallel_world_size", return_value=tp_size
        ),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        worker = TPURaidenConnectorWorker(cfg)
    return worker


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


def _make_req(req_id: str, prompt_len: int = 4096) -> SimpleNamespace:
    """Creates a lightweight mock Request structure for unit testing.

    Args:
        req_id: Unique string identifier for the mock request.
        prompt_len: Total prompt token length for the mock request.

    Returns:
        SimpleNamespace simulating a vLLM Request object.
    """
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=0,
        prompt_token_ids=[0] * prompt_len,
        kv_transfer_params=None,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )


def test_default_chunk_transfer_window_constant():
    """Verify that default CHUNK_TRANSFER_WINDOW constant is 2."""
    assert CHUNK_TRANSFER_WINDOW == 2


def test_get_chunk_transfer_window_config_precedence():
    """Verify hierarchy, precedence, and clamping for window size (W)."""
    # 1. None config defaults to CHUNK_TRANSFER_WINDOW (2)
    assert _get_chunk_transfer_window(None) == 2

    # 2. VllmConfig with empty kv_transfer_config defaults to 2
    cfg_empty = MagicMock(spec=VllmConfig)
    cfg_empty.kv_transfer_config = None
    assert _get_chunk_transfer_window(cfg_empty) == 2

    # 3. VllmConfig with direct chunk_transfer_window attribute
    cfg_direct = MagicMock(spec=VllmConfig)
    cfg_direct.kv_transfer_config = MagicMock()
    cfg_direct.kv_transfer_config.chunk_transfer_window = 4
    cfg_direct.kv_transfer_config.kv_connector_extra_config = {}
    assert _get_chunk_transfer_window(cfg_direct) == 4

    # 4. VllmConfig with kv_connector_extra_config override
    cfg_extra = MagicMock(spec=VllmConfig)
    cfg_extra.kv_transfer_config = MagicMock()
    cfg_extra.kv_transfer_config.chunk_transfer_window = None
    cfg_extra.kv_transfer_config.kv_connector_extra_config = {
        "chunk_transfer_window": 3
    }
    assert _get_chunk_transfer_window(cfg_extra) == 3

    # 5. Precedence: direct chunk_transfer_window overrides extra config
    cfg_both = MagicMock(spec=VllmConfig)
    cfg_both.kv_transfer_config = MagicMock()
    cfg_both.kv_transfer_config.chunk_transfer_window = 5
    cfg_both.kv_transfer_config.kv_connector_extra_config = {
        "chunk_transfer_window": 10
    }
    assert _get_chunk_transfer_window(cfg_both) == 5

    # 6. Clamping non-positive numbers (0 or negative) to minimum 1
    cfg_zero = MagicMock(spec=VllmConfig)
    cfg_zero.kv_transfer_config = MagicMock()
    cfg_zero.kv_transfer_config.chunk_transfer_window = 0
    assert _get_chunk_transfer_window(cfg_zero) == 1

    cfg_neg = MagicMock(spec=VllmConfig)
    cfg_neg.kv_transfer_config = MagicMock()
    cfg_neg.kv_transfer_config.chunk_transfer_window = -4
    assert _get_chunk_transfer_window(cfg_neg) == 1

    # 7. Type coercion from string integers
    cfg_str = MagicMock(spec=VllmConfig)
    cfg_str.kv_transfer_config = MagicMock()
    cfg_str.kv_transfer_config.chunk_transfer_window = "6"
    assert _get_chunk_transfer_window(cfg_str) == 6


def test_in_flight_window_limiting():
    """Attempt to emit 3 chunks for a request with W=2.

    Verify first 2 chunks are emitted and 3rd chunk is deferred until chunk 0
    completes.
    """
    with _patch_raiden_stage3_env(base_uuid=100001):
        scheduler = _make_raiden_scheduler(
            is_producer=True,
            pcp_size=4,
            interleave_size=256,  # min_transfer_tokens = 1024
            chunk_transfer_window=2,
        )
        assert scheduler.chunk_transfer_window == 2

        req = _make_req("req-window-test", prompt_len=4096)
        block_ids = list(range(256))
        scheduler.update_state_after_alloc(req, block_ids, 0)

        req_state = scheduler._stage3_send_requests["req-window-test"]
        assert req_state.tokens_sent == 0
        assert scheduler._in_flight_chunks.get("req-window-test", 0) == 0
        assert scheduler._can_emit_chunk("req-window-test") is True

        # Step 1: Compute 1024 tokens -> Chunk 0 emitted [0, 1024)
        sched_out1 = MagicMock(spec=SchedulerOutput)
        sched_out1.num_scheduled_tokens = {"req-window-test": 1024}
        meta1 = scheduler.build_connector_meta(sched_out1)

        assert "req-window-test" in meta1.reqs_to_send
        send0: SendMeta = meta1.reqs_to_send["req-window-test"]
        assert send0.token_base == 0
        assert send0.num_tokens == 1024
        assert send0.is_final is False
        assert scheduler._in_flight_chunks["req-window-test"] == 1
        assert scheduler._can_emit_chunk("req-window-test") is True
        assert req_state.tokens_sent == 1024

        # Step 2: Compute 1024 tokens (total 2048) -> Chunk 1 emitted
        sched_out2 = MagicMock(spec=SchedulerOutput)
        sched_out2.num_scheduled_tokens = {"req-window-test": 1024}
        meta2 = scheduler.build_connector_meta(sched_out2)

        assert "req-window-test" in meta2.reqs_to_send
        send1: SendMeta = meta2.reqs_to_send["req-window-test"]
        assert send1.token_base == 1024
        assert send1.num_tokens == 1024
        assert send1.is_final is False
        assert scheduler._in_flight_chunks["req-window-test"] == 2
        assert scheduler._can_emit_chunk("req-window-test") is False
        assert req_state.tokens_sent == 2048

        # Step 3: Compute 1024 tokens (total 3072) -> Chunk 2 deferred by W=2!
        sched_out3 = MagicMock(spec=SchedulerOutput)
        sched_out3.num_scheduled_tokens = {"req-window-test": 1024}
        meta3 = scheduler.build_connector_meta(sched_out3)

        # Emission is deferred: reqs_to_send is empty for this request
        assert "req-window-test" not in meta3.reqs_to_send
        assert scheduler._in_flight_chunks["req-window-test"] == 2
        assert req_state.tokens_sent == 2048
        assert req_state.num_computed_tokens == 3072

        # Receive completion signal for chunk 0
        conn_out = SimpleNamespace(finished_sending=["req-window-test"])
        scheduler.update_connector_output(conn_out)

        assert scheduler._in_flight_chunks["req-window-test"] == 1
        assert scheduler._can_emit_chunk("req-window-test") is True

        # Step 4: Next scheduler step without new scheduled tokens
        sched_out4 = MagicMock(spec=SchedulerOutput)
        sched_out4.num_scheduled_tokens = {"req-window-test": 0}
        meta4 = scheduler.build_connector_meta(sched_out4)

        # Deferred chunk 2 is now emitted: [2048, 3072)
        assert "req-window-test" in meta4.reqs_to_send
        send2: SendMeta = meta4.reqs_to_send["req-window-test"]
        assert send2.token_base == 2048
        assert send2.num_tokens == 1024
        assert send2.is_final is False
        assert scheduler._in_flight_chunks["req-window-test"] == 2
        assert req_state.tokens_sent == 3072


def test_slot_pool_scaling():
    """Verify _num_raiden_slots calculation scales linearly with window size."""
    window_sizes = [1, 2, 3, 4, 8]
    slot_counts = {}

    for window_size in window_sizes:
        worker = _make_raiden_worker(
            is_producer=True,
            tp_size=4,
            chunk_transfer_window=window_size,
        )
        assert worker.chunk_transfer_window == window_size

        # Mock runner with representative KV cache tensor
        runner = MagicMock()
        nope = torch.empty((4, 8, 4, 128), dtype=torch.uint8)
        rope = torch.empty((4, 2, 4, 128), dtype=torch.uint8)
        idx = torch.empty((4, 2, 4, 256), dtype=torch.uint8)
        runner.kv_caches = [(nope, rope), idx]
        worker.runner = runner

        with (
            patch(
                f"{_MOD}.dist_utils.get_raiden_transfer_num_slots",
                return_value=0,
            ),
            patch(f"{_MOD}.dist_utils.get_kv_shm_pool_gb", return_value=1.0),
        ):
            slots = worker._num_raiden_slots(max_blocks=4)
            slot_counts[window_size] = slots

    # Verify linear proportionality: slots(W) == W * slots(1)
    base_slots = slot_counts[1]
    assert base_slots > 0
    for window_size in window_sizes:
        assert slot_counts[window_size] == base_slots * window_size

    # Verify manual override via get_raiden_transfer_num_slots bypasses scaling
    worker_override = _make_raiden_worker(
        is_producer=True, chunk_transfer_window=4
    )
    worker_override.runner = MagicMock()
    with patch(
        f"{_MOD}.dist_utils.get_raiden_transfer_num_slots", return_value=7
    ):
        assert worker_override._num_raiden_slots(max_blocks=4) == 7


def test_multi_request_independent_window_accounting():
    """Verify independent in-flight tracking across distinct requests."""
    with _patch_raiden_stage3_env(base_uuid=200001):
        scheduler = _make_raiden_scheduler(
            is_producer=True,
            pcp_size=4,
            interleave_size=256,
            chunk_transfer_window=2,
        )

        req_a = _make_req("req-A", prompt_len=4096)
        req_b = _make_req("req-B", prompt_len=4096)
        scheduler.update_state_after_alloc(req_a, list(range(256)), 0)
        scheduler.update_state_after_alloc(req_b, list(range(256)), 0)

        # Emit 2 chunks for req-A
        for _ in range(2):
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {"req-A": 1024}
            scheduler.build_connector_meta(sched_out)

        assert scheduler._in_flight_chunks["req-A"] == 2
        assert scheduler._can_emit_chunk("req-A") is False

        # req-B has 0 in-flight chunks and is not blocked by req-A
        assert scheduler._in_flight_chunks.get("req-B", 0) == 0
        assert scheduler._can_emit_chunk("req-B") is True

        sched_out_b = MagicMock(spec=SchedulerOutput)
        sched_out_b.num_scheduled_tokens = {"req-B": 1024}
        meta_b = scheduler.build_connector_meta(sched_out_b)

        assert "req-B" in meta_b.reqs_to_send
        assert scheduler._in_flight_chunks["req-B"] == 1


def test_on_chunk_completed_helper():
    """Verify on_chunk_completed decrements and removes tracked request IDs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    scheduler._in_flight_chunks["req-1"] = 2

    scheduler.on_chunk_completed("req-1")
    assert scheduler._in_flight_chunks["req-1"] == 1

    scheduler.on_chunk_completed("req-1")
    assert "req-1" not in scheduler._in_flight_chunks

    # Extra decrement on untracked request does not raise
    scheduler.on_chunk_completed("req-1")
    assert "req-1" not in scheduler._in_flight_chunks


def test_request_finished_cleans_in_flight_state():
    """Verify request_finished purges in-flight tracking state."""
    req_id = "req-finish-test"
    # 2048 tokens on PCP4 (block_size=16): (2048 + 63) // 64 = 32 blocks.
    block_ids = list(range(32))

    req = MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = 2048
    req.num_computed_tokens = 2048
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=300001):
        scheduler = _make_raiden_scheduler(
            is_producer=True,
            pcp_size=4,
            interleave_size=256,
            chunk_transfer_window=2,
        )
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Emit 1 chunk
        sched_out = MagicMock(spec=SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: 1024}
        scheduler.build_connector_meta(sched_out)
        assert scheduler._in_flight_chunks[req_id] == 1

        # Finish request
        with patch(
            f"{_MOD}._resolved_reshard_controller_address",
            return_value="127.0.0.1:27000",
        ):
            ok, params = scheduler.request_finished(req, block_ids)
            assert ok is True

        assert req_id not in scheduler._in_flight_chunks



def test_can_emit_chunk_custom_window_sizes():
    """Verify _can_emit_chunk across window sizes W=1, W=3, W=4."""
    # Window W=1 (Strict lock-step)
    sched_w1 = _make_raiden_scheduler(chunk_transfer_window=1)
    assert sched_w1._can_emit_chunk("req-w1") is True
    sched_w1._in_flight_chunks["req-w1"] = 1
    assert sched_w1._can_emit_chunk("req-w1") is False

    # Window W=3
    sched_w3 = _make_raiden_scheduler(chunk_transfer_window=3)
    assert sched_w3._can_emit_chunk("req-w3") is True
    sched_w3._in_flight_chunks["req-w3"] = 2
    assert sched_w3._can_emit_chunk("req-w3") is True
    sched_w3._in_flight_chunks["req-w3"] = 3
    assert sched_w3._can_emit_chunk("req-w3") is False


def test_update_connector_output_none_and_untracked_safe():
    """Verify update_connector_output with None, empty, or untracked inputs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    scheduler._stage3_finished_sends["req-tracked"] = (1, (0,), 1024, {})
    scheduler._in_flight_chunks["req-tracked"] = 1

    # None connector_output is a safe no-op
    scheduler.update_connector_output(None)
    assert scheduler._in_flight_chunks["req-tracked"] == 1

    # connector_output with finished_sending=None is a safe no-op
    scheduler.update_connector_output(SimpleNamespace(finished_sending=None))
    assert scheduler._in_flight_chunks["req-tracked"] == 1

    # Untracked request ID is safely ignored
    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=["req-untracked"])
    )
    assert scheduler._in_flight_chunks["req-tracked"] == 1

    # Tracked request ID cleans up finished_sends and in_flight_chunks
    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=["req-tracked"])
    )
    assert "req-tracked" not in scheduler._stage3_finished_sends
    assert "req-tracked" not in scheduler._in_flight_chunks


def test_on_chunk_completed_non_string_id_interop():
    """Verify on_chunk_completed and _can_emit_chunk with non-string IDs."""
    scheduler = _make_raiden_scheduler(chunk_transfer_window=2)
    numeric_id = 98765

    # When tracked with string key (canonical in vLLM)
    scheduler._in_flight_chunks[str(numeric_id)] = 2
    assert scheduler._can_emit_chunk(str(numeric_id)) is False
    assert scheduler._can_emit_chunk(numeric_id) is False

    # on_chunk_completed with numeric ID decrements string key
    scheduler.on_chunk_completed(numeric_id)
    assert scheduler._in_flight_chunks[str(numeric_id)] == 1
    assert scheduler._can_emit_chunk(numeric_id) is True

    # Second completion cleans up
    scheduler.on_chunk_completed(numeric_id)
    assert str(numeric_id) not in scheduler._in_flight_chunks
    assert scheduler._can_emit_chunk(numeric_id) is True
