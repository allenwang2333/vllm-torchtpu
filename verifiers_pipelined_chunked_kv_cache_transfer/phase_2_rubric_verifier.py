# SPDX-License-Identifier: Apache-2.0
"""Phase 2 Rubric Verifier — Metadata Schema & Scheduler Stride Chunk Emission.

Validates all Phase 2 Rubric items for chunked KV cache transfer:
- R2.1 — Per-Step Aligned Stride Emission: During chunked prefill execution,
  the scheduler emits transfer descriptors for completed token spans whenever
  computed tokens cross interleave stride boundaries.
- R2.2 — Monotonic Token Cursor Tracking: The scheduler maintains a monotonic
  cursor of emitted tokens, ensuring intermediate chunks cover mutually
  exclusive, contiguous token intervals.
- R2.3 — Short-Prompt Bypass & Terminal Remainder Emission: Requests with prompt
  lengths below the interleave stride bypass intermediate emission, and the
  final request handler reliably emits any un-sent remainder tokens with the
  completion indicator set.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from unittest import mock

import pytest
from vllm.v1.core.sched import output as sched_output
from vllm.v1 import request as vllm_request
from vllm_torchtpu.distributed.kv_transfer import connector_metadata
from vllm_torchtpu.distributed.kv_transfer import tpu_connector

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


def _make_raiden_scheduler(
    *,
    is_producer: bool = True,
    block_size: int = 16,
    pcp_size: int = 4,
    interleave_size: int = 256,
) -> tpu_connector.TPURaidenConnectorScheduler:
    """Creates a mock-configured TPURaidenConnectorScheduler for testing.

    Initializes a connector scheduler instance with custom prefill context
    parallelism (PCP), interleave chunk size, block size, and mock network
    configuration for testing Stage-3 Raiden chunk emission in isolation.

    Args:
        is_producer: Whether the scheduler operates in KV producer mode.
        block_size: Number of tokens per KV cache block.
        pcp_size: Prefill context parallelism degree.
        interleave_size: Context parallelism interleave chunk size in tokens.

    Returns:
        An instantiated TPURaidenConnectorScheduler with mocked dist_utils.
    """
    cfg = mock.MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 32768
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        mock.patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        mock.patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
        mock.patch(
            f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"
        ),
    ):
        scheduler = tpu_connector.TPURaidenConnectorScheduler(cfg)
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
    """Patches Stage-3 Raiden connector environment variables and utilities.

    Yields a context with all necessary runtime environment variables,
    predicates, controller addresses, and deterministic UUID generation
    mocked for verifying Stage-3 transfer behavior without hardware
    dependencies.

    Args:
        base_uuid: Base integer UUID to return from get_uuid.
        parallelism: Prefill context parallelism degree for TPU transfer.
        timeout_s: P2P wait pull timeout in seconds.
        controller_address: Raiden controller host:port endpoint.
        job_name: Name of the Raiden distributed job.
        engine_id: Unique identifier for the local engine instance.

    Yields:
        None when context enters, restoring original bindings on exit.
    """
    with (
        mock.patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        mock.patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        mock.patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
            controller_address,
        ),
        mock.patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME", job_name),
        mock.patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", engine_id),
        mock.patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
        ),
        mock.patch(
            f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
            return_value=timeout_s,
        ),
        mock.patch(f"{_MOD}.get_uuid", return_value=base_uuid),
    ):
        yield


def _make_mock_request(
    *,
    req_id: str,
    prompt_tokens: int,
    computed_tokens: int = 0,
    status: vllm_request.RequestStatus = vllm_request.RequestStatus.RUNNING,
    is_p_side_truncated: bool = True,
) -> mock.MagicMock:
    """Creates a mock Request object for scheduler interactions.

    Constructs a lightweight test double conforming to the vLLM Request
    interface, carrying token counters, request status, and KV transfer
    parameter dictionary required by the scheduler.

    Args:
        req_id: Unique string identifier for the mock request.
        prompt_tokens: Total number of prompt tokens in the request.
        computed_tokens: Number of prompt tokens computed so far.
        status: Current RequestStatus enum value for the request.
        is_p_side_truncated: Whether prompt-side truncation is enabled.

    Returns:
        A configured MagicMock representing the vLLM Request.
    """
    req = mock.MagicMock()
    req.request_id = req_id
    req.num_prompt_tokens = prompt_tokens
    req.num_computed_tokens = computed_tokens
    req.status = status
    req.kv_transfer_params = {"_p_side_truncated": is_p_side_truncated}
    return req


# ==============================================================================
# Rubric Test 1: R2.1 — Per-Step Aligned Stride Emission
# ==============================================================================


def test_rubric_r2_1_aligned_stride_emission() -> None:
    """Verifies R2.1: Scheduler emits transfer descriptors on stride boundaries.

    Validates across multiple PCP topologies (PCP=2, PCP=4, PCP=8) and prompt
    lengths that when scheduled execution crosses interleave stride boundaries,
    the scheduler emits correctly dimensioned intermediate SendMeta descriptors
    with accurate token offsets, token counts, sub-chunk UUIDs, and
    is_final=False.
    """
    test_topologies = [
        # (pcp_size, interleave_tokens, prompt_tokens, step_tokens)
        (2, 256, 4096, 512),  # stride = 512, 8 steps
        (4, 256, 8192, 1024),  # stride = 1024, 8 steps
        (4, 128, 4096, 512),  # stride = 512, 8 steps
        (8, 256, 16384, 2048),  # stride = 2048, 8 steps
    ]

    for pcp_size, interleave_tokens, prompt_tokens, step_tokens in (
        test_topologies
    ):
        stride = pcp_size * interleave_tokens
        num_steps = prompt_tokens // step_tokens
        scheduler = _make_raiden_scheduler(
            is_producer=True,
            pcp_size=pcp_size,
            interleave_size=interleave_tokens,
        )
        req_id = f"req-r2-1-{pcp_size}-{prompt_tokens}"
        base_uuid = 900000 + pcp_size * 1000 + prompt_tokens
        block_tokens = 16 * pcp_size
        num_blocks = (prompt_tokens + block_tokens - 1) // block_tokens
        block_ids = list(range(200, 200 + num_blocks))

        req = _make_mock_request(
            req_id=req_id,
            prompt_tokens=prompt_tokens,
            computed_tokens=0,
        )

        with _patch_raiden_stage3_env(
            base_uuid=base_uuid, parallelism=pcp_size
        ):
            scheduler.update_state_after_alloc(req, block_ids, 0)
            assert req_id in scheduler._stage3_send_requests
            req_state = scheduler._stage3_send_requests[req_id]
            assert req_state.tokens_sent == 0
            assert req_state.next_chunk_idx == 0

            emitted_chunks: list[connector_metadata.SendMeta] = []
            for step_idx in range(num_steps):
                sched_out = mock.MagicMock(spec=sched_output.SchedulerOutput)
                sched_out.num_scheduled_tokens = {req_id: step_tokens}
                meta = scheduler.build_connector_meta(sched_out)

                assert req_id in meta.reqs_to_send, (
                    f"Step {step_idx}: expected chunk emission for "
                    f"req_id={req_id}"
                )
                chunk_meta = meta.reqs_to_send[req_id]
                emitted_chunks.append(chunk_meta)

                expected_token_base = step_idx * stride
                expected_uuid = tpu_connector.chunk_uuid(base_uuid, step_idx)

                assert chunk_meta.token_base == expected_token_base, (
                    f"Step {step_idx}: expected "
                    f"token_base={expected_token_base}, "
                    f"got {chunk_meta.token_base}"
                )
                assert chunk_meta.num_tokens == stride, (
                    f"Step {step_idx}: expected num_tokens={stride}, "
                    f"got {chunk_meta.num_tokens}"
                )
                assert chunk_meta.is_final is False, (
                    f"Step {step_idx}: intermediate chunk must have "
                    f"is_final=False"
                )
                assert chunk_meta.uuid == expected_uuid, (
                    f"Step {step_idx}: expected uuid={expected_uuid}, "
                    f"got {chunk_meta.uuid}"
                )
                assert chunk_meta.local_block_ids == block_ids

            assert len(emitted_chunks) == num_steps
            assert req_state.tokens_sent == prompt_tokens
            assert req_state.next_chunk_idx == num_steps


def test_rubric_r2_1_multi_stride_and_unaligned_step_accumulation() -> None:
    """Verifies R2.1: Multi-stride steps & unaligned token accumulation.

    Tests:
    1. Multi-stride single step: Scheduling 2048 or 3072 tokens in a single step
       on stride=1024 emits an aggregated chunk covering the multiple completed
       strides.
    2. Sub-stride accumulation: Step increments of 400, 400, 400 tokens
       accumulate until crossing the 1024 stride boundary, then emit [0, 1024),
       carrying over the 176-token remainder.
    """
    # 1. Multi-stride single step
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-1-multistride"
    base_uuid = 555111
    block_ids = list(range(100, 164))
    req = _make_mock_request(
        req_id=req_id, prompt_tokens=4096, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Schedule 2048 tokens in one step (2 strides)
        sched_out = mock.MagicMock(spec=sched_output.SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: 2048}
        meta = scheduler.build_connector_meta(sched_out)

        assert req_id in meta.reqs_to_send
        chunk = meta.reqs_to_send[req_id]
        assert chunk.token_base == 0
        assert chunk.num_tokens == 2048
        assert chunk.is_final is False
        assert chunk.uuid == tpu_connector.chunk_uuid(base_uuid, 0)
        assert scheduler._stage3_send_requests[req_id].tokens_sent == 2048

    # 2. Sub-stride accumulation across uneven steps
    scheduler_acc = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_acc_id = "req-r2-1-accum"
    base_uuid_acc = 555222
    req_acc = _make_mock_request(
        req_id=req_acc_id, prompt_tokens=4096, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid_acc, parallelism=4):
        scheduler_acc.update_state_after_alloc(req_acc, block_ids, 0)

        # Step 1: 400 tokens -> below 1024 stride -> 0 chunks emitted
        s1 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s1.num_scheduled_tokens = {req_acc_id: 400}
        m1 = scheduler_acc.build_connector_meta(s1)
        assert len(m1.reqs_to_send) == 0
        req_st = scheduler_acc._stage3_send_requests[req_acc_id]
        assert req_st.tokens_sent == 0
        assert req_st.num_computed_tokens == 400

        # Step 2: another 400 tokens (800 total) -> still below 1024 -> 0 chunks
        s2 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s2.num_scheduled_tokens = {req_acc_id: 400}
        m2 = scheduler_acc.build_connector_meta(s2)
        assert len(m2.reqs_to_send) == 0
        assert req_st.tokens_sent == 0
        assert req_st.num_computed_tokens == 800

        # Step 3: another 400 tokens (1200 total) -> crosses 1024 stride!
        # Emits [0, 1024), tokens_sent advances to 1024, remainder 176 tokens.
        s3 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s3.num_scheduled_tokens = {req_acc_id: 400}
        m3 = scheduler_acc.build_connector_meta(s3)
        assert req_acc_id in m3.reqs_to_send
        c3 = m3.reqs_to_send[req_acc_id]
        assert c3.token_base == 0
        assert c3.num_tokens == 1024
        assert c3.is_final is False
        assert c3.uuid == tpu_connector.chunk_uuid(base_uuid_acc, 0)
        assert req_st.tokens_sent == 1024
        assert req_st.num_computed_tokens == 1200


def test_rubric_r2_1_concurrent_multi_request_stride_emission() -> None:
    """Verifies R2.1: Concurrent requests maintain separate strides."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    base_uuid = 700000
    block_ids = list(range(100, 164))

    req1 = _make_mock_request(req_id="req-r2-1-c1", prompt_tokens=4096)
    req2 = _make_mock_request(req_id="req-r2-1-c2", prompt_tokens=8192)

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req1, block_ids, 0)
        scheduler.update_state_after_alloc(req2, block_ids, 0)

        # Step 1: req1 gets 1024 (emits), req2 gets 512 (below stride, defers)
        s1 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s1.num_scheduled_tokens = {"req-r2-1-c1": 1024, "req-r2-1-c2": 512}
        m1 = scheduler.build_connector_meta(s1)
        assert "req-r2-1-c1" in m1.reqs_to_send
        assert "req-r2-1-c2" not in m1.reqs_to_send
        assert m1.reqs_to_send["req-r2-1-c1"].token_base == 0
        assert m1.reqs_to_send["req-r2-1-c1"].num_tokens == 1024

        # Step 2: req1 gets 1024 (emits [1024, 2048))
        # req2 gets 1024 (now 1536 tokens, emits [0, 1024))
        s2 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s2.num_scheduled_tokens = {"req-r2-1-c1": 1024, "req-r2-1-c2": 1024}
        m2 = scheduler.build_connector_meta(s2)
        assert "req-r2-1-c1" in m2.reqs_to_send
        assert "req-r2-1-c2" in m2.reqs_to_send
        assert m2.reqs_to_send["req-r2-1-c1"].token_base == 1024
        assert m2.reqs_to_send["req-r2-1-c1"].num_tokens == 1024
        assert m2.reqs_to_send["req-r2-1-c2"].token_base == 0
        assert m2.reqs_to_send["req-r2-1-c2"].num_tokens == 1024


# ==============================================================================
# Rubric Test 2: R2.2 — Monotonic Token Cursor Tracking
# ==============================================================================


def test_rubric_r2_2_monotonic_token_cursor_and_contiguity() -> None:
    """Verifies R2.2: Monotonic cursor, contiguous non-overlapping chunks.

    Asserts that:
    1. tokens_sent is strictly monotonically non-decreasing.
    2. Consecutive chunks [base_i, base_i + len_i) and
       [base_{i+1}, base_{i+1} + len_{i+1}) are strictly contiguous:
       base_{i+1} == base_i + len_i.
    3. Intermediate chunks are mutually exclusive (zero overlap) and partition
       [0, tokens_sent).
    4. next_chunk_idx increments sequentially (0, 1, 2, ...) and each chunk UUID
       matches chunk_uuid(base_uuid, chunk_idx) deterministically.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-2-monotonic"
    base_uuid = 888111
    prompt_tokens = 16384
    block_ids = list(range(100, 356))
    req = _make_mock_request(
        req_id=req_id, prompt_tokens=prompt_tokens, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req, block_ids, 0)
        req_state = scheduler._stage3_send_requests[req_id]

        prev_tokens_sent = 0
        emitted_intervals: list[tuple[int, int]] = []
        emitted_uuids: list[int] = []

        # Execute in varying stride steps summing to 16384 tokens
        step_sizes = [1024, 2048, 1024, 3072, 1024, 2048, 2048, 4096]
        assert sum(step_sizes) == prompt_tokens

        for step_idx, step_tokens in enumerate(step_sizes):
            sched_out = mock.MagicMock(spec=sched_output.SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: step_tokens}
            meta = scheduler.build_connector_meta(sched_out)

            assert req_id in meta.reqs_to_send
            chunk = meta.reqs_to_send[req_id]

            # Monotonicity check
            assert req_state.tokens_sent > prev_tokens_sent, (
                f"tokens_sent must strictly increase on chunk emission: "
                f"{req_state.tokens_sent} > {prev_tokens_sent}"
            )
            assert chunk.token_base == prev_tokens_sent, (
                f"Chunk token_base must equal previous tokens_sent: "
                f"{chunk.token_base} == {prev_tokens_sent}"
            )
            assert chunk.num_tokens == step_tokens, (
                f"Chunk num_tokens mismatch: "
                f"{chunk.num_tokens} != {step_tokens}"
            )

            interval = (chunk.token_base, chunk.token_base + chunk.num_tokens)
            emitted_intervals.append(interval)

            expected_uuid = tpu_connector.chunk_uuid(base_uuid, step_idx)
            assert chunk.uuid == expected_uuid
            emitted_uuids.append(chunk.uuid)

            prev_tokens_sent = req_state.tokens_sent
            assert req_state.next_chunk_idx == step_idx + 1

        # Verify full interval contiguity and mutual exclusivity
        assert len(emitted_intervals) == len(step_sizes)
        for idx in range(len(emitted_intervals) - 1):
            curr_start, curr_end = emitted_intervals[idx]
            next_start, next_end = emitted_intervals[idx + 1]
            assert curr_end == next_start, (
                f"Gap or overlap detected between chunk {idx} "
                f"[{curr_start}, {curr_end}) and chunk {idx+1} "
                f"[{next_start}, {next_end})"
            )

        # Verify zero intersection between any two distinct chunks
        for i in range(len(emitted_intervals)):
            for j in range(i + 1, len(emitted_intervals)):
                s1, e1 = emitted_intervals[i]
                s2, e2 = emitted_intervals[j]
                overlap = max(0, min(e1, e2) - max(s1, s2))
                assert overlap == 0, f"Overlap between chunks {i} and {j}"

        # Verify UUID uniqueness across chunks
        assert len(set(emitted_uuids)) == len(emitted_uuids)


def test_rubric_r2_2_reallocation_preserves_monotonic_state() -> None:
    """Verifies R2.2: Reallocation during prefill preserves tracking progress.

    If update_state_after_alloc is invoked again during multi-step execution,
    existing tokens_sent, next_chunk_idx, and base_uuid must remain preserved.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-2-realloc"
    base_uuid = 444888
    initial_blocks = list(range(100, 164))
    expanded_blocks = list(range(100, 228))

    req = _make_mock_request(
        req_id=req_id, prompt_tokens=8192, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req, initial_blocks, 0)

        # Emit 2 chunks (2048 tokens)
        for _ in range(2):
            s = mock.MagicMock(spec=sched_output.SchedulerOutput)
            s.num_scheduled_tokens = {req_id: 1024}
            scheduler.build_connector_meta(s)

        req_state = scheduler._stage3_send_requests[req_id]
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2
        assert req_state.base_uuid == base_uuid

        # Reallocate with expanded block table
        scheduler.update_state_after_alloc(req, expanded_blocks, 0)

        # Assert cursors were NOT reset
        assert req_state.tokens_sent == 2048
        assert req_state.next_chunk_idx == 2
        assert req_state.base_uuid == base_uuid
        assert req_state.block_ids == expanded_blocks

        # Next chunk should start at token_base=2048 with chunk_idx=2
        s3 = mock.MagicMock(spec=sched_output.SchedulerOutput)
        s3.num_scheduled_tokens = {req_id: 1024}
        meta3 = scheduler.build_connector_meta(s3)
        assert req_id in meta3.reqs_to_send
        c3 = meta3.reqs_to_send[req_id]
        assert c3.token_base == 2048
        assert c3.num_tokens == 1024
        assert c3.uuid == tpu_connector.chunk_uuid(base_uuid, 2)
        assert c3.local_block_ids == expanded_blocks


# ==============================================================================
# Rubric Test 3: R2.3 — Short-Prompt Bypass & Terminal Remainder Emission
# ==============================================================================


def test_rubric_r2_3_short_prompt_bypass_and_terminal_emission() -> None:
    """Verifies R2.3: Short prompts bypass intermediate steps, emit terminally.

    For prompt lengths strictly below the interleave stride:
    1. Single-step and multi-step execution emit ZERO intermediate chunks.
    2. request_finished() emits a single terminal SendMeta([0, N),
       is_final=True) with uuid=chunk_uuid(base_uuid, 0) and cleans up state.
    """
    # All < 1024 stride (PCP=4, interleave 256)
    short_prompt_lengths = [64, 128, 256, 512, 1023]

    for prompt_len in short_prompt_lengths:
        scheduler = _make_raiden_scheduler(
            is_producer=True, pcp_size=4, interleave_size=256
        )
        req_id = f"req-r2-3-short-{prompt_len}"
        base_uuid = 300000 + prompt_len
        block_tokens = 64
        num_blocks = (prompt_len + block_tokens - 1) // block_tokens
        block_ids = list(range(100, 100 + num_blocks))

        req = _make_mock_request(
            req_id=req_id, prompt_tokens=prompt_len, computed_tokens=0
        )

        with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
            scheduler.update_state_after_alloc(req, block_ids, 0)

            # Step 1: scheduled execution -> bypassed!
            s = mock.MagicMock(spec=sched_output.SchedulerOutput)
            s.num_scheduled_tokens = {req_id: prompt_len}
            meta_step = scheduler.build_connector_meta(s)

            assert len(meta_step.reqs_to_send) == 0, (
                f"Short prompt ({prompt_len} tokens) must emit 0 "
                f"intermediate chunks"
            )
            assert (
                scheduler._stage3_send_requests[req_id].tokens_sent == 0
            )

            # Request completion: request_finished emits single terminal chunk
            req.num_computed_tokens = prompt_len
            req.status = vllm_request.RequestStatus.FINISHED_LENGTH_CAPPED
            delay, params = scheduler.request_finished(req, block_ids)
            meta_fin = scheduler.build_connector_meta()

            assert delay is True
            assert params["req_id"] == req_id
            assert params["uuid"] == tpu_connector.chunk_uuid(base_uuid, 0)
            assert params["num_tokens"] == prompt_len

            assert req_id in meta_fin.reqs_to_send
            chunk = meta_fin.reqs_to_send[req_id]
            assert chunk.token_base == 0
            assert chunk.num_tokens == prompt_len
            assert chunk.is_final is True
            assert chunk.uuid == tpu_connector.chunk_uuid(base_uuid, 0)
            assert chunk.local_block_ids == block_ids

            # State cleaned up
            assert req_id not in scheduler._stage3_send_requests


def test_rubric_r2_3_multi_chunk_terminal_remainder_emission() -> None:
    """Verifies R2.3: Terminal remainder emission for multi-chunk requests.

    A 3500-token prompt on stride 1024 emits 3 intermediate chunks [0, 1024),
    [1024, 2048), [2048, 3072) with is_final=False, and on request_finished()
    emits the final remainder [3072, 3500) (428 tokens) with is_final=True.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-3-remainder-3500"
    base_uuid = 999333
    prompt_tokens = 3500
    block_tokens = 64
    num_blocks = (prompt_tokens + block_tokens - 1) // block_tokens
    block_ids = list(range(100, 100 + num_blocks))

    req = _make_mock_request(
        req_id=req_id, prompt_tokens=prompt_tokens, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Emit 3 chunks of 1024 tokens (3072 tokens total)
        for chunk_idx in range(3):
            s = mock.MagicMock(spec=sched_output.SchedulerOutput)
            s.num_scheduled_tokens = {req_id: 1024}
            meta = scheduler.build_connector_meta(s)
            assert req_id in meta.reqs_to_send
            c = meta.reqs_to_send[req_id]
            assert c.token_base == chunk_idx * 1024
            assert c.num_tokens == 1024
            assert c.is_final is False
            assert c.uuid == tpu_connector.chunk_uuid(base_uuid, chunk_idx)

        assert scheduler._stage3_send_requests[req_id].tokens_sent == 3072
        assert scheduler._stage3_send_requests[req_id].next_chunk_idx == 3

        # Finalize request
        req.num_computed_tokens = prompt_tokens
        req.status = vllm_request.RequestStatus.FINISHED_LENGTH_CAPPED
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert params["req_id"] == req_id
        assert params["uuid"] == tpu_connector.chunk_uuid(base_uuid, 3)
        assert params["num_tokens"] == prompt_tokens

        assert req_id in meta_fin.reqs_to_send
        rem_chunk = meta_fin.reqs_to_send[req_id]
        assert rem_chunk.token_base == 3072
        assert rem_chunk.num_tokens == 428
        assert rem_chunk.is_final is True
        assert rem_chunk.uuid == tpu_connector.chunk_uuid(base_uuid, 3)
        assert rem_chunk.local_block_ids == block_ids

        assert req_id not in scheduler._stage3_send_requests


def test_rubric_r2_3_exact_stride_aligned_terminal_completion() -> None:
    """Verifies R2.3: Exact stride-aligned completion emits no empty remainder.

    When all prompt tokens (e.g. 4096 tokens on stride 1024) have already been
    emitted across intermediate chunks, request_finished() must clean up state,
    return (True, params) with the final UUID, and NOT emit a zero-token
    remainder.
    """
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-3-exact-4096"
    base_uuid = 777222
    prompt_tokens = 4096
    block_tokens = 64
    num_blocks = prompt_tokens // block_tokens
    block_ids = list(range(100, 100 + num_blocks))

    req = _make_mock_request(
        req_id=req_id, prompt_tokens=prompt_tokens, computed_tokens=0
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        scheduler.update_state_after_alloc(req, block_ids, 0)

        # Emit 4 chunks covering all 4096 tokens
        for chunk_idx in range(4):
            s = mock.MagicMock(spec=sched_output.SchedulerOutput)
            s.num_scheduled_tokens = {req_id: 1024}
            meta = scheduler.build_connector_meta(s)
            assert req_id in meta.reqs_to_send
            assert meta.reqs_to_send[req_id].token_base == chunk_idx * 1024
            assert meta.reqs_to_send[req_id].num_tokens == 1024
            assert meta.reqs_to_send[req_id].is_final is False
            assert meta.reqs_to_send[req_id].uuid == tpu_connector.chunk_uuid(
                base_uuid, chunk_idx
            )

        assert scheduler._stage3_send_requests[req_id].tokens_sent == 4096

        # Finish request
        req.num_computed_tokens = prompt_tokens
        req.status = vllm_request.RequestStatus.FINISHED_LENGTH_CAPPED
        delay, params = scheduler.request_finished(req, block_ids)
        meta_fin = scheduler.build_connector_meta()

        assert delay is True
        assert params["req_id"] == req_id
        assert params["uuid"] == tpu_connector.chunk_uuid(base_uuid, 4)
        assert params["num_tokens"] == prompt_tokens

        # No empty/zero-token chunk emitted
        assert req_id not in meta_fin.reqs_to_send
        assert len(meta_fin.reqs_to_send) == 0
        assert req_id not in scheduler._stage3_send_requests


def test_rubric_r2_3_untracked_fallback_and_idempotency() -> None:
    """Verifies R2.3: Untracked fallback to monolithic send and dedup."""
    scheduler = _make_raiden_scheduler(
        is_producer=True, pcp_size=4, interleave_size=256
    )
    req_id = "req-r2-3-untracked-idempotent"
    base_uuid = 999888
    prompt_tokens = 2048
    block_tokens = 64
    num_blocks = prompt_tokens // block_tokens
    block_ids = list(range(100, 100 + num_blocks))

    req = _make_mock_request(
        req_id=req_id,
        prompt_tokens=prompt_tokens,
        computed_tokens=prompt_tokens,
        status=vllm_request.RequestStatus.FINISHED_LENGTH_CAPPED,
    )

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        # 1. Fallback for untracked request
        delay1, params1 = scheduler.request_finished(req, block_ids)
        meta1 = scheduler.build_connector_meta()

        assert delay1 is True
        assert params1["uuid"] == base_uuid
        assert req_id in meta1.reqs_to_send
        c1 = meta1.reqs_to_send[req_id]
        assert c1.token_base == 0
        assert c1.num_tokens == prompt_tokens
        assert c1.is_final is True
        assert c1.uuid == base_uuid

        # 2. Duplicate callback returns identical cached params without
        # new emission
        delay2, params2 = scheduler.request_finished(req, block_ids)
        meta2 = scheduler.build_connector_meta()

        assert delay2 is True
        assert params1 == params2
        assert len(meta2.reqs_to_send) == 0


def run_all_rubric_checks() -> bool:
    """Executes all Phase 2 Rubric checks and prints formatted results.

    Runs comprehensive verifications for:
    - R2.1 (Per-Step Aligned Stride Emission)
    - R2.2 (Monotonic Token Cursor Tracking)
    - R2.3 (Short-Prompt Bypass & Terminal Remainder Emission)

    Returns:
        True if all rubric verification checks passed successfully, False
        otherwise.
    """
    banner = "=" * 70
    print(f"\n{banner}")
    print("PHASE 2 RUBRIC VERIFICATION SUITE — vLLM-TorchTPU Raiden Stage-3")
    print(banner)

    all_passed = True

    # R2.1
    print("\n--- Checking R2.1: Per-Step Aligned Stride Emission ---")
    try:
        test_rubric_r2_1_aligned_stride_emission()
        test_rubric_r2_1_multi_stride_and_unaligned_step_accumulation()
        test_rubric_r2_1_concurrent_multi_request_stride_emission()
        print(
            "[PASS] R2.1 — Per-Step Aligned Stride Emission: Verified that the "
            "scheduler emits correctly aligned sub-chunk descriptors whenever "
            "scheduled prefill steps cross interleave stride boundaries across "
            "PCP=2,4,8, handling multi-stride steps, sub-stride accumulation, "
            "and concurrent requests."
        )
    except Exception as error:
        print(f"[FAIL] R2.1 — Per-Step Aligned Stride Emission failed: {error}")
        all_passed = False

    # R2.2
    print("\n--- Checking R2.2: Monotonic Token Cursor Tracking ---")
    try:
        test_rubric_r2_2_monotonic_token_cursor_and_contiguity()
        test_rubric_r2_2_reallocation_preserves_monotonic_state()
        print(
            "[PASS] R2.2 — Monotonic Token Cursor Tracking: Verified that the "
            "scheduler maintains a strictly monotonic tokens_sent cursor, "
            "producing contiguous, non-overlapping token intervals with "
            "deterministic sequential UUIDs and preserving progress during "
            "reallocations."
        )
    except Exception as error:
        print(f"[FAIL] R2.2 — Monotonic Token Cursor Tracking failed: {error}")
        all_passed = False

    # R2.3
    print(
        "\n--- Checking R2.3: Short-Prompt Bypass & Terminal Remainder "
        "Emission ---"
    )
    try:
        test_rubric_r2_3_short_prompt_bypass_and_terminal_emission()
        test_rubric_r2_3_multi_chunk_terminal_remainder_emission()
        test_rubric_r2_3_exact_stride_aligned_terminal_completion()
        test_rubric_r2_3_untracked_fallback_and_idempotency()
        print(
            "[PASS] R2.3 — Short-Prompt Bypass & Terminal Remainder Emission: "
            "Verified that prompts shorter than the interleave stride bypass "
            "intermediate emission, multi-chunk requests emit un-sent "
            "remainder tokens on completion with is_final=True, exact stride "
            "completions avoid empty chunks, and untracked fallbacks remain "
            "safe."
        )
    except Exception as error:
        print(
            f"[FAIL] R2.3 — Short-Prompt Bypass & Terminal Remainder Emission "
            f"failed: {error}"
        )
        all_passed = False

    print(f"\n{banner}")
    if all_passed:
        print("ALL PHASE 2 RUBRIC CHECKS PASSED PERFECTLY!")
        print(banner)
        return True
    else:
        print("PHASE 2 RUBRIC VERIFICATION FAILED.")
        print(banner)
        return False


def test_phase_2_rubric_suite_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Executes the complete Phase 2 Rubric verification and prints banner."""
    with capsys.disabled():
        success = run_all_rubric_checks()
    assert success is True


if __name__ == "__main__":
    success = run_all_rubric_checks()
    sys.exit(0 if success else 1)
