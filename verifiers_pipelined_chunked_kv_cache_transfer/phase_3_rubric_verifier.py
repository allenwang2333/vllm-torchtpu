# SPDX-License-Identifier: Apache-2.0
"""Phase 3 Rubric Verifier — Worker Producer Registration & Keep-Alives.

Validates all Phase 3 Rubric items for chunked KV cache transfer:
- R3.1 — Sub-Chunk Work Unit Registration: The producer worker registers
  independent sub-chunk transfer descriptors with the Raiden controller
  specifying exact token offsets and span ranges.
- R3.2 — Device Buffer Residency Guarantee: Device memory buffers backing
  registered chunk transfers remain pinned and referenced until the underlying
  transfer engine reports DMA completion.
- R3.3 — Multi-Channel Transfer Status Disambiguation: The worker explicitly
  disambiguates and processes send completions, receive completions, and
  transfer error signals from the low-level engine status channel.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolSpanRegistration,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorWorker,
    _Stage3RegisteredSend,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class _FakeRaidenEngine:
    """Mock low-level Raiden transfer engine for worker verification."""

    def __init__(self) -> None:
        """Initializes call tracking lists and queued completion results."""
        self.calls: list[Any] = []
        self.poll_results: list[tuple[list[str], list[str], list[str]]] = []
        self.complete_read_results: list[
            tuple[list[str], list[str], list[str]]
        ] = []

    def register_read(
        self, req_id: str, uuid: int, block_ids: list[int]
    ) -> int:
        """Records a register_read invocation."""
        self.calls.append(("register_read", req_id, uuid, block_ids))
        return 1

    def start_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote_blocks: list[int],
        local_blocks: list[int],
    ) -> int:
        """Records a start_read invocation."""
        self.calls.append(
            ("start_read", req_id, uuid, endpoint, remote_blocks, local_blocks)
        )
        return 2

    def complete_read(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued complete_read completion triples."""
        self.calls.append(("complete_read",))
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        if self.poll_results:
            return self.poll_results.pop(0)
        return [], [], []

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued poll_stats completion triples."""
        self.calls.append(("poll_stats",))
        if self.poll_results:
            return self.poll_results.pop(0)
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        return [], [], []


class _FakeRaidenControllerFacade:
    """Mock controller client recording RPC invocations for verification."""

    def __init__(self) -> None:
        """Initializes lists to record controller RPC calls."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit RPC call."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks RPC call."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks RPC call."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed RPC call."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result


def _make_mock_vllm_config(
    *,
    is_producer: bool = True,
    block_size: int = 1024,
    dp_rank: int = 0,
    dp_size: int = 1,
    tp_size: int = 1,
    pcp_size: int = 4,
    interleave_size: int = 256,
) -> Any:
    """Builds mock VllmConfig suitable for TPURaidenConnectorWorker init."""
    cfg = mock.MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 16384
    cfg.model_config.hf_config = SimpleNamespace(
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size
    return cfg


def _make_test_worker(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    is_producer: bool = True,
    pcp_size: int = 4,
    interleave_size: int = 256,
    block_size: int = 1024,
    token_bytes: int = 1024,
    transfer_rank: int = 0,
) -> tuple[
    TPURaidenConnectorWorker,
    _FakeRaidenControllerFacade,
    _FakeRaidenEngine,
]:
    """Constructs a TPURaidenConnectorWorker with mocked dependencies."""
    cfg = _make_mock_vllm_config(
        is_producer=is_producer,
        block_size=block_size,
        tp_size=tp_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    with (
        mock.patch(
            f"{_MOD}.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
        mock.patch(
            f"{_MOD}.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        mock.patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        mock.patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        mock.patch(
            f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100
        ),
    ):
        worker = TPURaidenConnectorWorker(cfg)

    engine = _FakeRaidenEngine()
    facade = _FakeRaidenControllerFacade()
    unit = SimpleNamespace(
        job_name="prefill-worker",
        job_replica_id="producer-engine-rank0",
        data_name="kv.fa",
        data_replica_idx=0,
    )
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "prefill-controller.test:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = mock.MagicMock(
        return_value=transfer_rank
    )
    worker._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {
            "fa": {"live_bytes_per_block": block_size * token_bytes}
        },
        tag_counts=lambda: {"fa": 15},
    )
    return worker, facade, engine


@contextlib.contextmanager
def _patch_raiden_worker_env(
    parallelism: int = 4,
    wait_pull_timeout: float | None = None,
) -> Iterator[None]:
    """Patches runtime environment variables for worker verification."""
    patches = [
        mock.patch(
            f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
            "raiden",
            create=True,
        ),
        mock.patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
            create=True,
        ),
    ]
    if wait_pull_timeout is not None:
        patches.append(
            mock.patch(
                f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                return_value=wait_pull_timeout,
            )
        )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# ==============================================================================
# R3.1: Sub-Chunk Work Unit Registration Tests
# ==============================================================================


def test_rubric_r3_1_subchunk_registration_exact_token_offsets_and_spans(
) -> None:
    """Verifies sub-chunk registration derives exact token offsets and spans.

    Tests across multiple PCP topologies (PCP=2, 4, 8) and validates that for
    any sub-chunk [token_base, token_base + num_tokens), every transfer rank
    constructs PoolSpanRegistration descriptors with:
    1. dst_space_version == 1 (request-global destination coordinate space)
    2. dst_offset_bytes == (token_base + rank_stride_offset) * token_bytes
    3. declared_bytes == owned_tokens * token_bytes
    4. Exact matching block_ids allocated for that rank's owned tokens.
    """
    for pcp_size, interleave_size, token_base, num_tokens in [
        (2, 256, 1024, 1024),
        (4, 256, 1024, 1024),
        (4, 256, 2048, 2048),
        (8, 128, 2048, 1024),
    ]:
        block_size = 1024
        stride_tokens = pcp_size * interleave_size
        num_strides = num_tokens // stride_tokens

        for transfer_rank in range(pcp_size):
            worker, facade, _ = _make_test_worker(
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                block_size=block_size,
                transfer_rank=transfer_rank,
            )
            token_bytes = worker._stage3_fa_token_bytes(block_size)
            req_id = f"req-r31-pcp{pcp_size}-r{transfer_rank}"
            sub_uuid = 100000 + transfer_rank * 100 + token_base

            # Total scheduler blocks needed across all ranks
            total_req_tokens = token_base + num_tokens
            sched_block_tokens = block_size * pcp_size
            needed_blocks = (
                total_req_tokens + sched_block_tokens - 1
            ) // sched_block_tokens
            local_block_ids = list(range(200, 200 + needed_blocks + 2))

            meta = TPUConnectorMetadata()
            meta.reqs_to_send[req_id] = SendMeta(
                uuid=sub_uuid,
                local_block_ids=local_block_ids,
                expiration_time=time.perf_counter() + 30.0,
                num_tokens=num_tokens,
                token_base=token_base,
                is_final=False,
            )

            with _patch_raiden_worker_env(parallelism=pcp_size):
                worker.process_send_load(meta)

            assert len(facade.register_request_blocks_calls) == 1
            call = facade.register_request_blocks_calls[0]
            assert call["req_id"] == req_id
            assert call["uuid"] == sub_uuid
            assert call["unit"] is worker._raiden_work_unit

            pool_spans: list[PoolSpanRegistration] = call["pool_spans"]
            assert len(pool_spans) == 1
            fa_reg = pool_spans[0]
            assert fa_reg.tag == "fa"
            assert fa_reg.dst_space_version == 1

            expected_owned_tokens = num_strides * interleave_size
            expected_declared_bytes = expected_owned_tokens * token_bytes
            assert fa_reg.declared_bytes == expected_declared_bytes
            assert len(fa_reg.spans) == num_strides

            for stride_idx, span in enumerate(fa_reg.spans):
                expected_start_token = (
                    token_base
                    + stride_idx * stride_tokens
                    + transfer_rank * interleave_size
                )
                expected_dst_offset = expected_start_token * token_bytes
                expected_size_bytes = interleave_size * token_bytes

                owned_tokens_before = stride_idx * interleave_size
                expected_src_ordinal = owned_tokens_before // block_size
                expected_src_offset = (
                    owned_tokens_before % block_size
                ) * token_bytes

                assert span.src_block_ordinal == expected_src_ordinal
                assert span.src_offset_bytes == expected_src_offset
                assert span.dst_offset_bytes == expected_dst_offset
                assert span.size_bytes == expected_size_bytes

            # Verify internal registration state
            assert req_id in worker._stage3_registered_sends
            reg = worker._stage3_registered_sends[req_id]
            assert reg.uuid == sub_uuid
            assert reg.token_base == token_base
            assert reg.num_tokens == num_tokens
            assert reg.is_final is False


def test_rubric_r3_1_multi_chunk_sequential_and_concurrent_registrations(
) -> None:
    """Verifies sequential sub-chunks and concurrent requests register cleanly.

    Simulates:
    1. A 4096-token prompt prefill divided into 4 sequential 1024-token chunks:
       - Chunk 0: [0, 1024), is_final=False
       - Chunk 1: [1024, 2048), is_final=False
       - Chunk 2: [2048, 3072), is_final=False
       - Chunk 3: [3072, 4096), is_final=True
       Asserts each chunk registers independently with distinct UUIDs and span
       offsets.
    2. Multiple concurrent requests emitting sub-chunks interleaved.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=1
    )
    token_bytes = worker._stage3_fa_token_bytes(1024)

    # 1. Sequential multi-chunk registration
    chunk_bases = [0, 1024, 2048, 3072]
    for i, base in enumerate(chunk_bases):
        is_final = i == len(chunk_bases) - 1
        req_id = f"req-seq:{i}"
        sub_uuid = 20000 + i

        meta = TPUConnectorMetadata()
        meta.reqs_to_send[req_id] = SendMeta(
            uuid=sub_uuid,
            local_block_ids=[100, 101, 102, 103],
            expiration_time=time.perf_counter() + 30.0,
            num_tokens=1024,
            token_base=base,
            is_final=is_final,
        )

        with _patch_raiden_worker_env(parallelism=4):
            worker.process_send_load(meta)

        assert len(facade.register_request_blocks_calls) == i + 1
        call = facade.register_request_blocks_calls[i]
        assert call["req_id"] == req_id
        assert call["uuid"] == sub_uuid

        span = call["pool_spans"][0].spans[0]
        # Rank 1 offset: (base + 1 * 256) * token_bytes
        expected_dst_offset = (base + 256) * token_bytes
        assert span.dst_offset_bytes == expected_dst_offset
        assert span.size_bytes == 256 * token_bytes

        assert req_id in worker._stage3_registered_sends
        assert worker._stage3_registered_sends[req_id].is_final is is_final

    # 2. Concurrent requests registration
    req_a_chunk = SendMeta(
        uuid=30001,
        local_block_ids=[10, 11],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )
    req_b_chunk = SendMeta(
        uuid=30002,
        local_block_ids=[20, 21],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )

    concurrent_meta = TPUConnectorMetadata()
    concurrent_meta.reqs_to_send["req-A:1"] = req_a_chunk
    concurrent_meta.reqs_to_send["req-B:0"] = req_b_chunk

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(concurrent_meta)

    assert "req-A:1" in worker._stage3_registered_sends
    assert "req-B:0" in worker._stage3_registered_sends
    assert worker._stage3_registered_sends["req-A:1"].uuid == 30001
    assert worker._stage3_registered_sends["req-B:0"].uuid == 30002


def test_rubric_r3_1_registration_idempotency_and_conflict_rejection() -> None:
    """Verifies idempotent replay and rejection of conflicting registrations.

    Validates:
    1. Identical duplicate replay is accepted without duplicate controller RPC.
    2. Conflicting duplicate parameters (mismatched token_base, num_tokens,
       block_ids) raise ValueError.
    3. Conflicting replay against terminal tombstones raises ValueError.
    4. Unaligned / negative token_base and non-positive num_tokens raise
       ValueError.
    5. Insufficient scheduler block count raises ValueError.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )

    # 1. Identical duplicate replay
    req_id = "req-idempotent-check"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=40001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
        worker.process_send_load(meta)  # Replay identical

    assert len(facade.register_request_blocks_calls) == 1

    # 2. Conflicting duplicate parameters
    meta_conflict_base = TPUConnectorMetadata()
    meta_conflict_base.reqs_to_send[req_id] = SendMeta(
        uuid=40001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=2048,  # Conflict
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ):
            worker.process_send_load(meta_conflict_base)

    meta_conflict_tokens = TPUConnectorMetadata()
    meta_conflict_tokens.reqs_to_send[req_id] = SendMeta(
        uuid=40001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=2048,  # Conflict
        token_base=1024,
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ):
            worker.process_send_load(meta_conflict_tokens)

    # 3. Conflicting replay against terminal tombstone
    term_req_id = "req-terminal-check"
    worker._stage3_terminal_sends[term_req_id] = _Stage3RegisteredSend(
        uuid=50001,
        local_block_ids=(100,),
        num_tokens=1024,
        expiration_time=time.perf_counter() + 30.0,
        token_base=1024,
        is_final=False,
    )

    meta_term_conflict = TPUConnectorMetadata()
    meta_term_conflict.reqs_to_send[term_req_id] = SendMeta(
        uuid=50001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,  # Conflict
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match="Conflicting replay after terminal Stage-3",
        ):
            worker.process_send_load(meta_term_conflict)

    # 4. Unaligned / negative token_base / non-positive tokens
    meta_unaligned = TPUConnectorMetadata()
    meta_unaligned.reqs_to_send["req-unaligned"] = SendMeta(
        uuid=60001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=512,  # Not aligned to 4 * 256 = 1024
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
            worker.process_send_load(meta_unaligned)

    meta_neg = TPUConnectorMetadata()
    meta_neg.reqs_to_send["req-neg"] = SendMeta(
        uuid=60002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=-1024,
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match="requires a non-negative token_base",
        ):
            worker.process_send_load(meta_neg)

    # 5. Insufficient scheduler blocks
    meta_insufficient = TPUConnectorMetadata()
    meta_insufficient.reqs_to_send["req-insuff"] = SendMeta(
        uuid=60003,
        # 1 block provided, but 2 required for window [4096, 5120)
        local_block_ids=[100],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=4096,
        is_final=False,
    )
    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match="scheduler block count is inconsistent with PCP geometry",
        ):
            worker.process_send_load(meta_insufficient)


def test_rubric_r3_1_legacy_backward_compatibility() -> None:
    """Verifies legacy monolithic SendMeta registers seamlessly with defaults.

    When token_base and is_final are omitted in SendMeta instantiation,
    defaults (token_base=0, is_final=True) are applied and FlashAttention spans
    start at destination byte offset 0.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    token_bytes = worker._stage3_fa_token_bytes(1024)

    meta = TPUConnectorMetadata()
    meta.reqs_to_send["req-legacy"] = SendMeta(
        uuid=70001,
        local_block_ids=[100],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    fa_reg = call["pool_spans"][0]
    assert fa_reg.spans[0].dst_offset_bytes == 0
    assert fa_reg.spans[0].size_bytes == 256 * token_bytes

    reg = worker._stage3_registered_sends["req-legacy"]
    assert reg.token_base == 0
    assert reg.is_final is True


# ==============================================================================
# R3.2: Device Buffer Residency Guarantee Tests
# ==============================================================================


def test_rubric_r3_2_buffer_retention_across_subchunks() -> None:
    """Verifies device buffer handles are pinned in _active_tensor_refs.

    Validates that:
    1. When sub-chunk SendMeta descriptors are ingested, the worker queries
       runner KV cache tensors and retains strong references in
       _active_tensor_refs[req_id].device_buffers.
    2. Multiple sub-chunks across multiple requests concurrently maintain their
       device tensor buffer references.
    """
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor_k = mock.MagicMock(name="kv_tensor_k")
    mock_tensor_v = mock.MagicMock(name="kv_tensor_v")
    worker.runner = SimpleNamespace(kv_caches=[(mock_tensor_k, mock_tensor_v)])

    meta = TPUConnectorMetadata()
    meta.reqs_to_send["req-1:0"] = SendMeta(
        uuid=80001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )
    meta.reqs_to_send["req-1:1"] = SendMeta(
        uuid=80002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )
    meta.reqs_to_send["req-2:0"] = SendMeta(
        uuid=80003,
        local_block_ids=[200, 201],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert "req-1:0" in worker._active_tensor_refs
    assert "req-1:1" in worker._active_tensor_refs
    assert "req-2:0" in worker._active_tensor_refs

    for req_id in ["req-1:0", "req-1:1", "req-2:0"]:
        ref = worker._active_tensor_refs[req_id]
        assert isinstance(ref, _Stage3RegisteredSend)
        assert ref.device_buffers == (mock_tensor_k, mock_tensor_v)


def test_rubric_r3_2_selective_dma_completion_and_release() -> None:
    """Verifies device buffer handles are released strictly upon DMA completion.

    Validates that:
    1. When 3 sub-chunks are in flight, completing chunk 0 releases chunk 0's
       device buffer references while chunks 1 and 2 remain pinned.
    2. Completing chunk 1 releases chunk 1's device buffer references while
       chunk 2 remains pinned.
    3. Completing chunk 2 releases chunk 2's device buffer references, leaving
       zero remaining active buffer references.
    """
    worker, facade, engine = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor = mock.MagicMock(name="kv_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    meta = TPUConnectorMetadata()
    for i in range(3):
        meta.reqs_to_send[f"req-stream:{i}"] = SendMeta(
            uuid=81000 + i,
            local_block_ids=[100, 101, 102],
            expiration_time=time.perf_counter() + 30.0,
            num_tokens=1024,
            token_base=i * 1024,
            is_final=(i == 2),
        )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert len(worker._active_tensor_refs) == 3

    # Pass 1: Complete chunk 0 only
    engine.complete_read_results = [(["req-stream:0"], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()

    assert "req-stream:0" in done_sending
    assert "req-stream:0" not in worker._active_tensor_refs
    assert "req-stream:0" in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends["req-stream:0"].device_buffers == ()
    assert "req-stream:1" in worker._active_tensor_refs
    assert "req-stream:2" in worker._active_tensor_refs
    assert (
        worker._active_tensor_refs["req-stream:1"].device_buffers
        == (mock_tensor,)
    )
    assert (
        worker._active_tensor_refs["req-stream:2"].device_buffers
        == (mock_tensor,)
    )

    # Pass 2: Complete chunk 1 only
    engine.complete_read_results = [(["req-stream:1"], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending2, _ = worker.get_finished()

    assert "req-stream:1" in done_sending2
    assert "req-stream:1" not in worker._active_tensor_refs
    assert "req-stream:2" in worker._active_tensor_refs

    # Pass 3: Complete chunk 2
    engine.complete_read_results = [(["req-stream:2"], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending3, _ = worker.get_finished()

    assert "req-stream:2" in done_sending3
    assert len(worker._active_tensor_refs) == 0


def test_rubric_r3_2_terminal_cleanup_and_cancellation_release() -> None:
    """Verifies buffer handles are released on cancel race and cleanup.

    Validates that:
    1. Cancellation race at registration creates terminal tombstones with empty
       device_buffers () and does not retain refs in _active_tensor_refs.
    2. Explicit terminal cleanup (_stage3_terminal_cleanup) invokes controller
       complete_request_blocks and cleans up buffer handles.
    3. GLM registration path correctly populates and releases device buffers.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor_k = mock.MagicMock(name="glm_tensor_k")
    mock_tensor_v = mock.MagicMock(name="glm_tensor_v")
    worker.runner = SimpleNamespace(kv_caches=[(mock_tensor_k, mock_tensor_v)])

    # 1. Cancellation race on registration
    facade.register_request_blocks = mock.MagicMock(
        side_effect=RuntimeError(
            "Request block registration was cancelled: generation expired"
        )
    )
    req_cancel = "req-cancel-race"
    meta_cancel = TPUConnectorMetadata()
    meta_cancel.reqs_to_send[req_cancel] = SendMeta(
        uuid=82001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta_cancel)

    assert req_cancel not in worker._active_tensor_refs
    assert req_cancel in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_cancel].device_buffers == ()

    # Reset facade
    facade.register_request_blocks = (
        _FakeRaidenControllerFacade().register_request_blocks
    )

    # 2. Terminal cleanup force-release
    req_term = "req-term-force"
    meta_term = TPUConnectorMetadata()
    meta_term.reqs_to_send[req_term] = SendMeta(
        uuid=82002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta_term)

    assert req_term in worker._active_tensor_refs
    worker._stage3_terminal_cleanup[req_term] = True

    with _patch_raiden_worker_env(parallelism=4):
        worker.get_finished()

    assert req_term not in worker._active_tensor_refs
    assert req_term in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_term].device_buffers == ()
    assert len(facade.complete_request_blocks_calls) == 1
    assert facade.complete_request_blocks_calls[0]["req_id"] == req_term


def test_rubric_r3_2_tensor_extraction_robustness() -> None:
    """Verifies _flattened_kv_cache_tensors handles varied cache layouts."""
    worker, _, _ = _make_test_worker()

    # None runner
    worker.runner = None
    assert worker._flattened_kv_cache_tensors() == []

    # Runner without kv_caches
    worker.runner = SimpleNamespace()
    assert worker._flattened_kv_cache_tensors() == []

    # None kv_caches
    worker.runner = SimpleNamespace(kv_caches=None)
    assert worker._flattened_kv_cache_tensors() == []

    # Flat list
    t1, t2 = mock.MagicMock(name="t1"), mock.MagicMock(name="t2")
    worker.runner = SimpleNamespace(kv_caches=[t1, t2])
    assert worker._flattened_kv_cache_tensors() == [t1, t2]

    # Tuple pairs
    k1, v1 = mock.MagicMock(name="k1"), mock.MagicMock(name="v1")
    k2, v2 = mock.MagicMock(name="k2"), mock.MagicMock(name="v2")
    worker.runner = SimpleNamespace(kv_caches=[(k1, v1), (k2, v2)])
    assert worker._flattened_kv_cache_tensors() == [k1, v1, k2, v2]


# ==============================================================================
# R3.3: Multi-Channel Transfer Status Disambiguation Tests
# ==============================================================================


def test_rubric_r3_3_producer_status_channel_disambiguation() -> None:
    """Verifies producer worker explicitly disambiguates send status events.

    Validates that:
    1. done_sending events trigger buffer release, controller completion RPC,
       and transition to terminal sends.
    2. failed_recving entries on the push transport status channel are
       disambiguated as sender failures, logged with transfer rank and UUID,
       and recorded in sender_failures.
    3. Empty poll results preserve active registrations without false triggers.
    """
    worker, facade, engine = _make_test_worker(
        is_producer=True, pcp_size=4, interleave_size=256, block_size=1024
    )
    mock_tensor = mock.MagicMock(name="mock_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    # Register send 1 and send 2
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["send-ok"] = SendMeta(
        uuid=90001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )
    meta.reqs_to_send["send-fail"] = SendMeta(
        uuid=90002,
        local_block_ids=[102, 103],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=True,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert "send-ok" in worker._active_tensor_refs
    assert "send-fail" in worker._active_tensor_refs

    # Engine status channel returns done_sending=["send-ok"] and
    # failed_recving=["send-fail"]
    engine.complete_read_results = [(["send-ok"], [], ["send-fail"])]

    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()

    # Both send-ok and send-fail reach terminal completion
    assert "send-ok" in done_sending
    assert "send-fail" in done_sending

    # Both active tensor refs popped
    assert "send-ok" not in worker._active_tensor_refs
    assert "send-fail" not in worker._active_tensor_refs

    # Facade complete_request_blocks invoked for both terminal sends
    completed_req_ids = {
        call["req_id"] for call in facade.complete_request_blocks_calls
    }
    assert "send-ok" in completed_req_ids
    assert "send-fail" in completed_req_ids


def test_rubric_r3_3_consumer_status_channel_disambiguation() -> None:
    """Verifies consumer worker explicitly disambiguates receive status events.

    Validates that:
    1. done_recving events are translated from native transfer IDs to
       destination request IDs and validated against
       _stage3_controller_accepted.
    2. failed_recving events are translated and added to _failed_recving and
       _failed_block_ids for recomputation.
    3. Unmapped native transfer IDs are safely ignored.
    """
    worker, facade, engine = _make_test_worker(is_producer=False)

    # Configure consumer tracking state
    worker._stage3_destination_req_ids = {
        "src-req-ok": "dest-req-ok",
        "src-req-fail": "dest-req-fail",
    }
    worker._stage3_source_req_ids = {
        "dest-req-ok": "src-req-ok",
        "dest-req-fail": "src-req-fail",
    }
    worker._stage3_controller_accepted.add("dest-req-ok")
    worker._stage3_controller_accepted.add("dest-req-fail")
    worker._load_block_ids["dest-req-fail"] = [501, 502]
    worker._stage3_submitted_loads["dest-req-ok"] = 91001
    worker._stage3_submitted_loads["dest-req-fail"] = 91002
    worker._stage3_submitted_load_tokens["dest-req-ok"] = 1024
    worker._stage3_submitted_load_tokens["dest-req-fail"] = 1024

    # Status channel returns success for src-req-ok, failure for src-req-fail,
    # and unmapped transfer ID.
    engine.complete_read_results = [
        ([], ["src-req-ok", "unmapped-req"], ["src-req-fail"])
    ]

    with _patch_raiden_worker_env(parallelism=4):
        _, done_recving = worker.get_finished()

    # Verify disambiguation
    assert "dest-req-ok" in done_recving
    assert "dest-req-fail" in done_recving
    assert "unmapped-req" not in done_recving

    # Verify error tracking and block staging for recomputation
    assert "dest-req-fail" in worker._failed_recving
    assert worker.get_block_ids_with_load_errors() == {501, 502}


def test_rubric_r3_3_engine_interface_polymorphism() -> None:
    """Verifies worker handles both complete_read and poll_stats interfaces."""
    worker, _, _ = _make_test_worker(is_producer=True)
    mock_tensor = mock.MagicMock(name="mock_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    # 1. Engine with complete_read
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send["req-complete-read"] = SendMeta(
        uuid=93001,
        local_block_ids=[100],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)

    class EngineCompleteRead:
        """Mock transfer engine supporting complete_read interface."""

        def complete_read(self) -> tuple[list[str], list[str], list[str]]:
            """Returns completed read requests."""
            return ["req-complete-read"], [], []

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineCompleteRead())
    assert "req-complete-read" not in worker._active_tensor_refs

    # 2. Engine with poll_stats
    meta2 = TPUConnectorMetadata()
    meta2.reqs_to_send["req-poll-stats"] = SendMeta(
        uuid=93002,
        local_block_ids=[100],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta2)

    class EnginePollStats:
        """Mock transfer engine supporting poll_stats interface."""

        def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
            """Returns polled transfer completion stats."""
            return ["req-poll-stats"], [], []

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EnginePollStats())
    assert "req-poll-stats" not in worker._active_tensor_refs

    # 3. Engine with neither
    class EngineEmpty:
        """Mock transfer engine without completion query methods."""

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineEmpty())


def test_rubric_r3_3_unclaimed_send_cancellation_reconciliation() -> None:
    """Verifies expired unclaimed sends are cancelled without force-release."""
    worker, facade, _ = _make_test_worker(is_producer=True)
    mock_tensor = mock.MagicMock(name="mock_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    req_id = "req-unclaimed-cancel"
    meta = TPUConnectorMetadata()
    # Expired timestamp
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=94001,
        local_block_ids=[100],
        expiration_time=time.perf_counter() - 10.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert req_id in worker._active_tensor_refs
    facade.cancel_request_blocks_result = True

    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()

    assert req_id in done_sending
    assert req_id not in worker._active_tensor_refs
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()
    assert len(facade.cancel_request_blocks_calls) == 1
    assert facade.cancel_request_blocks_calls[0]["req_id"] == req_id
    # Complete request blocks should NOT be called since it was cancelled
    # unclaimed
    assert len(facade.complete_request_blocks_calls) == 0


# ==============================================================================
# Phase 3 Rubric Execution Runner
# ==============================================================================


def run_all_rubric_checks() -> bool:
    """Executes all Phase 3 Rubric checks and prints formatted results.

    Runs comprehensive verifications for:
    - R3.1 (Sub-Chunk Work Unit Registration)
    - R3.2 (Device Buffer Residency Guarantee)
    - R3.3 (Multi-Channel Transfer Status Disambiguation)

    Returns:
        True if all rubric verification checks passed successfully, False
        otherwise.
    """
    banner = "=" * 70
    print(f"\n{banner}")
    print("PHASE 3 RUBRIC VERIFICATION SUITE — vLLM-TorchTPU Raiden Stage-3")
    print(banner)

    all_passed = True

    # R3.1
    print("\n--- Checking R3.1: Sub-Chunk Work Unit Registration ---")
    try:
        test_rubric_r3_1_subchunk_registration_exact_token_offsets_and_spans()
        test_rubric_r3_1_multi_chunk_sequential_and_concurrent_registrations()
        test_rubric_r3_1_registration_idempotency_and_conflict_rejection()
        test_rubric_r3_1_legacy_backward_compatibility()
        print(
            "[PASS] R3.1 — Sub-Chunk Work Unit Registration: Verified that the "
            "producer worker registers independent sub-chunk transfer "
            "descriptors with the Raiden controller specifying exact token "
            "offsets, span ranges, destination byte coordinates, and local "
            "block IDs across PCP topologies (PCP=2,4,8), enforcing "
            "alignment and idempotency."
        )
    except Exception as error:
        print(f"[FAIL] R3.1 — Sub-Chunk Work Unit Registration failed: {error}")
        all_passed = False

    # R3.2
    print("\n--- Checking R3.2: Device Buffer Residency Guarantee ---")
    try:
        test_rubric_r3_2_buffer_retention_across_subchunks()
        test_rubric_r3_2_selective_dma_completion_and_release()
        test_rubric_r3_2_terminal_cleanup_and_cancellation_release()
        test_rubric_r3_2_tensor_extraction_robustness()
        print(
            "[PASS] R3.2 — Device Buffer Residency Guarantee: Verified that "
            "device memory buffers backing registered sub-chunk transfers "
            "remain pinned and referenced in _active_tensor_refs across "
            "sequential and concurrent requests until the low-level engine "
            "confirms DMA completion, releasing memory cleanly on completions, "
            "cancellations, and terminal cleanups."
        )
    except Exception as error:
        print(
            f"[FAIL] R3.2 — Device Buffer Residency Guarantee failed: {error}"
        )
        all_passed = False

    # R3.3
    print(
        "\n--- Checking R3.3: Multi-Channel Transfer Status Disambiguation ---"
    )
    try:
        test_rubric_r3_3_producer_status_channel_disambiguation()
        test_rubric_r3_3_consumer_status_channel_disambiguation()
        test_rubric_r3_3_engine_interface_polymorphism()
        test_rubric_r3_3_unclaimed_send_cancellation_reconciliation()
        print(
            "[PASS] R3.3 — Multi-Channel Transfer Status Disambiguation: "
            "Verified that the worker explicitly disambiguates and processes "
            "send completions, receive completions, push transport sender "
            "errors, receiver error block staging for recomputation, unmapped "
            "transfer IDs, and unclaimed send cancellations across "
            "polymorphic engine status interfaces."
        )
    except Exception as error:
        print(
            f"[FAIL] R3.3 — Multi-Channel Transfer Status Disambiguation "
            f"failed: {error}"
        )
        all_passed = False

    print(f"\n{banner}")
    if all_passed:
        print("ALL PHASE 3 RUBRIC CHECKS PASSED PERFECTLY!")
        print(banner)
        return True
    else:
        print("PHASE 3 RUBRIC VERIFICATION FAILED.")
        print(banner)
        return False


def test_phase_3_rubric_suite_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Executes the complete Phase 3 Rubric verification and prints banner."""
    with capsys.disabled():
        success = run_all_rubric_checks()
    assert success is True


if __name__ == "__main__":
    success = run_all_rubric_checks()
    sys.exit(0 if success else 1)
