# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Stage-3 worker consumer destination staging.

Verifies that TPURaidenConnectorWorker pre-allocates destination pages
matching ceil(total_tokens / block_size), tracks received sub-chunk intervals
in _stage3_recv_intervals across out-of-order and sequential completions,
dispatches start_read with chunk coordinates, and cleans up tracking state
upon request completion.
"""

import math
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnector,
    TPURaidenConnectorWorker,
    _validate_interval_tiling,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class _FakeRaidenEngine:
    """Mock low-level Raiden transfer engine for consumer worker testing."""

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
    """Mock controller client recording RPC invocations."""

    def __init__(self) -> None:
        """Initializes lists to record controller RPC calls."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.start_transfer_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True
        self.start_transfer_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit invocation."""
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks invocation."""
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks invocation."""
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed invocation."""
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

    def start_transfer(self, **kwargs: Any) -> bool:
        """Records a start_transfer invocation."""
        self.start_transfer_calls.append(kwargs)
        return self.start_transfer_result


def _make_mock_consumer_vllm_config(
    *,
    block_size: int = 1024,
    dp_rank: int = 0,
    dp_size: int = 4,
    tp_size: int = 1,
) -> Any:
    """Creates mock VllmConfig tailored for Stage-3 consumer decode tests."""
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = False
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.max_model_len = 16384
    cfg.model_config.hf_config = SimpleNamespace(
        num_key_value_heads=8,
        head_dim=128,
    )
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 256
    return cfg


def _make_consumer_test_worker(
    *,
    block_size: int = 1024,
    dp_rank: int = 0,
    dp_size: int = 4,
    tp_size: int = 1,
) -> tuple[
    TPURaidenConnectorWorker, _FakeRaidenControllerFacade, _FakeRaidenEngine
]:
    """Constructs consumer TPURaidenConnectorWorker with mocked state."""
    cfg = _make_mock_consumer_vllm_config(
        block_size=block_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        tp_size=tp_size,
    )
    with (
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
        patch(
            f"{_MOD}.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        worker = TPURaidenConnectorWorker(cfg)

    engine = _FakeRaidenEngine()
    facade = _FakeRaidenControllerFacade()
    unit = SimpleNamespace(
        job_name="decode-worker",
        job_replica_id="consumer-engine-rank0",
        data_name="kv.fa",
        data_replica_idx=dp_rank,
    )
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "decode-controller.test:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=0)
    worker._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {
            "fa": {"live_bytes_per_block": block_size * 1024}
        },
        tag_counts=lambda: {"fa": 15},
    )
    return worker, facade, engine


def test_consumer_accumulates_ranges() -> None:
    """Simulates out-of-order chunk arrivals for chunks 1 and 0.

    Verifies that self._stage3_recv_intervals[req_id] accumulates both
    intervals accurately regardless of arrival order.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-out-of-order"
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    assert worker._stage3_recv_intervals[req_id] == []

    # 1. Chunk 1 arrives first: [1024, 2048)
    worker._start_stage3_recv_if_ready(
        req_id="chunk-1",
        uuid=101,
        endpoint="10.0.0.1:9000",
        remote_blocks=[10, 11],
        local_blocks=[0, 1],
        token_base=1024,
        num_tokens=1024,
        engine=engine,
    )
    worker._stage3_in_flight_recv_chunks["chunk-1"] = (req_id, 1024, 1024)

    # Poll completion for chunk 1
    engine.complete_read_results = [([], ["chunk-1"], [])]
    worker._poll_finished(engine)

    assert (1024, 2048) in worker._stage3_recv_intervals[req_id]
    assert len(worker._stage3_recv_intervals[req_id]) == 1

    # 2. Chunk 0 arrives second: [0, 1024)
    worker._start_stage3_recv_if_ready(
        req_id="chunk-0",
        uuid=100,
        endpoint="10.0.0.1:9000",
        remote_blocks=[8, 9],
        local_blocks=[2, 3],
        token_base=0,
        num_tokens=1024,
        engine=engine,
    )
    worker._stage3_in_flight_recv_chunks["chunk-0"] = (req_id, 0, 1024)

    # Poll completion for chunk 0
    engine.complete_read_results = [([], ["chunk-0"], [])]
    worker._poll_finished(engine)

    assert (1024, 2048) in worker._stage3_recv_intervals[req_id]
    assert (0, 1024) in worker._stage3_recv_intervals[req_id]
    assert len(worker._stage3_recv_intervals[req_id]) == 2


def test_destination_pages_preallocated() -> None:
    """Verifies allocation size matches ceil(total_tokens / block_size).

    Asserts that destination pages are calculated identically regardless of the
    number of sub-chunks the prompt will be delivered across.
    """
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    test_cases = [
        (512, 1024, 1),
        (1024, 1024, 1),
        (1025, 1024, 2),
        (2048, 1024, 2),
        (3500, 1024, 4),
        (4096, 1024, 4),
        (4097, 1024, 5),
        (8192, 1024, 8),
    ]

    for total_tokens, block_size, expected_pages in test_cases:
        worker.vllm_config.cache_config.block_size = block_size
        req = SimpleNamespace(
            request_id=f"req-alloc-{total_tokens}",
            kv_transfer_params={"num_tokens": total_tokens},
        )
        pages = worker._update_stage3_state_after_alloc(req)
        assert pages == expected_pages
        assert pages == math.ceil(total_tokens / block_size)
        assert worker._stage3_recv_intervals[f"req-alloc-{total_tokens}"] == []


def test_subchunk_start_read_dispatch() -> None:
    """Verifies start_read dispatch with 50-bit UUID and block IDs."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    worker._start_stage3_recv_if_ready(
        req_id="chunk-test-dispatch",
        uuid=54321,
        endpoint="10.0.0.2:9100",
        remote_blocks=[5, 6],
        local_blocks=[12, 13],
        token_base=2048,
        num_tokens=1024,
        engine=engine,
    )

    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call[0] == "start_read"
    assert call[1] == "chunk-test-dispatch"
    assert call[2] == 54321
    assert call[3] == "10.0.0.2:9100"
    assert call[4] == [5, 6]
    assert call[5] == [12, 13]
    assert worker._stage3_in_flight_recv_chunks["chunk-test-dispatch"] == (
        "chunk-test-dispatch",
        2048,
        1024,
    )


def test_consumer_accumulates_three_sequential_chunks() -> None:
    """Verifies sequential accumulation across 3 sub-chunks in sequence."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-3-chunks"
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=3072
    )

    chunks = [
        ("c0", 0, 1024),
        ("c1", 1024, 1024),
        ("c2", 2048, 1024),
    ]

    for chunk_id, base, num in chunks:
        worker._start_stage3_recv_if_ready(
            req_id=chunk_id,
            uuid=200 + base,
            endpoint="10.0.0.1:9000",
            token_base=base,
            num_tokens=num,
            engine=engine,
        )
        worker._stage3_in_flight_recv_chunks[chunk_id] = (req_id, base, num)
        engine.complete_read_results = [([], [chunk_id], [])]
        worker._poll_finished(engine)

    intervals = worker._stage3_recv_intervals[req_id]
    assert len(intervals) == 3
    assert (0, 1024) in intervals
    assert (1024, 2048) in intervals
    assert (2048, 3072) in intervals


def test_legacy_consumer_process_send_load_accumulates_intervals() -> None:
    """Verifies non-Stage-3 process_send_load path registers intervals."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=False):
        meta = TPUConnectorMetadata(
            reqs_to_load={
                "legacy-req": LoadMeta(
                    uuid=777,
                    local_block_ids=[0, 1],
                    remote_block_ids=[10, 11],
                    remote_host="10.0.0.1",
                    remote_port=9000,
                    token_base=0,
                    is_final=True,
                )
            }
        )
        # Attach num_tokens for the test
        meta.reqs_to_load["legacy-req"].num_tokens = 2048

        worker.process_send_load(meta)
        engine.complete_read_results = [([], ["legacy-req"], [])]
        worker._poll_finished(engine)

        assert (0, 2048) in worker._stage3_recv_intervals["legacy-req"]


def test_consumer_cleanup_purges_recv_intervals() -> None:
    """Verifies get_finished cleans up _stage3_recv_intervals on retirement."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-clean"
    worker._stage3_recv_intervals[req_id] = [(0, 1024), (1024, 2048)]
    worker._stage3_in_flight_recv_chunks[req_id] = (req_id, 0, 1024)
    worker._stage3_recv_chunk_meta[req_id] = (0, 1024)

    # Finished and acknowledged by scheduler cleanup
    worker.get_finished(finished_req_ids={req_id})

    assert req_id not in worker._stage3_recv_intervals
    assert req_id not in worker._stage3_in_flight_recv_chunks
    assert req_id not in worker._stage3_recv_chunk_meta


def test_update_stage3_state_after_alloc_calling_conventions() -> None:
    """Verifies _update_stage3_state_after_alloc polymorphic calling styles."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    # 1. Dict-backed kv_transfer_params
    req_obj = SimpleNamespace(
        request_id="req-dict",
        kv_transfer_params={"num_tokens": 4096},
    )
    pages = worker._update_stage3_state_after_alloc(req_obj)
    assert pages == 4
    assert worker._stage3_recv_intervals["req-dict"] == []

    # 2. None kv_transfer_params with explicit expected_tokens
    req_none_params = SimpleNamespace(
        request_id="req-none-params",
        kv_transfer_params=None,
    )
    pages = worker._update_stage3_state_after_alloc(
        req_none_params, expected_tokens=3072
    )
    assert pages == 3
    assert worker._stage3_recv_intervals["req-none-params"] == []

    # 3. String request identifier positional with keyword tokens
    pages = worker._update_stage3_state_after_alloc(
        "req-str-pos", expected_tokens=2048
    )
    assert pages == 2
    assert worker._stage3_recv_intervals["req-str-pos"] == []

    # 4. Keyword arguments
    pages = worker._update_stage3_state_after_alloc(
        req_id="req-kwarg", expected_tokens=1024
    )
    assert pages == 1
    assert worker._stage3_recv_intervals["req-kwarg"] == []

    # 5. Positional arguments with extra args
    pages = worker._update_stage3_state_after_alloc(
        "req-extra-args",
        [1, 2],
        "extra1",
        "extra2",
        req_id="req-extra-args",
        expected_tokens=1024,
    )
    assert pages == 1
    assert worker._stage3_recv_intervals["req-extra-args"] == []


def test_update_stage3_state_after_alloc_malformed_and_empty_params() -> None:
    """Verifies allocation update handles zero or negative tokens."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    # 1. Zero tokens
    pages = worker._update_stage3_state_after_alloc(
        req_id="req-zero", expected_tokens=0
    )
    assert pages == 0
    assert worker._stage3_recv_intervals["req-zero"] == []

    # 2. Negative tokens
    pages = worker._update_stage3_state_after_alloc(
        req_id="req-neg", expected_tokens=-500
    )
    assert pages == 0
    assert worker._stage3_recv_intervals["req-neg"] == []

    # 3. None tokens and no params
    pages = worker._update_stage3_state_after_alloc(req_id="req-none-tok")
    assert pages == 0
    assert worker._stage3_recv_intervals["req-none-tok"] == []

    # 4. Malformed non-integer string in expected_tokens
    pages = worker._update_stage3_state_after_alloc(
        req_id="req-bad-str",
        expected_tokens="not_an_int",  # type: ignore[arg-type]
    )
    assert pages == 0
    assert worker._stage3_recv_intervals["req-bad-str"] == []


def test_register_in_flight_recv_chunk_coordinates_and_aliases() -> None:
    """Verifies coordinate tracking and alias mapping in _register_in_flight."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    worker._register_in_flight_recv_chunk(
        transfer_id="transfer-chunk-1",
        destination_req_id="dest-req-1",
        token_base=1024,
        num_tokens=1024,
        alias_id="alias-chunk-1",
    )

    # Destination interval accumulator initialized
    assert "dest-req-1" in worker._stage3_recv_intervals
    assert worker._stage3_recv_intervals["dest-req-1"] == []

    # In-flight chunk maps transfer ID and alias ID to coordinates
    assert worker._stage3_in_flight_recv_chunks["transfer-chunk-1"] == (
        "dest-req-1",
        1024,
        1024,
    )
    assert worker._stage3_in_flight_recv_chunks["alias-chunk-1"] == (
        "dest-req-1",
        1024,
        1024,
    )
    assert worker._stage3_recv_chunk_meta["dest-req-1"] == (1024, 1024)


def test_register_in_flight_recv_chunk_invalid_inputs_ignored() -> None:
    """Verifies invalid coordinates are ignored during chunk registration."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    # 1. Zero num_tokens: should NOT register in-flight chunk
    worker._register_in_flight_recv_chunk(
        transfer_id="chunk-zero-tok",
        destination_req_id="req-dest-1",
        token_base=0,
        num_tokens=0,
    )
    assert "chunk-zero-tok" not in worker._stage3_in_flight_recv_chunks

    # 2. Negative num_tokens: should NOT register in-flight chunk
    worker._register_in_flight_recv_chunk(
        transfer_id="chunk-neg-tok",
        destination_req_id="req-dest-2",
        token_base=0,
        num_tokens=-100,
    )
    assert "chunk-zero-tok" not in worker._stage3_in_flight_recv_chunks

    # 3. Negative token_base: should NOT register in-flight chunk
    worker._register_in_flight_recv_chunk(
        transfer_id="chunk-neg-base",
        destination_req_id="req-dest-3",
        token_base=-10,
        num_tokens=1024,
    )
    assert "chunk-neg-base" not in worker._stage3_in_flight_recv_chunks


def test_subchunk_start_read_with_destination_alias_resolution() -> None:
    """Verifies _start_stage3_recv_if_ready resolves destination alias."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    # Map producer transfer ID prod-chunk-1 to consumer consumer-req-42
    worker._stage3_destination_req_ids["prod-chunk-1"] = "consumer-req-42"

    worker._start_stage3_recv_if_ready(
        req_id="prod-chunk-1",
        uuid=88888,
        endpoint="10.0.0.5:9100",
        remote_blocks=[20, 21],
        local_blocks=[30, 31],
        token_base=0,
        num_tokens=1024,
        engine=engine,
    )

    # Assert start_read was called on the engine
    assert len(engine.calls) == 1
    assert engine.calls[0] == (
        "start_read",
        "prod-chunk-1",
        88888,
        "10.0.0.5:9100",
        [20, 21],
        [30, 31],
    )

    # Assert in-flight chunk tracks destination ID consumer-req-42
    assert worker._stage3_in_flight_recv_chunks["prod-chunk-1"] == (
        "consumer-req-42",
        0,
        1024,
    )
    assert worker._stage3_in_flight_recv_chunks["consumer-req-42"] == (
        "consumer-req-42",
        0,
        1024,
    )


def test_consumer_duplicate_chunk_completion_idempotent() -> None:
    """Verifies duplicate chunk completions do not insert duplicate ranges."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    req_id = "req-idempotent"
    worker._stage3_recv_intervals[req_id] = []

    # Register chunk
    worker._stage3_in_flight_recv_chunks["chunk-dup"] = (req_id, 0, 1024)

    # First completion
    worker._record_stage3_recv_chunk_completion("chunk-dup")
    assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]

    # Simulate duplicate arrival / recording
    worker._stage3_in_flight_recv_chunks["chunk-dup-2"] = (req_id, 0, 1024)
    worker._record_stage3_recv_chunk_completion("chunk-dup-2")
    assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]
    assert len(worker._stage3_recv_intervals[req_id]) == 1


def test_poll_finished_failed_chunk_purged_without_recording_interval() -> None:
    """Verifies failed chunks are purged without recording partial ranges."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-fail-purge"
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        worker._start_stage3_recv_if_ready(
            req_id="chunk-failing",
            uuid=99999,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        assert "chunk-failing" in worker._stage3_in_flight_recv_chunks

        # Engine reports failure for chunk-failing
        engine.complete_read_results = [([], [], ["chunk-failing"])]
        worker._poll_finished(engine)

        # Failed chunk should be purged from in-flight tracking
        assert "chunk-failing" not in worker._stage3_in_flight_recv_chunks
        # No interval should have been recorded for the failing chunk
        assert worker._stage3_recv_intervals.get(req_id, []) == []


def test_process_send_load_fallback_num_tokens_from_block_count() -> None:
    """Verifies fallback token count derivation from local_block_ids."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=False):
        meta = TPUConnectorMetadata(
            reqs_to_load={
                "legacy-fallback-req": LoadMeta(
                    uuid=12345,
                    local_block_ids=[0, 1, 2],  # 3 blocks * 1024 = 3072 tokens
                    remote_block_ids=[10, 11, 12],
                    remote_host="10.0.0.3",
                    remote_port=9100,
                    token_base=0,
                    is_final=True,
                )
            }
        )
        worker.process_send_load(meta)

        # In-flight chunk should have derived 3072 tokens
        assert worker._stage3_in_flight_recv_chunks["legacy-fallback-req"] == (
            "legacy-fallback-req",
            0,
            3072,
        )

        engine.complete_read_results = [([], ["legacy-fallback-req"], [])]
        worker._poll_finished(engine)

        assert worker._stage3_recv_intervals["legacy-fallback-req"] == [
            (0, 3072)
        ]


def test_consumer_cleanup_with_source_alias_mapping() -> None:
    """Verifies get_finished cleans up both destination and source aliases."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    dest_id = "dest-req-cleanup"
    source_id = "source-req-cleanup"
    worker._stage3_source_req_ids[dest_id] = source_id
    worker._stage3_destination_req_ids[source_id] = dest_id

    worker._stage3_recv_intervals[dest_id] = [(0, 1024)]
    worker._stage3_recv_intervals[source_id] = [(0, 1024)]
    worker._stage3_in_flight_recv_chunks[dest_id] = (dest_id, 0, 1024)
    worker._stage3_in_flight_recv_chunks[source_id] = (dest_id, 0, 1024)
    worker._stage3_recv_chunk_meta[dest_id] = (0, 1024)
    worker._stage3_recv_chunk_meta[source_id] = (0, 1024)

    worker.get_finished(finished_req_ids={dest_id})

    # Destination state cleaned
    assert dest_id not in worker._stage3_recv_intervals
    assert dest_id not in worker._stage3_in_flight_recv_chunks
    assert dest_id not in worker._stage3_recv_chunk_meta

    # Source alias state cleaned
    assert source_id not in worker._stage3_recv_intervals
    assert source_id not in worker._stage3_in_flight_recv_chunks
    assert source_id not in worker._stage3_recv_chunk_meta
    assert source_id not in worker._stage3_destination_req_ids
    assert dest_id not in worker._stage3_source_req_ids


def test_consumer_accumulates_non_contiguous_intervals() -> None:
    """Verifies accumulator records disjoint [0, 1024) and [2048, 3072)."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-disjoint"
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=3072
    )

    # Chunk 0: [0, 1024)
    worker._start_stage3_recv_if_ready(
        req_id="chunk-0",
        uuid=10,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )
    # Chunk 2: [2048, 3072) - chunk 1 is missing
    worker._start_stage3_recv_if_ready(
        req_id="chunk-2",
        uuid=12,
        token_base=2048,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )

    engine.complete_read_results = [([], ["chunk-0", "chunk-2"], [])]
    worker._poll_finished(engine)

    intervals = worker._stage3_recv_intervals[req_id]
    assert len(intervals) == 2
    assert (0, 1024) in intervals
    assert (2048, 3072) in intervals


def test_record_stage3_recv_chunk_completion_fallback_to_chunk_meta() -> None:
    """Verifies completion falls back to _stage3_recv_chunk_meta if needed."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    req_id = "req-meta-fallback"
    worker._stage3_recv_intervals[req_id] = []
    worker._stage3_recv_chunk_meta[req_id] = (1024, 1024)

    worker._record_stage3_recv_chunk_completion(req_id)

    assert worker._stage3_recv_intervals[req_id] == [(1024, 2048)]
    assert req_id not in worker._stage3_recv_chunk_meta



def test_tiling_check_contiguous_success() -> None:
    """Tests complete tiling of [0, 4096) via contiguous chunks returns True."""
    intervals = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is True


def test_tiling_check_gap_rejection() -> None:
    """Tests that intervals missing [1024, 2048) returns False due to gap."""
    intervals = [(0, 1024), (2048, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is False


def test_tiling_check_overlapping_success() -> None:
    """Tests that overlapping intervals correctly merge and return True."""
    intervals = [(0, 1536), (1024, 3072), (3072, 4096)]
    assert _validate_interval_tiling(intervals, 4096) is True
    # Verify two-chunk overlap covering [0, 3072)
    assert (
        _validate_interval_tiling([(0, 1536), (1024, 3072)], 3072) is True
    )


def test_tiling_check_shortfall_rejection() -> None:
    """Tests that intervals with shortfall tokens return False."""
    intervals = [(0, 2000), (2000, 4000)]
    assert _validate_interval_tiling(intervals, 4096) is False


def test_tiling_check_empty_and_invalid_inputs() -> None:
    """Tests that empty, None, or offset intervals return False."""
    assert _validate_interval_tiling([], 4096) is False
    assert _validate_interval_tiling(None, 4096) is False
    assert _validate_interval_tiling([(0, 1024)], 0) is False
    assert _validate_interval_tiling([(0, 1024)], -100) is False
    assert _validate_interval_tiling([(100, 4096)], 4096) is False
    # Duplicate and out-of-order intervals covering full range
    assert (
        _validate_interval_tiling(
            [(1024, 2048), (0, 1024), (0, 1024)], 2048
        )
        is True
    )
    assert (
        _validate_interval_tiling(
            [(3072, 4096), (0, 1024), (2048, 3072), (1024, 2048)], 4096
        )
        is True
    )


def test_get_finished_validates_interval_tiling_admission() -> None:
    """Verifies interval tiling admission during get_finished."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    # 1. Complete request: [0, 1024) + [1024, 2048), expected 2048
    req_complete = "req-complete"
    worker._update_stage3_state_after_alloc(
        req_id=req_complete, expected_tokens=2048
    )
    worker._stage3_recv_intervals[req_complete] = [(0, 1024), (1024, 2048)]
    worker._done_recving.add(req_complete)

    # 2. Incomplete request with gap: [0, 1024) + [2048, 3072), expected 3072
    req_gap = "req-gap"
    worker._update_stage3_state_after_alloc(
        req_id=req_gap, expected_tokens=3072
    )
    worker._stage3_recv_intervals[req_gap] = [(0, 1024), (2048, 3072)]
    worker._done_recving.add(req_gap)

    # 3. Shortfall request: [0, 1024), expected 2048
    req_shortfall = "req-shortfall"
    worker._update_stage3_state_after_alloc(
        req_id=req_shortfall, expected_tokens=2048
    )
    worker._stage3_recv_intervals[req_shortfall] = [(0, 1024)]
    worker._done_recving.add(req_shortfall)

    _, done_recving = worker.get_finished()

    assert req_complete in done_recving
    assert req_gap not in done_recving
    assert req_shortfall not in done_recving


def test_validate_interval_tiling_advanced_boundaries() -> None:
    """Tests comprehensive boundary conditions for _validate_interval_tiling."""
    # 1. Exact single interval match
    assert _validate_interval_tiling([(0, 2048)], 2048) is True

    # 2. Single interval overshooting expected token count
    assert _validate_interval_tiling([(0, 3000)], 2048) is True

    # 3. Fully subsumed / nested intervals
    assert _validate_interval_tiling([(0, 2048), (512, 1024)], 2048) is True

    # 4. Multi-gap disjoint intervals
    assert (
        _validate_interval_tiling(
            [(0, 512), (1024, 1536), (2048, 2560)], 2560
        )
        is False
    )

    # 5. Gap at origin (start offset > 0)
    assert _validate_interval_tiling([(1, 1024)], 1024) is False

    # 6. Inverted interval where end < start (does not advance cursor)
    assert _validate_interval_tiling([(1024, 512)], 1024) is False

    # 7. Zero-length intervals alongside covering interval
    assert _validate_interval_tiling([(0, 0), (0, 1024)], 1024) is True

    # 8. Negative start interval covering through target
    assert _validate_interval_tiling([(-50, 1024)], 1024) is True


def test_is_stage3_recv_complete_fallback_and_resolution_paths() -> None:
    """Verifies token resolution paths for _is_stage3_recv_complete."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    # 1. Untracked request ID returns True (interval tracking inactive)
    assert worker._is_stage3_recv_complete("untracked-req") is True

    # 2. Resolution via direct _stage3_submitted_load_tokens
    req_direct = "req-direct-tokens"
    worker._stage3_recv_intervals[req_direct] = [(0, 1024), (1024, 2048)]
    worker._stage3_submitted_load_tokens[req_direct] = 2048
    assert worker._is_stage3_recv_complete(req_direct) is True

    # Direct resolution fails on incomplete interval
    req_direct_incomplete = "req-direct-incomplete"
    worker._stage3_recv_intervals[req_direct_incomplete] = [(0, 1024)]
    worker._stage3_submitted_load_tokens[req_direct_incomplete] = 2048
    assert worker._is_stage3_recv_complete(req_direct_incomplete) is False

    # 3. Resolution via source alias mapping _stage3_source_req_ids
    dest_alias_req = "dest-alias-req"
    source_alias_req = "source-alias-req"
    worker._stage3_source_req_ids[dest_alias_req] = source_alias_req
    worker._stage3_submitted_load_tokens[source_alias_req] = 2048
    worker._stage3_recv_intervals[dest_alias_req] = [(0, 1024), (1024, 2048)]
    assert worker._is_stage3_recv_complete(dest_alias_req) is True

    # Source alias resolution fails on incomplete interval
    dest_alias_incomplete = "dest-alias-incomplete"
    source_alias_incomplete = "source-alias-incomplete"
    worker._stage3_source_req_ids[dest_alias_incomplete] = (
        source_alias_incomplete
    )
    worker._stage3_submitted_load_tokens[source_alias_incomplete] = 3072
    worker._stage3_recv_intervals[dest_alias_incomplete] = [
        (0, 1024),
        (1024, 2048),
    ]
    assert worker._is_stage3_recv_complete(dest_alias_incomplete) is False

    # 4. Resolution via block count fallback _load_block_ids
    req_block_fallback = "req-block-fallback"
    # 3 blocks * 1024 = 3072 expected tokens
    worker._load_block_ids[req_block_fallback] = [10, 11, 12]
    worker._stage3_recv_intervals[req_block_fallback] = [
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
    ]
    assert worker._is_stage3_recv_complete(req_block_fallback) is True

    # Block count fallback fails on shortfall
    req_block_shortfall = "req-block-shortfall"
    worker._load_block_ids[req_block_shortfall] = [10, 11, 12]  # 3072 expected
    worker._stage3_recv_intervals[req_block_shortfall] = [(0, 2048)]
    assert worker._is_stage3_recv_complete(req_block_shortfall) is False


def test_tpu_connector_worker_staticmethod_binding() -> None:
    """Verifies _validate_interval_tiling binding on connector worker class."""
    intervals = [(0, 1024), (1024, 2048)]
    # Call directly via class
    assert (
        TPURaidenConnectorWorker._validate_interval_tiling(intervals, 2048)
        is True
    )
    assert (
        TPURaidenConnectorWorker._validate_interval_tiling(intervals, 3072)
        is False
    )

    # Call via instance
    worker, _, _ = _make_consumer_test_worker(block_size=1024)
    assert worker._validate_interval_tiling(intervals, 2048) is True
    assert worker._validate_interval_tiling(intervals, 3072) is False


def test_get_finished_preserves_unadmitted_requests_until_completion() -> None:
    """Verifies unadmitted requests persist across passes until completed."""
    worker, _, _ = _make_consumer_test_worker(block_size=1024)

    req_id = "req-two-step-arrival"
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )

    # Step 1: Only chunk 0 [0, 1024) has arrived; reported done by engine
    worker._stage3_recv_intervals[req_id] = [(0, 1024)]
    worker._done_recving.add(req_id)

    _, done_recving_step1 = worker.get_finished()
    # Incomplete request should be filtered out
    assert req_id not in done_recving_step1

    # Step 2: Chunk 1 [1024, 2048) arrives and completes
    worker._stage3_recv_intervals[req_id].append((1024, 2048))
    worker._done_recving.add(req_id)

    _, done_recving_step2 = worker.get_finished()
    # Fully tiled request should now be admitted
    assert req_id in done_recving_step2


def test_failed_chunk_triggers_recomputation() -> None:
    """Simulates chunk 2 failing in _failed_recving and verifies recomputation.

    Registers 3 sub-chunks for a 3072-token prompt across blocks [101, 102,
    103]. Chunks 0 and 1 succeed, accumulating intervals [0, 1024) and [1024,
    2048). Chunk 2 fails in engine completion polling. Verifies that the base
    request is placed into _failed_recving, partial interval state is cleared,
    get_finished() reports the request as completed, and
    get_block_ids_with_load_errors() returns the full allocated block set
    {101, 102, 103} for local recomputation.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-subchunk-fail"
    blocks = [101, 102, 103]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=3072
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        # Register 3 sub-chunks
        worker._start_stage3_recv_if_ready(
            req_id="chunk-0",
            uuid=10001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        worker._start_stage3_recv_if_ready(
            req_id="chunk-1",
            uuid=10002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        worker._start_stage3_recv_if_ready(
            req_id="chunk-2",
            uuid=10003,
            endpoint="10.0.0.1:9000",
            token_base=2048,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )

        # Chunks 0 and 1 succeed
        engine.complete_read_results = [([], ["chunk-0", "chunk-1"], [])]
        worker._poll_finished(engine)
        assert set(worker._stage3_recv_intervals[req_id]) == {
            (0, 1024),
            (1024, 2048),
        }
        assert req_id not in worker._failed_recving

        # Chunk 2 fails
        engine.complete_read_results = [([], [], ["chunk-2"])]
        worker._poll_finished(engine)

        # Base request must be marked in _failed_recving
        assert req_id in worker._failed_recving
        # Partial interval state must be purged
        assert worker._stage3_recv_intervals.get(req_id) is None
        # Remaining in-flight chunks must be cleared
        assert "chunk-2" not in worker._stage3_in_flight_recv_chunks

        # get_finished() must report the failed request in done_recving
        done_sending, done_recving = worker.get_finished()
        assert req_id in done_recving

        # get_block_ids_with_load_errors() must return all affected blocks
        failed_blocks = worker.get_block_ids_with_load_errors()
        assert failed_blocks == {101, 102, 103}

        # Subsequent call must return empty set (drained)
        assert worker.get_block_ids_with_load_errors() == set()


def test_tiling_timeout_triggers_error() -> None:
    """Simulates missing final chunk causing timeout without deadlock.

    Registers 2 sub-chunks for a 2048-token prompt across blocks [201, 202].
    Chunk 0 completes, while Chunk 1 is lost / delayed. Simulates timeout
    expiration in _poll_finished / _wait_for_recving. Verifies that the request
    enters error recovery, is marked in _failed_recving, reports done_recving
    in get_finished(), and surfaces block IDs {201, 202} for local
    recomputation.
    """
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-tiling-timeout"
    blocks = [201, 202]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        # Register chunk 0 and chunk 1
        worker._start_stage3_recv_if_ready(
            req_id="chunk-0-ok",
            uuid=20001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        worker._start_stage3_recv_if_ready(
            req_id="chunk-1-lost",
            uuid=20002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )

        # Chunk 0 arrives and completes
        engine.complete_read_results = [([], ["chunk-0-ok"], [])]
        worker._poll_finished(engine)
        assert worker._stage3_recv_intervals[req_id] == [(0, 1024)]

        # Simulate timeout by setting start time far in the past
        worker._stage3_load_start_times[req_id] = time.perf_counter() - 10000.0

        # Poll finished detecting timeout
        engine.complete_read_results = [([], [], [])]
        worker._poll_finished(engine)

        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None

        # Verify get_finished reports request as completed without deadlock
        done_sending, done_recving = worker.get_finished()
        assert req_id in done_recving

        # Verify blocks are marked for recomputation
        assert worker.get_block_ids_with_load_errors() == {201, 202}


def test_wait_for_recving_timeout_recovers_without_deadlock() -> None:
    """Verifies that _wait_for_recving times out safely on incomplete chunks."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-wait-timeout"
    blocks = [301, 302]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)

    with (
        patch.object(worker, "_raiden_stage3_enabled", return_value=True),
        patch(
            f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
            return_value=0.01,
        ),
    ):
        worker._start_stage3_recv_if_ready(
            req_id="chunk-wait-0",
            uuid=30001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )

        # Chunk 0 arrives, chunk 1 never submitted
        engine.complete_read_results = [([], ["chunk-wait-0"], [])]
        worker._wait_for_recving({req_id})

        # Request must be marked failed
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None

        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {301, 302}


def test_legacy_consumer_chunk_failure_triggers_recomputation() -> None:
    """Verifies sub-chunk failure in non-Stage3 legacy path."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-legacy-fail"
    blocks = [401, 402]
    worker._load_block_ids[req_id] = list(blocks)
    worker._register_in_flight_recv_chunk(
        transfer_id="legacy-chunk-0",
        destination_req_id=req_id,
        token_base=0,
        num_tokens=1024,
    )
    worker._register_in_flight_recv_chunk(
        transfer_id="legacy-chunk-1",
        destination_req_id=req_id,
        token_base=1024,
        num_tokens=1024,
    )

    with patch.object(worker, "_raiden_stage3_enabled", return_value=False):
        # Chunk 0 succeeds, chunk 1 fails in poll_stats
        engine.poll_results = [
            ([], ["legacy-chunk-0"], []),
            ([], [], ["legacy-chunk-1"]),
        ]
        worker._poll_finished(engine)
        worker._poll_finished(engine)

        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None

        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {401, 402}


def test_absorbing_failure_state_ignores_late_chunks() -> None:
    """Verifies that late-arriving chunks do not resurrect a failed request."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-absorbing-fail"
    blocks = [501, 502]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        # Register chunk 0 and chunk 1
        worker._start_stage3_recv_if_ready(
            req_id="chunk-0-fails",
            uuid=50001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        worker._start_stage3_recv_if_ready(
            req_id="chunk-1-late",
            uuid=50002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )

        # 1. Chunk 0 fails -> request becomes failed
        engine.complete_read_results = [([], [], ["chunk-0-fails"])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None

        # 2. Chunk 1 arrives late -> must be ignored and not resurrect intervals
        engine.complete_read_results = [([], ["chunk-1-late"], [])]
        worker._poll_finished(engine)
        assert req_id in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_id) is None
        assert worker._is_stage3_recv_complete(req_id) is False

        # Verify get_finished reports request as completed (in failure state)
        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {501, 502}


def test_mixed_concurrent_requests_isolate_failure_and_success() -> None:
    """Verifies error isolation across concurrent requests."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_ok = "req-ok"
    req_fail = "req-fail"
    blocks_ok = [601, 602]
    blocks_fail = [701, 702]

    worker._update_stage3_state_after_alloc(
        req_id=req_ok, expected_tokens=2048
    )
    worker._load_block_ids[req_ok] = list(blocks_ok)
    worker._stage3_controller_accepted.add(req_ok)

    worker._update_stage3_state_after_alloc(
        req_id=req_fail, expected_tokens=2048
    )
    worker._load_block_ids[req_fail] = list(blocks_fail)
    worker._stage3_controller_accepted.add(req_fail)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        # Register chunks for req_ok
        worker._start_stage3_recv_if_ready(
            req_id="ok-0",
            uuid=60001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_ok,
        )
        worker._start_stage3_recv_if_ready(
            req_id="ok-1",
            uuid=60002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_ok,
        )

        # Register chunks for req_fail
        worker._start_stage3_recv_if_ready(
            req_id="fail-0",
            uuid=70001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_fail,
        )
        worker._start_stage3_recv_if_ready(
            req_id="fail-1",
            uuid=70002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_fail,
        )

        # ok-0, ok-1 succeed, fail-0 succeeds, fail-1 fails
        engine.complete_read_results = [(
            [],
            ["ok-0", "ok-1", "fail-0"],
            ["fail-1"],
        )]
        worker._poll_finished(engine)

        assert req_ok not in worker._failed_recving
        assert req_fail in worker._failed_recving
        assert worker._stage3_recv_intervals.get(req_fail) is None

        _, done_recving = worker.get_finished()
        assert req_ok in done_recving
        assert req_fail in done_recving

        # Only req_fail's blocks must be reported as load errors
        assert worker.get_block_ids_with_load_errors() == {701, 702}


def test_uncertain_controller_failure_timeout_triggers_recomputation() -> None:
    """Verifies expired pending controller failures trigger recomputation."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-uncertain-timeout"
    blocks = [801, 802]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)
    # Set expired pending controller failure deadline
    worker._stage3_pending_controller_failures[req_id] = (
        time.perf_counter() - 5.0
    )

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        engine.complete_read_results = [([], [], [])]
        worker._poll_finished(engine)

        assert req_id in worker._failed_recving
        assert req_id not in worker._stage3_pending_controller_failures

        _, done_recving = worker.get_finished()
        assert req_id in done_recving
        assert worker.get_block_ids_with_load_errors() == {801, 802}


def test_incomplete_multichunk_does_not_retire_to_terminal_loads() -> None:
    """Verifies that partial multi-chunk recvs do not enter terminal loads."""
    worker, _, engine = _make_consumer_test_worker(block_size=1024)

    req_id = "req-multichunk-progression"
    blocks = [901, 902]
    worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=2048
    )
    worker._load_block_ids[req_id] = list(blocks)
    worker._stage3_controller_accepted.add(req_id)

    with patch.object(worker, "_raiden_stage3_enabled", return_value=True):
        worker._start_stage3_recv_if_ready(
            req_id="chunk-step0",
            uuid=90001,
            endpoint="10.0.0.1:9000",
            token_base=0,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )
        worker._start_stage3_recv_if_ready(
            req_id="chunk-step1",
            uuid=90002,
            endpoint="10.0.0.1:9000",
            token_base=1024,
            num_tokens=1024,
            engine=engine,
            destination_req_id=req_id,
        )

        # Step 1: Only chunk 0 completes
        engine.complete_read_results = [([], ["chunk-step0"], [])]
        worker._poll_finished(engine)

        # Request must NOT be in _stage3_terminal_loads yet
        assert req_id not in worker._stage3_terminal_loads
        _, done_recving = worker.get_finished()
        assert req_id not in done_recving

        # Step 2: Chunk 1 completes
        engine.complete_read_results = [([], ["chunk-step1"], [])]
        worker._poll_finished(engine)

        # Request must NOW be in _stage3_terminal_loads
        assert req_id in worker._stage3_terminal_loads
        _, done_recving = worker.get_finished()
        assert req_id in done_recving


def test_tpu_connector_load_errors_group_index() -> None:
    """Verifies get_block_ids_with_load_errors_group_index returns FA group."""
    cfg = _make_mock_consumer_vllm_config()
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "use_raiden_connector": True
    }
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=MagicMock())  # FA group 0
    ]

    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        connector = TPUConnector(
            cfg, role=MagicMock(), kv_cache_config=kv_cache_config
        )
        connector._stage3_fa_group_index = 0
        assert connector.get_block_ids_with_load_errors_group_index() == 0

        # When Stage 3 is not enabled, returns None
        with patch(f"{_MOD}._use_raiden_stage3_transport", return_value=False):
            assert (
                connector.get_block_ids_with_load_errors_group_index() is None
            )

