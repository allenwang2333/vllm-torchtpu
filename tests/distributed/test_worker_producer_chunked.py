# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Stage-3 worker producer sub-chunk work unit registration.

Verifies that the TPURaidenConnectorWorker ingests range-aware SendMeta
descriptors (with token_base and is_final flags), derives sub-chunk pool
spans offset into the global destination coordinate space, registers work
units with the Raiden controller facade, and enforces registration
idempotency and conflict detection.
"""

from collections.abc import Iterator
import contextlib
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    SendMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolByteSpan,
    PoolSpanRegistration,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorWorker,
    _Stage3RegisteredSend,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class _FakeRaidenEngine:
    """Mock low-level Raiden transfer engine for worker testing."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.poll_results: list[tuple[list[str], list[str], list[str]]] = []
        self.complete_read_results: list[
            tuple[list[str], list[str], list[str]]
        ] = []

    def register_read(
        self, req_id: str, uuid: int, block_ids: list[int]
    ) -> int:
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
        self.calls.append(
            ("start_read", req_id, uuid, endpoint, remote_blocks, local_blocks)
        )
        return 2

    def complete_read(self) -> tuple[list[str], list[str], list[str]]:
        self.calls.append(("complete_read",))
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        if self.poll_results:
            return self.poll_results.pop(0)
        return [], [], []

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        self.calls.append(("poll_stats",))
        if self.poll_results:
            return self.poll_results.pop(0)
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        return [], [], []


class _FakeRaidenControllerFacade:
    """Mock controller client recording RPC invocations."""

    def __init__(self) -> None:
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
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
    cfg = MagicMock()
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
        patch(
            f"{_MOD}.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
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
        job_name="prefill-worker",
        job_replica_id="producer-engine-rank0",
        data_name="kv.fa",
        data_replica_idx=0,
    )
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "prefill-controller.test:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=transfer_rank)
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
    """Patches runtime environment variables for worker testing."""
    patches = [
        patch(
            f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT",
            "raiden",
            create=True,
        ),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
            create=True,
        ),
    ]
    if wait_pull_timeout is not None:
        patches.append(
            patch(
                f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                return_value=wait_pull_timeout,
            )
        )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def test_worker_registers_subchunk_work_unit() -> None:
    """Verifies worker registers sub-chunk work unit with range-addressed spans.

    Feeds sub-chunk SendMeta(token_base=1024, num_tokens=1024) into
    process_send_load(). Asserts that the controller client receives a
    registration containing pool spans offset at token 1024 in the destination
    byte space.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-chunk-001"
    sub_uuid = 54321
    token_base = 1024
    num_tokens = 1024
    local_block_ids = [200, 201]  # 2 scheduler blocks covering 2048 tokens

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=local_block_ids,
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=token_base,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    # Verify controller facade received exactly one registration call
    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    assert call["req_id"] == req_id
    assert call["uuid"] == sub_uuid
    assert call["unit"] is worker._raiden_work_unit
    assert call["block_ids"] == [200]  # Rank 0 owns 256 tokens -> 1 block

    # Verify byte spans are offset at token 1024
    pool_spans: list[PoolSpanRegistration] = call["pool_spans"]
    assert len(pool_spans) == 1
    fa_reg = pool_spans[0]
    assert fa_reg.tag == "fa"
    assert fa_reg.dst_space_version == 1
    token_bytes = worker._stage3_fa_token_bytes(1024)
    assert fa_reg.declared_bytes == 256 * token_bytes
    assert len(fa_reg.spans) == 1
    span: PoolByteSpan = fa_reg.spans[0]
    assert span.src_block_ordinal == 0
    assert span.src_offset_bytes == 0
    assert span.dst_offset_bytes == token_base * token_bytes
    assert span.size_bytes == 256 * token_bytes

    # Verify worker internal registration tracking
    assert req_id in worker._stage3_registered_sends
    reg_entry: _Stage3RegisteredSend = worker._stage3_registered_sends[req_id]
    assert reg_entry.uuid == sub_uuid
    assert reg_entry.token_base == token_base
    assert reg_entry.num_tokens == num_tokens
    assert reg_entry.is_final is False
    assert reg_entry.local_block_ids == (200,)


def test_worker_registration_idempotency() -> None:
    """Verifies duplicate sub-chunk SendMeta registrations are ignored.

    Feeds identical sub-chunk SendMeta twice into process_send_load(). Asserts
    that the controller facade receives only one registration RPC and no
    duplicate errors are raised.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-idempotent-002"
    sub_uuid = 98765
    token_base = 1024
    num_tokens = 1024

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=token_base,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)
        worker.process_send_load(meta)  # Replay identical registration

    assert len(facade.register_request_blocks_calls) == 1
    assert req_id in worker._stage3_registered_sends


def test_worker_registration_conflicting_duplicate_rejected() -> None:
    """Verifies that duplicate registrations with conflicting parameters fail.

    Registers a sub-chunk with token_base=1024, then attempts to register
    conflicting metadata under the same req_id. Asserts that ValueError is
    raised with an explicit conflict message.
    """
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-conflict-003"

    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(
        uuid=11111,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(
        uuid=11111,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=2048,  # Conflicting token_base
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ):
            worker.process_send_load(meta_conflict)


def test_worker_registration_terminal_idempotency_and_conflict() -> None:
    """Verifies replay behavior against terminal tombstone entries.

    Simulates a sub-chunk registration that has already completed and moved
    to _stage3_terminal_sends. Identical replays should be ignored, while
    conflicting replays must raise ValueError.
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-terminal-004"
    sub_uuid = 22222
    token_base = 2048
    num_tokens = 1024
    local_ids = (100,)

    # Populate terminal send tombstone
    worker._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
        uuid=sub_uuid,
        local_block_ids=local_ids,
        num_tokens=num_tokens,
        expiration_time=time.perf_counter() + 30.0,
        token_base=token_base,
        is_final=False,
    )

    meta_identical = TPUConnectorMetadata()
    meta_identical.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=[100, 101, 102],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=token_base,
        is_final=False,
    )

    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=[100, 101, 102],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=1024,  # Conflicting token_base vs terminal entry
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        # Identical replay: no error, no new facade RPC
        worker.process_send_load(meta_identical)
        assert len(facade.register_request_blocks_calls) == 0

        # Conflicting replay: raises ValueError
        with pytest.raises(
            ValueError,
            match=(
                "Conflicting replay after terminal Stage-3 producer"
                " registration"
            ),
        ):
            worker.process_send_load(meta_conflict)


@pytest.mark.parametrize("transfer_rank", [0, 1, 2, 3])
def test_worker_registration_multi_rank_offsets(transfer_rank: int) -> None:
    """Verifies sub-chunk span destination offsets across all PCP ranks.

    For chunk [1024, 2048) on PCP=4 with interleave 256:
    - Rank 0 owns [1024, 1280) -> dst_offset = 1024 * token_bytes
    - Rank 1 owns [1280, 1536) -> dst_offset = 1280 * token_bytes
    - Rank 2 owns [1536, 1792) -> dst_offset = 1536 * token_bytes
    - Rank 3 owns [1792, 2048) -> dst_offset = 1792 * token_bytes
    """
    worker, facade, _ = _make_test_worker(
        pcp_size=4,
        interleave_size=256,
        block_size=1024,
        transfer_rank=transfer_rank,
    )
    req_id = f"req-rank-{transfer_rank}"
    sub_uuid = 30000 + transfer_rank
    token_base = 1024
    num_tokens = 1024

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=[10, 11],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=token_base,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    fa_reg = call["pool_spans"][0]
    token_bytes = worker._stage3_fa_token_bytes(1024)
    expected_offset = (token_base + transfer_rank * 256) * token_bytes
    assert len(fa_reg.spans) == 1
    assert fa_reg.spans[0].dst_offset_bytes == expected_offset
    assert fa_reg.spans[0].size_bytes == 256 * token_bytes


def test_worker_registration_unaligned_token_base_rejected() -> None:
    """Verifies unaligned token_base raises ValueError during registration."""
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["req-unaligned"] = SendMeta(
        uuid=44444,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=512,  # Not a multiple of 4 * 256 = 1024
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
            worker.process_send_load(meta)


def test_worker_registration_negative_token_base_rejected() -> None:
    """Verifies negative token_base raises ValueError during registration."""
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["req-negative"] = SendMeta(
        uuid=55555,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=-1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match=(
                "Stage-3 producer send metadata requires a non-negative"
                " token_base"
            ),
        ):
            worker.process_send_load(meta)


def test_worker_registration_legacy_backward_compatibility() -> None:
    """Verifies legacy SendMeta registers properly with default base 0."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-legacy-001"
    uuid_val = 66001
    num_tokens = 1024

    meta = TPUConnectorMetadata()
    # Omit token_base and is_final to test legacy instantiation defaults
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=uuid_val,
        local_block_ids=[100],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert len(facade.register_request_blocks_calls) == 1
    call = facade.register_request_blocks_calls[0]
    fa_reg = call["pool_spans"][0]
    token_bytes = worker._stage3_fa_token_bytes(1024)
    assert fa_reg.spans[0].dst_offset_bytes == 0
    assert fa_reg.spans[0].size_bytes == 256 * token_bytes

    assert req_id in worker._stage3_registered_sends
    reg = worker._stage3_registered_sends[req_id]
    assert reg.token_base == 0
    assert reg.is_final is True


def test_worker_registration_insufficient_scheduler_blocks_rejected() -> None:
    """Verifies insufficient scheduler blocks for window raises ValueError."""
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-insufficient-blocks"
    # For PCP=4, block_size=1024 -> scheduler_block_tokens = 4096.
    # token_base = 4096, num_tokens = 1024 -> required window = 5120 tokens ->
    # requires 2 scheduler blocks.
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=66002,
        local_block_ids=[100],  # Only 1 block provided instead of 2
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=4096,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        with pytest.raises(
            ValueError,
            match=(
                "Stage-3 producer scheduler block count is inconsistent with"
                " PCP geometry"
            ),
        ):
            worker.process_send_load(meta)


def test_worker_registration_cancellation_race_records_tombstone() -> None:
    """Verifies cancellation errors record tombstone with sub-chunk fields."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-cancelled-001"
    sub_uuid = 66003
    token_base = 1024
    num_tokens = 1024

    facade.register_request_blocks = MagicMock(
        side_effect=RuntimeError(
            "Request block registration was cancelled: generation expired"
        )
    )

    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=sub_uuid,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=num_tokens,
        token_base=token_base,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta)

    # Must be recorded in terminal sends tombstone, NOT active sends
    assert req_id in worker._stage3_terminal_sends
    assert req_id not in worker._stage3_registered_sends
    assert req_id in worker._done_sending

    tombstone = worker._stage3_terminal_sends[req_id]
    assert tombstone.uuid == sub_uuid
    assert tombstone.token_base == token_base
    assert tombstone.num_tokens == num_tokens
    assert tombstone.is_final is False


def test_worker_registration_multiple_sequential_subchunks() -> None:
    """Verifies multiple sequential sub-chunks register independently."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    token_bytes = worker._stage3_fa_token_bytes(1024)

    # Chunk 0: [0, 1024), is_final=False
    meta0 = TPUConnectorMetadata()
    meta0.reqs_to_send["req-multi:0"] = SendMeta(
        uuid=70000,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )

    # Chunk 1: [1024, 2048), is_final=True
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send["req-multi:1"] = SendMeta(
        uuid=70001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=True,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta0)
        worker.process_send_load(meta1)

    assert len(facade.register_request_blocks_calls) == 2
    call0 = facade.register_request_blocks_calls[0]
    call1 = facade.register_request_blocks_calls[1]

    assert call0["req_id"] == "req-multi:0"
    assert call0["uuid"] == 70000
    assert call0["pool_spans"][0].spans[0].dst_offset_bytes == 0

    assert call1["req_id"] == "req-multi:1"
    assert call1["uuid"] == 70001
    assert (
        call1["pool_spans"][0].spans[0].dst_offset_bytes
        == 1024 * token_bytes
    )

    assert "req-multi:0" in worker._stage3_registered_sends
    assert "req-multi:1" in worker._stage3_registered_sends
    assert worker._stage3_registered_sends["req-multi:0"].is_final is False
    assert worker._stage3_registered_sends["req-multi:1"].is_final is True


def test_worker_registration_conflicting_num_tokens_rejected() -> None:
    """Verifies registration with mismatched num_tokens raises ValueError."""
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-conflict-num-tokens"

    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(
        uuid=80001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(
        uuid=80001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=2048,  # Conflicting num_tokens
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ):
            worker.process_send_load(meta_conflict)


def test_worker_registration_conflicting_local_block_ids_rejected() -> None:
    """Verifies registration with mismatched block_ids raises ValueError."""
    worker, _, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    req_id = "req-conflict-blocks"

    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id] = SendMeta(
        uuid=80002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    meta_conflict = TPUConnectorMetadata()
    meta_conflict.reqs_to_send[req_id] = SendMeta(
        uuid=80002,
        local_block_ids=[102, 103],  # Conflicting block IDs
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
        with pytest.raises(
            ValueError,
            match="Conflicting duplicate Stage-3 producer registration",
        ):
            worker.process_send_load(meta_conflict)



def test_device_buffer_retained_during_dma() -> None:
    """Verifies device buffer handles are retained during DMA."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor_k = MagicMock(name="mock_kv_tensor_k")
    mock_tensor_v = MagicMock(name="mock_kv_tensor_v")
    mock_runner = MagicMock()
    mock_runner.kv_caches = [(mock_tensor_k, mock_tensor_v)]
    worker.runner = mock_runner

    req_id = "req-dma-001"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=90001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert req_id in worker._active_tensor_refs
    assert req_id in worker._stage3_registered_sends
    ref = worker._active_tensor_refs[req_id]
    assert isinstance(ref, _Stage3RegisteredSend)
    assert ref.uuid == 90001
    assert ref.token_base == 1024
    assert ref.num_tokens == 1024
    assert ref.local_block_ids == (100,)
    assert ref.device_buffers == (mock_tensor_k, mock_tensor_v)


def test_device_buffer_released_on_dma_completion() -> None:
    """Verifies device buffer handles are released when DMA completes."""
    worker, facade, engine = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor = MagicMock(name="mock_tensor")
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner

    req_id = "req-dma-002"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=90002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert req_id in worker._active_tensor_refs
    assert worker._active_tensor_refs[req_id].device_buffers == (mock_tensor,)

    # Mock complete_read reporting DMA completion for req_id
    engine.complete_read_results = [([req_id], [], [])]

    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()

    assert req_id in done_sending
    assert req_id not in worker._active_tensor_refs
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()
    assert req_id not in worker._stage3_registered_sends


def test_device_buffer_multiple_subchunks_independent_lifecycle() -> None:
    """Verifies sub-chunks have independent buffer retention lifecycles."""
    worker, facade, engine = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor = MagicMock(name="mock_tensor")
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner

    meta0 = TPUConnectorMetadata()
    meta0.reqs_to_send["req-sub:0"] = SendMeta(
        uuid=91000,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=False,
    )
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send["req-sub:1"] = SendMeta(
        uuid=91001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=True,
    )

    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta0)
        worker.process_send_load(meta1)

    assert "req-sub:0" in worker._active_tensor_refs
    assert "req-sub:1" in worker._active_tensor_refs

    # Complete only chunk 0
    engine.complete_read_results = [(["req-sub:0"], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending, _ = worker.get_finished()

    assert "req-sub:0" in done_sending
    assert "req-sub:0" not in worker._active_tensor_refs
    assert "req-sub:1" in worker._active_tensor_refs  # Chunk 1 still pinned

    # Complete chunk 1
    engine.complete_read_results = [(["req-sub:1"], [], [])]
    with _patch_raiden_worker_env(parallelism=4):
        done_sending2, _ = worker.get_finished()

    assert "req-sub:1" in done_sending2
    assert "req-sub:1" not in worker._active_tensor_refs


def test_device_buffer_unregistered_on_cancellation_race() -> None:
    """Verifies buffer handles are dropped when registration is cancelled."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor = MagicMock(name="mock_tensor")
    mock_runner = MagicMock()
    mock_runner.kv_caches = [mock_tensor]
    worker.runner = mock_runner

    facade.register_request_blocks = MagicMock(
        side_effect=RuntimeError(
            "Request block registration was cancelled: generation expired"
        )
    )

    req_id = "req-cancel-race"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=92000,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=1024,
        is_final=False,
    )

    with _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0):
        worker.process_send_load(meta)

    assert req_id not in worker._active_tensor_refs
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()


def test_flattened_kv_cache_tensors_variations() -> None:
    """Verifies _flattened_kv_cache_tensors handles varied cache formats."""
    worker, _, _ = _make_test_worker()

    # 1. runner is None -> returns empty list
    worker.runner = None
    assert worker._flattened_kv_cache_tensors() == []

    # 2. runner without kv_caches attribute -> returns empty list
    worker.runner = SimpleNamespace()
    assert worker._flattened_kv_cache_tensors() == []

    # 3. runner.kv_caches is None -> returns empty list
    worker.runner = SimpleNamespace(kv_caches=None)
    assert worker._flattened_kv_cache_tensors() == []

    # 4. runner.kv_caches is a flat list of tensors -> returns flat list
    t1, t2 = MagicMock(name="t1"), MagicMock(name="t2")
    worker.runner = SimpleNamespace(kv_caches=[t1, t2])
    assert worker._flattened_kv_cache_tensors() == [t1, t2]

    # 5. runner.kv_caches is a list of (key, value) tuples -> flattens to 1D
    k1, v1 = MagicMock(name="k1"), MagicMock(name="v1")
    k2, v2 = MagicMock(name="k2"), MagicMock(name="v2")
    worker.runner = SimpleNamespace(kv_caches=[(k1, v1), (k2, v2)])
    assert worker._flattened_kv_cache_tensors() == [k1, v1, k2, v2]

    # 6. runner.kv_caches has mixed tuples and tensors -> flattens all entries
    t3 = MagicMock(name="t3")
    worker.runner = SimpleNamespace(kv_caches=[(k1, v1), t3])
    assert worker._flattened_kv_cache_tensors() == [k1, v1, t3]


def test_poll_finished_engine_interface_polymorphism() -> None:
    """Verifies _poll_finished handles polymorphic engine interfaces."""
    worker, _, _ = _make_test_worker(is_producer=True)
    mock_tensor = MagicMock(name="mock_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    # Register send 1 for complete_read test
    req_id_1 = "req-poly-complete-read"
    meta1 = TPUConnectorMetadata()
    meta1.reqs_to_send[req_id_1] = SendMeta(
        uuid=93001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta1)
    assert req_id_1 in worker._active_tensor_refs

    # Engine implementing only complete_read
    class EngineWithCompleteRead:
        def complete_read(self) -> tuple[list[str], list[str], list[str]]:
            return [req_id_1], [], []

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithCompleteRead())
    assert req_id_1 not in worker._active_tensor_refs

    # Register send 2 for poll_stats test
    req_id_2 = "req-poly-poll-stats"
    meta2 = TPUConnectorMetadata()
    meta2.reqs_to_send[req_id_2] = SendMeta(
        uuid=93002,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta2)
    assert req_id_2 in worker._active_tensor_refs

    # Engine implementing only poll_stats
    class EngineWithPollStats:
        def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
            return [req_id_2], [], []

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithPollStats())
    assert req_id_2 not in worker._active_tensor_refs

    # Engine implementing neither method
    class EngineWithoutPolling:
        pass

    with _patch_raiden_worker_env(parallelism=4):
        worker._poll_finished(EngineWithoutPolling())


def test_device_buffer_released_on_terminal_cleanup_force_release() -> None:
    """Verifies buffer handles are purged during terminal cleanup."""
    worker, facade, _ = _make_test_worker(
        pcp_size=4, interleave_size=256, block_size=1024, transfer_rank=0
    )
    mock_tensor = MagicMock(name="mock_tensor")
    worker.runner = SimpleNamespace(kv_caches=[mock_tensor])

    req_id = "req-term-cleanup"
    meta = TPUConnectorMetadata()
    meta.reqs_to_send[req_id] = SendMeta(
        uuid=94001,
        local_block_ids=[100, 101],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )
    with _patch_raiden_worker_env(parallelism=4):
        worker.process_send_load(meta)

    assert req_id in worker._active_tensor_refs
    assert worker._active_tensor_refs[req_id].device_buffers == (mock_tensor,)

    # Queue terminal send for cleanup with force_release=True
    worker._stage3_terminal_cleanup[req_id] = True

    with _patch_raiden_worker_env(parallelism=4):
        worker.get_finished()

    assert req_id not in worker._active_tensor_refs
    assert req_id not in worker._stage3_registered_sends
    assert req_id in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id].device_buffers == ()
    assert len(facade.complete_request_blocks_calls) == 1
    assert facade.complete_request_blocks_calls[0]["req_id"] == req_id
    assert facade.complete_request_blocks_calls[0]["uuid"] == 94001


def test_glm_device_buffer_retained_and_released_on_cancellation_race() -> None:
    """Verifies GLM producer registration retains and releases buffers."""
    worker, facade, _ = _make_test_worker(
        tp_rank=0,
        tp_size=4,
        pcp_size=4,
        interleave_size=256,
        block_size=1024,
        transfer_rank=0,
    )
    mock_tensor_k = MagicMock(name="mock_kv_tensor_k")
    mock_tensor_v = MagicMock(name="mock_kv_tensor_v")
    worker.runner = SimpleNamespace(kv_caches=[(mock_tensor_k, mock_tensor_v)])

    # Setup GLM geometry for all required tags
    worker._stage3_glm_geometry = {
        "mla.nope": (1024 * 128, 128),
        "mla.rope": (1024 * 64, 64),
        "dsa.idx": (1024 * 16, 16),
    }

    req_id_ok = "req-glm-ok"
    meta_ok = TPUConnectorMetadata()
    meta_ok.reqs_to_send[req_id_ok] = SendMeta(
        uuid=95001,
        local_block_ids=[200],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )

    with (
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        _patch_raiden_worker_env(parallelism=4),
    ):
        worker.process_send_load(meta_ok)

    assert req_id_ok in worker._active_tensor_refs
    assert req_id_ok in worker._stage3_registered_sends
    assert worker._active_tensor_refs[req_id_ok].device_buffers == (
        mock_tensor_k,
        mock_tensor_v,
    )

    # Test cancellation race in GLM registration path
    facade.register_request_blocks = MagicMock(
        side_effect=RuntimeError(
            "Request block registration was cancelled: generation expired"
        )
    )
    req_id_cancel = "req-glm-cancel"
    meta_cancel = TPUConnectorMetadata()
    meta_cancel.reqs_to_send[req_id_cancel] = SendMeta(
        uuid=95002,
        local_block_ids=[202],
        expiration_time=time.perf_counter() + 30.0,
        num_tokens=1024,
        token_base=0,
        is_final=True,
    )

    with (
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        _patch_raiden_worker_env(parallelism=4, wait_pull_timeout=5.0),
    ):
        worker.process_send_load(meta_cancel)

    assert req_id_cancel not in worker._active_tensor_refs
    assert req_id_cancel in worker._stage3_terminal_sends
    assert worker._stage3_terminal_sends[req_id_cancel].device_buffers == ()
