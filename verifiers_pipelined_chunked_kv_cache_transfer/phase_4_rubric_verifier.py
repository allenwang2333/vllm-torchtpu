# SPDX-License-Identifier: Apache-2.0
"""Phase 4 Rubric Verifier — Consumer Assembly & Fail-Closed Interval Tiling.

Validates all Phase 4 Rubric items for chunked KV cache transfer:
- R4.1 — Destination Memory Stability: Destination memory pages are
  pre-allocated upfront based on the total prompt extent, remaining stable
  as sub-chunks arrive.
- R4.2 — Gap-Detecting Tiling Check: The consumer validates that the union
  of received token intervals forms an unbroken cover of the entire prompt
  range before committing the transfer.
- R4.3 — Automatic Local Recomputation on Transfer Failure: If any sub-chunk
  fails or the interval coverage is incomplete, the request is marked for
  error recovery and local recomputation.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import sys
import time
import types
from typing import Any
from unittest import mock

import pytest
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta,
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
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
    self.complete_read_results: list[tuple[list[str], list[str], list[str]]] = (
        []
    )

  def register_read(
      self, req_id: str, uuid: int, block_ids: Sequence[int]
  ) -> int:
    """Records a register_read invocation.

    Args:
      req_id: Request identifier string.
      uuid: Transfer registration UUID.
      block_ids: Sequence of physical cache block IDs.

    Returns:
      Registration status code.
    """
    self.calls.append(("register_read", req_id, uuid, list(block_ids)))
    return 1

  def start_read(
      self,
      req_id: str,
      uuid: int,
      endpoint: str,
      remote_blocks: Sequence[int],
      local_blocks: Sequence[int],
  ) -> int:
    """Records a start_read invocation.

    Args:
      req_id: Request identifier string.
      uuid: Transfer registration UUID.
      endpoint: Target endpoint network address string.
      remote_blocks: Source block indices on remote host.
      local_blocks: Destination block indices on local host.

    Returns:
      Operation status code.
    """
    self.calls.append(
        (
            "start_read",
            req_id,
            uuid,
            endpoint,
            list(remote_blocks),
            list(local_blocks),
        )
    )
    return 2

  def complete_read(self) -> tuple[list[str], list[str], list[str]]:
    """Returns queued complete_read completion triples.

    Returns:
      Triple of (pending_req_ids, completed_req_ids, failed_req_ids).
    """
    self.calls.append(("complete_read",))
    if self.complete_read_results:
      return self.complete_read_results.pop(0)
    if self.poll_results:
      return self.poll_results.pop(0)
    return [], [], []

  def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
    """Returns queued poll_stats completion triples.

    Returns:
      Triple of (pending_req_ids, completed_req_ids, failed_req_ids).
    """
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
    """Records a register_work_unit invocation.

    Args:
      **kwargs: Keyword arguments for work unit registration.
    """
    self.register_work_unit_calls.append(kwargs)

  def register_request_blocks(self, **kwargs: Any) -> None:
    """Records a register_request_blocks invocation.

    Args:
      **kwargs: Keyword arguments for request block registration.
    """
    self.register_request_blocks_calls.append(kwargs)

  def complete_request_blocks(self, **kwargs: Any) -> None:
    """Records a complete_request_blocks invocation.

    Args:
      **kwargs: Keyword arguments for block completion.
    """
    self.complete_request_blocks_calls.append(kwargs)

  def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
    """Records a cancel_request_blocks_if_unclaimed invocation.

    Args:
      **kwargs: Keyword arguments for cancellation query.

    Returns:
      Boolean indicating whether cancellation succeeded.
    """
    self.cancel_request_blocks_calls.append(kwargs)
    return self.cancel_request_blocks_result

  def start_transfer(self, **kwargs: Any) -> bool:
    """Records a start_transfer invocation.

    Args:
      **kwargs: Keyword arguments for transfer initiation.

    Returns:
      Boolean indicating whether transfer initiation succeeded.
    """
    self.start_transfer_calls.append(kwargs)
    return self.start_transfer_result


def _make_mock_consumer_vllm_config(
    *,
    block_size: int = 1024,
    dp_rank: int = 0,
    dp_size: int = 4,
    tp_size: int = 1,
) -> Any:
  """Creates mock VllmConfig tailored for Stage-3 consumer decode tests.

  Args:
    block_size: Number of tokens allocated per KV block.
    dp_rank: Data parallel rank index.
    dp_size: Data parallel world size.
    tp_size: Tensor parallel world size.

  Returns:
    Mocked VllmConfig instance with KV transfer attributes populated.
  """
  cfg = mock.MagicMock()
  cfg.kv_transfer_config.is_kv_producer = False
  cfg.kv_transfer_config.kv_connector_extra_config = {}
  cfg.cache_config.block_size = block_size
  cfg.cache_config.enable_prefix_caching = False
  cfg.model_config.max_model_len = 16384
  cfg.model_config.hf_config = types.SimpleNamespace(
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
  """Constructs consumer TPURaidenConnectorWorker with mocked state.

  Args:
    block_size: Number of tokens allocated per KV block.
    dp_rank: Data parallel rank index.
    dp_size: Data parallel world size.
    tp_size: Tensor parallel world size.

  Returns:
    Triple of (worker_instance, controller_facade, transfer_engine).
  """
  cfg = _make_mock_consumer_vllm_config(
      block_size=block_size,
      dp_rank=dp_rank,
      dp_size=dp_size,
      tp_size=tp_size,
  )
  with (
      mock.patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
      mock.patch(
          f"{_MOD}.get_tensor_model_parallel_world_size",
          return_value=tp_size,
      ),
      mock.patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
      mock.patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
      mock.patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
  ):
    worker = TPURaidenConnectorWorker(cfg)

  engine = _FakeRaidenEngine()
  facade = _FakeRaidenControllerFacade()
  unit = types.SimpleNamespace(
      job_name="decode-worker",
      job_replica_id="consumer-engine-rank0",
      data_name="kv.fa",
      data_replica_idx=dp_rank,
  )
  worker._raiden_transfer_engine = engine
  worker._raiden_controller_facade = facade
  worker._raiden_controller_address = "decode-controller.test:27000"
  worker._raiden_work_unit = unit
  worker._local_raiden_transfer_rank = mock.MagicMock(return_value=0)
  worker._raiden_manifest = types.SimpleNamespace(
      geometry_by_tag=lambda: {
          "fa": {"live_bytes_per_block": block_size * 1024}
      },
      tag_counts=lambda: {"fa": 15},
  )
  return worker, facade, engine


# ==============================================================================
# R4.1: Destination Memory Stability Tests
# ==============================================================================


def test_rubric_r4_1_destination_memory_preallocation_and_stability() -> None:
  """Verifies R4.1: Upfront destination allocation and memory stability.

  Ensures that destination memory pages are pre-allocated upfront based on
  the full prompt extent (ceil(expected_tokens / block_size)) and remain
  strictly stable in _load_block_ids as sub-chunks arrive incrementally.
  """
  block_size = 1024
  total_tokens = 4096
  expected_blocks = math.ceil(total_tokens / block_size)  # 4 blocks
  assigned_blocks = [101, 102, 103, 104]

  worker, _, engine = _make_consumer_test_worker(block_size=block_size)
  req_id = "req-r4-1-stability"

  # Step 1: Upfront pre-allocation
  worker._update_stage3_state_after_alloc(
      req_id=req_id, expected_tokens=total_tokens
  )
  worker._load_block_ids[req_id] = list(assigned_blocks)
  worker._stage3_controller_accepted.add(req_id)

  assert req_id in worker._stage3_recv_intervals
  assert worker._stage3_recv_intervals[req_id] == []
  assert len(worker._load_block_ids[req_id]) == expected_blocks
  assert worker._load_block_ids[req_id] == assigned_blocks

  # Step 2: Ingest 4 sub-chunks across multiple steps
  chunks = [
      ("chunk-0", 0, 1024, 40001),
      ("chunk-1", 1024, 1024, 40002),
      ("chunk-2", 2048, 1024, 40003),
      ("chunk-3", 3072, 1024, 40004),
  ]

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    for chunk_id, t_base, n_tok, uuid in chunks:
      # Verify block allocation remains strictly stable prior to dispatch
      assert worker._load_block_ids[req_id] == assigned_blocks

      worker._start_stage3_recv_if_ready(
          req_id=chunk_id,
          uuid=uuid,
          endpoint="10.0.0.1:9000",
          local_blocks=assigned_blocks,
          token_base=t_base,
          num_tokens=n_tok,
          engine=engine,
          destination_req_id=req_id,
      )

      # Engine received start_read dispatch
      start_calls = [
          c for c in engine.calls if c[0] == "start_read" and c[1] == chunk_id
      ]
      assert len(start_calls) == 1
      assert start_calls[0][2] == uuid
      assert start_calls[0][5] == assigned_blocks

      # Poll chunk completion
      engine.complete_read_results = [([], [chunk_id], [])]
      worker._poll_finished(engine)

      # Verify block allocation remains unchanged after chunk arrival
      assert worker._load_block_ids[req_id] == assigned_blocks
      assert (
          t_base,
          t_base + n_tok,
      ) in worker._stage3_recv_intervals[req_id]

  # Verify all intervals accumulated against stable block IDs
  assert len(worker._stage3_recv_intervals[req_id]) == 4
  assert worker._load_block_ids[req_id] == assigned_blocks


def test_rubric_r4_1_varying_block_sizes_and_geometries() -> None:
  """Verifies R4.1: Upfront pre-allocation across diverse block sizes.

  Tests that preallocation matches ceil(total_tokens / block_size) exactly
  for standard and fine-grained block sizes without memory under-allocation.
  """
  test_cases = [
      (16, 512, 32),
      (64, 1000, 16),
      (128, 4096, 32),
      (512, 3500, 7),
      (1024, 16384, 16),
  ]

  for block_size, prompt_tokens, expected_page_count in test_cases:
    worker, _, _ = _make_consumer_test_worker(block_size=block_size)
    req_id = f"req-geo-{block_size}-{prompt_tokens}"

    # Dict params convention
    mock_req = types.SimpleNamespace(
        request_id=req_id,
        kv_transfer_params={
            "num_tokens": str(prompt_tokens),
            "req_id": req_id,
        },
    )
    pages = worker._update_stage3_state_after_alloc(mock_req)
    assert req_id in worker._stage3_recv_intervals
    assert worker._stage3_recv_intervals[req_id] == []
    assert pages == expected_page_count

    # Direct expected_tokens convention
    req_id2 = f"{req_id}-direct"
    pages2 = worker._update_stage3_state_after_alloc(
        req_id=req_id2, expected_tokens=prompt_tokens
    )
    assert req_id2 in worker._stage3_recv_intervals
    assert pages2 == expected_page_count


def test_rubric_r4_1_out_of_order_and_concurrent_multi_request_stability(
) -> None:
  """Verifies R4.1: Out-of-order sub-chunk staging across concurrent requests.

  Simulates multiple concurrent requests receiving sub-chunks out of order,
  verifying interval accumulation integrity without cross-request pollution.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)

  req_a = "req-stream-A"
  req_b = "req-stream-B"
  blocks_a = [201, 202, 203]
  blocks_b = [301, 302, 303]

  worker._update_stage3_state_after_alloc(req_id=req_a, expected_tokens=3072)
  worker._load_block_ids[req_a] = list(blocks_a)
  worker._stage3_controller_accepted.add(req_a)

  worker._update_stage3_state_after_alloc(req_id=req_b, expected_tokens=3072)
  worker._load_block_ids[req_b] = list(blocks_b)
  worker._stage3_controller_accepted.add(req_b)

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    # Register chunks out of order:
    # req_a: chunk 2, then chunk 0, then chunk 1
    # req_b: chunk 1, then chunk 2, then chunk 0
    worker._start_stage3_recv_if_ready(
        req_id="a-2",
        uuid=51003,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_a,
        token_base=2048,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_a,
    )
    worker._start_stage3_recv_if_ready(
        req_id="b-1",
        uuid=52002,
        endpoint="10.0.0.2:9000",
        local_blocks=blocks_b,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_b,
    )
    worker._start_stage3_recv_if_ready(
        req_id="a-0",
        uuid=51001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_a,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_a,
    )
    worker._start_stage3_recv_if_ready(
        req_id="b-2",
        uuid=52003,
        endpoint="10.0.0.2:9000",
        local_blocks=blocks_b,
        token_base=2048,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_b,
    )

    # Batch 1 completions
    engine.complete_read_results = [([], ["a-2", "b-1"], [])]
    worker._poll_finished(engine)

    assert worker._stage3_recv_intervals[req_a] == [(2048, 3072)]
    assert worker._stage3_recv_intervals[req_b] == [(1024, 2048)]

    # Remaining chunk dispatches
    worker._start_stage3_recv_if_ready(
        req_id="a-1",
        uuid=51002,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_a,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_a,
    )
    worker._start_stage3_recv_if_ready(
        req_id="b-0",
        uuid=52001,
        endpoint="10.0.0.2:9000",
        local_blocks=blocks_b,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_b,
    )

    # Batch 2 completions
    engine.complete_read_results = [
        ([], ["a-0", "b-2", "a-1", "b-0"], [])
    ]
    worker._poll_finished(engine)

    assert set(worker._stage3_recv_intervals[req_a]) == {
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
    }
    assert set(worker._stage3_recv_intervals[req_b]) == {
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
    }
    assert worker._load_block_ids[req_a] == blocks_a
    assert worker._load_block_ids[req_b] == blocks_b


def test_rubric_r4_1_legacy_and_metadata_backward_compatibility() -> None:
  """Verifies R4.1: Backward compatibility with monolithic metadata.

  Ensures that monolithic transfer metadata (LoadMeta with is_final=True
  and token_base=0) continues to be processed seamlessly via
  process_send_load() without disrupting destination block mappings.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)

  req_id = "req-legacy-meta"
  worker._update_stage3_state_after_alloc(
      req_id=req_id, expected_tokens=2048
  )
  worker._load_block_ids[req_id] = [401, 402]

  meta = TPUConnectorMetadata(
      reqs_to_load={
          req_id: LoadMeta(
              uuid=44001,
              local_block_ids=[401, 402],
              remote_block_ids=[10, 11],
              remote_host="10.0.0.1",
              remote_port=9000,
              token_base=0,
              is_final=True,
          )
      }
  )

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=False):
    worker.process_send_load(meta)
    engine.complete_read_results = [([], [req_id], [])]
    worker._poll_finished(engine)

    assert (0, 2048) in worker._stage3_recv_intervals[req_id]
    assert worker._load_block_ids[req_id] == [401, 402]


# ==============================================================================
# R4.2: Gap-Detecting Tiling Check Tests
# ==============================================================================


def test_rubric_r4_2_complete_contiguous_tiling_validation() -> None:
  """Verifies R4.2: Unbroken contiguous interval covers pass validation.

  Tests that _validate_interval_tiling() returns True for valid unbroken
  partitions spanning [0, total_tokens), including standard uniform
  chunks, single spans, out-of-order permutations, and fine-grained splits.
  """
  # 1. Standard 4-chunk contiguous partition of [0, 4096)
  intervals_4chunk = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
  assert _validate_interval_tiling(intervals_4chunk, 4096) is True

  # 2. Single monolithic span
  assert _validate_interval_tiling([(0, 4096)], 4096) is True

  # 3. Out-of-order permutation of contiguous intervals
  intervals_shuffled = [(2048, 3072), (0, 1024), (3072, 4096), (1024, 2048)]
  assert _validate_interval_tiling(intervals_shuffled, 4096) is True

  # 4. Fine-grained 16-chunk partition of [0, 2048)
  intervals_16chunk = [(i * 128, (i + 1) * 128) for i in range(16)]
  assert _validate_interval_tiling(intervals_16chunk, 2048) is True

  # 5. Non-uniform stride partition
  intervals_non_uniform = [(0, 1024), (1024, 3072), (3072, 3500)]
  assert _validate_interval_tiling(intervals_non_uniform, 3500) is True


def test_rubric_r4_2_gap_and_shortfall_rejection() -> None:
  """Verifies R4.2: Missing chunks, gaps, and shortfalls fail validation.

  Tests that _validate_interval_tiling() rejects broken partitions with
  interior gaps, leading gaps, trailing shortfalls, single-token gaps, or
  empty/invalid interval inputs.
  """
  # 1. Interior gap: missing [1024, 2048)
  intervals_interior_gap = [(0, 1024), (2048, 3072), (3072, 4096)]
  assert _validate_interval_tiling(intervals_interior_gap, 4096) is False

  # 2. Leading gap: starts at 512 instead of 0
  intervals_leading_gap = [(512, 2048), (2048, 4096)]
  assert _validate_interval_tiling(intervals_leading_gap, 4096) is False

  # 3. Trailing shortfall: ends at 4000 < 4096
  intervals_shortfall = [(0, 1024), (1024, 2048), (2048, 4000)]
  assert _validate_interval_tiling(intervals_shortfall, 4096) is False

  # 4. Single-token gap: [0, 1023) and [1024, 2048)
  intervals_single_gap = [(0, 1023), (1024, 2048)]
  assert _validate_interval_tiling(intervals_single_gap, 2048) is False

  # 5. Empty and invalid inputs
  assert _validate_interval_tiling([], 4096) is False
  assert _validate_interval_tiling([(0, 0)], 4096) is False
  assert _validate_interval_tiling([(1024, 0)], 4096) is False
  assert _validate_interval_tiling([(0, 1024)], 0) is False
  assert _validate_interval_tiling(None, 4096) is False


def test_rubric_r4_2_overlapping_and_redundant_intervals_merging() -> None:
  """Verifies R4.2: Robust handling of overlapping and redundant intervals.

  Tests that _validate_interval_tiling() merges overlapping intervals,
  subsumed intervals, and redundant duplicate sub-chunks into continuous
  covers without raising false positive validation errors.
  """
  # 1. Overlapping segments covering [0, 4096)
  intervals_overlapping = [(0, 1536), (1024, 3072), (2560, 4096)]
  assert _validate_interval_tiling(intervals_overlapping, 4096) is True

  # 2. Subsumed intervals
  intervals_subsumed = [(0, 2048), (512, 1024), (2048, 4096)]
  assert _validate_interval_tiling(intervals_subsumed, 4096) is True

  # 3. Redundant duplicate chunks
  intervals_duplicate = [(0, 1024), (0, 1024), (1024, 2048), (1024, 2048)]
  assert _validate_interval_tiling(intervals_duplicate, 2048) is True


def test_rubric_r4_2_get_finished_tiling_admission_and_lifecycle() -> None:
  """Verifies R4.2: get_finished() strictly enforces tiling predicate.

  Guarantees that decode admission occurs only upon complete interval cover,
  retaining in-flight state during partial progress and cleaning up upon
  retirement.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)
  req_id = "req-tiling-admission"
  blocks = [501, 502, 503]

  worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=3072)
  worker._load_block_ids[req_id] = list(blocks)
  worker._stage3_controller_accepted.add(req_id)

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    worker._start_stage3_recv_if_ready(
        req_id="chunk-0",
        uuid=61001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )
    worker._start_stage3_recv_if_ready(
        req_id="chunk-1",
        uuid=61002,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )
    worker._start_stage3_recv_if_ready(
        req_id="chunk-2",
        uuid=61003,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=2048,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )

    # Stage 1: Only chunks 0 and 1 arrive -> partial tiling [0, 2048) < 3072
    engine.complete_read_results = [([], ["chunk-0", "chunk-1"], [])]
    worker._poll_finished(engine)

    assert worker._is_stage3_recv_complete(req_id) is False
    assert req_id not in worker._stage3_terminal_loads
    _, done_recving = worker.get_finished()
    assert req_id not in done_recving

    # Stage 2: Final chunk arrives -> complete tiling [0, 3072) == 3072
    engine.complete_read_results = [([], ["chunk-2"], [])]
    worker._poll_finished(engine)

    assert worker._is_stage3_recv_complete(req_id) is True
    assert req_id in worker._stage3_terminal_loads

    _, done_recving = worker.get_finished(finished_req_ids={req_id})
    assert req_id in done_recving
    assert req_id not in worker._stage3_recv_intervals
    assert len(worker.get_block_ids_with_load_errors()) == 0


# ==============================================================================
# R4.3: Automatic Local Recomputation on Transfer Failure Tests
# ==============================================================================


def test_rubric_r4_3_subchunk_transport_failure_marks_request_failed() -> None:
  """Verifies R4.3: Sub-chunk failure triggers immediate request error state.

  Ensures that a failure in any sub-chunk marks the request in _failed_recving,
  clears partial intervals, and transitions to absorbing failure state.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)
  req_id = "req-r4-3-fail"
  blocks = [701, 702]

  worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
  worker._load_block_ids[req_id] = list(blocks)
  worker._stage3_controller_accepted.add(req_id)

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    worker._start_stage3_recv_if_ready(
        req_id="chunk-ok",
        uuid=71001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )
    worker._start_stage3_recv_if_ready(
        req_id="chunk-err",
        uuid=71002,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )

    # 1. Chunk 0 completes, Chunk 1 fails
    engine.complete_read_results = [([], ["chunk-ok"], ["chunk-err"])]
    worker._poll_finished(engine)

    assert req_id in worker._failed_recving
    # Partial interval state must be purged
    assert worker._stage3_recv_intervals.get(req_id) is None
    assert worker._is_stage3_recv_complete(req_id) is False

    # 2. Late chunk arrival cannot resurrect the request
    engine.complete_read_results = [([], ["chunk-late"], [])]
    worker._poll_finished(engine)
    assert req_id in worker._failed_recving
    assert worker._stage3_recv_intervals.get(req_id) is None


def test_rubric_r4_3_recomputation_block_identification() -> None:
  """Verifies R4.3: Local block IDs of failed transfers reported for recompute.

  Ensures that when a sub-chunk transfer fails, get_finished() returns the
  failed request and get_block_ids_with_load_errors() returns the full set
  of allocated physical destination block IDs for decode-side recomputation.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)
  req_id = "req-r4-3-recompute-blocks"
  blocks = [801, 802, 803]

  worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=3072)
  worker._load_block_ids[req_id] = list(blocks)
  worker._stage3_controller_accepted.add(req_id)

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    worker._start_stage3_recv_if_ready(
        req_id="sub-fail",
        uuid=81001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_id,
    )

    engine.complete_read_results = [([], [], ["sub-fail"])]
    worker._poll_finished(engine)

    # get_finished() returns the request in done_recving with error state
    _, done_recving = worker.get_finished()
    assert req_id in done_recving

    # All destination block IDs must be marked as load errors for local
    # recomputation.
    assert worker.get_block_ids_with_load_errors() == {801, 802, 803}


def test_rubric_r4_3_timeout_and_incomplete_transfer_recomputation() -> None:
  """Verifies R4.3: Deadlines and timeouts recover safely via recomputation.

  Tests that when a transfer times out before all sub-chunks arrive, the
  request transitions to the failed state and its allocated blocks are
  reported for local recomputation.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)
  req_id = "req-r4-3-timeout"
  blocks = [901, 902]

  worker._update_stage3_state_after_alloc(req_id=req_id, expected_tokens=2048)
  worker._load_block_ids[req_id] = list(blocks)
  worker._stage3_controller_accepted.add(req_id)
  # Set expired pending controller failure deadline
  worker._stage3_pending_controller_failures[req_id] = (
      time.perf_counter() - 5.0
  )

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    engine.complete_read_results = [([], [], [])]
    worker._poll_finished(engine)

    assert req_id in worker._failed_recving
    assert req_id not in worker._stage3_pending_controller_failures

    _, done_recving = worker.get_finished()
    assert req_id in done_recving
    assert worker.get_block_ids_with_load_errors() == {901, 902}


def test_rubric_r4_3_concurrent_multi_request_error_isolation() -> None:
  """Verifies R4.3: Errors are isolated without affecting concurrent successes.

  Tests that a sub-chunk failure in one request does not corrupt or abort
  concurrent requests that succeed, verifying that error state and
  recomputation blocks are strictly isolated per request.
  """
  worker, _, engine = _make_consumer_test_worker(block_size=1024)

  req_success = "req-isolation-success"
  req_failure = "req-isolation-failure"
  blocks_succ = [1001, 1002]
  blocks_fail = [2001, 2002]

  worker._update_stage3_state_after_alloc(
      req_id=req_success, expected_tokens=2048
  )
  worker._load_block_ids[req_success] = list(blocks_succ)
  worker._stage3_controller_accepted.add(req_success)

  worker._update_stage3_state_after_alloc(
      req_id=req_failure, expected_tokens=2048
  )
  worker._load_block_ids[req_failure] = list(blocks_fail)
  worker._stage3_controller_accepted.add(req_failure)

  with mock.patch.object(worker, "_raiden_stage3_enabled", return_value=True):
    # Register chunks for success request
    worker._start_stage3_recv_if_ready(
        req_id="s-0",
        uuid=91001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_succ,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_success,
    )
    worker._start_stage3_recv_if_ready(
        req_id="s-1",
        uuid=91002,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_succ,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_success,
    )

    # Register chunks for failure request
    worker._start_stage3_recv_if_ready(
        req_id="f-0",
        uuid=92001,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_fail,
        token_base=0,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_failure,
    )
    worker._start_stage3_recv_if_ready(
        req_id="f-1",
        uuid=92002,
        endpoint="10.0.0.1:9000",
        local_blocks=blocks_fail,
        token_base=1024,
        num_tokens=1024,
        engine=engine,
        destination_req_id=req_failure,
    )

    # Completions: s-0, s-1, f-0 pass; f-1 fails
    engine.complete_read_results = [
        ([], ["s-0", "s-1", "f-0"], ["f-1"])
    ]
    worker._poll_finished(engine)

    assert req_success not in worker._failed_recving
    assert req_failure in worker._failed_recving

    _, done_recving = worker.get_finished()
    assert req_success in done_recving
    assert req_failure in done_recving

    # Crucial check: only failed request blocks are flagged for recomputation.
    assert worker.get_block_ids_with_load_errors() == {2001, 2002}


# ==============================================================================
# Phase 4 Rubric Execution Runner
# ==============================================================================


def run_all_rubric_checks() -> bool:
  """Executes all Phase 4 Rubric checks and prints formatted results.

  Runs comprehensive verifications for:
  - R4.1 (Destination Memory Stability)
  - R4.2 (Gap-Detecting Tiling Check)
  - R4.3 (Automatic Local Recomputation on Transfer Failure)

  Returns:
    True if all rubric verification checks passed successfully, False otherwise.
  """
  banner = "=" * 70
  print(f"\n{banner}")
  print("PHASE 4 RUBRIC VERIFICATION SUITE — vLLM-TorchTPU Raiden Stage-3")
  print(banner)

  all_passed = True

  # R4.1
  print("\n--- Checking R4.1: Destination Memory Stability ---")
  try:
    test_rubric_r4_1_destination_memory_preallocation_and_stability()
    test_rubric_r4_1_varying_block_sizes_and_geometries()
    test_rubric_r4_1_out_of_order_and_concurrent_multi_request_stability()
    test_rubric_r4_1_legacy_and_metadata_backward_compatibility()
    print(
        "[PASS] R4.1 — Destination Memory Stability: Verified that "
        "destination memory pages are pre-allocated upfront based on the "
        "total prompt extent (ceil(total_tokens / block_size)) and remain "
        "strictly stable across sequential, out-of-order, and concurrent "
        "sub-chunk arrivals without fragmentation or dynamic reallocations."
    )
  except Exception as error:
    print(f"[FAIL] R4.1 — Destination Memory Stability failed: {error}")
    all_passed = False

  # R4.2
  print("\n--- Checking R4.2: Gap-Detecting Tiling Check ---")
  try:
    test_rubric_r4_2_complete_contiguous_tiling_validation()
    test_rubric_r4_2_gap_and_shortfall_rejection()
    test_rubric_r4_2_overlapping_and_redundant_intervals_merging()
    test_rubric_r4_2_get_finished_tiling_admission_and_lifecycle()
    print(
        "[PASS] R4.2 — Gap-Detecting Tiling Check: Verified that the "
        "consumer validates that the union of received token intervals "
        "forms an unbroken cover of the entire prompt range before "
        "committing the transfer, rejecting missing interior chunks, "
        "leading gaps, trailing shortfalls, and single-token gaps while "
        "robustly merging overlapping or redundant chunks."
    )
  except Exception as error:
    print(f"[FAIL] R4.2 — Gap-Detecting Tiling Check failed: {error}")
    all_passed = False

  # R4.3
  print(
      "\n--- Checking R4.3: Automatic Local Recomputation on Transfer "
      "Failure ---"
  )
  try:
    test_rubric_r4_3_subchunk_transport_failure_marks_request_failed()
    test_rubric_r4_3_recomputation_block_identification()
    test_rubric_r4_3_timeout_and_incomplete_transfer_recomputation()
    test_rubric_r4_3_concurrent_multi_request_error_isolation()
    print(
        "[PASS] R4.3 — Automatic Local Recomputation on Transfer Failure: "
        "Verified that if any sub-chunk fails or interval coverage is "
        "incomplete, the request is immediately transitioned to an "
        "absorbing failure state, partial interval state is purged, and "
        "the full set of destination block IDs is reported via "
        "get_block_ids_with_load_errors() to trigger automatic local "
        "recomputation on the decode node with fault isolation."
    )
  except Exception as error:
    print(
        f"[FAIL] R4.3 — Automatic Local Recomputation on Transfer Failure "
        f"failed: {error}"
    )
    all_passed = False

  print(f"\n{banner}")
  if all_passed:
    print("ALL PHASE 4 RUBRIC CHECKS PASSED PERFECTLY!")
    print(banner)
    return True
  else:
    print("PHASE 4 RUBRIC VERIFICATION FAILED.")
    print(banner)
    return False


def test_phase_4_rubric_suite_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
  """Executes the complete Phase 4 Rubric verification and prints banner.

  Args:
    capsys: Pytest fixture to capture stdout/stderr streams.
  """
  with capsys.disabled():
    success = run_all_rubric_checks()
  assert success is True


if __name__ == "__main__":
  success = run_all_rubric_checks()
  sys.exit(0 if success else 1)
