# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_ZIGZAG_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner,
    apply_pcp_zigzag_token_order,
    pcp_zigzag_page_table,
    pcp_zigzag_seq_arrays,
    pcp_zigzag_token_layout,
    prepare_pcp_zigzag_sequence_layout,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_ZIGZAG_SEQUENCE_LAYOUT_PROTOCOL,
    is_pcp_zigzag_sequence_layout_descriptor,
)


def _input_batch(*, computed, prompt):
    req_ids = [f"req{i}" for i in range(len(computed))]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )


def _scheduler_output(scheduled):
    return SimpleNamespace(
        num_scheduled_tokens={
            f"req{i}": int(tokens) for i, tokens in enumerate(scheduled)
        }
    )


def test_pcp_zigzag_token_layout_single_request():
    pcp_size = 2
    counts = [256]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size)

    assert chunk == [64]
    assert off == [0]
    assert s_live == 128


def test_pcp_zigzag_token_layout_multi_request():
    pcp_size = 4
    counts = [200, 300]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size)

    # two_p = 8. ceil(200 / 8) = 25. ceil(300 / 8) = 38.
    assert chunk == [25, 38]
    assert off == [0, 50]
    assert s_live == 126


def test_pcp_zigzag_seq_arrays():
    pcp_size = 2
    two_p = 4
    n_off = 4
    chunk = [32]
    off = [0]
    counts = [100]  # chunk is 32, so 4 chunks = 128 total capacity

    cu_q_lens, q_pos_offsets, pcp_write = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off, counts
    )

    assert cu_q_lens.shape == (pcp_size, n_off + 1)
    assert q_pos_offsets.shape == (pcp_size, n_off)
    assert pcp_write.shape == (pcp_size, n_off)

    # For rank 0: owns chunk 0 (head) and chunk 3 (tail)
    # Head chunk offset is 0 * 32 = 0
    assert q_pos_offsets[0, 0] == 0
    # Tail chunk offset is 3 * 32 = 96
    assert q_pos_offsets[0, 1] == 96

    # For rank 1: owns chunk 1 (head) and chunk 2 (tail)
    # Head chunk offset is 1 * 32 = 32
    assert q_pos_offsets[1, 0] == 32
    # Tail chunk offset is 2 * 32 = 64
    assert q_pos_offsets[1, 1] == 64

    # Writer flags: tail chunk has real tokens (100 - 3*32 = 4 > 0), so tail writes
    assert pcp_write[0, 1] == 1
    assert pcp_write[0, 0] == 0


def test_pcp_zigzag_seq_arrays_writer_fallback():
    pcp_size = 2
    two_p = 4
    n_off = 4
    chunk = [32]
    off = [0]
    # Total tokens is only 64, so chunks 2 and 3 have 0 real tokens
    counts = [64]

    cu_q_lens, q_pos_offsets, pcp_write = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off, counts
    )

    # Rank 0 owns chunk 0 (head) and chunk 3 (tail).
    # Chunk 3 has start offset 3*32 = 96 >= 64, so 0 real tokens.
    # Therefore, rank 0's writer flag should fall back to chunk 0 (head)!
    assert pcp_write[0, 1] == 0
    assert pcp_write[0, 0] == 1


def test_pcp_zigzag_page_table():
    pcp_size = 2
    block_size = 16
    n_off = 4
    chunk = [32]  # pages_per_chunk = 32 // 16 = 2

    table = pcp_zigzag_page_table(chunk, pcp_size, block_size, n_off)
    assert table.shape == (n_off, 2 * pcp_size * 2)
    # Chunk 0 (rank 0 head): pages 0, 1
    assert np.array_equal(table[0, 0:2], [0, 1])
    # Chunk 1 (rank 1 head): pages 4, 5
    assert np.array_equal(table[0, 2:4], [4, 5])
    # Chunk 2 (rank 1 tail): pages 6, 7
    assert np.array_equal(table[0, 4:6], [6, 7])
    # Chunk 3 (rank 0 tail): pages 2, 3
    assert np.array_equal(table[0, 6:8], [2, 3])


def test_apply_pcp_zigzag_token_order():
    pcp_size = 2
    counts = [128]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size)
    s_pad = 64
    padded_total_num_tokens = s_pad * pcp_size  # 128

    input_ids = np.arange(128, dtype=np.int32)
    positions_np = np.arange(128, dtype=np.int32)

    req_to_packed, packed_to_req = apply_pcp_zigzag_token_order(
        input_ids, positions_np, counts, chunk, off, pcp_size, padded_total_num_tokens
    )

    # Verify that the permutation is invertible
    assert np.array_equal(packed_to_req[req_to_packed], np.arange(128))

    # Verify rank 0 slot contains chunk 0 (head, 0..31) and chunk 3 (tail, 96..127)
    rank0_tokens = input_ids[:s_pad]
    assert np.array_equal(rank0_tokens[0:32], np.arange(0, 32))
    assert np.array_equal(rank0_tokens[32:64], np.arange(96, 128))

    # Verify rank 1 slot contains chunk 1 (head, 32..63) and chunk 2 (tail, 64..95)
    rank1_tokens = input_ids[s_pad : 2 * s_pad]
    assert np.array_equal(rank1_tokens[0:32], np.arange(32, 64))
    assert np.array_equal(rank1_tokens[32:64], np.arange(64, 96))


def test_pcp_zigzag_eligibility():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=4,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    assert eligibility.enabled
    assert eligibility.is_zigzag

    decision = eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=[0], prompt=[256]),
        scheduler_output=_scheduler_output([256]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([256], dtype=np.int32),
        num_tokens_paddings=[128, 256, 512],
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.ZIGZAG
    assert decision.local_required_tokens == 256 // 4  # 64 per rank


def test_pcp_zigzag_planner_prepare_dummy():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=2,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    planner = PcpSequenceLayoutPlanner(eligibility)

    plan = planner.prepare_dummy(
        num_tokens=128,
        num_reqs=2,
        kv_cache_initialized=True,
    )

    assert is_pcp_zigzag_sequence_layout_descriptor(plan.descriptor)
    assert plan.descriptor.protocol == PCP_ZIGZAG_SEQUENCE_LAYOUT_PROTOCOL
    assert plan.pcp_metadata is not None
    assert plan.pcp_metadata.cache_pages == 0
    assert plan.pcp_metadata.query_start_loc.shape[0] == 2
