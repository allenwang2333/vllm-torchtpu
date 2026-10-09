# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and coverage tests for Track TD7.22:
- SpeculativeDecodingManager (1D packed H2D upload, uniform-draft 2D broadcast
  fast path, cache-miss mutation isolation, take/stage/propose hooks)
- tpu_runner_async_output (single-chunk sampled_token_ids_cpu fast path and
  tail-sentinel short-circuit in parse_sampled_token_ids_cpu)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    parse_sampled_token_ids_cpu,
)


class _FakeTPUTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _as_fake_tpu(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_FakeTPUTensor)


def _reference_spec_decode_arrays(
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
    all_token_ids: np.ndarray,
    padded_logits_length: int,
    padded_num_reqs: int,
) -> dict[str, np.ndarray]:
    """Reference numpy implementation before TD7.22."""
    num_sampled_tokens = num_draft_tokens + 1
    cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
    cu_sampled_starts = cu_num_sampled_tokens - num_sampled_tokens
    rep_sampled_starts = np.repeat(cu_sampled_starts, num_sampled_tokens)
    arange = np.arange(rep_sampled_starts.shape[0], dtype=np.int32) - rep_sampled_starts
    logits_indices = (
        np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        + arange
    )
    bonus_logits_indices = cu_num_sampled_tokens - 1
    cu_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
    rep_draft_starts = np.repeat(cu_draft_tokens - num_draft_tokens, num_draft_tokens)
    group_indices = (
        np.arange(rep_draft_starts.shape[0], dtype=np.int32) - rep_draft_starts
    )
    target_logits_indices = (
        np.repeat(cu_sampled_starts, num_draft_tokens) + group_indices
    )
    draft_token_ids = all_token_ids[logits_indices[target_logits_indices + 1]]

    padded_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
    padded_logits_indices[: logits_indices.shape[0]] = logits_indices

    padded_bonus_logits_indices = np.zeros(padded_num_reqs, dtype=np.int32)
    padded_bonus_logits_indices[: bonus_logits_indices.shape[0]] = bonus_logits_indices

    padded_num_draft_tokens = np.zeros(padded_num_reqs, dtype=np.int32)
    padded_num_draft_tokens[: num_draft_tokens.shape[0]] = num_draft_tokens

    padded_draft_token_ids = np.zeros(padded_logits_length, dtype=np.int32)
    padded_draft_token_ids[: draft_token_ids.shape[0]] = draft_token_ids

    padded_target_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
    padded_target_logits_indices[: target_logits_indices.shape[0]] = (
        target_logits_indices
    )

    segment_ids = np.repeat(
        np.arange(num_draft_tokens.shape[0], dtype=np.int64), num_draft_tokens
    )
    padded_segment_ids = np.full(
        padded_logits_length, num_draft_tokens.shape[0], dtype=np.int64
    )
    padded_segment_ids[: segment_ids.shape[0]] = segment_ids

    padded_group_indices = np.zeros(padded_logits_length, dtype=np.int32)
    padded_group_indices[: group_indices.shape[0]] = group_indices

    return {
        "draft_token_ids": padded_draft_token_ids,
        "draft_lengths": padded_num_draft_tokens,
        "target_logits_indices": padded_target_logits_indices,
        "bonus_logits_indices": padded_bonus_logits_indices,
        "final_logits_indices": padded_logits_indices,
        "segment_ids": padded_segment_ids,
        "group_indices": padded_group_indices,
    }


def _make_runner(max_tokens: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_tokens=max_tokens,
        input_ids_cpu=torch.arange(1000, 1000 + max_tokens, dtype=torch.int32),
        arange_np=np.arange(max_tokens, dtype=np.int32),
        num_tokens_paddings=[8, 16, 32, 64, 128],
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            num_reqs=3,
            req_ids=["req-0", "req-1", "req-2", "req-unused"],
            num_tokens_no_spec=[10, 20, 30],
            token_ids_cpu=np.zeros((4, 64), dtype=np.int32),
        ),
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(method="dspark"),
        drafter=None,
        _async_drafter=None,
    )


@pytest.mark.parametrize(
    ("num_draft_list", "cu_sched_list", "padded_num_reqs"),
    [
        ([5], [6], 4),  # single request uniform k=5
        ([5, 5, 5, 5], [6, 12, 18, 24], 4),  # multi-request uniform k=5
        ([3, 3, 3], [10, 14, 22], 4),  # uniform drafts with mixed scheduled lengths
        ([0, 0, 0], [1, 2, 3], 4),  # uniform k=0 drafts
        ([5, 2, 0, 3], [6, 9, 10, 14], 8),  # variable draft lengths
        ([4, 4, 1], [5, 10, 12], 4),  # first two equal, last differs
    ],
)
def test_get_spec_decode_metadata_parity_and_packed_storage(
    num_draft_list: list[int],
    cu_sched_list: list[int],
    padded_num_reqs: int,
):
    runner = _make_runner()
    mgr = SpeculativeDecodingManager(runner)
    num_draft = np.array(num_draft_list, dtype=np.int32)
    cu_sched = np.array(cu_sched_list, dtype=np.int32)

    md = mgr.get_spec_decode_metadata(
        num_draft, cu_sched, padded_num_reqs=padded_num_reqs
    )
    padded_logits_len = md.final_logits_indices.shape[0]
    ref = _reference_spec_decode_arrays(
        num_draft,
        cu_sched,
        runner.input_ids_cpu.numpy(),
        padded_logits_len,
        padded_num_reqs,
    )

    for key, expected in ref.items():
        got = getattr(md, key)
        assert isinstance(got, torch.Tensor)
        np.testing.assert_array_equal(got.numpy(), expected, err_msg=key)
        assert got.is_contiguous()

    # All 6 int32 tensors are views into a single packed 1D storage buffer.
    base_storage_ptr = md.draft_token_ids.untyped_storage().data_ptr()
    for field_name in (
        "draft_lengths",
        "target_logits_indices",
        "bonus_logits_indices",
        "final_logits_indices",
        "group_indices",
    ):
        tensor = getattr(md, field_name)
        assert tensor.dtype == torch.int32
        assert tensor.untyped_storage().data_ptr() == base_storage_ptr
    assert md.segment_ids.dtype == torch.int64


def test_get_spec_decode_metadata_cache_miss_isolation():
    runner = _make_runner()
    mgr = SpeculativeDecodingManager(runner)
    num_draft = np.array([2, 2], dtype=np.int32)
    cu_sched = np.array([3, 6], dtype=np.int32)

    md_miss = mgr.get_spec_decode_metadata(
        num_draft, cu_sched, padded_num_reqs=2, skip_host_draft_tokens=False
    )
    assert isinstance(md_miss, SpecDecodeMetadata)
    assert md_miss is not mgr._meta_cache
    orig_cached_draft_tokens = mgr._meta_cache.draft_token_ids

    # Simulate execute_model overwriting md_miss.draft_token_ids after cache miss
    md_miss.draft_token_ids = torch.full_like(md_miss.draft_token_ids, 777)
    num_draft[0] = 99  # caller mutates input array

    # Re-create matching key array
    num_draft_hit = np.array([2, 2], dtype=np.int32)
    md_hit = mgr.get_spec_decode_metadata(
        num_draft_hit, cu_sched, padded_num_reqs=2, skip_host_draft_tokens=True
    )
    assert md_hit.draft_token_ids is orig_cached_draft_tokens
    assert (md_hit.draft_token_ids != 777).all()
    np.testing.assert_array_equal(md_hit.draft_lengths_cpu, np.array([2, 2]))

    # Cache hit with skip_host_draft_tokens=False re-extracts from input_ids_cpu
    runner.input_ids_cpu.fill_(42)
    md_hit_reupload = mgr.get_spec_decode_metadata(
        num_draft_hit, cu_sched, padded_num_reqs=2, skip_host_draft_tokens=False
    )
    assert (md_hit_reupload.draft_token_ids[:4] == 42).all()


def test_stage_and_take_draft_token_ids():
    runner = _make_runner()
    mgr = SpeculativeDecodingManager(runner)

    # 1. Nothing staged and _draft_token_ids is None -> returns None
    assert mgr.take_draft_token_ids() is None

    # 2. Host _draft_token_ids cached -> returns DraftTokenIds and clears cache
    mgr._draft_token_ids = [[10, 11], [20, 21], [30, 31]]
    out_host = mgr.take_draft_token_ids()
    assert out_host is not None
    assert out_host.req_ids == ["req-0", "req-1", "req-2"]
    assert out_host.draft_token_ids == [[10, 11], [20, 21], [30, 31]]
    assert mgr.take_draft_token_ids() is None

    # 3. stage_draft_token_ids_for_host on CPU (ready_event is None, 2D tensor)
    drafts_2d = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    mgr.stage_draft_token_ids_for_host(drafts_2d)
    out_staged_cpu = mgr.take_draft_token_ids()
    assert out_staged_cpu is not None
    assert out_staged_cpu.req_ids == ["req-0", "req-1", "req-2"]
    assert out_staged_cpu.draft_token_ids == [[1, 2], [3, 4], [5, 6]]
    assert mgr.take_draft_token_ids() is None

    # 4. stage_draft_token_ids_for_host on TPU (ready_event synced, 1D tensor)
    fake_event = MagicMock()
    drafts_1d_tpu = _as_fake_tpu(torch.tensor([7, 8, 9], dtype=torch.int32))
    with patch.object(
        torch,
        "tpu",
        SimpleNamespace(Event=MagicMock(return_value=fake_event)),
        create=True,
    ):
        mgr.stage_draft_token_ids_for_host(drafts_1d_tpu)
    fake_event.record.assert_called_once()
    out_staged_tpu = mgr.take_draft_token_ids()
    fake_event.synchronize.assert_called_once()
    assert out_staged_tpu is not None
    assert out_staged_tpu.draft_token_ids == [7, 8, 9]


def test_propose_draft_token_ids_all_branches():
    runner = _make_runner()
    mgr = SpeculativeDecodingManager(runner)

    # 1. ngram method
    runner._is_async_drafter = False
    runner.speculative_config = SimpleNamespace(method="ngram")
    fake_ngram = MagicMock(spec=NgramProposer)
    fake_ngram.propose.return_value = [[101, 102], [201, 202], [301, 302]]
    runner.drafter = fake_ngram
    sched_out = SimpleNamespace(num_spec_tokens_to_schedule=2)
    ret_ngram = mgr.propose_draft_token_ids(
        sampled_token_ids=[[1], [2], [3], [4]],
        scheduler_output=sched_out,
    )
    assert ret_ngram is None
    assert mgr._draft_token_ids == [[101, 102], [201, 202], [301, 302]]
    fake_ngram.propose.assert_called_once()

    # 2. Async drafter with return_device=False and non-empty sampled_token_ids
    runner._is_async_drafter = True
    runner.speculative_config = SimpleNamespace(method="dspark")
    async_drafter = MagicMock()
    async_drafter.propose.return_value = [[11], [22], [33]]
    runner._async_drafter = async_drafter
    ret_host = mgr.propose_draft_token_ids(
        sampled_token_ids=[[1], [2], [3], [4]],
        discard_sampled_tokens_req_indices=[1],
        return_device=False,
    )
    assert ret_host is None
    assert mgr._draft_token_ids == [[11], [22], [33]]
    assert async_drafter.propose.call_args.args[0] == [[1], [2], [3]]

    # 3. Async drafter with return_device=True and sampled_token_ids=None
    dev_tensor = torch.tensor([[11, 12], [21, 22], [31, 32]], dtype=torch.int32)
    async_drafter.propose.return_value = dev_tensor
    ret_dev = mgr.propose_draft_token_ids(
        sampled_token_ids=None,
        return_device=True,
        device_seed=torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    assert ret_dev is dev_tensor
    assert async_drafter.propose.call_args.args[0] is None

    # 4. Unsupported method raises NotImplementedError
    runner._is_async_drafter = False
    runner.speculative_config = SimpleNamespace(method="unknown_spec")
    with pytest.raises(NotImplementedError, match="unknown_spec"):
        mgr.propose_draft_token_ids(sampled_token_ids=[[1]])


def test_async_copy_state_single_and_multi_chunk_shapes():
    # 1. Single-chunk with chunk_shapes matching 1D tensor shape (t.shape == s)
    s1_1d = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([10, 20, 30, 40], dtype=torch.int32)],
        chunk_real_lens=[3],
        copy_ready_event=None,
        chunk_shapes=[(4,)],
        completed=True,
    )
    assert s1_1d.sampled_token_ids_cpu.tolist() == [10, 20, 30]

    # 2. Single-chunk with chunk_shapes reshaping 1D flat tensor to 2D (t.shape != s)
    s1_2d = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([10, 11, 20, 21, 99, 99], dtype=torch.int32)],
        chunk_real_lens=[2],
        copy_ready_event=None,
        chunk_shapes=[(3, 2)],
        completed=True,
    )
    assert s1_2d.sampled_token_ids_cpu.tolist() == [[10, 11], [20, 21]]

    # 3. Multi-chunk with chunk_shapes=None
    s2_no_shapes = AsyncTPUCopyState(
        chunks_cpu=[
            torch.tensor([[1, 2], [9, 9]], dtype=torch.int32),
            torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        ],
        chunk_real_lens=[1, 2],
        copy_ready_event=None,
        chunk_shapes=None,
        completed=True,
    )
    assert s2_no_shapes.sampled_token_ids_cpu.tolist() == [[1, 2], [3, 4], [5, 6]]

    # 4. Multi-chunk with chunk_shapes containing both reshaped and matching shapes
    s2_with_shapes = AsyncTPUCopyState(
        chunks_cpu=[
            torch.tensor([1, 2, 9, 9], dtype=torch.int32),
            torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        ],
        chunk_real_lens=[1, 2],
        copy_ready_event=None,
        chunk_shapes=[(2, 2), (2, 2)],
        completed=True,
    )
    assert s2_with_shapes.sampled_token_ids_cpu.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_parse_sampled_token_ids_cpu_tail_and_interior_sentinels():
    t = torch.tensor(
        [
            [10, 11, INVALID_TOKEN_ID, INVALID_TOKEN_ID],  # tail sentinel
            [20, INVALID_TOKEN_ID, 22, 23],  # interior sentinel only
            [30, 31, 32, 33],  # all valid (no sentinel)
        ],
        dtype=torch.int32,
    )
    rows, lens = parse_sampled_token_ids_cpu(t)
    assert rows == [[10, 11], [20, 22, 23], [30, 31, 32, 33]]
    assert lens == [2, 3, 4]
