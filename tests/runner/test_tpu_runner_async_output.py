# Copyright 2025 Google LLC
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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner import tpu_runner_async_output
from vllm_torchtpu.runner.tpu_runner import _spec_logprob_row_tokens
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    SpecLogprobsChunk,
    assemble_spec_next_tokens,
    assemble_spec_next_tokens_and_num_rejected,
    compute_num_rejected,
    extract_draft_token_ids,
    parse_sampled_token_ids_cpu,
    spec_logprobs_lists,
    subtract_num_rejected_tokens,
)


@pytest.fixture
def device():
    from vllm_torchtpu.tpu_info import get_num_chips

    if get_num_chips() == 0:
        return torch.device("cpu")
    try:
        import torch_tpu  # noqa: F401

        return torch.device("tpu")
    except Exception:
        return torch.device("cpu")


def test_assemble_spec_next_tokens(device):
    # 2 requests, K=3. Rejection rows: req0 has 2 accepted drafts + bonus (3
    # valid cols), req1 has bonus only (1 valid). Bonus = last non-(-1) col.
    next_tokens = torch.tensor(
        [
            [11, 12, 99, INVALID_TOKEN_ID],
            [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        ],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor(
        [[201, 202, 203], [211, 212, 213]], dtype=torch.int32, device=device
    )

    src = assemble_spec_next_tokens(next_tokens, drafts, num_reqs=2)

    # Per req: [bonus, draft_1..K], row-major flat. bonus = [99, 77].
    expected = torch.tensor(
        [99, 201, 202, 203, 77, 211, 212, 213], dtype=torch.int32, device=device
    )
    assert torch.equal(src, expected)


def test_assemble_spec_next_tokens_and_num_rejected(device):
    next_tokens = torch.tensor(
        [
            [11, 12, 99, INVALID_TOKEN_ID],
            [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
            [0, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        ],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor(
        [[201, 202, 203], [211, 212, 213], [999, 999, 999]],
        dtype=torch.int32,
        device=device,
    )
    num_draft = torch.tensor([3, 3, 3], dtype=torch.int32, device=device)

    src, num_rejected = assemble_spec_next_tokens_and_num_rejected(
        next_tokens, drafts, num_draft, num_reqs=2
    )
    assert torch.equal(src, assemble_spec_next_tokens(next_tokens, drafts, num_reqs=2))
    assert num_rejected.tolist() == [1, 3]

    src_no_draft, num_rejected_no_draft = assemble_spec_next_tokens_and_num_rejected(
        next_tokens, drafts, None, num_reqs=2
    )
    assert torch.equal(src_no_draft, src)
    assert num_rejected_no_draft.tolist() == [0, 0]


def test_assemble_spec_next_tokens_ignores_padded_rows(device):
    # next_tokens / drafts are bucket-padded to 3 rows; only 2 real requests.
    next_tokens = torch.tensor(
        [[5, INVALID_TOKEN_ID], [6, INVALID_TOKEN_ID], [0, INVALID_TOKEN_ID]],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor([[1], [2], [9]], dtype=torch.int32, device=device)

    src = assemble_spec_next_tokens(next_tokens, drafts, num_reqs=2)

    # K=1; bonus = [5, 6]; padded 3rd row dropped.
    expected = torch.tensor([5, 1, 6, 2], dtype=torch.int32, device=device)
    assert torch.equal(src, expected)


def test_subtract_num_rejected_tokens(device):
    # 2 reqs, K=3, n_sched=4 each → positions flat length 8. req0 rejected 1
    # draft, req1 rejected none; seq_lens/positions advanced optimistically.
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [6, 7, 8, 9, 16, 17, 18, 19], dtype=torch.int32, device=device
    )
    num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=device)
    seq_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
    pos_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)

    new_seq, new_pos = subtract_num_rejected_tokens(
        seq_lens, positions, num_rejected, seq_idx, pos_idx
    )

    # req0 over-counted by 1 (corrected), req1 unchanged.
    assert new_seq.tolist() == [9, 20]
    assert new_pos.tolist() == [5, 6, 7, 8, 16, 17, 18, 19]


def test_subtract_num_rejected_tokens_skips_invalid(device):
    # A -1 index leaves the slot unchanged (e.g. a new request with no prior
    # spec step).
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    positions = torch.tensor([5, 6], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([2], dtype=torch.int32, device=device)
    seq_idx = torch.tensor([-1, 0], dtype=torch.int32, device=device)
    pos_idx = torch.tensor([-1, 0], dtype=torch.int32, device=device)

    new_seq, new_pos = subtract_num_rejected_tokens(
        seq_lens, positions, num_rejected, seq_idx, pos_idx
    )

    assert new_seq.tolist() == [10, 18]  # slot 0 unchanged, slot 1 -= 2
    assert new_pos.tolist() == [5, 4]


def test_extract_draft_token_ids(device):
    input_ids = torch.tensor(
        [10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int32, device=device
    )
    # logits_indices picks the sampled+draft positions; target+1 selects the
    # draft positions among them.
    logits_indices = torch.tensor([1, 2, 3, 5, 6], dtype=torch.int32, device=device)
    target_logits_indices = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)

    out = extract_draft_token_ids(input_ids, logits_indices, target_logits_indices)

    # input_ids[logits_indices] = [11, 12, 13, 15, 16]; then index [1, 2, 4]
    # (target + 1) -> [12, 13, 16].
    expected = torch.tensor([12, 13, 16], dtype=torch.int32, device=device)
    assert torch.equal(out, expected)


def test_compute_num_rejected(device):
    # K=3. req0: 3 accepted + bonus (4 valid) -> 0 rejected; req1: 1 accepted +
    # bonus (2 valid) -> 2; req2: bonus only (1 valid) -> 3.
    next_tokens = torch.tensor(
        [[10, 11, 12, 99], [20, 21, -1, -1], [30, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    num_draft = torch.tensor([3, 3, 3], dtype=torch.int32, device=device)

    out = compute_num_rejected(next_tokens, num_draft)

    assert out.tolist() == [0, 2, 3]


def test_compute_num_rejected_variable_draft(device):
    # Mixed scheduled draft counts (near max-model-len). req0 K=3 all accepted
    # -> 0; req1 K=1, draft rejected (1 valid = bonus only) -> 1.
    next_tokens = torch.tensor(
        [[10, 11, 12, 99], [20, -1, -1, -1]], dtype=torch.int32, device=device
    )
    num_draft = torch.tensor([3, 1], dtype=torch.int32, device=device)

    out = compute_num_rejected(next_tokens, num_draft)

    assert out.tolist() == [0, 1]


# Spec-decode logprobs. A verify chunk of 3 real requests padded to 4, K=3:
# drafts [3, 1, 0], so request 0 owns logits rows 0-3, request 1 rows 4-5 and
# request 2 row 6; row 7 is padding. Request 0 accepted 1 draft + bonus,
# request 1 both slots, request 2 its bonus. The padded request carries junk.
_VERIFY_NEXT_TOKENS = [
    [11, 12, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [21, 22, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [31, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [99, 98, 97, 96],
]
_VERIFY_DRAFT_LENGTHS = [3, 1, 0, 0]
_VERIFY_BONUS_INDICES = [3, 5, 6, 0]


def test_spec_logprob_row_tokens_places_each_slot_on_its_row():
    row_tokens = _spec_logprob_row_tokens(
        torch.tensor(_VERIFY_NEXT_TOKENS, dtype=torch.int32),
        torch.tensor(_VERIFY_BONUS_INDICES, dtype=torch.int32),
        torch.tensor(_VERIFY_DRAFT_LENGTHS, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
    )
    assert row_tokens.shape == (8, 1)
    assert row_tokens.dtype == torch.int64
    # Rejected slots, the padded request and the padding row read token 0;
    # in particular the padded request's junk does not overwrite row 0.
    assert row_tokens.flatten().tolist() == [11, 12, 0, 0, 21, 22, 31, 0]


def _row_logprobs(num_rows, base=0):
    """Rows whose values encode their own index, max_num_logprobs=1."""
    rows = torch.arange(num_rows, dtype=torch.int32) + base
    return LogprobsTensors(
        logprob_token_ids=torch.stack([rows, rows + 1000], dim=1),
        logprobs=torch.stack([-rows.float(), -rows.float() - 0.5], dim=1),
        selected_token_ranks=rows.clone(),
    )


def test_spec_logprobs_lists_keeps_accepted_rows_in_request_order():
    verify = SpecLogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))
    no_drafts = SpecLogprobsChunk(_row_logprobs(4, base=100), np.arange(2))
    out = spec_logprobs_lists([verify, no_drafts], gen_lens=[2, 2, 1, 1, 1])
    assert out.cu_num_generated_tokens == [0, 2, 4, 5, 6, 7]
    assert out.logprob_token_ids[:, 0].tolist() == [0, 1, 4, 5, 6, 100, 101]
    assert out.sampled_token_ranks.tolist() == [0, 1, 4, 5, 6, 100, 101]
    # The scheduler takes each request's rows with slice_request.
    assert out.slice_request(1, 2).logprob_token_ids[:, 0].tolist() == [4, 5]
    assert out.slice_request(3, 1).logprob_token_ids[:, 0].tolist() == [100]


def test_get_output_attaches_spec_logprobs():
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor(_VERIFY_NEXT_TOKENS, dtype=torch.int32)],
        chunk_real_lens=[3],
        copy_ready_event=None,
        completed=True,
        spec_logprobs=[
            SpecLogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))
        ],
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b", "c"],
            req_id_to_index={"a": 0, "b": 1, "c": 2},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        # A discarded request keeps its rows in the layout, like vLLM's
        # rejection sampler; the scheduler never reads them.
        discard_sampled_tokens_req_indices=[1],
    ).get_output()
    assert out.sampled_token_ids == [[11, 12], [], [31]]
    assert out.logprobs.cu_num_generated_tokens == [0, 2, 4, 5]
    assert out.logprobs.slice_request(0, 2).logprob_token_ids[:, 0].tolist() == [
        0,
        1,
    ]
    assert out.logprobs.slice_request(2, 1).logprobs[:, 0].tolist() == [-6.0]


def test_get_output_skips_wait_when_all_requests_discarded():
    class _ForbiddenWaitEvent:
        def wait(self):
            raise AssertionError("copy_ready_event.wait() should not be called")

    copy_state = AsyncTPUCopyState(
        chunks_cpu=[],
        chunk_real_lens=[],
        copy_ready_event=_ForbiddenWaitEvent(),
        completed=False,
        spec_logprobs=None,
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["prefill-req-0"],
            req_id_to_index={"prefill-req-0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[0],
    ).get_output()
    assert out.sampled_token_ids == [[]]
    assert out.logprobs is None


class _FakeTPUTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _as_fake_tpu(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_FakeTPUTensor)


def test_parse_sampled_token_ids_cpu_shapes():
    rows_1d, lens_1d = parse_sampled_token_ids_cpu(
        torch.tensor([11, 22], dtype=torch.int32)
    )
    assert rows_1d == [[11], [22]]
    assert lens_1d == [1, 1]

    rows_col1, lens_col1 = parse_sampled_token_ids_cpu(
        torch.tensor([[33], [44]], dtype=torch.int32)
    )
    assert rows_col1 == [[33], [44]]
    assert lens_col1 == [1, 1]

    rows_spec, lens_spec = parse_sampled_token_ids_cpu(
        torch.tensor([[55, INVALID_TOKEN_ID], [66, 77]], dtype=torch.int32)
    )
    assert rows_spec == [[55], [66, 77]]
    assert lens_spec == [1, 2]


def test_spec_logprobs_lists_empty_chunk():
    empty_chunk = SpecLogprobsChunk(
        _row_logprobs(2), row_starts=np.zeros(0, dtype=np.int64)
    )
    out = spec_logprobs_lists([empty_chunk], gen_lens=[])
    assert out.cu_num_generated_tokens == [0]
    assert out.logprob_token_ids.shape[0] == 0


def test_async_tpu_copy_state_lifecycle_and_caching():
    t0 = torch.tensor([[10, 11, INVALID_TOKEN_ID], [99, 99, 99]], dtype=torch.int32)
    t1 = torch.tensor([[20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]], dtype=torch.int32)
    lp_tensors = _row_logprobs(4)
    spec_chunk = SpecLogprobsChunk(lp_tensors, row_starts=np.array([0, 2]))

    fake_event = MagicMock()
    with patch.object(
        torch,
        "tpu",
        SimpleNamespace(Event=MagicMock(return_value=fake_event)),
        create=True,
    ):
        state_no_lp = AsyncTPUCopyState.from_device_chunks(
            [t0], [1], spec_logprobs=None
        )
        assert state_no_lp.spec_logprobs is None
        fake_event.record.assert_called_once()

        state = AsyncTPUCopyState.from_device_chunks(
            [t0, t1], [1, 1], spec_logprobs=[spec_chunk]
        )
        assert state.spec_logprobs is not None
        assert not state.completed

    state.wait()
    assert state.completed
    assert state.chunks_tpu is None
    assert state._spec_logprobs_tpu is None
    fake_event.synchronize.assert_called_once()

    # Second wait() is a no-op.
    state.wait()
    fake_event.synchronize.assert_called_once()

    # Multi-chunk concatenation + caching on sampled_token_ids_cpu.
    cpu_ids_first = state.sampled_token_ids_cpu
    cpu_ids_second = state.sampled_token_ids_cpu
    assert cpu_ids_first is cpu_ids_second
    assert cpu_ids_first.tolist() == [
        [10, 11, INVALID_TOKEN_ID],
        [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    ]

    # parse_sampled_tokens caches the parsed result and returns defensive row copies.
    rows_a, lens_a = state.parse_sampled_tokens()
    assert rows_a == [[10, 11], [20]]
    assert lens_a == [2, 1]
    rows_a[0].clear()

    rows_b, lens_b = state.parse_sampled_tokens()
    assert rows_b == [[10, 11], [20]]
    assert lens_b is lens_a


def test_async_pre_results_wait_for_copy():
    pre_none = AsyncPreResults(
        req_ids=[],
        next_tokens_tpu=torch.empty(0, dtype=torch.int32),
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={},
        copy_state=None,
    )
    assert pre_none.wait_for_copy() is None

    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[7]], dtype=torch.int32)],
        chunk_real_lens=[1],
        copy_ready_event=None,
        completed=True,
    )
    pre_with_copy = AsyncPreResults(
        req_ids=["r0"],
        next_tokens_tpu=torch.tensor([7], dtype=torch.int32),
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"r0": 0},
        copy_state=copy_state,
    )
    out = pre_with_copy.wait_for_copy()
    assert out is not None and out.tolist() == [[7]]


def test_async_tpu_model_runner_output_edge_branches():
    # 1) copy_state is None, all requests discarded -> [[] for _ in range(num_reqs)]
    out_discard_all = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=None,
        discard_sampled_tokens_req_indices=[0, 1],
    ).get_output()
    assert out_discard_all.sampled_token_ids == [[], []]

    # 2) copy_state is None, not all requests discarded -> []
    out_none = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=None,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out_none.sampled_token_ids == []

    # 3) copy_state has 0 elements -> []
    empty_state = AsyncTPUCopyState(
        chunks_cpu=[torch.empty((0, 1), dtype=torch.int32)],
        chunk_real_lens=[0],
        copy_ready_event=None,
        completed=True,
    )
    out_empty = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=empty_state,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out_empty.sampled_token_ids == []

    # 4) copy_state without spec_logprobs, partial discard
    normal_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[11], [22]], dtype=torch.int32)],
        chunk_real_lens=[2],
        copy_ready_event=None,
        completed=True,
        spec_logprobs=None,
    )
    out_partial = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=normal_state,
        discard_sampled_tokens_req_indices=[0],
    ).get_output()
    assert out_partial.sampled_token_ids == [[], [22]]

    # 5) copy_state without spec_logprobs, no discard
    out_no_discard = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=normal_state,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out_no_discard.sampled_token_ids == [[11], [22]]


def test_compiled_helpers_padding_mrope_and_tpu_dispatch(monkeypatch):
    next_tokens = torch.tensor(
        [[11, 99, INVALID_TOKEN_ID], [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID]],
        dtype=torch.int32,
    )
    drafts = torch.tensor([[201, 202], [211, 212]], dtype=torch.int32)
    num_draft = torch.tensor([2, 2], dtype=torch.int32)

    # Padding branches (padded_src_len > src.shape[0] and padded_num_reqs > num_reqs)
    src_pad, rej_pad = assemble_spec_next_tokens_and_num_rejected(
        next_tokens,
        drafts,
        num_draft,
        num_reqs=2,
        padded_src_len=12,
        padded_num_reqs=4,
    )
    assert src_pad.tolist() == [99, 201, 202, 77, 211, 212, 0, 0, 0, 0, 0, 0]
    assert rej_pad.tolist() == [1, 2, 0, 0]

    # Exact-length branches (padded_src_len == src.shape[0] and
    # padded_num_reqs == num_reqs)
    src_exact, rej_exact = assemble_spec_next_tokens_and_num_rejected(
        next_tokens,
        drafts,
        num_draft,
        num_reqs=2,
        padded_src_len=6,
        padded_num_reqs=2,
    )
    assert src_exact.tolist() == [99, 201, 202, 77, 211, 212]
    assert rej_exact.tolist() == [1, 2]

    # num_draft=None with padded_num_reqs > num_reqs
    _, rej_none_pad = assemble_spec_next_tokens_and_num_rejected(
        next_tokens,
        drafts,
        None,
        num_reqs=2,
        padded_src_len=6,
        padded_num_reqs=4,
    )
    assert rej_none_pad.tolist() == [0, 0, 0, 0]

    # TPU dispatch branch for assemble_spec_next_tokens_and_num_rejected
    monkeypatch.setattr(
        tpu_runner_async_output,
        "_assemble_spec_next_tokens_and_num_rejected_compiled",
        tpu_runner_async_output._assemble_spec_next_tokens_and_num_rejected,
    )
    src_tpu, rej_tpu = assemble_spec_next_tokens_and_num_rejected(
        _as_fake_tpu(next_tokens),
        _as_fake_tpu(drafts),
        num_draft,
        num_reqs=2,
    )
    assert torch.equal(src_tpu, src_exact)
    assert torch.equal(rej_tpu, rej_exact)

    # Non-bucketed shape on TPU falls back to uncompiled helper
    src_tpu_unbucketed, _ = assemble_spec_next_tokens_and_num_rejected(
        _as_fake_tpu(next_tokens),
        _as_fake_tpu(drafts),
        num_draft,
        num_reqs=1,
    )
    assert src_tpu_unbucketed.tolist() == [99, 201, 202]

    # 2D mrope positions in subtract_num_rejected_tokens + TPU dispatch
    monkeypatch.setattr(
        tpu_runner_async_output,
        "_subtract_num_rejected_tokens_compiled",
        tpu_runner_async_output._subtract_num_rejected_tokens,
    )
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    positions_2d = torch.tensor([[6, 7], [6, 7], [6, 7]], dtype=torch.int32)
    num_rejected = torch.tensor([2, 1], dtype=torch.int32)
    seq_idx = torch.tensor([0, 1], dtype=torch.int32)
    pos_idx = torch.tensor([0, 1], dtype=torch.int32)
    new_seq, new_pos_2d = subtract_num_rejected_tokens(
        _as_fake_tpu(seq_lens),
        positions_2d,
        num_rejected,
        seq_idx,
        pos_idx,
    )
    assert new_seq.tolist() == [8, 19]
    assert new_pos_2d.tolist() == [[4, 6], [4, 6], [4, 6]]

    # TPU dispatch for extract_draft_token_ids
    monkeypatch.setattr(
        tpu_runner_async_output,
        "_extract_draft_token_ids_compiled",
        tpu_runner_async_output._extract_draft_token_ids,
    )
    input_ids = _as_fake_tpu(torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    logits_idx = torch.tensor([0, 1, 2], dtype=torch.int32)
    target_idx = torch.tensor([0, 1], dtype=torch.int32)
    assert extract_draft_token_ids(input_ids, logits_idx, target_idx).tolist() == [
        11,
        12,
    ]


def test_td8276_native_int32_1d_index_select_in_async_output_helpers(monkeypatch):
    select_calls = []
    orig_index_select = torch.Tensor.index_select

    def spy_index_select(self, dim, index):
        select_calls.append((self.ndim, dim, index.dtype))
        return orig_index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", spy_index_select)
    next_tokens = torch.tensor(
        [[11, 99, INVALID_TOKEN_ID], [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID]],
        dtype=torch.int32,
    )
    drafts = torch.tensor([[201, 202], [211, 212]], dtype=torch.int32)
    num_draft = torch.tensor([2, 2], dtype=torch.int32)

    select_calls.clear()
    out1 = tpu_runner_async_output.assemble_spec_next_tokens(
        next_tokens, drafts, num_reqs=2
    )
    assert out1.tolist() == [99, 201, 202, 77, 211, 212]
    assert select_calls == [(1, 0, torch.int32)]

    select_calls.clear()
    out2, rej2 = assemble_spec_next_tokens_and_num_rejected(
        next_tokens, drafts, num_draft, num_reqs=2
    )
    assert torch.equal(out2, out1)
    assert rej2.tolist() == [1, 2]
    assert select_calls == [(1, 0, torch.int32)]

    select_calls.clear()
    seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
    positions = torch.tensor([5, 6, 15, 16, 25], dtype=torch.int32)
    num_rej = torch.tensor([2, 1], dtype=torch.int64)
    seq_idx = torch.tensor([0, 1, -1], dtype=torch.int64)
    pos_idx = torch.tensor([0, 0, 1, 1, -1], dtype=torch.int64)
    new_seq, new_pos = subtract_num_rejected_tokens(
        seq_lens, positions, num_rej, seq_idx, pos_idx
    )
    assert new_seq.tolist() == [8, 19, 30]
    assert new_pos.tolist() == [3, 4, 14, 15, 25]
    assert select_calls == [(1, 0, torch.int32), (1, 0, torch.int32)]

    select_calls.clear()
    in_ids = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)
    logits_idx = torch.tensor([0, 2, 4], dtype=torch.int64)
    target_idx = torch.tensor([0, 1], dtype=torch.int64)
    extracted = extract_draft_token_ids(in_ids, logits_idx, target_idx)
    assert extracted.tolist() == [102, 104]
    assert select_calls == [(1, 0, torch.int32), (1, 0, torch.int32)]
