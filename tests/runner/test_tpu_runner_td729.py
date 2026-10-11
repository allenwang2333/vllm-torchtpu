# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Track TD7.29 TPUModelRunner steady-state host fast paths."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    INVALID_TOKEN_ID,
    TPUModelRunner,
    _cached_identity_map,
    _cached_uniform_draft_map,
    _uniform_placeholder_count,
)
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


def test_cached_maps_and_uniform_placeholder_count_all_branches():
    # _cached_identity_map and _cached_uniform_draft_map return cached dicts
    id_map = _cached_identity_map(3)
    assert id_map == {0: 0, 1: 1, 2: 2}
    assert _cached_identity_map(3) is id_map

    draft_map = _cached_uniform_draft_map(3, 2)
    assert draft_map == {0: 2, 1: 2, 2: 2}
    assert _cached_uniform_draft_map(3, 2) is draft_map

    # Empty / None -> 1
    assert _uniform_placeholder_count(None, 3) == 1
    assert _uniform_placeholder_count({}, 3) == 1

    # Identity hit on _cached_uniform_draft_map -> 1 + k0
    assert _uniform_placeholder_count(draft_map, 3) == 3

    # Plain dict with uniform values across 0..n_reqs-1 -> 1 + k0
    assert _uniform_placeholder_count({0: 4, 1: 4, 2: 4}, 3) == 5

    # Heterogeneous values -> None
    assert _uniform_placeholder_count({0: 4, 1: 2, 2: 4}, 3) is None

    # Length == n_reqs and 0 in dict, but (n_reqs - 1) missing -> None
    assert _uniform_placeholder_count({0: 4, 5: 4}, 2) is None

    # Length != n_reqs -> None
    assert _uniform_placeholder_count({0: 4}, 2) is None

    # 0 not in dict -> None
    assert _uniform_placeholder_count({1: 4, 2: 4}, 2) is None


def test_update_states_empty_and_nonempty_invalidation_branches():
    runner = object.__new__(TPUModelRunner)
    runner._mamba_state_pos = {"req-0": 5, "req-1": 7}
    runner._mamba_offset_seeded = {"req-0", "req-1"}
    runner._pre_async_results = None
    runner.requests = {
        "req-0": SimpleNamespace(prev_num_draft_len=2),
        "req-1": SimpleNamespace(prev_num_draft_len=3),
    }
    runner._install_spec_token_room_guard = lambda: None
    runner.is_unified_pool_used = lambda: False
    runner._dense_kv_copy_supported = lambda: False

    sched_empty = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        kv_cache_block_copies=None,
    )
    with patch.object(tpu_runner.GPUModelRunner, "_update_states", return_value="ok"):
        res = runner._update_states(sched_empty)
    assert res == "ok"
    assert runner._mamba_state_pos == {"req-0": 5, "req-1": 7}
    assert runner._mamba_offset_seeded == {"req-0", "req-1"}
    assert runner.requests["req-0"].prev_num_draft_len == 0

    # Non-empty invalidation with _pre_async_results is None
    sched_nonempty_no_pre = SimpleNamespace(
        finished_req_ids={"req-0"},
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        kv_cache_block_copies=None,
    )
    with patch.object(tpu_runner.GPUModelRunner, "_update_states", return_value="ok2"):
        runner._update_states(sched_nonempty_no_pre)
    assert "req-0" not in runner._mamba_state_pos
    assert "req-0" not in runner._mamba_offset_seeded

    # Non-empty invalidation with _pre_async_results is not None
    runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"req-1": 0, "req-2": 1},
        request_seq_lens=[
            (0, SimpleNamespace(), 10, "req-1"),
            (1, SimpleNamespace(), 12, "req-2"),
        ],
    )
    sched_nonempty_with_pre = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req-1"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        kv_cache_block_copies=None,
    )
    with patch.object(tpu_runner.GPUModelRunner, "_update_states", return_value="ok3"):
        runner._update_states(sched_nonempty_with_pre)
    assert runner._pre_async_results.req_id_to_index_copy == {"req-2": 1}
    assert len(runner._pre_async_results.request_seq_lens) == 1
    assert runner._pre_async_results.request_seq_lens[0][3] == "req-2"


def test_modify_prev_results_vectorized_fast_paths():
    # 1) Uniform draft count + batch_req_ids fast path + runner.arange_np reuse
    req0 = SimpleNamespace(req_id="req-0", output_token_ids=[1, 0, 0])
    req1 = SimpleNamespace(req_id="req-1", output_token_ids=[2, 0, 0])
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[
            torch.tensor(
                [[10, 11], [20, INVALID_TOKEN_ID]],
                dtype=torch.int32,
            )
        ],
        chunk_real_lens=[2],
        copy_ready_event=None,
        completed=True,
    )
    runner = SimpleNamespace(
        arange_np=np.arange(16, dtype=np.intp),
        input_batch=SimpleNamespace(
            req_ids=["req-0", "req-1"],
            req_id_to_index={"req-0": 0, "req-1": 1},
            num_tokens_no_spec=np.array([10, 20], dtype=np.int32),
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
        ),
        _pre_async_results=SimpleNamespace(
            req_ids=["req-0", "req-1"],
            request_seq_lens=[
                (0, req0, 10, "req-0"),
                (1, req1, 20, "req-1"),
            ],
            discard_sampled_tokens_req_indices=[],
            num_draft_per_req=_cached_uniform_draft_map(2, 1),
            copy_state=copy_state,
        ),
    )
    TPUModelRunner._modify_prev_results(runner)
    assert req0.output_token_ids == [1, 10, 11]
    assert req1.output_token_ids == [2, 20]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [10, 19]
    assert runner.input_batch.token_ids_cpu[0, 8:10].tolist() == [10, 11]
    assert runner.input_batch.token_ids_cpu[1, 18] == 20

    # 2) Heterogeneous draft count + batch_req_ids is None fallback + arange_np is None
    req0_h = SimpleNamespace(req_id="req-0", output_token_ids=[1, 0, 0, 0])
    req1_h = SimpleNamespace(req_id="req-1", output_token_ids=[2, 0, 0])
    copy_state_h = AsyncTPUCopyState(
        chunks_cpu=[
            torch.tensor(
                [[30, 31, INVALID_TOKEN_ID], [40, 41, INVALID_TOKEN_ID]],
                dtype=torch.int32,
            )
        ],
        chunk_real_lens=[2],
        copy_ready_event=None,
        completed=True,
    )
    runner_no_arange = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req-0": 0, "req-1": 1},
            num_tokens_no_spec=np.array([15, 25], dtype=np.int32),
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
        ),
        _pre_async_results=SimpleNamespace(
            req_ids=["req-0", "req-1"],
            request_seq_lens=[
                (0, req0_h, 15, "req-0"),
                (1, req1_h, 25, "req-1"),
            ],
            discard_sampled_tokens_req_indices=[],
            num_draft_per_req={0: 2, 1: 1},
            copy_state=copy_state_h,
        ),
    )
    TPUModelRunner._modify_prev_results(runner_no_arange)
    assert req0_h.output_token_ids == [1, 30, 31]
    assert req1_h.output_token_ids == [2, 40, 41]
    assert runner_no_arange.input_batch.num_tokens_no_spec.tolist() == [14, 25]
    assert runner_no_arange.input_batch.token_ids_cpu[0, 12:14].tolist() == [30, 31]
    assert runner_no_arange.input_batch.token_ids_cpu[1, 23:25].tolist() == [40, 41]


def test_update_placeholder_uniform_and_heterogeneous_fast_paths():
    req0 = SimpleNamespace(req_id="req-0", output_token_ids=[5])
    req1 = SimpleNamespace(req_id="req-1", output_token_ids=[6])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req-0": 0, "req-1": 1},
            num_tokens_no_spec=np.array([10, 20], dtype=np.int32),
        )
    )
    seq_lens = [(0, req0, 10, "req-0"), (1, req1, 20, "req-1")]

    # 1) Uniform draft map + cached identity map -> req_id_to_index.copy() fast path
    res_fast = TPUModelRunner._update_placeholder(
        runner,
        [],
        seq_lens,
        _cached_identity_map(2),
        _cached_uniform_draft_map(2, 2),
    )
    assert res_fast == {"req-0": 0, "req-1": 1}
    assert res_fast is not runner.input_batch.req_id_to_index
    assert runner.input_batch.num_tokens_no_spec.tolist() == [13, 23]
    assert req0.output_token_ids == [5, 0, 0, 0]
    assert req1.output_token_ids == [6, 0, 0, 0]

    # 2) Uniform draft map + non-cached custom next_token_indices dict
    custom_indices = {0: 10, 1: 20}
    res_custom = TPUModelRunner._update_placeholder(
        runner,
        [],
        seq_lens,
        custom_indices,
        None,
    )
    assert res_custom == {"req-0": 10, "req-1": 20}
    assert runner.input_batch.num_tokens_no_spec.tolist() == [14, 24]
    assert req0.output_token_ids == [5, 0, 0, 0, 0]
    assert req1.output_token_ids == [6, 0, 0, 0, 0]

    # 3) Heterogeneous num_draft_per_req (matching first_n_new and differing n_new)
    req2 = SimpleNamespace(req_id="req-2", output_token_ids=[7])
    runner.input_batch.req_id_to_index = {"req-0": 0, "req-1": 1, "req-2": 2}
    runner.input_batch.num_tokens_no_spec = np.array([10, 20, 30], dtype=np.int32)
    seq_lens_3 = [
        (0, req0, 10, "req-0"),
        (1, req1, 20, "req-1"),
        (2, req2, 30, "req-2"),
    ]
    res_hetero = TPUModelRunner._update_placeholder(
        runner,
        [],
        seq_lens_3,
        {0: 0, 1: 1, 2: 2},
        {0: 2, 1: 2, 2: 1},
    )
    assert res_hetero == {"req-0": 0, "req-1": 1, "req-2": 2}
    assert runner.input_batch.num_tokens_no_spec.tolist() == [13, 23, 32]


def test_assemble_async_spec_substitution_uniform_heterogeneous_and_multichunk():
    runner = object.__new__(TPUModelRunner)
    runner.max_num_reqs = 4
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)

    drafts = torch.tensor([[201, 202], [211, 212]], dtype=torch.int32)
    nt_chunk0 = torch.tensor([[11, 99, -1], [77, -1, -1]], dtype=torch.int32)

    # 1) Single chunk with uniform draft lengths -> returns _cached_uniform_draft_map
    md_uniform = SimpleNamespace(
        draft_lengths=torch.tensor([2, 2], dtype=torch.int32),
        draft_lengths_cpu=np.array([2, 2], dtype=np.int32),
    )
    state_uniform = SimpleNamespace(
        num_reqs_list=[2],
        spec_decode_metadata_list=[md_uniform],
    )
    _, idx_u, _, draft_u = runner._assemble_async_spec_substitution(
        drafts, [nt_chunk0], state_uniform
    )
    assert idx_u is _cached_identity_map(2)
    assert draft_u is _cached_uniform_draft_map(2, 2)

    # 2) Single chunk with heterogeneous draft lengths -> populates dict via update
    md_hetero = SimpleNamespace(
        draft_lengths=torch.tensor([2, 1], dtype=torch.int32),
        draft_lengths_cpu=np.array([2, 1], dtype=np.int32),
    )
    state_hetero = SimpleNamespace(
        num_reqs_list=[2],
        spec_decode_metadata_list=[md_hetero],
    )
    _, idx_h, _, draft_h = runner._assemble_async_spec_substitution(
        drafts, [nt_chunk0], state_hetero
    )
    assert idx_h is _cached_identity_map(2)
    assert draft_h == {0: 2, 1: 1}

    # 2b) Single chunk with md is None -> skips draft update
    # (exercises md is None branch)
    state_none = SimpleNamespace(
        num_reqs_list=[2],
        spec_decode_metadata_list=[None],
    )
    _, idx_none, rej_none, draft_none = runner._assemble_async_spec_substitution(
        drafts, [nt_chunk0], state_none
    )
    assert idx_none is _cached_identity_map(2)
    assert draft_none == {}
    assert rej_none is not None

    # 2c) Empty chunks -> spec_num_rejected is None
    state_empty = SimpleNamespace(
        num_reqs_list=[],
        spec_decode_metadata_list=[],
    )
    chunks_empty, idx_empty, rej_empty, draft_empty = (
        runner._assemble_async_spec_substitution(drafts, [], state_empty)
    )
    assert chunks_empty == []
    assert idx_empty is _cached_identity_map(0)
    assert rej_empty is None
    assert draft_empty == {}

    # 3) Multi-chunk -> populates dict via update and cats num_rejected_chunks
    md_c0 = SimpleNamespace(
        draft_lengths=torch.tensor([2], dtype=torch.int32),
        draft_lengths_cpu=np.array([2], dtype=np.int32),
    )
    md_c1 = SimpleNamespace(
        draft_lengths=torch.tensor([2], dtype=torch.int32),
        draft_lengths_cpu=np.array([2], dtype=np.int32),
    )
    state_multi = SimpleNamespace(
        num_reqs_list=[1, 1],
        spec_decode_metadata_list=[md_c0, md_c1],
    )
    chunks_m, idx_m, rej_m, draft_m = runner._assemble_async_spec_substitution(
        drafts,
        [nt_chunk0[:1], nt_chunk0[1:]],
        state_multi,
    )
    assert len(chunks_m) == 2
    assert idx_m is _cached_identity_map(2)
    assert rej_m is not None and rej_m.tolist() == [1, 2]
    assert draft_m == {0: 2, 1: 2}

    # 4) _assemble_async_prefill_bootstrap uses cached identity & uniform maps
    boot_chunks, boot_idx, boot_rej, boot_draft = (
        runner._assemble_async_prefill_bootstrap(
            drafts,
            [torch.tensor([[99], [77]], dtype=torch.int32)],
            [2],
        )
    )
    assert len(boot_chunks) == 1
    assert boot_idx is _cached_identity_map(2)
    assert boot_draft is _cached_uniform_draft_map(2, 0)
    assert boot_rej is not None and boot_rej.tolist() == [0, 0, 0, 0]

    _, boot_idx_e, boot_rej_e, boot_draft_e = runner._assemble_async_prefill_bootstrap(
        drafts, [], []
    )
    assert boot_idx_e is _cached_identity_map(0)
    assert boot_draft_e is _cached_uniform_draft_map(0, 0)
    assert boot_rej_e is None


def test_td8278_substitute_placeholder_and_sample_spec_verify_chunk_int32():
    import contextlib
    import inspect
    from unittest.mock import MagicMock

    sub_fn = inspect.unwrap(tpu_runner._substitute_placeholder_token)
    select_dtypes: list[torch.dtype] = []
    orig_index_select = torch.Tensor.index_select

    def _spy_index_select(self, dim, index):
        select_dtypes.append(index.dtype)
        return orig_index_select(self, dim, index)

    for idx_dtype in (torch.int32, torch.int64):
        input_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        ref_ids = input_ids.clone()
        cur_idx = torch.tensor([1, 3, 3, 3], dtype=idx_dtype)
        pre_idx = torch.tensor([2, 0, -1, -1], dtype=idx_dtype)
        next_tokens = torch.tensor([101, 102, 103, 104], dtype=torch.int64)

        mask = pre_idx > -1
        safe = torch.clamp_min(pre_idx, 0).long()
        ref_new = next_tokens[safe].to(ref_ids.dtype)
        ref_orig = ref_ids[cur_idx.long()]
        ref_ids.scatter_(0, cur_idx.long(), torch.where(mask, ref_new, ref_orig))

        select_dtypes.clear()
        with patch.object(torch.Tensor, "index_select", _spy_index_select):
            out = sub_fn(input_ids, cur_idx.long(), pre_idx, next_tokens)
        assert select_dtypes == [torch.int32, torch.int32]
        assert torch.equal(out, ref_ids)

    # _sample_spec_verify_chunk non-greedy path: int32 safe_seg index_select
    runner = object.__new__(TPUModelRunner)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    runner.drafter = object.__new__(tpu_runner.Eagle3Proposer)
    runner.spec_gather_bonus_and_target_logits = lambda logits, b_idx, t_idx: (
        logits[b_idx.long()],
        logits[t_idx.long()],
    )
    temps = torch.tensor([[0.7], [1.2]], dtype=torch.float32)
    top_k = torch.tensor([[10], [20]], dtype=torch.int32)
    top_p = torch.tensor([[0.9], [0.8]], dtype=torch.float32)
    runner._build_padded_sampling_params = lambda s, e, b: (temps, top_k, top_p)
    runner.sample_from_logits_func = lambda b, t, u, tk, tp, all_greedy=False: (
        torch.tensor([[7], [8]], dtype=torch.int32)
    )
    captured_rs_kwargs: list[dict] = []

    def _fake_rs(**kwargs):
        captured_rs_kwargs.append(kwargs)
        return torch.tensor([[1, 2, 7], [3, -1, -1]], dtype=torch.int32)

    runner.rejection_sampler = MagicMock(side_effect=_fake_rs)
    runner.rejection_sampler.synthetic_mode = False

    logits = torch.randn((5, 16), dtype=torch.float32)
    so = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [1, 2], "r1": [3]},
        num_scheduled_tokens={"r0": 3, "r1": 2},
    )
    for seg_dtype in (torch.int32, torch.int64):
        md = SimpleNamespace(
            bonus_logits_indices=torch.tensor([2, 4], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 1, 3, 3], dtype=torch.int32),
            draft_token_ids=torch.tensor([1, 2, 3, 0], dtype=torch.int32),
            draft_lengths=torch.tensor([2, 1], dtype=torch.int32),
            draft_lengths_cpu=np.array([2, 1], dtype=np.int32),
            # Include out-of-range sentinel 2 (= num_reqs) to test clamp
            segment_ids=torch.tensor([0, 0, 1, 2], dtype=seg_dtype),
            group_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )
        select_dtypes.clear()
        with patch.object(torch.Tensor, "index_select", _spy_index_select):
            res = runner._sample_spec_verify_chunk(
                logits, md, None, so, 0, 2, False, torch.Generator()
            )
        assert res.shape == (2, 3)
        assert select_dtypes == [torch.int32, torch.int32, torch.int32]
        kw = captured_rs_kwargs[-1]
        expected_seg = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        assert torch.equal(kw["temperatures"], temps[expected_seg])
        assert torch.equal(kw["top_k"], top_k[expected_seg])
        assert torch.equal(kw["top_p"], top_p[expected_seg])

    # _precompile_rejection_sampler passes segment_ids with dtype=torch.int32
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=2, use_eagle=lambda: True, method="eagle3"
    )
    runner._model_supports_top_tokens = lambda: False
    runner.num_tokens_paddings = [4]
    runner.num_reqs_paddings = [1]
    runner.max_num_reqs = 1
    runner.vocab_size = 16
    runner._hidden_states_dtype = torch.bfloat16
    runner.device = torch.device("cpu")
    runner._precompile_timed = lambda name: contextlib.nullcontext()
    runner._spec_verify_bucket_reachable = lambda nt, nr: True
    runner.spec_bonus_and_target_logits = lambda logits_in, b, t: (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((4, 16), dtype=torch.bfloat16),
    )
    captured_rs_kwargs.clear()
    with patch.object(tpu_runner, "synchronize_tensors"):
        runner._precompile_rejection_sampler()
    assert len(captured_rs_kwargs) == 2
    assert all(kw["segment_ids"].dtype == torch.int32 for kw in captured_rs_kwargs)


def test_td8299_substitute_placeholder_token_and_precompile_spec_int32():
    import contextlib
    import inspect

    sub_fn = inspect.unwrap(tpu_runner._substitute_placeholder_token)
    for next_dtype in (torch.int32, torch.int64):
        input_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        cur_idx = torch.tensor([1, 2, 3, 3], dtype=torch.int64)
        pre_idx = torch.tensor([2, 0, -1, -1], dtype=torch.int32)
        next_tokens = torch.tensor([101, 102, 103, 104], dtype=next_dtype)
        out = sub_fn(input_ids, cur_idx, pre_idx, next_tokens)
        assert out.dtype == torch.int32
        assert out.tolist() == [10, 103, 101, 40]

    runner = object.__new__(TPUModelRunner)
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=2, use_eagle=lambda: True, method="eagle3"
    )
    runner.drafter = object.__new__(tpu_runner.Eagle3Proposer)
    runner.num_tokens_paddings = [8]
    runner.num_reqs_paddings = [2, 4]
    runner.max_num_reqs = 4
    runner.device = torch.device("cpu")
    runner._precompile_timed = lambda name: contextlib.nullcontext()

    seen_calls: list[tuple[int, int, torch.dtype]] = []

    def _record_sub(inp, cur, pre, nxt):
        seen_calls.append((inp.shape[0], nxt.shape[0], nxt.dtype))
        return inp

    with (
        patch.object(tpu_runner, "_substitute_placeholder_token", _record_sub),
        patch.object(tpu_runner, "synchronize_tensors"),
    ):
        runner._precompile_substitute_placeholder_token()

    assert seen_calls == [
        (8, 2, torch.int64),
        (8, 4, torch.int64),
        (8, 12, torch.int32),
    ]
