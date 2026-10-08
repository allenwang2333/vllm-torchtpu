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

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _ValidTopTokenModel:
    def __init__(self, logits_processor, lm_head):
        self.logits_processor = logits_processor
        self.lm_head = lm_head

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor._get_logits(hidden_states, self.lm_head, None)


def _make_valid_top_token_model():
    logits_processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(logits_processor)
    logits_processor.vocab_size = 16
    logits_processor.org_vocab_size = 16
    logits_processor.scale = 1.0
    logits_processor.logits_as_input = False
    logits_processor.soft_cap = None
    logits_processor.head_dtype = None
    logits_processor.use_all_gather = True

    weight = torch.randn(16, 8, dtype=torch.float32)
    lm_head = SimpleNamespace(
        tp_size=1,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=16,
            num_org_vocab_padding=0,
        ),
        weight=weight,
        quant_method=SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.t()
        ),
    )
    return _ValidTopTokenModel(logits_processor, lm_head)


def test_spec_decode_metadata_skip_host_draft_tokens_cache_hit():
    runner = SimpleNamespace(
        max_num_tokens=32,
        input_ids_cpu=torch.arange(32, dtype=torch.int32),
        arange_np=np.arange(32, dtype=np.int32),
        num_tokens_paddings=[8, 16, 32],
        device=torch.device("cpu"),
    )
    mgr = SpeculativeDecodingManager(runner)
    num_draft = np.array([2, 2], dtype=np.int32)
    cu_tokens = np.array([3, 6], dtype=np.int32)

    # First call populates the cache and transfers host draft_token_ids.
    md1 = mgr.get_spec_decode_metadata(
        num_draft, cu_tokens, padded_num_reqs=2, skip_host_draft_tokens=False
    )
    orig_draft_tokens = md1.draft_token_ids
    assert orig_draft_tokens.shape == (8,)

    # Mutate input_ids_cpu to detect whether host draft_token_ids is re-uploaded.
    runner.input_ids_cpu.fill_(99)

    # Cache hit with skip_host_draft_tokens=True preserves the cached tensor
    # without re-uploading from host input_ids_cpu.
    md2 = mgr.get_spec_decode_metadata(
        num_draft, cu_tokens, padded_num_reqs=2, skip_host_draft_tokens=True
    )
    assert md2.draft_token_ids is orig_draft_tokens
    assert md2.final_logits_indices is md1.final_logits_indices
    assert (md2.draft_token_ids[:4] != 99).all()

    # Cache hit with skip_host_draft_tokens=False re-uploads from input_ids_cpu.
    md3 = mgr.get_spec_decode_metadata(
        num_draft, cu_tokens, padded_num_reqs=2, skip_host_draft_tokens=False
    )
    assert md3.final_logits_indices is md1.final_logits_indices
    assert (md3.draft_token_ids[:4] == 99).all()


def test_reorder_batch_for_rpa_precomputed_segments():
    swapped = []

    def swap_states(i, j):
        swapped.append((i, j))
        req_ids[i], req_ids[j] = req_ids[j], req_ids[i]

    req_ids = ["prefill-0", "spec-0", "decode-0", "spec-1", "decode-1"]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=5,
            req_ids=req_ids,
            swap_states=swap_states,
        )
    )
    sched = SimpleNamespace(
        num_scheduled_tokens={
            "prefill-0": 16,
            "spec-0": 4,
            "decode-0": 1,
            "spec-1": 4,
            "decode-1": 1,
        },
        scheduled_spec_decode_tokens={
            "spec-0": [1, 2, 3],
            "spec-1": [4, 5, 6],
        },
    )
    with patch.object(tpu_runner.mla_dispatch, "maybe_order_prefills_by_kv_len"):
        num_decode, num_windowed = TPUModelRunner._reorder_batch_for_rpa(runner, sched)

    assert num_decode == 2
    assert num_windowed == 4
    assert set(req_ids[:2]) == {"decode-0", "decode-1"}
    assert set(req_ids[2:4]) == {"spec-0", "spec-1"}
    assert req_ids[4] == "prefill-0"
    assert len(swapped) > 0

    # Empty batch early return
    empty_runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert TPUModelRunner._reorder_batch_for_rpa(empty_runner, sched) == (0, 0)

    # Single decode (i == j and segments[i] <= bound)
    single_decode_runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            req_ids=["decode-0"],
            swap_states=lambda i, j: None,
        )
    )
    with patch.object(tpu_runner.mla_dispatch, "maybe_order_prefills_by_kv_len"):
        assert TPUModelRunner._reorder_batch_for_rpa(single_decode_runner, sched) == (
            1,
            1,
        )

    # Single prefill (i == j and segments[i] > bound)
    single_prefill_runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            req_ids=["prefill-0"],
            swap_states=lambda i, j: None,
        )
    )
    with patch.object(tpu_runner.mla_dispatch, "maybe_order_prefills_by_kv_len"):
        assert TPUModelRunner._reorder_batch_for_rpa(single_prefill_runner, sched) == (
            0,
            0,
        )


def test_prepare_async_token_substitution_indices_all_layout_and_cached_prefix():
    all_plan = SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(0, 8),
        global_num_tokens=6,
        global_padded_num_tokens=8,
        local_num_tokens=6,
        local_padded_num_tokens=8,
    )
    assert all_plan.kind is SequenceLayoutKind.ALL

    # 0. _pre_async_results is None
    none_runner = SimpleNamespace(_pre_async_results=None)
    cur0, pre0 = TPUModelRunner._prepare_async_token_substitution_indices(
        none_runner, 0, 2, np.array([1, 1], dtype=np.int32)
    )
    assert cur0.tolist() == []
    assert pre0.tolist() == []

    # 1. Reuses _last_async_seq_idx_prefix when present and length matches.
    runner = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy={"r0": 1, "r1": 0},
            spec_decode_num_rejected_tokens=torch.tensor([0, 0], dtype=torch.int32),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        _last_sequence_layout_plan=all_plan,
        _last_async_seq_idx_prefix=np.array([1, 0], dtype=np.int32),
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
    )
    cur, pre = TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([3, 3], dtype=np.int32),
    )
    assert cur.tolist() == [0, 1, 2, 3, 4, 5]
    assert pre.tolist() == [3, 4, 5, 0, 1, 2]

    # 1b. Non-uniform scheduled tokens and no cached prefix
    runner._last_async_seq_idx_prefix = None
    cur_nu, pre_nu = TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([2, 3], dtype=np.int32),
    )
    assert cur_nu.tolist() == [0, 1, 2, 3, 4]
    assert pre_nu.tolist() == [3, 4, 0, 1, 2]

    # 1c. All new requests (not valid_reqs.any())
    runner.input_batch.req_ids = ["new0", "new1"]
    cur_new, pre_new = TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([3, 3], dtype=np.int32),
    )
    assert cur_new.tolist() == []
    assert pre_new.tolist() == []

    # 2. Mixed valid/invalid requests under SequenceLayoutKind.ALL falls through
    # to the per-request loop and uses request_major_index directly.
    runner.input_batch.req_ids = ["r0", "new_req"]
    cur2, pre2 = TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([3, 2], dtype=np.int32),
    )
    assert cur2.tolist() == [0, 1, 2]
    assert pre2.tolist() == [3, 4, 5]

    # 3. Non-ALL layout plan calls local_index_for_request_major_token.
    pcp_desc = SequenceLayoutDescriptor(
        kind=SequenceLayoutKind.PARTIAL,
        protocol="pcp_streaming",
        version=1,
    )
    pcp_plan = SequenceLayoutPlan(
        descriptor=pcp_desc,
        token_slice=slice(0, 2),
        global_num_tokens=3,
        global_padded_num_tokens=4,
        local_num_tokens=2,
        local_padded_num_tokens=2,
        _request_major_to_packed_token_indices=np.array([0, 3, 1], dtype=np.int32),
    )
    runner._last_sequence_layout_plan = pcp_plan
    runner.input_batch.req_ids = ["r0"]
    cur3, pre3 = TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([3], dtype=np.int32),
    )
    assert cur3.tolist() == [0, 1]
    assert pre3.tolist() == [3, 5]


def test_install_spec_token_room_guard_fast_append_row_and_may_reorder_batch():
    appended = []
    bt0 = SimpleNamespace(append_row=lambda bids, r: appended.append((0, bids, r)))
    bt1 = SimpleNamespace(append_row=lambda bids, r: appended.append((1, bids, r)))
    input_batch = SimpleNamespace(
        update_req_spec_token_ids=lambda req, spec: None,
        block_table=SimpleNamespace(
            block_tables=[bt0, bt1],
            append_row=None,
        ),
    )
    runner = SimpleNamespace(input_batch=input_batch)
    TPUModelRunner._install_spec_token_room_guard(runner)
    assert callable(input_batch.block_table.append_row)

    # Empty bids for bt1 is skipped; non-empty bids for bt0 is appended.
    input_batch.block_table.append_row(([10, 11], []), 3)
    assert appended == [(0, [10, 11], 3)]

    # Calling _install_spec_token_room_guard again is a no-op.
    TPUModelRunner._install_spec_token_room_guard(runner)

    # Also test when block_table has no block_tables list/tuple
    runner_no_bt = SimpleNamespace(
        input_batch=SimpleNamespace(
            update_req_spec_token_ids=lambda req, spec: None,
            block_table=None,
        )
    )
    TPUModelRunner._install_spec_token_room_guard(runner_no_bt)
    assert runner_no_bt.input_batch._tpu_spec_room_guard is True

    # _may_reorder_batch is a no-op.
    assert TPUModelRunner._may_reorder_batch(runner, SimpleNamespace()) is None


def test_prepare_inputs_spec_decode_metadata_caching_and_skip_host_drafts():
    max_num_reqs = 2
    max_model_len = 32
    token_ids_cpu = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)

    spec_mgr = MagicMock()
    fake_spec_md = SimpleNamespace(
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 0, 0], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 3, 4, 0, 0, 0, 0], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2, 5], dtype=torch.int32),
        draft_token_ids=torch.zeros(8, dtype=torch.int32),
    )
    spec_mgr.get_spec_decode_metadata.return_value = fake_spec_md

    runner = SimpleNamespace(
        reorder_batch_threshold=1,
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["req-0", "req-1"],
            num_computed_tokens_cpu=np.array([4, 8], dtype=np.int32),
            token_ids_cpu=token_ids_cpu,
            token_ids_cpu_tensor=torch.from_numpy(token_ids_cpu),
            num_prompt_tokens=np.full(max_num_reqs, 16, dtype=np.int32),
        ),
        _wait_input_staging_fence=lambda: None,
        most_model_len=None,
        num_reqs_max_model_len=max_num_reqs,
        num_reqs_most_model_len=max_num_reqs,
        max_num_reqs=max_num_reqs,
        uses_mrope=False,
        arange_np=np.arange(max_model_len, dtype=np.int32),
        positions_cpu=positions_cpu,
        positions_np=positions_cpu.numpy(),
        input_ids_cpu=torch.full((max_model_len,), -1, dtype=torch.int32),
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc_np=query_start_loc_cpu.numpy(),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_cpu.numpy(),
        _check_attention_schedule=lambda *a: None,
        _pcp_enabled=False,
        _mtp_enabled=False,
        device=torch.device("cpu"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        spec_decode_manager=spec_mgr,
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy={"req-0": 0, "req-1": 1},
            spec_decode_num_rejected_tokens=torch.tensor([1, 2], dtype=torch.int32),
            next_tokens_tpu=torch.zeros(6, dtype=torch.int32),
        ),
        _decode_device_cache_key=None,
        _request_distribution_cpu=torch.zeros(3, dtype=torch.int32),
        _has_mamba_state=False,
        mamba_slot_read_offsets=None,
        _mamba_ckpt_window=1,
        _unified_kv_layout=False,
        empty_slot_mappings=None,
        _build_attention_metadata=lambda **kw: ({}, None),
        lora_config=None,
        _prepare_async_token_substitution_indices=(
            lambda s, n, sched_arr: (
                TPUModelRunner._prepare_async_token_substitution_indices(
                    runner, s, n, sched_arr
                )
            )
        ),
        _collect_mamba_state_seed_copies=lambda *a: None,
        num_tokens_paddings=[8, 16, 32],
        pcp_size=1,
        dcp_size=1,
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=6,
        num_scheduled_tokens={"req-0": 3, "req-1": 3},
        scheduled_spec_decode_tokens={"req-0": [10, 11], "req-1": [20, 21]},
    )

    # First call populates _decode_device_cache_key for spec-decode verify step.
    res1 = TPUModelRunner._prepare_inputs(
        runner, sched, start_index=0, num_decode_reqs=0
    )
    assert spec_mgr.get_spec_decode_metadata.call_args.kwargs["skip_host_draft_tokens"]
    qsl1 = runner._cached_query_start_loc
    req_dist1 = runner._cached_request_distribution
    assert res1[1] is fake_spec_md.final_logits_indices

    # Second call hits _decode_device_cache_key and reuses cached device tensors.
    res2 = TPUModelRunner._prepare_inputs(
        runner, sched, start_index=0, num_decode_reqs=0
    )
    assert runner._attn_metadata_builder_ctx.query_start_loc is qsl1
    assert runner._attn_metadata_builder_ctx.request_distribution is req_dist1
    assert res2[1] is fake_spec_md.final_logits_indices

    # Test pure 1-token decode cache miss + hit (spec_decode_metadata is None)
    runner.speculative_config = None
    runner._pre_async_results = None
    runner._decode_device_cache_key = None
    sched_decode = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"req-0": 1, "req-1": 1},
        scheduled_spec_decode_tokens={},
    )
    res_dec1 = TPUModelRunner._prepare_inputs(
        runner, sched_decode, start_index=0, num_decode_reqs=2
    )
    res_dec2 = TPUModelRunner._prepare_inputs(
        runner, sched_decode, start_index=0, num_decode_reqs=2
    )
    assert res_dec2[1] is res_dec1[1]


def test_model_supports_top_tokens_and_can_use_greedy_top_tokens():
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner._dsv41_smoke_logits_hooked = False
    runner.model = _make_valid_top_token_model()
    runner.input_batch = SimpleNamespace(all_greedy=True, max_num_logprobs=None)
    runner._dp_lockstep_enabled = lambda: False

    sched = SimpleNamespace(has_structured_output_requests=False)
    assert runner._model_supports_top_tokens() is True
    assert runner._can_use_greedy_top_tokens(sched) is True

    # Test all guard conditions in _model_supports_top_tokens:
    runner.is_pooling_model = True
    assert runner._model_supports_top_tokens() is False
    runner.is_pooling_model = False

    runner._dsv41_smoke_logits_hooked = True
    assert runner._model_supports_top_tokens() is False
    runner._dsv41_smoke_logits_hooked = False

    runner.compute_selected_logits = lambda *a: None
    assert runner._model_supports_top_tokens() is False
    del runner.compute_selected_logits

    orig_model = runner.model
    runner.model = None
    assert runner._model_supports_top_tokens() is False
    runner.model = SimpleNamespace(logits_processor=None, lm_head=None)
    assert runner._model_supports_top_tokens() is False

    runner.model = _make_valid_top_token_model()
    runner.model.logits_processor.logits_as_input = True
    assert runner._model_supports_top_tokens() is False

    runner.model = _make_valid_top_token_model()
    runner.model.logits_processor.scale = -1.0
    assert runner._model_supports_top_tokens() is False

    runner.model = _make_valid_top_token_model()
    runner.model.lm_head.tp_size = MagicMock()
    assert runner._model_supports_top_tokens() is False

    # Instance attribute override on runner.model returns False:
    runner.model = _make_valid_top_token_model()
    runner.model.compute_logits = lambda h: None
    assert runner._model_supports_top_tokens() is False

    runner.model = orig_model
    assert runner._model_supports_top_tokens() is True

    # Test compute_selected_top_tokens, compute_top_tokens_from_hidden_states,
    # _greedy_sample, sample_from_logits, and spec_bonus_and_target_logits.
    hidden = torch.randn(4, 8, dtype=torch.float32)
    idx = torch.tensor([0, 2, 3], dtype=torch.int32)
    expected_full = torch.argmax(runner.model.compute_logits(hidden), dim=-1)
    top_selected = TPUModelRunner.compute_selected_top_tokens.__wrapped__(
        runner, hidden, idx
    )
    assert torch.equal(top_selected, expected_full[idx])

    top_all = TPUModelRunner.compute_top_tokens_from_hidden_states.__wrapped__(
        runner, hidden
    )
    assert torch.equal(top_all, expected_full)

    # 1D and 2D _greedy_sample
    assert torch.equal(runner._greedy_sample(top_all), top_all.unsqueeze(-1))
    logits_2d = runner.model.compute_logits(hidden)
    runner.sample_from_logits_func = lambda lg, t, u, tk, tp, all_greedy=False: (
        TPUModelRunner.sample_from_logits.__wrapped__(
            runner, lg, t, u, tk, tp, all_greedy=all_greedy
        )
    )
    assert torch.equal(runner._greedy_sample(logits_2d), top_all.unsqueeze(-1))

    # 1D and 2D sample_from_logits
    dummy = torch.empty((1, 1), dtype=torch.float32)
    sampled_1d = TPUModelRunner.sample_from_logits.__wrapped__(
        runner,
        top_all,
        dummy,
        dummy,
        torch.empty((1, 1), dtype=torch.int32),
        torch.empty((1, 1), dtype=torch.float32),
        all_greedy=True,
    )
    assert torch.equal(sampled_1d, top_all.unsqueeze(-1))

    sampled_2d = TPUModelRunner.sample_from_logits.__wrapped__(
        runner,
        logits_2d,
        dummy,
        dummy,
        torch.empty((1, 1), dtype=torch.int32),
        torch.empty((1, 1), dtype=torch.float32),
        all_greedy=True,
    )
    assert torch.equal(sampled_2d, top_all.unsqueeze(-1))

    # 1D spec_bonus_and_target_logits
    bonus_idx = torch.tensor([1, 3], dtype=torch.int32)
    target_idx = torch.tensor([0, 2], dtype=torch.int32)
    bonus_1d, target_1d = TPUModelRunner.spec_bonus_and_target_logits.__wrapped__(
        runner, top_all, bonus_idx, target_idx
    )
    bonus_2d, target_2d = TPUModelRunner.spec_bonus_and_target_logits.__wrapped__(
        runner, logits_2d, bonus_idx, target_idx
    )
    assert torch.equal(bonus_1d, bonus_2d)
    assert torch.equal(target_1d, target_2d)


def test_execute_model_routes_top_tokens_and_full_logits():
    for use_top in (True, False):
        for select_hidden in (False, True):
            calls = []
            planner = SimpleNamespace(
                maybe_select_logits_hidden_states=(
                    lambda h, plan, idx, sel=select_hidden: h if sel else None
                ),
                finalize_hidden_states=lambda h, plan: h,
            )
            runner = SimpleNamespace(
                execute_model_state=None,
                _update_states=lambda so: None,
                scheduler_config=SimpleNamespace(async_scheduling=False),
                _pcp_enabled=False,
                _mtp_enabled=False,
                _reorder_batch_for_rpa=lambda so: (1, 1),
                _count_input_chunks=lambda so: (1, 1),
                _dp_coordinated_step=lambda *a, **kw: (None, 1),
                supports_mm_inputs=False,
                _pp_is_first=True,
                _pp_is_last=True,
                _pp_wave=None,
                phase_based_profiler=None,
                _is_async_drafter=False,
                vllm_config=None,
                maybe_setup_kv_connector=lambda *a, **kw: None,
                input_batch=SimpleNamespace(num_reqs=1, req_ids=["r0"]),
                _prepare_inputs=lambda so, si, nd, nw: (
                    {},
                    torch.tensor([0], dtype=torch.int32),
                    1,
                    1,
                    1,
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    None,
                    None,
                    None,
                ),
                _flush_mamba_state_seed_copies=lambda: None,
                _reset_read_offsets_for_new_requests=lambda si, nr: None,
                input_ids=torch.zeros(4, dtype=torch.int32),
                position_ids=torch.zeros(4, dtype=torch.int32),
                _apply_async_token_substitution=lambda ids, c, p: ids,
                _record_input_staging_fence=lambda: None,
                _get_model_inputs=lambda ids, mm: (ids, None),
                _token_padding_update=lambda n: None,
                _dp_num_tokens_across_dp=lambda n: None,
                mesh=None,
                _kv_cache_bundle=None,
                _stage_timing=SimpleNamespace(
                    stage=lambda _n: contextlib.nullcontext()
                ),
                forward_model=lambda **kw: (torch.zeros((4, 8)), None),
                sequence_layout_planner=planner,
                _last_sequence_layout_plan=None,
                is_pooling_model=False,
                _can_use_greedy_top_tokens=lambda so, u=use_top: u,
                compute_selected_top_tokens=lambda h, idx, c=calls: (
                    c.append("selected_top") or torch.zeros(1, dtype=torch.int64)
                ),
                compute_selected_logits=lambda h, idx, c=calls: (
                    c.append("selected_logits") or torch.zeros((1, 16))
                ),
                compute_top_tokens_from_hidden_states=lambda h, c=calls: (
                    c.append("hidden_top") or torch.zeros(1, dtype=torch.int64)
                ),
                compute_logits_from_hidden_states=lambda h, c=calls: (
                    c.append("hidden_logits") or torch.zeros((1, 16))
                ),
                _attn_metadata_builder_ctx=SimpleNamespace(
                    mamba_state_indices=None,
                    unified_mamba_state_indices=None,
                ),
                mamba_slot_read_offsets=None,
            )
            sched = SimpleNamespace(
                total_num_scheduled_tokens=1,
                scheduled_encoder_inputs={},
            )
            with (
                patch.object(
                    tpu_runner,
                    "set_forward_context",
                    return_value=contextlib.nullcontext(),
                ),
                patch.object(
                    tpu_runner,
                    "set_vllm_model_wrapper_context",
                    return_value=contextlib.nullcontext(),
                ),
            ):
                TPUModelRunner.execute_model(runner, sched)

            if not select_hidden and use_top:
                assert calls == ["selected_top"]
            elif not select_hidden and not use_top:
                assert calls == ["selected_logits"]
            elif select_hidden and use_top:
                assert calls == ["hidden_top"]
            else:
                assert calls == ["hidden_logits"]


def test_precompile_top_tokens_subgraphs():
    sync_calls = []
    rejection_sampler = MagicMock(return_value=torch.zeros((2, 2), dtype=torch.int32))
    rejection_sampler.synthetic_mode = False
    runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 8),
        _hidden_states_dtype=torch.float32,
        num_tokens_paddings=[4],
        num_reqs_paddings=[2],
        max_num_reqs=2,
        vocab_size=16,
        device=torch.device("cpu"),
        _precompile_timed=lambda _name: contextlib.nullcontext(),
        compute_selected_logits=lambda h, idx: torch.zeros((idx.shape[0], 16)),
        compute_selected_top_tokens=lambda h, idx: torch.zeros(
            idx.shape[0], dtype=torch.int64
        ),
        compute_logits_from_hidden_states=lambda h: torch.zeros((h.shape[0], 16)),
        compute_top_tokens_from_hidden_states=lambda h: torch.zeros(
            h.shape[0], dtype=torch.int64
        ),
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        rejection_sampler=rejection_sampler,
        _spec_verify_bucket_reachable=lambda nt, nr: True,
        spec_bonus_and_target_logits=lambda lg, b, t: (
            torch.zeros(b.shape[0], dtype=torch.int64),
            torch.zeros(t.shape[0], dtype=torch.int32),
        ),
        spec_gather_bonus_and_target_logits=lambda lg, b, t: (
            torch.zeros(b.shape[0], dtype=torch.int64),
            torch.zeros((t.shape[0], 16), dtype=torch.float32),
        ),
    )

    with patch.object(
        tpu_runner, "synchronize_tensors", side_effect=lambda x: sync_calls.append(x)
    ):
        for supports in (True, False):
            runner._model_supports_top_tokens = lambda s=supports: s
            TPUModelRunner._precompile_compute_selected_logits(runner)
            TPUModelRunner._precompile_compute_logits_from_hidden_states(runner)
            TPUModelRunner._precompile_rejection_sampler(runner)

    assert len(sync_calls) > 0
