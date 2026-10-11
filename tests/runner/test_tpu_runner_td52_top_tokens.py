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


def test_td68_attention_metadata_and_runner_optimizations():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    from vllm_torchtpu.layers.core import attention_metadata as am
    from vllm_torchtpu.layers.core.attention_metadata import (
        AttentionMetadataBuilder,
        AttentionMetadataBuilderContext,
        _staged_walk_products,
        _staged_walk_products_compiled,
        stage_block_table_uploads,
    )

    # 1. _staged_walk_products with mamba_geom and ckpt_window == 1, plus
    # _staged_walk_products_compiled.__wrapped__
    staged_dev = torch.arange(16, dtype=torch.int32)
    seq_lens = torch.tensor([4, 16], dtype=torch.int32)
    table_geom = ((0, 2, 4), (8, 2, 4))
    mamba_geom = ((1, 4, 8),)
    out_w1 = _staged_walk_products(
        staged_dev, seq_lens, table_geom, mamba_geom, ckpt_window=1
    )
    out_compiled = _staged_walk_products_compiled.__wrapped__(
        staged_dev, seq_lens, table_geom, mamba_geom, ckpt_window=1
    )
    assert len(out_w1) == 3
    for a, b in zip(out_w1, out_compiled):
        assert torch.equal(a, b)

    # 2. _active_block_table_columns edge branches:
    #    - seq_lens_cpu is None and ctx.seq_lens is None -> returns target_num_blocks
    #    - max_seq_len <= 0 -> returns min(16, target_num_blocks)
    #    - is_mamba_group and mamba_ckpt_window > 1 -> adds mamba_ckpt_window - 1
    class _FakeBT:
        append_row = None

        def __init__(self, offset: int):
            self.max_num_blocks_per_req = 64
            self.num_blocks_per_row = np.zeros(4, dtype=np.int32)
            self._cpu = torch.arange(256, dtype=torch.int32).reshape(4, 64) + offset

        def get_cpu_tensor(self):
            return self._cpu

    bt0 = _FakeBT(0)
    bt1 = _FakeBT(100)
    bt2 = _FakeBT(200)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        max_num_reqs=4,
        most_model_len=512,
        position_ids=torch.arange(4, dtype=torch.int32),
        _block_table_stage_cpu=None,
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt0, bt1, bt2]),
        attn_groups=[],
    )
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=16384,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(4, 128)],
        dtypes=[torch.bfloat16],
        page_size_padded=16384,
    )
    b_attn0 = AttentionMetadataBuilder(
        attn_spec, ["attn.0"], MagicMock(), runner.device, runner, 0
    )
    b_attn1 = AttentionMetadataBuilder(
        attn_spec, ["attn.1"], MagicMock(), runner.device, runner, 1
    )
    b_mamba = AttentionMetadataBuilder(
        mamba_spec, ["mamba.0"], MagicMock(), runner.device, runner, 2
    )

    ctx_none_seq = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=None,  # type: ignore[arg-type]
        query_start_loc=torch.arange(3, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    assert b_attn0._active_block_table_columns(ctx_none_seq, 64) == 64

    ctx_zero_seq = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    assert b_attn0._active_block_table_columns(ctx_zero_seq, 64) == 16
    # Cache hit on second call with same geometry key
    assert b_attn1._active_block_table_columns(ctx_zero_seq, 64) == 16

    ctx_mamba_ckpt = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([256, 256, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        mamba_ckpt_window=3,
    )
    # 256 // 16 = 16 blocks + 2 ckpt blocks = 18 -> next power of 2 is 32
    assert b_mamba._active_block_table_columns(ctx_mamba_ckpt, 64) == 32

    # 3. build() branches:
    #    - active_cols == target_num_blocks and ctx.num_reqs < target_num_reqs
    #    - active_cols == target_num_blocks and ctx.num_reqs == target_num_reqs
    #    - mamba with ctx.mamba_state_indices is not None
    #    - mamba with staged is not None, unified_kv_layout, staged_mamba_products=None
    #    - mamba with ctx.mamba_state_indices is None and unified_kv_layout=False
    cm4 = SimpleNamespace(num_reqs=4)
    ctx_full_cols_partial_reqs = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([1024, 1024, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    runner._attn_metadata_builder_ctx = ctx_full_cols_partial_reqs
    md_partial = b_attn0.build(0, cm4)
    assert torch.equal(
        md_partial.block_tables.reshape(4, 64)[2:],
        torch.zeros((2, 64), dtype=torch.int32),
    )

    ctx_full_cols_all_reqs = AttentionMetadataBuilderContext(
        num_reqs=4,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.full((4,), 1024, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
    )
    runner._attn_metadata_builder_ctx = ctx_full_cols_all_reqs
    md_all = b_attn0.build(0, cm4)
    assert torch.equal(md_all.block_tables.reshape(4, 64), bt0.get_cpu_tensor())

    # Mamba with compact mamba_state_indices provided
    compact_slots = torch.tensor([3, 1, 0, 0], dtype=torch.int32)
    ctx_compact_mamba = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([16, 32, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        mamba_state_indices=compact_slots,
    )
    runner._attn_metadata_builder_ctx = ctx_compact_mamba
    md_compact = b_mamba.build(0, cm4)
    assert md_compact.mamba_state_indices is compact_slots

    # Mamba with staged table, unified_kv_layout=True, but staged_mamba_products=None
    runner._unified_kv_layout = True
    ctx_staged_mamba_no_products = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([16, 32, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        staged_block_tables={2: bt2.get_cpu_tensor().reshape(-1)},
        staged_mamba_products=None,
    )
    runner._attn_metadata_builder_ctx = ctx_staged_mamba_no_products
    md_staged_mamba = b_mamba.build(0, cm4)
    assert md_staged_mamba.mamba_state_indices is not None

    # Mamba with mamba_state_indices is None and unified_kv_layout=False
    runner._unified_kv_layout = False
    ctx_fallback_mamba = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([16, 32, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    runner._attn_metadata_builder_ctx = ctx_fallback_mamba
    md_fallback = b_mamba.build(0, cm4)
    assert md_fallback.mamba_state_indices is None

    # 4. stage_block_table_uploads & build_attention_metadata_for_layers:
    #    - gids=set() -> plan is None early return
    #    - multi-group non-mamba build() template reuse (with_block_tables)
    #    - full columns with ctx.num_reqs < num_reqs_padded vs == num_reqs_padded
    #    - staged_dev.device.type == "tpu" branch
    runner.attn_groups = [
        [SimpleNamespace(metadata_builders=[b_attn0], layer_names=["attn.0"])],
        [SimpleNamespace(metadata_builders=[b_attn1], layer_names=["attn.1"])],
    ]
    ctx_empty_gids = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([1024, 1024, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    stage_block_table_uploads(runner, ctx_empty_gids, 4, gids=set())
    assert ctx_empty_gids.staged_block_tables is None

    # Stage with num_reqs < num_reqs_padded and active_cols == target_num_blocks
    ctx_stage_partial = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([1024, 1024, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    stage_block_table_uploads(runner, ctx_stage_partial, 4)
    runner._attn_metadata_builder_ctx = ctx_stage_partial
    built0 = b_attn0.build(0, cm4)
    built1 = b_attn1.build(0, cm4)
    assert ctx_stage_partial.base_attn_metadata is built0
    assert built1 is not built0
    assert torch.equal(built1.block_tables, ctx_stage_partial.staged_block_tables[1])

    # Stage with num_reqs == num_reqs_padded and active_cols == target_num_blocks
    runner._block_table_delta_stager = None
    ctx_stage_full = AttentionMetadataBuilderContext(
        num_reqs=4,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.full((4,), 1024, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
    )
    stage_block_table_uploads(runner, ctx_stage_full, 4)
    assert ctx_stage_full.staged_block_tables is not None

    # TPU device dispatch in stage_block_table_uploads
    tpu_staged = MagicMock()
    tpu_staged.device.type = "tpu"
    with (
        patch.object(am.BlockTableDeltaStager, "try_stage", return_value=tpu_staged),
        patch.object(am.BlockTableDeltaStager, "get_cached_tables", return_value=None),
        patch.object(
            am,
            "_staged_walk_products_compiled",
            return_value=(torch.zeros(256, dtype=torch.int32),) * 2,
        ) as mock_compiled,
    ):
        ctx_tpu = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.full((4,), 1024, dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        )
        stage_block_table_uploads(runner, ctx_tpu, 4)
        mock_compiled.assert_called_once()

    # build_attention_metadata_for_layers cache miss + hit + pre-staged + empty
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([1024, 1024, 0, 0], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    md_layers1 = TPUModelRunner.build_attention_metadata_for_layers(
        runner, ["attn.0", "attn.1"], 4
    )
    assert set(md_layers1) == {"attn.0", "attn.1"}
    # TD8.341: cached group ids are a frozenset (used as a plan-cache key).
    assert all(
        isinstance(gids, frozenset)
        for _, gids in runner._layer_group_selection_cache.values()
    )
    # Second call hits _layer_group_selection_cache and pre-staged block tables
    md_layers2 = TPUModelRunner.build_attention_metadata_for_layers(
        runner, ["attn.0", "attn.1"], 4
    )
    assert set(md_layers2) == {"attn.0", "attn.1"}
    # Empty layer selection
    md_empty = TPUModelRunner.build_attention_metadata_for_layers(
        runner, ["nonexistent"], 4
    )
    assert md_empty == {}

    # 5. _apply_async_token_substitution: idx_pad_len > 0, cache hit, idx_pad_len == 0
    subst_runner = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([[100, 200], [300, 400]], dtype=torch.int32)
        ),
        _fast_token_substitution=False,
        device=torch.device("cpu"),
    )
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    cur_idx_padded = np.array([0, 2], dtype=np.int32)
    pre_idx_padded = np.array([0, 2], dtype=np.int32)
    with patch.object(
        tpu_runner,
        "_substitute_placeholder_token",
        side_effect=lambda ids, c, p, nxt: (ids, c, p),
    ):
        _, c1, p1 = TPUModelRunner._apply_async_token_substitution(
            subst_runner, input_ids, cur_idx_padded, pre_idx_padded
        )
        # Cache hit reuses exact same tensor objects
        _, c2, p2 = TPUModelRunner._apply_async_token_substitution(
            subst_runner, input_ids, cur_idx_padded, pre_idx_padded
        )
        assert c2 is c1 and p2 is p1
        # Unpadded (idx_pad_len == 0)
        cur_idx_full = np.array([0, 1, 2, 3], dtype=np.int32)
        pre_idx_full = np.array([0, 1, 2, 3], dtype=np.int32)
        _, c3, p3 = TPUModelRunner._apply_async_token_substitution(
            subst_runner, input_ids, cur_idx_full, pre_idx_full
        )
        assert c3.tolist() == [0, 1, 2, 3]
        assert p3.tolist() == [0, 1, 2, 3]

    # 6. fast_swap_row in _install_spec_token_room_guard:
    #    - i1 == i2 early return
    #    - bt with _bt_cpu_np and _rewrote
    #    - bt without _bt_cpu_np (uses block_table.np) and without _rewrote
    rewrote_calls = []
    arr0 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    arr1 = np.array([[10, 20], [30, 40]], dtype=np.int32)
    bt_tracked = SimpleNamespace(
        num_blocks_per_row=np.array([1, 2], dtype=np.int32),
        _bt_cpu_np=arr0,
        _rewrote=lambda r: rewrote_calls.append(r),
    )
    bt_untracked = SimpleNamespace(
        num_blocks_per_row=np.array([2, 1], dtype=np.int32),
        block_table=SimpleNamespace(np=arr1),
    )
    swap_runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            update_req_spec_token_ids=lambda req, spec: None,
            block_table=SimpleNamespace(
                block_tables=[bt_tracked, bt_untracked],
                append_row=None,
                swap_row=None,
            ),
        )
    )
    TPUModelRunner._install_spec_token_room_guard(swap_runner)
    swap_runner.input_batch.block_table.swap_row(1, 1)
    assert rewrote_calls == []
    swap_runner.input_batch.block_table.swap_row(0, 1)
    assert rewrote_calls == [0, 1]
    assert bt_tracked.num_blocks_per_row.tolist() == [2, 1]
    assert bt_untracked.num_blocks_per_row.tolist() == [1, 2]
    assert arr0[0].tolist() == [3, 4] and arr0[1].tolist() == [1, 2]
    assert arr1[0].tolist() == [30, 40] and arr1[1].tolist() == [10, 20]

    # 7. _prepare_inputs branches for non-cached prefill and non-ALL sequence layout
    max_num_reqs = 2
    max_model_len = 32
    token_ids_cpu = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    prep_runner = SimpleNamespace(
        reorder_batch_threshold=1,
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["req-0", "req-1"],
            num_computed_tokens_cpu=np.array([0, 0], dtype=np.int32),
            token_ids_cpu=token_ids_cpu,
            token_ids_cpu_tensor=torch.from_numpy(token_ids_cpu),
            num_prompt_tokens=np.full(max_num_reqs, 8, dtype=np.int32),
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
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        _pre_async_results=None,
        _decode_device_cache_key=None,
        _request_distribution_cpu=torch.zeros(3, dtype=torch.int32),
        _has_mamba_state=False,
        mamba_slot_read_offsets=None,
        _mamba_ckpt_window=1,
        _unified_kv_layout=False,
        empty_slot_mappings=None,
        _build_attention_metadata=lambda **kw: ({}, None),
        lora_config=None,
        _collect_mamba_state_seed_copies=lambda *a: None,
        num_tokens_paddings=[8, 16, 32],
        pcp_size=1,
        dcp_size=1,
        _prepare_async_token_substitution_indices=lambda s, n, arr: (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ),
    )
    sched_prefill = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"req-0": 4, "req-1": 4},
        scheduled_spec_decode_tokens={},
    )
    # Prefill step: can_cache_decode_metadata is False
    TPUModelRunner._prepare_inputs(
        prep_runner, sched_prefill, start_index=0, num_decode_reqs=0
    )
    assert prep_runner._attn_metadata_builder_ctx.query_start_loc_cpu == (0, 4, 8)
    assert prep_runner._attn_metadata_builder_ctx.prompt_lens_cpu == (8, 8)

    # Non-ALL layout plan: query_start_loc_cpu, seq_lens_cpu, prompt_lens_cpu are None
    partial_desc = SequenceLayoutDescriptor(
        kind=SequenceLayoutKind.PARTIAL,
        protocol="pcp_streaming",
        version=1,
    )
    partial_plan = SequenceLayoutPlan(
        descriptor=partial_desc,
        token_slice=slice(0, 8),
        global_num_tokens=8,
        global_padded_num_tokens=8,
        local_num_tokens=8,
        local_padded_num_tokens=8,
    )
    prep_runner.sequence_layout_planner = SimpleNamespace(
        reserve_host_token_capacity=lambda *a, **kw: None,
        prepare_real=lambda **kw: partial_plan,
    )
    TPUModelRunner._prepare_inputs(
        prep_runner, sched_prefill, start_index=0, num_decode_reqs=0
    )
    assert prep_runner._attn_metadata_builder_ctx.query_start_loc_cpu is None
    assert prep_runner._attn_metadata_builder_ctx.seq_lens_cpu is None
    assert prep_runner._attn_metadata_builder_ctx.prompt_lens_cpu is None


def test_td73_spec_bookkeeping_and_acceptance_fast_paths():
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        INVALID_TOKEN_ID,
        AsyncPreResults,
        AsyncTPUCopyState,
    )

    num_tokens_no_spec = np.array([4, 6], dtype=np.int32)
    token_ids_cpu = np.zeros((2, 16), dtype=np.int32)
    token_ids_cpu[0, :4] = [1, 2, 3, 4]
    token_ids_cpu[1, :6] = [10, 20, 30, 40, 50, 60]
    req0 = SimpleNamespace(req_id="req-0", output_token_ids=[101, 102])
    req1 = SimpleNamespace(req_id="req-1", output_token_ids=[201, 202])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req-0": 0, "req-1": 1},
            num_tokens_no_spec=num_tokens_no_spec,
            token_ids_cpu=token_ids_cpu,
        ),
        _pre_async_results=None,
    )

    # 1. Empty request_seq_lens in _update_placeholder
    assert TPUModelRunner._update_placeholder(runner, [], [], {}) == {}

    # 2. Heterogeneous num_draft_per_req in contiguous _update_placeholder fast path
    req_seq_lens = [(0, req0, 4, "req-0"), (1, req1, 6, "req-1")]
    num_draft_hetero = {0: 3, 1: 1}
    idx_map = TPUModelRunner._update_placeholder(
        runner, [], req_seq_lens, {0: 0, 1: 1}, num_draft_hetero
    )
    assert idx_map == {"req-0": 0, "req-1": 1}
    # req-0 advanced by 1+3=4 (to 8), req-1 advanced by 1+1=2 (to 8)
    assert num_tokens_no_spec.tolist() == [8, 8]
    assert req0.output_token_ids == [101, 102, 0, 0, 0, 0]
    assert req1.output_token_ids == [201, 202, 0, 0]

    # 3. _modify_prev_results contiguous fast path with heterogeneous num_draft_per_req
    #    and AsyncTPUCopyState (1D-flattened chunk reshaped back to 2D)
    sampled_2d = torch.tensor(
        [
            [11, 12, 13, INVALID_TOKEN_ID],
            [21, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    with patch.object(torch, "tpu", SimpleNamespace(Event=MagicMock()), create=True):
        copy_state = AsyncTPUCopyState.from_device_chunks([sampled_2d], [2])
    runner._pre_async_results = AsyncPreResults(
        req_ids=["req-0", "req-1"],
        next_tokens_tpu=torch.zeros(8, dtype=torch.int32),
        request_seq_lens=req_seq_lens,
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"req-0": 0, "req-1": 1},
        copy_state=copy_state,
        num_draft_per_req=num_draft_hetero,
    )
    TPUModelRunner._modify_prev_results(runner)
    # req-0 committed 3 tokens (4+3=7), req-1 committed 1 token (6+1=7)
    assert num_tokens_no_spec.tolist() == [7, 7]
    assert req0.output_token_ids == [101, 102, 11, 12, 13]
    assert req1.output_token_ids == [201, 202, 21]
    assert token_ids_cpu[0, :7].tolist() == [1, 2, 3, 4, 11, 12, 13]
    assert token_ids_cpu[1, :7].tolist() == [10, 20, 30, 40, 50, 60, 21]

    # 4. _update_placeholder slow path with discard_sampled_tokens_req_indices
    idx_map_discard = TPUModelRunner._update_placeholder(
        runner, [0], req_seq_lens, {0: 0, 1: 1}, None
    )
    assert idx_map_discard == {"req-1": 1}
    assert num_tokens_no_spec.tolist() == [7, 8]


def test_td713_prepare_inputs_uniform_2d_broadcast_and_fast_subst():
    # 1. _apply_async_token_substitution with _fast_token_substitution=True
    fast_runner = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        ),
        _fast_token_substitution=True,
        arange_np=np.arange(16, dtype=np.int32),
        device=torch.device("cpu"),
    )
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    idx4 = np.arange(4, dtype=np.int32)
    out_fast = TPUModelRunner._apply_async_token_substitution(
        fast_runner, input_ids, idx4, idx4
    )
    assert out_fast.tolist() == [101, 102, 103, 104]

    # 2. _prepare_inputs uniform multi-token 2D broadcast and empty spec tokens
    max_num_reqs = 2
    max_model_len = 32
    token_ids_cpu = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    prep_runner = SimpleNamespace(
        reorder_batch_threshold=1,
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["req-0", "req-1"],
            num_computed_tokens_cpu=np.array([4, 10], dtype=np.int32),
            token_ids_cpu=token_ids_cpu,
            token_ids_cpu_tensor=torch.from_numpy(token_ids_cpu),
            num_prompt_tokens=np.full(max_num_reqs, 4, dtype=np.int32),
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
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        _pre_async_results=None,
        _decode_device_cache_key=None,
        _request_distribution_cpu=torch.zeros(3, dtype=torch.int32),
        _has_mamba_state=False,
        mamba_slot_read_offsets=None,
        _mamba_ckpt_window=1,
        _unified_kv_layout=False,
        empty_slot_mappings=None,
        _build_attention_metadata=lambda **kw: ({}, None),
        lora_config=None,
        _collect_mamba_state_seed_copies=lambda *a: None,
        num_tokens_paddings=[8, 16, 32],
        pcp_size=1,
        dcp_size=1,
        _prepare_async_token_substitution_indices=lambda s, n, arr: (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ),
    )
    sched_uniform = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"req-0": 4, "req-1": 4},
        scheduled_spec_decode_tokens={"req-0": [], "req-1": []},
    )
    TPUModelRunner._prepare_inputs(
        prep_runner, sched_uniform, start_index=0, num_decode_reqs=0
    )
    assert prep_runner.positions_np[:8].tolist() == [4, 5, 6, 7, 10, 11, 12, 13]
    assert prep_runner.input_ids_cpu[:8].tolist() == [
        4,
        5,
        6,
        7,
        32 + 10,
        32 + 11,
        32 + 12,
        32 + 13,
    ]
