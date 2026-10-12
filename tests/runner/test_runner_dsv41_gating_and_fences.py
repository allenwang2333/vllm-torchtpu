# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 runner gating, metadata cache, and H2D fences."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.core.attention_metadata import (
    _get_staged_walk_plan,
    stage_block_table_uploads,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
)
from vllm_torchtpu.runner.speculative_decoding_manager import (
    DraftTokenIds,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUModelRunnerOutput,
)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

# ---------------------------------------------------------------------------
# 1. TPUModelRunner: _is_deepseek_v41_model Truth Table
# ---------------------------------------------------------------------------


def test_is_deepseek_v41_model_truth_table():
    # Instance dict override
    r1 = SimpleNamespace(_is_deepseek_v41=True)
    assert TPUModelRunner._is_deepseek_v41_model(r1) is True

    r2 = SimpleNamespace(_is_deepseek_v41=False)
    assert TPUModelRunner._is_deepseek_v41_model(r2) is False

    # model_config is None -> True fallback
    r3 = SimpleNamespace()
    assert TPUModelRunner._is_deepseek_v41_model(r3) is True

    # hf_config is None -> True fallback
    r4 = SimpleNamespace(model_config=SimpleNamespace(hf_config=None))
    assert TPUModelRunner._is_deepseek_v41_model(r4) is True

    # Architectures match
    r5 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        )
    )
    assert TPUModelRunner._is_deepseek_v41_model(r5) is True

    # Model type match
    r6 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[], model_type="deepseek_v41")
        )
    )
    assert TPUModelRunner._is_deepseek_v41_model(r6) is True

    # Non-DSv4.1 model -> False
    r7 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"], model_type="llama"
            )
        )
    )
    assert TPUModelRunner._is_deepseek_v41_model(r7) is False


# ---------------------------------------------------------------------------
# 2. TPUModelRunner: Gating of DSv4.1 Fast Paths
# ---------------------------------------------------------------------------


def test_may_reorder_batch_gating():
    # DSv4.1: no-op, GPUModelRunner._may_reorder_batch not called
    r_dsv = SimpleNamespace(_is_deepseek_v41=True)
    with patch.object(GPUModelRunner, "_may_reorder_batch") as mock_gpu:
        TPUModelRunner._may_reorder_batch(r_dsv, None)
        mock_gpu.assert_not_called()

    # Non-DSv4.1: calls GPUModelRunner._may_reorder_batch(self, scheduler_output)
    r_non = SimpleNamespace(_is_deepseek_v41=False)
    sched = SimpleNamespace()
    with patch.object(GPUModelRunner, "_may_reorder_batch") as mock_gpu:
        TPUModelRunner._may_reorder_batch(r_non, sched)
        mock_gpu.assert_called_once_with(r_non, sched)


def test_install_spec_token_room_guard_gating():
    # Non-DSv4.1: does not wrap block tables
    tables_non = [SimpleNamespace(append_row=None, swap_row=None)]
    r_non = SimpleNamespace(
        _is_deepseek_v41=False,
        input_batch=SimpleNamespace(
            update_req_spec_token_ids=lambda r, s: None,
            block_table=SimpleNamespace(
                block_tables=tables_non, append_row=None, swap_row=None
            ),
        ),
    )
    TPUModelRunner._install_spec_token_room_guard(r_non)
    assert r_non.input_batch.block_table.append_row is None

    # DSv4.1: wraps block tables
    tables_dsv = [
        SimpleNamespace(
            append_row=None,
            swap_row=None,
            num_blocks_per_row=np.array([1], dtype=np.int32),
            _bt_cpu_np=np.array([[10]], dtype=np.int32),
        )
    ]
    r_dsv = SimpleNamespace(
        _is_deepseek_v41=True,
        input_batch=SimpleNamespace(
            update_req_spec_token_ids=lambda r, s: None,
            block_table=SimpleNamespace(
                block_tables=tables_dsv, append_row=None, swap_row=None
            ),
        ),
    )
    TPUModelRunner._install_spec_token_room_guard(r_dsv)
    assert r_dsv.input_batch.block_table.append_row is not None


def test_try_fast_update_states_gating_and_strides():
    # 1. Non-DSv4.1 returns False early
    r_non = SimpleNamespace(
        _is_deepseek_v41=False,
        input_batch=SimpleNamespace(req_ids=["r0"]),
        requests={"r0": object()},
    )
    sched_non = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r0"]),
        scheduled_new_reqs=[],
    )
    assert TPUModelRunner._try_fast_update_states(r_non, sched_non) is False

    # 2. Large stride in _try_fast_update_states: n_reqs * max_len >= 2**31 vs < 2**31
    class MockTokenIdsCpu:
        shape = (2, 1 << 31)

        def ravel(self):
            m = MagicMock()
            return m

    mock_tok = MockTokenIdsCpu()
    num_tokens = np.array([0, 0], dtype=np.int32)
    spec_token_ids = [[1, 2], [1, 2]]
    r_large = SimpleNamespace(
        _is_deepseek_v41=True,
        _pp_is_last=True,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
            num_prompt_tokens=np.zeros(2, dtype=np.int32),
            num_tokens_no_spec=num_tokens,
            block_table=SimpleNamespace(append_row=MagicMock()),
            token_ids_cpu=mock_tok,
            is_token_ids=MagicMock(),
            spec_token_ids=spec_token_ids,
            refresh_metadata=MagicMock(),
        ),
        requests={
            "r0": SimpleNamespace(output_token_ids=[], block_ids=[]),
            "r1": SimpleNamespace(output_token_ids=[], block_ids=[]),
        },
    )
    sched = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens=[0, 0],
            num_output_tokens=[0, 0],
            new_block_ids=[None, None],
        ),
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        preempted_req_ids=set(),
        new_block_ids_to_zero=None,
        free_encoder_mm_hashes=None,
        num_scheduled_tokens={"r0": 3, "r1": 3},
        num_computed_tokens={"r0": 0, "r1": 0},
        scheduled_spec_decode_tokens={"r0": [1, 2], "r1": [1, 2]},
    )
    res = TPUModelRunner._try_fast_update_states(r_large, sched)
    assert res is True

    # Normal small stride
    tok_small = np.zeros((2, 16), dtype=np.int32)
    r_small = SimpleNamespace(
        _is_deepseek_v41=True,
        _pp_is_last=True,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
            num_prompt_tokens=np.zeros(2, dtype=np.int32),
            num_tokens_no_spec=np.array([0, 0], dtype=np.int32),
            block_table=SimpleNamespace(append_row=MagicMock()),
            token_ids_cpu=tok_small,
            is_token_ids=np.zeros((2, 16), dtype=bool),
            spec_token_ids=[[1], [1]],
            refresh_metadata=MagicMock(),
        ),
        requests={
            "r0": SimpleNamespace(output_token_ids=[], block_ids=[]),
            "r1": SimpleNamespace(output_token_ids=[], block_ids=[]),
        },
    )
    sched_small = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens=[0, 0],
            num_output_tokens=[0, 0],
            new_block_ids=[None, None],
        ),
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        preempted_req_ids=set(),
        new_block_ids_to_zero=None,
        free_encoder_mm_hashes=None,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        num_computed_tokens={"r0": 0, "r1": 0},
        scheduled_spec_decode_tokens={"r0": [1], "r1": [1]},
    )
    assert TPUModelRunner._try_fast_update_states(r_small, sched_small) is True


def test_modify_prev_results_large_and_small_strides():
    # 1. Large stride in _modify_prev_results (lines 3107-3112):
    # n_reqs * max_model_len >= 2**31
    class LargeTokens:
        shape = (2, 1 << 31)

        def ravel(self):
            return MagicMock()

    r_large = SimpleNamespace(
        arange_np=np.arange(16, dtype=np.int32),
        input_batch=SimpleNamespace(
            token_ids_cpu=LargeTokens(),
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            num_tokens_no_spec=np.array([2, 2], dtype=np.int32),
        ),
    )
    req0 = SimpleNamespace(output_token_ids=[1, 2])
    req1 = SimpleNamespace(output_token_ids=[3, 4])
    sampled_large = torch.tensor([[10], [20]], dtype=torch.int32)
    copy_state_large = SimpleNamespace(
        parse_sampled_tokens=lambda copy_rows=False: ([[10], [20]], [1, 1]),
        _sampled_token_ids_cpu=sampled_large,
    )
    r_large._pre_async_results = AsyncPreResults(
        req_ids=["r0", "r1"],
        next_tokens_tpu=torch.zeros(2, dtype=torch.int32),
        request_seq_lens=[(0, req0, 2, "r0"), (1, req1, 2, "r1")],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"r0": 0, "r1": 1},
        copy_state=copy_state_large,
        num_draft_per_req=None,
    )
    TPUModelRunner._modify_prev_results(r_large)

    # 2. Normal small stride
    small_tokens = np.zeros((2, 16), dtype=np.int32)
    r_small = SimpleNamespace(
        arange_np=np.arange(16, dtype=np.int32),
        input_batch=SimpleNamespace(
            token_ids_cpu=small_tokens,
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            num_tokens_no_spec=np.array([2, 2], dtype=np.int32),
        ),
    )
    r_small._pre_async_results = AsyncPreResults(
        req_ids=["r0", "r1"],
        next_tokens_tpu=torch.zeros(2, dtype=torch.int32),
        request_seq_lens=[(0, req0, 2, "r0"), (1, req1, 2, "r1")],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"r0": 0, "r1": 1},
        copy_state=copy_state_large,
        num_draft_per_req=None,
    )
    TPUModelRunner._modify_prev_results(r_small)
    assert small_tokens[0, 1] == 10


# ---------------------------------------------------------------------------
# 3. TPUModelRunner: Decode Cache Invalidation and Prepare Inputs Strides
# ---------------------------------------------------------------------------


def test_prepare_inputs_strides_and_qsl_cache():
    max_num_reqs = 2
    max_model_len = 32

    # 1. Stride calculation for (start_index + num_reqs) * max_model_len >= 2**31
    class LargeTokenIdsCpu:
        shape = (2, 1 << 31)

        def __getitem__(self, idx):
            return MagicMock()

    token_ids_large = LargeTokenIdsCpu()
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)

    runner = SimpleNamespace(
        reorder_batch_threshold=1,
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["r0", "r1"],
            num_computed_tokens_cpu=np.array([0, 0], dtype=np.int32),
            token_ids_cpu=token_ids_large,
            token_ids_cpu_tensor=MagicMock(),
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

    sched_decode = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 1, "r1": 1},
        scheduled_spec_decode_tokens={},
    )
    with patch("numpy.take"):
        # Decode path with large stride
        TPUModelRunner._prepare_inputs(
            runner, sched_decode, start_index=0, num_decode_reqs=2
        )
        assert runner._cached_qsl_cpu_key is not None

        # Cache hit on second decode call
        TPUModelRunner._prepare_inputs(
            runner, sched_decode, start_index=0, num_decode_reqs=2
        )

        # Cache miss / invalidation on prefill step (can_cache_decode_metadata is False)
        sched_prefill = SimpleNamespace(
            total_num_scheduled_tokens=8,
            num_scheduled_tokens={"r0": 4, "r1": 4},
            scheduled_spec_decode_tokens={},
        )
        TPUModelRunner._prepare_inputs(
            runner, sched_prefill, start_index=0, num_decode_reqs=0
        )
        assert runner._cached_qsl_cpu_key is None

        # Heterogeneous scheduled tokens (exercises line 3618 in else branch)
        sched_hetero = SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens={"r0": 1, "r1": 2},
            scheduled_spec_decode_tokens={},
        )
        TPUModelRunner._prepare_inputs(
            runner, sched_hetero, start_index=0, num_decode_reqs=0
        )


# ---------------------------------------------------------------------------
# 4. TPUModelRunner: Async Engram Staged Next & Warmup Cleanup Fences
# ---------------------------------------------------------------------------


def test_modify_prev_results_engram_fence_and_cleanup():
    sync_mock = MagicMock()
    runner = SimpleNamespace(
        _modify_prev_results=lambda: None,
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy={"r_old": 0},
            wait_for_copy=MagicMock(),
        ),
        _host_engram_staged_next=(None, None, sync_mock),
        requests={"r0": object()},
        input_batch=SimpleNamespace(
            req_id_to_index={"r0": 0},
            num_reqs=1,
            req_ids=["r0"],
            spec_token_ids=[],
        ),
        spec_decode_manager=SimpleNamespace(
            _draft_token_ids=[1, 2],
            _staged_draft_copy=(None, sync_mock, None, None, None),
        ),
        _release_warmup_chunk_activations=MagicMock(),
    )
    # Flush disjoint async results
    TPUModelRunner._flush_disjoint_async_results(runner)
    sync_mock.synchronize.assert_called_once()
    assert runner._host_engram_staged_next is None
    assert runner._pre_async_results is None

    # Branches for staged_next is None and staged_next[2] is None
    # in _flush_disjoint_async_results
    runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"r_old": 0},
        wait_for_copy=MagicMock(),
    )
    runner._host_engram_staged_next = None
    TPUModelRunner._flush_disjoint_async_results(runner)
    assert runner._pre_async_results is None

    runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"r_old": 0},
        wait_for_copy=MagicMock(),
    )
    runner._host_engram_staged_next = (None, None, None)
    TPUModelRunner._flush_disjoint_async_results(runner)
    assert runner._pre_async_results is None

    # Test _warmup_spec_decode_cleanup
    sync_mock.reset_mock()
    runner._pre_async_results = SimpleNamespace(wait_for_copy=MagicMock())
    runner._host_engram_staged_next = (None, None, sync_mock)
    runner.input_batch.num_reqs = 0
    TPUModelRunner._warmup_spec_decode_cleanup(runner, ["r0"])
    sync_mock.synchronize.assert_called()
    assert runner.spec_decode_manager._staged_draft_copy is None
    assert runner.spec_decode_manager._draft_token_ids is None

    # Branch for staged_next is None and staged_next[2] is None
    # in _warmup_spec_decode_cleanup
    runner._pre_async_results = SimpleNamespace(wait_for_copy=MagicMock())
    runner._host_engram_staged_next = None
    TPUModelRunner._warmup_spec_decode_cleanup(runner, ["r0"])

    runner._pre_async_results = SimpleNamespace(wait_for_copy=MagicMock())
    runner._host_engram_staged_next = (None, None, None)
    TPUModelRunner._warmup_spec_decode_cleanup(runner, ["r0"])


# ---------------------------------------------------------------------------
# 5. TPUModelRunner: Staged Sampling Params Fences
# ---------------------------------------------------------------------------


def test_stage_sampling_params_fences():
    sync_mock = MagicMock()
    runner = SimpleNamespace(
        _sampling_params_staging=(sync_mock, None, None, None),
        vocab_size=32000,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.ones(4),
            top_k_cpu_tensor=torch.zeros(4, dtype=torch.int32),
            top_p_cpu_tensor=torch.ones(4),
        ),
    )
    # CPU logits device -> fence is None
    logits_cpu = torch.zeros(4, 10, device="cpu")
    t, k, p = TPUModelRunner._build_padded_sampling_params(runner, 0, 4, logits_cpu)
    sync_mock.synchronize.assert_called_once()
    assert runner._sampling_params_staging[0] is None

    # Non-CPU / TPU device -> creates torch.tpu.Event()
    tpu_event_mock = MagicMock()
    with patch.object(
        torch,
        "tpu",
        SimpleNamespace(Event=MagicMock(return_value=tpu_event_mock)),
        create=True,
    ):
        dev = torch.device("meta")
        logits_meta = torch.zeros(4, 10, device=dev)
        t, k, p = TPUModelRunner._build_padded_sampling_params(
            runner, 0, 4, logits_meta
        )
        tpu_event_mock.record.assert_called_once()
        assert runner._sampling_params_staging[0] is tpu_event_mock


# ---------------------------------------------------------------------------
# 6. TPUModelRunner: _has_custom_logits_hook & _model_supports_top_tokens
# ---------------------------------------------------------------------------


def test_custom_logits_hook_and_model_supports_top_tokens():
    # 1. _dsv41_smoke_logits_hooked on instance
    r1 = TPUModelRunner.__new__(TPUModelRunner)
    r1._dsv41_smoke_logits_hooked = True
    assert r1._has_custom_logits_hook() is True
    assert r1._model_supports_top_tokens() is False

    # 2. _dsv41_smoke_logits_hooked on class
    class HookedRunner(TPUModelRunner):
        _dsv41_smoke_logits_hooked = True

    r2 = HookedRunner.__new__(HookedRunner)
    assert r2._has_custom_logits_hook() is True

    # 3. compute_selected_logits in instance __dict__
    r3 = TPUModelRunner.__new__(TPUModelRunner)
    r3.compute_selected_logits = lambda *a: None
    assert r3._has_custom_logits_hook() is True

    # 4. compute_logits_from_hidden_states in instance __dict__
    r4 = TPUModelRunner.__new__(TPUModelRunner)
    r4.compute_logits_from_hidden_states = lambda *a: None
    assert r4._has_custom_logits_hook() is True

    # 5. Class override of compute_selected_logits
    class OverrideRunner1(TPUModelRunner):
        def compute_selected_logits(self, *a):
            pass

    r5 = OverrideRunner1.__new__(OverrideRunner1)
    assert r5._has_custom_logits_hook() is True

    # 6. Class override of compute_logits_from_hidden_states
    class OverrideRunner2(TPUModelRunner):
        def compute_logits_from_hidden_states(self, *a):
            pass

    r6 = OverrideRunner2.__new__(OverrideRunner2)
    assert r6._has_custom_logits_hook() is True

    # 7. Standard unhooked runner with model and supported head
    r7 = TPUModelRunner.__new__(TPUModelRunner)
    r7.is_pooling_model = False
    r7.model = SimpleNamespace(
        logits_processor=SimpleNamespace(
            logits_as_input=False, scale=1.0, get_top_tokens=lambda *a: None
        ),
        lm_head=SimpleNamespace(
            tp_size=1,
            shard_indices=SimpleNamespace(num_org_vocab_padding=0),
        ),
    )
    assert r7._has_custom_logits_hook() is False
    assert r7._model_supports_top_tokens() is True

    # Pooling model
    r7.is_pooling_model = True
    assert r7._model_supports_top_tokens() is False


# ---------------------------------------------------------------------------
# 7. TPUModelRunner: gather_logprobs Int64 Index Arithmetic
# ---------------------------------------------------------------------------


def test_gather_logprobs_int64_index_arithmetic():
    fn = TPUModelRunner.gather_logprobs.__wrapped__

    # 1. Normal int32 logits
    runner = MagicMock()
    runner.model_config.max_logprobs = 2
    logits_small = torch.randn(2, 4)
    sampled = torch.tensor([[1], [2]], dtype=torch.int64)
    res_small = fn(runner, logits_small, sampled)
    assert isinstance(res_small, LogprobsTensors)

    # 2. Large logits mock triggering int64 arithmetic:
    # logits.shape[0] * logits.shape[1] >= 2**31
    class MockLargeLogits:
        shape = (2, 1 << 31)
        device = torch.device("cpu")

        def reshape(self, *a):
            m = MagicMock()
            m.index_select.return_value.unsqueeze.return_value = torch.zeros(2, 1)
            return m

        def __ge__(self, other):
            return torch.ones(2, 1, dtype=torch.bool)

        def float(self):
            return torch.zeros(2, 1)

    runner.model_config.max_logprobs = 0
    with patch("torch.logsumexp", return_value=torch.zeros(2, 1)):
        res_large = fn(runner, MockLargeLogits(), sampled)
        assert isinstance(res_large, LogprobsTensors)


# ---------------------------------------------------------------------------
# 8. AsyncTPUModelRunnerOutput: Defensive Row Copying
# ---------------------------------------------------------------------------


def test_async_tpu_model_runner_output_defensive_copies():
    copy_state = MagicMock()
    copy_state.parse_sampled_tokens.return_value = ([[10, 20]], [2])
    copy_state.spec_logprobs = None
    copy_state.sampled_token_ids_cpu = torch.tensor([[10, 20]])

    model_runner_output = SimpleNamespace(req_ids=["r0"])
    output = AsyncTPUModelRunnerOutput(
        model_runner_output=model_runner_output,
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[],
    )
    # get_output should call copy_state.parse_sampled_tokens without copy_rows=False
    res = output.get_output()
    copy_state.parse_sampled_tokens.assert_called_once_with()
    assert res.sampled_token_ids == [[10, 20]]


# ---------------------------------------------------------------------------
# 9. DFlashProposer: Request Window Bounds & Try Borrow Staged Tables
# ---------------------------------------------------------------------------


def test_dflash_request_window_bounds():
    proposer = DFlashProposer.__new__(DFlashProposer)

    # 1. num_reqs is None, query_start_loc int32, position_ids int32
    qsl = torch.tensor([0, 4, 8, 8], dtype=torch.int32)
    pos = torch.arange(10, dtype=torch.int32)
    safe_idx, mask, start_p, last_p = proposer._request_window_bounds(
        query_start_loc=qsl,
        position_ids=pos,
        padded_num_reqs=4,
        num_reqs=None,
    )
    assert safe_idx.tolist() == [0, 1, 1, 1]
    assert mask.squeeze(-1).tolist() == [1, 1, 0, 0]

    # 2. num_reqs as Tensor, qsl int64, pos int64
    qsl_i64 = torch.tensor([0, 2, 2], dtype=torch.int64)
    pos_i64 = torch.arange(5, dtype=torch.int64)
    num_reqs_t = torch.tensor(1, dtype=torch.int32)
    safe_idx, mask, start_p, last_p = proposer._request_window_bounds(
        query_start_loc=qsl_i64,
        position_ids=pos_i64,
        padded_num_reqs=2,
        num_reqs=num_reqs_t,
    )
    assert mask.squeeze(-1).tolist() == [1, 0]

    # 3. num_reqs as int, with 0-token request row (has_tokens is False)
    qsl_empty_row = torch.tensor([0, 0, 2], dtype=torch.int32)
    safe_idx, mask, start_p, last_p = proposer._request_window_bounds(
        query_start_loc=qsl_empty_row,
        position_ids=pos,
        padded_num_reqs=2,
        num_reqs=2,
    )
    # Row 0 has 0 tokens -> masked out
    assert mask.squeeze(-1).tolist() == [0, 1]

    # 4. Out-of-bounds pos clamping
    short_pos = torch.zeros(1, dtype=torch.int32)
    safe_idx, mask, start_p, last_p = proposer._request_window_bounds(
        query_start_loc=qsl,
        position_ids=short_pos,
        padded_num_reqs=4,
        num_reqs=2,
    )
    assert start_p.shape == (4, 1)


def test_dflash_try_borrow_staged_tables():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._get_frozen_draft_attn_layer_names = lambda: ("layer0",)
    proposer._draft_attn_layer_names = ["layer0"]
    proposer._draft_single_group_meta_cache = {}

    class _MockBuilder:
        def __init__(self):
            self.is_mamba_group = False
            self.target_block_size = 16
            self.kv_cache_group_id = 0
            self._active_block_table_columns = MagicMock(return_value=8)

        def _block_table_geometry(self, ctx):
            return (1, 8)

    b0 = _MockBuilder()
    proposer._get_draft_group_builders = lambda groups: [b0]
    runner = SimpleNamespace(
        position_ids=torch.zeros(4, dtype=torch.int32),
        attn_groups=[object()],
    )
    proposer.runner = runner

    staged_tables = {0: torch.zeros(4 * 8, dtype=torch.int32)}
    chunk_ctx = SimpleNamespace(
        position_ids_override=None,
        use_max_model_len=True,
        staged_block_tables=staged_tables,
    )

    # Condition <= holds: borrows staged tables
    b0._active_block_table_columns.side_effect = [8, 8]
    meta = proposer._build_attn_metadata_for_draft(
        chunk_ctx=chunk_ctx,
        position_ids_override=None,
        use_max_model_len=True,
        max_active_seq_len=64,
        num_reqs=4,
        start_index=0,
        padded_num_reqs=4,
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    assert "layer0" in meta

    # Condition > holds: skips borrowing staged tables
    proposer._draft_single_group_meta_cache.clear()
    b0._active_block_table_columns.side_effect = [16, 8]
    runner.build_attention_metadata_for_layers = MagicMock(
        return_value={"layer0": None}
    )
    runner._attn_metadata_builder_ctx = None
    proposer._can_reuse_staged_block_tables = MagicMock(return_value=None)
    meta2 = proposer._build_attn_metadata_for_draft(
        chunk_ctx=chunk_ctx,
        position_ids_override=None,
        use_max_model_len=True,
        max_active_seq_len=64,
        num_reqs=4,
        start_index=0,
        padded_num_reqs=4,
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    assert "layer0" in meta2
    runner.build_attention_metadata_for_layers.assert_called_once()


# ---------------------------------------------------------------------------
# 10. SpeculativeDecodingManager: Staged Fences & All-Gather Packing
# ---------------------------------------------------------------------------


def test_spec_decode_manager_fences():
    mgr = SpeculativeDecodingManager.__new__(SpeculativeDecodingManager)
    mgr.runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["r0", "r1"]),
    )

    # 1. take_draft_token_ids with ready_event != None vs None
    event_sync = MagicMock()
    mgr._staged_draft_copy = (
        torch.tensor([1, 2]),
        event_sync,
        ["r0", "r1"],
        (2, 1),
        None,
    )
    drafts = mgr.take_draft_token_ids()
    event_sync.synchronize.assert_called_once()
    assert isinstance(drafts, DraftTokenIds)
    assert mgr._staged_draft_copy is None

    # take_draft_token_ids with ready_event == None
    mgr._staged_draft_copy = (
        torch.tensor([1, 2]),
        None,
        ["r0", "r1"],
        (2, 1),
        None,
    )
    drafts2 = mgr.take_draft_token_ids()
    assert isinstance(drafts2, DraftTokenIds)

    # 2. stage_draft_token_ids_for_host syncs previous copy if present
    mgr._staged_draft_copy = (None, event_sync, None, None, None)
    event_sync.reset_mock()
    mgr.stage_draft_token_ids_for_host(torch.zeros(2, 1, dtype=torch.int32))
    event_sync.synchronize.assert_called_once()

    # stage_draft_token_ids_for_host when _staged_draft_copy is None
    mgr._staged_draft_copy = None
    mgr.stage_draft_token_ids_for_host(torch.zeros(2, 1, dtype=torch.int32))

    # 3. get_spec_decode_metadata H2D fence on TPU vs non-TPU
    num_draft = np.array([1, 1], dtype=np.int32)
    cu_sched = np.array([0, 1, 2], dtype=np.int32)
    mgr._meta_cache_key = (
        num_draft,
        cu_sched,
        2,
        memoryview(num_draft),
        memoryview(cu_sched),
    )
    mgr._meta_cache = SimpleNamespace(
        draft_token_ids=torch.zeros(2, 2, dtype=torch.int32),
        draft_lengths=None,
        draft_lengths_cpu=None,
        target_logits_indices=None,
        bonus_logits_indices=None,
        final_logits_indices=None,
        segment_ids=None,
        group_indices=None,
    )
    mgr._meta_cache_np = (
        np.array([0, 1], dtype=np.int32),
        None,
        np.zeros(4, dtype=np.int32),
    )
    mgr.runner.input_ids_cpu = torch.zeros(4, dtype=torch.int32)
    mgr.runner.device = torch.device("cpu")

    h2d_fence = MagicMock()
    mgr._draft_token_ids_h2d_fence = h2d_fence

    # Non-TPU device: syncs existing fence
    mgr.get_spec_decode_metadata(
        num_draft,
        cu_sched,
        2,
        skip_host_draft_tokens=False,
    )
    h2d_fence.synchronize.assert_called_once()
    assert mgr._draft_token_ids_h2d_fence is h2d_fence

    # Non-TPU device without existing fence
    mgr._draft_token_ids_h2d_fence = None
    mgr.get_spec_decode_metadata(
        num_draft,
        cu_sched,
        2,
        skip_host_draft_tokens=False,
    )
    assert mgr._draft_token_ids_h2d_fence is None

    # skip_host_draft_tokens=True: reuses cached draft_token_ids
    res_skip = mgr.get_spec_decode_metadata(
        num_draft,
        cu_sched,
        2,
        skip_host_draft_tokens=True,
    )
    assert res_skip.draft_token_ids is mgr._meta_cache.draft_token_ids

    # TPU device: records fence
    tpu_event = MagicMock()
    mgr.runner.device = SimpleNamespace(type="tpu")
    with (
        patch.object(
            torch,
            "tpu",
            SimpleNamespace(Event=MagicMock(return_value=tpu_event)),
            create=True,
        ),
        patch.object(torch.Tensor, "to", return_value=torch.zeros(2, 2)),
    ):
        mgr.get_spec_decode_metadata(
            num_draft,
            cu_sched,
            2,
            skip_host_draft_tokens=False,
        )
        tpu_event.record.assert_called_once()
        assert mgr._draft_token_ids_h2d_fence is tpu_event

    # 4. Cache-miss packed_int32 staging and TPU event recording / sync
    prev_staging_event = MagicMock()
    mgr._packed_int32_h2d_staging = (prev_staging_event, None)
    mgr._meta_cache_key = None
    mgr.runner.arange_np = np.arange(128, dtype=np.int32)
    mgr.runner.num_tokens_paddings = [16, 32, 64]
    mgr.runner.input_ids_cpu = torch.zeros(128, dtype=torch.int32)
    mgr.runner.device = SimpleNamespace(type="tpu")

    tpu_event = MagicMock()
    with (
        patch.object(
            torch,
            "tpu",
            SimpleNamespace(Event=MagicMock(return_value=tpu_event)),
            create=True,
        ),
        patch.object(
            torch.Tensor, "to", return_value=torch.zeros(512, dtype=torch.int32)
        ),
    ):
        mgr.get_spec_decode_metadata(
            num_draft_tokens=np.array([1, 1], dtype=np.int32),
            cu_num_scheduled_tokens=np.array([2, 4], dtype=np.int32),
            padded_num_reqs=2,
        )
        prev_staging_event.synchronize.assert_called_once()
        tpu_event.record.assert_called_once()
        assert mgr._packed_int32_h2d_staging[0] is tpu_event

    # Cache miss on CPU device with prev_packed is None
    mgr._packed_int32_h2d_staging = None
    mgr._meta_cache_key = None
    mgr.runner.device = torch.device("cpu")
    mgr.get_spec_decode_metadata(
        num_draft_tokens=np.array([1, 1], dtype=np.int32),
        cu_num_scheduled_tokens=np.array([2, 4], dtype=np.int32),
        padded_num_reqs=2,
    )
    assert mgr._packed_int32_h2d_staging[0] is None


# ---------------------------------------------------------------------------
# 11. AttentionMetadata: Plan Cache Invalidation & Stage Block Table Fences
# ---------------------------------------------------------------------------


def test_attention_metadata_plan_cache_and_fences():
    # 1. _get_staged_walk_plan cache hit and invalidation
    plan_entry = (0, object())
    cached_plan = SimpleNamespace(entries=[plan_entry])
    builder = SimpleNamespace(
        kv_cache_group_id=0,
        _block_table_geometry=lambda ctx: (plan_entry[1], 4),
        is_mamba_group=False,
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[plan_entry[1]]),
        _staged_walk_plan_cache={},
        attn_groups=[[SimpleNamespace(metadata_builders=[builder])]],
        _unified_kv_layout=False,
    )
    ctx = SimpleNamespace(
        use_max_model_len=True,
        mamba_ckpt_window=1,
        num_tokens_paddings_key=(1,),
        mamba_state_indices=None,
    )

    # Compute key by running _get_staged_walk_plan with a mock
    with patch(
        "vllm_torchtpu.layers.core.attention_metadata._StagedWalkPlan",
        return_value=cached_plan,
    ):
        first_plan = _get_staged_walk_plan(runner, ctx, 2, None)
        assert first_plan is cached_plan

    # Second call hits cache (same block_table)
    hit_plan = _get_staged_walk_plan(runner, ctx, 2, None)
    assert hit_plan is cached_plan

    # Invalidate by replacing block_table
    runner.input_batch.block_table = [object()]
    with patch(
        "vllm_torchtpu.layers.core.attention_metadata._StagedWalkPlan"
    ) as mock_new_plan:
        _get_staged_walk_plan(runner, ctx, 2, None)
        mock_new_plan.assert_called_once()

    # 2. stage_block_table_uploads fences
    stage_fence_mock = MagicMock()
    runner_fence = SimpleNamespace(
        attn_groups=[object()],
        device=torch.device("cpu"),
        _block_table_stage_fence=stage_fence_mock,
        _block_table_delta_stager=None,
        _block_table_stage_cpu=None,
    )
    ctx_stage = SimpleNamespace(
        staged_block_tables=None,
        position_ids_override=object(),
        num_reqs=2,
        start_index=0,
    )
    mock_plan = SimpleNamespace(
        total=16,
        entries=[(0, object(), 8)],
        entry_builders=[],
        table_geom=[(0, 2, 8)],
        mamba_geom=[],
        is_uniform=True,
        is_single_group=True,
        first_group_id=0,
        first_shape=(2, 8),
        first_strides=(8, 1),
        first_elem_bytes=4,
        tracked_entries_key=None,
        delta_key_zero_start=None,
    )
    with patch(
        "vllm_torchtpu.layers.core.attention_metadata._get_staged_walk_plan",
        return_value=mock_plan,
    ):
        stage_block_table_uploads(runner_fence, ctx_stage, 2)
        stage_fence_mock.synchronize.assert_called_once()

    # TPU device stage fence record
    tpu_event = MagicMock()
    runner_fence._block_table_stage_fence = None
    runner_fence.device = SimpleNamespace(type="tpu")
    with (
        patch.object(
            torch,
            "tpu",
            SimpleNamespace(Event=MagicMock(return_value=tpu_event)),
            create=True,
        ),
        patch.object(
            torch.Tensor, "to", return_value=torch.zeros(16, dtype=torch.int32)
        ),
        patch(
            "vllm_torchtpu.layers.core.attention_metadata._get_staged_walk_plan",
            return_value=mock_plan,
        ),
    ):
        stage_block_table_uploads(runner_fence, ctx_stage, 2)
        tpu_event.record.assert_called_once()
        assert runner_fence._block_table_stage_fence is tpu_event


# ---------------------------------------------------------------------------
# 12. BlockTableDeltaStager: Fence Synchronization & TPU Record
# ---------------------------------------------------------------------------


def test_block_table_delta_stager_fence(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.layers.core.test_block_table_delta import (
        _Builder,
        _make_block_table,
        _make_runner,
        _stage,
    )

    monkeypatch.setenv("TPU_INCREMENTAL_BLOCK_TABLE_UPLOAD", "1")
    monkeypatch.setenv("TPU_VERIFY_INCREMENTAL_BLOCK_TABLE", "0")

    group_builders = [
        _Builder(0, _make_block_table(16), 16),
        _Builder(1, _make_block_table(16), 16),
    ]
    runner = _make_runner(group_builders)
    for b in group_builders:
        b.bt.append_row([1, 2], 0)
        b.bt.append_row([10], 1)

    # Initial stage (full upload)
    _stage(runner, 2, [8, 4], 2)
    stager = runner._block_table_delta_stager
    assert stager is not None
    assert stager._update_fence is None

    # 1. stage synchronizes existing fence
    fence_mock = MagicMock()
    stager._update_fence = fence_mock
    for b in group_builders:
        b.bt.append_row([3], 0)
    _stage(runner, 2, [12, 4], 2)
    fence_mock.synchronize.assert_called_once()
    assert stager._update_fence is None

    # 2. Delta update on TPU records fence
    tpu_event = MagicMock()
    layout = next(iter(stager._layouts.values()))
    dev_mock = MagicMock()
    dev_mock.device = SimpleNamespace(type="tpu")
    layout.dev = dev_mock

    for b in group_builders:
        b.bt.append_row([4], 0)

    with (
        patch.object(
            torch,
            "tpu",
            SimpleNamespace(Event=MagicMock(return_value=tpu_event)),
            create=True,
        ),
        patch.object(
            torch.Tensor, "to", return_value=torch.zeros(2, 8, dtype=torch.int32)
        ),
        patch(
            "vllm_torchtpu.layers.core.block_table_delta.apply_block_table_delta",
            return_value=dev_mock,
        ),
        patch(
            "vllm_torchtpu.layers.core.attention_metadata._staged_walk_products_compiled",
            return_value=[torch.zeros(1), torch.zeros(1)],
        ),
    ):
        _stage(runner, 2, [16, 4], 2)
        tpu_event.record.assert_called_once()
        assert stager._update_fence is tpu_event
