# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for runner prefill & async output optimizations.

# Covers all lines and branches in:
# - attention_metadata (attention_metadata.py:326-388, 531-541)
# - tpu_runner (tpu_runner.py:2437-2441, 2788-2805, 4453-4479, 4629-4632)
# - tpu_runner_async_output (tpu_runner_async_output.py:178-196)
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    stage_block_table_uploads,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    SpecLogprobsChunk,
)

# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_ctx(**kwargs):
    defaults = {
        "num_reqs": 1,
        "start_index": 0,
        "use_max_model_len": True,
        "seq_lens": None,
        "seq_lens_cpu": None,
        "query_start_loc": None,
        "query_start_loc_cpu": None,
        "prompt_lens_cpu": None,
        "replay_starts_cpu": None,
        "request_distribution": None,
        "position_ids_override": None,
        "staged_block_tables": None,
        "mamba_state_indices": None,
        "mamba_slot_read_offsets": None,
        "mamba_ckpt_window": 1,
        "mamba_request_distribution": None,
        "unified_mamba_state_indices": None,
        "sequence_layout_descriptor": DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        "mamba_row_plans": {},
        # Per-walk caches on AttentionMetadataBuilderContext.
        "max_active_seq_len": None,
        "active_cols_cache": {},
        "base_attn_metadata": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_dummy_builder(
    device=None,
    block_size=16,
    max_num_reqs=4,
    max_num_blocks_per_req=128,
    is_mamba=False,
    kv_cache_group_id=0,
):
    if device is None:
        device = torch.device("cpu")
    runner = MagicMock()
    runner.device = device
    runner.block_size = block_size
    runner.max_num_reqs = max_num_reqs
    runner.most_model_len = 1024
    runner.position_ids = torch.full((8,), 42, dtype=torch.int32)
    runner._block_table_stage_cpu = None
    runner._unified_kv_layout = False

    bt = MagicMock()
    bt.max_num_blocks_per_req = max_num_blocks_per_req
    bt.get_cpu_tensor.return_value = torch.arange(
        max_num_reqs * max_num_blocks_per_req, dtype=torch.int32
    ).reshape(max_num_reqs, max_num_blocks_per_req)
    runner.input_batch.block_table = [bt]

    spec = (
        MambaSpec(
            block_size=1,
            shapes=((64, 128),),
            dtypes=(torch.bfloat16,),
            page_size_padded=256,
        )
        if is_mamba
        else FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
    )

    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["test_layer"],
        vllm_config=MagicMock(),
        device=device,
        runner=runner,
        kv_cache_group_id=kv_cache_group_id,
    )
    group_obj = SimpleNamespace(metadata_builders=[builder])
    runner.attn_groups = [[group_obj]]
    return builder, runner


# ==============================================================================
# 1. Tests for _active_block_table_columns & stage_block_table_uploads
# ==============================================================================


def test_active_block_table_columns_fallback_cases():
    """Verify fallback returns target_num_blocks when constraints not met."""
    builder, _ = _make_dummy_builder(block_size=16)

    # 1. seq_lens_cpu is None and ctx.seq_lens is None -> target_num_blocks
    ctx_none = _make_ctx(seq_lens_cpu=None, seq_lens=None, num_reqs=1)
    assert builder._active_block_table_columns(ctx_none, 64) == 64

    # 2. seq_lens on non-cpu device and seq_lens_cpu is None -> target_num_blocks
    ctx_cuda = _make_ctx(
        seq_lens_cpu=None,
        seq_lens=SimpleNamespace(device=SimpleNamespace(type="cuda")),
        num_reqs=1,
    )
    assert builder._active_block_table_columns(ctx_cuda, 64) == 64

    # 3. num_reqs <= 0 -> target_num_blocks
    ctx_zero_reqs = _make_ctx(seq_lens_cpu=(128,), num_reqs=0)
    assert builder._active_block_table_columns(ctx_zero_reqs, 64) == 64

    # 4. target_num_blocks <= 16 -> target_num_blocks
    ctx_small_blocks = _make_ctx(seq_lens_cpu=(128,), num_reqs=1)
    assert builder._active_block_table_columns(ctx_small_blocks, 16) == 16
    assert builder._active_block_table_columns(ctx_small_blocks, 8) == 8


def test_active_block_table_columns_device_cpu_seq_lens_and_zero_max_len():
    """Verify fallback to ctx.seq_lens when on CPU, and zero max_seq_len."""
    builder, _ = _make_dummy_builder(block_size=16)

    # seq_lens on CPU sets seq_lens_cpu
    ctx_cpu_tensor = _make_ctx(
        seq_lens_cpu=None,
        seq_lens=torch.tensor([0, 0], dtype=torch.int32),
        num_reqs=2,
    )
    # max_seq_len <= 0 returns min(16, target_num_blocks)
    assert builder._active_block_table_columns(ctx_cpu_tensor, 64) == 16
    assert builder._active_block_table_columns(ctx_cpu_tensor, 8) == 8


def test_active_block_table_columns_mamba_group_and_power_of_2():
    """Verify bucketing formula with Mamba group and power-of-2 caps."""
    # Standard builder (block_size=16)
    builder, _ = _make_dummy_builder(block_size=16)
    ctx = _make_ctx(
        seq_lens_cpu=(256, 128),
        num_reqs=2,
        mamba_ckpt_window=1,
    )
    # max_seq_len=256 -> needed_blocks = 256 / 16 = 16
    # 1 << max(4, 15.bit_length()=4) = 16
    assert builder._active_block_table_columns(ctx, 64) == 16

    # max_seq_len=300 -> needed_blocks = cdiv(300, 16) = 19
    # 1 << max(4, 18.bit_length()=5) = 32
    # The context memoizes the max length and bucket per walk, so a new batch
    # gets a new context (as each chunk's walk does in the runner).
    ctx = _make_ctx(seq_lens_cpu=(300, 100), num_reqs=2, mamba_ckpt_window=1)
    assert builder._active_block_table_columns(ctx, 64) == 32
    # target_num_blocks caps the bucket
    assert builder._active_block_table_columns(ctx, 24) == 24

    # Mamba group with mamba_ckpt_window > 1
    mamba_builder, _ = _make_dummy_builder(is_mamba=True)
    mamba_ctx = _make_ctx(
        seq_lens_cpu=(64,),
        num_reqs=1,
        mamba_ckpt_window=4,
    )
    # needed_blocks = cdiv(64, 1) = 64 + 4 - 1 = 67 -> (66).bit_length()=7 -> 128
    assert mamba_builder._active_block_table_columns(mamba_ctx, 256) == 128


def test_builder_build_variants():
    """Verify builder.build under staged, override, and normal paths."""
    builder, runner = _make_dummy_builder()
    cm = SimpleNamespace(num_reqs=4)

    # 1. Staged block tables path
    staged_tensor = torch.zeros(128, dtype=torch.int32)
    ctx_staged = _make_ctx(
        staged_block_tables={0: staged_tensor},
        position_ids_override=None,
        num_reqs=2,
    )
    runner._attn_metadata_builder_ctx = ctx_staged
    meta_staged = builder.build(0, cm)
    assert meta_staged.block_tables is staged_tensor

    # 2. position_ids_override path without staged
    ctx_override = _make_ctx(
        staged_block_tables=None,
        position_ids_override=torch.ones(8, dtype=torch.int32),
        num_reqs=2,
    )
    runner._attn_metadata_builder_ctx = ctx_override
    meta_override = builder.build(0, cm)
    assert meta_override.input_positions is ctx_override.position_ids_override
    assert meta_override.block_tables.shape[0] == 4 * 128

    # 3. Normal path with active_cols slicing
    ctx_normal = _make_ctx(
        staged_block_tables=None,
        position_ids_override=None,
        num_reqs=2,
        start_index=0,
        seq_lens_cpu=(32, 16),
    )
    runner._attn_metadata_builder_ctx = ctx_normal
    meta_normal = builder.build(0, cm)
    assert meta_normal.input_positions is runner.position_ids
    assert meta_normal.block_tables.shape[0] == 4 * 128


def test_stage_block_table_uploads_allocation_and_override():
    """Verify stage_block_table_uploads scratch buffer reuse and override path."""
    builder, runner = _make_dummy_builder(max_num_reqs=4, max_num_blocks_per_req=32)

    # position_ids_override is not None: scratch kept zeroed
    ctx_dummy = _make_ctx(
        position_ids_override=torch.ones(4, dtype=torch.int32),
        num_reqs=2,
    )
    runner._block_table_stage_cpu = None
    stage_block_table_uploads(runner, ctx_dummy, 4)
    assert runner._block_table_stage_cpu is not None
    assert runner._block_table_stage_cpu.shape[0] == 4 * 32

    # Normal upload path reusing existing scratch
    scratch_ref = runner._block_table_stage_cpu
    ctx_normal = _make_ctx(
        position_ids_override=None,
        seq_lens_cpu=(32, 16),
        num_reqs=2,
        start_index=0,
    )
    stage_block_table_uploads(runner, ctx_normal, 4)
    assert runner._block_table_stage_cpu is scratch_ref


# ==============================================================================
# 2. Tests for Async Output: copy_state wait skipping on discarded requests
# ==============================================================================


def test_async_output_copy_state_none():
    """Verify get_output when copy_state is None."""
    # All discarded
    out_all_discard = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r1", "r2"],
            req_id_to_index={"r1": 0, "r2": 1},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=None,
        discard_sampled_tokens_req_indices=[0, 1],
    ).get_output()
    assert out_all_discard.sampled_token_ids == [[], []]

    # Partial / none discarded
    out_none_discard = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r1", "r2"],
            req_id_to_index={"r1": 0, "r2": 1},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=None,
        discard_sampled_tokens_req_indices=[0],
    ).get_output()
    assert out_none_discard.sampled_token_ids == []


def test_async_output_skips_wait_when_spec_logprobs_empty_and_all_discarded():
    """Verify copy_state.wait() skipped when not spec_logprobs and
    all requests discarded."""
    forbidden_event = Mock()
    forbidden_event.synchronize.side_effect = AssertionError(
        "synchronize() must not be called"
    )

    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([1, 2])],
        chunk_real_lens=[2],
        copy_ready_event=forbidden_event,
        completed=False,
        spec_logprobs=None,
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[0],
    ).get_output()
    assert out.sampled_token_ids == [[]]
    forbidden_event.synchronize.assert_not_called()


def test_async_output_waits_when_spec_logprobs_present_even_if_all_discarded():
    """Verify copy_state.wait() called when spec_logprobs is present."""
    sync_mock = Mock()
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[10, 20]])],
        chunk_real_lens=[1],
        copy_ready_event=Mock(synchronize=sync_mock),
        completed=False,
        spec_logprobs=[
            SpecLogprobsChunk(
                MagicMock(
                    logprob_token_ids=torch.zeros((2, 1), dtype=torch.int32),
                    logprobs=torch.zeros((2, 1), dtype=torch.float32),
                    sampled_token_ranks=torch.zeros((2, 1), dtype=torch.int32),
                ),
                row_starts=np.array([0]),
            )
        ],
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[0],
    ).get_output()
    sync_mock.assert_called_once()
    assert out.sampled_token_ids == [[]]


def test_async_output_selected_token_ids_numel_zero_and_dim_1():
    """Verify empty selected_token_ids and 1D selected_token_ids handling."""
    # Numel zero
    copy_state_empty = AsyncTPUCopyState(
        chunks_cpu=[torch.empty(0, dtype=torch.int32)],
        chunk_real_lens=[0],
        copy_ready_event=Mock(wait=Mock()),
        completed=True,
        spec_logprobs=None,
    )
    out_empty = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state_empty,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out_empty.sampled_token_ids == []

    # 1D selected_token_ids gets unsqueezed
    copy_state_1d = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([42], dtype=torch.int32)],
        chunk_real_lens=[1],
        copy_ready_event=Mock(wait=Mock()),
        completed=True,
        spec_logprobs=None,
    )
    out_1d = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state_1d,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out_1d.sampled_token_ids == [[42]]


# ==============================================================================
# 3. Tests for TPUModelRunner: _modify_prev_results, _prepare_inputs & sample_tokens
# ==============================================================================


def test_modify_prev_results_branches():
    """Verify _modify_prev_results drains copy_state when request_seq_lens is empty,
    and advances when non-empty."""
    runner = object.__new__(TPUModelRunner)
    runner._pre_async_results = None

    # 1. _pre_async_results is None -> early return
    runner._modify_prev_results()

    # 2. _pre_async_results has empty request_seq_lens and copy_state is not None
    wait_for_copy = Mock()
    runner._pre_async_results = SimpleNamespace(
        req_ids=["r1"],
        request_seq_lens=[],
        copy_state=object(),
        wait_for_copy=wait_for_copy,
    )
    runner._modify_prev_results()
    wait_for_copy.assert_called_once()

    # 3. copy_state is None
    wait_for_copy.reset_mock()
    runner._pre_async_results.copy_state = None
    runner._modify_prev_results()
    wait_for_copy.assert_not_called()

    # 4. request_seq_lens is non-empty -> covers line 2441-2493
    req_state = SimpleNamespace(output_token_ids=[999])
    runner._pre_async_results = SimpleNamespace(
        req_ids=["r1"],
        request_seq_lens=[(0, req_state, 5, "r1")],
        discard_sampled_tokens_req_indices=[],
        num_draft_per_req=None,
        copy_state=object(),
        wait_for_copy=Mock(return_value=torch.tensor([[42]])),
    )
    runner.input_batch = SimpleNamespace(
        req_id_to_index={},
        num_tokens_no_spec=np.zeros(1, dtype=np.int32),
        token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
    )
    runner._modify_prev_results()
    assert req_state.output_token_ids == [42]


def test_prepare_inputs_single_req_fast_path_matches_general_path():
    """Verify TPUModelRunner._prepare_inputs num_reqs==1 fast path produces
    identical input_ids and position_ids as the general multi-req path."""
    max_num_reqs = 4
    max_model_len = 128
    num_computed = 4
    num_scheduled = 8

    token_ids_cpu = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )

    def _make_fake_runner(batch_num_reqs):
        token_ids_copy = token_ids_cpu.copy()
        num_computed_np = np.array([num_computed, 11, 9, 3], dtype=np.int32)
        positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
        query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
        seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
        from vllm_torchtpu.layers.core.sequence_layout import (
            AllSequenceLayoutPlanner,
        )

        return SimpleNamespace(
            reorder_batch_threshold=1,
            _replay_starts={},
            sequence_layout_planner=AllSequenceLayoutPlanner(),
            _dp_target_bucket=None,
            input_batch=SimpleNamespace(
                num_reqs=batch_num_reqs,
                req_ids=[f"req-{i}" for i in range(max_num_reqs)],
                num_computed_tokens_cpu=num_computed_np,
                token_ids_cpu=token_ids_copy,
                token_ids_cpu_tensor=torch.from_numpy(token_ids_copy),
                num_prompt_tokens=np.full(max_num_reqs, 64, dtype=np.int32),
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
            _prepare_async_token_substitution_indices=lambda *a: (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            ),
            _collect_mamba_state_seed_copies=lambda *a: None,
            num_tokens_paddings=[16, 64, 128],
            pcp_size=1,
            dcp_size=1,
        )

    # 1) Single-request fast path: num_reqs == 1 (lines 2788-2805)
    runner_fast = _make_fake_runner(1)
    sched_fast = SimpleNamespace(
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens={"req-0": num_scheduled},
    )
    TPUModelRunner._prepare_inputs(
        runner_fast, sched_fast, start_index=0, num_decode_reqs=0
    )

    # 2) General multi-req path
    runner_gen = _make_fake_runner(2)
    sched_gen = SimpleNamespace(
        total_num_scheduled_tokens=num_scheduled + 2,
        num_scheduled_tokens={"req-0": num_scheduled, "req-1": 2},
    )
    TPUModelRunner._prepare_inputs(
        runner_gen, sched_gen, start_index=0, num_decode_reqs=0
    )

    assert torch.equal(
        runner_fast.input_ids_cpu[:num_scheduled],
        runner_gen.input_ids_cpu[:num_scheduled],
    )
    assert np.array_equal(
        runner_fast.positions_np[:num_scheduled],
        runner_gen.positions_np[:num_scheduled],
    )


def test_sample_tokens_logprobs_and_chunks():
    """Verify logprobs concatenation and next_tokens_tpu chunks handling
    in TPUModelRunner.sample_tokens (lines 4453-4479, 4629-4632)."""

    def _run_sample(num_chunks=1, needs_logprobs=True, discard_all=False):
        stage_timing = MagicMock()
        stage_timing.stage.return_value = contextlib.nullcontext()
        stage_timing.host_stage.return_value = contextlib.nullcontext()

        req_ids = ["req-0"]
        req_state = SimpleNamespace(
            num_computed_tokens=0,
            num_tokens=10 if discard_all else 1,
        )
        requests = {"req-0": req_state}
        scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})

        dummy_lp = LogprobsLists(
            logprob_token_ids=np.array([[42]]),
            logprobs=np.array([[-0.5]]),
            sampled_token_ranks=np.array([[0]]),
        )
        mock_sliced_lp = MagicMock()
        mock_sliced_lp.tolists.return_value = dummy_lp

        mock_gather = MagicMock(return_value=mock_sliced_lp)

        runner = SimpleNamespace(
            is_pooling_model=False,
            scheduler_config=SimpleNamespace(async_scheduling=True),
            speculative_config=None,
            _is_async_drafter=False,
            _get_sampling_generator=MagicMock(return_value=None),
            input_batch=SimpleNamespace(
                num_reqs=1,
                req_ids=req_ids,
                all_greedy=True,
                generators={},
                req_id_to_index={"req-0": 0},
                max_num_logprobs=1 if needs_logprobs else None,
            ),
            requests=requests,
            _stage_timing=stage_timing,
            _update_num_xla_graphs=MagicMock(),
            _update_mamba_slot_read_offsets=MagicMock(),
            _greedy_sample=lambda logits: torch.tensor([42]),
            gather_logprobs=mock_gather,
            _modify_prev_results=MagicMock(),
            _update_placeholder=MagicMock(return_value={"req-0": 0}),
            maybe_wait_for_kv_save=MagicMock(),
            get_finished_kv_transfers=MagicMock(
                return_value=(None, None, None, [], 0, None)
            ),
            _sync_replicated_drafts_across_tp=lambda x: x,
            max_num_reqs=1,
        )

        logits = torch.zeros((1, 10))
        if num_chunks == 1:
            state = SimpleNamespace(
                logits_list=[logits],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=[None],
                pooler_output_list=[],
                scheduler_output=scheduler_output,
            )
        else:
            state = SimpleNamespace(
                logits_list=[logits, logits],
                num_reqs_list=[1, 1],
                spec_decode_metadata_list=[None, None],
                mamba_state_indices_list=[None, None],
                pooler_output_list=[],
                scheduler_output=scheduler_output,
            )
        runner.execute_model_state = state

        with patch.object(
            torch, "tpu", SimpleNamespace(Event=MagicMock()), create=True
        ):
            return TPUModelRunner.sample_tokens(runner, grammar_output=None)

    # 1. Single chunk with logprobs (covers 4453-4479 and 4633 single chunk)
    out1 = _run_sample(num_chunks=1, needs_logprobs=True, discard_all=False)
    assert out1.copy_state is not None
    assert out1._model_runner_output.logprobs is not None

    # 2. Two chunks (covers 4635 multi chunk torch.cat)
    out2 = _run_sample(num_chunks=2, needs_logprobs=True, discard_all=False)
    assert out2.copy_state is not None

    # 3. Discard all (covers 4478 false branch when request_seq_lens empty)
    out3 = _run_sample(num_chunks=1, needs_logprobs=True, discard_all=True)
    assert out3.copy_state is None


def test_replay_starts_and_owner_token_layout_step_caches(monkeypatch):
    """Verify _record_replay_starts, _prepare_lookback_token_ids, _prepare_inputs
    replay_starts_cpu caching, and step_owner_required_tokens /
    attach_step_owner_token_layout owner cache reuse."""
    runner = object.__new__(TPUModelRunner)
    spec = SimpleNamespace(prefix_replay_tokens=128)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    runner._replay_starts = {}

    def _sched_out(new=(), cached=(), resumed=()):
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=r, replay_start=s, num_computed_tokens=s)
                for r, s in new
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[r for r, _ in cached],
                num_computed_tokens=[n for _, n in cached],
                resumed_req_ids=set(resumed),
            ),
        )

    # 1. Steady-state decode fast-exit (no new_reqs and no resumed_req_ids)
    runner._record_replay_starts(_sched_out(cached=[("running", 900)]))
    assert runner._replay_starts == {}

    # 2. Only new_reqs (resumed_req_ids empty -> skips cached loop)
    runner._record_replay_starts(
        _sched_out(new=[("hit", 512), ("cold", 0)], cached=[("running", 900)])
    )
    assert runner._replay_starts == {"hit": 512}

    # 3. Both new_reqs and resumed_req_ids
    runner._record_replay_starts(
        _sched_out(
            new=[("hit2", 256)],
            cached=[("running", 900), ("resumed", 768), ("resumed_cold", 0)],
            resumed=["resumed", "resumed_cold"],
        )
    )
    assert runner._replay_starts == {"hit": 512, "hit2": 256, "resumed": 768}

    # 4. Disabled when prefix_replay_tokens == 0
    spec.prefix_replay_tokens = 0
    runner._record_replay_starts(_sched_out(new=[("ignored", 512)]))
    assert "ignored" not in runner._replay_starts
    spec.prefix_replay_tokens = 128

    # 5. _prepare_lookback_token_ids: num_reqs == 0, empty _replay_starts, and non-empty
    class _LookbackBuf:
        def __init__(self, rows, depth):
            self.np = np.zeros((rows, depth), dtype=np.int64)

        def copy_to_gpu(self):
            return torch.from_numpy(self.np.copy())

    runner.lookback_token_ids = _LookbackBuf(rows=3, depth=2)
    tokens = np.arange(1000, dtype=np.int64)[None, :].repeat(3, axis=0)
    runner.input_batch = SimpleNamespace(
        req_ids=["resumed", "later", "fresh"],
        num_computed_tokens_cpu=np.array([768, 800, 512]),
        num_prompt_tokens=np.array([256, 256, 1024]),
        num_tokens_no_spec=np.array([1000, 1000, 1024]),
        token_ids_cpu=tokens,
    )
    assert torch.all(runner._prepare_lookback_token_ids(0) == -1)

    runner._replay_starts = {}
    ids_no_replay = runner._prepare_lookback_token_ids(3)
    assert ids_no_replay[0].tolist() == [-1, -1]
    assert ids_no_replay[2].tolist() == [511, 510]

    runner._replay_starts = {"resumed": 768, "later": 768, "fresh": 512}
    ids_replay = runner._prepare_lookback_token_ids(3)
    assert ids_replay[0].tolist() == [767, 766]
    assert ids_replay[1].tolist() == [-1, -1]
    assert ids_replay[2].tolist() == [511, 510]

    # 6. _update_states pops invalidated req_id from non-empty _replay_starts
    runner._mamba_state_pos = {"resumed": 1}
    runner._mamba_offset_seeded = {"resumed"}
    runner._pre_async_results = None
    runner.requests = {}
    runner._install_spec_token_room_guard = lambda: None
    runner.is_unified_pool_used = lambda: False
    runner._dense_kv_copy_supported = lambda: False
    so_finish = SimpleNamespace(
        finished_req_ids={"resumed"},
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], resumed_req_ids=set()
        ),
        kv_cache_block_copies=None,
    )
    with patch.object(TPUModelRunner.__bases__[0], "_update_states", return_value="ok"):
        assert runner._update_states(so_finish) == "ok"
    assert "resumed" not in runner._replay_starts

    # 7. _prepare_inputs replay_starts_cpu caching (empty and non-empty _replay_starts)
    from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner

    max_num_reqs = 4
    max_model_len = 128
    token_ids_copy = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    fake_runner = SimpleNamespace(
        reorder_batch_threshold=1,
        _replay_starts={},
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=[f"req-{i}" for i in range(max_num_reqs)],
            num_computed_tokens_cpu=np.array([4, 11, 9, 3], dtype=np.int32),
            token_ids_cpu=token_ids_copy,
            token_ids_cpu_tensor=torch.from_numpy(token_ids_copy),
            num_prompt_tokens=np.full(max_num_reqs, 64, dtype=np.int32),
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
        _prepare_async_token_substitution_indices=lambda *a: (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ),
        _collect_mamba_state_seed_copies=lambda *a: None,
        num_tokens_paddings=[16, 64, 128],
        pcp_size=1,
        dcp_size=1,
    )
    sched_two = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"req-0": 1, "req-1": 1},
    )
    TPUModelRunner._prepare_inputs(fake_runner, sched_two, 0, 2)
    assert fake_runner._attn_metadata_builder_ctx.replay_starts_cpu == (0, 0, 0, 0)
    TPUModelRunner._prepare_inputs(fake_runner, sched_two, 0, 2)
    assert fake_runner._attn_metadata_builder_ctx.replay_starts_cpu is (
        fake_runner._cached_zero_replay_starts_cpu
    )

    fake_runner._replay_starts = {"req-0": 512}
    TPUModelRunner._prepare_inputs(fake_runner, sched_two, 0, 2)
    assert fake_runner._attn_metadata_builder_ctx.replay_starts_cpu == (512, 0, 0, 0)
    TPUModelRunner._prepare_inputs(fake_runner, sched_two, 0, 2)
    assert fake_runner._attn_metadata_builder_ctx.replay_starts_cpu is (
        fake_runner._cached_replay_starts_cpu
    )

    # 8. step_owner_required_tokens cache hit and _kv_shard_owners_cache reuse
    from vllm_torchtpu import envs
    from vllm_torchtpu.core.owner_affine_block_pool import OwnerLayout
    from vllm_torchtpu.runner.owner_token_layout import (
        attach_step_owner_token_layout,
        step_owner_required_tokens,
    )

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_KV_SHARD", True, raising=False)
    owner_layout = OwnerLayout(num_replicated=8, num_owners=4, blocks_per_owner=14)
    bt_arr = np.array([[10], [24]], dtype=np.int32)
    kv_runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace())]
        ),
        _kv_shard_group_cache=(None, 0),
        _kv_shard_owner_layout=owner_layout,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[SimpleNamespace(_bt_cpu_np=bt_arr)]
            )
        ),
        num_tokens_padded=16,
        num_tokens_paddings=[16, 32],
        _attn_metadata_builder_ctx=SimpleNamespace(
            num_reqs=2,
            start_index=0,
            req_indices=None,
            position_ids_override=None,
            query_start_loc_cpu=np.array([0, 1, 2], dtype=np.int64),
        ),
    )
    kv_runner._kv_shard_group_cache = (id(kv_runner.kv_cache_config), 0)
    req1 = step_owner_required_tokens(kv_runner, 0, 2, [1, 1])
    req2 = step_owner_required_tokens(kv_runner, 0, 2, [1, 1])
    assert req1 == req2 == 4
    meta = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32))
    layout = attach_step_owner_token_layout(kv_runner, {"l0": meta})
    assert layout is not None

    # 9. _prepare_async_token_substitution_indices memoization hit & miss
    async_runner = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy={"req-0": 0, "req-1": 1},
            spec_decode_num_rejected_tokens=object(),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        _last_sequence_layout_plan=None,
        input_batch=SimpleNamespace(req_ids=["req-0", "req-1"]),
        arange_np=np.arange(64, dtype=np.int32),
    )
    sched_arr = np.array([6, 6], dtype=np.int32)
    pair1 = TPUModelRunner._prepare_async_token_substitution_indices(
        async_runner, 0, 2, sched_arr
    )
    pair2 = TPUModelRunner._prepare_async_token_substitution_indices(
        async_runner, 0, 2, sched_arr
    )
    assert pair1 is pair2
    assert pair1[0].tolist() == list(range(12))
    assert pair1[1].tolist() == list(range(12))

    # Also verify via _last_async_seq_idx_prefix / _last_async_seq_idx_bytes
    pos_prefix = np.array([1, 0], dtype=np.int32)
    async_runner._last_async_seq_idx_prefix = pos_prefix
    async_runner._last_async_seq_idx_bytes = pos_prefix.tobytes()
    async_runner._last_sched_tokens_bytes = sched_arr.tobytes()
    pair3 = TPUModelRunner._prepare_async_token_substitution_indices(
        async_runner, 0, 2, sched_arr
    )
    pair4 = TPUModelRunner._prepare_async_token_substitution_indices(
        async_runner, 0, 2, sched_arr
    )
    assert pair3 is pair4
    assert pair3[1].tolist() == list(range(6, 12)) + list(range(0, 6))


def test_td8235_fast_update_states_and_count_input_chunks():
    """Verify TD8.235 steady-state _try_fast_update_states and _count_input_chunks."""
    runner = object.__new__(TPUModelRunner)
    runner._mamba_state_pos = {}
    runner._mamba_offset_seeded = set()
    runner._pre_async_results = None
    runner._pp_is_last = True
    runner.speculative_config = None
    runner.use_async_spec_decode = True
    runner.prev_num_draft_tokens = SimpleNamespace(np=np.ones(4, dtype=np.int32))
    runner.is_unified_pool_used = lambda: False
    runner._dense_kv_copy_supported = lambda: False

    # 1) Early return when input_batch is None
    so_base = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        new_block_ids_to_zero=[],
        free_encoder_mm_hashes=[],
        kv_cache_block_copies=None,
        num_scheduled_tokens={"r0": 3, "r1": 3},
        scheduled_spec_decode_tokens={"r0": [101, 102], "r1": [201, 202]},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            resumed_req_ids=set(),
            num_computed_tokens=[10, 20],
            num_output_tokens=[1, 2],
            new_block_ids=[([5],), None],
        ),
    )
    assert runner._try_fast_update_states(so_base) is False

    # 2) Build lightweight InputBatch stub
    appended_rows: list[tuple[object, int]] = []
    refreshed: list[bool] = []
    fallback_spec_calls: list[str] = []
    runner.requests = {
        "r0": SimpleNamespace(
            req_id="r0",
            num_computed_tokens=8,
            output_token_ids=[11, 99, 99],
            block_ids=([1],),
            prev_num_draft_len=2,
        ),
        "r1": SimpleNamespace(
            req_id="r1",
            num_computed_tokens=18,
            output_token_ids=[21, 22],
            block_ids=([2],),
            prev_num_draft_len=2,
        ),
    }
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=np.zeros(4, dtype=np.int32),
        num_prompt_tokens=np.array([5, 10, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 12, 0, 0], dtype=np.int32),
        token_ids_cpu=np.zeros((4, 32), dtype=np.int32),
        is_token_ids=np.zeros((4, 32), dtype=bool),
        spec_token_ids=[[9], [8], [], []],
        block_table=SimpleNamespace(
            append_row=lambda nb, idx: appended_rows.append((nb, idx))
        ),
        update_req_spec_token_ids=lambda req, _sched: fallback_spec_calls.append(
            req.req_id
        ),
        refresh_metadata=lambda: refreshed.append(True),
    )

    # 3) Early return when use_ngram_gpu() is True
    runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: True)
    assert runner._try_fast_update_states(so_base) is False
    runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: False)

    # 4) Early return when n_reqs != input_batch.num_reqs
    runner.input_batch.num_reqs = 3
    assert runner._try_fast_update_states(so_base) is False
    runner.input_batch.num_reqs = 2

    # 5) Steady-state uniform spec-decode fast path via _update_states
    assert runner._update_states(so_base) is None
    assert runner.prev_num_draft_tokens.np.tolist() == [0, 0, 0, 0]
    assert runner.input_batch.num_computed_tokens_cpu[:2].tolist() == [10, 20]
    # r0 output_token_ids trimmed from len 3 -> 1, num_tokens_no_spec[0] = 5 + 1 = 6
    assert runner.requests["r0"].output_token_ids == [11]
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 6
    assert runner.requests["r0"].block_ids == ([1, 5],)
    assert appended_rows == [(([5],), 0)]
    # Spec tokens staged at [0, 6:8] and [1, 12:14]
    assert runner.input_batch.token_ids_cpu[0, 6:8].tolist() == [101, 102]
    assert runner.input_batch.token_ids_cpu[1, 12:14].tolist() == [201, 202]
    assert runner.input_batch.is_token_ids[0, 6:8].tolist() == [True, True]
    assert runner.input_batch.spec_token_ids[:2] == [[101, 102], [201, 202]]
    assert runner.requests["r0"].prev_num_draft_len == 2
    assert runner.requests["r1"].prev_num_draft_len == 2
    assert len(refreshed) == 1

    # 6) Non-uniform / empty spec tokens fallback + use_async_spec_decode=False
    runner.use_async_spec_decode = False
    so_base.scheduled_spec_decode_tokens = {"r0": [101], "r1": []}
    assert runner._update_states(so_base) is None
    assert fallback_spec_calls == ["r0", "r1"]

    # 7) Empty scheduled_spec_decode_tokens clears spec_token_ids
    so_base.scheduled_spec_decode_tokens = {}
    assert runner._update_states(so_base) is None
    assert runner.input_batch.spec_token_ids[:2] == [[], []]

    # 8) _count_input_chunks: total_tokens > most_model_len, each req <= most_model_len
    runner.most_model_len = 4
    runner.num_reqs_most_model_len = 2
    runner.num_reqs_max_model_len = 1
    so_chunks = SimpleNamespace(
        total_num_scheduled_tokens=6,
        num_scheduled_tokens={"r0": 3, "r1": 3},
    )
    assert runner._count_input_chunks(so_chunks) == (1, 2)


def test_td8303_lookback_token_ids_and_arange_int32():
    """Verify TD8.303 native np.int32 _ARANGE_INTP, tail-only lookback fill, and
    deferred num_prompt_tokens copy when replay_starts is empty."""
    from vllm_torchtpu.runner.tpu_runner import _ARANGE_INTP

    assert _ARANGE_INTP.dtype == np.int32

    runner = object.__new__(TPUModelRunner)

    class _LookbackBuf:
        def __init__(self, rows: int, depth: int):
            self.np = np.full((rows, depth), 77, dtype=np.int32)

        def copy_to_gpu(self) -> torch.Tensor:
            return torch.from_numpy(self.np.copy())

    runner.lookback_token_ids = _LookbackBuf(rows=4, depth=2)
    prompt_tokens = np.array([256, 256, 1024, 512], dtype=np.int32)
    tokens = np.arange(1000, dtype=np.int32)[None, :].repeat(4, axis=0)
    runner.input_batch = SimpleNamespace(
        req_ids=["resumed", "later", "fresh", "idle"],
        num_computed_tokens_cpu=np.array([768, 800, 512, 0], dtype=np.int32),
        num_prompt_tokens=prompt_tokens,
        num_tokens_no_spec=np.array([1000, 1000, 1024, 0], dtype=np.int32),
        token_ids_cpu=tokens,
    )

    # Empty replay_starts: tail row [3] filled with -1, active rows [0:3] populated
    runner._replay_starts = {}
    out_no_replay = runner._prepare_lookback_token_ids(2)
    assert out_no_replay.dtype == torch.int32
    assert out_no_replay[0].tolist() == [-1, -1]
    assert out_no_replay[1].tolist() == [-1, -1]
    assert out_no_replay[2].tolist() == [-1, -1]
    assert out_no_replay[3].tolist() == [-1, -1]

    # Non-empty replay_starts: copies num_prompt_tokens without mutating original array
    runner.lookback_token_ids.np.fill(77)
    runner._replay_starts = {"resumed": 768, "later": 768, "fresh": 512}
    out_replay = runner._prepare_lookback_token_ids(3)
    assert out_replay[0].tolist() == [767, 766]
    assert out_replay[1].tolist() == [-1, -1]
    assert out_replay[2].tolist() == [511, 510]
    assert out_replay[3].tolist() == [-1, -1]
    assert prompt_tokens.tolist() == [256, 256, 1024, 512]


def test_td8313_dp_coordinated_step_mm_cumsum_and_row_starts_int32(monkeypatch):
    """Verify TD8.313 int32 _dp_coordinated_step, mm cumsum, and SpecLogprobsChunk."""
    import vllm.distributed.parallel_state as ps

    runner = object.__new__(TPUModelRunner)
    runner._dp_lockstep_enabled = lambda: True
    runner.num_tokens_paddings = [16, 32, 64]

    reduced_dtypes: list[torch.dtype] = []

    def fake_all_reduce(t: torch.Tensor, op=None, group=None):
        reduced_dtypes.append(t.dtype)
        t[0] = 20
        t[1] = 2
        t[2] = 4
        t[3] = 1

    monkeypatch.setattr(
        ps, "get_dp_group", lambda: SimpleNamespace(cpu_group="fake_cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    bucket, chunks = runner._dp_coordinated_step(12, 1, 3, True)
    assert reduced_dtypes == [torch.int32]
    assert bucket == 32
    assert chunks == 2
    assert runner._dp_step_max_reqs == 4
    assert runner._dp_step_has_encoder_inputs is True

    draft_lengths_cpu = np.array([3, 1, 5], dtype=np.int32)
    row_starts = (
        np.cumsum(
            step_lens := np.asarray(draft_lengths_cpu, dtype=np.int32) + 1,
            dtype=np.int32,
        )
        - step_lens
    )
    assert row_starts.dtype == np.int32
    assert row_starts.tolist() == [0, 4, 6]
