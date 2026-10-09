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
"""Unit tests for TD7.48 optimizations and 100% full-module coverage of
`src/vllm_torchtpu/runner/tpu_runner_async_output.py` and
`src/vllm_torchtpu/runner/utils.py`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner import tpu_runner_async_output as async_out
from vllm_torchtpu.runner import utils as rutils
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
from vllm_torchtpu.runner.utils import (
    InferencePhase,
    PhaseBasedProfiler,
    determine_phase_from_batch_composition_stats,
    get_batch_composition_stats,
    relayout_hash_tables,
    relayout_rope_caches,
    relayout_rope_tables,
)


class _FakeTPUTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _as_fake_tpu(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_FakeTPUTensor)


def _row_logprobs(num_rows, base=0):
    rows = torch.arange(num_rows, dtype=torch.int32) + base
    return LogprobsTensors(
        logprob_token_ids=torch.stack([rows, rows + 1000], dim=1),
        logprobs=torch.stack([-rows.float(), -rows.float() - 0.5], dim=1),
        selected_token_ranks=rows.clone(),
    )


def test_async_output_parse_and_logprobs_full():
    # 1D, 2D col=1, 2D multi-col with and without INVALID_TOKEN_ID
    r1, l1 = parse_sampled_token_ids_cpu(torch.tensor([5, 6], dtype=torch.int32))
    assert r1 == [[5], [6]] and l1 == [1, 1]

    r2, l2 = parse_sampled_token_ids_cpu(torch.tensor([[7], [8]], dtype=torch.int32))
    assert r2 == [[7], [8]] and l2 == [1, 1]

    r3, l3 = parse_sampled_token_ids_cpu(
        torch.tensor([[10, INVALID_TOKEN_ID], [20, 21]], dtype=torch.int32)
    )
    assert r3 == [[10], [20, 21]] and l3 == [1, 2]

    # spec_logprobs_lists: non-empty and empty chunk
    c1 = SpecLogprobsChunk(_row_logprobs(4), row_starts=np.array([0, 2]))
    c_empty = SpecLogprobsChunk(
        _row_logprobs(2), row_starts=np.zeros(0, dtype=np.int64)
    )
    out = spec_logprobs_lists([c1, c_empty], gen_lens=[2, 1])
    assert out.cu_num_generated_tokens == [0, 2, 3]
    assert out.logprob_token_ids.shape[0] == 3


def test_async_tpu_copy_state_and_runner_output_all_branches():
    fake_event = MagicMock()
    t_1d = torch.tensor([10, 11], dtype=torch.int32)
    t_2d = torch.tensor([[10, 11], [20, INVALID_TOKEN_ID]], dtype=torch.int32)
    lp_chunk = SpecLogprobsChunk(_row_logprobs(4), row_starts=np.array([0]))

    with patch.object(
        torch,
        "tpu",
        SimpleNamespace(Event=MagicMock(return_value=fake_event)),
        create=True,
    ):
        # Single-chunk 1D (t.ndim == 1 -> no reshape needed on sampled_token_ids_cpu)
        s_1d = AsyncTPUCopyState.from_device_chunks([t_1d], [1], spec_logprobs=None)
        assert s_1d.sampled_token_ids_cpu.tolist() == [10]

        # Single-chunk 2D (t.ndim > 1 -> flattened to 1D then reshaped on CPU)
        s_2d = AsyncTPUCopyState.from_device_chunks(
            [t_2d], [1], spec_logprobs=[lp_chunk]
        )
        assert s_2d.sampled_token_ids_cpu.tolist() == [[10, 11]]

        # Multi-chunk (1D and 2D mixed) with chunk_shapes
        s_multi = AsyncTPUCopyState.from_device_chunks(
            [t_2d, t_2d], [1, 1], spec_logprobs=None
        )
        assert s_multi.sampled_token_ids_cpu.tolist() == [[10, 11], [10, 11]]

        # Multi-chunk with 1D tensors (t.shape == s branch in comprehension)
        s_multi_1d = AsyncTPUCopyState.from_device_chunks([t_1d, t_1d], [1, 1])
        assert s_multi_1d.sampled_token_ids_cpu.tolist() == [10, 10]

    # Multi-chunk with chunk_shapes=None
    s_no_shapes = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[1], [2]]), torch.tensor([[3], [4]])],
        chunk_real_lens=[1, 1],
        copy_ready_event=fake_event,
        chunk_shapes=None,
        completed=True,
    )
    assert s_no_shapes.sampled_token_ids_cpu.tolist() == [[1], [3]]

    # Single-chunk with chunk_shapes=None
    s_single_no_shapes = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[5], [6]])],
        chunk_real_lens=[1],
        copy_ready_event=fake_event,
        chunk_shapes=None,
        completed=True,
    )
    assert s_single_no_shapes.sampled_token_ids_cpu.tolist() == [[5]]
    assert s_single_no_shapes.sampled_token_ids_cpu.tolist() == [[5]]

    # wait() and parse_sampled_tokens(copy_rows=True / False)
    s_2d.wait()
    s_2d.wait()
    rows_copy, lens_copy = s_2d.parse_sampled_tokens(copy_rows=True)
    rows_nocopy, _ = s_2d.parse_sampled_tokens(copy_rows=False)
    assert rows_copy == [[10, 11]] and rows_nocopy == [[10, 11]] and lens_copy == [2]

    # AsyncPreResults.wait_for_copy
    pre_none = AsyncPreResults(
        req_ids=[],
        next_tokens_tpu=t_1d,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={},
        copy_state=None,
    )
    assert pre_none.wait_for_copy() is None
    pre_some = AsyncPreResults(
        req_ids=["r0"],
        next_tokens_tpu=t_1d,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"r0": 0},
        copy_state=s_1d,
    )
    assert pre_some.wait_for_copy().tolist() == [10]

    # AsyncTPUModelRunnerOutput.get_output branches
    for copy_state, req_ids, discard, expected in [
        (None, ["r0", "r1"], [0, 1], [[], []]),
        (None, ["r0"], [], []),
        (s_multi, ["r0", "r1"], [0, 1], [[], []]),
        (
            AsyncTPUCopyState(
                chunks_cpu=[torch.empty((0, 1), dtype=torch.int32)],
                chunk_real_lens=[0],
                copy_ready_event=fake_event,
                completed=True,
            ),
            ["r0"],
            [],
            [],
        ),
        (s_multi, ["r0", "r1"], [], [[10, 11], [10, 11]]),
        (s_2d, ["r0"], [0], [[]]),
    ]:
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[],
            logprobs=None,
        )
        out = AsyncTPUModelRunnerOutput(mro, copy_state, discard).get_output()
        assert out.sampled_token_ids == expected


def test_spec_token_assembly_and_rejection_all_branches(monkeypatch):
    # Exact shape vs padded shape, matching dtype vs different dtype (int64 vs int32)
    nt_exact = torch.tensor(
        [[11, 99, INVALID_TOKEN_ID], [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID]],
        dtype=torch.int32,
    )
    drafts_exact = torch.tensor([[201, 202], [211, 212]], dtype=torch.int32)
    assert assemble_spec_next_tokens(nt_exact, drafts_exact, num_reqs=2).tolist() == [
        99,
        201,
        202,
        77,
        211,
        212,
    ]

    # Padded rows + mismatched dtype (nt is int64, drafts is int32)
    nt_padded_i64 = torch.tensor(
        [
            [11, 99, INVALID_TOKEN_ID],
            [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    drafts_padded = torch.tensor([[201, 202], [211, 212], [0, 0]], dtype=torch.int32)
    assert assemble_spec_next_tokens(
        nt_padded_i64, drafts_padded, num_reqs=2
    ).tolist() == [99, 201, 202, 77, 211, 212]

    # _assemble_spec_next_tokens_and_num_rejected:
    # 1. Exact shapes, int32 num_draft, padded_src_len > src, padded_num_reqs > num_reqs
    nd_exact = torch.tensor([2, 2], dtype=torch.int32)
    src1, rej1 = assemble_spec_next_tokens_and_num_rejected(
        nt_exact,
        drafts_exact,
        nd_exact,
        num_reqs=2,
        padded_src_len=8,
        padded_num_reqs=3,
    )
    assert src1.tolist() == [99, 201, 202, 77, 211, 212, 0, 0]
    assert rej1.tolist() == [1, 2, 0]

    # 2. Padded shapes, int64 nt/num_draft, exact padded_src_len / padded_num_reqs
    nd_padded_i64 = torch.tensor([2, 2, 2], dtype=torch.int64)
    src2, rej2 = assemble_spec_next_tokens_and_num_rejected(
        nt_padded_i64,
        drafts_padded,
        nd_padded_i64,
        num_reqs=2,
        padded_src_len=6,
        padded_num_reqs=2,
    )
    assert src2.tolist() == [99, 201, 202, 77, 211, 212]
    assert rej2.dtype == torch.int32 and rej2.tolist() == [1, 2]

    # 3. num_draft=None (both padded_num_reqs > num_reqs and None)
    _, rej_none1 = assemble_spec_next_tokens_and_num_rejected(
        nt_exact, drafts_exact, None, num_reqs=2, padded_num_reqs=4
    )
    assert rej_none1.tolist() == [0, 0, 0, 0]
    _, rej_none2 = assemble_spec_next_tokens_and_num_rejected(
        nt_exact, drafts_exact, None, num_reqs=2, padded_num_reqs=None
    )
    assert rej_none2.tolist() == [0, 0]

    # 4. TPU dispatch branch vs unbucketed fallback
    monkeypatch.setattr(
        async_out,
        "_assemble_spec_next_tokens_and_num_rejected_compiled",
        async_out._assemble_spec_next_tokens_and_num_rejected,
    )
    src_tpu, rej_tpu = assemble_spec_next_tokens_and_num_rejected(
        _as_fake_tpu(nt_exact), _as_fake_tpu(drafts_exact), nd_exact, num_reqs=2
    )
    assert src_tpu.tolist() == [99, 201, 202, 77, 211, 212]
    assert rej_tpu.tolist() == [1, 2]

    # subtract_num_rejected_tokens (1D CPU and 2D TPU)
    monkeypatch.setattr(
        async_out,
        "_subtract_num_rejected_tokens_compiled",
        async_out._subtract_num_rejected_tokens,
    )
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    pos_1d = torch.tensor([5, 6], dtype=torch.int32)
    pos_2d = torch.tensor([[5, 6], [5, 6], [5, 6]], dtype=torch.int32)
    rej = torch.tensor([1, 2], dtype=torch.int32)
    idx = torch.tensor([0, 1], dtype=torch.int32)
    s_cpu, p_cpu = subtract_num_rejected_tokens(seq_lens, pos_1d, rej, idx, idx)
    assert s_cpu.tolist() == [9, 18] and p_cpu.tolist() == [4, 4]
    s_tpu, p_tpu = subtract_num_rejected_tokens(
        _as_fake_tpu(seq_lens), pos_2d, rej, idx, idx
    )
    assert s_tpu.tolist() == [9, 18] and p_tpu.tolist() == [[4, 4], [4, 4], [4, 4]]

    # extract_draft_token_ids (CPU and TPU)
    monkeypatch.setattr(
        async_out,
        "_extract_draft_token_ids_compiled",
        async_out._extract_draft_token_ids,
    )
    ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    l_idx = torch.tensor([0, 1, 2], dtype=torch.int32)
    t_idx = torch.tensor([0, 1], dtype=torch.int32)
    assert extract_draft_token_ids(ids, l_idx, t_idx).tolist() == [11, 12]
    assert extract_draft_token_ids(_as_fake_tpu(ids), l_idx, t_idx).tolist() == [11, 12]

    # compute_num_rejected: exact vs padded nt, int32 vs int64 num_draft
    assert compute_num_rejected(nt_exact, nd_exact).tolist() == [1, 2]
    res_i64 = compute_num_rejected(
        nt_padded_i64, torch.tensor([2, 2], dtype=torch.int64)
    )
    assert res_i64.dtype == torch.int32 and res_i64.tolist() == [1, 2]


def test_runner_utils_batch_composition_and_phases():
    # determine_phase_from_batch_composition_stats: all 6 phases
    for pf, tot, expected in [
        (10, 10, InferencePhase.PREFILL_ONLY),
        (0, 10, InferencePhase.DECODE_ONLY),
        (95, 100, InferencePhase.PREFILL_HEAVY),
        (10, 100, InferencePhase.DECODE_HEAVY),
        (50, 100, InferencePhase.BALANCED),
        (30, 100, InferencePhase.AMBIGUOUS),
    ]:
        assert (
            determine_phase_from_batch_composition_stats(
                {"num_prefill_tokens": pf, "total_num_scheduled_tokens": tot}
            )
            == expected
        )

    # get_batch_composition_stats:
    # Case 1: num_reqs == 0 -> total_num_scheduled_tokens=1 to avoid div-by-zero
    empty_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([], dtype=np.int32),
        req_ids=[],
    )
    empty_sched = SimpleNamespace(
        num_scheduled_tokens={}, scheduled_spec_decode_tokens={}
    )
    stats_empty = get_batch_composition_stats(0, empty_batch, 1, 0, 1, empty_sched)
    assert stats_empty["min_kv_len"] == 0

    # Case 2: 4 requests covering prefill, spec decode, chunked prefill, and decode
    input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0, 10, 20, 30], dtype=np.int32),
        req_ids=["r_prefill", "r_spec", "r_chunk", "r_decode"],
    )
    sched_out = SimpleNamespace(
        num_scheduled_tokens={
            "r_prefill": 8,
            "r_spec": 4,
            "r_chunk": 6,
            "r_decode": 1,
        },
        scheduled_spec_decode_tokens={"r_spec": [101, 102, 103]},
    )
    stats = get_batch_composition_stats(1, input_batch, 19, 4, 32, sched_out)
    assert stats["num_prefill_tokens"] == 14  # 8 + 6
    assert stats["num_decode_tokens"] == 5  # 4 + 1
    assert stats["min_kv_len"] == 0
    assert stats["phase"] == InferencePhase.AMBIGUOUS.name


def test_phased_profiler_edge_cases_and_exceptions(tmp_path, monkeypatch):
    # 1. __init__ logging branches with positive skip / thresholds
    prof = PhaseBasedProfiler(
        profile_dir=str(tmp_path),
        num_decode_steps_to_skip=2,
        decode_kv_len_threshold=16,
        prefill_kv_len_threshold=32,
    )
    # _write_batch_composition_stats_to_file_helper when suffix dir is None
    prof._write_batch_composition_stats_to_file_helper({"a": 1})
    # _step_or_stop_profiling when current_phase == ""
    prof._step_or_stop_profiling({"a": 1})
    # _finish_current_phase when profile_context is None
    prof._finish_current_phase()
    # _merge_profile_directories when profile_dir_with_phase_suffix is None
    prof._merge_profile_directories()
    # finish() when idle vs active
    prof.finish()

    # 2. _write_batch_composition_stats_to_file_helper exception handling
    prof.profile_dir_with_phase_suffix = str(tmp_path)
    with patch("builtins.open", side_effect=OSError("disk full")):
        prof._write_batch_composition_stats_to_file_helper({"a": 1})

    # 3. _start_profiling on AMBIGUOUS phase (exhausts loop without matching)
    prof.step({"num_prefill_tokens": 30, "total_num_scheduled_tokens": 100})
    assert prof.current_phase == ""

    # 4. DECODE_HEAVY skip, DECODE_ONLY kv_len skip, PREFILL_ONLY kv_len skip
    prof.step(
        {"num_prefill_tokens": 10, "total_num_scheduled_tokens": 100, "min_kv_len": 0}
    )
    assert prof.decode_steps_skipped == 1
    prof.step(
        {"num_prefill_tokens": 0, "total_num_scheduled_tokens": 100, "min_kv_len": 4}
    )
    assert prof.current_phase == ""
    prof.step(
        {"num_prefill_tokens": 100, "total_num_scheduled_tokens": 100, "min_kv_len": 4}
    )
    assert prof.current_phase == ""

    # 5. _start_profiling where profile_context.__enter__() raises Exception
    bad_ctx = MagicMock()
    bad_ctx.__enter__.side_effect = RuntimeError("enter failed")
    prof2 = PhaseBasedProfiler(
        profile_dir=str(tmp_path),
        num_steps_to_profile_for=2,
        standard_opts={
            "host_tracer_level": 2,
            "device_tracer_level": 1,
            "python_tracer_level": 1,
        },
    )
    with (
        patch("torch.profiler.profile", return_value=bad_ctx),
        patch.object(
            PhaseBasedProfiler, "_resolve_canonical_dst_ts", return_value="ts"
        ),
    ):
        prof2.step({"num_prefill_tokens": 50, "total_num_scheduled_tokens": 100})
    assert prof2.profile_context is None and prof2.current_phase == ""

    # 6. Full cycle across all 5 phases + __exit__ raising once and succeeding after
    good_ctx = MagicMock()
    good_ctx.__exit__.side_effect = [
        RuntimeError("exit failed"),
        None,
        None,
        None,
        None,
    ]
    prof3 = PhaseBasedProfiler(
        profile_dir=str(tmp_path),
        num_steps_to_profile_for=2,
        decode_kv_len_threshold=0,
        standard_opts={
            "host_tracer_level": 2,
            "device_tracer_level": 1,
            "python_tracer_level": 1,
        },
    )
    with (
        patch("torch.profiler.profile", return_value=good_ctx),
        patch.object(
            PhaseBasedProfiler, "_resolve_canonical_dst_ts", return_value="ts"
        ),
        patch.object(PhaseBasedProfiler, "_merge_profile_directories"),
    ):
        for pf in (100, 95, 0, 10, 50):
            stats = {"num_prefill_tokens": pf, "total_num_scheduled_tokens": 100}
            prof3.step(stats)
            assert prof3.current_phase != ""
            # Step 1 (decrements profiling_n_steps_left from 2 to 1)
            prof3.step(stats)
            assert prof3.profiling_n_steps_left == 1
            # Step 2 (decrements to 0 and calls _finish_current_phase)
            prof3.step(stats)
            assert prof3.current_phase == ""
        assert prof3._all_phases_seen is True
        # Subsequent steps with _all_phases_seen=True and current_phase="" are no-ops
        prof3.step({"num_prefill_tokens": 100, "total_num_scheduled_tokens": 100})
        # Also test no-work step
        prof3.step({"num_prefill_tokens": 0, "total_num_scheduled_tokens": 0})

    # Test finish() when current_phase is non-empty
    prof3.current_phase = "balanced"
    with patch.object(prof3, "_finish_current_phase") as mock_fin:
        prof3.finish()
        mock_fin.assert_called_once()

    # Test _resolve_canonical_dst_ts and _merge_profile_directories (rank 0 and rank 1)
    phase_dir = tmp_path / "merge_test"
    phase_dir.mkdir()
    p_rank0 = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0, world_size=2)
    ts0 = p_rank0._resolve_canonical_dst_ts(str(phase_dir))
    r0_dir = phase_dir / "rank_0"
    (r0_dir / "plugins" / "profile" / "pt0").mkdir(parents=True)
    (r0_dir / "plugins" / "profile" / "pt0" / "a.xplane.pb").write_text("x")
    p_rank0.profile_dir_with_phase_suffix = str(r0_dir)
    p_rank0._canonical_dst_ts = ts0

    p_rank1 = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=1, world_size=2)
    r1_dir = phase_dir / "rank_1"
    (r1_dir / "plugins" / "profile" / "pt1").mkdir(parents=True)
    (r1_dir / "plugins" / "profile" / "pt1" / "a.xplane.pb").write_text("y")
    p_rank1.profile_dir_with_phase_suffix = str(r1_dir)
    p_rank1._canonical_dst_ts = ts0

    p_rank1._merge_profile_directories()
    p_rank0._merge_profile_directories()


def test_relayout_helpers_and_entry_layout_pin(monkeypatch):
    # 1. _to_row_major
    from torch_tpu._internal.compile import compiler as tt_compiler
    from torch_tpu._internal.compile import tpu_torch_compile as ttc
    from torch_tpu._internal.device_utils import annotations as tt_annot

    monkeypatch.setattr(rutils, "_PINNED_ENTRY_LAYOUTS", {})
    monkeypatch.setattr(rutils, "_entry_pin_installed", False)

    fake_ctx = MagicMock()
    with (
        patch.object(ttc, "get_default_layout", return_value=([0, 1], [(8, 128)], 16)),
        patch.object(tt_annot, "TpuLayout", return_value="row_major_layout"),
        patch.object(tt_annot, "LayoutContext", return_value=fake_ctx),
    ):
        t = torch.ones((4, 8), dtype=torch.bfloat16)
        out = rutils._to_row_major(t)
        assert torch.equal(out, t)
        assert rutils._PINNED_ENTRY_LAYOUTS[(torch.bfloat16, (4, 8))] == [1, 0]

    # 2. _install_entry_layout_pin and _call_with_pinned_layouts
    calls = []

    def fake_orig_call(self, gm, ex_inputs, *args, **kwargs):
        calls.append((gm, ex_inputs, args, kwargs))
        return "compiled"

    monkeypatch.setattr(tt_compiler.StaticCompiler, "__call__", fake_orig_call)
    rutils._install_entry_layout_pin()
    assert rutils._entry_pin_installed is True
    # Second call is a no-op
    rutils._install_entry_layout_pin()

    compiler_inst = tt_compiler.StaticCompiler.__new__(tt_compiler.StaticCompiler)
    pinned_t = torch.ones((4, 8), dtype=torch.bfloat16)
    unpinned_t = torch.ones((2, 2), dtype=torch.float32)

    # (a) With pinned tensor + non-tensor + unpinned tensor
    assert compiler_inst("gm", [123, unpinned_t, pinned_t]) == "compiled"
    assert calls[-1][3]["argument_layouts"] == [[], [1, 0]]

    # (b) With only unpinned tensor (pinned_any is False)
    assert compiler_inst("gm", [unpinned_t]) == "compiled"
    assert "argument_layouts" not in calls[-1][3]

    # (c) With explicit argument_layouts already in kwargs (skips pinning)
    assert compiler_inst("gm", [pinned_t], argument_layouts=[[0, 1]]) == "compiled"
    assert calls[-1][3]["argument_layouts"] == [[0, 1]]

    # 3. relayout_rope_caches
    class _DummyRotary(RotaryEmbeddingBase):
        def __init__(self, cache=None):
            torch.nn.Module.__init__(self)
            if cache is not None:
                self.cos_sin_cache = cache

    with (
        patch.object(rutils, "_to_row_major", side_effect=lambda x: x.clone()),
        patch.object(rutils, "_install_entry_layout_pin") as mock_pin,
    ):
        # Empty model / non-2D / missing cache -> num_changed == 0
        m_no_change = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 2),
                _DummyRotary(None),
                _DummyRotary(torch.ones(8)),
            ]
        )
        relayout_rope_caches(m_no_change)
        mock_pin.assert_not_called()

        # 2D rotary cache -> relayouts and installs pin
        m_change = torch.nn.ModuleList([_DummyRotary(torch.ones(8, 4))])
        relayout_rope_caches(m_change)
        mock_pin.assert_called_once()

    # 4. relayout_rope_tables
    class _AttnWithTable(torch.nn.Module):
        def __init__(self, tbl):
            super().__init__()
            self._tpu_rope_cache = tbl

    with (
        patch.object(rutils, "_to_row_major", side_effect=lambda x: x.clone()),
        patch.object(rutils, "_install_entry_layout_pin") as mock_pin,
    ):
        relayout_rope_tables(torch.nn.ModuleList([_AttnWithTable(torch.ones(4))]))
        mock_pin.assert_not_called()

        shared_tbl = torch.ones(8, 4)
        relayout_rope_tables(
            torch.nn.ModuleList(
                [_AttnWithTable(shared_tbl), _AttnWithTable(shared_tbl)]
            )
        )
        mock_pin.assert_called_once()

    # 5. relayout_hash_tables
    class _MoEWithHash(torch.nn.Module):
        def __init__(self, t1=None, t2=None):
            super().__init__()
            self.tid2eid = t1
            self.hash_indices_table = t2

    with (
        patch.object(rutils, "_to_row_major", side_effect=lambda x: x.clone()),
        patch.object(rutils, "_install_entry_layout_pin") as mock_pin,
    ):
        # No 2D tables -> num_changed == 0
        relayout_hash_tables(torch.nn.ModuleList([_MoEWithHash(torch.ones(4), None)]))
        mock_pin.assert_not_called()

        # Shared 2D table + separate 2D table -> relayouts once per unique id
        shared_hash = torch.ones(8, 2)
        relayout_hash_tables(
            torch.nn.ModuleList([_MoEWithHash(shared_hash, shared_hash)])
        )
        mock_pin.assert_called_once()
