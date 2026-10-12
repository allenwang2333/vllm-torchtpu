# SPDX-License-Identifier: Apache-2.0
"""runner host fast paths.

(1) `_build_attention_metadata` builds staged non-Mamba metadata once per
    kv_cache_group_id and shares the object across that group's layers.
(2) `_prepare_lookback_token_ids` skips the gather when no request has prompt
    history in its window, and skips the fill + H2D on consecutive all -1
    steps.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as tr_mod
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Builder:
    def __init__(self, gid, *, mamba=False):
        self.kv_cache_group_id = gid
        self.is_mamba_group = mamba
        self.calls = 0

    def build(self, common_prefix_len, common_attn_metadata):
        self.calls += 1
        return SimpleNamespace(gid=self.kv_cache_group_id, builder=self)


def _walk_runner(builders, staged):
    runner = object.__new__(TPUModelRunner)
    runner.attn_groups = [
        [
            SimpleNamespace(metadata_builders=[b], layer_names=[f"l{i}a", f"l{i}b"])
            for i, b in enumerate(builders)
        ]
    ]
    runner._attn_metadata_builder_ctx = SimpleNamespace(staged_block_tables=staged)
    return runner


def _walk(runner):
    meta, spec = TPUModelRunner._build_attention_metadata(
        runner, num_tokens=8, num_reqs=4, num_reqs_padded=4
    )
    assert spec is None
    return meta


def test_staged_walk_builds_once_per_group_id():
    b = [
        _Builder(0),
        _Builder(0),
        _Builder(1),
        _Builder(0, mamba=True),
        _Builder(0, mamba=True),
        _Builder(7),  # not staged
        _Builder(7),
        _Builder(2),  # staged entry is None
    ]
    staged = {0: torch.zeros(2), 1: torch.zeros(2), 2: None}
    meta = _walk(_walk_runner(b, staged))

    assert [x.calls for x in b] == [1, 0, 1, 1, 1, 1, 1, 1]
    assert meta["l0a"] is meta["l0b"] is meta["l1a"] is meta["l1b"]
    assert meta["l2a"] is not meta["l0a"]
    assert meta["l3a"].builder is b[3] and meta["l4a"].builder is b[4]
    assert meta["l5a"] is not meta["l6a"]
    assert meta["l7a"].builder is b[7]
    # Layer order is unchanged.
    assert list(meta) == [f"l{i}{s}" for i in range(8) for s in "ab"]


def test_unstaged_walk_builds_every_group():
    b = [_Builder(0), _Builder(0)]
    runner = _walk_runner(b, None)
    with patch.object(tr_mod, "stage_block_table_uploads") as stage:
        meta = _walk(runner)
    stage.assert_called_once()
    assert [x.calls for x in b] == [1, 1]
    assert meta["l0a"] is not meta["l1a"]


def test_shared_metadata_equals_per_builder_clone():
    """Real builders: the shared object holds exactly what the old second
    `build` (a `with_block_tables` clone of the base) returned."""
    staged = {0: torch.arange(4, dtype=torch.int32)}
    desc = SimpleNamespace(kind=SimpleNamespace(value=0), protocol=1, version=2)
    ctx = SimpleNamespace(
        staged_block_tables=staged,
        base_attn_metadata=None,
        position_ids_override=None,
        seq_lens=torch.ones(4, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
        sequence_layout_descriptor=desc,
        query_start_loc_cpu=(0, 1, 2, 3, 4),
        seq_lens_cpu=(1, 1, 1, 1),
        prompt_lens_cpu=(1, 1, 1, 1),
        replay_starts_cpu=None,
    )
    runner = object.__new__(TPUModelRunner)
    runner.position_ids = torch.zeros(8, dtype=torch.int32)
    runner._attn_metadata_builder_ctx = ctx

    def builder():
        bld = object.__new__(AttentionMetadataBuilder)
        bld.runner = runner
        bld.kv_cache_group_id = 0
        bld.is_mamba_group = False
        return bld

    first, second = builder(), builder()
    runner.attn_groups = [
        [
            SimpleNamespace(metadata_builders=[first], layer_names=["a"]),
            SimpleNamespace(metadata_builders=[second], layer_names=["b"]),
        ]
    ]
    meta = _walk(runner)
    assert meta["a"] is meta["b"]
    assert isinstance(meta["a"], AttentionMetadata)

    # with_block_tables returns base when block_tables is identical.
    ctx.base_attn_metadata = None
    common = SimpleNamespace(num_reqs=4)
    base = first.build(common_prefix_len=0, common_attn_metadata=common)
    clone = second.build(common_prefix_len=0, common_attn_metadata=common)
    assert clone is base
    assert clone.__dict__.keys() == meta["a"].__dict__.keys()
    for k, v in clone.__dict__.items():
        got = meta["a"].__dict__[k]
        if isinstance(v, torch.Tensor):
            assert torch.equal(got, v)
        else:
            assert got == v


class _LookbackBuffer:
    def __init__(self, rows, depth):
        self.cpu = torch.zeros((rows, depth), dtype=torch.int32)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros((rows, depth), dtype=torch.int32)
        self.uploads = 0

    def copy_to_gpu(self):
        self.uploads += 1
        return self.gpu.copy_(self.cpu)


def _lookback_runner(computed, prompt, rows=4, depth=2):
    runner = object.__new__(TPUModelRunner)
    runner.lookback_token_ids = _LookbackBuffer(rows, depth)
    n = len(computed)
    runner.input_batch = SimpleNamespace(
        req_ids=[f"r{i}" for i in range(n)],
        num_computed_tokens_cpu=np.array(computed, dtype=np.int32),
        num_prompt_tokens=np.array(prompt, dtype=np.int32),
        num_tokens_no_spec=np.array(prompt, dtype=np.int32),
        token_ids_cpu=np.arange(n * 100, dtype=np.int32).reshape(n, 100),
    )
    runner._replay_starts = {}
    return runner


def _old_lookback(runner, num_reqs):
    """Previous reference values."""
    buf = runner.lookback_token_ids
    depth = buf.np.shape[1]
    out = np.full(buf.np.shape, -1, dtype=np.int32)
    if num_reqs > 0:
        ib = runner.input_batch
        starts = ib.num_computed_tokens_cpu[:num_reqs, None]
        pos = starts - np.arange(1, depth + 1)
        valid = (pos >= 0) & (pos < ib.num_prompt_tokens[:num_reqs, None])
        ids = ib.token_ids_cpu[np.arange(num_reqs)[:, None], np.maximum(pos, 0)]
        out[:num_reqs] = np.where(valid, ids, -1)
    return out


def test_lookback_with_prompt_history_gathers_every_step():
    runner = _lookback_runner([3, 50], [10, 10])
    buf = runner.lookback_token_ids
    buf.np[:] = 99  # stale rows must be cleared
    for step in (1, 2):
        ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
        assert ids is buf.gpu
        np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))
        assert buf.uploads == step
    assert ids.numpy().tolist() == [[2, 1], [-1, -1], [-1, -1], [-1, -1]]
    assert runner._lookback_all_neg1_synced is False


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_lookback_all_invalid_uploads_once(num_reqs):
    runner = _lookback_runner([50, 60], [10, 10])
    buf = runner.lookback_token_ids
    buf.np[:] = 99
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, num_reqs)
    assert ids is buf.gpu
    assert (ids == -1).all() and (buf.np == -1).all()
    assert buf.uploads == 1
    assert runner._lookback_all_neg1_synced is True

    ids2 = TPUModelRunner._prepare_lookback_token_ids(runner, num_reqs)
    assert ids2 is buf.gpu
    assert buf.uploads == 1
    np.testing.assert_array_equal(ids2.numpy(), _old_lookback(runner, num_reqs))


def test_lookback_valid_step_resets_synced_state():
    runner = _lookback_runner([50, 60], [10, 10])
    buf = runner.lookback_token_ids
    TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 1

    runner.input_batch.num_computed_tokens_cpu[1] = 5  # back inside its prompt
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 2
    assert runner._lookback_all_neg1_synced is False
    np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))

    runner.input_batch.num_computed_tokens_cpu[1] = 60
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 3  # the -1 buffer must be re-uploaded
    assert (ids == -1).all()


# --- 1D all-past-prompt check ahead of the 2D window -----------------


def test_all_past_prompt_skips_window(monkeypatch):
    runner = _lookback_runner([50, 60], [10, 47], depth=3)  # 60 - 47 >= 3
    # The 2D window would index _ARANGE_INTP; poison it to prove it is skipped.
    monkeypatch.setattr(tr_mod, "_ARANGE_INTP", None)
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert (ids == -1).all()
    assert runner.lookback_token_ids.uploads == 1


def test_inside_window_still_gathers():
    runner = _lookback_runner([50, 49], [10, 47], depth=3)  # 49 - 47 < 3
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))
    assert ids.numpy()[1].tolist() == [-1, -1, 146]  # only pos 46 < prompt 47


def test_before_prompt_start_falls_through_to_all_minus_one():
    # computed 0 < depth: the 1D check fails, the window finds nothing valid.
    runner = _lookback_runner([0, 60], [10, 10], depth=2)
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert (ids == -1).all()
    np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))


def test_attn_group_walk_cache_and_for_layers_gid_dedup():
    """Verify _build_attention_metadata caches _attn_group_walk_cache
    across steps, and build_attention_metadata_for_layers reuses frozenset
    layer_names and deduplicates staged non-Mamba builder.build calls by gid."""
    b = [
        _Builder(0),
        _Builder(0),
        _Builder(1),
        _Builder(0, mamba=True),
        _Builder(7),
        _Builder(7),
    ]
    staged = {0: torch.zeros(2), 1: torch.zeros(2)}
    runner = _walk_runner(b, staged)

    meta1 = _walk(runner)
    walk_cache1 = runner._attn_group_walk_cache
    assert walk_cache1 is not None and len(walk_cache1[1]) == 6
    meta2 = _walk(runner)
    assert runner._attn_group_walk_cache is walk_cache1
    assert list(meta1) == list(meta2)

    # Empty attn_groups exercises the `else 0` branches in walk_key
    empty_runner = _walk_runner([], staged)
    empty_runner.attn_groups = []
    assert _walk(empty_runner) == {}
    assert empty_runner._attn_group_walk_cache[0][2:] == (0, 0)

    # build_attention_metadata_for_layers: frozenset key + gid deduplication
    for x in b:
        x.calls = 0
    fset = frozenset(["l0a", "l1a", "l2a", "l3a", "l4a", "l5a"])
    sub_meta1 = TPUModelRunner.build_attention_metadata_for_layers(runner, fset, 4)
    assert [x.calls for x in b] == [1, 0, 1, 1, 1, 1]
    assert sub_meta1["l0a"] is sub_meta1["l1a"]
    assert sub_meta1["l2a"] is not sub_meta1["l0a"]
    assert sub_meta1["l3a"].builder is b[3]
    assert sub_meta1["l4a"] is not sub_meta1["l5a"]
    # Verify frozenset object was used directly without re-allocating in selection_key
    cached_key = next(iter(runner._layer_group_selection_cache))
    assert cached_key[2] is fset

    # Second call hits cached_selection and unstaged ctx calls stage_block_table_uploads
    runner._attn_metadata_builder_ctx.staged_block_tables = None
    with patch.object(tr_mod, "stage_block_table_uploads") as stage_fn:
        sub_meta2 = TPUModelRunner.build_attention_metadata_for_layers(runner, fset, 4)
    stage_fn.assert_called_once()
    assert list(sub_meta2) == list(sub_meta1)


def test_modify_prev_results_full_accept_and_lookback_short_circuit(monkeypatch):
    """(a) _modify_prev_results skips np.asarray(gen_lens) and boolean
    mask indexing when gen_lens.count(w) == n_reqs and sampled_np has no -1;
    (b) _prepare_lookback_token_ids short-circuits in O(1) when
    _lookback_all_neg1_synced is True and _fast_uniform_spec_k is not None."""
    # Part (b): _prepare_lookback_token_ids O(1) short-circuit
    runner_lb = _lookback_runner([50, 60], [10, 10])
    TPUModelRunner._prepare_lookback_token_ids(runner_lb, 2)
    assert runner_lb._lookback_all_neg1_synced is True
    runner_lb._fast_uniform_spec_k = ({}, ["r0", "r1"], 5)
    # Poison num_prompt_tokens to prove the subtraction/all() is short-circuited
    runner_lb.input_batch.num_prompt_tokens = None
    ids = TPUModelRunner._prepare_lookback_token_ids(runner_lb, 2)
    assert ids is runner_lb.lookback_token_ids.gpu
    assert runner_lb.lookback_token_ids.uploads == 1

    # Part (a): _modify_prev_results full-accept vs partial-accept
    def make_prev_runner(sampled_tensor, draft_k=2):
        r = object.__new__(TPUModelRunner)
        r.arange_np = np.arange(32, dtype=np.int32)
        req0 = SimpleNamespace(output_token_ids=[1, 2, -1, -1, -1])
        req1 = SimpleNamespace(output_token_ids=[3, 4, -1, -1, -1])
        r.input_batch = SimpleNamespace(
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            num_tokens_no_spec=np.array([7, 9], dtype=np.int32),
            token_ids_cpu=np.zeros((2, 16), dtype=np.int32),
        )
        r._pre_async_results = SimpleNamespace(
            req_ids=["r0", "r1"],
            request_seq_lens=[(0, req0, 7, "r0"), (1, req1, 9, "r1")],
            discard_sampled_tokens_req_indices=set(),
            num_draft_per_req={0: draft_k, 1: draft_k},
            copy_state=None,
            wait_for_copy=lambda: sampled_tensor,
        )
        return r, req0, req1

    # 1. Full accept (all 3 tokens valid per request): np.asarray is NOT called
    sampled_full = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    r_full, req0_f, req1_f = make_prev_runner(sampled_full)
    orig_asarray = np.asarray
    asarray_calls = []

    def spy_asarray(a, *args, **kwargs):
        asarray_calls.append(a)
        return orig_asarray(a, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", spy_asarray)
    TPUModelRunner._modify_prev_results(r_full)
    monkeypatch.undo()
    assert asarray_calls == []
    assert req0_f.output_token_ids == [1, 2, 10, 11, 12]
    assert req1_f.output_token_ids == [3, 4, 20, 21, 22]
    assert r_full.input_batch.num_tokens_no_spec.tolist() == [7, 9]
    assert r_full.input_batch.token_ids_cpu[0, 4:7].tolist() == [10, 11, 12]
    assert r_full.input_batch.token_ids_cpu[1, 6:9].tolist() == [20, 21, 22]

    # 2. Partial accept with arange_np=None & non-uniform drafts: falls back to mask
    sampled_part = torch.tensor([[10, 11, 12], [20, 21, -1]], dtype=torch.int32)
    r_part, req0_p, req1_p = make_prev_runner(sampled_part)
    r_part.arange_np = None
    r_part._pre_async_results.num_draft_per_req = {0: 2, 1: 1}
    req1_p.output_token_ids = [3, 4, 5, -1, -1]
    r_part.input_batch.num_tokens_no_spec = np.array([7, 8], dtype=np.int32)
    TPUModelRunner._modify_prev_results(r_part)
    assert req0_p.output_token_ids == [1, 2, 10, 11, 12]
    assert req1_p.output_token_ids == [3, 4, 5, 20, 21]
    assert r_part.input_batch.num_tokens_no_spec.tolist() == [7, 8]
    assert r_part.input_batch.token_ids_cpu[0, 4:7].tolist() == [10, 11, 12]
    assert r_part.input_batch.token_ids_cpu[1, 6:8].tolist() == [20, 21]
    assert r_part.input_batch.token_ids_cpu[1, 8] == 0


def test_fast_uniform_spec_schedule_across_reorder_chunks_and_prepare(
    monkeypatch,
):
    """_try_fast_update_states records whether all scheduled_tokens equal
    1 + k in _fast_uniform_spec_k[3], and _reorder_batch_for_rpa, _count_input_chunks,
    and _prepare_inputs reuse it to skip per-request num_scheduled_tokens lookups."""
    monkeypatch.setattr(
        tr_mod.mla_dispatch, "maybe_order_prefills_by_kv_len", lambda *a: None
    )
    runner = object.__new__(TPUModelRunner)
    runner.requests = {
        "r0": SimpleNamespace(
            num_computed_tokens=0,
            prev_num_draft_len=0,
            output_token_ids=[1, 2],
            block_ids=[[]],
        ),
        "r1": SimpleNamespace(
            num_computed_tokens=0,
            prev_num_draft_len=0,
            output_token_ids=[3, 4],
            block_ids=[[]],
        ),
    }
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
        num_prompt_tokens=np.array([4, 4], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 6], dtype=np.int32),
        token_ids_cpu=np.zeros((2, 16), dtype=np.int32),
        is_token_ids=np.zeros((2, 16), dtype=bool),
        spec_token_ids=[[], []],
        block_table=SimpleNamespace(append_row=lambda *a: None),
        refresh_metadata=lambda: None,
        swap_states=lambda i, j: None,
    )
    runner.reorder_batch_threshold = 1
    runner.most_model_len = 4
    runner.num_reqs_max_model_len = 1
    runner.num_reqs_most_model_len = 2
    runner._wait_input_staging_fence = lambda: None

    spec_tokens = {"r0": [101, 102], "r1": [201, 202]}
    sched_out = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens=[6, 6],
            num_output_tokens=[2, 2],
            new_block_ids=[None, None],
        ),
        scheduled_spec_decode_tokens=spec_tokens,
        num_scheduled_tokens={"r0": 3, "r1": 3},
        total_num_scheduled_tokens=6,
    )
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert runner._fast_uniform_spec_k[:4] == (spec_tokens, ["r0", "r1"], 2, True)

    # Poison num_scheduled_tokens to prove all 3 consumers skip dict indexing
    sched_out.num_scheduled_tokens = None
    assert TPUModelRunner._reorder_batch_for_rpa(runner, sched_out) == (0, 2)
    assert TPUModelRunner._count_input_chunks(runner, sched_out) == (1, 2)
    runner.most_model_len = 2
    assert TPUModelRunner._count_input_chunks(runner, sched_out) == (2, 1)
    runner.most_model_len = 4

    seen_totals = []

    class _StopPlanner(Exception):
        pass

    def _stop_reserve(_self, total_tokens):
        seen_totals.append(total_tokens)
        raise _StopPlanner

    monkeypatch.setattr(
        tr_mod,
        "_get_sequence_layout_planner_for_runner",
        lambda _self: SimpleNamespace(reserve_host_token_capacity=_stop_reserve),
    )
    with pytest.raises(_StopPlanner):
        TPUModelRunner._prepare_inputs(runner, sched_out, 0, 0)
    assert seen_totals == [6]

    # Non-uniform scheduled tokens sets fast_spec[3] = False and falls back cleanly
    sched_out.num_scheduled_tokens = {"r0": 3, "r1": 2}
    sched_out.total_num_scheduled_tokens = 5
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert runner._fast_uniform_spec_k[:4] == (spec_tokens, ["r0", "r1"], 2, False)
    assert TPUModelRunner._reorder_batch_for_rpa(runner, sched_out) == (0, 2)
    assert TPUModelRunner._count_input_chunks(runner, sched_out) == (1, 2)
    with pytest.raises(_StopPlanner):
        TPUModelRunner._prepare_inputs(runner, sched_out, 0, 0)
    assert seen_totals == [6, 5]


def test_sample_tokens_fast_request_seq_lens_and_fast_update_states_spec():
    """(1) _try_fast_update_states uses direct indexing when
    len(scheduled_spec_tokens) == n_reqs, short-circuits all(...) when
    spec_list.count(first_spec) == n_reqs, sets prev_num_draft_len = k in the
    main request loop, and skips the spec_token_ids update loop when cur_specs
    already equal first_spec; (2) sample_tokens builds request_seq_lens in O(N)
    without dict lookups or req_state.num_tokens on speculative decode steps."""
    import contextlib

    req0 = SimpleNamespace(
        req_id="r0",
        num_computed_tokens=0,
        prev_num_draft_len=0,
        output_token_ids=[1, 2],
        block_ids=[[]],
    )
    req1 = SimpleNamespace(
        req_id="r1",
        num_computed_tokens=0,
        prev_num_draft_len=0,
        output_token_ids=[3, 4],
        block_ids=[[]],
    )
    updated_fallback = []
    runner = object.__new__(TPUModelRunner)
    runner.requests = {"r0": req0, "r1": req1}
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
        num_prompt_tokens=np.array([4, 4], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 6], dtype=np.int32),
        token_ids_cpu=np.zeros((2, 16), dtype=np.int32),
        is_token_ids=np.zeros((2, 16), dtype=bool),
        spec_token_ids=[[], []],
        block_table=SimpleNamespace(append_row=lambda *a: None),
        refresh_metadata=lambda: None,
        update_req_spec_token_ids=lambda rs, st: updated_fallback.append(rs.req_id),
        all_greedy=True,
        max_num_logprobs=None,
        generators={},
    )

    # Step 1: identical placeholder drafts [-1, -1] populate spec_token_ids
    same_tokens = {"r0": [-1, -1], "r1": [-1, -1]}
    sched_out = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens=[6, 6],
            num_output_tokens=[2, 2],
            new_block_ids=[None, None],
        ),
        scheduled_spec_decode_tokens=same_tokens,
        num_scheduled_tokens={"r0": 3, "r1": 3},
        has_structured_output_requests=False,
    )
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert runner.input_batch.spec_token_ids == [[-1, -1], [-1, -1]]
    assert (req0.prev_num_draft_len, req1.prev_num_draft_len) == (2, 2)
    assert runner._fast_uniform_spec_k == (
        same_tokens,
        ["r0", "r1"],
        2,
        True,
        [req0, req1],
    )

    # Step 2: cur_specs already equal first_spec -> skips loop, prev_num_draft_len=2
    req0.prev_num_draft_len = 0
    req1.prev_num_draft_len = 0
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert (req0.prev_num_draft_len, req1.prev_num_draft_len) == (2, 2)

    # Extra key in scheduled_spec_tokens exercises map(scheduled_spec_tokens.get, ...)
    # and distinct spec lists exercise cur_spec == ids vs != ids, plus num_out / blocks
    sched_out.scheduled_spec_decode_tokens = {
        "r0": [-1, -1],
        "r1": [-2, -2],
        "extra": [-1, -1],
    }
    sched_out.scheduled_cached_reqs.num_output_tokens = [1, 2]
    sched_out.scheduled_cached_reqs.new_block_ids = [[[5]], [[]]]
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert req0.output_token_ids == [1]
    assert req0.block_ids == [[5]]
    assert runner.input_batch.spec_token_ids == [[-1, -1], [-2, -2]]

    # Heterogeneous spec lengths exercise k=0 -> update_req_spec_token_ids fallback
    sched_out.scheduled_spec_decode_tokens = {"r0": [-1, -1], "r1": [-1]}
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert updated_fallback == ["r0", "r1"]
    assert (req0.prev_num_draft_len, req1.prev_num_draft_len) == (0, 0)

    # Empty scheduled_spec_decode_tokens clears spec_token_ids and sets k=0
    sched_out.scheduled_spec_decode_tokens = {}
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    assert runner.input_batch.spec_token_ids == [[], []]

    # Part (2): sample_tokens fast request_seq_lens construction (5-tuple and 3-tuple)
    sched_out.scheduled_spec_decode_tokens = same_tokens
    assert TPUModelRunner._try_fast_update_states(runner, sched_out) is True
    # Poison num_scheduled_tokens; req0/req1 also have no .num_tokens attribute
    sched_out.num_scheduled_tokens = None

    def _make_exec_state():
        return SimpleNamespace(
            scheduler_output=sched_out,
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            logits_list=[],
            num_reqs_list=[],
            pooler_output_list=[],
            draft_chunks=[],
        )

    runner._get_sampling_generator = lambda: None
    runner.speculative_config = None
    runner.is_pooling_model = False
    runner._stage_timing = SimpleNamespace(
        stage=lambda _n: contextlib.nullcontext(),
        host_stage=lambda _n: contextlib.nullcontext(),
    )
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner._modify_prev_results = lambda: None
    runner.maybe_wait_for_kv_save = lambda: None
    runner.get_finished_kv_transfers = lambda _so: (None, None, None, set(), None, None)
    runner._sync_replicated_drafts_across_tp = lambda d: d
    seen_req_seq_lens = []
    runner._update_placeholder = (
        lambda _disc, rsl, _nti, _nd: seen_req_seq_lens.append(rsl) or {}
    )

    runner.execute_model_state = _make_exec_state()
    TPUModelRunner.sample_tokens(runner, grammar_output=None)
    assert seen_req_seq_lens[-1] == [(0, req0, 0, "r0"), (1, req1, 0, "r1")]

    # 3-tuple _fast_uniform_spec_k exercises map(requests.__getitem__, req_ids)
    runner._fast_uniform_spec_k = (same_tokens, ["r0", "r1"], 2)
    runner.execute_model_state = _make_exec_state()
    TPUModelRunner.sample_tokens(runner, grammar_output=None)
    assert seen_req_seq_lens[-1] == [(0, req0, 0, "r0"), (1, req1, 0, "r1")]

    # _fast_uniform_spec_k = None falls back to per-request loop (incl. partial prefill)
    runner._fast_uniform_spec_k = None
    sched_out.num_scheduled_tokens = {"r0": 3, "r1": 1}
    req0.num_tokens = 9
    req1.num_tokens = 20
    gen_offsets = [12]
    runner.input_batch.generators = {
        1: SimpleNamespace(
            get_offset=lambda: gen_offsets[0],
            set_offset=lambda v: gen_offsets.__setitem__(0, v),
        )
    }
    runner.execute_model_state = _make_exec_state()
    TPUModelRunner.sample_tokens(runner, grammar_output=None)
    assert seen_req_seq_lens[-1] == [(0, req0, 9, "r0")]
    assert gen_offsets == [8]
    runner.input_batch.generators = {}
    runner.execute_model_state = _make_exec_state()
    TPUModelRunner.sample_tokens(runner, grammar_output=None)
    assert seen_req_seq_lens[-1] == [(0, req0, 9, "r0")]
