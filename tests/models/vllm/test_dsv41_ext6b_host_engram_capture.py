# SPDX-License-Identifier: Apache-2.0
"""EXT-6b: vectorized host capture for OFFLOAD=1 Engram under DSpark K=5.

``capture_host_engram_inputs`` runs on the host between the previous step's
completion and the next launch. These tests pin its outputs against an
independent per-request Python reference over randomized C=1..224 batches and
cover the helper branches the existing suites do not reach.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import host_engram as he


def _make_runner(
    num_reqs: int,
    *,
    k: int = 5,
    depth: int = 4,
    seed: int = 0,
    spec: bool = True,
    with_no_spec: bool = True,
    drop_every: int = 0,
    max_len: int = 64,
) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    stride = 1 + k if spec else 1
    t = num_reqs * stride
    computed = rng.integers(8, max_len - 8, size=num_reqs).astype(np.int64)
    token_ids = rng.integers(0, 1000, size=(num_reqs, max_len)).astype(np.int32)
    n_acc = rng.integers(0, k + 1, size=num_reqs)
    sampled = np.full((num_reqs, 1 + k), -1, dtype=np.int32)
    for r in range(num_reqs):
        sampled[r, : n_acc[r] + 1] = rng.integers(1000, 2000, size=n_acc[r] + 1)
    discard = [int(i) for i in np.flatnonzero(rng.random(num_reqs) < 0.1)]
    next_tokens = torch.from_numpy(rng.integers(2000, 3000, size=t).astype(np.int32))
    pre_index = rng.permutation(num_reqs)
    req_id_to_index = {f"req-{i}": int(pre_index[i]) for i in range(num_reqs)}
    if drop_every:
        for i in range(0, num_reqs, drop_every):
            del req_id_to_index[f"req-{i}"]
    sampled_t = torch.from_numpy(sampled)
    pre_async = SimpleNamespace(
        req_id_to_index_copy=req_id_to_index,
        wait_for_copy=lambda: sampled_t,
        copy_state=None,
        next_tokens_tpu=next_tokens,
        spec_decode_num_rejected_tokens=(
            torch.zeros(num_reqs, dtype=torch.int32) if spec else None
        ),
        num_draft_per_req={int(pre_index[i]): k for i in range(num_reqs)},
        discard_sampled_tokens_req_indices=discard,
    )
    batch = SimpleNamespace(
        num_prompt_tokens=np.full(num_reqs, 4, dtype=np.int64),
        num_computed_tokens_cpu=computed,
        num_tokens_no_spec=(computed + 1) if with_no_spec else None,
        token_ids_cpu=token_ids,
        req_ids=[f"req-{i}" for i in range(num_reqs)],
    )
    return SimpleNamespace(
        _last_sequence_layout_plan=SimpleNamespace(
            kind=SequenceLayoutKind.ALL, start_index=0
        ),
        _attn_metadata_builder_ctx=None,
        input_batch=batch,
        seq_lens_np=computed + stride,
        input_ids_cpu=torch.zeros(t, dtype=torch.int32),
        positions_cpu=torch.from_numpy(
            (computed[:, None] + np.arange(stride)[None, :]).reshape(-1)
        ),
        query_start_loc_cpu=torch.arange(
            0, (num_reqs + 1) * stride, stride, dtype=torch.int32
        ),
        lookback_token_ids=SimpleNamespace(
            cpu=torch.full((num_reqs, depth), -1, dtype=torch.int32)
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        _pre_async_results=pre_async,
    )


def _reference(runner, num_reqs: int):
    """Per-request Python statement of what the host capture must produce."""
    pa = runner._pre_async_results
    batch = runner.input_batch
    stride = 1 + runner.speculative_config.num_speculative_tokens
    if pa.spec_decode_num_rejected_tokens is None:
        stride = 1
    sampled = pa.wait_for_copy().numpy()
    flat_src = pa.next_tokens_tpu.numpy()
    ids = runner.input_ids_cpu.numpy().copy()
    positions = runner.positions_cpu.numpy().copy()
    qsl = runner.query_start_loc_cpu.numpy()
    depth = runner.lookback_token_ids.cpu.shape[1]
    lookback = np.full((num_reqs, depth), -1, dtype=np.int64)
    discard = set(pa.discard_sampled_tokens_req_indices)
    for i in range(num_reqs):
        req_id = batch.req_ids[i]
        pre = pa.req_id_to_index_copy.get(req_id)
        q0, q1 = int(qsl[i]), int(qsl[i + 1])
        rejected = 0
        overlay: list[int] = []
        write_start = 0
        if pre is not None and q1 > q0:
            ids[q0:q1] = flat_src[pre * stride : pre * stride + (q1 - q0)]
            accepted = [int(x) for x in sampled[pre] if x != -1]
            n_draft = pa.num_draft_per_req.get(pre, 0)
            if pa.spec_decode_num_rejected_tokens is not None and n_draft > 0:
                rejected = max(0, n_draft - len(accepted) + 1)
            positions[q0:q1] -= rejected
            if pre not in discard:
                overlay = accepted
                if batch.num_tokens_no_spec is not None:
                    write_start = int(batch.num_tokens_no_spec[i]) - 1 - n_draft
                else:
                    write_start = int(batch.num_computed_tokens_cpu[i]) - n_draft
        start = int(batch.num_computed_tokens_cpu[i]) - rejected
        for d in range(depth):
            p = start - 1 - d
            if write_start <= p < write_start + len(overlay):
                lookback[i, d] = overlay[p - write_start]
            elif 0 <= p < batch.token_ids_cpu.shape[1]:
                lookback[i, d] = batch.token_ids_cpu[i, p]
    return ids, positions, lookback


@pytest.mark.parametrize("num_reqs", [1, 3, 48, 224])
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"with_no_spec": False},
        {"drop_every": 5},
        {"spec": False},
        {"k": 1, "depth": 2},
        {"k": 5, "depth": 6, "max_len": 24},
    ],
)
def test_capture_matches_per_request_reference(num_reqs, kw):
    for seed in range(3):
        runner = _make_runner(num_reqs, seed=seed, **kw)
        res = he.capture_host_engram_inputs(runner, num_reqs)
        assert res is not None
        if not runner._pre_async_results.req_id_to_index_copy:
            # No request survived from the previous step: nothing to resolve
            # on the host, the device hash path takes over.
            assert res.host_ready is False
            continue
        assert res.host_ready is True

        ids, positions, lookback = _reference(runner, num_reqs)
        np.testing.assert_array_equal(res.input_ids.numpy(), ids)
        np.testing.assert_array_equal(res.positions.numpy(), positions)
        np.testing.assert_array_equal(res.lookback_token_ids.numpy(), lookback)
        assert res.input_ids.dtype == runner.input_ids_cpu.dtype
        assert res.positions.dtype == runner.positions_cpu.dtype
        assert res.lookback_token_ids.dtype == runner.lookback_token_ids.cpu.dtype


def test_positions_untouched_without_rejections():
    # Every draft accepted -> rejected == 0 everywhere -> positions is the
    # runner's own tensor (no clone), input ids are still substituted.
    runner = _make_runner(6, seed=7)
    full = torch.from_numpy(np.tile(np.arange(1000, 1006, dtype=np.int32), (6, 1)))
    runner._pre_async_results.wait_for_copy = lambda: full
    res = he.capture_host_engram_inputs(runner, 6)
    assert res is not None and res.positions is runner.positions_cpu
    assert res.input_ids is not runner.input_ids_cpu
    assert torch.equal(
        res.input_ids,
        runner._pre_async_results.next_tokens_tpu.index_select(
            0,
            torch.from_numpy(
                he._ragged_arange(
                    np.array(
                        [
                            runner._pre_async_results.req_id_to_index_copy[f"req-{i}"]
                            * 6
                            for i in range(6)
                        ]
                    ),
                    np.full(6, 6),
                )
            ),
        ),
    )


def test_resolve_async_spec_state_branches():
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=2)
    )
    sampled = torch.tensor([[5, -1, 6], [7, 8, 9], [-1, -1, -1]], dtype=torch.int32)
    pre = SimpleNamespace(
        req_id_to_index_copy={"a": 0, "b": 1, "c": 2},
        wait_for_copy=lambda: sampled,
        spec_decode_num_rejected_tokens=torch.zeros(3),
        num_draft_per_req={0: 2, 1: 2, 2: 2, 7: 2, -1: 2},  # 7 / -1 out of range
        discard_sampled_tokens_req_indices=[1, 9],  # 9 is out of range
    )
    state = he._resolve_async_spec_state(runner, pre)
    assert state is not None and state.stride == 3
    # Valid tokens compact to the front in order; discarded row 1 has length 0
    # but keeps its rejection count (computed before discarding).
    assert state.tokens[0, :2].tolist() == [5, 6]
    assert state.lengths.tolist() == [2, 0, 0]
    assert state.rejected.tolist() == [1, 0, 3]
    assert state.num_draft.tolist() == [2, 2, 2]

    # No num_draft_per_req at all and no rejection tensor -> stride 1, zeros.
    pre_plain = SimpleNamespace(
        req_id_to_index_copy={"a": 0},
        wait_for_copy=lambda: torch.tensor([4, -1], dtype=torch.int32),  # 1-D
        spec_decode_num_rejected_tokens=None,
        num_draft_per_req=None,
        discard_sampled_tokens_req_indices=None,
    )
    plain = he._resolve_async_spec_state(runner, pre_plain)
    assert plain is not None and plain.stride == 1
    assert plain.tokens.shape == (2, 1)
    assert plain.lengths.tolist() == [1, 0]
    assert plain.rejected.tolist() == [0, 0]

    # Non-empty discard list where all indices are out of range -> lengths unchanged.
    pre_oob_discard = SimpleNamespace(
        req_id_to_index_copy={"a": 0},
        wait_for_copy=lambda: torch.tensor([[7, 8, -1]], dtype=torch.int32),
        spec_decode_num_rejected_tokens=None,
        num_draft_per_req=None,
        discard_sampled_tokens_req_indices=[99, -5],
    )
    oob_discard = he._resolve_async_spec_state(runner, pre_oob_discard)
    assert oob_discard is not None
    assert oob_discard.lengths.tolist() == [2]

    # Out-of-range previous index -> None (host falls back to the device hash).
    pre_oob = SimpleNamespace(
        req_id_to_index_copy={"a": 3},
        wait_for_copy=lambda: sampled,
        spec_decode_num_rejected_tokens=None,
    )
    assert he._resolve_async_spec_state(runner, pre_oob) is None
    pre_neg = SimpleNamespace(
        req_id_to_index_copy={"a": -1},
        wait_for_copy=lambda: sampled,
        spec_decode_num_rejected_tokens=None,
    )
    assert he._resolve_async_spec_state(runner, pre_neg) is None


def test_ragged_arange():
    out = he._ragged_arange(np.array([10, 20, 30]), np.array([2, 0, 3]))
    assert out.tolist() == [10, 11, 30, 31, 32]
    uniform = he._ragged_arange(np.array([10, 20]), np.array([2, 2]))
    assert uniform.tolist() == [10, 11, 20, 21]
    empty = np.array([], dtype=np.int64)
    assert he._ragged_arange(empty, empty).tolist() == []


def test_lookback_overlay_edges():
    runner = SimpleNamespace(
        lookback_token_ids=SimpleNamespace(
            cpu=torch.full((2, 3), -1, dtype=torch.int32)
        )
    )
    batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([6, 6], dtype=np.int64),
        token_ids_cpu=np.arange(100, 120, dtype=np.int32).reshape(2, 10),
    )
    # Row 0: positions 5,4,3; overlay [7,8] written at 4..5 -> [8, 7, 103].
    # Row 1: overlay empty -> table values [115, 114, 113].
    overlay = he._SampledOverlay(
        tokens=np.array([[7, 8, -1], [9, 9, 9]]),
        starts=np.array([4, 0]),
        lengths=np.array([2, 0]),
    )
    out = he._resolve_decode_lookback_cpu(
        runner, batch, 2, 0, rejected_by_req=np.array([0, 0]), sampled_overlay=overlay
    )
    assert out.tolist() == [[8, 7, 103], [115, 114, 113]]
    # Overlay entirely outside the window -> no change (the `hit.any()` gate).
    far = he._SampledOverlay(
        tokens=np.array([[7, 8, -1], [9, 9, 9]]),
        starts=np.array([50, 50]),
        lengths=np.array([2, 2]),
    )
    out_far = he._resolve_decode_lookback_cpu(runner, batch, 2, 0, sampled_overlay=far)
    assert out_far.tolist() == [[105, 104, 103], [115, 114, 113]]


def test_host_scratch_buffer_reuse_and_resync():
    runner = _make_runner(4, seed=11)
    # Pad input_ids_cpu and positions_cpu to a larger max_num_batched_tokens
    runner.input_ids_cpu = torch.zeros(64, dtype=torch.int32)
    pos_base = (
        runner.input_batch.num_computed_tokens_cpu[:, None] + np.arange(6)
    ).reshape(-1)
    runner.positions_cpu = torch.zeros(64, dtype=torch.int64)
    runner.positions_cpu[:24] = torch.from_numpy(pos_base)

    res1 = he.capture_host_engram_inputs(runner, 4)
    assert res1 is not None and res1.host_ready is True
    ids_scratch = runner._host_engram_ids_scratch
    pos_scratch = runner._host_engram_pos_scratch
    lb_scratch = runner._host_engram_lookback_scratch
    assert res1.input_ids is ids_scratch
    assert res1.positions is pos_scratch
    assert res1.lookback_token_ids is lb_scratch

    # Second step with fewer active requests (2 instead of 4) reuses the exact
    # same scratch tensors and restores the previously dirtied tail [12:24].
    res2 = he.capture_host_engram_inputs(runner, 2)
    assert res2 is not None and res2.host_ready is True
    assert res2.input_ids is ids_scratch
    assert res2.positions is pos_scratch
    assert res2.lookback_token_ids is lb_scratch
    assert torch.equal(res2.input_ids[12:24], runner.input_ids_cpu[12:24])
    assert torch.equal(res2.positions[12:24], runner.positions_cpu[12:24])


def test_td8246_host_engram_zero_alloc_fast_paths():
    runner = _make_runner(8, k=5, depth=4, seed=99)
    res1 = he.capture_host_engram_inputs(runner, 8)
    assert res1 is not None and res1.host_ready is True
    spec_ws = runner._host_engram_spec_ws
    lb_ws = runner._host_engram_lookback_ws
    offsets = runner._host_engram_lookback_offsets
    local_rows = runner._host_engram_local_rows

    # Second call reuses the exact same preallocated NumPy workspace buffers
    res2 = he.capture_host_engram_inputs(runner, 8)
    assert res2 is not None and res2.host_ready is True
    assert runner._host_engram_spec_ws is spec_ws
    assert runner._host_engram_lookback_ws is lb_ws
    assert runner._host_engram_lookback_offsets is offsets
    assert runner._host_engram_local_rows is local_rows
    ids_ref, pos_ref, lb_ref = _reference(runner, 8)
    np.testing.assert_array_equal(res2.input_ids.numpy(), ids_ref)
    np.testing.assert_array_equal(res2.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(res2.lookback_token_ids.numpy(), lb_ref)

    # Also verify non-uniform num_draft_per_req with full 0..R-1 keys
    runner._pre_async_results.num_draft_per_req[3] = 2
    res3 = he.capture_host_engram_inputs(runner, 8)
    assert res3 is not None and res3.host_ready is True
    ids_ref3, pos_ref3, lb_ref3 = _reference(runner, 8)
    np.testing.assert_array_equal(res3.input_ids.numpy(), ids_ref3)
    np.testing.assert_array_equal(res3.positions.numpy(), pos_ref3)
    np.testing.assert_array_equal(res3.lookback_token_ids.numpy(), lb_ref3)


def test_td8247_capture_host_engram_contiguous_fast_path():
    runner = _make_runner(8, k=5, depth=4, seed=123)
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(8)}
    runner._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)
    # 1. Steady-state identity-order & uniform-stride fast path
    res_fast1 = he.capture_host_engram_inputs(runner, 8)
    res_fast = he.capture_host_engram_inputs(runner, 8)
    assert (
        res_fast1 is not None and res_fast is not None and res_fast.host_ready is True
    )
    ids_ref, pos_ref, lb_ref = _reference(runner, 8)
    np.testing.assert_array_equal(res_fast.input_ids.numpy(), ids_ref)
    np.testing.assert_array_equal(res_fast.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(res_fast.lookback_token_ids.numpy(), lb_ref)

    # Shrinking active_len under full_overwrite=True resyncs the dirty tail [24:48]
    res_shrunk = he.capture_host_engram_inputs(runner, 4)
    assert res_shrunk is not None and res_shrunk.host_ready is True
    ids_ref_s, pos_ref_s, lb_ref_s = _reference(runner, 4)
    np.testing.assert_array_equal(res_shrunk.input_ids.numpy(), ids_ref_s)
    np.testing.assert_array_equal(res_shrunk.positions.numpy(), pos_ref_s)
    np.testing.assert_array_equal(
        res_shrunk.lookback_token_ids.numpy()[:4], lb_ref_s[:4]
    )

    # 2. Reordered requests (all_matched=True, is_identity=False, uniform_stride=False)
    runner_reord = _make_runner(8, k=5, depth=4, seed=123)
    res_reord = he.capture_host_engram_inputs(runner_reord, 8)
    assert res_reord is not None and res_reord.host_ready is True
    ids_ref_r, pos_ref_r, lb_ref_r = _reference(runner_reord, 8)
    np.testing.assert_array_equal(res_reord.input_ids.numpy(), ids_ref_r)
    np.testing.assert_array_equal(res_reord.positions.numpy(), pos_ref_r)
    np.testing.assert_array_equal(res_reord.lookback_token_ids.numpy(), lb_ref_r)


def test_td8283_host_engram_int32_zero_copy_and_scratch_dtypes(monkeypatch):
    # 1. _ragged_arange returns np.int32 across uniform, non-uniform, and empty inputs
    u_arr = he._ragged_arange(
        np.array([10, 20], dtype=np.int64), np.array([2, 2], dtype=np.int64)
    )
    assert u_arr.dtype == np.int32 and u_arr.tolist() == [10, 11, 20, 21]
    nu_arr = he._ragged_arange(
        np.array([10, 20, 30], dtype=np.int64), np.array([2, 0, 3], dtype=np.int64)
    )
    assert nu_arr.dtype == np.int32 and nu_arr.tolist() == [10, 11, 30, 31, 32]
    e_arr = he._ragged_arange(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert e_arr.dtype == np.int32 and e_arr.tolist() == []

    # 2. _dummy_host_inputs allocates int32 positions
    dummy_runner = SimpleNamespace(
        max_num_tokens=16,
        max_num_reqs=4,
        lookback_token_ids=SimpleNamespace(cpu=torch.zeros((4, 4), dtype=torch.int32)),
    )
    dummy_ctx = SimpleNamespace(
        position_ids_override=torch.zeros(8, dtype=torch.int32),
        query_start_loc=None,
        num_reqs=2,
    )
    dummy_inputs = he._dummy_host_inputs(dummy_runner, dummy_ctx)
    assert dummy_inputs.positions.dtype == torch.int32

    # 3. Spec state, lookback, and capture use np.int32 scratch & zero-copy views
    seen_lookback_calls = []
    orig_lookback = he._resolve_decode_lookback_cpu

    def spy_lookback(
        runner,
        batch,
        num_reqs,
        start_index,
        *,
        rejected_by_req=None,
        sampled_overlay=None,
        raw_starts=None,
    ):
        seen_lookback_calls.append((rejected_by_req, sampled_overlay, raw_starts))
        return orig_lookback(
            runner,
            batch,
            num_reqs,
            start_index,
            rejected_by_req=rejected_by_req,
            sampled_overlay=sampled_overlay,
            raw_starts=raw_starts,
        )

    monkeypatch.setattr(he, "_resolve_decode_lookback_cpu", spy_lookback)

    for with_no_spec in (True, False):
        for identity in (True, False):
            runner = _make_runner(8, k=5, depth=4, seed=42, with_no_spec=with_no_spec)
            runner.input_batch.num_computed_tokens_cpu = (
                runner.input_batch.num_computed_tokens_cpu.astype(np.int32)
            )
            if runner.input_batch.num_tokens_no_spec is not None:
                runner.input_batch.num_tokens_no_spec = (
                    runner.input_batch.num_tokens_no_spec.astype(np.int32)
                )
            if identity:
                runner._pre_async_results.req_id_to_index_copy = {
                    f"req-{i}": i for i in range(8)
                }
                runner._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)
            else:
                runner._pre_async_results.num_draft_per_req[0] = 3

            seen_lookback_calls.clear()
            res = he.capture_host_engram_inputs(runner, 8)
            assert res is not None and res.host_ready is True

            _, lengths_ws, num_draft_ws, rejected_ws = runner._host_engram_spec_ws
            assert lengths_ws.dtype == np.int32
            assert num_draft_ws.dtype == np.int32
            assert rejected_ws.dtype == np.int32

            assert runner._host_engram_lookback_offsets.dtype == np.int32
            assert runner._host_engram_local_rows.dtype == np.int32
            pos_buf, clipped_buf, _, _ = runner._host_engram_lookback_ws
            assert pos_buf.dtype == np.int32
            assert clipped_buf.dtype == np.int32

            assert len(seen_lookback_calls) == 1
            rej_arg, ov_arg, raw_starts_arg = seen_lookback_calls[0]
            assert rej_arg is not None and rej_arg.dtype == np.int32
            assert ov_arg is not None
            assert ov_arg.starts.dtype == np.int32
            assert ov_arg.lengths.dtype == np.int32
            assert raw_starts_arg is not None and raw_starts_arg.dtype == np.int32
            assert np.shares_memory(
                np.asarray(
                    runner.input_batch.num_computed_tokens_cpu[:8], dtype=np.int32
                ),
                runner.input_batch.num_computed_tokens_cpu,
            )


def test_td8342_host_engram_c_dict_fast_paths_and_lookback_dirty_tracking():
    """TD8.342: C-level dict equality fast paths, single-pass positions subtract,
    and _host_engram_lookback_dirty tracking."""
    runner = _make_runner(8, k=5, depth=4, seed=99)
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(8)}
    runner._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)

    # 1. Steady-state decode with num_reqs=6 < cap_rows=8
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(6)}
    res1 = he.capture_host_engram_inputs(runner, 6)
    assert res1 is not None and res1.host_ready is True
    assert runner._host_engram_lookback_dirty == 6
    assert runner._host_engram_ref_req_dict[0] == tuple(f"req-{i}" for i in range(6))
    assert 5 in runner._host_engram_ref_drafts
    ref_draft_5 = runner._host_engram_ref_drafts[5]

    # Full 8-request step reuses _host_engram_ref_drafts[5] and caches ref_req_8
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(8)}
    res_full1 = he.capture_host_engram_inputs(runner, 8)
    ref_req_8 = runner._host_engram_ref_req_dict
    res_full2 = he.capture_host_engram_inputs(runner, 8)
    assert runner._host_engram_ref_drafts[5] is ref_draft_5
    assert runner._host_engram_ref_req_dict is ref_req_8
    ids_ref, pos_ref, lb_ref = _reference(runner, 8)
    np.testing.assert_array_equal(res_full1.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(res_full2.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(res_full2.lookback_token_ids.numpy(), lb_ref)

    # 2. Steady-state num_reqs=4 < cap_rows=8: skips tail copy on repeat
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(4)}
    res_s1 = he.capture_host_engram_inputs(runner, 4)
    assert runner._host_engram_lookback_dirty == 4
    res_s1.lookback_token_ids.numpy()[4:] = -777
    res_s2 = he.capture_host_engram_inputs(runner, 4)
    assert np.all(res_s2.lookback_token_ids.numpy()[4:] == -777)

    # 3. Identity reqs with non-uniform scheduled tokens (uniform_stride=False)
    runner.query_start_loc_cpu[:5] = torch.tensor([0, 6, 10, 16, 22], dtype=torch.int32)
    res_hetero = he.capture_host_engram_inputs(runner, 4)
    assert res_hetero is not None and res_hetero.host_ready is True
    ids_ref_h, pos_ref_h, lb_ref_h = _reference(runner, 4)
    np.testing.assert_array_equal(res_hetero.input_ids.numpy()[:22], ids_ref_h[:22])
    np.testing.assert_array_equal(res_hetero.positions.numpy()[:22], pos_ref_h[:22])
    np.testing.assert_array_equal(
        res_hetero.lookback_token_ids.numpy()[:4], lb_ref_h[:4]
    )

    # 4. TD8.359: All-accepted valid.all() fast path & pointer-identity ref_req[3] reuse
    runner_all = _make_runner(8, k=5, depth=4, seed=77)
    sampled_all = torch.ones((8, 6), dtype=torch.int32) * 19
    runner_all._pre_async_results.wait_for_copy = lambda: sampled_all
    shared_req_map = {f"req-{i}": i for i in range(8)}
    runner_all._pre_async_results.req_id_to_index_copy = shared_req_map
    runner_all._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)
    out_all1 = he.capture_host_engram_inputs(runner_all, 8)
    assert out_all1 is not None and out_all1.host_ready is True
    assert runner_all._host_engram_ref_req_dict[3] is shared_req_map
    # Second call with same shared_req_map hits `req_id_to_index is ref_req[3]`
    out_all2 = he.capture_host_engram_inputs(runner_all, 8)
    assert out_all2 is not None and out_all2.host_ready is True
    ids_all_ref, pos_all_ref, lb_all_ref = _reference(runner_all, 8)
    np.testing.assert_array_equal(out_all2.input_ids.numpy(), ids_all_ref)
    np.testing.assert_array_equal(out_all2.positions.numpy(), pos_all_ref)
    np.testing.assert_array_equal(out_all2.lookback_token_ids.numpy(), lb_all_ref)


def test_td8372_full_overlay_lookback_skips_token_ids_cpu_gather() -> None:
    """Verify full-overlay lookback fast path skips batch.token_ids_cpu."""
    from types import SimpleNamespace

    runner = SimpleNamespace(
        lookback_token_ids=SimpleNamespace(cpu=torch.zeros((4, 3), dtype=torch.int32))
    )
    overlay_tokens = np.arange(100, 124, dtype=np.int32).reshape(4, 6)
    overlay = he._SampledOverlay(
        tokens=overlay_tokens,
        starts=np.array([10, 20, 30, 40], dtype=np.int32),
        lengths=np.array([6, 6, 6, 6], dtype=np.int32),
    )

    class _NoTokenIdsBatch:
        num_computed_tokens_cpu = np.array([15, 24, 36, 43], dtype=np.int32)

        @property
        def token_ids_cpu(self):
            raise AssertionError(
                "batch.token_ids_cpu should not be accessed on full-overlay fast path"
            )

    # 1. Full-overlay fast path (delta = [5, 4, 6, 3] >= depth=3 and <= lengths=6)
    lb_fast = he._resolve_decode_lookback_cpu(
        runner,
        _NoTokenIdsBatch(),
        4,
        0,
        sampled_overlay=overlay,
    )
    expected_fast = np.array(
        [
            [overlay_tokens[0, 4], overlay_tokens[0, 3], overlay_tokens[0, 2]],
            [overlay_tokens[1, 3], overlay_tokens[1, 2], overlay_tokens[1, 1]],
            [overlay_tokens[2, 5], overlay_tokens[2, 4], overlay_tokens[2, 3]],
            [overlay_tokens[3, 2], overlay_tokens[3, 1], overlay_tokens[3, 0]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(lb_fast.numpy(), expected_fast)

    # 2. Partial-overlay fallback (req 0 has delta=2 < depth=3)
    token_table = np.arange(1000, 1000 + 4 * 50, dtype=np.int32).reshape(4, 50)
    partial_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([12, 24, 36, 43], dtype=np.int32),
        token_ids_cpu=token_table,
    )
    lb_partial = he._resolve_decode_lookback_cpu(
        runner,
        partial_batch,
        4,
        0,
        sampled_overlay=overlay,
    )
    expected_partial = expected_fast.copy()
    expected_partial[0] = [
        overlay_tokens[0, 1],
        overlay_tokens[0, 0],
        token_table[0, 9],
    ]
    np.testing.assert_array_equal(lb_partial.numpy(), expected_partial)


def test_td8377_ref_qsl_memoryview_fast_path_and_shared_raw_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD8.377: Verify _host_engram_ref_qsl memoryview fast path and raw_starts."""
    runner = _make_runner(8, k=5, depth=4, seed=77, with_no_spec=False)
    runner.input_batch.num_computed_tokens_cpu = (
        runner.input_batch.num_computed_tokens_cpu.astype(np.int32)
    )
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(8)}
    runner._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)

    # 1. First call populates _host_engram_ref_qsl
    out1 = he.capture_host_engram_inputs(runner, 8)
    assert out1 is not None and out1.host_ready is True
    ref_qsl = runner._host_engram_ref_qsl
    assert ref_qsl is not None
    assert ref_qsl[:3] == (8, 6, np.dtype(np.int32))
    assert memoryview(runner.query_start_loc_cpu[:9].numpy()) == memoryview(ref_qsl[3])

    # 2. Second call with unchanged qsl hits memoryview(qsl) == memoryview(ref_qsl[3])
    # and skips np.all during uniform-stride check, while sharing raw_starts
    seen_raw_starts = []
    orig_lookback = he._resolve_decode_lookback_cpu

    def spy_lookback(
        r,
        b,
        n,
        s,
        *,
        rejected_by_req=None,
        sampled_overlay=None,
        raw_starts=None,
    ):
        seen_raw_starts.append(raw_starts)
        return orig_lookback(
            r,
            b,
            n,
            s,
            rejected_by_req=rejected_by_req,
            sampled_overlay=sampled_overlay,
            raw_starts=raw_starts,
        )

    monkeypatch.setattr(he, "_resolve_decode_lookback_cpu", spy_lookback)
    out2 = he.capture_host_engram_inputs(runner, 8)
    assert out2 is not None and out2.host_ready is True
    assert runner._host_engram_ref_qsl is ref_qsl
    assert len(seen_raw_starts) == 1
    assert seen_raw_starts[0] is not None and seen_raw_starts[0].dtype == np.int32
    assert np.shares_memory(
        seen_raw_starts[0], runner.input_batch.num_computed_tokens_cpu
    )
    ids_ref, pos_ref, lb_ref = _reference(runner, 8)
    np.testing.assert_array_equal(out2.input_ids.numpy(), ids_ref)
    np.testing.assert_array_equal(out2.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(out2.lookback_token_ids.numpy(), lb_ref)

    # 3. Non-uniform qsl invalidates fast path and falls back cleanly
    runner.query_start_loc_cpu[:9] = torch.tensor(
        [0, 6, 10, 16, 22, 28, 34, 40, 46], dtype=torch.int32
    )
    out3 = he.capture_host_engram_inputs(runner, 8)
    assert out3 is not None and out3.host_ready is True
    ids_ref3, pos_ref3, lb_ref3 = _reference(runner, 8)
    np.testing.assert_array_equal(out3.input_ids.numpy()[:46], ids_ref3[:46])
    np.testing.assert_array_equal(out3.positions.numpy()[:46], pos_ref3[:46])
    np.testing.assert_array_equal(out3.lookback_token_ids.numpy(), lb_ref3)
