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
    assert torch.equal(res.input_ids, runner._pre_async_results.next_tokens_tpu[
        torch.from_numpy(he._ragged_arange(
            np.array([runner._pre_async_results.req_id_to_index_copy[f"req-{i}"] * 6
                      for i in range(6)]), np.full(6, 6)))
    ])  # fmt: skip


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
