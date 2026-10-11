# SPDX-License-Identifier: Apache-2.0
"""TD8.245: ``OFFLOAD=1`` host critical path of the Engram hash and gather.

``NgramHashState.hash_on_host`` now memoizes each row's request and start for
a repeated ``query_start_loc``, reuses its lookback window buffer and adds the
table offsets while narrowing straight into the ``[T, layers, heads]`` int32
output; ``ParallelEngramEmbedding.gather_raw`` reuses its storage views. These
tests pin bitwise parity with the previous expressions and the cache rules.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as ce


def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    s = object.__new__(ce.NgramHashState)
    nn.Module.__init__(s)
    s.token_map = torch.randint(1, 5000, (5000,), generator=g)
    s.multipliers = torch.randint(1, 1 << 40, (4, 4), generator=g) * 2 + 1
    s.primes = torch.randint(1000, 90_000, (4, 3, 8), generator=g)
    s.offsets = torch.randint(0, 1 << 22, (4, 24), generator=g)
    s.lookback_depth = 3
    s.pad_id = 0
    s.use_slot_cache = False
    return s


def _inputs(reqs, tokens_per_req, seed=0, dead_frac=0.0):
    rng = np.random.default_rng(seed)
    count = reqs * tokens_per_req
    ids = rng.integers(0, 5000, count, dtype=np.int32)
    start = rng.integers(0, 40, reqs)
    pos = np.repeat(start, tokens_per_req) + np.tile(np.arange(tokens_per_req), reqs)
    qsl = np.arange(0, count + 1, tokens_per_req, dtype=np.int32)
    dead = rng.random(count) < dead_frac
    lookback = rng.integers(-1, 5000, (reqs, 3), dtype=np.int32)
    lb_dead = rng.random((reqs, 3)) < dead_frac
    return (
        torch.from_numpy(ids),
        torch.from_numpy(pos.astype(np.int64)),
        torch.from_numpy(qsl),
        torch.from_numpy(dead),
        torch.from_numpy(lookback),
        torch.from_numpy(lb_dead),
    )


@pytest.mark.parametrize("dead_frac", [0.0, 0.2])
@pytest.mark.parametrize("reqs,tpr", [(1, 4), (5, 6), (32, 6), (7, 3)])
def test_hash_on_host_matches_device_path_formula(reqs, tpr, dead_frac, monkeypatch):
    """Bitwise equal to the old int64-add + astype tail on the same remainder."""
    state = _state()
    args = _inputs(reqs, tpr, dead_frac=dead_frac)
    captured = {}
    orig_add = np.add

    def spy_add(a, b, *rest, **kw):
        if kw.get("casting") == "unsafe" and "out" in kw:
            captured["hashed"] = np.array(a, copy=True)
            captured["offsets"] = np.array(b, copy=True)
        return orig_add(a, b, *rest, **kw)

    monkeypatch.setattr(ce.np, "add", spy_add)
    got = state.hash_on_host(*args)
    monkeypatch.setattr(ce.np, "add", orig_add)
    hashed = captured["hashed"] + captured["offsets"]
    layers = hashed.shape[0]
    want = hashed.reshape(layers, -1, args[0].numel()).transpose(2, 0, 1)
    want = want.astype(np.int32, order="C")
    assert got.dtype == torch.int32 and got.is_contiguous()
    assert np.array_equal(got.numpy(), want)
    # The returned array is fresh per call: a second call does not alias it.
    again = state.hash_on_host(*args)
    assert again.data_ptr() != got.data_ptr()
    assert torch.equal(again, got)


def test_hash_on_host_matches_device_forward():
    """End-to-end: the host hash equals the device ``_hash`` path."""
    state = _state(1)
    ids, pos, qsl, dead, lookback, lb_dead = _inputs(6, 6, seed=3)
    host = state.hash_on_host(ids, pos, qsl, dead, lookback, lb_dead)
    tables = (state.token_map, state.multipliers, state.primes, state.offsets)
    dev = state._hash(
        tables,
        ids,
        pos,
        qsl,
        dead,
        lookback,
        lb_dead,
        lambda past, reqs_col, window: torch.full_like(window, state.pad_id),
    )
    assert torch.equal(host, dev)


def test_request_rows_memo_hit_and_invalidation():
    tables = ce._HostHashTables.from_state(_state())
    rows = np.arange(12)
    qsl = np.array([0, 6, 12], dtype=np.int32)
    reqs, starts, base = tables.request_rows(qsl, rows, 12, 2, 3)
    assert reqs.tolist() == [0] * 6 + [1] * 6
    assert starts.tolist() == [0] * 6 + [6] * 6
    assert base.tolist() == [[0] * 6 + [4] * 6]  # TD8.248: req * (depth + 1)
    # Same layout (a different array object): cache hit, same arrays.
    again = tables.request_rows(qsl.copy(), rows, 12, 2, 3)
    assert again[0] is reqs and again[1] is starts and again[2] is base
    # A changed layout, active count, token count or depth recomputes.
    moved = tables.request_rows(np.array([0, 4, 12], np.int32), rows, 12, 2, 3)
    assert moved[0].tolist() == [0] * 4 + [1] * 8
    clamped = tables.request_rows(np.array([0, 4, 12], np.int32), rows, 12, 1, 3)
    assert clamped[0].tolist() == [0] * 12
    shorter = tables.request_rows(np.array([0, 4, 8], np.int32), rows[:8], 8, 2, 3)
    assert shorter[1].tolist() == [0] * 4 + [4] * 4
    deeper = tables.request_rows(np.array([0, 4, 8], np.int32), rows[:8], 8, 2, 5)
    assert deeper[2].tolist() == [[0] * 4 + [6] * 4]


def test_window_table_is_reused_per_shape():
    tables = ce._HostHashTables.from_state(_state())
    first = tables.window_table(4, 3)
    assert first.shape == (4, 4) and first.dtype == tables.token_map.dtype
    assert tables.window_table(4, 3) is first
    other = tables.window_table(5, 3)
    assert other.shape == (5, 4) and other is not first


def _embedding(weight, scales):
    emb = object.__new__(ce.ParallelEngramEmbedding)
    emb.head_start = 0
    emb.part_n_hash_cols = 2
    emb.n_hash_cols = 4
    emb.vocab_start_idx = 0
    emb.vocab_end_idx = 100
    emb.part_num_embeddings = 100
    emb.dim = 32
    emb.block_size = 16
    storage = {"value": (weight, scales)}
    emb._storage = lambda: storage["value"]
    return emb, storage


def test_gather_raw_reuses_storage_views_until_storage_changes():
    weight = torch.arange(100 * 32, dtype=torch.int32).remainder(251).to(torch.uint8)
    weight = weight.reshape(100, 32)
    scales = torch.arange(200, dtype=torch.int32).remainder(250).to(torch.uint8)
    scales = scales.reshape(100, 2)
    emb, storage = _embedding(weight, scales)
    indices = torch.tensor([[10, 20, 200, 30]], dtype=torch.int32)
    raw = torch.zeros((1, 2, 32), dtype=torch.uint8)
    scale = torch.zeros((1, 2, 2), dtype=torch.uint8)
    emb.gather_raw(indices, raw, scale)
    views = emb._raw_views
    assert torch.equal(raw[0, 0], weight[10]) and torch.equal(raw[0, 1], weight[20])
    assert torch.equal(scale[0], scales[[10, 20]])
    emb.gather_raw(indices, raw, scale)
    assert emb._raw_views is views
    # New storage tensors (e.g. a reload) rebuild the views.
    weight2, scales2 = weight.flip(0).contiguous(), scales.flip(0).contiguous()
    storage["value"] = (weight2, scales2)
    emb.gather_raw(indices, raw, scale)
    assert emb._raw_views is not views
    assert torch.equal(raw[0, 0], weight2[10])
    assert torch.equal(scale[0], scales2[[10, 20]])
    storage["value"] = (weight2, scales)
    emb.gather_raw(indices, raw, scale)
    assert emb._raw_views[1] is scales
    assert torch.equal(scale[0], scales[[10, 20]])


def test_td8248_work_buffers_are_reused_per_count():
    tables = ce._HostHashTables.from_state(_state())
    first = tables.work(12)
    assert [b.shape for b in first] == [(4, 12)] * 5
    assert [b.dtype for b in first] == [
        np.dtype(np.int32),
        np.dtype(np.bool_),
        tables.token_map.dtype,
        np.dtype(np.bool_),
        np.dtype(np.bool_),
    ]
    assert tables.work(12) is first
    assert tables.work(8)[0].shape == (4, 8)


def test_td8248_select_local_heads_np_clamps_in_place():
    emb, _ = _embedding(torch.zeros(100, 32, dtype=torch.uint8), torch.zeros(100, 2))
    emb.part_n_hash_cols = 3  # one padded head (n_hash_cols - head_start = 4 > 3)
    emb.head_start = 2
    indices = np.array([[1, 2, 5, 150], [3, 4, -7, 99]], dtype=np.int32)
    rows, valid = emb._select_local_heads_np(indices)
    assert rows.tolist() == [[5, 99, 0], [0, 99, 0]]
    assert valid.tolist() == [[True, False, False], [False, True, False]]


def test_td8294_host_hash_and_select_local_heads_np_int32_buffers(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _state(seed=7)
    tables = ce._HostHashTables.from_state(state)
    assert tables.shifts.dtype == np.int32
    _, rows, _, _, _ = tables._ensure_scratch(12)
    assert rows.dtype == np.int32
    qsl_i64 = np.array([0, 6, 12], dtype=np.int64)
    reqs, starts, req_base = tables.request_rows(qsl_i64, rows, 12, 2, 3)
    assert reqs.dtype == np.int32
    assert starts.dtype == np.int32
    assert req_base.dtype == np.int32
    cols, _, _, _, _ = tables.work(12)
    assert cols.dtype == np.int32

    emb, _ = _embedding(
        torch.arange(100 * 32, dtype=torch.int32)
        .remainder(251)
        .to(torch.uint8)
        .reshape(100, 32),
        torch.arange(200, dtype=torch.int32)
        .remainder(250)
        .to(torch.uint8)
        .reshape(100, 2),
    )
    emb.vocab_start_idx = 10
    emb.vocab_end_idx = 110
    for idx_dtype in (np.int32, np.int64):
        idx_np = np.array([[10, 50, 120, 5], [9, 109, 110, -3]], dtype=idx_dtype)
        sel_rows, sel_valid = emb._select_local_heads_np(idx_np)
        assert sel_rows.dtype == np.int32
        assert sel_valid.dtype == np.bool_
        assert sel_rows.tolist() == [[0, 40], [0, 99]]
        assert sel_valid.tolist() == [[True, True], [False, True]]

    # Verify gather_raw passes torch.int32 index tensor to torch.index_select
    seen_index_dtypes: list[torch.dtype] = []
    orig_index_select = torch.index_select

    def spy_index_select(input_t, dim, index_t, *, out=None):
        seen_index_dtypes.append(index_t.dtype)
        return orig_index_select(input_t, dim, index_t, out=out)

    monkeypatch.setattr(torch, "index_select", spy_index_select)
    raw_out = torch.zeros((2, 2, 32), dtype=torch.uint8)
    scale_out = torch.zeros((2, 2, 2), dtype=torch.uint8)
    emb.gather_raw(
        torch.tensor([[10, 50, 120, 5], [9, 109, 110, -3]], dtype=torch.int64),
        raw_out,
        scale_out,
    )
    assert seen_index_dtypes == [torch.int32]


def test_td8340_request_rows_array_equal_and_gather_bufs_reuse():
    """TD8.340: request_rows stores qsl_i32.copy() and compares via np.array_equal;
    ParallelEngramEmbedding reuses (_gather_bufs) across steady-shape steps."""
    tables = ce._HostHashTables.from_state(_state(seed=11))
    rows = np.arange(12, dtype=np.int32)
    qsl = np.array([0, 6, 12], dtype=np.int32)
    reqs1, starts1, base1 = tables.request_rows(qsl, rows, 12, 2, 3)
    cached = tables._memo["request_rows"]
    assert cached[0] == (12, 2, 3)
    assert cached[1] is not qsl and cached[1].dtype == np.int32
    assert np.array_equal(cached[1], qsl)
    # In-place mutation of caller's qsl invalidates cache on next call
    qsl[1] = 4
    reqs2, starts2, _ = tables.request_rows(qsl, rows, 12, 2, 3)
    assert reqs2 is not reqs1
    assert reqs2.tolist() == [0] * 4 + [1] * 8
    assert starts2.tolist() == [0] * 4 + [4] * 8

    weight = (
        torch.arange(100 * 32, dtype=torch.int32)
        .remainder(251)
        .to(torch.uint8)
        .reshape(100, 32)
    )
    scales = (
        torch.arange(200, dtype=torch.int32)
        .remainder(250)
        .to(torch.uint8)
        .reshape(100, 2)
    )
    emb, _ = _embedding(weight, scales)
    emb.vocab_start_idx = 10
    emb.vocab_end_idx = 110
    emb.head_start = 2
    emb.part_n_hash_cols = 3  # 2 active cols + 1 padded col

    idx1 = torch.tensor([[1, 2, 15, 25], [3, 4, 5, 109]], dtype=torch.int32)
    raw_out = torch.zeros((2, 3, 32), dtype=torch.uint8)
    scale_out = torch.zeros((2, 3, 2), dtype=torch.uint8)
    emb.gather_raw(idx1, raw_out, scale_out)
    bufs1 = emb._gather_bufs
    assert bufs1[0] == (2, 3, 2)
    assert torch.equal(raw_out[0, 0], weight[5])
    assert torch.equal(raw_out[0, 1], weight[15])
    assert torch.equal(scale_out[0, :2], scales[[5, 15]])
    assert torch.all(scale_out[:, 2] == ce._INVALID_SCALE)
    assert torch.all(scale_out[1, 0] == ce._INVALID_SCALE)
    assert torch.equal(scale_out[1, 1], scales[99])

    idx2 = torch.tensor([[1, 2, 30, 40], [3, 4, 50, 200]], dtype=torch.int32)
    emb.gather_raw(idx2, raw_out, scale_out)
    assert emb._gather_bufs is bufs1
    assert torch.equal(raw_out[0, 0], weight[20])
    assert torch.equal(raw_out[0, 1], weight[30])
    assert torch.equal(raw_out[1, 0], weight[40])
    assert torch.all(scale_out[1, 1:] == ce._INVALID_SCALE)


def test_td8368_hash_on_host_all_valid_fast_path_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    """TD8.368: Verify hash_on_host skips blocked mask construction, propagation,
    and pad_id copyto when not dead_np.any(), not lb_dead_np.any(), and
    pos_np.min() >= ngram - 1, while falling back cleanly on any invalid/pre-seq
    history."""
    state = _state(seed=68)
    ids, pos, qsl, dead, lookback, lb_dead = _inputs(4, 6, seed=68, dead_frac=0.0)
    pos = pos + 100  # Ensure all positions >= ngram - 1
    lookback = lookback.clamp(min=0)
    tables = (state.token_map, state.multipliers, state.primes, state.offsets)

    def _ref(i_t, p_t, q_t, d_t, lb_t, lbd_t):
        return state._hash(
            tables,
            i_t,
            p_t,
            q_t,
            d_t,
            lb_t,
            lbd_t,
            lambda past, reqs_col, window: torch.full_like(window, state.pad_id),
        )

    # 1) All-valid fast path: np.greater and np.equal must not be called
    greater_calls = 0
    equal_calls = 0
    orig_greater = np.greater
    orig_equal = np.equal

    def spy_greater(*a, **kw):
        nonlocal greater_calls
        greater_calls += 1
        return orig_greater(*a, **kw)

    def spy_equal(*a, **kw):
        nonlocal equal_calls
        equal_calls += 1
        return orig_equal(*a, **kw)

    monkeypatch.setattr(ce.np, "greater", spy_greater)
    monkeypatch.setattr(ce.np, "equal", spy_equal)

    got_fast = state.hash_on_host(ids, pos, qsl, dead, lookback, lb_dead)
    assert greater_calls == 0
    assert equal_calls == 0
    assert torch.equal(got_fast, _ref(ids, pos, qsl, dead, lookback, lb_dead))

    # 2) Fallback when dead_np has a True element
    dead_one = dead.clone()
    dead_one[3] = True
    got_dead = state.hash_on_host(ids, pos, qsl, dead_one, lookback, lb_dead)
    assert greater_calls == 1 and equal_calls == 1
    assert torch.equal(got_dead, _ref(ids, pos, qsl, dead_one, lookback, lb_dead))

    # 3) Fallback when lb_dead_np has a True element
    lb_dead_one = lb_dead.clone()
    lb_dead_one[1, 0] = True
    got_lbd = state.hash_on_host(ids, pos, qsl, dead, lookback, lb_dead_one)
    assert greater_calls == 2 and equal_calls == 2
    assert torch.equal(got_lbd, _ref(ids, pos, qsl, dead, lookback, lb_dead_one))

    # 4) Fallback when pos_np[0] >= ngram - 1 but a later request has pos < ngram - 1
    pos_mixed = pos.clone()
    pos_mixed[6:12] = torch.arange(6, dtype=pos.dtype)
    got_mixed = state.hash_on_host(ids, pos_mixed, qsl, dead, lookback, lb_dead)
    assert greater_calls == 3 and equal_calls == 3
    assert torch.equal(got_mixed, _ref(ids, pos_mixed, qsl, dead, lookback, lb_dead))


def test_td8374_request_rows_cols_and_hash_on_host_dead_mask_fast_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    """TD8.374: Verify _HostHashTables.request_rows memoizes request_rows_cols
    (default_offset, default_cols, default_in_chunk) and hash_on_host reuses it
    on contiguous chunk offsets while falling back on non-contiguous offsets,
    and skips no-op dead/lb_dead/lb_np<0 mask writes when clean."""
    state = _state(seed=74)
    ids, pos, qsl, dead, lookback, lb_dead = _inputs(4, 6, seed=74, dead_frac=0.0)
    pos = pos + 100
    lookback_pos = lookback.clamp(min=0)
    tables = (state.token_map, state.multipliers, state.primes, state.offsets)

    def _ref(i_t, p_t, q_t, d_t, lb_t, lbd_t):
        return state._hash(
            tables,
            i_t,
            p_t,
            q_t,
            d_t,
            lb_t,
            lbd_t,
            lambda past, reqs_col, window: torch.full_like(window, state.pad_id),
        )

    # First call populates _host_tables and _memo["request_rows_cols"]
    got1 = state.hash_on_host(ids, pos, qsl, dead, lookback_pos, lb_dead)
    assert torch.equal(got1, _ref(ids, pos, qsl, dead, lookback_pos, lb_dead))
    host_tables = state._host_tables
    assert "request_rows_cols" in host_tables._memo
    rc_memo = host_tables._memo["request_rows_cols"]
    assert len(rc_memo) == 3

    # Steady-state contiguous offset: np.subtract and np.less should not be called
    sub_calls = 0
    less_calls = 0
    orig_sub = np.subtract
    orig_less = np.less

    def spy_sub(*a, **kw):
        nonlocal sub_calls
        sub_calls += 1
        return orig_sub(*a, **kw)

    def spy_less(*a, **kw):
        nonlocal less_calls
        less_calls += 1
        return orig_less(*a, **kw)

    monkeypatch.setattr(ce.np, "subtract", spy_sub)
    monkeypatch.setattr(ce.np, "less", spy_less)

    got2 = state.hash_on_host(ids, pos + 6, qsl, dead, lookback_pos, lb_dead)
    assert sub_calls == 0
    assert less_calls == 0
    assert torch.equal(got2, _ref(ids, pos + 6, qsl, dead, lookback_pos, lb_dead))

    # Non-contiguous offset fallback: np.subtract (2x) and np.less (1x) are called
    pos_gap = pos.clone()
    pos_gap[5] += 2
    got_gap = state.hash_on_host(ids, pos_gap, qsl, dead, lookback_pos, lb_dead)
    assert sub_calls == 2
    assert less_calls == 1
    assert torch.equal(got_gap, _ref(ids, pos_gap, qsl, dead, lookback_pos, lb_dead))
    # Verify memoized default_cols/default_in_chunk were not mutated by fallback
    got3 = state.hash_on_host(ids, pos, qsl, dead, lookback_pos, lb_dead)
    assert torch.equal(got3, got1)

    # Also exercise negative lookback token (< 0) fallback with dead and lb_dead
    lookback_neg = lookback_pos.clone()
    lookback_neg[0, 1] = -1
    dead_some = dead.clone()
    dead_some[2] = True
    lb_dead_some = lb_dead.clone()
    lb_dead_some[2, 0] = True
    got_neg = state.hash_on_host(ids, pos, qsl, dead_some, lookback_neg, lb_dead_some)
    assert torch.equal(
        got_neg, _ref(ids, pos, qsl, dead_some, lookback_neg, lb_dead_some)
    )


def test_td8377_ref_qsl_memoryview_fast_path_and_shared_raw_starts():
    """TD8.377: Verify _host_engram_ref_qsl memoryview fast path and shared raw_starts
    in v2.host_engram.capture_host_engram_inputs."""
    from tests.models.vllm.test_dsv41_ext6b_host_engram_capture import (
        _make_runner,
        _reference,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import host_engram as he

    runner = _make_runner(8, k=5, depth=4, seed=77, with_no_spec=False)
    runner.input_batch.num_computed_tokens_cpu = (
        runner.input_batch.num_computed_tokens_cpu.astype(np.int32)
    )
    runner._pre_async_results.req_id_to_index_copy = {f"req-{i}": i for i in range(8)}
    runner._pre_async_results.num_draft_per_req = dict.fromkeys(range(8), 5)

    out1 = he.capture_host_engram_inputs(runner, 8)
    assert out1 is not None and out1.host_ready is True
    ref_qsl = runner._host_engram_ref_qsl
    assert ref_qsl[:3] == (8, 6, np.dtype(np.int32))
    assert memoryview(runner.query_start_loc_cpu[:9].numpy()) == memoryview(ref_qsl[3])

    out2 = he.capture_host_engram_inputs(runner, 8)
    assert out2 is not None and out2.host_ready is True
    assert runner._host_engram_ref_qsl is ref_qsl
    ids_ref, pos_ref, lb_ref = _reference(runner, 8)
    np.testing.assert_array_equal(out2.input_ids.numpy(), ids_ref)
    np.testing.assert_array_equal(out2.positions.numpy(), pos_ref)
    np.testing.assert_array_equal(out2.lookback_token_ids.numpy(), lb_ref)
