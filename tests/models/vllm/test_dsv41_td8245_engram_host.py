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
