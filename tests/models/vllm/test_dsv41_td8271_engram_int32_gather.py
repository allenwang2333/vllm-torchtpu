# SPDX-License-Identifier: Apache-2.0
"""TD8.271: Engram int32 ``index_select`` gathers are bitwise-equal to int64 indexing.

``NgramHashState.forward`` / ``_hash`` and ``ParallelEngramEmbedding.lookup``
now gather with native int32 flat 1D ``index_select``. The references below are
the pre-TD8.271 int64 advanced-indexing implementations.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram

upstream = common_engram.upstream


class _State(common_engram.NgramHashState):
    def write_cache(self, cache, slots, values, valid):
        cache[slots[valid].long()] = values[valid].to(cache.dtype)


def _ref_hash(
    self,
    tables,
    input_ids,
    positions,
    query_start_loc,
    dead_mask,
    lookback_token_ids,
    lookback_dead_mask,
    fallback,
    mapped_batch=None,
):
    token_map, multipliers, primes, offsets = tables
    count = input_ids.numel()
    _, ngram = multipliers.shape
    dev = input_ids.device
    rows = torch.arange(count, device=dev)
    reqs = (
        (rows.unsqueeze(1) >= query_start_loc[1:].unsqueeze(0))
        .sum(-1)
        .clamp(max=lookback_token_ids.shape[0] - 1)
    )
    starts = query_start_loc[reqs].long().clamp(max=count - 1)
    chunk_positions = positions[starts]
    shifts = torch.arange(ngram, device=dev)
    past = positions[:, None] - shifts
    in_chunk = past >= chunk_positions[:, None]
    indices = (rows[:, None] - shifts).clamp(min=0)
    if mapped_batch is None:
        mapped_batch = token_map[input_ids.long()].masked_fill(
            dead_mask, upstream.DEAD_ID
        )
    batch = mapped_batch[indices]
    cols = chunk_positions[:, None] - 1 - past
    in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
    cols = cols.clamp(min=0, max=self.lookback_depth - 1).long()
    reqs_col = reqs[:, None]
    window_ids = lookback_token_ids[reqs_col, cols]
    known = in_window & (window_ids >= 0)
    mapped_lookback = token_map[lookback_token_ids.clamp(min=0).long()].masked_fill(
        lookback_dead_mask, upstream.DEAD_ID
    )
    window = mapped_lookback[reqs_col, cols]
    source = torch.where(
        in_chunk, batch, torch.where(known, window, fallback(past, reqs_col, window))
    ).long()
    blocked = ((past < 0) | (source == upstream.DEAD_ID)).cumsum(dim=1) > 0
    source = source.masked_fill(blocked, self.pad_id)
    terms = source[:, None, :] * multipliers[None, :, :]
    rolling = terms[:, :, 0]
    steps = []
    for shift in range(1, ngram):
        rolling = rolling ^ terms[:, :, shift]
        steps.append(rolling)
    rolled = torch.stack(steps, dim=2)
    hashed = (rolled[:, :, :, None] % primes[None, :, :, :]).flatten(2) + offsets[
        None, :, :
    ]
    return hashed.to(torch.int32)


def _ref_forward(
    self,
    input_ids,
    positions,
    query_start_loc,
    dead_mask,
    lookback_token_ids,
    lookback_dead_mask,
    slot_mapping,
    block_table,
):
    count = input_ids.numel()
    slot_mapping = slot_mapping[:count]
    mapped = self.token_map[input_ids.long()].masked_fill(dead_mask, upstream.DEAD_ID)
    self.write_cache(self._cache, slot_mapping, mapped, slot_mapping >= 0)

    def fallback(past, reqs, window):
        logical = past.clamp(
            min=0, max=block_table.shape[1] * self.block_size - 1
        ).long()
        blocks = block_table[reqs, logical // self.block_size].long()
        slots = (blocks * self.block_size + logical % self.block_size).clamp(
            min=0, max=self._cache.numel() - 1
        )
        return self._cache[slots]

    tables = (self.token_map, self.multipliers, self.primes, self.offsets)
    return _ref_hash(
        self,
        tables,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        fallback,
        mapped_batch=mapped,
    )


def _state(seed, block_size=4, pages=16):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(_State)
    nn.Module.__init__(state)
    state.token_map = torch.randint(1, 500, (1000,), generator=g)
    state.token_map[7] = upstream.DEAD_ID
    state.multipliers = torch.randint(1, 1 << 20, (2, 3), generator=g)
    state.primes = torch.randint(1000, 5000, (2, 2, 4), generator=g)
    state.offsets = torch.randint(0, 9999, (2, 8), generator=g)
    state.lookback_depth = 3
    state.pad_id = 0
    state.use_slot_cache = True
    state.block_size = block_size
    state._cache = torch.randint(1, 500, (pages * block_size,), generator=g)
    state._kv_cache_ref = None
    state.swa_cache_module = SimpleNamespace()
    return state, g


def _inputs(g, idx_dtype, block_size=4, pages=16):
    # 3 requests: a decode at a deep position, a prefill chunk mid-sequence and
    # a fresh prefill; the lookback window only covers part of the history so
    # the slot-cache fallback is exercised, plus unknown (-1) and dead ids.
    lens = [1, 4, 3]
    starts = [9, 5, 0]
    qsl = torch.tensor([0, 1, 5, 8], dtype=idx_dtype)
    pos = torch.cat(
        [torch.arange(s, s + n) for s, n in zip(starts, lens, strict=True)]
    ).to(idx_dtype)
    count = sum(lens)
    ids = torch.randint(0, 1000, (count,), generator=g).to(idx_dtype)
    ids[2] = 7
    dead = torch.zeros(count, dtype=torch.bool)
    dead[6] = True
    lookback = torch.randint(0, 1000, (3, 3), generator=g).to(idx_dtype)
    lookback[0, 1] = -1
    lookback[1, 2] = 7
    dead_lb = torch.zeros((3, 3), dtype=torch.bool)
    dead_lb[0, 0] = True
    width = 4
    block_table = torch.randperm(pages, generator=g)[: 3 * width].view(3, width)
    block_table = block_table.to(idx_dtype)
    slots = torch.randint(0, pages * block_size, (count + 2,), generator=g)
    slots[1] = -1
    return ids, pos, qsl, dead, lookback, dead_lb, slots.to(idx_dtype), block_table


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_bitwise_matches_int64_reference(idx_dtype, seed):
    new, g = _state(seed)
    ref, _ = _state(seed)
    args = _inputs(g, idx_dtype)
    out = new.forward(*args)
    expected = _ref_forward(ref, *args)
    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    assert torch.equal(new._cache, ref._cache)


def test_forward_noncontiguous_tables_match_reference():
    new, g = _state(5)
    ref, _ = _state(5)
    ids, pos, qsl, dead, lookback, dead_lb, slots, block_table = _inputs(g, torch.int64)
    lookback_nc = torch.cat([lookback, lookback], dim=1)[:, ::2]
    block_table_nc = block_table.t().contiguous().t()
    assert not lookback_nc.is_contiguous() and not block_table_nc.is_contiguous()
    args = (ids, pos, qsl, dead, lookback_nc, dead_lb, slots, block_table_nc)
    assert torch.equal(new.forward(*args), _ref_forward(ref, *args))


def _emb(pad):
    emb = object.__new__(common_engram.ParallelEngramEmbedding)
    emb.head_start = 2
    emb.part_n_hash_cols = 3
    emb.n_hash_cols = 5 - pad
    emb.vocab_start_idx = 40
    emb.vocab_end_idx = 90
    emb.part_num_embeddings = 50
    emb.dim = 16
    emb.block_size = 8
    emb._host_rows = None
    return emb


@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_lookup_bitwise_matches_int64_reference(pad, idx_dtype):
    g = torch.Generator().manual_seed(pad)
    emb = _emb(pad)
    weight = torch.randn((50, 16), generator=g).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 135, (50, 2), generator=g).to(torch.uint8)
    emb._storage = lambda: (weight, scales)
    indices = torch.randint(0, 130, (6, 5), generator=g).to(idx_dtype)

    rows, valid = emb._select_local_heads(indices)
    assert rows.dtype == torch.int32

    # Pre-TD8.271 reference: int64 rows and 2D advanced indexing.
    start, end = 2, min(5, 5 - pad)
    sel = indices[:, start:end].long()
    ref_valid = (sel >= 40) & (sel < 90)
    if pad:
        sel = torch.nn.functional.pad(sel, (0, pad))
        ref_valid = torch.nn.functional.pad(ref_valid, (0, pad), value=False)
    ref_rows = (sel - 40).clamp(min=0, max=49)
    assert torch.equal(rows.long(), ref_rows)
    assert torch.equal(valid, ref_valid)
    values = weight[ref_rows].unflatten(-1, (-1, 8)).float()
    scale = (scales[ref_rows].to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    expected = ((values * scale).flatten(-2).masked_fill(~ref_valid[:, :, None], 0)).to(
        torch.bfloat16
    )

    out = torch.empty(6, 3, 16, dtype=torch.bfloat16)
    emb.lookup(indices, out)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_td8314_select_local_heads_int32_cast_guard():
    """TD8.314: Verify _select_local_heads skips .int() when indices is already
    torch.int32 and casts before slicing when indices is torch.int64."""
    emb = _emb(pad=0)
    idx_i32 = torch.tensor(
        [[40, 45, 50, 55, 60], [10, 41, 89, 95, 42]], dtype=torch.int32
    )
    idx_i64 = idx_i32.to(torch.int64)

    class _NoIntCastI32:
        dtype = torch.int32

        def __getitem__(self, item):
            return idx_i32[item]

        def int(self):
            raise AssertionError(".int() must not be called when dtype == torch.int32")

    rows_guard, valid_guard = emb._select_local_heads(_NoIntCastI32())
    rows_i32, valid_i32 = emb._select_local_heads(idx_i32)
    rows_i64, valid_i64 = emb._select_local_heads(idx_i64)
    assert rows_guard.dtype == rows_i32.dtype == rows_i64.dtype == torch.int32
    assert torch.equal(rows_guard, rows_i32)
    assert torch.equal(rows_i32, rows_i64)
    assert torch.equal(valid_guard, valid_i32)
    assert torch.equal(valid_i32, valid_i64)
