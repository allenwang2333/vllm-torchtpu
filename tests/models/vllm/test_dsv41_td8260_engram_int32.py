# SPDX-License-Identifier: Apache-2.0
"""TD8.260: device ``NgramHashState`` hash addressing in native int32.

``_hash`` builds rows / shifts / positions as int32 and the slot-cache
``fallback`` keeps its block divide, modulo and multiply-add in int32; only
gather indices are int64. These tests pin bitwise parity with the previous
int64 expressions (kept verbatim below as the reference).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as ce

upstream = ce.upstream


class _NoWriteState(ce.NgramHashState):
    def write_cache(self, cache, slots, values, valid):
        pass


def _state(use_slot_cache, block_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = object.__new__(_NoWriteState)
    nn.Module.__init__(s)
    s.token_map = torch.randint(1, 3000, (3000,), generator=g)
    s.multipliers = torch.randint(1, 1 << 40, (3, 4), generator=g) * 2 + 1
    s.primes = torch.randint(1000, 90_000, (3, 3, 4), generator=g)
    s.offsets = torch.randint(0, 1 << 22, (3, 12), generator=g)
    s.lookback_depth = 3
    s.pad_id = 0
    s.use_slot_cache = use_slot_cache
    s.block_size = block_size
    s._cache = torch.randint(1, 3000, (64,), generator=g)
    return s


def _old_forward(state, ids, pos, qsl, dead, lookback, lb_dead, block_table):
    """Pre-TD8.260 ``forward`` (no cache write) + ``_hash``, int64 math."""
    token_map, multipliers, primes, offsets = (
        state.token_map,
        state.multipliers,
        state.primes,
        state.offsets,
    )

    def fallback(past, reqs, window):
        if not state.use_slot_cache:
            return torch.full_like(window, state.pad_id)
        logical = past.clamp(
            min=0, max=block_table.shape[1] * state.block_size - 1
        ).long()
        blocks = block_table[reqs, logical // state.block_size].long()
        slots = (blocks * state.block_size + logical % state.block_size).clamp(
            min=0, max=state._cache.numel() - 1
        )
        return state._cache[slots]

    count = ids.numel()
    _, ngram = multipliers.shape
    rows = torch.arange(count)
    reqs = (
        (rows.unsqueeze(1) >= qsl[1:].unsqueeze(0))
        .sum(-1)
        .clamp(max=lookback.shape[0] - 1)
    )
    starts = qsl[reqs].long().clamp(max=count - 1)
    chunk_positions = pos[starts]
    shifts = torch.arange(ngram)
    past = pos[:, None] - shifts
    in_chunk = past >= chunk_positions[:, None]
    indices = (rows[:, None] - shifts).clamp(min=0)
    mapped_batch = token_map[ids.long()].masked_fill(dead, upstream.DEAD_ID)
    batch = mapped_batch[indices]
    cols = chunk_positions[:, None] - 1 - past
    in_window = ~in_chunk & (cols >= 0) & (cols < state.lookback_depth)
    cols = cols.clamp(min=0, max=state.lookback_depth - 1).long()
    reqs_col = reqs[:, None]
    window_ids = lookback[reqs_col, cols]
    known = in_window & (window_ids >= 0)
    mapped_lookback = token_map[lookback.clamp(min=0).long()].masked_fill(
        lb_dead, upstream.DEAD_ID
    )
    window = mapped_lookback[reqs_col, cols]
    source = torch.where(
        in_chunk, batch, torch.where(known, window, fallback(past, reqs_col, window))
    ).long()
    blocked = ((past < 0) | (source == upstream.DEAD_ID)).cumsum(dim=1) > 0
    source = source.masked_fill(blocked, state.pad_id)
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


def _inputs(reqs, tpr, seed, pos_dtype):
    rng = np.random.default_rng(seed)
    count = reqs * tpr
    ids = torch.from_numpy(rng.integers(0, 3000, count, dtype=np.int32))
    start = rng.integers(0, 12, reqs)
    pos = np.repeat(start, tpr) + np.tile(np.arange(tpr), reqs)
    qsl = torch.arange(0, count + 1, tpr, dtype=torch.int32)
    dead = torch.from_numpy(rng.random(count) < 0.1)
    lookback = torch.from_numpy(rng.integers(-1, 3000, (reqs, 3), dtype=np.int32))
    lb_dead = torch.from_numpy(rng.random((reqs, 3)) < 0.1)
    block_table = torch.from_numpy(rng.integers(0, 16, (reqs, 6), dtype=np.int32))
    return (
        ids,
        torch.from_numpy(pos).to(pos_dtype),
        qsl,
        dead,
        lookback,
        lb_dead,
        block_table,
    )


@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("use_slot_cache", [False, True])
@pytest.mark.parametrize("reqs,tpr,seed", [(1, 4, 0), (5, 6, 1), (8, 3, 2)])
def test_device_hash_int32_matches_int64_reference(
    reqs, tpr, seed, use_slot_cache, pos_dtype
):
    state = _state(use_slot_cache, seed=seed)
    ids, pos, qsl, dead, lookback, lb_dead, block_table = _inputs(
        reqs, tpr, seed, pos_dtype
    )
    slot_mapping = torch.arange(ids.numel()) if use_slot_cache else None
    got = state.forward(
        ids, pos, qsl, dead, lookback, lb_dead, slot_mapping, block_table
    )
    want = _old_forward(
        state, ids, pos.long(), qsl, dead, lookback, lb_dead, block_table
    )
    assert got.dtype == torch.int32
    assert torch.equal(got, want)
