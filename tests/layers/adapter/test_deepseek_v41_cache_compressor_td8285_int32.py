# SPDX-License-Identifier: Apache-2.0
"""TD8.285: V4.1 adapter cache/compressor int32 addressing equals the int64 one.

``token_requests`` / ``cache_slots`` / ``row_slots`` and the ratio-2 ring in
``save_compress_norm`` now use native int32 rows/requests and flat 1D
``index_select`` gathers. The ``_old_*`` functions are the pre-TD8.285 bodies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_cache as cache_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_compressor as comp_mod,
)


def _old_token_requests(metadata, positions, *, rows=None):
    if rows is None:
        rows = torch.arange(positions.shape[0], device=positions.device)
    return torch.bucketize(rows, metadata.query_start_loc[1:], right=True)


def _old_cache_slots(
    metadata, positions, *, rows_per_page, compress_ratio=1, ring=False, requests=None
):
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = cache_mod.block_table(metadata)
    rows = torch.arange(positions.shape[0], device=positions.device)
    if requests is None:
        requests = _old_token_requests(metadata, positions, rows=rows)
    requests = requests.clamp(max=table.shape[0] - 1)
    logical = positions if compress_ratio == 1 else positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    blocks = table[requests, columns.clamp(max=table.shape[1] - 1).long()].long()
    slots = blocks * rows_per_page + logical % rows_per_page
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


def _old_row_slots(block_table, requests, logical, *, rows_per_page):
    safe = logical.clamp(min=0, max=block_table.shape[1] * rows_per_page - 1)
    blocks = block_table[requests[:, None], safe // rows_per_page].long()
    return blocks * rows_per_page + safe % rows_per_page


def _meta(seed, idx_dtype, lens=(3, 1, 5, 2), width=6, pad=3):
    g = torch.Generator().manual_seed(seed)
    reqs = len(lens)
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=idx_dtype)
    starts = torch.randint(0, 20, (reqs,), generator=g)
    pos = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lens, strict=True)])
    # Padded tail tokens repeat the last real position.
    pos = torch.cat([pos, pos[-1:].expand(pad)]).to(idx_dtype)
    tables = torch.randperm(200, generator=g)[: reqs * width].to(torch.int32)
    meta = SimpleNamespace(
        seq_lens=(starts + torch.tensor(lens)).to(torch.int32),
        block_tables=tables,
        query_start_loc=qsl,
    )
    return meta, pos


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("ratio", "ring", "rows_per_page"), [(1, False, 4), (2, False, 2), (2, True, 8)]
)
def test_cache_slots_and_token_requests_match_int64(
    idx_dtype, seed, ratio, ring, rows_per_page
):
    meta, pos = _meta(seed, idx_dtype)
    reqs = cache_mod.token_requests(meta, pos)
    assert reqs.dtype == torch.int32
    assert torch.equal(reqs.long(), _old_token_requests(meta, pos))
    for given in (None, reqs):
        got = cache_mod.cache_slots(
            meta,
            pos,
            rows_per_page=rows_per_page,
            compress_ratio=ratio,
            ring=ring,
            requests=given,
        )
        ref = _old_cache_slots(
            meta,
            pos,
            rows_per_page=rows_per_page,
            compress_ratio=ratio,
            ring=ring,
            requests=None if given is None else given.long(),
        )
        assert got.dtype == torch.int32
        assert torch.equal(got.long(), ref.long())


def test_cache_slots_empty_metadata_is_int32():
    empty = SimpleNamespace(seq_lens=torch.zeros((0,), dtype=torch.int32))
    out = cache_mod.cache_slots(empty, torch.arange(4), rows_per_page=4)
    assert out.dtype == torch.int32 and torch.equal(out, torch.full((4,), -1).int())


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_row_slots_match_int64(idx_dtype):
    g = torch.Generator().manual_seed(3)
    table = torch.randperm(100, generator=g)[:24].view(4, 6).to(torch.int32)
    requests = torch.tensor([0, 3, 1, 2, 3], dtype=idx_dtype)
    logical = torch.randint(-5, 40, (5, 7), generator=g).to(idx_dtype)
    got = cache_mod.row_slots(table, requests, logical, rows_per_page=4)
    ref = _old_row_slots(table, requests, logical, rows_per_page=4)
    assert got.dtype == torch.int32
    assert torch.equal(got.long(), ref)


def _old_ratio2(kv_score, positions, cache, slot_mapping, qsl, token_to_req, dim):
    """Pre-TD8.285 ratio-2 previous-row selection (CPU ``read_cache``)."""
    kv = kv_score.float()
    count = kv.shape[0]
    capacity = cache.shape[1]
    rows = torch.arange(count)
    requests = token_to_req.long().clamp(max=qsl.numel() - 2)
    starts = qsl[requests]
    previous_slots = (
        slot_mapping.clamp(min=0) // capacity * capacity + (positions - 1) % capacity
    )
    return torch.where(
        (rows > starts)[:, None],
        kv[(rows - 1).clamp(min=0)],
        cache_mod.read_cache(cache, previous_slots),
    )


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_save_compress_norm_ratio2_matches_int64(monkeypatch, pos_dtype):
    g = torch.Generator().manual_seed(7)
    dim = 4
    kv_score = torch.randn(6, 2 * dim, generator=g).to(torch.bfloat16)
    positions = torch.tensor([5, 6, 7, 0, 1, 2], dtype=pos_dtype)
    cache = torch.randn(3, 4, 2 * dim, generator=g)
    slot_mapping = torch.tensor([1, 2, 3, 4, -1, 6], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    expected_prev = _old_ratio2(
        kv_score, positions, cache.clone(), slot_mapping, qsl, token_to_req, dim
    )

    seen = {}
    real_read = comp_mod.read_cache

    def spy_read(c, slots, layout=None):
        seen["slots"] = slots
        return real_read(c, slots, layout=layout)

    monkeypatch.setattr(comp_mod, "read_cache", spy_read)
    latent = torch.empty(6, dim, dtype=torch.bfloat16)
    comp_mod.save_compress_norm(
        kv_score,
        positions,
        cache,
        slot_mapping,
        qsl,
        token_to_req,
        torch.ones(dim),
        1e-6,
        2,
        latent,
    )
    assert seen["slots"].dtype == torch.int32

    # Recompute the expected latent from the reference previous rows.
    kv = kv_score.float()
    scores = torch.stack((expected_prev[:, dim:], kv[:, dim:]), 1)
    pair = torch.stack((expected_prev[:, :dim], kv[:, :dim]), 1)
    values = (pair * scores.softmax(1)).sum(1)
    valid = (slot_mapping >= 0) & (positions % 2 == 1)
    ref = comp_mod.rms_norm(values, torch.ones(dim), 1e-6)
    ref = ref.masked_fill(~valid[:, None], 0).to(torch.bfloat16)
    assert torch.equal(latent.view(torch.int16), ref.view(torch.int16))
