# SPDX-License-Identifier: Apache-2.0
"""TD8.231: stage-shared pre-mapped write records for PagedCacheRows inserts.

The global-KV (L2/8/14/20), SWA and DSpark context-KV inserts now hand
``write_cache`` the pool rows of their slots (``records=``), computed once per
stage from the block/remainder addresses (``_fast_swa_row_slots``) plus each
layer's own ``row_offset``, instead of ``PagedCacheRows.write`` re-mapping the
slots through ``row_indices`` on every layer. These tests replay every captured
insert into two real ``PagedCacheRows`` pools (with and without ``records``)
and pin bitwise-equal pool bytes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    PagedCacheRows,
    write_cache,
)

BLOCKS = 16
WIDTH = 256  # bf16 elements per 512-byte record

# name -> (page_stride multiplier, row_offset in pages, sharded, rep, off)
GEOMETRIES = {
    "plain": (1, 0, False, 0, 0),
    "offset": (1, 2, False, 0, 0),
    "strided": (2, 1, False, 0, 0),
    "sharded": (1, 1, True, 1, 3),
}


def _real_cache(rows, geometry, *, offset_pages=None):
    stride_mul, pages, sharded, rep, off = GEOMETRIES[geometry]
    if offset_pages is not None:
        pages = offset_pages
    stride = stride_mul * rows
    offset = pages * rows
    raw = torch.zeros(offset + BLOCKS * stride, 128, dtype=torch.int32)
    return PagedCacheRows(
        raw,
        (
            (BLOCKS, rows, WIDTH),
            torch.bfloat16,
            stride * 512,
            offset * 512,
            off,
            sharded,
            rep,
        ),
    )


def _mock_cache(rows, geometry, *, offset_pages=None):
    """A spec'd stand-in whose geometry and ``row_indices`` match ``_real_cache``."""
    real = _real_cache(rows, geometry, offset_pages=offset_pages)
    cache: Any = MagicMock(spec=PagedCacheRows)
    cache.numel.return_value = 1024
    cache.shape = real.shape
    cache.row_offset = real.row_offset
    cache.page_stride = real.page_stride
    cache.sharded = real.sharded
    cache.replicated_blocks = real.replicated_blocks
    cache.block_offset = real.block_offset
    cache.row_indices = real.row_indices
    cache.pages.return_value = None
    cache.real = real
    return cache


def _replay_parity(call):
    """Replay one captured insert into fresh pools with and without records."""
    cache, slots, _, valid, records = call
    assert records is not None
    real = cache.real if hasattr(cache, "real") else cache
    total_rows = real.raw.shape[0]
    ref = PagedCacheRows(
        torch.zeros(total_rows, 128, dtype=torch.int32),
        (
            real.shape,
            real.dtype,
            real.page_stride * 512,
            real.row_offset * 512,
            real.block_offset,
            real.sharded,
            real.replicated_blocks,
        ),
    )
    got = PagedCacheRows(
        torch.zeros(total_rows, 128, dtype=torch.int32),
        (
            real.shape,
            real.dtype,
            real.page_stride * 512,
            real.row_offset * 512,
            real.block_offset,
            real.sharded,
            real.replicated_blocks,
        ),
    )
    count = slots.shape[0]
    values = torch.arange(count * WIDTH, dtype=torch.float32).reshape(count, WIDTH)
    values = (values + 1).to(torch.bfloat16)
    write_cache(ref, slots, values, valid)
    write_cache(got, slots, values, valid, records=records)
    assert torch.equal(got.raw, ref.raw)
    return ref.raw


def _capture(calls):
    def insert(cache, slots, values, valid, *, records=None):
        calls.append((cache, slots, values, valid, records))

    return insert


# ---------------------------------------------------------------------------
# PagedCacheRows.write / write_cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_paged_write_records_parity(geometry):
    rows = 4
    blocks = torch.tensor([1, 2, -1, 5, 3, 7, 0, 15, 2, 4], dtype=torch.long)
    rem = torch.tensor([0, 3, 1, 2, 1, 0, 3, 2, 2, 1], dtype=torch.long)
    slots = (blocks * rows + rem).masked_fill(blocks < 0, -1)
    valid = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    write_slots, write_invalid = v2_attn._fast_swa_row_slots(
        _real_cache(rows, geometry), blocks, rem, ~valid | (slots < 0)
    )
    cache = _real_cache(rows, geometry)
    records = v2_attn._offset_records(cache, write_slots, write_invalid)
    raw = _replay_parity((cache, slots, None, valid, records))
    assert raw.any()


def test_write_cache_ignores_records_for_plain_tensors():
    slots = torch.tensor([3, 0, -1, 5], dtype=torch.long)
    valid = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1
    ref = torch.zeros(8, 2)
    got = torch.zeros(8, 2)
    write_cache(ref, slots, values, valid)
    write_cache(got, slots, values, valid, records=torch.full((4,), 7))
    assert torch.equal(got, ref)
    assert ref.any()


def test_insert_forwards_records(monkeypatch):
    seen: list[Any] = []
    monkeypatch.setattr(v2_attn, "write_cache", lambda *a, **kw: seen.append((a, kw)))
    marker = torch.ones(1)
    v2_attn._insert("c", "s", "v", "m", records=marker)
    v2_attn._insert("c", "s", "v", "m")
    assert seen == [
        (("c", "s", "v", "m"), {"records": marker}),
        (("c", "s", "v", "m"), {"records": None}),
    ]


# ---------------------------------------------------------------------------
# _write_global_kv
# ---------------------------------------------------------------------------

KV_ROWS = 4
KV_COUNT = 8


def _kv_meta(table, qsl=(0, 4, 8)):
    return SimpleNamespace(
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_tables=table,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
    )


def _make_global_attn(kv_cache):
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "kv"
    attn.compress_ratio = 2
    attn.window_size = 128
    attn._config = {"index_topk": 4}
    attn._max_model_len = 1000
    attn._rotate = lambda x, p: x
    attn._quantize_indexer = lambda x: x
    attn.compressor = lambda score, pos: torch.ones(KV_COUNT, 8)
    attn.kv_cache = kv_cache
    index_cache = _mock_cache(KV_ROWS, "plain")
    attn.indexer = SimpleNamespace(
        head_dim=8,
        wk=lambda lat: lat,
        k_norm=lambda k: k,
        k_cache=SimpleNamespace(prefix="idx", compress_ratio=2, kv_cache=index_cache),
    )
    return attn


@pytest.mark.parametrize("with_starts", [False, True])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_write_global_kv_records_parity(monkeypatch, geometry, with_starts):
    kv_table = torch.tensor([[1, 2, 6], [3, 4, 9]], dtype=torch.int32)
    metadata = {"kv": _kv_meta(kv_table, qsl=(0, 4, 7)), "idx": _kv_meta(kv_table)}
    starts = torch.tensor([4, 0], dtype=torch.int32) if with_starts else None
    forward_ctx = SimpleNamespace(
        attn_metadata=metadata,
        additional_kwargs={"deepseek_v41_window_starts": starts},
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    calls: list[Any] = []
    monkeypatch.setattr(v2_attn, "_insert", _capture(calls))
    positions = torch.tensor([0, 1, 5, 9, 2, 3, 15, 17], dtype=torch.int32)
    # Two layers sharing one pool at different row offsets.
    layer_a = _make_global_attn(_real_cache(KV_ROWS, geometry))
    layer_b = _make_global_attn(_real_cache(KV_ROWS, geometry, offset_pages=3))

    layer_a._write_global_kv(torch.ones(KV_COUNT, 10), positions)
    assert calls[-1][4] is None  # outside a stage: PagedCacheRows maps slots
    uncached_slots = calls[-1][1]

    with v2_attn.stage_attention_cache_scope():
        layer_a._write_global_kv(torch.ones(KV_COUNT, 10), positions)
        call_a = calls[-1]
        layer_b._write_global_kv(torch.ones(KV_COUNT, 10), positions)
        call_b = calls[-1]
    assert torch.equal(call_a[1].long(), uncached_slots.long())
    assert call_b[1] is call_a[1]  # stage-shared slots
    raw_a = _replay_parity(call_a)
    _replay_parity(call_b)
    assert raw_a.any()
    if with_starts:
        # Replayed prefix rows are masked out of the records too.
        assert (call_a[4][~call_a[3][:KV_COUNT]] == -1).all()


# ---------------------------------------------------------------------------
# _prepare_tpu SWA insert
# ---------------------------------------------------------------------------

SWA_ROWS = 32


def _make_swa_attn(swa_cache):
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "swa_pfx"
    attn.compressed_cache_prefix = "src_pfx"
    attn.index_source_layer_id = 0
    attn.window_size = 64
    attn.compress_ratio = 0
    attn._max_model_len = 16
    attn._config = {"index_topk": 2}
    attn._fp8_kv = True
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn._local_q_attn = False
    attn.n_local_heads = 2
    attn.head_dim = 8
    attn.scale = 0.25
    attn.attn_sink = torch.zeros(2)
    attn._swa_max_requests = 16
    attn._precompact_records = False
    attn._token_parallel_attn = False
    attn._full_attn_sink = False
    attn._tp_rank = torch.tensor(0, dtype=torch.int32)
    attn._tp_rank_1d = torch.tensor([0], dtype=torch.int32)
    attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
    attn._rotate = lambda x, p: x
    attn.wq_b = lambda qr: qr
    attn.compressor = None
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_pfx", kv_cache=swa_cache)
    attn.kv_cache = swa_cache
    attn._static_forward_context = {
        "src_pfx": SimpleNamespace(kv_cache=swa_cache),
        "swa_pfx": SimpleNamespace(kv_cache=swa_cache),
    }
    attn.indexer = None
    attn._indexer_weight_scale = 1.0
    return attn


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_prepare_tpu_swa_records_parity(monkeypatch, geometry):
    count = 4
    positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    forward_ctx = SimpleNamespace(
        attn_metadata={"swa_pfx": meta, "src_pfx": meta}, additional_kwargs={}
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    calls: list[Any] = []
    monkeypatch.setattr(v2_attn, "_insert", _capture(calls))
    monkeypatch.setattr(
        v2_attn, "paged_attention", lambda q, *a, **kw: torch.zeros_like(q)
    )
    monkeypatch.setattr(
        v2_attn, "sparse_fp8_attention", lambda q, *a, **kw: torch.zeros_like(q)
    )
    qr = torch.ones((count, 16), dtype=torch.bfloat16)
    kv = torch.ones((count, 8), dtype=torch.bfloat16)
    out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
    layer_a = _make_swa_attn(_mock_cache(SWA_ROWS, geometry))
    layer_b = _make_swa_attn(_mock_cache(SWA_ROWS, geometry, offset_pages=3))

    layer_a._prepare_tpu(None, qr, kv, None, None, None, positions, out)
    assert calls[-1][4] is None  # outside a stage: PagedCacheRows maps slots
    uncached_slots = calls[-1][1]

    with v2_attn.stage_attention_cache_scope():
        layer_a._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        call_a = calls[-1]
        layer_b._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        call_b = calls[-1]
    assert torch.equal(call_a[1].long(), uncached_slots.long())
    assert call_b[1] is call_a[1]
    raw_a = _replay_parity(call_a)
    _replay_parity(call_b)
    if geometry != "sharded":
        assert raw_a.any()


def test_prepare_tpu_swa_without_fast_offset_keeps_plain_insert(monkeypatch):
    count = 4
    positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    forward_ctx = SimpleNamespace(
        attn_metadata={"swa_pfx": meta, "src_pfx": meta}, additional_kwargs={}
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    calls: list[Any] = []
    monkeypatch.setattr(v2_attn, "_insert", _capture(calls))
    monkeypatch.setattr(
        v2_attn, "paged_attention", lambda q, *a, **kw: torch.zeros_like(q)
    )
    cache = _mock_cache(SWA_ROWS, "plain")
    cache.row_offset = None  # no static offset
    attn = _make_swa_attn(cache)
    qr = torch.ones((count, 16), dtype=torch.bfloat16)
    kv = torch.ones((count, 8), dtype=torch.bfloat16)
    out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
    assert calls[-1][4] is None


# ---------------------------------------------------------------------------
# DSpark _insert_context_kv
# ---------------------------------------------------------------------------


def _dspark_layer(swa_cache):
    return SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=swa_cache),
        _rotate=lambda x, p: x,
        window_size=8,
    )


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_dspark_context_kv_records_parity(monkeypatch, geometry):
    rows = 4
    calls: list[Any] = []
    monkeypatch.setattr(v2_dspark, "_insert", _capture(calls))
    table = torch.tensor([[1, 2, 5], [3, 4, -1]], dtype=torch.int32)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([9, 6], dtype=torch.int32),
        block_tables=table,
        query_start_loc=torch.tensor([0, 5, 9], dtype=torch.int32),
    )
    positions = torch.tensor([2, 4, 6, 7, 8, 1, 3, 5, 9, 0], dtype=torch.int32)
    kv = torch.ones(10, 8)
    layer_a = _dspark_layer(_real_cache(rows, geometry))
    layer_b = _dspark_layer(_real_cache(rows, geometry, offset_pages=3))

    # Without a stage the rows are still pre-mapped (one call, no reuse).
    v2_dspark._insert_context_kv(layer_a, kv, positions, meta)
    uncached = calls[-1]
    _replay_parity(uncached)

    with v2_attn.stage_attention_cache_scope():
        v2_dspark._insert_context_kv(layer_a, kv, positions, meta)
        call_a = calls[-1]
        v2_dspark._insert_context_kv(layer_b, kv, positions, meta)
        call_b = calls[-1]
        # A new block table re-derives the rows from the cached addresses.
        meta.block_tables = table.clone()
        v2_dspark._insert_context_kv(layer_b, kv, positions, meta)
        call_c = calls[-1]
    assert torch.equal(call_a[1], uncached[1])
    assert torch.equal(call_a[3], uncached[3])
    assert call_b[1] is call_a[1]
    assert call_c[1] is not call_a[1]
    raw_a = _replay_parity(call_a)
    _replay_parity(call_b)
    _replay_parity(call_c)
    assert raw_a.any()


def test_dspark_context_kv_without_fast_offset_keeps_plain_insert(monkeypatch):
    calls: list[Any] = []
    monkeypatch.setattr(v2_dspark, "_insert", _capture(calls))
    cache = _mock_cache(4, "plain")
    cache.row_offset = None
    meta = SimpleNamespace(
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    positions = torch.tensor([2, 3], dtype=torch.int32)
    with v2_attn.stage_attention_cache_scope():
        v2_dspark._insert_context_kv(
            _dspark_layer(cache), torch.ones(2, 8), positions, meta
        )
        v2_dspark._insert_context_kv(
            _dspark_layer(cache), torch.ones(2, 8), positions, meta
        )
    assert calls[-1][4] is None and calls[-2][4] is None
    assert calls[-1][1] is calls[-2][1]
