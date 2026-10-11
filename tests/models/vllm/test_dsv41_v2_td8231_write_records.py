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


def _replay_parity(call, *, expect_records: bool = True):
    """Replay one captured insert into fresh pools with and without records."""
    cache, slots, _, valid, records = call
    if expect_records:
        assert records is not None
    else:
        assert records is None
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
    expect_rec = geometry != "sharded"
    raw_a = _replay_parity(call_a, expect_records=expect_rec)
    _replay_parity(call_b, expect_records=expect_rec)
    assert raw_a.any()
    if with_starts and expect_rec:
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
    expect_rec = geometry != "sharded"
    raw_a = _replay_parity(call_a, expect_records=expect_rec)
    _replay_parity(call_b, expect_records=expect_rec)
    if expect_rec:
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

    # Without a stage the rows are still pre-mapped when fast_offset is True.
    expect_rec = geometry != "sharded"
    v2_dspark._insert_context_kv(layer_a, kv, positions, meta)
    uncached = calls[-1]
    _replay_parity(uncached, expect_records=expect_rec)

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
    raw_a = _replay_parity(call_a, expect_records=expect_rec)
    _replay_parity(call_b, expect_records=expect_rec)
    _replay_parity(call_c, expect_records=expect_rec)
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


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_td8243_dspark_context_kv_int32_slots_match_int64(monkeypatch, geometry):
    """TD8.243: int32 block/slot math equals the former int64 math bitwise."""
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
    cache = _real_cache(rows, geometry)
    v2_dspark._insert_context_kv(
        _dspark_layer(cache), torch.ones(10, 8), positions, meta
    )
    _, slots, _, valid, records = calls[-1]
    assert slots.dtype == torch.int32

    # The pre-TD8.243 int64 formula.
    requests = v2_attn._token_requests(meta, positions).clamp(max=1)
    pos_col = (positions // rows).clamp(max=table.shape[1] - 1).long()
    blocks64 = table[requests, pos_col].long()
    valid_token = torch.arange(10) < meta.query_start_loc[-1]
    slots64 = (blocks64 * rows + positions % rows).masked_fill(~valid_token, -1)
    assert torch.equal(slots.long(), slots64)
    if records is not None:
        write64 = v2_attn._fast_swa_row_slots(
            cache, blocks64, positions % rows, ~valid | (slots64 < 0)
        )
        assert torch.equal(records, v2_attn._offset_records(cache, *write64))


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_td8249_dspark_context_kv_int64_positions_produce_int32_slots(
    monkeypatch, geometry
):
    """TD8.249: int64 positions cast to int32 so swa_slots and pos_rem stay int32."""
    rows = 4
    calls: list[Any] = []
    monkeypatch.setattr(v2_dspark, "_insert", _capture(calls))
    table = torch.tensor([[1, 2, 5], [3, 4, -1]], dtype=torch.int32)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([9, 6], dtype=torch.int32),
        block_tables=table,
        query_start_loc=torch.tensor([0, 5, 9], dtype=torch.int32),
    )
    positions_i64 = torch.tensor([2, 4, 6, 7, 8, 1, 3, 5, 9, 0], dtype=torch.int64)
    positions_i32 = positions_i64.int()
    cache = _real_cache(rows, geometry)
    with v2_attn.stage_attention_cache_scope():
        v2_dspark._insert_context_kv(
            _dspark_layer(cache), torch.ones(10, 8), positions_i64, meta
        )
        assert v2_attn._active_stage_attn_cache is not None
        cached_entry = next(iter(v2_attn._active_stage_attn_cache.values()))
    _, slots_i64, valid_i64, _, addr_i64, write_i64 = cached_entry
    assert slots_i64.dtype == torch.int32
    assert addr_i64[2].dtype == torch.int32  # pos_rem

    v2_dspark._insert_context_kv(
        _dspark_layer(cache), torch.ones(10, 8), positions_i32, meta
    )
    _, slots_i32, _, valid_i32, records_i32 = calls[-1]
    assert torch.equal(slots_i64, slots_i32)
    assert torch.equal(valid_i64, valid_i32)
    if write_i64 is not None:
        assert torch.equal(
            v2_attn._offset_records(cache, *write_i64),
            records_i32,
        )


def test_td8242_write_cache_and_paged_write_zero_redundant_mask():
    """Verify write_cache skips slots slicing/masking when records is provided
    and avoids double masked_fill when records is None across all geometries.
    """
    rows = 4
    # Include positive slots where valid is False (e.g. replayed prefix or
    # non-publish rows) as well as negative slots where valid is True/False.
    slots = torch.tensor([4, 9, -1, 13, 21, 5, 0, 18], dtype=torch.int32)
    valid = torch.tensor(
        [True, False, True, True, False, True, False, True], dtype=torch.bool
    )
    values = (torch.arange(8 * WIDTH, dtype=torch.float32).reshape(8, WIDTH) + 1).to(
        torch.bfloat16
    )

    class _UntouchedSlots:
        def __getitem__(self, item):
            raise AssertionError("slots must not be sliced when records is provided")

        def masked_fill(self, *args, **kwargs):
            raise AssertionError("slots must not be masked when records is provided")

    for geometry in sorted(GEOMETRIES):
        ref_cache = _real_cache(rows, geometry)
        got_records_cache = _real_cache(rows, geometry)
        got_valid_cache = _real_cache(rows, geometry)

        # Reference: pre-mask slots before calling PagedCacheRows.write directly
        masked_slots = slots.masked_fill(~valid, -1)
        ref_cache.write(masked_slots, values)

        # 1. records is not None: slots must not be touched at all
        pre_records = got_records_cache.row_indices(slots.clamp(min=0)).masked_fill(
            slots < 0, -1
        )
        write_cache(
            got_records_cache,
            _UntouchedSlots(),
            values,
            valid,
            records=pre_records,
        )
        assert torch.equal(got_records_cache.raw, ref_cache.raw)

        # 2. records is None: write_cache passes valid to PagedCacheRows.write
        # without pre-masking slots in write_cache
        write_cache(got_valid_cache, slots, values, valid)
        assert torch.equal(got_valid_cache.raw, ref_cache.raw)

    # Also test pure-sharded (replicated_blocks == 0) with block_offset == 0 and > 0
    for block_offset in (0, 2):
        raw_ref = torch.zeros(BLOCKS * rows, 128, dtype=torch.int32)
        raw_got = torch.zeros(BLOCKS * rows, 128, dtype=torch.int32)
        layout = (
            (BLOCKS, rows, WIDTH),
            torch.bfloat16,
            rows * 512,
            0,
            block_offset,
            True,
            0,
        )
        c_ref = PagedCacheRows(raw_ref, layout)
        c_got = PagedCacheRows(raw_got, layout)
        c_ref.write(slots.masked_fill(~valid, -1), values)
        write_cache(c_got, slots, values, valid)
        assert torch.equal(c_got.raw, c_ref.raw)
        assert torch.equal(
            c_got.row_indices(slots, valid=valid),
            c_ref.row_indices(slots.masked_fill(~valid, -1)),
        )


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_td8264_dspark_context_kv_int64_metadata_keeps_int32(monkeypatch, geometry):
    """TD8.264: int64 seq_lens / query_start_loc still give int32 offsets / slots."""
    rows = 4
    calls: list[Any] = []
    monkeypatch.setattr(v2_dspark, "_insert", _capture(calls))
    table = torch.tensor([[1, 2, 5], [3, 4, -1]], dtype=torch.int32)
    positions = torch.tensor([2, 4, 6, 7, 8, 1, 3, 5, 9, 0], dtype=torch.int64)
    cache = _real_cache(rows, geometry)
    got = {}
    for dtype in (torch.int64, torch.int32):
        meta = SimpleNamespace(
            seq_lens=torch.tensor([9, 6], dtype=dtype),
            block_tables=table,
            query_start_loc=torch.tensor([0, 5, 9], dtype=dtype),
        )
        with v2_attn.stage_attention_cache_scope():
            v2_dspark._insert_context_kv(
                _dspark_layer(cache), torch.ones(10, 8), positions, meta
            )
            assert v2_attn._active_stage_attn_cache is not None
            got[dtype] = next(iter(v2_attn._active_stage_attn_cache.values()))
    _, slots64, valid64, in_window64, addr64, _ = got[torch.int64]
    _, slots32, valid32, in_window32, addr32, _ = got[torch.int32]
    assert slots64.dtype == torch.int32
    assert torch.equal(slots64, slots32)
    assert torch.equal(valid64, valid32)
    assert torch.equal(in_window64, in_window32)
    assert torch.equal(addr64[3], addr32[3])  # valid_token
