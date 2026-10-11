# SPDX-License-Identifier: Apache-2.0
"""TD8.230: stage-cached compressor / global-KV write slots.

`DeepseekCompressor.forward` and `DeepseekV41Attention._write_global_kv` run
on the four compressed layers (L2/8/14/20) of a stage with the same tokens and
block tables. Inside `stage_attention_cache_scope()` they now share the
block-table gathers and the deferred index-cache write rows. These tests pin
bitwise parity with the uncached path and the cache hit/invalidation rules.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

ROWS = 4  # rows per page in the synthetic caches
COUNT = 8


def _meta(table, qsl=(0, 4, 8)):
    return SimpleNamespace(
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_tables=table,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
    )


# ---------------------------------------------------------------------------
# Compressor.forward
# ---------------------------------------------------------------------------


def _make_compressor(monkeypatch, metadata, captured):
    forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)

    def save(kv_score, pos, state_cache, slots, qsl, reqs, nw, eps, ratio, latent):
        captured.append(slots)

    monkeypatch.setattr(v2_attn, "save_compress_norm", save)
    comp: Any = object.__new__(v2_attn.DeepseekCompressor)
    torch.nn.Module.__init__(comp)
    comp.head_dim = 8
    comp.compress_ratio = 2
    comp.rms_norm_eps = 1e-6
    comp.norm = SimpleNamespace(weight=torch.ones(8))
    state_kv = MagicMock()
    state_kv.shape = (16, ROWS, 128)
    comp.state_cache = SimpleNamespace(prefix="state", kv_cache=state_kv)
    return comp


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_compressor_write_slots_shared_per_block_table(monkeypatch, pos_dtype):
    table = torch.tensor([[3, 5], [7, 9]], dtype=torch.int32)
    positions = torch.arange(COUNT, dtype=pos_dtype) + 2
    metadata = {"state": _meta(table, qsl=(0, 4, 6))}
    captured: list[torch.Tensor] = []
    comp = _make_compressor(monkeypatch, metadata, captured)

    # Reference: uncached ring slots.
    comp.forward(torch.ones(COUNT, 10), positions)
    expected = captured[-1].clone()
    assert expected.dtype == torch.int32

    with v2_attn.stage_attention_cache_scope():
        comp.forward(torch.ones(COUNT, 10), positions)
        comp.forward(torch.ones(COUNT, 10), positions)
        # Second layer reuses the first layer's slots tensor.
        assert captured[-1] is captured[-2]
        assert captured[-1].dtype == torch.int32
        assert torch.equal(captured[-1], expected)

        # A new block_tables tensor invalidates the shared slots.
        metadata["state"].block_tables = table + 10
        comp.forward(torch.ones(COUNT, 10), positions)
        assert captured[-1] is not captured[-2]
        assert captured[-1].dtype == torch.int32
        assert not torch.equal(captured[-1], expected)
    metadata["state"].block_tables = table + 10
    comp.forward(torch.ones(COUNT, 10), positions)
    assert captured[-1].dtype == torch.int32
    assert torch.equal(captured[-1], captured[-2])


def test_td8250_save_compress_norm_int32_ring_and_empty_latent_parity(monkeypatch):
    monkeypatch.setattr(v2_attn, "write_cache", lambda *a, **kw: None)
    kv_score = torch.randn(COUNT, 9, dtype=torch.float32)
    positions_i64 = torch.arange(COUNT, dtype=torch.int64) + 2
    positions_i32 = positions_i64.int()
    slots_i64 = torch.tensor([12, 13, 14, 15, 28, 29, -1, -1], dtype=torch.int64)
    slots_i32 = slots_i64.int()
    qsl = torch.tensor([0, 4, 6], dtype=torch.int32)
    reqs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    nw = torch.ones(8, dtype=torch.float32)

    seen_indices: list[tuple[torch.dtype, torch.dtype]] = []
    backing = torch.randn(16, ROWS, 9, dtype=torch.float32)

    class _SpyStateCache:
        shape = (16, ROWS, 9)

        def __getitem__(self, idx):
            b, r = idx
            seen_indices.append((b.dtype, r.dtype))
            return backing[b, r]

    latent_zero = torch.zeros((COUNT, 8), dtype=torch.bfloat16)
    latent_dirty = torch.full((COUNT, 8), -77.0, dtype=torch.bfloat16)
    v2_attn.save_compress_norm(
        kv_score,
        positions_i32,
        _SpyStateCache(),
        slots_i32,
        qsl,
        reqs,
        nw,
        1e-6,
        2,
        latent_zero,
    )
    v2_attn.save_compress_norm(
        kv_score,
        positions_i64,
        _SpyStateCache(),
        slots_i64,
        qsl,
        reqs,
        nw,
        1e-6,
        2,
        latent_dirty,
    )
    assert seen_indices == [
        (torch.int32, torch.int32),
        (torch.int32, torch.int32),
    ]
    assert torch.equal(latent_dirty, latent_zero)


# ---------------------------------------------------------------------------
# _write_global_kv
# ---------------------------------------------------------------------------


def _paged_cache(*, row_offset=0, page_stride=ROWS, sharded=False, rep=0, off=0):
    cache: Any = MagicMock(spec=PagedCacheRows)
    cache.shape = (16, ROWS, 128)
    cache.row_offset = row_offset
    cache.page_stride = page_stride
    cache.sharded = sharded
    cache.replicated_blocks = rep
    cache.block_offset = off
    cache.row_indices = lambda slots: PagedCacheRows.row_indices(cache, slots)
    return cache


def _make_attn(monkeypatch, metadata, index_cache, inserted, *, starts=None):
    forward_ctx = SimpleNamespace(
        attn_metadata=metadata,
        additional_kwargs={"deepseek_v41_window_starts": starts},
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    monkeypatch.setattr(
        v2_attn, "_insert", lambda cache, slots, val, mask, **kw: inserted.append(slots)
    )
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "kv"
    attn.compress_ratio = 2
    attn.window_size = 128
    attn._config = {"index_topk": 4}  # _all_indices_fit is False
    attn._max_model_len = 1000
    attn._rotate = lambda x, p: x
    attn._quantize_indexer = lambda x: x
    attn.compressor = lambda score, pos: torch.ones(COUNT, 8)
    kv_cache = MagicMock(spec=PagedCacheRows)
    kv_cache.shape = (16, ROWS, 128)
    attn.kv_cache = kv_cache
    attn.indexer = SimpleNamespace(
        head_dim=8,
        wk=lambda lat: lat,
        k_norm=lambda k: k,
        k_cache=SimpleNamespace(prefix="idx", compress_ratio=2, kv_cache=index_cache),
    )
    return attn


CACHES = {
    "plain": dict(),
    "offset": dict(row_offset=32),
    "strided": dict(page_stride=2 * ROWS, row_offset=8),
    "sharded": dict(sharded=True, rep=2, off=4, row_offset=16),
}


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shared_table", [True, False])
@pytest.mark.parametrize("with_starts", [False, True])
@pytest.mark.parametrize("cache_kind", sorted(CACHES))
def test_write_global_kv_deferred_records_parity(
    monkeypatch, cache_kind, with_starts, shared_table, pos_dtype
):
    kv_table = torch.tensor([[1, 2, 6], [3, 4, 9]], dtype=torch.int32)
    idx_table = kv_table if shared_table else kv_table + 1
    metadata = {"kv": _meta(kv_table, qsl=(0, 4, 7)), "idx": _meta(idx_table)}
    positions = torch.tensor([0, 1, 5, 9, 2, 3, 15, 17], dtype=pos_dtype)
    starts = torch.tensor([4, 0], dtype=torch.int32) if with_starts else None
    index_cache = _paged_cache(**CACHES[cache_kind])
    inserted: list[torch.Tensor] = []
    rotated_positions: list[torch.Tensor] = []
    attn = _make_attn(monkeypatch, metadata, index_cache, inserted, starts=starts)
    attn._rotate = lambda x, p: rotated_positions.append(p) or x

    ref_records, _ = attn._write_global_kv(torch.ones(COUNT, 10), positions)
    ref_kv_slots = inserted[-1]
    ref_group_pos = rotated_positions[-1]

    with v2_attn.stage_attention_cache_scope():
        records1, _ = attn._write_global_kv(torch.ones(COUNT, 10), positions)
        kv_slots1 = inserted[-1]
        group_pos1 = rotated_positions[-1]
        records2, _ = attn._write_global_kv(torch.ones(COUNT, 10), positions)
        kv_slots2 = inserted[-1]

    assert records1.dtype == torch.int32
    assert ref_records.dtype == torch.int32
    assert group_pos1.dtype == torch.int32
    assert ref_group_pos.dtype == torch.int32
    assert torch.equal(group_pos1, ref_group_pos)
    assert torch.equal(records1, ref_records)
    assert torch.equal(records2, records1)
    assert kv_slots1.dtype == torch.int32
    assert ref_kv_slots.dtype == torch.int32
    assert torch.equal(kv_slots1, ref_kv_slots)
    # Cache hit: the second layer reuses the first layer's slot tensor.
    assert kv_slots2 is kv_slots1


def test_write_global_kv_index_slots_reuse_and_invalidation(monkeypatch):
    kv_table = torch.tensor([[1, 2, 6], [3, 4, 9]], dtype=torch.int32)
    metadata = {"kv": _meta(kv_table), "idx": _meta(kv_table)}
    positions = torch.tensor([0, 1, 5, 9, 2, 3, 15, 17], dtype=torch.int32)
    index_cache = _paged_cache()
    inserted: list[torch.Tensor] = []
    attn = _make_attn(monkeypatch, metadata, index_cache, inserted)

    with v2_attn.stage_attention_cache_scope():
        # Same table, ratio and rows: the index slots *are* the KV slots.
        assert (
            attn._write_global_kv(
                torch.ones(COUNT, 10), positions, defer_index_write=False
            )
            is None
        )
        kv_slots, index_slots = inserted[-2:]
        assert index_slots is kv_slots

        # Replacing the KV block table invalidates the cached slots.
        metadata["kv"].block_tables = kv_table + 2
        attn._write_global_kv(torch.ones(COUNT, 10), positions, defer_index_write=False)
        new_kv_slots, new_index_slots = inserted[-2:]
        assert new_kv_slots is not kv_slots
        # The index table is now a different tensor: a separate gather.
        assert new_index_slots is not new_kv_slots
        assert torch.equal(new_index_slots, index_slots)


def test_write_global_kv_without_fast_offset_uses_row_indices(monkeypatch):
    kv_table = torch.tensor([[1, 2, 6], [3, 4, 9]], dtype=torch.int32)
    metadata = {"kv": _meta(kv_table), "idx": _meta(kv_table)}
    positions = torch.tensor([0, 1, 5, 9, 2, 3, 15, 17], dtype=torch.int32)
    index_cache = _paged_cache(row_offset=8)
    index_cache.row_offset = None  # no static offset: keep row_indices
    index_cache.row_indices = lambda slots: slots + 100
    inserted: list[torch.Tensor] = []
    attn = _make_attn(monkeypatch, metadata, index_cache, inserted)

    ref, _ = attn._write_global_kv(torch.ones(COUNT, 10), positions)
    with v2_attn.stage_attention_cache_scope():
        got, _ = attn._write_global_kv(torch.ones(COUNT, 10), positions)
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


def test_td8254_prepare_tpu_3page_swa_int64_positions_addressing(monkeypatch):
    from test_dsv41_v2_stage_attn_cache import _make_swa_attn

    swa_table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)
    swa_meta = SimpleNamespace(
        seq_lens=torch.tensor([130, 130], dtype=torch.int32),
        block_tables=swa_table,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    ctx = SimpleNamespace(attn_metadata={"swa_pfx": swa_meta}, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
    captured_records: list[torch.Tensor] = []
    monkeypatch.setattr(
        v2_attn,
        "sparse_fp8_attention",
        lambda q, swa_c, main_c, recs, *a, **kw: (
            captured_records.append(recs.clone()) or torch.zeros_like(q)
        ),
    )
    attn = _make_swa_attn(window_size=128, ced_decoder=False, has_fast_swa_offset=True)
    pos_i64 = torch.tensor([65, 129, 64, 128], dtype=torch.int64)
    pos_i32 = pos_i64.int()
    qr = torch.ones((4, 16), dtype=torch.bfloat16)
    kv = torch.ones((4, 8), dtype=torch.bfloat16)
    out = torch.zeros((4, 2, 8), dtype=torch.bfloat16)

    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(None, qr, kv, None, None, None, pos_i64, out)
        stage_cache = v2_attn._active_stage_attn_cache
        assert stage_cache is not None
        swa_entries = [v for k, v in stage_cache.items() if k[0] == "swa_token_addr"]
        assert len(swa_entries) == 1
        valid_token, requests, pos_col, pos_rem, logical, _, swa_3page = swa_entries[0]
        assert pos_rem.dtype == torch.int32
        assert logical.dtype == torch.int32
        assert swa_3page is not None
        c_last, prev_cols, is_c_last, is_prev_cols, swa_page_rem = swa_3page
        assert c_last.dtype == torch.int64
        assert all(col.dtype == torch.int64 for col in prev_cols)
        assert is_c_last.dtype == torch.bool
        assert all(m.dtype == torch.bool for m in is_prev_cols)
        assert swa_page_rem.dtype == torch.int32

    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(None, qr, kv, None, None, None, pos_i32, out)

    assert len(captured_records) == 2
    assert captured_records[0].dtype == torch.int32
    assert torch.equal(captured_records[0], captured_records[1])
