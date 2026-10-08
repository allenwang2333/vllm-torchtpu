# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for scripts/dsv41_bench/kv_shard.py (owner-local KV layout)."""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import torch

_BENCH_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import kv_shard  # noqa: E402

PAGE = 16


def _unique_row_inputs(num_seqs: int, pages_per_seq: int, window: int) -> dict:
    """Bench-shaped inputs whose pool rows are all distinct.

    `build_decode_inputs` tiles a 1024-row template, so pages 64 apart hold
    equal bytes and an off-by-64-pages remap would still compare equal.
    """
    total_pages = pages_per_seq + num_seqs * 64
    total_rows = total_pages * PAGE
    swa_rows = num_seqs * window + 64
    pages = (
        torch.arange(pages_per_seq, dtype=torch.int32)[None, :]
        + 61 * torch.arange(num_seqs, dtype=torch.int32)[:, None]
    ) % total_pages
    swa_records = (
        torch.arange(num_seqs, dtype=torch.int32)[:, None] * window
        + torch.arange(window, dtype=torch.int32)[None, :]
    )
    swa_records[-1, -3:] = -1
    lanes = torch.arange(128, dtype=torch.int32)
    return {
        "pages": pages,
        "swa_records": swa_records,
        "main_pool": torch.arange(total_rows, dtype=torch.int32)[:, None] * 128 + lanes,
        "rope_pool": -(
            torch.arange(total_rows // 4, dtype=torch.int32)[:, None] * 128 + lanes
        ),
        "swa_pool": torch.arange(swa_rows * 2, dtype=torch.int32)[:, None] * 128
        + lanes
        + 7,
    }


def _main_rows(pool, pages, s, rows):
    raw = pages[s, rows // PAGE].long() * PAGE + rows % PAGE
    return pool[raw]


def _rope_rows(pool, pages, s, rows):
    raw = pages[s, rows // PAGE].long() * PAGE + rows % PAGE
    return pool.reshape(-1, 32)[raw]


def _swa_rows(pool, records, s):
    rec = records[s].long()
    words = pool.reshape(-1, 2 * 128)
    return torch.where((rec >= 0)[:, None], words[rec.clamp(min=0)], -1)


class TestOwnerRange:
    def test_owner_major_split(self):
        assert [kv_shard.owner_range(r, 16, 8) for r in (0, 3, 7)] == [
            (0, 2),
            (6, 8),
            (14, 16),
        ]

    @pytest.mark.parametrize(
        ("rank", "num_seqs", "tp", "match"),
        [
            (0, 12, 8, "num_seqs % tp == 0"),
            (0, 8, 0, "num_seqs % tp == 0"),
            (8, 16, 8, "outside"),
            (-1, 16, 8, "outside"),
        ],
    )
    def test_errors(self, rank, num_seqs, tp, match):
        with pytest.raises(ValueError, match=match):
            kv_shard.owner_range(rank, num_seqs, tp)


class TestLocalize:
    def test_localize_pages_maps_owned_rows_only(self):
        pages = torch.tensor([[9, 4, 7], [4, 2, 9], [5, 6, 8]], dtype=torch.int32)
        page_ids, local = kv_shard.localize_pages(pages, 0, 2)
        assert page_ids.tolist() == [2, 4, 7, 9]
        assert local.dtype == torch.int32
        assert local.tolist() == [[3, 1, 2], [1, 0, 3], [0, 0, 0]]
        assert torch.equal(page_ids[local[:2].long()].to(torch.int32), pages[:2])

    def test_localize_records_keeps_skip_entries(self):
        records = torch.tensor([[5, -1, 3], [8, 5, -1], [1, 2, 3]], dtype=torch.int32)
        row_ids, local = kv_shard.localize_records(records, 0, 2)
        assert row_ids.tolist() == [3, 5, 8]
        assert local.dtype == torch.int32
        assert local.tolist() == [[1, -1, 0], [2, 1, -1], [-1, -1, -1]]

    def test_localize_records_all_skipped(self):
        records = torch.full((2, 4), -1, dtype=torch.int32)
        row_ids, local = kv_shard.localize_records(records, 1, 2)
        assert row_ids.numel() == 0
        assert (local == -1).all()


class TestPackUnits:
    def test_packs_and_zero_pads(self):
        pool = torch.arange(6 * 2 * 3, dtype=torch.int32).reshape(12, 3)
        packed = kv_shard.pack_units(pool, torch.tensor([4, 1]), 2, 3)
        assert packed.shape == (6, 3)
        assert torch.equal(packed[:2], pool[8:10])
        assert torch.equal(packed[2:4], pool[2:4])
        assert (packed[4:] == 0).all()

    def test_overflow(self):
        pool = torch.zeros((8, 4), dtype=torch.int32)
        with pytest.raises(ValueError, match="do not fit"):
            kv_shard.pack_units(pool, torch.arange(3), 2, 2)


class TestOwnerShard:
    @pytest.mark.parametrize(("num_seqs", "tp"), [(16, 8), (8, 2)])
    def test_owned_requests_read_identical_rows(self, num_seqs, tp):
        window = 32
        pages_per_seq = 8
        inputs = _unique_row_inputs(num_seqs, pages_per_seq, window)
        rows = torch.arange(pages_per_seq * PAGE)
        shapes = set()
        for rank in range(tp):
            shard = kv_shard.owner_shard(inputs, rank=rank, tp=tp, page_size=PAGE)
            assert set(shard) == set(kv_shard.LOCAL_KEYS)
            shapes.add(tuple(tuple(shard[k].shape) for k in kv_shard.LOCAL_KEYS))
            assert shard["main_pool"].shape[0] % (8 * PAGE) == 0
            start, stop = kv_shard.owner_range(rank, num_seqs, tp)
            for s in range(start, stop):
                assert torch.equal(
                    _main_rows(shard["main_pool"], shard["pages"], s, rows),
                    _main_rows(inputs["main_pool"], inputs["pages"], s, rows),
                )
                assert torch.equal(
                    _rope_rows(shard["rope_pool"], shard["pages"], s, rows),
                    _rope_rows(inputs["rope_pool"], inputs["pages"], s, rows),
                )
                assert torch.equal(
                    _swa_rows(shard["swa_pool"], shard["swa_records"], s),
                    _swa_rows(inputs["swa_pool"], inputs["swa_records"], s),
                )
            others = [s for s in range(num_seqs) if not start <= s < stop]
            assert (shard["swa_records"][others] == -1).all()
            assert (shard["pages"][others] == 0).all()
        assert len(shapes) == 1

    def test_remap_is_monotonic(self):
        inputs = _unique_row_inputs(16, 8, 32)
        shard = kv_shard.owner_shard(inputs, rank=5, tp=8, page_size=PAGE)
        start, stop = kv_shard.owner_range(5, 16, 8)
        glob = inputs["pages"][start:stop].flatten()
        loc = shard["pages"][start:stop].flatten()
        order = torch.argsort(glob, stable=True)
        assert torch.equal(torch.argsort(loc, stable=True), order)

    def test_pool_bytes_shrink(self):
        inputs = _unique_row_inputs(16, 8, 32)
        shard = kv_shard.owner_shard(inputs, rank=0, tp=8, page_size=PAGE)
        local = kv_shard.pool_bytes(shard)
        full = kv_shard.pool_bytes(inputs)
        assert full == sum(
            inputs[k].numel() * inputs[k].element_size() for k in kv_shard.POOL_KEYS
        )
        assert local < full

    def test_rejects_bad_rank(self):
        inputs = _unique_row_inputs(16, 8, 32)
        with pytest.raises(ValueError, match="outside"):
            kv_shard.owner_shard(inputs, rank=-1, tp=8, page_size=PAGE)
