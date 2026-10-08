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
"""Owner-local (request-affine) KV layout for the decode attention benches.

At TP=8 every rank keeps an identical copy of the shared MQA KV cache. A
request-affine layout keeps the TP=8 weights but stores each request's rows
only on its owner rank. With an owner-major batch (rank ``r`` owns requests
``[r * C / tp, (r + 1) * C / tp)``), the contiguous token split of the
token-parallel attention body already routes every decode token to its owner,
so attention runs unchanged once each rank's pools hold just its owned rows
and the page table / window records address them.

The helpers build that layout on CPU from replicated bench inputs:
`owner_range` (owner-major split), `localize_pages` / `localize_records`
(global to local ids; rows of other owners are never read), `pack_units`
(gather fixed-size row units into a zero-padded pool), and `owner_shard`
(all pools of one rank, padded to one shape on every rank so the SPMD program
and its collectives match). Both remaps are monotonic, so any computation
ordered by physical row sees the same order as on the replicated pools.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Int32 word rows per unit: a BF16 window record of 512 elements spans two
# 128-lane word rows (`cache.gather_cache_pieces`); a page of the FP8 NoPE pool
# spans ``page_size`` rows and a page of its RoPE companion ``page_size // 4``.
SWA_RECORD_ROWS = 2
POOL_KEYS = ("main_pool", "rope_pool", "swa_pool")
LOCAL_KEYS = (*POOL_KEYS, "pages", "swa_records")


def owner_range(rank: int, num_seqs: int, tp: int) -> tuple[int, int]:
    """Requests ``[start, stop)`` owned by ``rank`` in owner-major order."""
    if tp < 1 or num_seqs % tp:
        raise ValueError(
            f"owner-major layout needs num_seqs % tp == 0, got {num_seqs} % {tp}"
        )
    if not 0 <= rank < tp:
        raise ValueError(f"rank {rank} outside [0, {tp})")
    per_rank = num_seqs // tp
    return rank * per_rank, (rank + 1) * per_rank


def localize_pages(
    pages: torch.Tensor, start: int, stop: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pages of requests ``[start, stop)`` and the page table remapped onto them.

    Returns ``(page_ids, local_pages)``: ``page_ids`` holds the sorted unique
    global pages the owned requests reference; ``local_pages[s, j]`` is the
    index of ``pages[s, j]`` in ``page_ids`` for owned ``s`` and 0 otherwise
    (those requests attend on their own owner, so the entries are never read).
    """
    import torch

    owned = pages[start:stop].long()
    page_ids = torch.unique(owned)
    local_pages = torch.zeros_like(pages)
    local_pages[start:stop] = torch.searchsorted(page_ids, owned).to(pages.dtype)
    return page_ids, local_pages


def localize_records(
    records: torch.Tensor, start: int, stop: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows referenced by requests ``[start, stop)`` and records remapped onto them.

    ``-1`` (skip) records stay ``-1``; records of other owners become ``-1``.
    """
    import torch

    owned = records[start:stop].long()
    valid = owned >= 0
    row_ids = torch.unique(owned[valid])
    local = torch.searchsorted(row_ids, owned.clamp(min=0))
    local_records = torch.full_like(records, -1)
    local_records[start:stop] = torch.where(valid, local, -1).to(records.dtype)
    return row_ids, local_records


def pack_units(
    pool: torch.Tensor, unit_ids: torch.Tensor, rows_per_unit: int, num_units: int
) -> torch.Tensor:
    """Pack units ``unit_ids`` of ``rows_per_unit`` rows into ``num_units`` units.

    Unit ``i`` of the result is unit ``unit_ids[i]`` of ``pool``; the tail is 0.
    """
    count = unit_ids.numel()
    if count > num_units:
        raise ValueError(f"{count} units do not fit in {num_units}")
    lanes = pool.shape[-1]
    packed = pool.new_zeros((num_units, rows_per_unit, lanes))
    packed[:count] = pool.reshape(-1, rows_per_unit, lanes)[unit_ids.long()]
    return packed.reshape(num_units * rows_per_unit, lanes)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def owner_shard(
    inputs: Mapping[str, torch.Tensor], *, rank: int, tp: int, page_size: int
) -> dict[str, torch.Tensor]:
    """Owner-local pools, page table and window records of ``rank``.

    ``inputs`` are replicated `build_decode_inputs` tensors. Every rank pads
    its pools to the largest owner's page and window-row counts (rounded up to
    8 pages and 64 rows, plus 64 spare rows like the replicated window pool),
    so all ranks compile identical shapes.
    """
    num_seqs = inputs["pages"].shape[0]
    owner_range(rank, num_seqs, tp)
    plans = []
    for r in range(tp):
        start, stop = owner_range(r, num_seqs, tp)
        plans.append(
            (
                localize_pages(inputs["pages"], start, stop),
                localize_records(inputs["swa_records"], start, stop),
            )
        )
    num_pages = _round_up(max(pg[0].numel() for pg, _ in plans), 8)
    num_rows = _round_up(max(sw[0].numel() for _, sw in plans), 64) + 64
    (page_ids, local_pages), (row_ids, local_records) = plans[rank]
    return {
        "main_pool": pack_units(inputs["main_pool"], page_ids, page_size, num_pages),
        "rope_pool": pack_units(
            inputs["rope_pool"], page_ids, page_size // 4, num_pages
        ),
        "swa_pool": pack_units(inputs["swa_pool"], row_ids, SWA_RECORD_ROWS, num_rows),
        "pages": local_pages,
        "swa_records": local_records,
    }


def pool_bytes(tensors: Mapping[str, torch.Tensor]) -> int:
    """Bytes of the main, RoPE and window pools in ``tensors``."""
    return sum(tensors[k].numel() * tensors[k].element_size() for k in POOL_KEYS)
