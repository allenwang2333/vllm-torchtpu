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
"""Head-unique Stage-3 transfer: the page owner rule and the landing gather.

A head-sharded TP plane with fewer KV heads than ranks holds every KV head on
``group_size = tp_size // total_num_kv_heads`` consecutive ranks, its
replicas. The replicated transfer sends every page from each source rank to
the decode rank with the same index, so the wire carries ``group_size``
copies of each head. The head-unique transfer sends page ``p`` (its index in
the transfer's destination block list) once, from source replica
``p % group_size`` to decode replica ``p % group_size`` of the same head, and
the decode replicas copy the other ``group_size - 1`` copies over the device
interconnect once the pages have landed (``gather_plan``).

The source store applies the rule to the registered spans
(``dst_replica_group_size`` on the coordinate request); this module holds
the decode side's half: which layouts admit it, which loads use it, and the
row copies of the landing gather.
"""

from __future__ import annotations

import dataclasses


def head_unique_group_size(
    *,
    tp_size: int,
    total_num_kv_heads: int,
    dcp_size: int,
    dst_shards: int,
) -> int:
    """The KV-head group size of a decode layout that admits the head-unique transfer.

    Raises ``ValueError`` naming the reason for a layout that does not: the
    rule needs every rank of a group to hold the same pages of one head.
    """
    if tp_size <= 0:
        raise ValueError("head-unique transfer requires a positive TP size")
    if dcp_size != 1:
        raise ValueError(
            "head-unique transfer requires decode_context_parallel_size=1: DCP "
            f"ranks hold different pages of one KV head (dcp_size={dcp_size})"
        )
    if dst_shards != tp_size:
        raise ValueError(
            "head-unique transfer requires TPU_RAIDEN_DST_SHARDS to equal the "
            f"decode TP size: dst_shards={dst_shards}, tp_size={tp_size}"
        )
    if total_num_kv_heads <= 0 or tp_size % total_num_kv_heads:
        raise ValueError(
            "head-unique transfer requires the KV head count to divide the TP "
            f"size: num_key_value_heads={total_num_kv_heads}, tp_size={tp_size}"
        )
    group_size = tp_size // total_num_kv_heads
    if group_size <= 1:
        raise ValueError(
            "head-unique transfer requires a KV head replicated across TP "
            f"ranks: num_key_value_heads={total_num_kv_heads}, tp_size={tp_size}"
        )
    return group_size


def load_group_size(num_pages: int, group_size: int) -> int:
    """The group size one load transfers under: head-unique only when every replica lands a page.

    A replica armed for no page would never report, so a load shorter than
    the group stays replicated. Scheduler and workers both derive this from
    the destination page count, so they agree without a flag on the wire.
    """
    return group_size if group_size > 1 and num_pages >= group_size else 1


def owned_page_indices(num_pages: int, group_size: int, replica: int) -> list[int]:
    """The page indices replica ``replica`` lands, in page order."""
    if not 0 <= replica < group_size:
        raise ValueError(f"replica {replica} is outside group size {group_size}")
    return list(range(replica, num_pages, group_size))


# Blocks of the gather buffer per replica, per pass: a load takes the smallest
# bucket that holds its share, and a load larger than the last bucket goes
# in several passes, so the kernels and the collective compile for these
# shapes only. A pass boundary is a multiple of the group size, which keeps
# the owner rule (page index mod group size) the same across passes.
GATHER_BLOCK_BUCKETS = (8, 16, 32, 48)


def pages_per_replica(num_pages: int, group_size: int, block_bucket: int = 1) -> int:
    """Blocks of the gather buffer per replica: the largest replica share, bucketed.

    ``block_bucket`` rounds the share up so the buffer shapes, and with them
    the compiled kernels and collectives, take few distinct values.
    """
    share = (num_pages + group_size - 1) // group_size
    return (share + block_bucket - 1) // block_bucket * block_bucket


def gather_bucket(num_pages: int, group_size: int) -> int:
    """The smallest bucket whose share holds ``num_pages`` of one pass."""
    share = (num_pages + group_size - 1) // group_size
    for bucket in GATHER_BLOCK_BUCKETS:
        if share <= bucket:
            return bucket
    raise ValueError(
        f"{num_pages} pages exceed the largest gather pass of "
        f"{GATHER_BLOCK_BUCKETS[-1] * group_size}"
    )


def gather_passes(block_ids: list[int], group_size: int) -> list[list[int]]:
    """Splits a load's block list into passes of at most the largest bucket."""
    span = GATHER_BLOCK_BUCKETS[-1] * group_size
    return [block_ids[i : i + span] for i in range(0, len(block_ids), span)]


@dataclasses.dataclass(frozen=True)
class GatherPlan:
    """Kernel-page row copies of one replica's landing gather.

    ``pack`` moves this replica's own pages from the pool into its send
    buffer; ``unpack`` moves the other replicas' pages from the gathered
    buffer (``group_size`` send buffers back to back, replica order) into
    the pool. Rows are kernel pages; a destination block spans
    ``kernel_pages_per_block`` consecutive rows.
    """

    pack_src_rows: tuple[int, ...]
    pack_dst_rows: tuple[int, ...]
    unpack_src_rows: tuple[int, ...]
    unpack_dst_rows: tuple[int, ...]
    buffer_rows: int


def gather_plan(
    *,
    block_ids: list[int],
    group_size: int,
    replica: int,
    kernel_pages_per_block: int,
    block_row_stride: int,
    base_row: int = 0,
    buffer_blocks: int | None = None,
) -> GatherPlan:
    """The row copies of the landing gather for one pool and one replica.

    ``block_ids`` is the transfer's destination block list (page index =
    position). Block ``b`` starts at pool row ``base_row + b *
    block_row_stride``. The send buffer holds ``pages_per_replica`` blocks of
    ``kernel_pages_per_block`` rows; replica ``r``'s ``j``-th own page sits
    at buffer block ``j``, and at block ``r * pages_per_replica + j`` of the
    gathered buffer. ``buffer_blocks`` fixes the buffer's blocks per replica
    (a gather bucket); the copy lists are padded to the bucket's full row
    counts so each bucket is one compiled shape.
    """
    if group_size <= 1:
        raise ValueError("gather_plan needs a group size above one")
    if kernel_pages_per_block <= 0 or block_row_stride < kernel_pages_per_block:
        raise ValueError(
            "a block must span at least its kernel pages: "
            f"kernel_pages_per_block={kernel_pages_per_block}, "
            f"block_row_stride={block_row_stride}"
        )
    num_pages = len(block_ids)
    per_replica = pages_per_replica(num_pages, group_size)
    if buffer_blocks is not None:
        if buffer_blocks < per_replica:
            raise ValueError(
                f"gather bucket of {buffer_blocks} blocks is smaller than the "
                f"replica share {per_replica}"
            )
        per_replica = buffer_blocks
    kps = kernel_pages_per_block

    def pool_rows(page: int) -> range:
        start = base_row + int(block_ids[page]) * block_row_stride
        return range(start, start + kps)

    pack_src: list[int] = []
    pack_dst: list[int] = []
    for j, page in enumerate(owned_page_indices(num_pages, group_size, replica)):
        pack_src.extend(pool_rows(page))
        pack_dst.extend(range(j * kps, (j + 1) * kps))
    unpack_src: list[int] = []
    unpack_dst: list[int] = []
    for other in range(group_size):
        if other == replica:
            continue
        for j, page in enumerate(owned_page_indices(num_pages, group_size, other)):
            start = (other * per_replica + j) * kps
            unpack_src.extend(range(start, start + kps))
            unpack_dst.extend(pool_rows(page))
    # A pass shorter than the group leaves replicas without a page (a load's
    # tail pass of one to group_size - 1 pages). Such a replica still joins
    # the collective with its buffer, so its empty copy lists get one
    # harmless copy to pad from: a read of page 0's first row into its unused
    # send buffer when it owns nothing, and its own page 0's row from its own
    # slot of the gathered buffer (the same bytes) when nobody else owns a
    # page.
    if not pack_src:
        pack_src.append(pool_rows(0).start)
        pack_dst.append(0)
    if not unpack_src:
        unpack_src.append(replica * per_replica * kps)
        unpack_dst.append(
            pool_rows(owned_page_indices(num_pages, group_size, replica)[0]).start
        )
    pack_src_t, pack_dst_t = tuple(pack_src), tuple(pack_dst)
    unpack_src_t, unpack_dst_t = tuple(unpack_src), tuple(unpack_dst)
    if buffer_blocks is not None:
        pack_src_t, pack_dst_t = padded_copy_rows_to(
            pack_src_t, pack_dst_t, per_replica * kps
        )
        unpack_src_t, unpack_dst_t = padded_copy_rows_to(
            unpack_src_t, unpack_dst_t, (group_size - 1) * per_replica * kps
        )
    return GatherPlan(
        pack_src_rows=pack_src_t,
        pack_dst_rows=pack_dst_t,
        unpack_src_rows=unpack_src_t,
        unpack_dst_rows=unpack_dst_t,
        buffer_rows=per_replica * kps,
    )


def padded_copy_rows(
    src_rows: tuple[int, ...], dst_rows: tuple[int, ...], multiple: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pads a row copy list to a multiple of ``multiple`` by repeating its last pair.

    A repeated copy writes the bytes its destination already receives, so
    the destination is unchanged and the kernel's chunk count takes a few
    distinct values instead of one per page count.
    """
    if len(src_rows) != len(dst_rows):
        raise ValueError("source and destination row lists differ in length")
    if not src_rows or multiple <= 1:
        return src_rows, dst_rows
    pad = (-len(src_rows)) % multiple
    return src_rows + (src_rows[-1],) * pad, dst_rows + (dst_rows[-1],) * pad


def padded_copy_rows_to(
    src_rows: tuple[int, ...], dst_rows: tuple[int, ...], length: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pads a row copy list to exactly ``length`` pairs by repeating its last pair."""
    if len(src_rows) != len(dst_rows):
        raise ValueError("source and destination row lists differ in length")
    if len(src_rows) > length:
        raise ValueError(f"{len(src_rows)} row copies exceed the bucket's {length}")
    if not src_rows:
        raise ValueError("an empty row copy list cannot be padded")
    pad = length - len(src_rows)
    return src_rows + (src_rows[-1],) * pad, dst_rows + (dst_rows[-1],) * pad
