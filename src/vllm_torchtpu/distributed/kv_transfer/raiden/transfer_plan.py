# SPDX-License-Identifier: Apache-2.0
"""Which KV cache pages Stage 3 transfers, and how many tokens they hold.

Page geometry, not transport. A cache group is the set of layers the
scheduler pages together, and the two peers of a disagg pair each derive
their page list from the same request length and the same group geometry.
That is the only thing keeping the lists aligned index for index, so the
arithmetic lives here in one place and both sides call it.

Sliding-window groups are where it gets interesting: the allocator frees
pages that fall behind the window and points their block-table entries at
the null block, so a windowed group's table opens with dead pages that must
come off before anything is sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TransferGroup:
    """One paged KV cache group Stage 3 moves, and the size of its pages.

    A cache group is the set of layers the scheduler pages together: one
    request gets one block table per group, and every layer in the group
    reads the same page numbers.
    """

    cache_group_index: int
    page_tokens: int
    window_tokens: Optional[int] = None


# vLLM reserves block 0 as the null block and points every freed sliding-window
# entry at it, so a transfer range containing it is a range whose KV is gone.
NULL_BLOCK_ID = 0


def skipped_window_pages(*, num_tokens: int, page_tokens: int,
                         window_tokens: Optional[int]) -> int:
    """Leading pages of a sliding-window block table that hold no live KV.

    vLLM's `SlidingWindowManager` frees every page that falls out of the
    window and writes the null block over its entry, so the table starts with
    a run of one repeated ID. Transferring those pages would hand Raiden a
    destination block list with duplicates, which it rejects, and would move
    bytes the decode replica cannot read anyway.

    Mirrors `SlidingWindowManager.get_num_skipped_tokens` and
    `SingleTypeKVCacheManager.remove_skipped_blocks`. Both peers derive the
    count from the same request length and page geometry, so they trim the
    same pages and the surviving pages stay paired.
    """
    if not window_tokens:
        return 0
    skipped_tokens = max(0, int(num_tokens) - int(window_tokens) + 1)
    return skipped_tokens // int(page_tokens)


def trim_window_pages(
    block_ids: Sequence[int],
    *,
    num_tokens: int,
    page_tokens: int,
    window_tokens: Optional[int],
    group_label: str,
) -> list[int]:
    """`block_ids` without the dead prefix a sliding window leaves.

    A windowed group's block table still has one entry per page of the whole
    request, but the allocator has already freed the pages that fell behind
    the window and points them at the null block. Those entries are dropped here.
    """
    ids = [int(block_id) for block_id in block_ids]
    skipped = min(
        skipped_window_pages(num_tokens=num_tokens,
                             page_tokens=page_tokens,
                             window_tokens=window_tokens), len(ids))
    if not skipped:
        return ids
    dropped, kept = ids[:skipped], ids[skipped:]
    # Out-of-window pages hold nothing the consumer can attend to, so dropping
    # them is right either way, but the allocator normally has already written
    # the null block over all of them. A varied prefix means this arithmetic
    # and vLLM's have drifted apart, which is worth seeing before it turns
    # into a silent mis-transfer.
    if len(set(dropped)) != 1:
        logger.warning(
            "Stage-3 sliding-window trim dropped pages that are not all the "
            "null block: group=%s, dropped=%s, num_tokens=%d, page_tokens=%d, "
            "window=%s", group_label, dropped, num_tokens, page_tokens,
            window_tokens)
    if len(set(kept)) != len(kept):
        raise ValueError(
            "Stage-3 sliding-window block table still repeats a page after "
            f"trimming: group={group_label}, kept={kept}, ids={ids}, "
            f"skipped={skipped}, "
            f"num_tokens={num_tokens}, page_tokens={page_tokens}, "
            f"window={window_tokens}")
    return kept


def recycled_transfer_pages(
    block_ids: Sequence[int],
    *,
    num_tokens: int,
    page_tokens: int,
    window_tokens: Optional[int],
) -> int:
    """How many pages this group would transfer that the allocator already freed.

    Both peers derive their skip from `num_tokens`, the transfer prefix, but the
    allocator freed pages against wherever the producer actually got to. Those
    two agree only while the producer stops at the handoff token. Let it run one
    page further and it has recycled a page the send still lists, which arrives
    as the null block: `num_tokens=374` with a 16-token page and an 8-token
    window skips 22 pages while the allocator freed 23, so page 22 goes over the
    wire dead. Let it run far enough and two dead pages collide, which is a
    duplicate destination that Raiden rejects outright.

    Counting the null blocks left in the range beats re-deriving the allocator's
    position, because it reads what the block table actually says rather than
    assuming the two formulas line up.
    """
    skipped = skipped_window_pages(num_tokens=num_tokens,
                                   page_tokens=page_tokens,
                                   window_tokens=window_tokens)
    return sum(1 for block_id in tuple(block_ids)[skipped:]
               if int(block_id) == NULL_BLOCK_ID)


def group_window_tokens(group_spec: Any) -> Optional[int]:
    """The sliding window of a cache group, or None if it is full attention.

    DSv4 hands its groups over as `UniformTypeKVCacheSpecs`, which carries no
    `sliding_window` of its own; the layers under it are uniform by
    construction, so any one of them answers for the group.
    """
    specs = getattr(group_spec, "kv_cache_specs", None)
    if specs:
        group_spec = next(iter(specs.values()))
    window = getattr(group_spec, "sliding_window", None)
    return int(window) if window else None


def plan_transfer_groups(
    kv_cache_groups: Sequence[Any],
    *,
    multi_group: bool,
) -> tuple[TransferGroup, ...]:
    """The KV cache groups whose pages Stage 3 transfers, in group order.

    Mamba groups are excluded: their state rides the state-class tags and is
    addressed one committed checkpoint at a time, not as a page range.

    ``page_tokens`` travels with the group index because DeepSeek-V4's groups
    page at different sizes -- its SWA group's block is an eighth of the CSA
    group's -- so a single connector-wide page size cannot validate them all.

    ``multi_group`` says which plan this is, and it is exact in both
    directions. Qwen3.5 and GLM have one such group and must have one. DSv4
    has several by construction, because the whole reason it needs its own
    admission path is that its groups disagree about page geometry; a
    single-group DSv4 plan would mean the scheduler sends no per-group block
    tables while the worker's DSv4 path requires them, and the transfer would
    fail at runtime rather than here.
    """
    groups = tuple(
        TransferGroup(cache_group_index=index,
                      page_tokens=int(group.kv_cache_spec.block_size),
                      window_tokens=group_window_tokens(group.kv_cache_spec))
        for index, group in enumerate(kv_cache_groups)
        if not isinstance(group.kv_cache_spec, MambaSpec))
    if not groups:
        raise ValueError(
            "Stage-3 resharding requires at least one non-Mamba KV cache "
            f"group, got {len(tuple(kv_cache_groups))} group(s)")
    if not multi_group and len(groups) != 1:
        raise ValueError("Stage-3 Qwen3.5 resharding requires exactly one "
                         f"full-attention KV cache group, got "
                         f"{[group.cache_group_index for group in groups]}")
    if multi_group and len(groups) < 2:
        raise ValueError(
            "Stage-3 DSv4 resharding expects several KV cache groups paging "
            "at different sizes, got "
            f"{[group.cache_group_index for group in groups]}. A model with "
            "one group does not need DSv4 admission; run it without "
            "TPU_RAIDEN_DSV4_ADMISSION=1")
    return groups


@dataclass(frozen=True)
class Dsv4Pool:
    """One tag in the DSv4 transfer plan, and the geometry to move it."""

    # Pool tag, e.g. `dsv4.csa.nope.g0`. The `.g0` suffix names the kv cache
    # group whose block table addresses these bytes.
    tag: str
    # Which block table to read, as an index into `SendMeta.group_block_ids`.
    # That list skips Mamba groups, so this is not a kv-cache-group index.
    # Tags of one group share a position: `dsv4.csa.nope.g0` and
    # `dsv4.csa.rope.g0` do, because a CSA layer's two arrays page together.
    plan_position: int
    page_tokens: int
    live_bytes_per_block: int
    # `live_bytes_per_block // row_bytes` is rows per page, which is not
    # `page_tokens`: each DSv4 role packs a different number of tokens into a
    # row, four for CSA NoPE and sixteen for RoPE.
    row_bytes: int
    # None for full attention.
    window_tokens: Optional[int] = None


def dsv4_pool_pages(
    pool: Dsv4Pool,
    block_ids: Sequence[int],
    num_tokens: int,
) -> tuple[list[int], int]:
    """The pages one DSv4 pool actually transfers, and the tokens they hold.

    A windowed group's block table opens with null blocks standing in for
    pages the allocator already freed, so both the pages and the token count
    they would account for come off the front together. Producer and consumer
    call this with the same pool geometry and the same request length, which
    is what keeps their page lists aligned index for index.
    """
    trimmed = trim_window_pages(block_ids,
                                num_tokens=num_tokens,
                                page_tokens=pool.page_tokens,
                                window_tokens=pool.window_tokens,
                                group_label=pool.tag)
    dropped_pages = len(block_ids) - len(trimmed)
    return trimmed, int(num_tokens) - dropped_pages * int(pool.page_tokens)
