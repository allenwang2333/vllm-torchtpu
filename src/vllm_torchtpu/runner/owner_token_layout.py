# SPDX-License-Identifier: Apache-2.0
"""Owner-major token order for token-parallel attention over owner-sharded KV.

With ``TPU_DEEPSEEK_V41_KV_SHARD=1``, a request's owner-sharded KV blocks live
only on its owner rank (see ``vllm_torchtpu.core.owner_affine_block_pool``), so
token-parallel attention has to run every token on the rank that owns its
request. Splitting the batch into contiguous slices by position does not: the
batch interleaves requests of different owners, and owners have different
token counts.

``build_owner_token_layout`` permutes one step's tokens, on the host, into one
equal segment per owner: segment ``o`` holds the tokens of owner ``o``'s
requests in batch order, then padding. Requests stay whole, so a segment is
described by a ``query_start_loc`` row in batch request order in which the
other owners' requests are empty, the same form as the per-rank
``query_start_loc`` of the positional split. Attention gathers q and its
per-token inputs with ``perm`` before the head-to-token all-to-all, and
restores batch order with ``inv`` after the return all-to-all.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class OwnerTokenLayout:
    """Owner-major order of one step's tokens.

    Attributes:
        padded_tokens: Length of the permuted token axis, a multiple of the
            TP size.
        segment_tokens: Rows per owner, ``padded_tokens // tp``. Rank ``o``
            attends over rows ``[o * segment_tokens, (o + 1) * segment_tokens)``.
        perm: ``int32[padded_tokens]``, the batch token in each permuted row.
            Rows past an owner's tokens hold the step's padding tokens.
        inv: ``int32[num_padded_tokens]``, the permuted row of each batch
            token: ``perm[inv[t]] == t`` for every real token ``t``.
        query_start_loc: ``int32[tp, num_reqs_padded + 1]``. Row ``o`` is where
            each request starts within segment ``o``, in batch request order;
            requests of other owners and padding requests are empty.
        owner_tokens: ``int32[tp]``, the real tokens in each segment.
    """

    padded_tokens: int
    segment_tokens: int
    perm: np.ndarray
    inv: np.ndarray
    query_start_loc: np.ndarray
    owner_tokens: np.ndarray


def build_owner_token_layout(
    owners: Sequence[int] | np.ndarray,
    num_scheduled_tokens: Sequence[int] | np.ndarray,
    tp_size: int,
    token_paddings: Sequence[int],
    *,
    num_padded_tokens: int | None = None,
    num_reqs_padded: int | None = None,
    segment_multiple: int = 1,
) -> OwnerTokenLayout:
    """Lay out one step's tokens owner-major, one segment per owner.

    Args:
        owners: The owner rank of each request, in batch order.
        num_scheduled_tokens: The tokens of each request in this step.
        tp_size: The number of owners, the size of the TP group.
        token_paddings: The allowed lengths of the permuted axis. The layout
            uses the smallest one that is a multiple of
            ``tp_size * segment_multiple`` and fits every owner's tokens in
            one segment.
        num_padded_tokens: The length of the batch token axis, at least the
            number of real tokens. Defaults to the permuted length, which makes
            ``perm`` and ``inv`` inverse permutations.
        num_reqs_padded: The requests in each ``query_start_loc`` row, at least
            the real requests. Defaults to the real requests.
        segment_multiple: What ``segment_tokens`` must be a multiple of.
    """
    owners = np.asarray(owners, dtype=np.int64)
    tokens = np.asarray(num_scheduled_tokens, dtype=np.int64)
    if owners.ndim != 1 or owners.shape != tokens.shape:
        raise ValueError(
            f"Expected one owner per request, got owners {owners.shape} and "
            f"token counts {tokens.shape}"
        )
    if tp_size < 1 or segment_multiple < 1:
        raise ValueError(
            f"tp_size ({tp_size}) and segment_multiple ({segment_multiple}) "
            "must be positive"
        )
    num_reqs = owners.size
    if num_reqs and (owners.min() < 0 or owners.max() >= tp_size):
        raise ValueError(f"Owners must be in [0, {tp_size}): {owners}")
    if num_reqs and tokens.min() < 0:
        raise ValueError(f"Token counts must be non-negative: {tokens}")
    total = int(tokens.sum())
    if total < 1:
        raise ValueError("The step has no scheduled tokens")
    reqs_padded = num_reqs if num_reqs_padded is None else num_reqs_padded
    if reqs_padded < num_reqs:
        raise ValueError(
            f"num_reqs_padded ({reqs_padded}) is below the {num_reqs} requests"
        )

    # Row o: exclusive prefix sums of owner o's token counts, in batch order.
    request_ids = np.arange(num_reqs)
    per_owner = np.zeros((tp_size, num_reqs), dtype=np.int64)
    per_owner[owners, request_ids] = tokens
    query_start_loc = np.zeros((tp_size, reqs_padded + 1), dtype=np.int64)
    query_start_loc[:, 1 : num_reqs + 1] = np.cumsum(per_owner, axis=1)
    query_start_loc[:, num_reqs + 1 :] = query_start_loc[:, num_reqs, None]
    owner_tokens = query_start_loc[:, num_reqs]

    need = tp_size * int(owner_tokens.max())
    step = tp_size * segment_multiple
    fits = [size for size in token_paddings if size >= need and size % step == 0]
    if not fits:
        raise ValueError(
            f"No token padding holds {need} tokens in a multiple of {step}: "
            f"{sorted(token_paddings)}"
        )
    padded = min(fits)
    segment = padded // tp_size
    padded_batch = padded if num_padded_tokens is None else num_padded_tokens
    if padded_batch < total:
        raise ValueError(
            f"num_padded_tokens ({padded_batch}) is below the {total} tokens"
        )

    # Request i starts at row owner * segment + (its start within the segment).
    request_rows = owners * segment + query_start_loc[owners, request_ids]
    token_request = np.repeat(request_ids, tokens)
    batch_starts = np.cumsum(tokens) - tokens
    rows = request_rows[token_request] + (
        np.arange(total) - batch_starts[token_request]
    )
    used = np.zeros(padded, dtype=bool)
    used[rows] = True
    pad_rows = np.flatnonzero(~used)

    perm = np.empty(padded, dtype=np.int32)
    perm[rows] = np.arange(total)
    # Pair padding rows with padding tokens. When their counts differ, repeat
    # them; with none, use row/token 0. Either way the pairs are never read
    # back, as padding is outside every request of every segment.
    perm[pad_rows] = np.resize(np.arange(total, padded_batch), pad_rows.size)
    inv = np.empty(padded_batch, dtype=np.int32)
    inv[:total] = rows
    inv[total:] = np.resize(pad_rows, padded_batch - total)
    return OwnerTokenLayout(
        padded_tokens=padded,
        segment_tokens=segment,
        perm=perm,
        inv=inv,
        query_start_loc=query_start_loc.astype(np.int32),
        owner_tokens=owner_tokens.astype(np.int32),
    )


def _to_2d_numpy(table: Any) -> np.ndarray:
    if hasattr(table, "block_tables") and table.block_tables is not None:
        table = table.block_tables
    raw = table.get_cpu_tensor() if hasattr(table, "get_cpu_tensor") else table
    arr = (
        raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    )
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D block table [reqs, blocks], got shape {arr.shape}"
        )
    return arr


def extract_request_owners_from_block_table(
    block_table: Any,
    num_blocks_per_req_or_layout: Any,
    owner_layout: Any = None,
    *,
    num_reqs: int | None = None,
    num_blocks_per_req: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    start_index: int = 0,
    default_owner: int = 0,
) -> np.ndarray:
    """Resolve each active request's owner rank in ``[0, tp_size)`` from its
    allocated block table(s) and ``owner_layout``.

    Scans the request's allocated block IDs for the first block in
    ``[owner_layout.num_replicated, owner_layout.num_blocks)`` and maps it via
    ``owner_layout.region_of(block_id)``. Falls back to ``default_owner`` when a
    short or dummy request only occupies replicated blocks ``< num_replicated``.
    """
    if hasattr(num_blocks_per_req_or_layout, "region_of"):
        layout = num_blocks_per_req_or_layout
        blocks_per_req = num_blocks_per_req
    else:
        layout = owner_layout
        blocks_per_req = (
            num_blocks_per_req_or_layout
            if num_blocks_per_req is None
            else num_blocks_per_req
        )
    if layout is None or not hasattr(layout, "region_of"):
        raise ValueError(f"Expected a valid OwnerLayout, got {layout}")
    if isinstance(block_table, dict):
        raw_tables = [
            m
            for m in block_table.values()
            if getattr(m, "block_tables", None) is not None
        ]
    elif (
        isinstance(block_table, (np.ndarray, torch.Tensor))
        or hasattr(block_table, "get_cpu_tensor")
        or getattr(block_table, "block_tables", None) is not None
    ):
        raw_tables = [block_table]
    else:
        raw_tables = list(block_table)
    tables = tuple(_to_2d_numpy(t) for t in raw_tables)
    if not tables:
        raise ValueError("Expected at least one block table")
    total_reqs = tables[0].shape[0] - start_index if num_reqs is None else int(num_reqs)
    if (
        total_reqs < 0
        or start_index < 0
        or start_index + total_reqs > tables[0].shape[0]
    ):
        raise ValueError(
            f"Invalid request slice [{start_index}, {start_index + total_reqs}) "
            f"for block table with {tables[0].shape[0]} rows"
        )
    limits = (
        None if blocks_per_req is None else np.asarray(blocks_per_req, dtype=np.int64)
    )
    if limits is not None and limits.shape != (total_reqs,):
        raise ValueError(
            f"Expected num_blocks_per_req of shape ({total_reqs},), got {limits.shape}"
        )
    owners = np.full(total_reqs, int(default_owner), dtype=np.int64)
    num_rep = int(layout.num_replicated)
    num_total = int(layout.num_blocks)
    for req_idx in range(total_reqs):
        row_idx = start_index + req_idx
        for tbl in tables:
            max_cols = (
                tbl.shape[1]
                if limits is None
                else max(0, min(tbl.shape[1], int(limits[req_idx])))
            )
            row = tbl[row_idx, :max_cols]
            owned_mask = (row >= num_rep) & (row < num_total)
            if np.any(owned_mask):
                first_block = int(row[int(np.argmax(owned_mask))])
                owners[req_idx] = int(layout.region_of(first_block))
                break
    return owners


def owner_layout_tensors(
    layout: OwnerTokenLayout,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ``layout`` into ``(owner_perm, owner_inv, owner_q_loc)`` int32
    tensors with padding rows/tokens masked to ``-1``."""
    target_device = torch.device("cpu") if device is None else torch.device(device)
    seg_col = np.arange(layout.padded_tokens) % layout.segment_tokens
    seg_owner = np.arange(layout.padded_tokens) // layout.segment_tokens
    valid_perm = seg_col < layout.owner_tokens[seg_owner]
    perm_masked = np.where(valid_perm, layout.perm, -1).astype(np.int32)
    total_real = int(layout.owner_tokens.sum())
    valid_inv = np.arange(layout.inv.shape[0]) < total_real
    inv_masked = np.where(valid_inv, layout.inv, -1).astype(np.int32)
    return (
        torch.as_tensor(perm_masked, dtype=torch.int32, device=target_device),
        torch.as_tensor(inv_masked, dtype=torch.int32, device=target_device),
        torch.as_tensor(
            layout.query_start_loc, dtype=torch.int32, device=target_device
        ),
    )


def attach_owner_token_layout(
    attn_metadata: Any,
    layout: OwnerTokenLayout,
    *,
    device: torch.device | str | None = None,
) -> Any:
    """Attach ``layout`` and device ``int32`` tensors onto ``attn_metadata``."""
    targets = (
        tuple(attn_metadata.values())
        if isinstance(attn_metadata, dict)
        else (attn_metadata,)
    )
    first_q_loc: Any = getattr(targets[0], "query_start_loc", None) if targets else None
    if device is not None:
        target_device: torch.device | str = device
    elif isinstance(first_q_loc, torch.Tensor):
        target_device = first_q_loc.device
    else:
        target_device = torch.device("cpu")
    perm_t, inv_t, q_loc_t = owner_layout_tensors(layout, device=target_device)
    for meta in targets:
        meta.owner_token_layout = layout
        meta.owner_perm = perm_t
        meta.owner_inv = inv_t
        meta.owner_q_loc = q_loc_t
    return attn_metadata


def prepare_kv_shard_owner_token_layout(
    block_table: Any,
    owner_layout: Any,
    num_scheduled_tokens: Sequence[int] | np.ndarray | torch.Tensor,
    token_paddings: Sequence[int],
    *,
    num_padded_tokens: int | None = None,
    num_reqs_padded: int | None = None,
    segment_multiple: int = 1,
    num_blocks_per_req: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    start_index: int = 0,
    default_owner: int = 0,
    attn_metadata: Any = None,
    device: torch.device | str | None = None,
) -> OwnerTokenLayout:
    """Extract request owners from ``block_table``, build ``OwnerTokenLayout``,
    and optionally attach it onto ``attn_metadata``."""
    tokens_arr = (
        num_scheduled_tokens.detach().cpu().numpy()
        if isinstance(num_scheduled_tokens, torch.Tensor)
        else np.asarray(num_scheduled_tokens, dtype=np.int64)
    )
    owners = extract_request_owners_from_block_table(
        block_table,
        owner_layout,
        num_reqs=int(tokens_arr.shape[0]),
        num_blocks_per_req=num_blocks_per_req,
        start_index=start_index,
        default_owner=default_owner,
    )
    layout = build_owner_token_layout(
        owners,
        tokens_arr,
        int(owner_layout.num_owners),
        token_paddings,
        num_padded_tokens=num_padded_tokens,
        num_reqs_padded=num_reqs_padded,
        segment_multiple=segment_multiple,
    )
    meta_target = (
        attn_metadata
        if attn_metadata is not None
        else (
            block_table
            if isinstance(block_table, dict)
            or getattr(block_table, "block_tables", None) is not None
            else None
        )
    )
    if meta_target is not None:
        attach_owner_token_layout(meta_target, layout, device=device)
    return layout
