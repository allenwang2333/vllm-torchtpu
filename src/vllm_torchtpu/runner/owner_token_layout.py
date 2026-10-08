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

import numpy as np


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
