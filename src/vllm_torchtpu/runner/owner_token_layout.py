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

from vllm_torchtpu import envs


@dataclasses.dataclass(frozen=True, slots=True)
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
    owners: np.ndarray | None = None


def build_owner_token_layout(
    owners: Sequence[int] | np.ndarray,
    num_scheduled_tokens: Sequence[int] | np.ndarray,
    tp_size: int,
    token_paddings: Sequence[int],
    *,
    num_padded_tokens: int | None = None,
    num_reqs_padded: int | None = None,
    segment_multiple: int = 1,
    scale_paddings: bool = False,
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
        scale_paddings: Whether to scale ``token_paddings`` by
            ``tp_size * segment_multiple`` when ``need`` exceeds the largest
            aligned bucket.
    """
    owners = (
        owners.astype(np.int32, copy=True)
        if isinstance(owners, np.ndarray)
        and owners.ndim == 1
        and np.issubdtype(owners.dtype, np.integer)
        else np.asarray(owners, dtype=np.int32).copy()
    )
    tokens = (
        num_scheduled_tokens.astype(np.int32, copy=False)
        if isinstance(num_scheduled_tokens, np.ndarray)
        and num_scheduled_tokens.ndim == 1
        and np.issubdtype(num_scheduled_tokens.dtype, np.integer)
        else np.asarray(num_scheduled_tokens, dtype=np.int32)
    )
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
    if not num_reqs:
        raise ValueError("The step has no scheduled tokens")
    if int(owners.min()) < 0 or int(owners.max()) >= tp_size:
        raise ValueError(f"Owners must be in [0, {tp_size}): {owners}")
    if int(tokens.min()) < 0:
        raise ValueError(f"Token counts must be non-negative: {tokens}")
    batch_ends = np.cumsum(tokens, dtype=np.int32)
    total = int(batch_ends[-1])
    if total < 1:
        raise ValueError("The step has no scheduled tokens")
    reqs_padded = num_reqs if num_reqs_padded is None else num_reqs_padded
    if reqs_padded < num_reqs:
        raise ValueError(
            f"num_reqs_padded ({reqs_padded}) is below the {num_reqs} requests"
        )

    # Row o: exclusive prefix sums of owner o's token counts, in batch order.
    request_ids = np.arange(num_reqs, dtype=np.int32)
    query_start_loc = np.zeros((tp_size, reqs_padded + 1), dtype=np.int32)
    q_active = query_start_loc[:, 1 : num_reqs + 1]
    q_active[owners, request_ids] = tokens
    np.cumsum(q_active, axis=1, dtype=np.int32, out=q_active)
    query_start_loc[:, num_reqs + 1 :] = query_start_loc[:, num_reqs, None]
    owner_tokens = query_start_loc[:, num_reqs]

    need = tp_size * int(owner_tokens.max())
    step = tp_size * segment_multiple
    paddings = (
        _owner_paddings(token_paddings, tp_size, need, segment_multiple)
        if scale_paddings
        else token_paddings
    )
    padded = select_owner_padded_tokens(need, step, paddings)
    segment = padded // tp_size
    padded_batch = padded if num_padded_tokens is None else num_padded_tokens
    if padded_batch < total:
        raise ValueError(
            f"num_padded_tokens ({padded_batch}) is below the {total} tokens"
        )

    # Request i starts at row owner * segment + (its start within the segment).
    token_idx = np.arange(total, dtype=np.int32)
    request_offsets = (
        owners * segment + query_start_loc[owners, request_ids] - (batch_ends - tokens)
    )
    rows = np.repeat(request_offsets, tokens) + token_idx
    perm = np.zeros(padded, dtype=np.int32)
    perm[rows] = token_idx
    if padded_batch == total:
        inv = rows
    else:
        used = np.zeros(padded, dtype=bool)
        used[rows] = True
        pad_rows = np.flatnonzero(~used)
        # Pair padding rows with padding tokens. When their counts differ,
        # repeat them; with none, use row/token 0. Either way the pairs are
        # never read back, as padding is outside every request of every segment.
        perm[pad_rows] = np.resize(
            np.arange(total, padded_batch, dtype=np.int32), pad_rows.size
        )
        inv = np.empty(padded_batch, dtype=np.int32)
        inv[:total] = rows
        inv[total:] = np.resize(pad_rows, padded_batch - total)
    return OwnerTokenLayout(
        padded_tokens=padded,
        segment_tokens=segment,
        perm=perm,
        inv=inv,
        query_start_loc=query_start_loc,
        owner_tokens=owner_tokens.copy(),
        owners=owners,
    )


def select_owner_padded_tokens(
    need: int, step: int, token_paddings: Sequence[int]
) -> int:
    """Smallest padding in ``token_paddings`` that is ``>= need`` and a multiple
    of ``step``."""
    fits = [size for size in token_paddings if size >= need and size % step == 0]
    if not fits:
        raise ValueError(
            f"No token padding holds {need} tokens in a multiple of {step}: "
            f"{sorted(token_paddings)}"
        )
    return min(fits)


def _owner_paddings(
    token_paddings: Sequence[int],
    tp_size: int,
    need: int,
    segment_multiple: int = 1,
) -> Sequence[int]:
    """Return ``token_paddings`` when it covers ``need``, else scale by
    ``tp_size * segment_multiple`` so an overflowing owner gets a whole-bucket
    segment."""
    step = tp_size * segment_multiple
    if any(size >= need and size % step == 0 for size in token_paddings):
        return token_paddings
    return [step * size for size in token_paddings]


def _owners_from_first_blocks(first: np.ndarray, layout: Any) -> np.ndarray:
    invalid = (first < layout.num_replicated) | (first >= layout.num_blocks)
    if invalid.any():
        bad = int(first[invalid][0])
        raise ValueError(
            f"Block {bad} in the sharded block table is not in an owner "
            f"region of {layout}"
        )
    return ((first - layout.num_replicated) // layout.blocks_per_owner).astype(
        np.int32, copy=False
    )


def request_owners(
    block_ids: np.ndarray,
    rows: slice | Sequence[int] | np.ndarray,
    layout: Any,
) -> np.ndarray:
    """Owner rank of each request in ``rows`` from column 0 of a sharded KV
    cache group's host block table."""
    return _owners_from_first_blocks(
        np.asarray(block_ids[rows, 0], dtype=np.int32), layout
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
    num_table_rows = tables[0].shape[0]
    total_reqs = num_table_rows - start_index if num_reqs is None else int(num_reqs)
    if total_reqs < 0 or start_index < 0 or start_index + total_reqs > num_table_rows:
        raise ValueError(
            f"Invalid request slice [{start_index}, {start_index + total_reqs}) "
            f"for block table with {num_table_rows} rows"
        )
    limits = (
        None if blocks_per_req is None else np.asarray(blocks_per_req, dtype=np.int32)
    )
    if limits is not None and limits.shape != (total_reqs,):
        raise ValueError(
            f"Expected num_blocks_per_req of shape ({total_reqs},), got {limits.shape}"
        )
    owners = np.full(total_reqs, int(default_owner), dtype=np.int32)
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
    total_real = int(layout.owner_tokens.sum())
    perm_np = layout.perm.astype(np.int32, copy=True)
    perm_2d = perm_np.reshape(-1, layout.segment_tokens)
    for owner_idx, count in enumerate(layout.owner_tokens.tolist()):
        perm_2d[owner_idx, count:] = -1
    inv_np = layout.inv.astype(np.int32, copy=True)
    inv_np[total_real:] = -1
    q_loc_np = np.ascontiguousarray(layout.query_start_loc, dtype=np.int32)
    perm = torch.from_numpy(perm_np).to(device=target_device, non_blocking=True)
    inv = torch.from_numpy(inv_np).to(device=target_device, non_blocking=True)
    q_loc = torch.from_numpy(q_loc_np).to(device=target_device, non_blocking=True)
    _mark_token_dim_dynamic(perm, inv)
    return perm, inv, q_loc


def _mark_token_dim_dynamic(*tensors: torch.Tensor) -> None:
    """Keep owner perm/inv token dims symbolic under torch.compile.

    Left static, the first traced bucket's length (e.g. 288) unifies with the
    dynamic ``input_ids`` dim through ``_unpermute_owner_tokens_torch`` and
    dynamo raises ConstraintViolationError (KV_SHARD 40L, c7x8 Job 1250).
    """
    for tensor in tensors:
        if tensor.shape[0] >= 2:
            torch._dynamo.mark_dynamic(tensor, 0)


def attach_owner_token_layout(
    attn_metadata: Any,
    layout: OwnerTokenLayout,
    *,
    device: torch.device | str | None = None,
) -> Any:
    """Attach ``layout`` and device ``int32`` tensors onto ``attn_metadata``."""
    targets = (
        tuple({id(m): m for m in attn_metadata.values()}.values())
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
        else np.asarray(num_scheduled_tokens, dtype=np.int32)
    )
    owners = extract_request_owners_from_block_table(
        block_table,
        owner_layout,
        num_reqs=int(tokens_arr.shape[0]),
        num_blocks_per_req=num_blocks_per_req,
        start_index=start_index,
        default_owner=default_owner,
    )
    tp_size = int(owner_layout.num_owners)
    layout = build_owner_token_layout(
        owners,
        tokens_arr,
        tp_size,
        token_paddings,
        num_padded_tokens=num_padded_tokens,
        num_reqs_padded=num_reqs_padded,
        segment_multiple=segment_multiple,
        scale_paddings=True,
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


def _positional_layout(
    query_start_loc: torch.Tensor,
    num_padded_tokens: int,
    tp_size: int,
    token_paddings: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity permutation for dummy/profile runs without allocated blocks."""
    segment = max(1, -(-num_padded_tokens // tp_size))
    need = tp_size * segment
    padded = select_owner_padded_tokens(
        need, tp_size, _owner_paddings(token_paddings, tp_size, need)
    )
    segment = padded // tp_size
    device = query_start_loc.device
    perm = torch.arange(padded, dtype=torch.int32, device=device)
    if padded > num_padded_tokens:
        perm[num_padded_tokens:] = -1
    inv = torch.arange(num_padded_tokens, dtype=torch.int32, device=device)
    starts = torch.arange(tp_size, dtype=torch.int32, device=device) * segment
    q_loc = (query_start_loc[None, :] - starts[:, None]).clamp(0, segment)
    _mark_token_dim_dynamic(perm, inv)
    return perm, inv, q_loc


def _sharded_group_id(runner: Any) -> int:
    cfg = runner.kv_cache_config
    cached = getattr(runner, "_kv_shard_group_cache", None)
    if cached is None or cached[0] != id(cfg):
        from vllm_torchtpu.core.owner_affine_block_pool import is_replicated_spec

        group_id = next(
            i
            for i, group in enumerate(cfg.kv_cache_groups)
            if not is_replicated_spec(group.kv_cache_spec)
        )
        cached = (id(cfg), group_id)
        runner._kv_shard_group_cache = cached
    return int(cached[1])


def _col0_token(table: Any, start: int, num_reqs: int) -> tuple | None:
    """Identity of column 0 of block-table rows ``[start, start + num_reqs)``.

    Tracked tables (`block_table_delta.ensure_tracked`) bump ``_bt_rewrites``
    on every row mutation except ``append_row``, which writes from column
    ``num_blocks_per_row[row]`` on. So while the counter is unchanged, column 0
    of a row that already held a block cannot change. None when untracked.
    """
    rewrites = getattr(table, "_bt_rewrites", None)
    return None if rewrites is None else (table, rewrites, start, num_reqs)


def _col0_snapshot_token(table: Any, start: int, num_reqs: int) -> tuple | None:
    """`_col0_token` to cache with a column-0 snapshot: None unless every row
    already holds a block, since `append_row` on an empty row writes column 0."""
    token = _col0_token(table, start, num_reqs)
    if token is None or not bool(
        (table.num_blocks_per_row[start : start + num_reqs] > 0).all()
    ):
        return None
    return token


def step_owner_required_tokens(
    runner: Any,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req: Any,
) -> int:
    """Minimum padded batch tokens ``tp_size * max(owner_tokens)`` required so
    the step's ``owner_perm`` length matches the selected token bucket."""
    if not envs.TPU_DEEPSEEK_V41_KV_SHARD or num_reqs <= 0:
        return 0
    parallel_cfg = getattr(runner, "parallel_config", None)
    tp_size = int(getattr(parallel_cfg, "tensor_parallel_size", 1))
    if (
        tp_size <= 1
        or getattr(runner, "kv_cache_config", None) is None
        or getattr(runner, "input_batch", None) is None
    ):
        return 0
    owner_layout = getattr(runner, "_kv_shard_owner_layout", None)
    if owner_layout is None:
        from vllm_torchtpu.core.owner_affine_block_pool import worker_owner_layout

        owner_layout = worker_owner_layout(runner.vllm_config, runner.kv_cache_config)
        runner._kv_shard_owner_layout = owner_layout
    table = runner.input_batch.block_table.block_tables[_sharded_group_id(runner)]
    block_ids = getattr(table, "_bt_cpu_np", None)
    if block_ids is None:
        block_ids = table.block_table.np
    raw_first_blocks = block_ids[start_index : start_index + num_reqs, 0]
    scheduled_arr = np.asarray(num_scheduled_tokens_per_req)
    req_meta_key = (tp_size, owner_layout, start_index, num_reqs)
    col0_token = _col0_token(table, start_index, num_reqs)
    cached_req = getattr(runner, "_kv_shard_required_tokens_cache", None)
    if (
        cached_req is not None
        and cached_req[0] == req_meta_key
        and (
            (col0_token is not None and cached_req[4] == col0_token)
            or np.array_equal(cached_req[1], raw_first_blocks)
        )
        and (
            # An owning read-only array cannot have changed since cached.
            (
                cached_req[5] is scheduled_arr
                and scheduled_arr.base is None
                and not scheduled_arr.flags.writeable
            )
            or np.array_equal(cached_req[2], scheduled_arr)
        )
    ):
        return cached_req[3]
    first_blocks = np.array(raw_first_blocks, dtype=np.int32, copy=True)
    scheduled = np.array(scheduled_arr, dtype=np.int32, copy=True)
    snapshot_token = _col0_snapshot_token(table, start_index, num_reqs)
    if (
        (first_blocks < owner_layout.num_replicated)
        | (first_blocks >= owner_layout.num_blocks)
    ).any():
        runner._kv_shard_required_tokens_cache = (
            req_meta_key,
            first_blocks,
            scheduled,
            0,
            snapshot_token,
            scheduled_arr,
        )
        return 0
    owners = _owners_from_first_blocks(first_blocks, owner_layout)
    runner._kv_shard_owners_cache = (owner_layout, first_blocks, owners)
    per_owner = np.bincount(owners, weights=scheduled, minlength=tp_size)
    required = int(tp_size * int(per_owner.max()))
    runner._kv_shard_required_tokens_cache = (
        req_meta_key,
        first_blocks,
        scheduled,
        required,
        snapshot_token,
        scheduled_arr,
    )
    return required


def attach_step_owner_token_layout(
    runner: Any,
    per_layer_attn_metadata: dict[str, Any],
    *,
    num_tokens_padded: int | None = None,
) -> OwnerTokenLayout | None:
    """Build and attach the step's ``OwnerTokenLayout`` onto
    ``per_layer_attn_metadata`` when ``TPU_DEEPSEEK_V41_KV_SHARD=1``."""
    if not envs.TPU_DEEPSEEK_V41_KV_SHARD:
        return None
    tp_size = int(runner.parallel_config.tensor_parallel_size)
    if tp_size <= 1 or not per_layer_attn_metadata:
        return None
    ctx = getattr(runner, "_attn_metadata_builder_ctx", None)
    padded_tokens = (
        int(getattr(runner, "num_tokens_padded", 0))
        if num_tokens_padded is None
        else int(num_tokens_padded)
    )
    if ctx is None or ctx.num_reqs <= 0 or padded_tokens <= 0:
        return None
    # Layers share per-group metadata objects; re-assigning the same fields on
    # a shared object is idempotent and cheaper than an id()-keyed dedup.
    metas = per_layer_attn_metadata.values()
    first = next(iter(metas))
    raw_paddings = runner.num_tokens_paddings
    paddings_cache = getattr(runner, "_kv_shard_paddings_cache", None)
    if paddings_cache is None or paddings_cache[0] != id(raw_paddings):
        paddings_cache = (id(raw_paddings), tuple(raw_paddings))
        runner._kv_shard_paddings_cache = paddings_cache
    paddings_tuple = paddings_cache[1]
    if ctx.position_ids_override is not None:
        perm, inv, q_loc = _positional_layout(
            first.query_start_loc,
            padded_tokens,
            tp_size,
            paddings_tuple,
        )
        for meta in metas:
            meta.owner_perm = perm
            meta.owner_inv = inv
            meta.owner_q_loc = q_loc
        return None
    q_loc_cpu = (
        ctx.query_start_loc_cpu
        if ctx.query_start_loc_cpu is not None
        else getattr(runner, "query_start_loc_np", None)
    )
    if q_loc_cpu is None:
        return None
    owner_layout = getattr(runner, "_kv_shard_owner_layout", None)
    if owner_layout is None:
        from vllm_torchtpu.core.owner_affine_block_pool import worker_owner_layout

        owner_layout = worker_owner_layout(runner.vllm_config, runner.kv_cache_config)
        runner._kv_shard_owner_layout = owner_layout
    table = runner.input_batch.block_table.block_tables[_sharded_group_id(runner)]
    block_ids = getattr(table, "_bt_cpu_np", None)
    if block_ids is None:
        block_ids = table.block_table.np
    req_indices = getattr(ctx, "req_indices", None)
    rows = (
        slice(ctx.start_index, ctx.start_index + ctx.num_reqs)
        if req_indices is None
        else req_indices
    )
    col0_token = (
        _col0_token(table, ctx.start_index, ctx.num_reqs)
        if req_indices is None
        else None
    )
    q_loc_tuple = (
        (
            q_loc_cpu
            if len(q_loc_cpu) == ctx.num_reqs + 1
            else q_loc_cpu[: ctx.num_reqs + 1]
        )
        if isinstance(q_loc_cpu, tuple)
        else None
    )
    q_loc_np = (
        None
        if q_loc_tuple is not None
        else (
            q_loc_cpu[: ctx.num_reqs + 1]
            if isinstance(q_loc_cpu, np.ndarray)
            and q_loc_cpu.ndim == 1
            and np.issubdtype(q_loc_cpu.dtype, np.integer)
            else np.asarray(
                q_loc_cpu.detach().cpu().numpy()
                if isinstance(q_loc_cpu, torch.Tensor)
                else q_loc_cpu,
                dtype=np.int32,
            )[: ctx.num_reqs + 1]
        )
    )
    first_blocks = block_ids[rows, 0]
    num_reqs_padded = int(first.query_start_loc.shape[0]) - 1
    step_key = (
        tp_size,
        padded_tokens,
        num_reqs_padded,
        first.query_start_loc.device,
        owner_layout,
        paddings_tuple,
    )
    cached_step = getattr(runner, "_kv_shard_step_layout_cache", None)
    if (
        cached_step is not None
        and cached_step[0] == step_key
        and (
            isinstance(cached_step[2], tuple) and cached_step[2] == q_loc_tuple
            if q_loc_tuple is not None
            else np.array_equal(cached_step[2], q_loc_np)
        )
        and (
            (col0_token is not None and cached_step[7] == col0_token)
            or np.array_equal(cached_step[1], first_blocks)
        )
    ):
        layout, perm_t, inv_t, q_loc_t = cached_step[3:7]
        for meta in metas:
            meta.owner_token_layout = layout
            meta.owner_perm = perm_t
            meta.owner_inv = inv_t
            meta.owner_q_loc = q_loc_t
        return layout
    owners_cache = getattr(runner, "_kv_shard_owners_cache", None)
    if (
        owners_cache is not None
        and owners_cache[0] == owner_layout
        and np.array_equal(owners_cache[1], first_blocks)
    ):
        first_blocks_copy, owners = owners_cache[1], owners_cache[2]
    else:
        first_blocks_copy = np.array(first_blocks, dtype=np.int32, copy=True)
        owners = _owners_from_first_blocks(first_blocks_copy, owner_layout)
        runner._kv_shard_owners_cache = (owner_layout, first_blocks_copy, owners)
    q_loc_arr = (
        np.asarray(q_loc_tuple, dtype=np.int32) if q_loc_np is None else q_loc_np
    )
    step_paddings = (
        tuple(p for p in paddings_tuple if p >= padded_tokens) or paddings_tuple
    )
    layout = build_owner_token_layout(
        owners,
        q_loc_arr[1 : ctx.num_reqs + 1] - q_loc_arr[: ctx.num_reqs],
        tp_size,
        step_paddings,
        num_padded_tokens=padded_tokens,
        num_reqs_padded=num_reqs_padded,
        scale_paddings=True,
    )
    perm_t, inv_t, q_loc_t = owner_layout_tensors(
        layout, device=first.query_start_loc.device
    )
    for meta in metas:
        meta.owner_token_layout = layout
        meta.owner_perm = perm_t
        meta.owner_inv = inv_t
        meta.owner_q_loc = q_loc_t
    runner._kv_shard_step_layout_cache = (
        step_key,
        first_blocks_copy,
        q_loc_tuple if q_loc_tuple is not None else q_loc_arr.copy(),
        layout,
        perm_t,
        inv_t,
        q_loc_t,
        None
        if col0_token is None
        else _col0_snapshot_token(table, ctx.start_index, ctx.num_reqs),
    )
    return layout
