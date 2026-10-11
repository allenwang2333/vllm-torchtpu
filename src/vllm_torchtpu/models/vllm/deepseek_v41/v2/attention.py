"""TPU MLA and compressor adapters with upstream cache ownership."""

import contextlib
import dataclasses
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41 import attention as upstream
from vllm.models.deepseek_v41 import compressor as compressor_upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR

from .attention_kernels import (
    paged_attention,
    paged_indexer,
    sparse_fp8_attention,
    sparse_fp8_attention_qo,
)
from .cache import ROW_BYTES, NativeCacheView, PagedCacheRows, write_cache
from .ced import prefix_replay_enabled
from .ops import device_hooks, rms_norm, rotary, round_fp8_activation

_active_stage_attn_cache: dict[Any, Any] | None = None


@contextlib.contextmanager
def stage_attention_cache_scope():
    """Share stage-invariant SWA and compressed-KV addressing across layers."""
    global _active_stage_attn_cache
    prev = _active_stage_attn_cache
    _active_stage_attn_cache = {}
    try:
        yield
    finally:
        _active_stage_attn_cache = prev


def select_candidate_blocks(logits, lengths, topk_blocks, block_size):
    if block_size < 1 or topk_blocks < 0:
        raise ValueError("invalid candidate block parameters")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    lengths = torch.as_tensor(lengths, device=logits.device, dtype=torch.int32)
    visible = torch.arange(width, device=logits.device, dtype=torch.int32) < lengths
    scores = F.pad(
        logits.masked_fill(~visible, -torch.inf),
        (0, -width % block_size),
        value=-torch.inf,
    )
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    last = (lengths - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32) == last,
        torch.inf,
    )
    selected = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, selected.indices, selected.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_compressed_indices(scores, lengths, topk, *, candidates=None):
    """Return logical compressed positions; unavailable picks are always -1.

    Do not carry a source layer's SWA concatenation offset into reuse layers.
    Equal scores retain PyTorch topk's unspecified tie ordering.
    """
    if topk < 0:
        raise ValueError("topk must be nonnegative")
    valid = torch.arange(
        scores.shape[-1], device=scores.device, dtype=torch.int32
    ) < torch.as_tensor(lengths, device=scores.device, dtype=torch.int32)
    if candidates is not None:
        valid = valid & candidates
    masked = scores.masked_fill(~valid, -torch.inf)
    indices = (
        masked.topk(min(topk, scores.shape[-1]), dim=-1)
        .indices.to(torch.int32)
        .sort(-1)
        .values
    )
    return indices.masked_fill(
        masked.reshape(-1)
        .index_select(
            0,
            (
                (
                    torch.arange(
                        masked.reshape(-1, masked.shape[-1]).shape[0],
                        device=masked.device,
                        dtype=torch.int32,
                    )
                    * masked.shape[-1]
                ).reshape(*masked.shape[:-1], 1)
                + indices
            ).reshape(-1),
        )
        .reshape(indices.shape)
        == -torch.inf,
        -1,
    )


def sparse_attention(q, kv, sink, indices, scale):
    """Shared K/V latent with a per-query-head zero-value attention sink."""
    if kv.shape[1] == 0 or indices.shape[-1] == 0:
        return torch.zeros_like(q)
    valid = (indices >= 0) & (indices < kv.shape[1])
    batch = (
        torch.arange(kv.shape[0], device=kv.device, dtype=torch.int32) * kv.shape[1]
    )[:, None, None]
    selected = (
        kv.flatten(0, 1)
        .index_select(
            0,
            (batch + indices.to(torch.int32).clamp(0, kv.shape[1] - 1)).reshape(-1),
        )
        .reshape(*indices.shape, *kv.shape[2:])
    )
    selected = selected.masked_fill(~valid.unsqueeze(-1), 0).float()
    logits = torch.einsum("bshd,bskd->bshk", q.float(), selected) * scale
    logits = logits.masked_fill(~valid.unsqueeze(-2), -torch.inf)
    if q.dtype == torch.bfloat16:
        # Match the released kernel: online blocks of 64, BF16 unnormalized
        # exponentials for the value GEMM, FP32 denominator, sink added last.
        maximum = torch.full_like(logits[..., 0], -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q, dtype=torch.float32)
        for start in range(0, logits.shape[-1], 64):
            scores = logits[..., start : start + 64]
            new_maximum = torch.maximum(maximum, scores.amax(-1))
            correction = torch.exp(maximum - new_maximum)
            exponentials = torch.exp(scores - new_maximum.unsqueeze(-1))
            denominator = denominator * correction + exponentials.sum(-1)
            numerator = numerator * correction.unsqueeze(-1) + torch.einsum(
                "bshk,bskd->bshd",
                exponentials.to(torch.bfloat16).float(),
                selected[:, :, start : start + 64],
            )
            maximum = new_maximum
        denominator = denominator + torch.exp(sink.float() - maximum)
        return (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    sinks = sink.float().expand(*logits.shape[:-1]).unsqueeze(-1)
    probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, selected).to(q.dtype)


def _block_table(metadata):
    return metadata.block_tables.reshape(metadata.seq_lens.shape[0], -1)


def _token_requests(metadata, positions):
    """Request index of every token row: ``bucketize(rows, qsl[1:], right=True)``.

    Counts the (non-decreasing) request ends at or below each row with one
    ``[tokens, requests]`` compare + reduce. ``torch.bucketize`` lowers on TPU
    to a fusion taking one scalar operand per boundary, which cost 31-37 us
    per call at 288 tokens x 48 requests (C=48 DSpark K=5 xprof).
    """
    rows = torch.arange(positions.shape[0], dtype=torch.int32, device=positions.device)
    ends = metadata.query_start_loc[1:]
    return (rows.unsqueeze(1) >= ends.unsqueeze(0)).sum(-1, dtype=torch.int32)


def _window_starts():
    """Per-request first position holding SWA KV, or None (SWA bounded replay)."""
    stage_cache = _active_stage_attn_cache
    if stage_cache is not None:
        if "window_starts" not in stage_cache:
            stage_cache["window_starts"] = get_forward_context().additional_kwargs.get(
                "deepseek_v41_window_starts"
            )
        return stage_cache["window_starts"]
    return get_forward_context().additional_kwargs.get("deepseek_v41_window_starts")


def _replayed(metadata, positions, starts, window):
    """Tokens a prefix hit replays: their prefix-cached KV must stay as is."""
    requests = _token_requests(metadata, positions).clamp(max=starts.shape[0] - 1)
    start = starts.index_select(0, requests).int()
    pos_i32 = positions.int()
    return (start > 0) & (pos_i32 >= start) & (pos_i32 < start + window)


def _cache_slots(cache, metadata, positions, *, compress_ratio=1, ring=False):
    """Derive this cache's addresses from the shared TPU metadata contract."""
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1, dtype=torch.int32)
    table = _block_table(metadata)
    requests = _token_requests(metadata, positions).clamp(max=table.shape[0] - 1)
    rows_per_page = cache.shape[1]
    logical = positions.int() // compress_ratio
    columns = (
        torch.zeros_like(logical)
        if ring
        else (logical // rows_per_page).clamp(max=table.shape[1] - 1)
    )
    blocks = (
        table.reshape(-1).index_select(0, requests * table.shape[1] + columns).int()
    )
    slots = blocks * rows_per_page + logical % rows_per_page
    # Padded request boundaries repeat the last real token offset. Keep the
    # mask on device, including dummy runs and batches with no active tokens.
    rows = torch.arange(positions.shape[0], dtype=torch.int32, device=positions.device)
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


class CompressorStateCache(NativeCacheView, compressor_upstream.CompressorStateCache):
    """Upstream CircularBufferSpec, allocation lifecycle and cache binding."""

    def get_attn_backend(self):
        return AttentionBackend


def save_compress_norm(
    kv_score,
    positions,
    state_cache,
    slot_mapping,
    query_start_loc,
    token_to_req_indices,
    norm_weight,
    eps,
    ratio,
    latent,
):
    """Tensor implementation of the compressor_upstream fused compression operation.

    Read the previous row before publishing ring writes. For a long prefill,
    previous rows come directly from this chunk and only the final ring-sized
    suffix is stored, so writes never race aliases from earlier ring wraps.
    """
    count = kv_score.shape[0]
    if slot_mapping is not None:
        slot_mapping = slot_mapping[:count]
    if token_to_req_indices is not None:
        token_to_req_indices = token_to_req_indices[:count]
    dim = norm_weight.numel()
    values = kv_score[:, :dim].float()
    valid = slot_mapping >= 0
    if ratio == 2:
        capacity = state_cache.shape[1]
        cache = state_cache
        rows = torch.arange(
            kv_score.shape[0], dtype=torch.int32, device=kv_score.device
        )
        requests = token_to_req_indices.int().clamp(max=query_start_loc.numel() - 2)
        starts = query_start_loc.index_select(0, requests)
        ends = query_start_loc.index_select(0, requests + 1)
        previous_slots = (
            slot_mapping.clamp(min=0) // capacity * capacity
            + (positions - 1) % capacity
        )
        stored = cache.reshape(-1, *cache.shape[2:]).index_select(
            0, previous_slots.int()
        )
        previous = torch.where(
            (rows > starts)[:, None],
            kv_score.index_select(0, (rows - 1).clamp(min=0).int()).float(),
            stored,
        )
        scores = torch.stack((previous[:, dim:], kv_score[:, dim:].float()), 1)
        pair = torch.stack((previous[:, :dim], values), 1)
        values = (pair * scores.softmax(1)).sum(1)
        # All reads above refer to the old ring. Keep unique physical writes.
        publish = valid & (rows >= ends - capacity)
        write_cache(cache, slot_mapping, kv_score.float(), publish, after=stored)
        valid = valid & (positions % 2 == 1)
    elif ratio != 1:
        raise ValueError(f"V4.1 compression ratio must be 1 or 2, got {ratio}")
    result = rms_norm(values, norm_weight, eps)
    latent.copy_(result.masked_fill(~valid[:, None], 0).to(latent.dtype))


def _memo_stage_cache_key(obj: Any, kind: str, *parts: Any) -> tuple[Any, ...]:
    """Return a memoized ``(kind, *parts)`` stage-cache key tuple on ``obj``."""
    cache = obj.__dict__.get("_stage_key_memo")
    if cache is None:
        cache = {}
        obj.__dict__["_stage_key_memo"] = cache
    entry = cache.get(kind)
    if entry is not None and entry[0] == parts:
        return entry[1]
    key = (kind, *parts)
    cache[kind] = (parts, key)
    return key


class DeepseekCompressor(compressor_upstream.DeepseekCompressor):
    """Inherit weights and state ownership; consume the shared TPU metadata."""

    def __init__(self, *args, **kwargs):
        with device_hooks(
            compressor_upstream, CompressorStateCache=CompressorStateCache
        ):
            super().__init__(*args, **kwargs)

    def project(self, hidden_states):
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        layer = self.fused_wkv_wgate
        weight = layer.weight.float()
        if not getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            weight = weight.t()
        # Upstream accumulates the unquantized KV/score projection in FP32.
        # Keep that precision through pooling and normalization.
        return torch.mm(hidden_states.float(), weight)

    def forward(self, kv_score, positions):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        if self.state_cache is None:
            state_cache = query_start_loc = requests = None
            cache = self._static_forward_context[self.k_cache_prefix].kv_cache
            slots = _cache_slots(cache, metadata[self.k_cache_prefix], positions).int()
        else:
            state_metadata = metadata[self.state_cache.prefix]
            state_cache = self.state_cache.kv_cache
            stage_cache = _active_stage_attn_cache
            state_rows = (
                state_cache.shape[1]
                if isinstance(getattr(state_cache, "shape", None), tuple)
                else None
            )
            if (
                stage_cache is not None
                and state_rows is not None
                and state_metadata.seq_lens.numel() > 0
            ):
                table = _block_table(state_metadata)
                state_key = _memo_stage_cache_key(
                    self, "compressor_state_addr", table.shape, state_rows
                )
                cached_state = stage_cache.get(state_key)
                if cached_state is None:
                    query_start_loc = state_metadata.query_start_loc
                    requests = _token_requests(state_metadata, positions)
                    clamped_requests = requests.clamp(max=table.shape[0] - 1)
                    ring_rem = positions.int() % state_rows
                    token_rows = torch.arange(
                        positions.shape[0],
                        dtype=torch.int32,
                        device=positions.device,
                    )
                    valid_token = token_rows < query_start_loc[-1]
                    cached_state = (
                        query_start_loc,
                        requests,
                        clamped_requests,
                        ring_rem,
                        valid_token,
                    )
                    stage_cache[state_key] = cached_state
                else:
                    (
                        query_start_loc,
                        requests,
                        clamped_requests,
                        ring_rem,
                        valid_token,
                    ) = cached_state
                # Every compressed layer (L2/8/14/20) of a stage gathers the
                # same block-table column for the same tokens: share the slots
                # while the stage keeps the same block_tables tensor.
                slot_key = _memo_stage_cache_key(
                    self, "compressor_write_slots", table.shape, state_rows
                )
                cached_slots = stage_cache.get(slot_key)
                if (
                    cached_slots is not None
                    and cached_slots[0] is state_metadata.block_tables
                ):
                    slots = cached_slots[1]
                else:
                    blocks = table[:, 0].index_select(0, clamped_requests).int()
                    slots = (blocks * state_rows + ring_rem).masked_fill(
                        ~valid_token, -1
                    )
                    stage_cache[slot_key] = (state_metadata.block_tables, slots)
            else:
                slots = _cache_slots(
                    state_cache, state_metadata, positions, ring=True
                ).int()
                query_start_loc = state_metadata.query_start_loc
                requests = _token_requests(state_metadata, positions)
        latent = torch.empty(
            (positions.shape[0], self.head_dim),
            dtype=torch.bfloat16,
            device=positions.device,
        )
        save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slots,
            query_start_loc,
            requests,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
        )
        return latent


def indexer_uses_fp4(vllm_config):
    """Resolve the upstream format without imposing a CUDA capability check."""
    dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype != "fp8":
        raise ValueError(f"TPU V4.1 indexer requires fp8, got {dtype}")
    return False


class AttentionBackend(compressor_upstream.CompressorBackend):
    @staticmethod
    def get_name():
        return "TORCH_TPU_DEEPSEEK_V41"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, *args, **kwargs
    ):
        return num_blocks, num_kv_heads, block_size, head_size


class SWACache(NativeCacheView, DeepseekV4SWACache):
    def __init__(self, *args, **kwargs):
        kwargs["block_size"] = kwargs["cache_config"].block_size
        # Upstream ties bounded replay to its V2 model runner; the TPU model
        # implements the replay itself (see `prefix_replay_enabled`).
        if prefix_replay_enabled(get_current_vllm_config_or_none()):
            kwargs["bounded_replay"] = True
        # FP8 applies to the compressed main KV only; SWA rows stay BF16.
        kwargs["dtype"] = torch.bfloat16
        kwargs["backend_cls"] = AttentionBackend
        super().__init__(*args, **kwargs)


class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    def __init__(self, *args, **kwargs):
        # The first TPU path stores rounded values in BF16; ownership and
        # compression ratio still come from upstream. Packed storage follows
        # when the corresponding TPU kernel is connected.
        kwargs["head_dim"] = (
            get_current_vllm_config().model_config.hf_config.index_head_dim
        )
        kwargs["dtype"] = torch.bfloat16
        super().__init__(*args, **kwargs)

    def get_attn_backend(self):
        return AttentionBackend


def _value(output):
    return output[0] if isinstance(output, tuple) else output


def _insert(cache, slots, values, valid, *, records=None):
    write_cache(cache, slots, values, valid, records=records)


def _offset_records(cache, write_slots, write_invalid):
    """Pool rows from stage-shared ``_fast_swa_row_slots`` output for ``cache``."""
    offset = cache.row_offset
    records = write_slots if offset == 0 else offset + write_slots
    return records.masked_fill(write_invalid, -1)


def _gather_query_rows(rows):
    """All-gather per-rank query rows (dim 0), padded to 128 lanes."""
    width = rows.shape[-1]
    padding = (-width) % 128
    if padding:
        rows = F.pad(rows, (0, padding))
    rows = tensor_model_parallel_all_gather(rows.contiguous(), dim=0)
    return rows[:, :width].contiguous() if padding else rows


def _row_slots(cache, block_table, requests, logical):
    """Gather fixed-width token rows without inspecting device values on host."""
    safe = logical.int().clamp(min=0, max=block_table.shape[1] * cache.shape[1] - 1)
    # Gather each token's table row, then index within it 128 columns at a
    # time. A [tokens, width] elementwise 2-D gather is 16-40x slower on TPU.
    rows = block_table.index_select(0, requests)
    pages = (safe // cache.shape[1]).long()
    blocks = torch.cat(
        [
            rows.gather(1, pages[:, start : start + 128])
            for start in range(0, pages.shape[1], 128)
        ],
        dim=1,
    ).int()
    return blocks * cache.shape[1] + safe % cache.shape[1]


def _index_page_records(index_cache):
    """Records per page when page gathers are exact for `_request_row_slots`.

    `PagedCacheRows.row_indices` keeps each page's records contiguous; the
    page gather additionally needs every page to start a pool word row and
    to cover whole word rows (`gather_cache_page_values`). Returns 0 when a
    layout does not qualify (the per-record gather is then used).
    """
    if (
        not isinstance(index_cache, PagedCacheRows)
        or getattr(index_cache, "dtype", None) != torch.bfloat16
    ):
        return 0
    row_bytes = ROW_BYTES
    per_page = index_cache.shape[1]
    record_bytes = index_cache.row_bytes
    if row_bytes % record_bytes or (per_page * record_bytes) % row_bytes:
        return 0
    per_row = row_bytes // record_bytes
    page_words = per_page // per_row
    if (
        index_cache.row_offset % per_row
        or index_cache.page_stride % per_row
        or (
            getattr(index_cache, "raw_hi", None) is not None
            and index_cache.raw.shape[0] % page_words
        )
    ):
        return 0
    return per_page


def _query_split(count, ranks, *, kv_shard: bool = False):
    """``(part, padding)``: rows per TP rank and the static pad that covers them.

    ``part`` is ``ceil(count / ranks)`` rounded up to even, so never 1: a part
    of 1 row (``ceil`` at T=4/8 on 8 ranks) makes Dynamo 0/1-specialize and
    trace a second encoder/decoder graph for those buckets.
    ``ranks * part - count`` is at most ``2 * ranks - 1``, so
    ``padding = 2 * ranks`` rows always cover it.
    Under ``kv_shard``, when ``count >= ranks`` and ``count % ranks == 0``,
    each rank owns the exact contiguous slice ``part = count // ranks`` matching
    `slice_token_parallel_attention_inputs`.
    """
    if kv_shard and count >= ranks and count % ranks == 0:
        return count // ranks, ranks
    return (count + 2 * ranks - 1) // (2 * ranks) * 2, 2 * ranks


def _has_fast_cache_offset(cache) -> bool:
    return isinstance(getattr(cache, "row_offset", None), int) and not getattr(
        cache, "sharded", False
    )


def _fast_cache_offset_key(cache):
    return cache.page_stride


def _map_cache_blocks(cache, blocks):
    if not getattr(cache, "sharded", False):
        return blocks * cache.page_stride, None
    rep = cache.replicated_blocks
    sharded_blocks = cache.shape[0] - rep
    sharded_local = blocks - cache.block_offset
    in_rep = (blocks >= 0) & (blocks < rep)
    in_sharded = (sharded_local >= 0) & (sharded_local < sharded_blocks)
    owned = (blocks >= 0) & (in_rep | in_sharded)
    local_blocks = torch.where(in_rep, blocks, rep + sharded_local).clamp(
        min=0, max=max(cache.shape[0] - 1, 0)
    )
    return local_blocks * cache.page_stride, ~owned


def _fast_swa_row_slots(swa_cache, blocks, rem, invalid_window):
    mapped_blocks, swa_unowned = _map_cache_blocks(swa_cache, blocks)
    return (
        (mapped_blocks + rem).int(),
        invalid_window if swa_unowned is None else invalid_window | swa_unowned,
    )


def _request_row_slots(cache, block_table, width, page_records: int = 0):
    """Slots of logical rows ``0..width-1`` for every request: ``[requests, width]``.

    Every token of a request scans the same capacity-wide rows, so address
    them once per request instead of once per token. When ``page_records``
    matches ``cache.shape[1]`` (> 1), returns one first-record slot per page
    (``[requests, num_pages]``).
    """
    rows_per_page = cache.shape[1]
    stride = cache.page_stride if _has_fast_cache_offset(cache) else rows_per_page
    if page_records > 1 and rows_per_page == page_records:
        num_pages = (width + rows_per_page - 1) // rows_per_page
        table = block_table[:, :num_pages]
        if num_pages > table.shape[1]:
            table = torch.cat(
                (
                    table,
                    table[:, -1:].expand(-1, num_pages - table.shape[1]),
                ),
                dim=1,
            )
        return table.int() * stride
    page_bases = block_table.int() * stride
    # Logical row r lives in table column r // rows_per_page, so repeating each
    # column rows_per_page times lines blocks up with rows 0..width-1 without
    # a full-width gather (16-40x slower on TPU, see _row_slots).
    bases = (
        page_bases[:, :, None]
        .expand(-1, -1, rows_per_page)
        .reshape(block_table.shape[0], -1)
    )
    offsets = (
        torch.arange(width, dtype=bases.dtype, device=block_table.device).clamp(
            max=bases.shape[1] - 1
        )
        % rows_per_page
    )
    if width > bases.shape[1]:
        # Match _row_slots' clamp to the table's last row.
        bases = torch.cat(
            (bases, bases[:, -1:].expand(-1, width - bases.shape[1])), dim=1
        )
    return bases[:, :width] + offsets


def _read_rows(cache, block_table, requests, logical):
    slots = _row_slots(cache, block_table, requests, logical)
    return cache[slots // cache.shape[1], slots % cache.shape[1]]


def _sp_gather_inputs(
    qr, kv, kv_score, indexer_weights, *, local_q: bool, num_tokens: int
):
    """All-gather this rank's SP token shard of the input projections.

    One collective per layer. ``qr`` (``q_lora_rank``), ``kv``
    (``head_dim``) and ``indexer_weights`` (``indexer.n_head``) share the
    activation dtype and are concatenated along the feature axis.
    ``kv_score`` is the fp32 ``[T_local, head_dim + 1]`` compressor
    projection; the collective only moves bytes, so it rides along as
    ``2 * (head_dim + 1)`` activation-dtype columns through a dtype
    bit-view and is viewed back to fp32 after the gather. That is
    bit-exact and saves the second all-gather launch (and its barrier)
    on every compressed-KV source layer of every prefill chunk.

    Non-source layers with local_q do not consume full-token ``qr`` on
    the host, so they gather only ``kv`` (512 cols instead of 2048).
    """
    gather_qr = indexer_weights is not None or not local_q
    parts = []
    if gather_qr:
        parts.append(qr)
    parts.append(kv)
    if indexer_weights is not None:
        parts.append(indexer_weights)
    if kv_score is not None:
        parts.append(kv_score.contiguous().view(kv.dtype))
    assert all(part.dtype == kv.dtype for part in parts)
    payload = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    gathered = sp_all_gather(payload)[:num_tokens]
    pieces = iter(
        piece.contiguous()
        for piece in gathered.split([part.shape[-1] for part in parts], dim=-1)
    )
    if gather_qr:
        qr = next(pieces)
    kv = next(pieces)
    if indexer_weights is not None:
        indexer_weights = next(pieces)
    if kv_score is not None:
        kv_score = next(pieces).view(kv_score.dtype)
    return qr, kv, kv_score, indexer_weights


class DeepseekV41Attention(NativeCacheView, upstream.DeepseekV4Attention):
    backend_cls = AttentionBackend
    swa_backend_cls = AttentionBackend
    use_fp8_ds_mla_layout = False
    _ced_decoder = False
    _ced_global_from_encoder = False
    _sp_sharded_input = False
    # True once `_init_local_q_weights` built the gathered weights.
    _local_q_attn = False
    # True when that path takes the TP-replicated full ``qr`` and slices each
    # rank's token shard itself (`TPU_DEEPSEEK_V41_DECODE_LOCAL_Q`).
    _replicated_local_q = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def __init__(self, vllm_config, *args, **kwargs):
        with (
            device_hooks(
                upstream,
                DeepseekCompressor=DeepseekCompressor,
                dsa_indexer_uses_fp4=indexer_uses_fp4,
                DeepseekV4IndexerCache=IndexerCache,
                DeepseekV4SWACache=SWACache,
            ),
            device_hooks(torch.cuda, Event=lambda: None),
        ):
            super().__init__(vllm_config, *args, **kwargs)
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "TPU V4.1 supports --kv-cache-dtype auto, bfloat16 or fp8"
            )
        # FP8 main KV uses DSV4's packed record (448 e4m3 + 64 e8m0 bytes)
        # plus a BF16 RoPE companion; see cache.encode_fp8_kv.
        self._fp8_kv = self.kv_cache_torch_dtype == torch.float8_e4m3fn
        self.register_buffer(
            "_tpu_rope_cache", self.rotary_emb.cos_sin_cache, persistent=False
        )
        # The TP rank as data, not a Python constant: every rank must compile
        # the same program, or XLA may order its collectives differently per
        # rank and deadlock (see `_select_paged_indices`).
        self.register_buffer(
            "_tp_rank",
            torch.tensor(
                get_tensor_model_parallel_rank(),
                dtype=torch.int32,
                device=self.rotary_emb.cos_sin_cache.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_tp_rank_1d",
            self._tp_rank.reshape(1),
            persistent=False,
        )
        tp_size = get_tensor_model_parallel_world_size()
        self.register_buffer(
            "_full_attn_sink",
            self.attn_sink.detach().float().repeat(tp_size).contiguous(),
            persistent=False,
        )
        self._config = vllm_config.model_config.hf_config.to_dict()
        self._max_model_len = vllm_config.model_config.max_model_len
        # Branch-free SWA patch (`_attend_window`) only while its static
        # capacity, 63 tokens per extra request, stays small. Measured
        # (tpu7x-2): max_num_seqs 1 +4.6%, 4 +1.6%, 8 -4.3% prefill.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._swa_max_requests = (
            None
            if envs.TPU_DEEPSEEK_V41_KV_SHARD
            else (max_num_seqs if max_num_seqs <= 4 else None)
        )
        self._precompact_records = self._fp8_kv
        self._token_parallel_attn = self._fp8_kv
        self._indexer_weight_scale = (
            (self.indexer.head_dim * self.indexer.n_head) ** -0.5
            if getattr(self, "indexer", None) is not None
            else 1.0
        )
        object.__setattr__(self, "mla_attn", self)
        self._prepare_and_attn_fn = self._prepare_tpu

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None or not self._fp8_kv:
            return spec
        return dataclasses.replace(spec, dtype=torch.uint8, page_size_padded=None)

    # Mirrors upstream vllm/models/deepseek_v41/attention.py:561-596
    # (DeepseekV41Attention.forward) with SP-sharded input projection support
    # and SymInt-safe o_padded allocation.
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = positions.shape[0]
        # Allocate o_padded from positions via zeros_like + expand so TorchTPU's
        # lowering does not call torch.empty on a compound SymInt expression.
        o_padded = (
            torch.zeros_like(
                positions[:, None, None],
                dtype=hidden_states.dtype,
            )
            .expand(-1, self.padded_heads, self.head_dim)
            .contiguous()
        )
        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(
            hidden_states
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        # This rank's SP token shard of qr, before the gather; or, for the
        # replicated decode path, the full qr the op slices per rank.
        swa_kv = getattr(getattr(self, "swa_cache_layer", None), "kv_cache", None)
        qr_local = (
            qr
            if (self._sp_sharded_input or self._replicated_local_q)
            and getattr(self, "_local_q_attn", False)
            and (
                swa_kv is None
                or (isinstance(swa_kv, PagedCacheRows) and self.window_size % 64 == 0)
            )
            else None
        )
        if self._sp_sharded_input:
            qr, kv, kv_score, indexer_weights = _sp_gather_inputs(
                qr,
                kv,
                kv_score,
                indexer_weights,
                local_q=qr_local is not None,
                num_tokens=num_tokens,
            )
        args = (
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        if qr_local is not None:
            projected = self._prepare_and_attn_fn(*args, qr_local=qr_local)
            if projected is not None:
                # The fused q/attention/wo_a op already produced wo_b's input.
                return _value(self.wo_b(projected))
        else:
            self._prepare_and_attn_fn(*args)
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _run_parallel_input_projections(self, hidden_states):
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = (
            self.compressor.project(hidden_states)
            if self.compressor is not None and not self._ced_global_from_encoder
            else None
        )
        index_weights = (
            _value(self.indexer.weights_proj(hidden_states))
            if self.indexer is not None and not self._all_indices_fit
            else None
        )
        return qr_kv, kv_score, index_weights

    @property
    def _all_indices_fit(self):
        return bool(self.compress_ratio) and (
            self._max_model_len // self.compress_ratio <= self._config["index_topk"]
        )

    def _quantize_indexer(self, values):
        # Values are kept dequantized in BF16 storage, not silently changed to
        # another quantizer. Power-of-two FP8 scales remain with Q/K, so weights
        # must NOT fold the Q scale a second time.
        return round_fp8_activation(values, self.indexer.head_dim)

    def _project_index_q(self, qr, positions):
        index_q = _value(self.indexer.wq_b(qr)).reshape(
            -1, self.indexer.n_head, self.indexer.head_dim
        )
        return self._quantize_indexer(self._rotate(index_q, positions))

    def _split_qkv_and_norm(self, qr_kv):
        return (
            self.q_norm(qr_kv[:, : self.q_lora_rank]),
            None,
            self.kv_norm(qr_kv[:, self.q_lora_rank :]),
        )

    @property
    def kv_rope_op(self):
        op = self.__dict__.get("_kv_rope_op_instance")
        if op is not None:
            return op
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        return cast(Callable, VllmDeepseekV4MLAAttention.kv_rope_op.fget)(self)

    def _stage_cache_key(self, kind: str, *parts: Any) -> tuple[Any, ...]:
        return _memo_stage_cache_key(self, kind, *parts)

    def _get_stage_tp_rank(self, stage_cache: dict[Any, Any] | None = None):
        rank = self._tp_rank
        if stage_cache is not None and isinstance(rank, torch.Tensor):
            return stage_cache.setdefault(
                self._stage_cache_key("stage_tp_rank", rank.device, rank.dtype), rank
            )
        return rank

    def _get_stage_tp_rank_1d(self, stage_cache: dict[Any, Any] | None = None):
        rank_1d = self._tp_rank_1d
        if stage_cache is not None and isinstance(rank_1d, torch.Tensor):
            key = self._stage_cache_key("stage_tp_rank_1d", rank_1d.device)
            cached = stage_cache.get(key)
            if cached is None:
                cached = (
                    rank_1d
                    if rank_1d.shape == (1,) and rank_1d.dtype == torch.int32
                    else rank_1d.reshape(1).int()
                )
                stage_cache[key] = cached
            return cached
        return rank_1d

    def _get_stage_rope_cache(self, stage_cache: dict[Any, Any] | None = None):
        rope_cache = self._tpu_rope_cache
        if stage_cache is not None and isinstance(rope_cache, torch.Tensor):
            # Same shape/dtype does not mean same values: compressed layers use
            # compress_rope_theta (upstream deepseek_v41/common/rope.py), so the
            # key must carry the rope config or layers read the wrong cos/sin.
            return stage_cache.setdefault(
                self._stage_cache_key(
                    "stage_tpu_rope_cache",
                    rope_cache.device,
                    rope_cache.shape,
                    rope_cache.dtype,
                    bool(self.compress_ratio),
                    getattr(getattr(self, "rotary_emb", None), "base", None),
                ),
                rope_cache,
            )
        return rope_cache

    def _rotate(self, values, positions, inverse=False):
        if values.device.type != "cpu" and not inverse:
            return self.kv_rope_op(
                values, positions, self._get_stage_rope_cache(_active_stage_attn_cache)
            )
        return rotary(
            values.unsqueeze(0),
            positions,
            self._config,
            compressed=bool(self.compress_ratio),
            inverse=inverse,
        ).squeeze(0)

    def _write_global_kv(self, kv_score, positions, *, defer_index_write=True):
        metadata = get_forward_context().attn_metadata
        latent = self.compressor(kv_score, positions)
        indexer = self.indexer
        index_cache = indexer.k_cache.kv_cache
        kv_meta = metadata[self.prefix]
        idx_meta = metadata[indexer.k_cache.prefix]
        stage_cache = _active_stage_attn_cache
        kv_rows = (
            self.kv_cache.shape[1]
            if isinstance(getattr(self.kv_cache, "shape", None), tuple)
            else None
        )
        idx_rows = (
            index_cache.shape[1]
            if isinstance(getattr(index_cache, "shape", None), tuple)
            else None
        )
        idx_ratio = indexer.k_cache.compress_ratio
        if (
            stage_cache is not None
            and kv_rows is not None
            and idx_rows is not None
            and kv_meta.seq_lens.numel() > 0
            and idx_meta.seq_lens.numel() > 0
        ):
            kv_table = _block_table(kv_meta)
            idx_table = _block_table(idx_meta)
            slot_key = self._stage_cache_key(
                "global_kv_addr",
                kv_table.shape,
                self.compress_ratio,
                kv_rows,
                idx_table.shape,
                idx_ratio,
                idx_rows,
            )
            cached_addr = stage_cache.get(slot_key)
            if cached_addr is None:
                pos_i32 = positions.int()
                requests = _token_requests(kv_meta, positions).clamp(
                    max=kv_table.shape[0] - 1
                )
                logical = pos_i32 // self.compress_ratio
                kv_col = (logical // kv_rows).clamp(max=kv_table.shape[1] - 1)
                kv_idx = requests * kv_table.shape[1] + kv_col
                kv_rem = logical % kv_rows
                if (
                    idx_ratio == self.compress_ratio
                    and idx_rows == kv_rows
                    and idx_table.shape == kv_table.shape
                ):
                    idx_idx = kv_idx
                    idx_rem = kv_rem
                else:
                    idx_logical = pos_i32 // idx_ratio
                    idx_col = (idx_logical // idx_rows).clamp(
                        max=idx_table.shape[1] - 1
                    )
                    idx_idx = requests * idx_table.shape[1] + idx_col
                    idx_rem = idx_logical % idx_rows
                token_rows = torch.arange(
                    positions.shape[0], dtype=torch.int32, device=positions.device
                )
                valid_token = token_rows < kv_meta.query_start_loc[-1]
                valid_group = valid_token & ((pos_i32 + 1) % self.compress_ratio == 0)
                group_positions = logical * self.compress_ratio
                cached_addr = (
                    kv_idx,
                    kv_rem,
                    idx_idx,
                    idx_rem,
                    valid_token,
                    valid_group,
                    group_positions,
                )
                stage_cache[slot_key] = cached_addr
            else:
                (
                    kv_idx,
                    kv_rem,
                    idx_idx,
                    idx_rem,
                    valid_token,
                    valid_group,
                    group_positions,
                ) = cached_addr
            # L2/8/14/20 all write the same token rows through the same block
            # tables: compute their addresses (and the deferred index-cache
            # write rows) once per stage while the block_tables are unchanged.
            fast_idx = _has_fast_cache_offset(index_cache)
            fast_kv = _has_fast_cache_offset(self.kv_cache)
            write_key = self._stage_cache_key(
                "global_kv_write_slots",
                slot_key,
                _fast_cache_offset_key(index_cache) if fast_idx else None,
                _fast_cache_offset_key(self.kv_cache) if fast_kv else None,
            )
            kv_tables = kv_meta.block_tables
            idx_tables = idx_meta.block_tables
            cached_write = stage_cache.get(write_key)
            if (
                cached_write is not None
                and cached_write[0] is kv_tables
                and cached_write[1] is idx_tables
            ):
                (
                    _,
                    _,
                    compressed_slots,
                    valid,
                    index_slots,
                    idx_write_slots,
                    idx_write_invalid,
                    kv_write_slots,
                    kv_write_invalid,
                ) = cached_write
            else:
                kv_blocks = kv_table.reshape(-1).index_select(0, kv_idx).int()
                compressed_slots = (kv_blocks * kv_rows + kv_rem).masked_fill(
                    ~valid_token, -1
                )
                valid = (compressed_slots >= 0) & valid_group
                if (
                    idx_idx is kv_idx
                    and idx_tables is kv_tables
                    and idx_rows == kv_rows
                ):
                    # Same table, column and remainder: the index cache's slots
                    # are the compressed KV slots (no second 2-D gather).
                    idx_blocks = kv_blocks
                    index_slots = compressed_slots
                else:
                    idx_blocks = idx_table.reshape(-1).index_select(0, idx_idx).int()
                    index_slots = (idx_blocks * idx_rows + idx_rem).masked_fill(
                        ~valid_token, -1
                    )
                idx_write_slots = idx_write_invalid = None
                if fast_idx:
                    # index_cache.row_indices without the per-layer divmod.
                    idx_write_slots, idx_write_invalid = _fast_swa_row_slots(
                        index_cache, idx_blocks, idx_rem, ~valid | (index_slots < 0)
                    )
                kv_write_slots = kv_write_invalid = None
                if fast_kv:
                    # The kv_cache write rows, shared by L2/8/14/20 (each adds
                    # its own row_offset) instead of a per-layer row_indices.
                    kv_write_slots, kv_write_invalid = _fast_swa_row_slots(
                        self.kv_cache,
                        kv_blocks,
                        kv_rem,
                        ~valid | (compressed_slots < 0),
                    )
                stage_cache[write_key] = (
                    kv_tables,
                    idx_tables,
                    compressed_slots,
                    valid,
                    index_slots,
                    idx_write_slots,
                    idx_write_invalid,
                    kv_write_slots,
                    kv_write_invalid,
                )
        else:
            pos_i32 = positions.int()
            compressed_slots = _cache_slots(
                self.kv_cache,
                kv_meta,
                positions,
                compress_ratio=self.compress_ratio,
            ).int()
            valid = (compressed_slots >= 0) & ((pos_i32 + 1) % self.compress_ratio == 0)
            group_positions = pos_i32 // self.compress_ratio * self.compress_ratio
            index_slots = _cache_slots(
                index_cache,
                idx_meta,
                positions,
                compress_ratio=idx_ratio,
            ).int()
            idx_write_slots = idx_write_invalid = None
            kv_write_slots = kv_write_invalid = None
        starts = _window_starts()
        if starts is not None:
            # Shared prefix rows must survive replay unchanged in both paths.
            valid = valid & ~_replayed(kv_meta, positions, starts, self.window_size)
            if idx_write_invalid is not None:
                idx_write_invalid = idx_write_invalid | ~valid
            if kv_write_invalid is not None:
                kv_write_invalid = kv_write_invalid | ~valid
        rotated = self._rotate(latent, group_positions)
        if kv_write_slots is not None and kv_write_invalid is not None:
            _insert(
                self.kv_cache,
                compressed_slots,
                rotated,
                valid,
                records=_offset_records(
                    self.kv_cache, kv_write_slots, kv_write_invalid
                ),
            )
        else:
            _insert(self.kv_cache, compressed_slots, rotated, valid)
        keys = indexer.k_norm(_value(indexer.wk(latent)))
        keys = self._quantize_indexer(self._rotate(keys, group_positions))
        if (
            defer_index_write
            and isinstance(index_cache, PagedCacheRows)
            and self.window_size % 64 == 0
            and not self._all_indices_fit
        ):
            if idx_write_slots is not None and idx_write_invalid is not None:
                return (
                    _offset_records(index_cache, idx_write_slots, idx_write_invalid),
                    keys,
                )
            records = index_cache.row_indices(index_slots.clamp(min=0))
            return records.masked_fill(~valid | (index_slots < 0), -1), keys
        _insert(index_cache, index_slots, keys, valid)
        return None

    def write_encoder_global_kv(
        self, hidden_states, positions, *, sp_sharded: bool = False
    ):
        """Publish decoder global KV for all encoder tokens before replay."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not self.kv_cache.numel():
            return
        kv_score = self.compressor.project(hidden_states)
        if sp_sharded:
            kv_score = sp_all_gather(kv_score)[: positions.shape[0]]
        self._write_global_kv(kv_score, positions, defer_index_write=False)

    def _prepare_tpu(
        self,
        hidden_states,
        qr,
        kv,
        qr_scale,
        kv_score,
        index_weights,
        positions,
        output,
        qr_local=None,
    ):
        """Attend into ``output``, or return the ``wo_b`` input when the fused
        q/attention/``wo_a`` op (``qr_local``) ran; else ``None``."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not self.swa_cache_layer.kv_cache.numel():
            output.zero_()
            return None
        swa = metadata[self.swa_cache_layer.prefix]
        swa_cache = self.swa_cache_layer.kv_cache
        swa_table = _block_table(swa)
        tiled = isinstance(swa_cache, PagedCacheRows) and self.window_size % 64 == 0
        has_fast_swa_offset = _has_fast_cache_offset(swa_cache)
        stage_cache = _active_stage_attn_cache
        rows_per_page = (
            swa_cache.shape[1]
            if isinstance(getattr(swa_cache, "shape", None), tuple)
            else None
        )
        non_causal = bool(getattr(self, "non_causal_multi_token_decode", False))
        swa_width = (
            -(-(self.window_size + int(getattr(self, "dspark_block_size", 5))) // 64)
            * 64
            if non_causal
            else self.window_size
        )
        swa_row_slots_masked = None
        swa_slots_key = None
        if (
            stage_cache is not None
            and rows_per_page is not None
            and swa.seq_lens.numel() > 0
        ):
            use_3page_swa = (
                tiled and swa_table.shape[1] > 0 and swa_width <= 6 * rows_per_page
            )
            swa_key = self._stage_cache_key(
                "swa_token_addr",
                swa_table.shape,
                rows_per_page,
                _fast_cache_offset_key(swa_cache) if has_fast_swa_offset else None,
                self.window_size,
                self._ced_decoder,
                use_3page_swa,
                non_causal,
                swa_width,
                tiled,
            )
            cached_swa = stage_cache.get(swa_key)
            if cached_swa is None:
                token_rows = torch.arange(
                    positions.shape[0], dtype=torch.int32, device=positions.device
                )
                valid_token = token_rows < swa.query_start_loc[-1]
                requests = _token_requests(swa, positions).clamp(
                    max=swa_table.shape[0] - 1
                )
                pos_col = (
                    (positions // rows_per_page).clamp(max=swa_table.shape[1] - 1).int()
                )
                pos_rem = (positions % rows_per_page).int()
                if non_causal:
                    req_idx = requests.clamp(max=swa.seq_lens.shape[0] - 1)
                    end_pos = swa.seq_lens.index_select(0, req_idx)
                    query_len = swa.query_start_loc.index_select(
                        0, req_idx + 1
                    ) - swa.query_start_loc.index_select(0, req_idx)
                    prefix_len = (end_pos - query_len).clamp(min=0)
                    start_pos = (prefix_len - self.window_size).clamp(min=0).int()
                    logical = (
                        start_pos[:, None]
                        + torch.arange(
                            swa_width, dtype=torch.int32, device=positions.device
                        )[None, :]
                    )
                    base_valid_window = (
                        (logical >= 0)
                        & (logical < end_pos[:, None])
                        & valid_token[:, None]
                    )
                else:
                    logical = (
                        positions.int()[:, None]
                        - self.window_size
                        + 1
                        + torch.arange(
                            self.window_size,
                            dtype=torch.int32,
                            device=positions.device,
                        )[None, :]
                    )
                    base_valid_window = (logical >= 0) & valid_token[:, None]
                if self._ced_decoder:
                    replay_starts = get_forward_context().additional_kwargs[
                        "deepseek_v41_replay_starts"
                    ]
                    base_valid_window = base_valid_window & (
                        logical >= replay_starts.index_select(0, requests)[:, None]
                    )
                elif (win_starts := _window_starts()) is not None:
                    base_valid_window = base_valid_window & (
                        logical
                        >= win_starts.index_select(
                            0, requests.clamp(max=win_starts.shape[0] - 1)
                        )[:, None]
                    )
                if use_3page_swa:
                    safe = logical.clamp(
                        min=0, max=swa_table.shape[1] * rows_per_page - 1
                    )
                    swa_pages = (safe // rows_per_page).int()
                    swa_page_rem = safe % rows_per_page
                    max_prev_pages = max(2, (swa_width - 1) // rows_per_page + 1)
                    c_last = swa_pages[:, -1]
                    prev_cols = tuple(
                        (c_last - k).clamp(min=0) for k in range(1, max_prev_pages + 1)
                    )
                    is_c_last = swa_pages == c_last[:, None]
                    is_prev_cols = tuple(
                        swa_pages == col[:, None] for col in prev_cols[:-1]
                    )
                    swa_3page = (
                        c_last,
                        prev_cols,
                        is_c_last,
                        is_prev_cols,
                        swa_page_rem,
                    )
                else:
                    swa_3page = None
                cached_swa = (
                    valid_token,
                    requests,
                    pos_col,
                    pos_rem,
                    logical,
                    base_valid_window,
                    swa_3page,
                )
                stage_cache[swa_key] = cached_swa
            else:
                (
                    valid_token,
                    requests,
                    pos_col,
                    pos_rem,
                    logical,
                    base_valid_window,
                    swa_3page,
                ) = cached_swa
            swa_slots_key = self._stage_cache_key("swa_token_slots", *swa_key[1:])
            cached_swa_slots = stage_cache.get(swa_slots_key)
            if cached_swa_slots is None or cached_swa_slots[0] is not swa.block_tables:
                swa_table_flat, req_row_base = (
                    swa_table.reshape(-1),
                    requests.int() * swa_table.shape[1],
                )
                b2 = swa_table_flat.index_select(0, req_row_base + pos_col).int()
                swa_slots = (b2 * rows_per_page + pos_rem).masked_fill(~valid_token, -1)
                swa_valid = swa_slots >= 0
                swa_valid_int32 = swa_valid.int()
                swa_invalid_3d = (~swa_valid)[:, None, None]
                valid_window = base_valid_window & swa_valid[:, None]
                invalid_window = ~valid_window
                if swa_3page is not None:
                    (
                        c_last,
                        prev_cols,
                        is_c_last,
                        is_prev_cols,
                        swa_page_rem,
                    ) = swa_3page
                    b_last = (
                        swa_table_flat.index_select(0, req_row_base + c_last).int()
                        if non_causal
                        else b2
                    )
                    swa_blocks = swa_table_flat.index_select(
                        0, req_row_base + prev_cols[-1]
                    ).int()[:, None]
                    for col, is_col in zip(
                        reversed(prev_cols[:-1]),
                        reversed(is_prev_cols),
                        strict=True,
                    ):
                        b_col = swa_table_flat.index_select(
                            0, req_row_base + col
                        ).int()[:, None]
                        swa_blocks = torch.where(is_col, b_col, swa_blocks)
                    swa_blocks = torch.where(is_c_last, b_last[:, None], swa_blocks)
                    swa_row_slots, invalid_window = (
                        _fast_swa_row_slots(
                            swa_cache, swa_blocks, swa_page_rem, invalid_window
                        )
                        if has_fast_swa_offset
                        else (swa_blocks * rows_per_page + swa_page_rem, invalid_window)
                    )
                else:
                    raw_row_slots = (
                        _row_slots(swa_cache, swa_table, requests, logical)
                        if tiled
                        else None
                    )
                    swa_row_slots, invalid_window = (
                        _fast_swa_row_slots(
                            swa_cache,
                            raw_row_slots // rows_per_page,
                            raw_row_slots % rows_per_page,
                            invalid_window,
                        )
                        if raw_row_slots is not None and has_fast_swa_offset
                        else (raw_row_slots, invalid_window)
                    )
                swa_row_slots_masked = (
                    swa_row_slots.masked_fill(invalid_window, -1)
                    if swa_row_slots is not None and has_fast_swa_offset
                    else None
                )
                # The token write rows (b2, pos_rem) shared by every SWA layer;
                # each layer adds its row_offset instead of re-deriving the
                # block and remainder from swa_slots in PagedCacheRows.write.
                swa_write = (
                    _fast_swa_row_slots(swa_cache, b2, pos_rem, ~swa_valid)
                    if has_fast_swa_offset
                    else None
                )
                stage_cache[swa_slots_key] = (
                    swa.block_tables,
                    swa_slots,
                    swa_valid,
                    swa_valid_int32,
                    swa_invalid_3d,
                    valid_window,
                    invalid_window,
                    swa_write,
                    swa_row_slots,
                    swa_row_slots_masked,
                )
            else:
                (
                    _,
                    swa_slots,
                    swa_valid,
                    swa_valid_int32,
                    swa_invalid_3d,
                    valid_window,
                    invalid_window,
                    swa_write,
                    swa_row_slots,
                    swa_row_slots_masked,
                ) = cached_swa_slots
        else:
            swa_write = None
            swa_slots = _cache_slots(swa_cache, swa, positions).int()
            swa_valid = swa_slots >= 0
            swa_valid_int32 = swa_valid
            swa_invalid_3d = (~swa_valid)[:, None, None]
            requests = _token_requests(swa, positions).clamp(max=swa_table.shape[0] - 1)
            if non_causal and swa.seq_lens.numel() > 0:
                req_idx = requests.clamp(max=swa.seq_lens.shape[0] - 1)
                end_pos = swa.seq_lens.index_select(0, req_idx)
                query_len = swa.query_start_loc.index_select(
                    0, req_idx + 1
                ) - swa.query_start_loc.index_select(0, req_idx)
                prefix_len = (end_pos - query_len).clamp(min=0)
                start_pos = (prefix_len - self.window_size).clamp(min=0).int()
                logical = (
                    start_pos[:, None]
                    + torch.arange(
                        swa_width, dtype=torch.int32, device=positions.device
                    )[None, :]
                )
                valid_window = (
                    (logical >= 0) & (logical < end_pos[:, None]) & swa_valid[:, None]
                )
            else:
                logical = (
                    positions.int()[:, None]
                    - self.window_size
                    + 1
                    + torch.arange(
                        self.window_size, dtype=torch.int32, device=positions.device
                    )[None, :]
                )
                valid_window = (logical >= 0) & swa_valid[:, None]
            if self._ced_decoder:
                replay_starts = get_forward_context().additional_kwargs[
                    "deepseek_v41_replay_starts"
                ]
                valid_window = valid_window & (
                    logical >= replay_starts.index_select(0, requests)[:, None]
                )
            elif (win_starts := _window_starts()) is not None:
                valid_window = valid_window & (
                    logical
                    >= win_starts.index_select(
                        0, requests.clamp(max=win_starts.shape[0] - 1)
                    )[:, None]
                )
            invalid_window = ~valid_window
            raw_row_slots = (
                _row_slots(swa_cache, swa_table, requests, logical) if tiled else None
            )
            swa_row_slots, invalid_window = (
                _fast_swa_row_slots(
                    swa_cache,
                    raw_row_slots // rows_per_page,
                    raw_row_slots % rows_per_page,
                    invalid_window,
                )
                if raw_row_slots is not None
                and has_fast_swa_offset
                and rows_per_page is not None
                else (raw_row_slots, invalid_window)
            )
        if swa_row_slots is None:
            records = None
        elif has_fast_swa_offset and rows_per_page is not None:
            if swa_cache.row_offset == 0 and swa_row_slots_masked is not None:
                records = swa_row_slots_masked
            elif stage_cache is not None and swa_slots_key is not None:
                swa_rec_key = self._stage_cache_key(
                    "swa_records", swa_cache.row_offset, *swa_slots_key[1:]
                )
                cached_swa_rec = stage_cache.get(swa_rec_key)
                if cached_swa_rec is None or cached_swa_rec[0] is not swa_row_slots:
                    records = (swa_cache.row_offset + swa_row_slots).masked_fill(
                        invalid_window, -1
                    )
                    stage_cache[swa_rec_key] = (swa_row_slots, records)
                else:
                    records = cached_swa_rec[1]
            else:
                records = (
                    swa_row_slots
                    if swa_cache.row_offset == 0
                    else swa_cache.row_offset + swa_row_slots
                ).masked_fill(invalid_window, -1)
        else:
            records = swa_cache.row_indices(swa_row_slots).masked_fill(
                invalid_window, -1
            )
        # Token-local q path: q projection, attention and
        # wo_a run in one op that picks token-parallel vs head-sharded on
        # concrete shapes.
        fused_qo = (
            qr_local is not None
            and self._local_q_attn
            and tiled
            and self._fp8_kv
            and bool(self.compress_ratio)
        )
        q = None
        if not fused_qo:
            q = _value(self.wq_b(qr)).reshape(-1, self.n_local_heads, self.head_dim)
            # V4.1 normalizes the low-rank qr before wq_b, but does not
            # normalize the projected query heads. Upstream's fused insert
            # passes apply_q_norm=False; V4's qnorm_rope kernel changes
            # attention scores.
            q = self._rotate(q, positions)
        # Match upstream's plain BF16 cache selected by this backend. Packed
        # FP8/NVFP4 cache numerics must not be imposed on BF16 cache rows.
        kv = self._rotate(kv, positions)
        if swa_write is not None:
            _insert(
                swa_cache,
                swa_slots,
                kv,
                swa_valid,
                records=_offset_records(swa_cache, *swa_write),
            )
        else:
            _insert(swa_cache, swa_slots, kv, swa_valid)
        weights = None
        pending_index_write = None
        if self.compressor is not None and not self._ced_global_from_encoder:
            pending_index_write = self._write_global_kv(kv_score, positions)
        if self.indexer is not None and not self._all_indices_fit:
            weights = index_weights.float() * getattr(
                self,
                "_indexer_weight_scale",
                (self.indexer.head_dim * self.indexer.n_head) ** -0.5,
            )
        count = positions.shape[0]
        window = None if tiled else _read_rows(swa_cache, swa_table, requests, logical)
        rows = window
        valid_rows = valid_window
        main_cache = swa_cache
        main_records = (
            None
            if self.compress_ratio
            else positions.new_empty((count, 0), dtype=torch.int32)
        )
        pages = starts = None
        swa_is_sharded = bool(
            getattr(swa_cache, "sharded", False)
            and getattr(swa_cache, "replicated_blocks", 0) == 0
        )
        use_owner_layout = bool(self.compress_ratio or swa_is_sharded)
        owner_layout = (
            getattr(swa, "owner_token_layout", None) if use_owner_layout else None
        )
        owner_perm = (
            getattr(swa, "owner_perm", getattr(owner_layout, "perm", None))
            if use_owner_layout
            else None
        )
        owner_inv = (
            getattr(swa, "owner_inv", getattr(owner_layout, "inv", None))
            if use_owner_layout
            else None
        )
        owner_q_loc = (
            getattr(swa, "owner_q_loc", getattr(owner_layout, "query_start_loc", None))
            if use_owner_layout
            else None
        )
        segmented_main = False
        if self.compress_ratio:
            source_meta = metadata[self.compressed_cache_prefix]
            src_layout = owner_layout or getattr(
                source_meta, "owner_token_layout", None
            )
            owner_perm = (
                owner_perm
                if owner_perm is not None
                else getattr(
                    source_meta, "owner_perm", getattr(src_layout, "perm", None)
                )
            )
            owner_inv = (
                owner_inv
                if owner_inv is not None
                else getattr(source_meta, "owner_inv", getattr(src_layout, "inv", None))
            )
            owner_q_loc = (
                owner_q_loc
                if owner_q_loc is not None
                else getattr(
                    source_meta,
                    "owner_q_loc",
                    getattr(src_layout, "query_start_loc", None),
                )
            )
            comp_key = self._stage_cache_key(
                "compressed_attn",
                self.compressed_cache_prefix,
                getattr(self, "index_source_layer_id", None),
                tiled,
                self._fp8_kv,
                owner_perm is not None,
            )
            cached_comp = (
                stage_cache.get(comp_key)
                if stage_cache is not None and self.indexer is None
                else None
            )
            if cached_comp is not None and tiled:
                main_cache, pages, starts, main_records, segmented_main = cached_comp
            else:
                source = self._static_forward_context[self.compressed_cache_prefix]
                capacity = max(1, self._max_model_len // self.compress_ratio)
                tp_ranks = (
                    get_tensor_model_parallel_world_size()
                    if owner_perm is not None
                    else 1
                )
                segmented_main = bool(
                    owner_perm is not None
                    and tiled
                    and tp_ranks > 1
                    and (self.indexer is None or not self._all_indices_fit)
                )
                comp_inputs_key = self._stage_cache_key(
                    "compressed_inputs",
                    self.compress_ratio,
                    segmented_main,
                    tp_ranks,
                    id(owner_perm) if segmented_main else None,
                )
                cached_comp_inputs = (
                    stage_cache.get(comp_inputs_key)
                    if stage_cache is not None
                    else None
                )
                if cached_comp_inputs is None:
                    lengths = ((positions.int() + 1) // self.compress_ratio)[:, None]
                    if segmented_main and owner_perm is not None:
                        seg_count = owner_perm.shape[0] // tp_ranks
                        # index_select, not a tensor-bound slice: slicing calls
                        # Tensor.__index__ (.item()), creating unbacked SymInts
                        # that break stage-cache reuse under compile (KV_SHARD).
                        seg_perm = (
                            owner_perm[: tp_ranks * seg_count]
                            .reshape(tp_ranks, seg_count)
                            .index_select(0, self._get_stage_tp_rank_1d(stage_cache))
                            .squeeze(0)
                        )
                        seg_valid = seg_perm >= 0
                        seg_idx = seg_perm.clamp(min=0)
                        lengths_comp = lengths.index_select(0, seg_idx).masked_fill(
                            ~seg_valid[:, None], 0
                        )
                        swa_valid_comp = swa_valid.index_select(0, seg_idx) & seg_valid
                        requests_comp = requests.index_select(0, seg_idx).masked_fill(
                            ~seg_valid, 0
                        )
                    else:
                        lengths_comp = lengths
                        swa_valid_comp = swa_valid
                        requests_comp = requests
                    if stage_cache is not None:
                        stage_cache[comp_inputs_key] = (
                            lengths,
                            lengths_comp,
                            swa_valid_comp,
                            requests_comp,
                        )
                else:
                    (
                        lengths,
                        lengths_comp,
                        swa_valid_comp,
                        requests_comp,
                    ) = cached_comp_inputs
                if self.indexer is not None:
                    if self._all_indices_fit:
                        logical_compressed = torch.arange(
                            capacity, dtype=torch.int32, device=positions.device
                        )[None, :].expand(count, -1)
                        selected = logical_compressed.masked_fill(
                            (logical_compressed >= lengths) | (~swa_valid[:, None]),
                            -1,
                        )
                    else:
                        indexer = self.indexer
                        weights = cast(torch.Tensor, weights)
                        keys_meta = metadata[indexer.k_cache.prefix]
                        index_cache = indexer.k_cache.kv_cache
                        if tiled:
                            # Rows are row_indices of _row_slots(..., logical),
                            # computed per request: [requests, capacity]. The
                            # indexer shares each request's key tiles.
                            has_fast_idx_offset = _has_fast_cache_offset(index_cache)
                            idx_page_records = _index_page_records(index_cache)
                            idx_req_key = self._stage_cache_key(
                                "index_req_slots",
                                indexer.k_cache.prefix,
                                index_cache.shape[1] if has_fast_idx_offset else None,
                                (
                                    _fast_cache_offset_key(index_cache)
                                    if has_fast_idx_offset
                                    else None
                                ),
                                capacity,
                                idx_page_records,
                            )
                            cached_idx_req = (
                                stage_cache.get(idx_req_key)
                                if stage_cache is not None
                                else None
                            )
                            if cached_idx_req is None:
                                idx_table = _block_table(keys_meta)
                                cached_idx_req = (
                                    _request_row_slots(
                                        index_cache,
                                        idx_table,
                                        capacity,
                                        idx_page_records,
                                    )
                                    if idx_page_records > 1
                                    else _request_row_slots(
                                        index_cache, idx_table, capacity
                                    )
                                )
                                if stage_cache is not None:
                                    stage_cache[idx_req_key] = cached_idx_req
                            if has_fast_idx_offset:
                                idx_rec_key = self._stage_cache_key(
                                    "index_records",
                                    index_cache.row_offset,
                                    *idx_req_key[1:],
                                )
                                index_records = (
                                    stage_cache.get(idx_rec_key)
                                    if stage_cache is not None
                                    else None
                                )
                                if index_records is None:
                                    index_records = (
                                        cached_idx_req
                                        if index_cache.row_offset == 0
                                        else index_cache.row_offset + cached_idx_req
                                    )
                                    if stage_cache is not None:
                                        stage_cache[idx_rec_key] = index_records
                            else:
                                index_records = index_cache.row_indices(cached_idx_req)
                            selected = self._select_paged_indices(
                                qr,
                                positions,
                                index_cache,
                                index_records,
                                requests,
                                weights,
                                lengths,
                                swa_slots,
                                pending_index_write,
                                **(
                                    {"owner_perm": owner_perm}
                                    if owner_perm is not None
                                    else {}
                                ),
                            )
                        else:
                            index_q = self._project_index_q(qr, positions)
                            logical_compressed = torch.arange(
                                capacity, dtype=torch.int32, device=positions.device
                            )[None, :].expand(count, -1)
                            keys = _read_rows(
                                index_cache,
                                _block_table(keys_meta),
                                requests,
                                logical_compressed,
                            )
                            scores = (
                                torch.einsum(
                                    "thd,tcd->thc", index_q.float(), keys.float()
                                ).relu()
                                * weights[:, :, None]
                            ).sum(1)[None]
                            selected = self._select_indices(
                                scores, lengths, logical_compressed, count
                            )
                    sel_count = selected.shape[0]
                    if stage_cache is None or not tiled:
                        # Keep the persistent buffer's physical shape across compile
                        # buckets when not sharing stage-scoped main_records directly.
                        self.topk_indices_buffer.copy_(
                            torch.nn.functional.pad(
                                selected,
                                (
                                    0,
                                    self.topk_indices_buffer.shape[-1]
                                    - selected.shape[-1],
                                    0,
                                    self.topk_indices_buffer.shape[0] - sel_count,
                                ),
                                value=-1,
                            )
                        )
                        selected = self.topk_indices_buffer[
                            :sel_count,
                            : min(capacity, self.topk_indices_buffer.shape[-1]),
                        ]
                    else:
                        target_width = min(capacity, self.topk_indices_buffer.shape[-1])
                        if selected.shape[-1] < target_width:
                            selected = F.pad(
                                selected,
                                (0, target_width - selected.shape[-1]),
                                value=-1,
                            )
                        else:
                            selected = selected[:, :target_width]
                else:
                    sel_count = (
                        owner_perm.shape[0] // tp_ranks
                        if segmented_main and owner_perm is not None
                        else count
                    )
                    selected = self.topk_indices_buffer[
                        :sel_count, : min(capacity, self.topk_indices_buffer.shape[-1])
                    ]
                main_cache = source.kv_cache
                valid_compressed = (
                    (selected >= 0)
                    & (selected < lengths_comp)
                    & swa_valid_comp[:, None]
                )
                if tiled:
                    if self._fp8_kv:
                        pages_key = self._stage_cache_key(
                            "compressed_pages",
                            self.compressed_cache_prefix,
                        )
                        pages = (
                            stage_cache.get(pages_key)
                            if stage_cache is not None
                            else None
                        )
                        if pages is None:
                            pages = main_cache.pages(_block_table(source_meta))
                            if stage_cache is not None and pages is not None:
                                if pages.dtype != torch.int32:
                                    pages = pages.int()
                                stage_cache[pages_key] = pages
                    else:
                        pages = None
                    if pages is not None:
                        # SparseMLA resolves pages itself (Pallas), avoiding a
                        # [tokens, topk] XLA gather of the block table here.
                        main_records = selected.masked_fill(~valid_compressed, -1)
                        starts = (
                            owner_q_loc
                            if owner_q_loc is not None
                            else source_meta.query_start_loc
                        )
                        if stage_cache is not None and starts.dtype != torch.int32:
                            starts = starts.int()
                    else:
                        slots = _row_slots(
                            main_cache,
                            _block_table(source_meta),
                            requests_comp,
                            selected,
                        )
                        main_records = main_cache.row_indices(slots).masked_fill(
                            ~valid_compressed, -1
                        )
                    if stage_cache is not None:
                        stage_cache[comp_key] = (
                            main_cache,
                            pages,
                            starts,
                            main_records,
                            segmented_main,
                        )
                else:
                    compressed_rows = _read_rows(
                        source.kv_cache,
                        _block_table(source_meta),
                        requests,
                        selected,
                    )
                    rows = torch.cat((window, compressed_rows), 1)
                    valid_rows = torch.cat((valid_window, valid_compressed), 1)
        if tiled:
            assert records is not None and main_records is not None
            tp_rank_1d = self._get_stage_tp_rank_1d(stage_cache)
            if owner_perm is not None or owner_inv is not None:
                if stage_cache is not None:
                    owner_kw_key = self._stage_cache_key(
                        "owner_kwargs",
                        id(owner_perm),
                        id(owner_inv),
                        segmented_main,
                    )
                    owner_kwargs = stage_cache.get(owner_kw_key)
                    if owner_kwargs is None:
                        owner_kwargs = {
                            "owner_perm": owner_perm,
                            "owner_inv": owner_inv,
                            "main_records_segmented": segmented_main,
                        }
                        stage_cache[owner_kw_key] = owner_kwargs
                else:
                    owner_kwargs = {
                        "owner_perm": owner_perm,
                        "owner_inv": owner_inv,
                        "main_records_segmented": segmented_main,
                    }
            else:
                owner_kwargs = {}
            fused_swa = (
                stage_cache.setdefault("fused_swa", envs.TPU_DEEPSEEK_V41_FUSED_SWA)
                if stage_cache is not None
                else envs.TPU_DEEPSEEK_V41_FUSED_SWA
            )
            if fused_qo:
                return sparse_fp8_attention_qo(
                    qr,
                    qr_local,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    self._wq_b_full,
                    self._wq_b_full_scale,
                    positions,
                    self._get_stage_rope_cache(stage_cache),
                    swa_cache,
                    main_cache,
                    records,
                    main_records,
                    self.attn_sink,
                    self._full_attn_sink,
                    tp_rank_1d,
                    self.wo_a.weight,
                    self.wo_a.weight_scale,
                    self._wo_a_full,
                    self._wo_a_full_scale,
                    swa_valid_int32,
                    self.scale,
                    pages,
                    starts,
                    self._swa_max_requests,
                    self._precompact_records,
                    int(self.head_dim),
                    replicated_qr=self._replicated_local_q,
                    fused_swa=fused_swa,
                    **owner_kwargs,
                )
            assert q is not None
            if self._fp8_kv and self.compress_ratio:
                result = sparse_fp8_attention(
                    q,
                    swa_cache,
                    main_cache,
                    records,
                    main_records,
                    self.attn_sink,
                    self.scale,
                    pages,
                    starts,
                    self._swa_max_requests,
                    precompacted=self._precompact_records,
                    token_parallel=self._token_parallel_attn,
                    full_sink=self._full_attn_sink,
                    tp_rank=tp_rank_1d,
                    fused_swa=fused_swa,
                    **owner_kwargs,
                )
            else:
                main_is_sharded = bool(
                    getattr(main_cache, "sharded", False)
                    and (
                        self.compress_ratio
                        or getattr(main_cache, "replicated_blocks", 0) == 0
                    )
                )
                result = paged_attention(
                    q,
                    swa_cache,
                    main_cache,
                    records,
                    main_records,
                    self.attn_sink,
                    self.scale,
                    getattr(self, "query_block_size", 1) if non_causal else 1,
                    token_parallel=bool(
                        main_is_sharded or swa_is_sharded or owner_perm is not None
                    ),
                    full_sink=self._full_attn_sink,
                    tp_rank=tp_rank_1d,
                    **owner_kwargs,
                )
            output.copy_(result.masked_fill(swa_invalid_3d, 0))
            return None
        if (
            self._fp8_kv
            or envs.TPU_DEEPSEEK_V41_KV_SHARD
            or getattr(main_cache, "sharded", False)
        ) and self.compress_ratio:
            raise NotImplementedError("FP8 V4.1 KV cache requires the tiled path")
        assert q is not None
        rows = cast(torch.Tensor, rows)
        q = cast(torch.Tensor, q)
        indices = (
            torch.arange(rows.shape[1], dtype=torch.int32, device=rows.device)[None, :]
            .expand(count, -1)
            .masked_fill(~valid_rows, -1)
        )
        # Treat every query as its own batch: each has a fixed-width paged gather.
        result = sparse_attention(
            q[:, None], rows, self.attn_sink, indices[:, None], self.scale
        )[:, 0]
        output.copy_(result.masked_fill(swa_invalid_3d, 0))
        return None

    def _init_local_q_weights(self):
        """All-gather ``wq_b`` and ``wo_a`` for the token-local q path.

        Column ``r`` blocks of the gathered weights are rank ``r``'s shards, the
        same rank-major head/group order the token-parallel ``all_to_all``
        uses, so every per-head/per-group product is computed from identical
        weights. FP8 tensors are gathered as bytes.

        ``TPU_DEEPSEEK_V41_LOCAL_Q`` serves SP-sharded (encoder) layers;
        ``TPU_DEEPSEEK_V41_DECODE_LOCAL_Q`` serves TP-replicated layers (the
        CED decoder at decode), where the op slices each rank's token shard
        from the full ``qr`` instead of taking an SP shard.
        """
        tp_size = get_tensor_model_parallel_world_size()
        replicated = bool(
            envs.TPU_DEEPSEEK_V41_DECODE_LOCAL_Q and not self._sp_sharded_input
        )
        sharded = bool(
            envs.TPU_DEEPSEEK_V41_LOCAL_Q and not getattr(self, "_ced_decoder", False)
        )
        if (
            not (replicated or sharded)
            or tp_size <= 1
            or not (self._fp8_kv and self.compress_ratio)
            or self.n_local_groups < 1
        ):
            return
        wq_scale = getattr(self.wq_b, "weight_scale", None)
        wo_scale = getattr(self.wo_a, "weight_scale", None)
        if (
            wq_scale is None
            or wq_scale.ndim != 1
            or wo_scale is None
            or getattr(self.wq_b, KEEP_VLLM_LAYOUT_ATTR, False)
        ):
            return

        def gather_columns(weight):
            weight = weight.contiguous()
            gathered = tensor_model_parallel_all_gather(weight.view(torch.uint8), dim=1)
            return gathered.view(weight.dtype).contiguous()

        buffers = {
            "_wq_b_full": gather_columns(self.wq_b.weight),
            "_wq_b_full_scale": tensor_model_parallel_all_gather(
                wq_scale.contiguous(), dim=0
            ),
            "_wo_a_full": gather_columns(self.wo_a.weight),
            "_wo_a_full_scale": tensor_model_parallel_all_gather(
                wo_scale.reshape(-1).contiguous(), dim=0
            ),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)
        self._local_q_attn = True
        self._replicated_local_q = replicated

    def _select_paged_indices(
        self,
        qr,
        positions,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_slots,
        pending_index_write,
        owner_perm=None,
    ):
        """Publish explicit index-cache and logical-selection outputs."""
        indexer = self.indexer
        count = qr.shape[0]
        has_candidates = indexer.indexer_op.candidate_blocks is not None
        candidate_write = has_candidates and getattr(
            indexer.indexer_op, "candidate_write", False
        )
        write_records, write_keys = (
            pending_index_write
            if pending_index_write is not None
            else (
                swa_slots.new_empty((0,), dtype=torch.int32),
                qr.new_empty((0, indexer.head_dim)),
            )
        )
        # Scoring and selection are independent per query, but every TP rank
        # holds the whole batch: split the queries over ranks and gather the
        # selections. Every rank still writes all index keys (replicated cache).
        # Contiguous parts of `_query_split` rows, computed rather than branched
        # on the token count, which would freeze the traced bucket's choice
        # (compilation/shape_variants.py). Out-of-range rows in the last part
        # are clamped for `index_select` and masked with `invalid_mine`; padding
        # rows have zero length and select nothing.
        # A 16-row-aligned part (SparseCore top-k row tile) put a whole decode
        # batch of ``count <= 16`` on rank 0 while the other ranks scored only
        # padding (split change alone at C=16: L20 3.20 -> 1.48 ms,
        # L24 0.75 -> 0.31 ms).
        ranks = get_tensor_model_parallel_world_size()
        kv_shard = bool(
            envs.TPU_DEEPSEEK_V41_KV_SHARD or getattr(index_cache, "sharded", False)
        )
        split = ranks > 1
        stage_cache = _active_stage_attn_cache
        query_split_key = _memo_stage_cache_key(
            self,
            "indexer_query_split",
            getattr(self, "compress_ratio", None),
            ranks,
            kv_shard,
            owner_perm is not None,
        )
        cached_query_split = (
            stage_cache.get(query_split_key) if stage_cache is not None else None
        )
        if cached_query_split is None:
            # Padding queries neither select nor publish candidates.
            active_lengths = lengths.masked_fill(swa_slots[:, None] < 0, 0)
            batch_max = active_lengths.max()
            if split:
                tp_rank = (
                    self._get_stage_tp_rank(stage_cache)
                    if hasattr(self, "_get_stage_tp_rank")
                    else getattr(self, "_tp_rank", 0)
                )
                if owner_perm is not None:
                    part = owner_perm.shape[0] // ranks
                    mine = (
                        owner_perm[: ranks * part]
                        .reshape(ranks, part)
                        .index_select(0, self._get_stage_tp_rank_1d(stage_cache))
                        .squeeze(0)
                        .to(torch.int32)
                    )
                else:
                    part, _ = _query_split(count, ranks, kv_shard=kv_shard)
                    mine = tp_rank * part + torch.arange(
                        part, dtype=torch.int32, device=qr.device
                    )
                invalid_mine, clamped_mine = (
                    (mine < 0) | (mine >= count),
                    mine.clamp(min=0, max=count - 1),
                )
                invalid_mine_2d = invalid_mine[:, None]
                positions_part, lengths_part, requests_part = (
                    positions.index_select(0, clamped_mine).masked_fill(
                        invalid_mine, 0
                    ),
                    active_lengths.index_select(0, clamped_mine).masked_fill(
                        invalid_mine.reshape(-1, *([1] * (active_lengths.ndim - 1))), 0
                    ),
                    requests.index_select(0, clamped_mine).masked_fill(invalid_mine, 0),
                )
            else:
                clamped_mine = None
                invalid_mine_2d = None
                positions_part = positions
                lengths_part = active_lengths
                requests_part = requests
            cached_query_split = (
                active_lengths,
                batch_max,
                clamped_mine,
                invalid_mine_2d,
                positions_part,
                lengths_part,
                requests_part,
            )
            if stage_cache is not None:
                stage_cache[query_split_key] = cached_query_split
        else:
            (
                active_lengths,
                batch_max,
                clamped_mine,
                invalid_mine_2d,
                positions_part,
                lengths_part,
                requests_part,
            ) = cached_query_split
        if split:
            qr_part, weights_part = (
                qr.index_select(0, clamped_mine).masked_fill(invalid_mine_2d, 0),
                weights.index_select(0, clamped_mine).masked_fill(invalid_mine_2d, 0),
            )
        else:
            qr_part, weights_part = qr, weights
        cand_key = (
            _memo_stage_cache_key(
                self,
                "indexer_candidates_part",
                id(self.candidate_block_buffer),
                *query_split_key[1:],
            )
            if stage_cache is not None and has_candidates
            else None
        )
        if not has_candidates:
            candidates_part = swa_slots.new_empty(
                (qr_part.shape[0], 0), dtype=torch.int32
            )
        elif (
            stage_cache is not None
            and not candidate_write
            and (cached_candidates_part := stage_cache.get(cand_key)) is not None
        ):
            candidates_part = cached_candidates_part
        else:
            if not split or owner_perm is not None:
                candidates_part = self.candidate_block_buffer[: qr_part.shape[0]]
            else:
                candidates_part = (
                    self.candidate_block_buffer[:count]
                    .index_select(0, clamped_mine)
                    .masked_fill(invalid_mine_2d, -1)
                )
            if stage_cache is not None and not candidate_write:
                stage_cache[cand_key] = candidates_part
        q_part = self._project_index_q(qr_part, positions_part)
        page_records = _index_page_records(index_cache)
        max_model_len = getattr(self, "_max_model_len", 0)
        compress_ratio = getattr(self, "compress_ratio", 0)
        capacity = (
            max(1, max_model_len // compress_ratio)
            if (
                isinstance(max_model_len, int)
                and isinstance(compress_ratio, int)
                and max_model_len > 0
                and compress_ratio > 0
            )
            else 0
        )
        prestrided = bool(
            page_records > 1 and capacity > 0 and records.shape[1] < capacity
        )
        updated_cache, selected, published = paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            topk=self._config["index_topk"],
            block_size=max(1, self.candidate_block_size),
            candidate_write=candidate_write,
            candidate_filter=has_candidates and not candidate_write,
            # FP8 attention re-sorts the selection (compact_records).
            unordered=self._fp8_kv,
            batch_max=batch_max,
            # Score tiles gather whole pages: one index per 512-byte pool row
            # instead of one per record.
            page_records=page_records,
            **(
                {"prestrided_pages": True, "logical_width": capacity}
                if prestrided
                else {}
            ),
        )
        if split and owner_perm is None:
            selected = _gather_query_rows(selected)[:count]
            if candidate_write and stage_cache is None:
                published = _gather_query_rows(published)[:count]
        if isinstance(updated_cache, tuple):
            index_cache.copy_raw(updated_cache)
        else:
            index_cache.raw.copy_(updated_cache)
        if candidate_write:
            if stage_cache is not None:
                stage_cache[cand_key] = (
                    published.masked_fill(invalid_mine_2d, -1)
                    if (split and owner_perm is None)
                    else published
                )
            else:
                self.candidate_block_buffer.copy_(
                    torch.nn.functional.pad(
                        published,
                        (
                            0,
                            0,
                            0,
                            self.candidate_block_buffer.shape[0] - published.shape[0],
                        ),
                        value=-1,
                    )
                )
        return selected

    def _select_indices(self, scores, lengths, logical_compressed, count):
        indexer = self.indexer
        candidates = None
        # If every block fits, publication and filtering are identities. The
        # Flash config has 2048 candidate slots but only 512 blocks at 4K.
        capacity = logical_compressed.shape[-1]
        if (
            indexer.indexer_op.candidate_blocks is not None
            and self.candidate_block_buffer.shape[-1] * self.candidate_block_size
            < capacity
        ):
            if getattr(indexer.indexer_op, "candidate_write", False):
                published = select_candidate_blocks(
                    scores,
                    lengths,
                    self._config["candidate_topk_blocks"],
                    self.candidate_block_size,
                )
                blocks = published[..., :: self.candidate_block_size]
                ids = (
                    torch.arange(
                        blocks.shape[-1], dtype=torch.int32, device=blocks.device
                    )
                    .expand_as(blocks)
                    .masked_fill(~blocks, -1)
                )
                selected = ids.topk(
                    min(ids.shape[-1], self.candidate_block_buffer.shape[-1]), dim=-1
                ).values
                self.candidate_block_buffer.copy_(
                    torch.nn.functional.pad(
                        selected[0],
                        (
                            0,
                            self.candidate_block_buffer.shape[-1] - selected.shape[-1],
                            0,
                            self.candidate_block_buffer.shape[0] - count,
                        ),
                        value=-1,
                    )
                )
            else:
                ids = self.candidate_block_buffer[:count]
                # Membership is a 2-D block table, not [tokens, context, topk].
                # The extra column absorbs all invalid/padded candidate ids.
                num_blocks = (
                    capacity + self.candidate_block_size - 1
                ) // self.candidate_block_size
                valid_ids = (ids >= 0) & (ids < num_blocks)
                block_mask = torch.zeros(
                    (count, num_blocks + 1), dtype=torch.bool, device=ids.device
                )
                block_mask.scatter_(
                    1, ids.masked_fill(~valid_ids, num_blocks).long(), True
                )
                candidates = (
                    block_mask.reshape(-1)
                    .index_select(
                        0,
                        (
                            torch.arange(count, device=ids.device, dtype=torch.int32)[
                                :, None
                            ]
                            * (num_blocks + 1)
                            + logical_compressed.to(torch.int32)
                            // self.candidate_block_size
                        ).reshape(-1),
                    )
                    .reshape(1, count, capacity)
                )
        if candidates is not None:
            candidates = candidates | (lengths.max() <= self._config["index_topk"])
        return select_compressed_indices(
            scores, lengths, self._config["index_topk"], candidates=candidates
        )[0]

    def forward_mqa(self, q, kv, positions, output):
        raise RuntimeError("TPU sparse attention is dispatched by _prepare_tpu")

    @property
    def o_proj_op(self):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        getter = cast(Callable, VllmDeepseekV4MLAAttention.o_proj_op.fget)
        return getter(self)

    def _assert_o_proj_supported(self):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        VllmDeepseekV4MLAAttention._assert_o_proj_supported(self)

    def process_weights_after_loading(self, act_dtype):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        VllmDeepseekV4MLAAttention.process_weights_after_loading(self, act_dtype)
        self._assert_o_proj_supported()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            self._full_attn_sink.copy_(
                tensor_model_parallel_all_gather(
                    self.attn_sink.detach().float().contiguous(), dim=0
                )
            )
        else:
            self._full_attn_sink.copy_(self.attn_sink.detach().float())
        self._init_local_q_weights()
        # Register before TorchDynamo captures the first model forward.
        if self.wo_a.weight.device.type == "tpu":
            _ = self.o_proj_op

    def _o_proj(self, output, positions):
        assert output.dtype == torch.bfloat16
        assert output.shape[1:] == (self.n_local_heads, self.head_dim)
        # Keep V4.1's bounded rotary table while using V4's FP8 projection kernel.
        projected = self.o_proj_op(
            output,
            positions,
            self._get_stage_rope_cache(_active_stage_attn_cache),
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        return _value(self.wo_b(projected))
