# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native packed FP8 index-cache ownership and model scoring bridge.

As in vLLM's DeepseekV4IndexerCache, allocation, insertion and scoring share
one byte format. The TPU page stores feature words by token column; it is
not NVIDIA's token-major page layout.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_quantization import quantize_index_rows
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_compact import (
    candidate_topk,
    compact_candidate_scores,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    packed_fp8_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_sparse import (
    candidate_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PACKED_FP8_MQA_CONFIG,
)

from . import attention_kernels
from .cache import PagedCacheRows, RawCachePool

_write_op = None
_write_2bank_op = None
_read_op = None
_read_2bank_op = None
_select_op = None
_select_2bank_op = None


class PackedFP8IndexCacheRows(PagedCacheRows):
    """Logical D128 keys backed by transposed FP8 pages in shared native owners."""

    def __init__(
        self,
        raw: RawCachePool,
        layout: tuple[tuple[int, ...], torch.dtype, int, int],
    ) -> None:
        physical_shape, dtype, stride_bytes, offset_bytes = layout
        if (
            dtype != torch.uint8
            or physical_shape[1:] != (PAGE_RECORDS, 132)
            or stride_bytes % 512
            or offset_bytes % 512
        ):
            raise ValueError(
                "Packed FP8 index cache requires 128 compressed records per page, "
                "132 bytes per record and 512-byte aligned owners; set block_size "
                "to 128 * the index compression ratio"
            )
        banks = raw if isinstance(raw, tuple) else (raw,)
        if len(banks) not in (1, 2) or any(
            bank.dtype != torch.int32 or bank.ndim != 2 or bank.shape[1] != 128
            for bank in banks
        ):
            raise ValueError("Packed FP8 index cache requires int32 [rows,128] owners")
        self.layout = PackedFP8Layout(
            physical_shape[0], stride_bytes // 512, offset_bytes // 512
        )
        self.layout.validate(tuple(bank.shape[0] for bank in banks))
        self.raw, self.raw_hi = banks[0], banks[1] if len(banks) == 2 else None
        self.shape = (physical_shape[0], PAGE_RECORDS, 128)
        self.dtype = torch.bfloat16
        # Addressing above the format boundary is in logical record slots.
        self.row_offset = 0
        self.page_stride = PAGE_RECORDS

    def row_indices(self, slots: torch.Tensor) -> torch.Tensor:
        return slots

    def __getitem__(self, indices: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        blocks, rows = indices
        records = (blocks * PAGE_RECORDS + rows).int()
        args = (records.flatten(), *self._layout_args())
        if self.raw_hi is None:
            assert _read_op is not None
            values = _read_op(self.raw, *args)
        else:
            assert _read_2bank_op is not None
            values = _read_2bank_op(self.raw, self.raw_hi, *args)
        return values.reshape(*records.shape, 128)

    def _layout_args(self) -> tuple[int, int, int]:
        return self.layout.pages, self.layout.page_stride, self.layout.page_offset

    def write(
        self,
        slots: torch.Tensor,
        values: torch.Tensor,
        after: torch.Tensor | None = None,
    ) -> None:
        after = values.new_empty((0,)) if after is None else after
        args = (slots.int(), values, after, *self._layout_args())
        if self.raw_hi is None:
            assert _write_op is not None
            self.raw.copy_(_write_op(self.raw, *args))
        else:
            assert _write_2bank_op is not None
            self.copy_raw(_write_2bank_op(self.raw, self.raw_hi, *args))


def _write(
    pool: jax.Array,
    records: jax.Array,
    keys: jax.Array,
    after: jax.Array,
    pages: int,
    stride: int,
    offset: int,
) -> jax.Array:
    pool, _ = jax.lax.optimization_barrier((pool, after))
    return write_fp8_records(
        (pool,), records, keys, PackedFP8Layout(pages, stride, offset)
    )[0]


def _write_two(
    lo: jax.Array,
    hi: jax.Array,
    records: jax.Array,
    keys: jax.Array,
    after: jax.Array,
    pages: int,
    stride: int,
    offset: int,
) -> tuple[jax.Array, jax.Array]:
    (lo, hi), _ = jax.lax.optimization_barrier(((lo, hi), after))
    updated_lo, updated_hi = write_fp8_records(
        (lo, hi), records, keys, PackedFP8Layout(pages, stride, offset)
    )
    return updated_lo, updated_hi


def _read(
    pool: jax.Array, records: jax.Array, pages: int, stride: int, offset: int
) -> jax.Array:
    return read_fp8_records((pool,), records, PackedFP8Layout(pages, stride, offset))


def _read_two(
    lo: jax.Array,
    hi: jax.Array,
    records: jax.Array,
    pages: int,
    stride: int,
    offset: int,
) -> jax.Array:
    return read_fp8_records((lo, hi), records, PackedFP8Layout(pages, stride, offset))


def _score_and_select(
    q,
    banks,
    records,
    weights,
    lengths,
    candidates,
    requests,
    batch_max,
    layout,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
    unordered,
):
    tokens, width = q.shape[0], records.shape[1]
    if tokens == 0:
        return jnp.empty((0, min(topk, width)), jnp.int32), candidates
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    key_tile = PACKED_FP8_MQA_CONFIG.key_tile
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    capacity = (width + key_tile - 1) // key_tile * key_tile
    pages = records[:, ::PAGE_RECORDS] // PAGE_RECORDS
    pages = jnp.pad(
        pages,
        ((0, 0), (0, capacity // PAGE_RECORDS - pages.shape[1])),
        constant_values=-1,
    )

    # Bound dense score workspace using the existing indexer's memory budget.
    rows = min(
        max(tokens, query_tile),
        max(query_tile, attention_kernels._INDEX_UNTILED_SCORE_ELEMENTS // capacity),
    )
    padded = (tokens + rows - 1) // rows * rows
    safe_requests = jnp.clip(requests, 0, records.shape[0] - 1)
    lengths = jnp.where(
        (requests >= 0) & (requests < records.shape[0]), jnp.clip(lengths, 0, width), 0
    )
    inputs = (q, weights, lengths, candidates, safe_requests)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1)) for x in inputs
    )
    order = jnp.argsort(inputs[-1], stable=True)
    inputs = tuple(x[order] for x in inputs)
    output = jnp.full((padded, min(topk, width)), -1, jnp.int32)
    published = jnp.full((padded, candidates.shape[1]), -1, jnp.int32)

    def body(index, outputs):
        queries, row_weights, row_lengths, row_candidates, owners = tuple(
            jax.lax.dynamic_slice_in_dim(x, index * rows, rows) for x in inputs
        )
        counts = jnp.bincount(owners, length=records.shape[0])
        offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(counts)))
        quantized, scales, _ = quantize_index_rows(queries)
        pool = banks[0] if len(banks) == 1 else banks

        def dense():
            scores = packed_fp8_mqa_logits(
                quantized,
                row_weights.astype(jnp.float32),
                pool,
                pages,
                row_lengths,
                offsets,
                q_scale=scales,
                cache_layout=layout,
            )[:, :width]
            return attention_kernels.select_indices_jax(
                scores,
                row_lengths,
                row_candidates,
                topk,
                block_size,
                candidate_write,
                candidate_filter,
                unordered,
                batch_max,
            )

        def sparse():
            if row_candidates.shape[1] == 0:
                return jnp.full((rows, min(topk, width)), -1, jnp.int32), row_candidates
            tasks = candidate_pages(
                row_candidates,
                row_lengths,
                owners,
                pages[:, : (width + PAGE_RECORDS - 1) // PAGE_RECORDS],
                block_size=block_size,
                cache_pages=layout.pages,
                query_start_loc=offsets,
            )
            scores = packed_fp8_mqa_logits(
                quantized,
                row_weights.astype(jnp.float32),
                pool,
                tasks.physical,
                tasks.lengths,
                tasks.query_offsets,
                q_scale=scales,
                cache_layout=layout,
                skip_invalid_pages=True,
            )
            scores = compact_candidate_scores(
                scores,
                tasks.candidates,
                block_size=block_size,
            )
            if unordered:
                selected = candidate_topk(
                    scores,
                    tasks.candidates,
                    k=min(topk, scores.shape[1]),
                    block_size=block_size,
                )
                selected = jnp.pad(
                    selected,
                    ((0, 0), (0, min(topk, width) - selected.shape[1])),
                    constant_values=-1,
                )
                return selected, row_candidates
            selected, _ = attention_kernels.select_indices_jax(
                scores,
                jnp.full_like(row_lengths, scores.shape[1]),
                tasks.candidates,
                min(topk, width),
                block_size,
                False,
                False,
                unordered,
                batch_max,
            )
            selected = jnp.pad(
                selected,
                ((0, 0), (0, min(topk, width) - selected.shape[1])),
                constant_values=-1,
            )
            blocks = jnp.take_along_axis(
                tasks.candidates, jnp.maximum(selected, 0) // block_size, axis=1
            )
            selected = jnp.where(
                selected >= 0, blocks * block_size + selected % block_size, -1
            )
            return selected, row_candidates

        if candidate_filter and candidates.shape[1] * block_size < width:
            # Match upstream's whole-batch short-context bypass, including TP
            # query shards and chunks whose local lengths are all short.
            selected, candidate_rows = jax.lax.cond(
                jnp.minimum(batch_max.max(), width) <= topk, dense, sparse
            )
        else:
            selected, candidate_rows = dense()
        return tuple(
            jax.lax.dynamic_update_slice_in_dim(dst, src, index * rows, axis=0)
            for dst, src in zip(outputs, (selected, candidate_rows))
        )

    output, published = jax.lax.fori_loop(0, padded // rows, body, (output, published))
    inverse = jnp.argsort(order)
    return output[inverse][:tokens], published[inverse][:tokens]


def _select(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    pages: int,
    stride: int,
    offset: int,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    layout = PackedFP8Layout(pages, stride, offset)
    updated = write_fp8_records((pool,), write_records, keys, layout)
    selected, published = _score_and_select(
        q,
        updated,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    return updated[0], selected, published


def _select_two(
    q: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    pages: int,
    stride: int,
    offset: int,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    layout = PackedFP8Layout(pages, stride, offset)
    updated = write_fp8_records((lo, hi), write_records, keys, layout)
    selected, published = _score_and_select(
        q,
        updated,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    updated_lo, updated_hi = updated
    return updated_lo, updated_hi, selected, published


def initialize_packed_fp8_indexer() -> None:
    """Register native-owner ops once, before tracing model forwards."""
    global _write_op, _write_2bank_op, _read_op, _read_2bank_op
    global _select_op, _select_2bank_op
    if _select_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

    def register(name, function, arrays, statics, donate=()):
        return pallas.jax_op(
            "pallas::dsv41_packed_fp8_" + name,
            function,
            donate_argnums=donate,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(Any, (P(),) * arrays + (None,) * statics),
        )

    _write_op = register("write", _write, 4, 3, (0,))
    _write_2bank_op = register("write_two", _write_two, 5, 3, (0, 1))
    _read_op = register("read", _read, 2, 3)
    _read_2bank_op = register("read_two", _read_two, 3, 3)
    _select_op = register("select", _select, 10, 8, (1,))
    _select_2bank_op = register("select_two", _select_two, 11, 8, (1, 2))

    def fake_write(pool, records, keys, after, pages, stride, offset):
        return torch.empty_like(pool)

    def fake_write_two(lo, hi, records, keys, after, pages, stride, offset):
        return torch.empty_like(lo), torch.empty_like(hi)

    def fake_read(pool, records, pages, stride, offset):
        return pool.new_empty((records.numel(), 128), dtype=torch.bfloat16)

    def fake_read_two(lo, hi, records, pages, stride, offset):
        return lo.new_empty((records.numel(), 128), dtype=torch.bfloat16)

    _write_op.register_fake(fake_write)
    _write_2bank_op.register_fake(fake_write_two)
    _read_op.register_fake(fake_read)
    _read_2bank_op.register_fake(fake_read_two)

    def fake_select(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        requests,
        batch_max,
        pages,
        stride,
        offset,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    ):
        return (
            torch.empty_like(pool),
            records.new_empty((q.shape[0], min(topk, records.shape[1]))),
            torch.empty_like(candidates),
        )

    def fake_select_two(
        q,
        lo,
        hi,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        requests,
        batch_max,
        pages,
        stride,
        offset,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    ):
        return (
            torch.empty_like(lo),
            torch.empty_like(hi),
            records.new_empty((q.shape[0], min(topk, records.shape[1]))),
            torch.empty_like(candidates),
        )

    _select_op.register_fake(fake_select)
    _select_2bank_op.register_fake(fake_select_two)


def paged_fp8_indexer(
    q: torch.Tensor,
    cache: PackedFP8IndexCacheRows,
    records: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    write_records: torch.Tensor,
    keys: torch.Tensor,
    requests: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    batch_max: torch.Tensor,
) -> tuple[RawCachePool, torch.Tensor, torch.Tensor]:
    """Keep the existing model's mutation and logical top-k output contract."""
    args = (
        records.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        write_records.int(),
        keys,
        requests.int(),
        batch_max.reshape(1).int(),
        *cache._layout_args(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    if cache.raw_hi is None:
        assert _select_op is not None
        return _select_op(q, cache.raw, *args)
    assert _select_2bank_op is not None
    lo, hi, selected, published = _select_2bank_op(q, cache.raw, cache.raw_hi, *args)
    return (lo, hi), selected, published
