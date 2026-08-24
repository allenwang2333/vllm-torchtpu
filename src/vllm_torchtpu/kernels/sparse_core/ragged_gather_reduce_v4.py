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
"""SparseCore Ragged Gather-Reduce v4 with Continuous Inter-Block Prefetching.

This kernel implements an optimized destination-major gather-reduce on TPU SparseCore
(v7x Ironwood) featuring:
1. True Continuous Inter-Block Prefetching: While the SparseCore subcore computes the
   final chunk of Block N, it proactively issues the indirect gather DMA for Chunk 0
   of Block N+1 into the alternate gather buffer slot.
2. Inter-Block Output Overlapping: Packed BF16 output DMAs stream to HBM asynchronously
   and overlap with the subsequent block's accumulator initialization and compute.
3. Asynchronous Metadata Double-Buffering: Compaction metadata, routing weights, and
   valid counts for Block N+1/N+2 are fetched asynchronously into 1D VMEM staging buffers.
4. Elimination of Nested Pipeline Overhead: Avoids per-block emit_pipeline teardown
   and drains, keeping execution pipelines fully saturated without cold-start bubbles.
5. Strict VMEM Capacity Budgeting: Limits total subcore VMEM allocations to ~458 KiB,
   well within the 512 KiB hardware limit.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS  # 64 destination tokens
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1
_VMEM_1D_ALIGNMENT = 128  # 1D VMEM tiling boundary in elements


def _align_to(value: int, alignment: int) -> int:
    """Aligns an integer value up to the nearest multiple of alignment.

    Args:
        value: The integer value to align.
        alignment: The alignment boundary (must be positive).

    Returns:
        The aligned integer value greater than or equal to value.
    """
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Fallback reference implementation running via standard JAX primitives.

    Args:
        x: Input activations tensor of shape [num_source_rows, hidden_size].
        indices: 1D route indices tensor of shape [num_tokens * topk].
        topk_weights: 1D routing weights tensor of shape [num_tokens * topk].
        valid_rows_mask: 1D boolean validity mask of shape [num_tokens * topk].
        reduce_group_size: Number of routes per token (topk).

    Returns:
        Reduced output tensor of shape [num_tokens, hidden_size].
    """
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0.0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    """Calculates optimal column partitioning factor across SparseCore subcores.

    Args:
        hidden_size: Feature dimension size (H).
        num_cores: Total number of active subcores across SparseCores.
        num_lanes: Number of vector lanes per SparseCore subcore (16).

    Returns:
        The number of column partitions (power of 2).
    """
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def _main_kernel_v4(
    x_hbm_ref: jax.Ref,
    route_metadata_hbm_ref: jax.Ref,
    route_weights_hbm_ref: jax.Ref,
    route_counts_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    accum_vmem_ref: jax.Ref,
    gather_buf_vmem_ref: jax.Ref,
    packed_out_vmem_ref: jax.Ref,
    meta_buf_vmem_ref: jax.Ref,
    weights_buf_vmem_ref: jax.Ref,
    counts_buf_vmem_ref: jax.Ref,
    gather_sem_ref: jax.Ref,
    output_sem_ref: jax.Ref,
    meta_sem_ref: jax.Ref,
    weights_sem_ref: jax.Ref,
    counts_sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    metadata_block_size: int,
    topk: int,
    skip_compute: bool = False,
):
    """Core kernel function executed on each SparseCore subcore with continuous prefetch.

    Orchestrates continuous double-buffering across chunk and block boundaries:
    - Double-buffered VMEM gather staging: (2, 16, col_size)
    - Double-buffered VMEM packed BF16 output staging: (2, 8, col_size)
    - Double-buffered 1D VMEM route metadata and weights: (2 * metadata_block_size,)
    - Overlapped asynchronous DMA gather and DMA writeout across blocks.

    Args:
        x_hbm_ref: Input tensor ref in HBM [num_source_rows, hidden_size].
        route_metadata_hbm_ref: Compacted route metadata ref in HBM.
        route_weights_hbm_ref: Compacted route weights ref in HBM.
        route_counts_hbm_ref: Valid route counts per block ref in HBM.
        out_hbm_ref: Output tensor ref in HBM [padded_num_tokens, hidden_size].
        accum_vmem_ref: FP32 accumulator tile ref in VMEM (64, col_size).
        gather_buf_vmem_ref: Double-buffered gather staging ref in VMEM (2, 16, col_size).
        packed_out_vmem_ref: Double-buffered packed BF16 output ref in VMEM (2, 8, col_size).
        meta_buf_vmem_ref: Double-buffered 1D metadata ref in VMEM (2 * metadata_block_size,).
        weights_buf_vmem_ref: Double-buffered 1D weights ref in VMEM (2 * metadata_block_size,).
        counts_buf_vmem_ref: Double-buffered 1D counts ref in VMEM (2 * 16,).
        gather_sem_ref: DMA semaphores for gather buffer ping-pong (2,).
        output_sem_ref: DMA semaphores for output writeout ping-pong (2,).
        meta_sem_ref: DMA semaphores for metadata prefetch (2,).
        weights_sem_ref: DMA semaphores for weights prefetch (2,).
        counts_sem_ref: DMA semaphores for counts prefetch (2,).
        core_axis_name: Pallas mesh core axis name.
        subcore_axis_name: Pallas mesh subcore axis name.
        num_row_partitions: Total number of destination row partitions.
        num_column_partitions: Total number of feature column partitions.
        blocks_per_row_partition: Number of 64-token blocks assigned per row partition.
        metadata_block_size: Aligned 1D size per block metadata in elements.
        topk: Number of routes per destination token.
        skip_compute: When True, executes only HBM memory transfers for profiling.
    """
    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    col_start = pl.multiple_of(col_partition_id * col_size, _TOKEN_SUBCHUNK)

    x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    def _zero_accumulator():
        """Clears the 64xCol FP32 accumulator tile in VMEM to zero."""
        if skip_compute:
            return

        with jax.named_scope("v4_zero_accum"):
            def zero_col(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
                for r in range(_TOKEN_BLOCK):
                    accum_vmem_ref[r, col_slice] = zero

            plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(zero_col)

    def _issue_metadata_dma(block_idx, slot):
        """Asynchronously transfers metadata, weights, and counts for block_idx into VMEM."""
        with jax.named_scope("v4_meta_prefetch"):
            global_block_id = row_partition_id * blocks_per_row_partition + block_idx
            meta_offset = global_block_id * metadata_block_size
            counts_offset = global_block_id * _TOKEN_SUBCHUNK
            vmem_meta_offset = slot * metadata_block_size
            vmem_counts_offset = slot * _TOKEN_SUBCHUNK

            pltpu.make_async_copy(
                route_metadata_hbm_ref.at[pl.ds(meta_offset, metadata_block_size)],
                meta_buf_vmem_ref.at[pl.ds(vmem_meta_offset, metadata_block_size)],
                meta_sem_ref.at[slot],
            ).start()
            pltpu.make_async_copy(
                route_weights_hbm_ref.at[pl.ds(meta_offset, metadata_block_size)],
                weights_buf_vmem_ref.at[pl.ds(vmem_meta_offset, metadata_block_size)],
                weights_sem_ref.at[slot],
            ).start()
            pltpu.make_async_copy(
                route_counts_hbm_ref.at[pl.ds(counts_offset, _TOKEN_SUBCHUNK)],
                counts_buf_vmem_ref.at[pl.ds(vmem_counts_offset, _TOKEN_SUBCHUNK)],
                counts_sem_ref.at[slot],
            ).start()

    def _wait_metadata_dma(slot):
        """Waits for metadata, weights, and counts DMAs for slot to complete."""
        with jax.named_scope("v4_meta_wait"):
            vmem_meta_offset = slot * metadata_block_size
            vmem_counts_offset = slot * _TOKEN_SUBCHUNK
            pltpu.make_async_copy(
                route_metadata_hbm_ref.at[pl.ds(0, metadata_block_size)],
                meta_buf_vmem_ref.at[pl.ds(vmem_meta_offset, metadata_block_size)],
                meta_sem_ref.at[slot],
            ).wait()
            pltpu.make_async_copy(
                route_weights_hbm_ref.at[pl.ds(0, metadata_block_size)],
                weights_buf_vmem_ref.at[pl.ds(vmem_meta_offset, metadata_block_size)],
                weights_sem_ref.at[slot],
            ).wait()
            pltpu.make_async_copy(
                route_counts_hbm_ref.at[pl.ds(0, _TOKEN_SUBCHUNK)],
                counts_buf_vmem_ref.at[pl.ds(vmem_counts_offset, _TOKEN_SUBCHUNK)],
                counts_sem_ref.at[slot],
            ).wait()

    def _issue_gather_dma(meta_slot, chunk_idx, gather_slot):
        """Issues indirect DMA gather for 16 activation rows into gather_buf_vmem_ref[gather_slot]."""
        with jax.named_scope("v4_prefetch_dma"):
            start = meta_slot * metadata_block_size + chunk_idx * _TOKEN_SUBCHUNK
            meta_slice = meta_buf_vmem_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            source_indices = jnp.bitwise_right_shift(
                jnp.bitwise_and(meta_slice, _ROUTE_INDEX_MASK), 1
            )
            pltpu.make_async_copy(
                x_32b_hbm_ref.at[source_indices, pl.ds(col_start, col_size)],
                gather_buf_vmem_ref.at[gather_slot],
                gather_sem_ref.at[gather_slot],
            ).start()

    def _wait_gather_dma(gather_slot):
        """Waits for DMA gather into gather_buf_vmem_ref[gather_slot] to complete."""
        with jax.named_scope("v4_wait_in"):
            pltpu.make_async_copy(
                x_32b_hbm_ref.at[pl.ds(0, _TOKEN_SUBCHUNK), pl.ds(col_start, col_size)],
                gather_buf_vmem_ref.at[gather_slot],
                gather_sem_ref.at[gather_slot],
            ).wait()

    def _compute_chunk(meta_slot, chunk_idx, gather_slot, valid_count):
        """Unpacks BF16 values, multiplies by route weights, and vector scatter-adds to accumulator."""
        if skip_compute:
            return

        with jax.named_scope("v4_run_compute"):
            start = meta_slot * metadata_block_size + chunk_idx * _TOKEN_SUBCHUNK
            route_meta = meta_buf_vmem_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            source_indices = jnp.bitwise_and(route_meta, _ROUTE_INDEX_MASK)
            route_destinations = jnp.bitwise_right_shift(route_meta, _ROUTE_INDEX_BITS)
            route_weights = weights_buf_vmem_ref[pl.ds(start, _TOKEN_SUBCHUNK)]

            def compute_col(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)

                for batch_start in range(0, _TOKEN_SUBCHUNK, 4):
                    for batch_lane in range(4):
                        lane = batch_start + batch_lane
                        val_i32 = gather_buf_vmem_ref[gather_slot, lane, col_slice]
                        shift = jnp.where(
                            jnp.bitwise_and(source_indices[lane], 1) == 0, 16, 0
                        )
                        val_f32 = plsc.bitcast(
                            jnp.bitwise_and(
                                jnp.left_shift(val_i32, shift), jnp.int32(-65536)
                            ),
                            jnp.float32,
                        ) * route_weights[lane]
                        dest = jnp.full_like(feature_indices, route_destinations[lane])
                        route_valid = (chunk_idx * _TOKEN_SUBCHUNK + lane < valid_count)
                        scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                        plsc.addupdate_scatter(
                            accum_vmem_ref,
                            (dest, feature_indices),
                            val_f32,
                            mask=scatter_mask,
                        )

            plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(compute_col)

    def _pack_and_write_output(block_idx):
        """Packs pairs of FP32 accum rows into uint32 BF16 and streams tiles to HBM."""
        with jax.named_scope("v4_output_pack_write"):
            global_block_id = row_partition_id * blocks_per_row_partition + block_idx
            for token_subchunk in range(_NUM_TOKEN_SUBCHUNKS):
                out_slot = token_subchunk % 2
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

                # Wait on previous DMA if reusing this output buffer slot
                @pl.when((token_subchunk >= 2) | (block_idx > 0))
                def _wait_prev_out_dma():
                    pltpu.make_async_copy(
                        packed_out_vmem_ref.at[out_slot],
                        out_32b_hbm_ref.at[
                            pl.ds(0, _TOKEN_SUBCHUNK // 2), pl.ds(col_start, col_size)
                        ],
                        output_sem_ref.at[out_slot],
                    ).wait()

                if not skip_compute:
                    def pack_col(col_offset, row_start=accum_row_start, slot=out_slot):
                        col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                        for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                            accum_row = row_start + token_lane
                            packed_bf16 = plsc.pack(
                                accum_vmem_ref[accum_row, col_slice],
                                accum_vmem_ref[accum_row + 1, col_slice],
                                format=plsc.PackFormat.INTERLEAVED,
                            )
                            packed_out_vmem_ref[
                                slot, token_lane // 2, col_slice
                            ] = plsc.bitcast(packed_bf16, jnp.uint32)

                    plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(pack_col)

                output_row = (
                    global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
                )
                output_row_packed = pl.multiple_of(
                    output_row // 2, _TOKEN_SUBCHUNK // 2
                )

                pltpu.make_async_copy(
                    packed_out_vmem_ref.at[out_slot],
                    out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(col_start, col_size),
                    ],
                    output_sem_ref.at[out_slot],
                ).start()

    # -------------------------------------------------------------------------
    # PROLOGUE: Fetch Block 0 metadata and prefetch Chunk 0
    # -------------------------------------------------------------------------
    _issue_metadata_dma(block_idx=0, slot=0)
    _wait_metadata_dma(slot=0)

    b0_valid_count = counts_buf_vmem_ref[pl.ds(0, _TOKEN_SUBCHUNK)][0]
    b0_num_chunks = pl.cdiv(b0_valid_count, _TOKEN_SUBCHUNK)

    # Prefetch Block 1 metadata if there are >= 2 blocks
    if blocks_per_row_partition > 1:
        _issue_metadata_dma(block_idx=1, slot=1)

    @pl.when(b0_num_chunks > 0)
    def _init_b0_gather():
        _issue_gather_dma(meta_slot=0, chunk_idx=0, gather_slot=0)

    cur_gather_slot = 0

    # -------------------------------------------------------------------------
    # MAIN BLOCK ITERATION: Process destination token blocks via fori_loop
    # -------------------------------------------------------------------------
    def _block_body(b, cur_slot):
        meta_slot = b % 2
        next_meta_slot = (b + 1) % 2

        _zero_accumulator()

        cur_valid_count = counts_buf_vmem_ref[
            pl.ds(meta_slot * _TOKEN_SUBCHUNK, _TOKEN_SUBCHUNK)
        ][0]
        cur_num_chunks = pl.cdiv(cur_valid_count, _TOKEN_SUBCHUNK)

        @pl.when(b + 1 < blocks_per_row_partition)
        def _handle_next_meta():
            _wait_metadata_dma(slot=next_meta_slot)

        next_valid_count = jnp.where(
            b + 1 < blocks_per_row_partition,
            counts_buf_vmem_ref[
                pl.ds(next_meta_slot * _TOKEN_SUBCHUNK, _TOKEN_SUBCHUNK)
            ][0],
            0,
        )
        next_num_chunks = pl.cdiv(next_valid_count, _TOKEN_SUBCHUNK)

        # Chunk loop with inter-chunk and inter-block prefetch
        def _chunk_body(chunk_idx, slot):
            _wait_gather_dma(gather_slot=slot)
            next_slot = 1 - slot

            is_last_chunk = chunk_idx + 1 == cur_num_chunks

            @pl.when(jnp.logical_not(is_last_chunk))
            def _prefetch_intra():
                _issue_gather_dma(
                    meta_slot=meta_slot,
                    chunk_idx=chunk_idx + 1,
                    gather_slot=next_slot,
                )

            @pl.when(is_last_chunk & (next_num_chunks > 0))
            def _prefetch_inter():
                _issue_gather_dma(
                    meta_slot=next_meta_slot,
                    chunk_idx=0,
                    gather_slot=next_slot,
                )

            _compute_chunk(
                meta_slot=meta_slot,
                chunk_idx=chunk_idx,
                gather_slot=slot,
                valid_count=cur_valid_count,
            )
            return next_slot

        cur_slot = jax.lax.fori_loop(0, cur_num_chunks, _chunk_body, cur_slot)

        # If current block had 0 chunks but next block has chunks, issue Chunk 0 prefetch
        @pl.when((cur_num_chunks == 0) & (next_num_chunks > 0))
        def _prefetch_when_cur_empty():
            _issue_gather_dma(
                meta_slot=next_meta_slot,
                chunk_idx=0,
                gather_slot=cur_slot,
            )

        # Pack and stream output (overlaps with next block's Chunk 0 gather)
        _pack_and_write_output(block_idx=b)

        # Prefetch metadata for block b+2
        @pl.when(b + 2 < blocks_per_row_partition)
        def _prefetch_meta_b2():
            _issue_metadata_dma(block_idx=b + 2, slot=meta_slot)

        return cur_slot

    cur_gather_slot = jax.lax.fori_loop(
        0, blocks_per_row_partition, _block_body, cur_gather_slot
    )

    # -------------------------------------------------------------------------
    # EPILOGUE: Drain pending output DMAs
    # -------------------------------------------------------------------------
    with jax.named_scope("v4_output_drain"):
        for s in range(2):
            pltpu.make_async_copy(
                packed_out_vmem_ref.at[s],
                out_32b_hbm_ref.at[
                    pl.ds(0, _TOKEN_SUBCHUNK // 2), pl.ds(col_start, col_size)
                ],
                output_sem_ref.at[s],
            ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("reduce_group_size", "num_sparse_cores", "skip_compute"),
)
def ragged_gather_reduce_v4(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_sparse_cores: int = 1,
    skip_compute: bool = False,
) -> jax.Array:
    """SparseCore Ragged Gather-Reduce v4 with Continuous Inter-Block Prefetching.

    Performs destination-major MoE gather-reduce by assigning each SparseCore
    subcore a contiguous block of 64 destination tokens and streaming inputs
    and outputs with continuous double-buffering.

    Args:
        x: Input activations tensor of shape [num_source_rows, hidden_size] and
          dtype bfloat16.
        indices: 1D int32 route indices tensor of shape [num_tokens * topk].
        topk_weights: 1D bfloat16 routing weights tensor of shape [num_tokens *
          topk].
        valid_rows_mask: 1D boolean validity mask of shape [num_tokens * topk].
        reduce_group_size: Number of routes per destination token (topk).
        num_sparse_cores: Target number of physical SparseCores to utilize (1 or
          2).
        skip_compute: If True, executes memory movement without arithmetic for
          profiling.

    Returns:
        Reduced output tensor of shape [num_tokens, hidden_size] and dtype
        bfloat16.
    """
    sc_info = pltpu.get_tpu_info().sparse_core
    if (
        sc_info is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % pltpu.get_tpu_info().num_lanes != 0
    ):
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if (
        jnp.size(x) * dtype_bytes * 2
        < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6
    ):
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        raise ValueError(
            f"{input_size=} must be divisible by {reduce_group_size=}"
        )

    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))

    num_tokens = input_size // reduce_group_size
    target_num_sc_cores = (
        min(num_sparse_cores, sc_info.num_cores)
        if num_sparse_cores is not None
        else 1
    )
    num_cores = target_num_sc_cores * sc_info.num_subcores
    num_lanes = pltpu.get_tpu_info().num_lanes
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        raise ValueError(
            f"destination-major prototype supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )

    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, num_lanes
    )
    num_row_partitions = num_cores // num_column_partitions

    token_alignment = num_row_partitions * _TOKEN_BLOCK
    padded_num_tokens = _align_to(num_tokens, token_alignment)
    token_padding = padded_num_tokens - num_tokens

    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    unaligned_meta_size = _TOKEN_BLOCK * reduce_group_size
    metadata_block_size = _align_to(unaligned_meta_size, _VMEM_1D_ALIGNMENT)
    meta_padding = metadata_block_size - unaligned_meta_size

    indices_blocks = indices_2d.reshape(num_blocks, unaligned_meta_size)
    valid_blocks = valid_2d.reshape(num_blocks, unaligned_meta_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(
        destination_template, indices_blocks.shape
    )

    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(
        destination_blocks, route_order, axis=1
    )
    weights_blocks = weights_2d.reshape(num_blocks, unaligned_meta_size)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
    route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(
        jnp.int32
    )
    route_metadata = jnp.bitwise_or(
        route_indices,
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS),
    )
    route_weights = jnp.where(compact_valid, route_weights, 0).astype(
        jnp.float32
    )
    route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)

    if meta_padding > 0:
        route_metadata = jnp.pad(route_metadata, ((0, 0), (0, meta_padding)))
        route_weights = jnp.pad(route_weights, ((0, 0), (0, meta_padding)))

    route_metadata = route_metadata.reshape(-1)
    route_weights = route_weights.reshape(-1)
    route_counts = jnp.broadcast_to(
        route_counts[:, None], (num_blocks, _TOKEN_SUBCHUNK)
    ).reshape(-1)

    col_size = hidden_size // num_column_partitions
    blocks_per_row_partition = num_blocks // num_row_partitions
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=target_num_sc_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    out = core_map_helper.kernel(
        functools.partial(
            _main_kernel_v4,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_partitions=num_row_partitions,
            num_column_partitions=num_column_partitions,
            blocks_per_row_partition=blocks_per_row_partition,
            metadata_block_size=metadata_block_size,
            topk=reduce_group_size,
            skip_compute=skip_compute,
        ),
        out_type=jax.ShapeDtypeStruct(
            (padded_num_tokens, hidden_size), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(
            pltpu.VMEM((_TOKEN_BLOCK, col_size), jnp.float32),  # accum_vmem_ref
            pltpu.VMEM(
                (2, _TOKEN_SUBCHUNK, col_size), jnp.int32
            ),  # gather_buf_vmem_ref
            pltpu.VMEM(
                (2, _TOKEN_SUBCHUNK // 2, col_size), jnp.uint32
            ),  # packed_out_vmem_ref
            pltpu.VMEM(
                (2 * metadata_block_size,), jnp.int32
            ),  # meta_buf_vmem_ref (1D)
            pltpu.VMEM(
                (2 * metadata_block_size,), jnp.float32
            ),  # weights_buf_vmem_ref (1D)
            pltpu.VMEM(
                (2 * _TOKEN_SUBCHUNK,), jnp.int32
            ),  # counts_buf_vmem_ref (1D)
            pltpu.SemaphoreType.DMA((2,)),  # gather_sem_ref
            pltpu.SemaphoreType.DMA((2,)),  # output_sem_ref
            pltpu.SemaphoreType.DMA((2,)),  # meta_sem_ref
            pltpu.SemaphoreType.DMA((2,)),  # weights_sem_ref
            pltpu.SemaphoreType.DMA((2,)),  # counts_sem_ref
        ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v4",
    )(x, route_metadata, route_weights, route_counts)

    return out[:num_tokens]
