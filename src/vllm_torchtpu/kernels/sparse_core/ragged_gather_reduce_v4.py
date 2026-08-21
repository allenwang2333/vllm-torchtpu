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
"""SparseCore Ragged Gather-Reduce v4 with Continuous Double-Buffering.

This kernel optimizes ragged gather-reduce on TPU SparseCore by eliminating
the per-block nested pipeline initialization overhead present in v3. It maintains
continuous double-buffered DMA prefetching for input tokens and double-buffered
asynchronous writeout for output blocks, achieving peak memory and compute overlap.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS  # 64
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    """Aligns an integer value up to the nearest multiple of alignment.

    Args:
        value: The integer value to align.
        alignment: The alignment boundary.

    Returns:
        The aligned integer value.
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
        x: Input tensor of shape [num_source_rows, hidden_size].
        indices: Route indices of shape [num_tokens * topk].
        topk_weights: Routing weights of shape [num_tokens * topk].
        valid_rows_mask: Validity mask of shape [num_tokens * topk].
        reduce_group_size: Number of routes per token (topk).

    Returns:
        Reduced output tensor of shape [num_tokens, hidden_size].
    """
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(hidden_size: int, num_cores: int,
                                     num_lanes: int) -> int:
    """Calculates optimal column partitioning factor across SparseCore subcores.

    Args:
        hidden_size: Feature dimension size.
        num_cores: Total number of active subcores across SparseCores.
        num_lanes: Number of vector lanes per SparseCore subcore (16).

    Returns:
        The number of column partitions.
    """
    preferred_num_stages = 4
    num_column_partitions = 1
    while (num_cores % (num_column_partitions * 2) == 0
           and hidden_size % (num_lanes * num_column_partitions * 2) == 0
           and hidden_size //
           (num_column_partitions * 2 * num_lanes) >= preferred_num_stages):
        num_column_partitions *= 2
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
            num_column_partitions // 2):
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
    gather_sem_ref: jax.Ref,
    output_sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    topk: int,
    skip_compute: bool = False,
):
    """Core kernel function executed on each SparseCore subcore.

    Employs a single outer emit_pipeline across destination token blocks with
    an internal jax.lax.fori_loop and double-buffering ping-pong DMA.

    Args:
        x_hbm_ref: Input tensor in HBM.
        route_metadata_hbm_ref: Compacted route metadata in HBM.
        route_weights_hbm_ref: Compacted route weights in HBM.
        route_counts_hbm_ref: Valid route counts per block in HBM.
        out_hbm_ref: Output tensor in HBM.
        accum_vmem_ref: FP32 accumulator tile in VMEM.
        gather_buf_vmem_ref: Double-buffered gather staging buffer in VMEM.
        packed_out_vmem_ref: Double-buffered packed BF16 output tile in VMEM.
        gather_sem_ref: DMA semaphores for gather buffer double-buffering.
        output_sem_ref: DMA semaphores for output writeout double-buffering.
        core_axis_name: Pallas mesh core axis name.
        subcore_axis_name: Pallas mesh subcore axis name.
        num_row_partitions: Number of row partitions.
        num_column_partitions: Number of column partitions.
        blocks_per_row_partition: Number of blocks assigned to each row partition.
        topk: Number of experts/routes per token.
        skip_compute: When True, benchmarks memory bandwidth without arithmetic.
    """
    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    col_start = pl.multiple_of(col_partition_id * col_size, _TOKEN_SUBCHUNK)
    metadata_block_size = topk * _TOKEN_BLOCK

    x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    def metadata_block_index(block_id):
        return (row_partition_id * blocks_per_row_partition + block_id,)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(blocks_per_row_partition,),
        in_specs=(
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((_TOKEN_SUBCHUNK,), metadata_block_index),
        ),
        out_specs=(),
    )
    def token_block_pipeline(
        route_metadata_ref,
        route_weights_ref,
        route_counts_ref,
        output_sem_ref_in,
    ):
        block_id = pl.program_id(0)
        global_block_id = row_partition_id * blocks_per_row_partition + block_id
        valid_route_count = route_counts_ref[...][0]
        fixed_two_routes = valid_route_count == 2 * _TOKEN_BLOCK
        num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

        # 1. Zero accumulator in VMEM
        if not skip_compute:
            def zero_column(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
                for token_row in range(_TOKEN_BLOCK):
                    accum_vmem_ref[token_row, col_slice] = zero

            plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(zero_column)

        def _issue_chunk_gather(slot: int, chunk_idx: int):
            start = chunk_idx * _TOKEN_SUBCHUNK
            meta = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            source_indices = jnp.bitwise_right_shift(
                jnp.bitwise_and(meta, _ROUTE_INDEX_MASK), 1
            )
            dma_gather = pltpu.make_async_copy(
                x_32b_hbm_ref.at[source_indices, pl.ds(col_start, col_size)],
                gather_buf_vmem_ref.at[slot],
                gather_sem_ref.at[slot],
            )
            dma_gather.start()

        def _compute_chunk(chunk_idx: int, slot: int):
            if skip_compute:
                return
            start = chunk_idx * _TOKEN_SUBCHUNK
            route_metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            source_indices = jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK)
            route_destinations = jnp.bitwise_right_shift(
                route_metadata, _ROUTE_INDEX_BITS
            )
            route_weights = route_weights_ref[pl.ds(start, _TOKEN_SUBCHUNK)]

            def fixed_two_route_col(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
                for pair_idx in range(8):
                    route_0 = pair_idx * 2
                    route_1 = route_0 + 1
                    val0_i32 = gather_buf_vmem_ref[slot, route_0, col_slice]
                    shift0 = jnp.where(jnp.bitwise_and(source_indices[route_0], 1) == 0, 16, 0)
                    val0_f32 = plsc.bitcast(
                        jnp.bitwise_and(jnp.left_shift(val0_i32, shift0), jnp.int32(-65536)),
                        jnp.float32,
                    ) * route_weights[route_0]

                    val1_i32 = gather_buf_vmem_ref[slot, route_1, col_slice]
                    shift1 = jnp.where(jnp.bitwise_and(source_indices[route_1], 1) == 0, 16, 0)
                    val1_f32 = plsc.bitcast(
                        jnp.bitwise_and(jnp.left_shift(val1_i32, shift1), jnp.int32(-65536)),
                        jnp.float32,
                    ) * route_weights[route_1]

                    dest = jnp.full_like(feature_indices, route_destinations[route_0])
                    plsc.addupdate_scatter(
                        accum_vmem_ref,
                        (dest, feature_indices),
                        val0_f32 + val1_f32,
                    )

            def generic_col(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(_TOKEN_SUBCHUNK, dtype=jnp.int32)
                for route_lane in range(_TOKEN_SUBCHUNK):
                    val_i32 = gather_buf_vmem_ref[slot, route_lane, col_slice]
                    shift = jnp.where(jnp.bitwise_and(source_indices[route_lane], 1) == 0, 16, 0)
                    val_f32 = plsc.bitcast(
                        jnp.bitwise_and(jnp.left_shift(val_i32, shift), jnp.int32(-65536)),
                        jnp.float32,
                    ) * route_weights[route_lane]
                    dest = jnp.full_like(feature_indices, route_destinations[route_lane])
                    route_valid = (start + route_lane < valid_route_count)
                    scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                    plsc.addupdate_scatter(
                        accum_vmem_ref,
                        (dest, feature_indices),
                        val_f32,
                        mask=scatter_mask,
                    )

            @pl.when(fixed_two_routes)
            def _run_fixed():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(fixed_two_route_col)

            @pl.when(jnp.logical_not(fixed_two_routes))
            def _run_generic():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(generic_col)

        # 2. Hardware loop using jax.lax.fori_loop with double-buffering ping-pong
        def _chunk_loop_body(chunk_idx, _):
            cur_slot = chunk_idx % 2
            next_slot = (chunk_idx + 1) % 2

            @pl.when(chunk_idx + 1 < num_route_chunks)
            def _prefetch_next():
                _issue_chunk_gather(slot=next_slot, chunk_idx=chunk_idx + 1)

            pltpu.make_async_copy(
                x_32b_hbm_ref.at[pl.ds(0, _TOKEN_SUBCHUNK), pl.ds(col_start, col_size)],
                gather_buf_vmem_ref.at[cur_slot],
                gather_sem_ref.at[cur_slot],
            ).wait()

            _compute_chunk(chunk_idx, cur_slot)
            return ()

        @pl.when(num_route_chunks > 0)
        def _run_block_chunks():
            _issue_chunk_gather(slot=0, chunk_idx=0)
            jax.lax.fori_loop(0, num_route_chunks, _chunk_loop_body, ())

        # 3. Output Stage: Overlapped double-buffered packing and async DMA output
        if not skip_compute:
            for token_subchunk in range(_NUM_TOKEN_SUBCHUNKS):
                out_slot = token_subchunk % 2
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

                if token_subchunk >= 2:
                    pltpu.make_async_copy(
                        packed_out_vmem_ref.at[out_slot],
                        out_32b_hbm_ref.at[pl.ds(0, _TOKEN_SUBCHUNK // 2), pl.ds(col_start, col_size)],
                        output_sem_ref_in.at[out_slot],
                    ).wait()

                def pack_columns(col_offset, row_start=accum_row_start, slot=out_slot):
                    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                    for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                        accum_row = row_start + token_lane
                        packed_bf16 = plsc.pack(
                            accum_vmem_ref[accum_row, col_slice],
                            accum_vmem_ref[accum_row + 1, col_slice],
                            format=plsc.PackFormat.INTERLEAVED,
                        )
                        packed_out_vmem_ref[slot, token_lane // 2, col_slice] = plsc.bitcast(
                            packed_bf16, jnp.uint32
                        )

                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(pack_columns)

                output_row = (global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK)
                output_row_packed = pl.multiple_of(output_row // 2, _TOKEN_SUBCHUNK // 2)

                copy = pltpu.make_async_copy(
                    packed_out_vmem_ref.at[out_slot],
                    out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(col_start, col_size),
                    ],
                    output_sem_ref_in.at[out_slot],
                )
                copy.start()

            for s in range(2):
                pltpu.make_async_copy(
                    packed_out_vmem_ref.at[s],
                    out_32b_hbm_ref.at[pl.ds(0, _TOKEN_SUBCHUNK // 2), pl.ds(col_start, col_size)],
                    output_sem_ref_in.at[s],
                ).wait()

    token_block_pipeline(
        route_metadata_hbm_ref,
        route_weights_hbm_ref,
        route_counts_hbm_ref,
        scratches=(output_sem_ref,),
    )


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
    """SparseCore Ragged Gather-Reduce v4 with Continuous Prefetching.

    Args:
        x: Input tensor of shape [num_source_rows, hidden_size] and dtype bfloat16.
        indices: 1D flat int32 indices tensor of shape [num_tokens * topk].
        topk_weights: 1D flat bfloat16 routing weights tensor of shape [num_tokens * topk].
        valid_rows_mask: 1D flat bool validity mask of shape [num_tokens * topk].
        reduce_group_size: Number of routes per token (topk).
        num_sparse_cores: Target number of physical SparseCores to utilize (default 1).
        skip_compute: If True, executes memory movement without arithmetic for profiling.

    Returns:
        Reduced output tensor of shape [num_tokens, hidden_size] and dtype bfloat16.
    """
    sc_info = pltpu.get_tpu_info().sparse_core
    if (sc_info is None or x.dtype != jnp.bfloat16
            or x.shape[-1] % pltpu.get_tpu_info().num_lanes != 0):
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        raise ValueError(f"{input_size=} must be divisible by {reduce_group_size=}")

    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))

    num_tokens = input_size // reduce_group_size
    target_num_sc_cores = (min(num_sparse_cores, sc_info.num_cores)
                           if num_sparse_cores is not None else 1)
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
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
    valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(destination_template, indices_blocks.shape)

    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(destination_blocks, route_order, axis=1)
    weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
    route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(jnp.int32)
    route_metadata = jnp.bitwise_or(
        route_indices,
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS),
    ).reshape(-1)
    route_weights = jnp.where(compact_valid, route_weights, 0).reshape(-1).astype(jnp.float32)
    route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)
    paired_destinations = route_destinations[:, :2 * _TOKEN_BLOCK].reshape(
        num_blocks, _TOKEN_BLOCK, 2
    )
    fixed_two_route_blocks = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.all(paired_destinations[:, :, 0] == paired_destinations[:, :, 1], axis=1),
    )
    needs_zero_weight_sentinel = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.logical_not(fixed_two_route_blocks),
    )
    route_counts += needs_zero_weight_sentinel.astype(jnp.int32)
    route_counts = jnp.broadcast_to(route_counts[:, None], (num_blocks, _TOKEN_SUBCHUNK)).reshape(-1)

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
            topk=reduce_group_size,
            skip_compute=skip_compute,
        ),
        out_type=jax.ShapeDtypeStruct((padded_num_tokens, hidden_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(
            pltpu.VMEM((_TOKEN_BLOCK, col_size), jnp.float32),  # accum_vmem_ref
            pltpu.VMEM((2, _TOKEN_SUBCHUNK, col_size), jnp.int32),  # gather_buf_vmem_ref
            pltpu.VMEM((2, _TOKEN_SUBCHUNK // 2, col_size), jnp.uint32),  # packed_out_vmem_ref
            pltpu.SemaphoreType.DMA((2,)),  # gather_sem_ref
            pltpu.SemaphoreType.DMA((2,)),  # output_sem_ref
        ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v4",
    )(x, route_metadata, route_weights, route_counts)

    return out[:num_tokens]
