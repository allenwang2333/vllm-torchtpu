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
"""TPU-Friendly StreamIndex Top-K kernel."""

import enum
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

Enum = enum.Enum
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
PCP_AXIS_NAME = "pcp"
DCP_AXIS_NAME = "dcp"


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


# --------------------------------------------------------------------------- #
# PCP (prefill context parallel) compressed-coordinate helpers.
#
# PCP shards the KV cache chunk-interleaved over `interleave_size` *uncompressed*
# tokens (see kernels/experimental/pcp_streaming_rpa/schedule.py). The indexer
# stores one compressed row per `compression_ratio` tokens, so the chunk width in
# compressed space is `interleave_c = interleave_size // compression_ratio` and
# global compressed position `g` is owned by rank `(g // interleave_c) % pcp_size`.
#
# `pcp_rank` is a traced i32 scalar throughout, so a single compiled program
# serves every rank; only `pcp_size` and `interleave_c` are static.
# --------------------------------------------------------------------------- #


def pcp_local_to_global(local_idx, pcp_rank, pcp_size: int, interleave_c: int):
    """Map a rank-local compressed index to its global compressed position.

    Monotonic in `local_idx`, so the l-th position this rank owns is exactly
    ``pcp_local_to_global(l, ...)``. That is what lets the kernel keep using a
    plain ``global_pos < kv_len`` test as the local bounds check.
    """
    if pcp_size == 1:
        return local_idx
    cycle_c = pcp_size * interleave_c
    return ((local_idx // interleave_c) * cycle_c + pcp_rank * interleave_c +
            local_idx % interleave_c)


def pcp_owner_rank(global_idx, pcp_size: int, interleave_c: int):
    """Which rank owns global compressed position `global_idx`."""
    if pcp_size == 1:
        return jnp.zeros_like(global_idx)
    return (global_idx // interleave_c) % pcp_size


def pcp_global_to_local(global_idx, pcp_size: int, interleave_c: int):
    """Inverse of `pcp_local_to_global`, evaluated on the owning rank.

    Takes no rank argument, and does not need one: the owner is recoverable
    from `global_idx` alone (`pcp_owner_rank`), and once the owner is fixed the
    local index is the same arithmetic everywhere. That is what lets the DCP
    merge convert a winner to the coordinate space of whichever rank holds it
    without being on that rank.
    """
    if pcp_size == 1:
        return global_idx
    cycle_c = pcp_size * interleave_c
    return (global_idx // cycle_c) * interleave_c + global_idx % interleave_c


def pcp_local_length(global_len, pcp_rank, pcp_size: int, interleave_c: int):
    """Count compressed positions ``< global_len`` owned by `pcp_rank`.

    Mirrors ``_local_page_valid_len`` in pcp_streaming_rpa/schedule.py.
    """
    if pcp_size == 1:
        return global_len
    cycle_c = pcp_size * interleave_c
    full = global_len // cycle_c
    rem = global_len - full * cycle_c
    return full * interleave_c + jnp.clip(rem - pcp_rank * interleave_c, 0,
                                          interleave_c)


def pcp_rank_as_data(pcp_axis_name: str, pcp_size: int):
    """This shard's index along `pcp_axis_name`, without `lax.axis_index`.

    `lax.axis_index` is the obvious spelling, but it lowers to `partition-id`,
    which XLA's SPMD partitioner rejects when torch_tpu recompiles a shard_map
    through its `jax.export` bridge ("partitionId instruction is not supported
    for SPMD partitioning since the meaning is ambiguous"). An all-to-all over
    a replicated iota says the same thing with an instruction the partitioner
    accepts: every device splits `[0..pcp_size)` into single elements and
    keeps the chunks addressed to it, so device `r` receives `r` from each
    peer and every lane of the result is `r`.

    Same trick as `pcp_streaming_rpa/wrapper.py`'s
    `_select_replicated_shard_for_pcp_rank`, on an iota rather than real data.
    Must be called from inside a `shard_map` body.
    """
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)
    exchanged = lax.all_to_all(ranks, pcp_axis_name, 0, 0, tiled=True)
    return lax.dynamic_slice_in_dim(exchanged, 0, 1, axis=0)[0]


class MlaCase(Enum):
    """Represents the different cases for MLA."""

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


def kernel(
    # Prefetch
    seq_lens_ref,  # Shape: [max_num_seqs], Memory: SMEM
    page_indices_ref,  # Shape: [max_num_seqs * pages_per_seq], Memory: SMEM
    cu_q_lens_ref,  # Shape: [max_num_seqs + 1], Memory: SMEM
    start_end_seq_idx_ref,  # Shape: [2], Memory: SMEM
    sem_ids_ref,  # Shape: [3], Memory: SMEM
    bo_sz_ref,  # Shape: [2], Memory: SMEM
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim], Memory: HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM
    # Scratch
    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim], Memory: VMEM
    bq_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads, head_dim], Memory: VMEM
    bq_weights_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads], Memory: VMEM
    scores_block_x2_ref,  # Shape: [2, seq_batch_size * bq_sz, num_sublanes_bkv, 128], Memory: VMEM
    sems,  # Shape: [4, 2, seq_batch_size], Memory: Semaphore
    *,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
):
    """Core kernel logic that operates on memory references."""

    _, num_q_heads, head_dim = q_hbm_ref.shape
    lkv_dim = cache_kv_hbm_ref.shape[-1]

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    num_sublanes_bkv = bkv_sz // 128

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * seq_batch_size
    batch_end_seq_idx = batch_start_seq_idx + seq_batch_size - 1

    q_lens = []
    kv_lens = []
    seq_lens = []
    for batch_idx in range(seq_batch_size):
        q_start = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
        q_end = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
        q_len = q_end - q_start
        q_lens.append(q_len)
        seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
        seq_lens.append(seq_len)
        kv_len = seq_len // compression_ratio
        kv_lens.append(kv_len)

    def wait_send_scores(bo_sem_idx):
        old_sz = bo_sz_ref[bo_sem_idx]

        @pl.when(old_sz >= 0)
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

    def start_send_scores(bo_sem_idx, sz, token_start, bkv_idx):
        # All sequences in the batch have the same sz. Issue a single DMA for the
        # entire batch.
        bo_sz_ref[bo_sem_idx] = sz
        sublane_start = bkv_idx * num_sublanes_bkv
        _async_copy(
            scores_block_x2_ref.at[bo_sem_idx,
                                   pl.ds(0, seq_batch_size * sz)],
            scores_hbm_ref.at[
                pl.ds(token_start, seq_batch_size * sz),
                pl.ds(sublane_start, num_sublanes_bkv),
            ],
            sems.at[2, bo_sem_idx, 0],
            wait=False,
        )

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            kv_packing,
            lkv_dim,
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

        for batch_idx in range(seq_batch_size):
            sem = sems.at[0, bkv_sem_idx, batch_idx]
            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]

            kv_p_start = bkv_idx * bkv_p
            page_indices_offset = (seq_idx +
                                   batch_idx) * pages_per_seq + kv_p_start

            if not wait:
                for i in range(bkv_p):
                    sz_per_kv_packing = page_size_per_kv_packing
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    safe_page_offset = jnp.minimum(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        jnp.maximum(0,
                                    max_hbm_pages - page_size_per_kv_packing),
                    )

                    _async_copy(
                        reshaped_cache_hbm_ref.at[pl.ds(
                            safe_page_offset, sz_per_kv_packing)],
                        bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                              sz_per_kv_packing)],
                        sem,
                        wait=False,
                    )
            else:
                dma_bkv_sz = bkv_p * page_size_per_kv_packing
                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        for batch_idx in range(seq_batch_size):
            sem = sems.at[1, bq_sem_idx, batch_idx]
            weights_sem = sems.at[3, bq_sem_idx, batch_idx]
            bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, batch_idx]
            bq_weights_vmem_ref = bq_weights_x2_ref.at[bq_sem_idx, batch_idx]

            q_len_start = cu_q_lens_ref[seq_idx + batch_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[seq_idx + batch_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))

            if not wait:
                _async_copy(
                    q_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    indexer_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_weights_vmem_ref.at[pl.ds(0, sz)],
                    weights_sem,
                    wait=False,
                )
            else:
                dst = bq_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst, dst, sem, wait=True)
                dst_w = bq_weights_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst_w, dst_w, weights_sem, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def load_bq(bq_sem_idx):
        data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].reshape(
            seq_batch_size, bq_sz * num_q_heads, head_dim)
        bqs = []
        for batch_idx in range(seq_batch_size):
            bqs.append(data[batch_idx])
        return bqs

    def load_bq_weights(bq_sem_idx):
        data = bq_weights_x2_ref.at[bq_sem_idx, :, :bq_sz][...]
        bq_weights = []
        for batch_idx in range(seq_batch_size):
            bq_weights.append(data[batch_idx])
        return bq_weights

    def load_bkv(bkv_sem_idx):
        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            bkv = bkv_x2_ref.at[bkv_sem_idx,
                                batch_idx, :bkv_sz_per_kv_packing][...]

            flat_bkv = bkv.reshape(-1, bkv.shape[-1])
            fp8_val = flat_bkv[:, :head_dim]
            fp8_val = pltpu.bitcast(fp8_val, jnp.float8_e4m3fn)
            scale_val = pltpu.bitcast(flat_bkv[:, head_dim:head_dim + 1].T,
                                      jnp.float8_e8m0fnu).astype(jnp.bfloat16)

            bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
            bkv_scales.append(scale_val)
        return bkvs, bkv_scales

    def process():
        kv_len_max = jnp.max(jnp.stack(kv_lens))
        num_bkv = jnp.maximum(1, cdiv(kv_len_max, bkv_sz))
        if static_q_len is None:
            assert seq_batch_size == 1
            num_bq = jnp.maximum(1, cdiv(q_lens[0], bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_scores(
            bq_vec,
            bkv_vec,
            scale_val_vec,
            bq_weights_vec,
            bq_pos_compressed_vec,
            bkv_idx,
        ):
            # Vectorized batched matmul
            bq_stacked = jnp.stack(
                bq_vec
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, head_dim)
            bkv_stacked = jnp.stack(
                bkv_vec)  # Shape: (seq_batch_size, bkv_sz, head_dim)
            scale_val_stacked = jnp.stack(
                scale_val_vec)  # Shape: (seq_batch_size, 1, bkv_sz)
            bq_weights_stacked = jnp.stack(
                bq_weights_vec)  # Shape: (seq_batch_size, bq_sz, num_q_heads)
            bq_pos_compressed_stacked = jnp.stack(
                bq_pos_compressed_vec)  # Shape: (seq_batch_size, bq_sz)

            # Compute in Registers
            s = jnp.einsum(
                "bnd,bmd->bnm",
                bq_stacked,
                bkv_stacked,
                preferred_element_type=jnp.float32,
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, bkv_sz)
            s = s.reshape(seq_batch_size, bq_sz, num_q_heads, bkv_sz)
            s = jnp.maximum(s, 0.0)
            s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
            s_summed = s.sum(axis=2)  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            s_summed = (s_summed * scale_val_stacked
                        )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            k_span = bkv_idx * bkv_sz + jnp.arange(
                bkv_sz, dtype=jnp.int32)  # Shape: (bkv_sz,)

            kv_lens_stacked = jnp.stack(kv_lens)  # Shape: (seq_batch_size,)
            valid_mask = (k_span[None, None, :] < kv_lens_stacked[:, None,
                                                                  None]
                          )  # Shape: (seq_batch_size, 1, bkv_sz)
            causal_mask = (k_span[None, None, :]
                           <= bq_pos_compressed_stacked[:, :, None]
                           )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            mask = jnp.logical_and(valid_mask, causal_mask)
            s_summed = jnp.where(mask, s_summed, -jnp.inf)

            return s_summed  # Shape: (seq_batch_size, bq_sz, bkv_sz)

        def compute_with_bq(bq_idx, _):

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            bq_pos_compressed_vec = []
            for batch_idx in range(seq_batch_size):
                q_pos = (seq_lens[batch_idx] - q_lens[batch_idx] +
                         bq_idx * bq_sz + jnp.arange(bq_sz, dtype=jnp.int32))
                bq_pos_compressed_vec.append(q_pos // compression_ratio)

            # Wait for cur bq if not ready yet
            wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)
            bq_vec = load_bq(bq_sem_idx)
            bq_weights_vec = load_bq_weights(bq_sem_idx)

            # If seq_batch_size > 1, static_q_len is always 1, therefore sz is always
            # 1 for all sequences within the batch.
            token_start = cu_q_lens_ref[batch_start_seq_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[batch_start_seq_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - token_start))

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)
                bkv_vec, scale_val_vec = load_bkv(bkv_sem_idx)

                s_summed = compute_scores(
                    bq_vec,
                    bkv_vec,
                    scale_val_vec,
                    bq_weights_vec,
                    bq_pos_compressed_vec,
                    bkv_idx,
                )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

                bo_sem_idx = sem_ids_ref[2]
                wait_send_scores(bo_sem_idx)
                scores_block_x2_ref[bo_sem_idx, ...] = s_summed.astype(
                    scores_block_x2_ref.dtype).reshape(seq_batch_size * bq_sz,
                                                       num_sublanes_bkv, 128)
                start_send_scores(bo_sem_idx, sz, token_start, bkv_idx)
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(start_seq_idx, 0, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_scores(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, head_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
) -> jax.Array:
    """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    num_kv_pages_per_block: number of kv pages to be processed in one block in
      the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    num_queries_per_block: number of queries to be processed in one block in the
      pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
      kv_len for all sequences in the batch. Defaults to False.
    decode_req_batch_size: maximum decode batch size per iteration.

  Returns:
    Top-K indices (in compressed space).
  """
    # Scale factors for the FP8 index cache format are packed directly inside
    # `cache_kv` along the width dimension, keeping HBM transactions fused.

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )

    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype

    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    q = prepare_q_inputs(q)
    lkv_dim = cache_kv.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        out_dtype,
        case=MlaCase.MIXED,
    ):
        _, num_q_heads, head_dim = q.shape
        # Only support batching for decode sequences.
        # TODO: support batching for decode sequences with speculative decoding
        # enabled, e.g. static_q_len = gamma + 1.
        if seq_batch_size > 1:
            assert static_q_len == 1

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        # If seq_batch_size > 1, caller already guaranteed that
        # end_seq_idx - start_seq_idx % seq_batch_size == 0.
        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(
                memory_space=pltpu.HBM),  # scores_init (aliased to out)
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # scores

        bkv_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bkv_buf_sz_per_kv_packing,
                kv_packing,
                lkv_dim,
            ),
            cache_kv.dtype,
        )
        bq_double_bufq = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
                head_dim,
            ),
            q.dtype,
        )
        bq_weights_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
            ),
            prepared_indexer_weights.dtype,
        )
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), out_dtype)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),  # (bq, bkv, bo) sem indices
            jnp.full((2, ), -1, jnp.int32),  # in-flight out DMA row counts
        )

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
        )

    def _common_path(_):
        out_dtype = jnp.float32

        # Pre-fill the output with -inf and alias it, so masked / unwritten
        # columns are already -inf.
        scores_init = jnp.full(
            (q.shape[0], num_sublanes_total, 128),
            -jnp.inf,
            dtype=out_dtype,
        )

        # TODO: we shall sort the sequences by length, so that multiple decode
        # sequences in one batch have similar lengths to reduce waste of compute.
        # With the same batch size, the longest sequence will determine number of
        # blocks to run computation for.
        decode_batch_end = (distribution[0] // decode_req_batch_size *
                            decode_req_batch_size)
        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=jnp.array(0),
            end_seq_idx=decode_batch_end,
            static_q_len=1,
            seq_batch_size=decode_req_batch_size,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )
        # Handle num_decode_seqs % decode_req_batch_size != 0 case.
        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[1],
            static_q_len=1,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )

        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            static_q_len=None,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.MIXED,
        )

        scores = scores.reshape(q.shape[0], -1)
        if scores.shape[1] < k:
            scores = jnp.pad(
                scores,
                ((0, 0), (0, k - scores.shape[1])),
                constant_values=-jnp.inf,
            )

        scores_to_reduce = scores
        if scores_to_reduce.dtype != jnp.float32:
            scores_to_reduce = scores_to_reduce.astype(jnp.float32)

        eff_row_lengths = _effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=q.shape[0],
            num_positions=scores_to_reduce.shape[1],
            compression_ratio=compression_ratio,
        )
        topk_idxs = sparsecore_topk(
            scores_to_reduce,
            k,
            row_lengths=eff_row_lengths,
            write_empty_rows=True,
        )

        return topk_idxs[:q.shape[0], :k]

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        seq_idx = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            q_abs_pos // compression_ratio,
        )
        max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        _fast_path,
        _common_path,
        None,
    )


def streamindex_topk_pcp(
    q: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    indexer_weights: jax.Array,  # [padded_num_tokens, num_q_heads]
    cache_kv: jax.Array,  # sharded on axis 0 over the PCP axis
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * virtual_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    token_order: jax.Array,  # i64/i32[padded_num_tokens]
    *,
    mesh,
    k: int,
    compression_ratio: int,
    pcp_size: int,
    interleave_size: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    pcp_axis_name: str = PCP_AXIS_NAME,
) -> jax.Array:
    """StreamIndex top-k over a PCP-sharded compressed KV cache.

    Top-k over a sharded KV axis is a global reduction, so each rank scores
    every query token against its own shard, and the per-rank winners are then
    merged. The merge is a set union, not an accumulation: PCP *partitions* the
    KV positions, so no two ranks hold a partial score for the same position.
    It is exact because a global winner is beaten by at most `k - 1` positions
    overall, hence by at most `k - 1` on its owning rank, so it always survives
    that rank's local top-k.

    Q is replicated rather than sharded because it is O(num_tokens) and does not
    grow with context, whereas ring-streaming the KV would move O(context) per
    layer. The candidate exchange is an all-to-all rather than an all-gather:
    query tokens are PCP-sharded too, so rank `r` only needs to send rank `r'`
    the candidates belonging to `r'`'s tokens -- a factor of `pcp_size` less
    traffic. At 1M context / `pcp_size=16` / `k=2048` that is the difference
    between ~268MB and ~17MB per layer.

    This is the cross-rank analogue of `sparsecore_topk`'s own two-stage
    subcore merge (sparsecore_topk.py:415-457), and the structural analogue of
    the LSE output the DCP attention kernels return for their merge.

    Args:
      q, indexer_weights: replicated, in natural request-major token order.
      cache_kv: this rank's shard of the compressed KV cache.
      page_indices: replicated *virtual* page ordinals. A virtual page spans
        `page_size * pcp_size` compressed positions, of which each rank holds
        `page_size`; the same ordinal therefore resolves against every rank's
        own shard, as in pcp_streaming_rpa/schedule.py.
      token_order: `token_order` from `build_pcp_rank_major_token_order`,
        mapping a rank-major packed slot to its natural token index (-1 pads).
      mesh: mesh containing `pcp_axis_name`.

    Returns:
      i32[padded_num_tokens // pcp_size, k] global compressed positions for this
      rank's own query tokens, in rank-major packed order. `-1` pads.
    """
    if pcp_size <= 1:
        raise ValueError(
            f"streamindex_topk_pcp requires pcp_size > 1, got {pcp_size}; "
            "use streamindex_topk for the unsharded case.")
    if pcp_axis_name not in mesh.axis_names:
        raise ValueError(
            f"mesh {mesh.axis_names} has no {pcp_axis_name!r} axis.")
    if mesh.shape[pcp_axis_name] != pcp_size:
        raise ValueError(f"pcp_size={pcp_size} does not match mesh axis "
                         f"{pcp_axis_name!r}={mesh.shape[pcp_axis_name]}.")
    if q.shape[0] % pcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={q.shape[0]} must be divisible by "
            f"pcp_size={pcp_size}.")
    if token_order.shape[0] != q.shape[0]:
        raise ValueError(
            f"token_order has {token_order.shape[0]} entries but q has "
            f"{q.shape[0]} tokens.")

    def _local(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
               distribution, token_order):
        pcp_rank = pcp_rank_as_data(pcp_axis_name, pcp_size)

        # A virtual page ordinal resolves to a physical block within this rank's
        # own shard; `cache_kv.shape[0]` is the local block count inside
        # shard_map. Mirrors schedule.py's `jnp.mod(block_tables, ...)`.
        local_page_indices = jnp.mod(page_indices,
                                     jnp.int32(cache_kv.shape[0]))

        idxs, scores = streamindex_topk(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            local_page_indices,
            cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=False,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
            return_scores=True,
        )

        # Natural request-major -> PCP rank-major packed order, so that the
        # all-to-all's contiguous split along axis 0 delivers each rank exactly
        # the tokens it owns.
        keep = (token_order >= 0)[:, None]
        take = jnp.maximum(token_order, 0).astype(jnp.int32)
        idxs = jnp.where(keep, jnp.take(idxs, take, axis=0), -1)
        scores = jnp.where(keep, jnp.take(scores, take, axis=0), -jnp.inf)

        # [T, k] -> [T // pcp_size, pcp_size * k].
        idxs = lax.all_to_all(idxs, pcp_axis_name, 0, 1, tiled=True)
        scores = lax.all_to_all(scores, pcp_axis_name, 0, 1, tiled=True)

        # Stage 2. Full width as the row length: -inf entries (unfilled local
        # slots, padding tokens) are never selected, so they need no masking.
        slots = sparsecore_topk(
            scores,
            k,
            row_lengths=jnp.full((scores.shape[0], ),
                                 scores.shape[1],
                                 dtype=jnp.int32),
            write_empty_rows=True,
        )
        merged = jnp.take_along_axis(idxs, jnp.maximum(slots, 0), axis=1)
        return jnp.where(slots >= 0, merged, -1)

    replicated = P()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            replicated,  # q
            replicated,  # indexer_weights
            P(pcp_axis_name),  # cache_kv
            replicated,  # seq_lens
            replicated,  # page_indices
            replicated,  # cu_q_lens
            replicated,  # distribution
            replicated,  # token_order
        ),
        out_specs=P(pcp_axis_name),
        check_vma=False,
    )(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
      distribution, token_order)


def dcp_topk_capacity(k: int, dcp_size: int) -> int:
    """Per-rank list width for `streamindex_topk_dcp`: 2x the mean occupancy.

    Mean occupancy is `k / dcp_size`, so headroom has to scale with the shard
    count rather than be a fixed width -- a constant that is generous at
    `dcp_size=16` is a hard truncation at `dcp_size=2`.

    2x is not arbitrary. `m_r = |topk & shard_r|` was swept over run-clustered
    winners (attention sinks, a local window, and contiguous runs of length L)
    and the spread turns out to be governed by the run length against the cycle
    width `interleave_size * dcp_size`, not by clustering as such: runs much
    longer than the cycle spread almost perfectly. At `interleave_size=4` --
    which is also the floor forced by `dsa_gather.TILE_SUBROWS` -- the worst
    `m_r` observed over all L with `k=2048, dcp_size=16` was 199 against a mean
    of 128, so 256 leaves 1.29x. Winners correlated with `position mod
    (interleave_size * dcp_size)` could still concentrate arbitrarily, which is
    why `streamindex_topk_dcp` reports `dropped` instead of trusting this.
    """
    if dcp_size <= 1:
        return k
    return min(k, 2 * k // dcp_size)


def _partition_winners_by_owner(merged, dcp_size: int, interleave_c: int,
                                capacity: int):
    """Compact a token's global top-k list into one padded list per owner rank.

    Args:
      merged: i32[num_rows, k] global compressed positions in `sparsecore_topk`
        order, `-1` padded.
      dcp_size, interleave_c: the shard layout.
      capacity: static width of each owner's list.

    Returns:
      `(lists, dropped)` where `lists` is i32[num_rows, dcp_size, capacity] of
      *rank-local* indices (`-1` padded) and `dropped` counts entries that did
      not fit `capacity`.
    """
    num_rows, _ = merged.shape
    valid = merged >= 0
    owner = jnp.where(valid,
                      pcp_owner_rank(merged, dcp_size, interleave_c), -1)
    local = pcp_global_to_local(jnp.maximum(merged, 0), dcp_size, interleave_c)

    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]
    # One dump column past the end absorbs everything that is masked off or
    # overflows, and is sliced away below.
    out = jnp.full((num_rows, dcp_size, capacity + 1), -1, jnp.int32)
    dropped = jnp.zeros((), jnp.int32)
    for r in range(dcp_size):
        mine = owner == r
        # Exclusive count of this rank's earlier winners, i.e. its slot.
        # `sparsecore_topk` returns each row unsorted, so this order carries no
        # score information and truncating at `capacity` sheds an arbitrary
        # subset, not the weakest entries. That is precisely why `dropped` is
        # returned instead of being swallowed: any overflow at all is a sizing
        # bug, not graceful degradation.
        slot = jnp.cumsum(mine, axis=1, dtype=jnp.int32) - mine
        keep = jnp.logical_and(mine, slot < capacity)
        dropped = dropped + jnp.sum(jnp.logical_and(mine, ~keep))
        out = out.at[rows, r,
                     jnp.where(keep, slot, capacity)].set(
                         jnp.where(keep, local, -1))
    return out[:, :, :capacity], dropped


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "k",
        "compression_ratio",
        "dcp_size",
        "interleave_size",
        "capacity",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "dcp_axis_name",
    ),
)
def streamindex_topk_dcp(
    q: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim], replicated
    indexer_weights: jax.Array,  # [padded_num_tokens, num_q_heads], replicated
    cache_kv: jax.Array,  # sharded on axis 0 over the DCP axis
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * virtual_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    mesh,
    k: int,
    compression_ratio: int,
    dcp_size: int,
    interleave_size: int,
    capacity: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    dcp_axis_name: str = DCP_AXIS_NAME,
) -> tuple[jax.Array, jax.Array]:
    """Exact global top-k over a DCP-sharded KV cache, delivered rank-local.

    The DCP twin of `streamindex_topk_pcp`. Same three-stage core -- score
    locally, all-to-all the candidates, merge with `sparsecore_topk` -- and
    exact for the same reason: DCP *partitions* the KV positions, so a global
    winner is beaten by at most `k - 1` positions overall and therefore by at
    most `k - 1` on its owner, and always survives that rank's local top-k.

    What differs is the query axis. Under PCP queries are sharded, so each rank
    wants the merged list for *its own* tokens and the exchange ends there.
    Under DCP queries are replicated and the KV is sharded, which inverts the
    question: every rank needs, for *every* token, the slice of that token's
    global top-k that it happens to hold. So the merge does not stop at the
    merged list -- it partitions each list by owning rank and sends the pieces
    home.

    Both halves of that are nearly free. After the first all-to-all the
    candidate axis is a concatenation over source ranks, so `slots // k` *is*
    the owning rank; no ownership scan is needed. And the return exchange is a
    second all-to-all of the already-compacted `capacity`-wide lists, which
    costs `capacity * 4` bytes per token -- against `k * 4` bytes per token to
    all-gather the merged list and have every rank redo the same compaction.
    At `k=2048, capacity=256` that is 1KB/token instead of 8KB/token, with the
    compaction done once instead of `dcp_size` times.

    Args:
      q, indexer_weights: replicated, in natural request-major token order.
        There is no rank-major packing here -- with replicated queries the
        split of tokens across merge ranks is arbitrary, so it is a plain
        contiguous split and needs no permutation tables.
      cache_kv: this rank's shard of the compressed KV cache.
      page_indices: replicated *virtual* page ordinals, resolved against each
        rank's own shard exactly as in `streamindex_topk_pcp`.
      capacity: static per-rank list width. Sized above `k / dcp_size` to
        absorb clustering; see `dropped` for what a too-small value costs.
      mesh: mesh containing `dcp_axis_name`.

    Returns:
      `(local_topk, dropped)`.
      `local_topk` is i32[padded_num_tokens, capacity] of this rank's own
      *local* cache indices for every token, `-1` padded, sharded over
      `dcp_axis_name` (global leading dim `dcp_size * padded_num_tokens`).
      Callers derive per-token counts from the `-1` padding; a row that is all
      `-1` means this rank owns none of that token's top-k, which the attention
      side must mask out of the LSE merge rather than attend over.
      `dropped` is i32[dcp_size], the number of winners that did not fit
      `capacity` on each rank. `sparsecore_topk` rows are unsorted, so an
      overflow discards an arbitrary subset of real winners rather than the
      weakest ones -- there is no graceful degradation here, and a non-zero
      `dropped` means the result is no longer the exact global top-k. Callers
      must treat it as a sizing failure.
    """
    if dcp_size <= 1:
        raise ValueError(
            f"streamindex_topk_dcp requires dcp_size > 1, got {dcp_size}; "
            "use streamindex_topk for the unsharded case.")
    if dcp_axis_name not in mesh.axis_names:
        raise ValueError(
            f"mesh {mesh.axis_names} has no {dcp_axis_name!r} axis.")
    if mesh.shape[dcp_axis_name] != dcp_size:
        raise ValueError(f"dcp_size={dcp_size} does not match mesh axis "
                         f"{dcp_axis_name!r}={mesh.shape[dcp_axis_name]}.")
    if q.shape[0] % dcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={q.shape[0]} must be divisible by "
            f"dcp_size={dcp_size}.")
    if capacity > k:
        raise ValueError(
            f"capacity={capacity} exceeds k={k}; a single rank can never own "
            "more than the whole top-k list.")
    interleave_c = interleave_size // compression_ratio

    def _local(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
               distribution):
        dcp_rank = pcp_rank_as_data(dcp_axis_name, dcp_size)
        local_page_indices = jnp.mod(page_indices,
                                     jnp.int32(cache_kv.shape[0]))

        idxs, scores = streamindex_topk(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            local_page_indices,
            cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=False,
            pcp_size=dcp_size,
            pcp_rank=dcp_rank,
            interleave_size=interleave_size,
            return_scores=True,
        )

        # Stage 1. [T, k] -> [T // dcp, dcp * k]; the candidate axis is now a
        # concatenation over source ranks, in source order.
        idxs = lax.all_to_all(idxs, dcp_axis_name, 0, 1, tiled=True)
        scores = lax.all_to_all(scores, dcp_axis_name, 0, 1, tiled=True)

        # Stage 2. Full width as the row length: -inf entries (unfilled local
        # slots, padding tokens) are never selected, so they need no masking.
        slots = sparsecore_topk(
            scores,
            k,
            row_lengths=jnp.full((scores.shape[0], ),
                                 scores.shape[1],
                                 dtype=jnp.int32),
            write_empty_rows=True,
        )
        merged = jnp.take_along_axis(idxs, jnp.maximum(slots, 0), axis=1)
        merged = jnp.where(slots >= 0, merged, -1)

        # Stage 3. Split each merged list by owner and send the pieces home.
        # [T // dcp, dcp, capacity] -> [T, capacity], the inverse of stage 1's
        # exchange, so tokens come back in their original order.
        lists, dropped = _partition_winners_by_owner(merged, dcp_size,
                                                     interleave_c, capacity)
        local_topk = lax.all_to_all(lists.reshape(lists.shape[0], -1),
                                    dcp_axis_name,
                                    1,
                                    0,
                                    tiled=True)
        return local_topk, jnp.reshape(dropped, (1, ))

    replicated = P()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            replicated,  # q
            replicated,  # indexer_weights
            P(dcp_axis_name),  # cache_kv
            replicated,  # seq_lens
            replicated,  # page_indices
            replicated,  # cu_q_lens
            replicated,  # distribution
        ),
        out_specs=(P(dcp_axis_name), P(dcp_axis_name)),
        check_vma=False,
    )(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
      distribution)
