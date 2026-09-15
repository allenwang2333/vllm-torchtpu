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
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import bref_override
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

MlaCase = config.MlaCase
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DCP_AXIS_NAME = "dcp"

cp_local_to_global = metadata.cp_local_to_global
cp_local_length = metadata.cp_local_length
cp_owner_rank = metadata.cp_owner_rank
cp_global_to_local = metadata.cp_global_to_local
cp_rank_as_data = metadata.cp_rank_as_data


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


def kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    metadata_ref,  # MetadataRef
    cp_rank_ref,  # [1]
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim], Memory: HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # [max_num_tokens, num_sublanes_total, 128]
    *,
    compression_ratio: int,
    static_q_len: int | None,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    case: MlaCase = MlaCase.MIXED,
    cp_size: int = 1,
    interleave_c: int = 1,
):
  """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
  buffer_cnt = buffer_count

  total_num_pages, page_size_per_kv_packing, kv_packing, kv_cache_width = (
      cache_kv_hbm_ref.shape
  )
  page_size = page_size_per_kv_packing * kv_packing

  _, num_q_heads, head_dim = q_hbm_ref.shape

  max_num_seqs = seq_lens_ref.shape[0]
  num_page_indices = page_indices_ref.shape[0]

  pages_per_seq = num_page_indices // max_num_seqs

  bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
  bkv_sz = bkv_sz_per_kv_packing * kv_packing
  num_sublanes_bkv = bkv_sz // 128
  shift_comp = (compression_ratio - 1).bit_length()

  q_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads, head_dim),
      lambda p_id: (
          metadata_ref.batch_tile_idx[p_id] * seq_batch_size * bq_sz,
          metadata_ref.bq_idx[p_id],
          0,
      ),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  weights_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads),
      lambda p_id: (
          metadata_ref.batch_tile_idx[p_id] * seq_batch_size * bq_sz,
          metadata_ref.bq_idx[p_id],
          0,
      ),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  kv_spec = pl.BlockSpec(
      (
          pl.BoundedSlice(seq_batch_size),
          bkv_sz_per_kv_packing,
          kv_packing,
          kv_cache_width,
      ),
      lambda p_id: (
          metadata_ref.batch_tile_idx[p_id],
          0,
          metadata_ref.bkv_idx[p_id],
          0,
      ),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  out_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_sublanes_bkv, 128),
      lambda p_id: (
          metadata_ref.batch_tile_idx[p_id] * seq_batch_size * bq_sz,
          metadata_ref.bq_idx[p_id],
          metadata_ref.bkv_idx[p_id],
          0,
      ),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  q_alloc = bref_override.StreamIndexQBufferedRef.input(
      spec=q_spec,
      dtype_or_type=q_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
  )
  weights_alloc = bref_override.StreamIndexQBufferedRef.input(
      spec=weights_spec,
      dtype_or_type=indexer_weights_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
  )
  kv_alloc = bref_override.StreamIndexKVBufferedRef.input(
      spec=kv_spec,
      dtype_or_type=cache_kv_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bkv_p=bkv_p,
      page_size_per_kv_packing=page_size_per_kv_packing,
      kv_packing=kv_packing,
      pages_per_seq=pages_per_seq,
      seq_batch_size=seq_batch_size,
  )
  o_alloc = bref_override.StreamIndexOBufferedRef.output(
      spec=out_spec,
      dtype_or_type=scores_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      bq_sz=bq_sz,
      num_sublanes=num_sublanes_bkv,
      seq_batch_size=seq_batch_size,
  )

  def compute_scores(
      bq_vec,
      bkv_vec,
      scale_val_vec,
      bq_weights_vec,
      bq_pos_compressed_vec,
      bkv_idx,
      batch_start_seq_idx,
      cp_rank,
  ):
    assert len(bq_vec) == seq_batch_size
    assert len(bkv_vec) == seq_batch_size
    assert len(scale_val_vec) == seq_batch_size
    assert len(bq_weights_vec) == seq_batch_size
    assert len(bq_pos_compressed_vec) == seq_batch_size
    ret = []

    for batch_idx in range(seq_batch_size):
      bq = bq_vec[batch_idx].reshape(-1, head_dim)
      bkv = bkv_vec[batch_idx]
      scale_val = scale_val_vec[batch_idx]
      bq_weights = bq_weights_vec[batch_idx]
      bq_pos_compressed = bq_pos_compressed_vec[batch_idx]

      s = jnp.einsum(
          "nd,md->nm",
          bq,
          bkv,
          preferred_element_type=jnp.float32,
      )
      s = s.reshape(-1, num_q_heads, s.shape[-1])
      s = jnp.maximum(s, 0.0)
      s = s * bq_weights.astype(jnp.float32)[:, :, None]
      s_summed = s.sum(axis=1)
      s_summed = s_summed * scale_val
      k_local = bkv_idx * bkv_sz + lax.broadcasted_iota(
          jnp.int32, s_summed.shape, 1
      )
      k_span = cp_local_to_global(k_local, cp_rank, cp_size, interleave_c)
      seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
      kv_len = seq_len >> shift_comp

      valid_mask = k_span < kv_len
      causal_mask = k_span <= bq_pos_compressed[:, None]
      mask = jnp.logical_and(valid_mask, causal_mask)
      s_summed = jnp.where(mask, s_summed, -jnp.inf)
      ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
    return jnp.concatenate(ret, axis=0)

  def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem):
    p_id = pl.program_id(0)
    batch_start_seq_idx = metadata_ref.batch_tile_idx[p_id]
    bq_idx = metadata_ref.bq_idx[p_id]
    bkv_idx = metadata_ref.bkv_idx[p_id]
    cp_rank = cp_rank_ref[0]

    q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
    bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
    w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
    bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

    bq_pos_compressed_vec = []
    for batch_idx in range(seq_batch_size):
      s_l = seq_lens_ref[batch_start_seq_idx + batch_idx]
      q_s = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
      q_e = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
      q_l = q_e - q_s
      q_pos = s_l - q_l + bq_idx * bq_sz + jnp.arange(bq_sz, dtype=jnp.int32)
      bq_pos_compressed_vec.append(q_pos >> shift_comp)

    bkvs = []
    bkv_scales = []
    for batch_idx in range(seq_batch_size):
      bkv = bkv_vmem[batch_idx, :bkv_sz_per_kv_packing][...]
      flat_bkv = bkv.reshape(-1, bkv.shape[-1])
      fp8_val = pltpu.bitcast(flat_bkv[:, :head_dim], jnp.float8_e4m3fn)
      scale_val = pltpu.bitcast(
          flat_bkv[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
      ).astype(jnp.bfloat16)
      bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
      bkv_scales.append(scale_val)

    scores = compute_scores(
        bq_vec,
        bkvs,
        bkv_scales,
        bq_weights_vec,
        bq_pos_compressed_vec,
        bkv_idx,
        batch_start_seq_idx,
        cp_rank,
    )
    scores_vmem[...] = scores

  @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
  def _run_pipeline(final_allocs):
    q_buf, weights_buf, kv_buf, o_buf = final_allocs
    pipeline_fn = pltpu.emit_pipeline(
        step_body,
        grid=(metadata_ref.num_steps[0],),
        in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
        out_specs=o_buf.spec,
    )
    pipeline_fn(
        (q_hbm_ref, cu_q_lens_ref, metadata_ref),
        (indexer_weights_hbm_ref, cu_q_lens_ref, metadata_ref),
        (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
        (scores_hbm_ref, cu_q_lens_ref, metadata_ref),
        allocations=(q_buf, weights_buf, kv_buf, o_buf),
    )

  _run_pipeline()

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
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
) -> jax.Array:  # i32[num_tokens]
  """Visible compressed KV positions per token row of the score matrix.

  Under context parallelism the score matrix is rank-local, so the row length
  is the number of globally-visible positions that *this* rank owns.
  """
  max_num_seqs = seq_lens.shape[0]
  num_seqs = distribution[2]
  token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
  seq_mask = jnp.where(
      jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False
  )
  seq_ids = jnp.sum(seq_mask, axis=1)

  q_start = cu_q_lens[seq_ids]
  q_len = cu_q_lens[seq_ids + 1] - q_start
  seq_len = seq_lens[seq_ids]
  token_pos = seq_len - q_len + (token_ids - q_start)
  kv_len = seq_len // compression_ratio
  visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
  visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
  valid = token_ids < cu_q_lens[num_seqs]
  return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        "cp_size",
        "interleave_size",
        "return_scores",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, head_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
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
    buffer_count: buffer count for the pallas kernel. This is a tuple of
      (decode, prefill, mixed) cases or a single integer. Defaults to (4, 3, 3).
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
      kv_len for all sequences in the batch. Defaults to False.
    decode_req_batch_size: maximum decode batch size per iteration.
    cp_size: number of context-parallel ranks the compressed KV cache is sharded
      over. 1 (default) means no sharding and the whole CP path compiles away.
      When > 1, `cache_kv` / `page_indices` describe only this rank's shard, and
      the returned top-k is *local* -- correct only after the cross-rank merge
      in `streamindex_topk_dcp`.
    cp_rank: this rank's index in the CP group. Traced, so one compiled program
      serves every rank.
    interleave_size: CP chunk-interleave width in uncompressed tokens. Must be a
      multiple of `compression_ratio`.
    return_scores: also return the score of each selected position, which is
      what makes a cross-rank merge possible. A local top-k over a sharded KV
      axis is not the global top-k, so the merge needs the scores the same way
      the DCP attention kernels need an LSE output.

  Returns:
    Top-K indices in global compressed space, or `(indices, scores)` if
    `return_scores`. Unfilled slots are `-1` with score `-inf`.
  """
  # Scale factors for the FP8 index cache format are packed directly inside
  # `cache_kv` along the width dimension, keeping HBM transactions fused.

  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size > 1:
    if interleave_size % compression_ratio != 0:
      raise ValueError(
          f"interleave_size ({interleave_size}) must be a multiple of "
          f"compression_ratio ({compression_ratio}) for the CP chunk "
          "boundary to fall on a compressed-row boundary."
      )
    if enable_early_exit:
      raise NotImplementedError(
          "enable_early_exit is not supported with cp_size > 1."
      )
  if enable_early_exit and return_scores:
    raise NotImplementedError(
        "return_scores is not supported with enable_early_exit."
    )
  interleave_c = interleave_size // compression_ratio if cp_size > 1 else 1

  if num_kv_pages_per_block is None or num_queries_per_block is None:
    raise ValueError(
        "num_kv_pages_per_block and num_queries_per_block must be specified."
    )
  if (
      compression_ratio < 1
      or (compression_ratio & (compression_ratio - 1)) != 0
  ):
    raise ValueError("compression_ratio must be a power of 2.")

  if isinstance(num_kv_pages_per_block, int):
    num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
  else:
    num_kv_pages_per_blocks = num_kv_pages_per_block

  if isinstance(num_queries_per_block, int):
    num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
  else:
    num_queries_per_blocks = num_queries_per_block

  if buffer_count is None:
    buffer_counts = list(DEFAULT_BUFFER_COUNT)
  elif isinstance(buffer_count, int):
    buffer_counts = [buffer_count for _ in range(3)]
  else:
    buffer_counts = list(buffer_count)

  if len(buffer_counts) != 3:
    raise ValueError(
        "buffer_count must be a 3-tuple or a single integer, got"
        f" {buffer_count}"
    )

  max_num_seqs = seq_lens.shape[0]

  original_dtype = q.dtype

  prepared_indexer_weights = prepare_index_weights(
      indexer_weights, original_dtype
  )
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
          " of 128."
      )

  num_sublanes_total = max(
      align_to(pages_per_seq, bkv_p) * page_size // 128
      for bkv_p in num_kv_pages_per_blocks
  )

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
      buffer_count,
      seq_batch_size,
      out_dtype,
      case=MlaCase.MIXED,
  ):
    max_num_tokens, num_q_heads, head_dim = q.shape
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
    bkv_sz = page_size * bkv_p

    meta = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=pages_per_seq,
        compression_ratio=compression_ratio,
        static_q_len=static_q_len,
        seq_batch_size=seq_batch_size,
        page_size=page_size,
        max_num_tokens=max_num_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
    )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [
        hbm_spec,  # q
        hbm_spec,  # prepared_indexer_weights
        hbm_spec,  # cache_kv
        hbm_spec,  # scores_init (aliased to out)
    ]
    out_specs = hbm_spec

    scratch_shapes = []

    scalar_prefetches = (
        seq_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
        meta,
        jnp.asarray([cp_rank], jnp.int32),
    )

    num_scalar_prefetch = len(scalar_prefetches)
    num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))

    scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
    pallas_kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
                buffer_count=buffer_count,
                case=case,
                cp_size=cp_size,
                interleave_c=interleave_c,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={num_flat_prefetches + 3: 0},
            name=scope_name,
        )
    )
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
    decode_batch_end = (
        distribution[0] // decode_req_batch_size * decode_req_batch_size
    )
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
        buffer_count=buffer_counts[0],
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
        buffer_count=buffer_counts[0],
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
        buffer_count=buffer_counts[2],
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
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
    )
    topk_idxs = sparsecore_topk(
        scores_to_reduce,
        k,
        row_lengths=eff_row_lengths,
        write_empty_rows=True,
    )
    topk_idxs = topk_idxs[: q.shape[0], :k]

    if cp_size == 1 and not return_scores:
      return topk_idxs

    # `sparsecore_topk` suffix-pads with -1; clamp before the gather so the
    # negative index cannot wrap onto the last column, then restore -inf.
    filled = topk_idxs >= 0
    topk_scores = jnp.take_along_axis(
        scores_to_reduce, jnp.maximum(topk_idxs, 0), axis=1
    )
    topk_scores = jnp.where(filled, topk_scores, -jnp.inf)
    # Columns are rank-local; callers (and the merge) want global positions.
    global_idxs = jnp.where(
        filled,
        cp_local_to_global(topk_idxs, cp_rank, cp_size, interleave_c),
        -1,
    )
    if not return_scores:
      return global_idxs
    return global_idxs, topk_scores

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


def _select_owned_winners(
    merged: jax.Array,
    dcp_size: int,
    interleave_c: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
  """Keep the winners this rank owns, left-packed as rank-local indices.

  Every rank filters the identical `merged` global top-k list with its own
  `dcp_rank`. Because `cp_owner_rank` partitions `[0, total_kv_positions)` onto
  `range(dcp_size)`, the results across all ranks form an exact partition of
  the global top-k: no winner is duplicated and none is lost. This is an
  arithmetic identity regardless of buffer occupancy, which is why this requires
  no capacity bounding and never truncates -- a rank owns at most all `k`
  winners, and the output width is `k`.

  Args:
    merged: i32[num_rows, k] global compressed positions in `sparsecore_topk`
      order, `-1` padded. Identical on every rank.
    dcp_size: number of context-parallel ranks.
    interleave_c: chunk-interleave width in compressed tokens.
    dcp_rank: i32 scalar, this shard's index along the DCP axis.

  Returns:
    i32[num_rows, k] of rank-local cache indices for the winners this rank owns,
    packed into a prefix and `-1` padded.
  """
  num_rows, width = merged.shape
  mine = (merged >= 0) & (
      cp_owner_rank(merged, dcp_size, interleave_c) == dcp_rank
  )
  local = cp_global_to_local(jnp.maximum(merged, 0), dcp_size, interleave_c)

  # Exclusive prefix count of owned winners gives each entry's destination slot.
  # At most `width` entries can be `mine`, so `slot <= width - 1`.
  slot = jnp.cumsum(mine, axis=1, dtype=jnp.int32) - mine
  rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

  # An extra column past the end (`width`) absorbs entries owned by other ranks;
  # they all write -1 to the dump column, which is sliced off at the end.
  out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
  out = out.at[rows, jnp.where(mine, slot, width)].set(
      jnp.where(mine, local, -1)
  )
  return out[:, :width]


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "k",
        "compression_ratio",
        "dcp_size",
        "interleave_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
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
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    dcp_axis_name: str = DCP_AXIS_NAME,
) -> jax.Array:
  """Exact global top-k over a DCP-sharded KV cache, delivered rank-local.

  Architecture:
    1. Local Scoring: Every rank scores the replicated query tokens against its
       own local KV cache shard via `streamindex_topk`, producing candidate
       indices and scores of shape `[padded_num_tokens, k]`. Because DCP
       partitions the KV cache, a true global top-k winner is beaten by at most
       `k - 1` positions globally, and therefore by at most `k - 1` positions
       on its owner rank. Thus, every true global winner is guaranteed to
       survive into its owner's local candidate list.
    2. Candidate All-Gather: An `all_gather` along the candidate axis (axis 1)
       exchanges candidate indices and scores across all `dcp_size` ranks:
       `[T, k] -> [T, dcp_size * k]`.
    3. SparseCore Merge: Each rank runs `sparsecore_topk` over the concatenated
       `[T, dcp_size * k]` scores to select the exact top-k global winners:
       `[T, dcp_size * k] -> [T, k]`.
    4. Rank-Local Filtering: Each rank filters the merged winners with
       `_select_owned_winners`, keeping only the positions it holds
       (`cp_owner_rank(g) == dcp_rank`), mapping them to local shard
       coordinates (`cp_global_to_local`), and packing them into a `-1` padded
       prefix.

  By gathering candidates along axis 1 upfront rather than partitioning tokens
  over an `all_to_all`, all communication is consolidated into a single upfront
  collective phase, avoiding the barrier latency of a multi-phase pipeline and
  preserving the token axis `T` unpartitioned.

  Args:
    q: replicated query tokens, in natural request-major order
      `[padded_num_tokens, num_q_heads, head_dim]`.
    indexer_weights: replicated query weights `[padded_num_tokens,
      num_q_heads]`.
    cache_kv: this rank's shard of the compressed KV cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: replicated *virtual* page ordinals, resolved against each
      rank's own shard by `local = virtual % num_local_pages`.
    cu_q_lens: the cumulative sum of the effective query lengths.
    distribution: (i, j, k) partition indices.
    mesh: mesh containing `dcp_axis_name`.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    dcp_size: number of context-parallel ranks.
    interleave_size: chunk-interleave width in uncompressed tokens.
    num_kv_pages_per_block: number of kv pages per block in pallas kernel.
    num_queries_per_block: number of queries per block in pallas kernel.
    buffer_count: buffer count for pallas kernel.
    vmem_limit_bytes: vmem limit for pallas kernel.
    decode_req_batch_size: maximum decode batch size per iteration.
    dcp_axis_name: name of the DCP mesh axis.

  Returns:
    i32[padded_num_tokens, k] of this rank's own *local* cache indices for
    every token, packed into a prefix and `-1` padded, sharded over
    `dcp_axis_name` (global leading dim `dcp_size * padded_num_tokens`).
    Callers derive per-token counts from the `-1` padding; a row that is all
    `-1` means this rank owns none of that token's top-k, which the attention
    side must mask out of the LSE merge rather than attend over. Rows are
    `k` wide but hold `k / dcp_size` entries on average, so a caller that
    gathers the full width pays `dcp_size`x more than it needs to; the
    attention side should bound its gather by the per-row count.
  """
  if dcp_size <= 1:
    raise ValueError(
        f"streamindex_topk_dcp requires dcp_size > 1, got {dcp_size}; "
        "use streamindex_topk for the unsharded case."
    )
  if dcp_axis_name not in mesh.axis_names:
    raise ValueError(f"mesh {mesh.axis_names} has no {dcp_axis_name!r} axis.")
  if mesh.shape[dcp_axis_name] != dcp_size:
    raise ValueError(
        f"dcp_size={dcp_size} does not match mesh axis"
        f" {dcp_axis_name!r}={mesh.shape[dcp_axis_name]}."
    )
  if q.shape[0] % dcp_size != 0:
    raise ValueError(
        f"padded_num_tokens={q.shape[0]} must be divisible by"
        f" dcp_size={dcp_size}."
    )
  interleave_c = interleave_size // compression_ratio

  def _local(
      q,
      indexer_weights,
      cache_kv,
      seq_lens,
      page_indices,
      cu_q_lens,
      distribution,
  ):
    dcp_rank = cp_rank_as_data(dcp_axis_name, dcp_size)
    local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0]))

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
        buffer_count=buffer_count,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=False,
        cp_size=dcp_size,
        cp_rank=dcp_rank,
        interleave_size=interleave_size,
        return_scores=True,
    )

    # Stage 1. All-gather candidate indices and scores along axis 1.
    # [T, k] -> [T, dcp * k]; candidates concatenated over source ranks.
    idxs = lax.all_gather(idxs, dcp_axis_name, axis=1, tiled=True)
    scores = lax.all_gather(scores, dcp_axis_name, axis=1, tiled=True)

    # Stage 2. Full width as the row length: -inf entries (unfilled local
    # slots, padding tokens) are never selected, so they need no masking.
    slots = sparsecore_topk(
        scores,
        k,
        row_lengths=jnp.full(
            (scores.shape[0],), scores.shape[1], dtype=jnp.int32
        ),
        write_empty_rows=True,
    )
    merged = jnp.take_along_axis(idxs, jnp.maximum(slots, 0), axis=1)
    merged = jnp.where(slots >= 0, merged, -1)

    return _select_owned_winners(merged, dcp_size, interleave_c, dcp_rank)

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
      out_specs=P(dcp_axis_name),
      check_vma=False,
  )(
      q,
      indexer_weights,
      cache_kv,
      seq_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )
