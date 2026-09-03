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
"""Option D -- hoist `streamindex_topk`'s layer-invariant metadata above the
`lax.cond`, and above the layer loop.

**This variant edits the caller, not the Pallas kernel.**  `kernel()` --- the
Mosaic body --- is imported unchanged from
`vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`, and so is
`_topk_kernel`, the SparseCore body, from `...sparsecore_topk`.  Nothing below
this line changes a single instruction *inside* either kernel.  What changes is
the XLA graph the caller builds around them.

Two independent edits, measurable separately:

**(D1) the `lax.cond` goes away.**  `streamindex_topk` wraps its whole body in
`jax.lax.cond(jnp.max(seq_lens) // compression_ratio <= k, _fast_path,
_common_path, None)` when `enable_early_exit=True`, which
`mla_attention_op.py:57` sets.  XLA compiles each branch into a private
computation *per call site*, and CSE does not span computations, so the 21
indexer layers get 21 private copies of identical metadata.  R2 measures the
`conditional` control op at **6.591 us of device self time** per call --- self
time, i.e. excluding both branch bodies --- and its predicate `reduce_max.7` at
1.330 us.

D1 needs no new code at all: `enable_early_exit=False` already selects
`return _common_path(None)` (`streamindex_topk.py:1010`).  The production value
is a module constant, `STREAMIDX_ENABLE_EARLY_EXIT`.  `_fast_path` is dead at
the nightly shape --- `compression_ratio=1` and `k=2048`, so the predicate is
`max(seq_lens) <= 2048` against a 8704--9216 token context --- which `area6 E4`
and `area7`'s CSE dump independently confirm (`branch_1_fun` has zero
`hlo_stats` rows).

**(D2) the layer-invariant metadata is computed once.**  Everything derived
from `(seq_lens, cu_q_lens, distribution)` alone is identical in all 21 indexer
layers.  `precompute()` below computes it once; `streamindex_topk_d()` accepts
it and skips the per-layer rebuild.  R2 measures this bucket at 10.470 us/call
(8.172 metadata + 2.298 scalar prefetch), of which 20/21 is redundant.

D2 is a *superset* of what XLA can do for itself.  Once the cond is gone the 21
layers are in one computation and CSE is free to collapse the identical
metadata subgraphs without any help --- whether it actually does is the
question `--arm dcond21` answers, and `--arm d21` is the belt-and-braces arm
that does not depend on the answer.

What is deliberately *not* hoisted, because R2 shows it is genuinely per-layer
data:

  * the `-inf` `scores_init` fill (1.655 us).  The kernel carries
    `output_to_operand_aliasing={{}: (10, {})}`; the buffer is donated into the
    kernel's output, so each layer needs its own.
  * stage-B data shaping (4.637 us) --- the `bitcast_convert_type` of *this*
    layer's 2.5 MiB score matrix and the reshape of its output.
  * the indexer-weights layout copy and bf16->f32 convert (1.452 us) --- each
    layer has its own weights.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sc
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as _si

# Imported, not copied: the Mosaic kernel body and every pure helper are the
# incumbent's, byte for byte.
kernel = _si.kernel
prepare_q_inputs = _si.prepare_q_inputs
prepare_index_weights = _si.prepare_index_weights
_effective_row_lengths = _si._effective_row_lengths
align_to = _si.align_to
get_dtype_packing = _si.get_dtype_packing
MlaCase = _si.MlaCase
DEFAULT_VMEM_LIMIT_BYTES = _si.DEFAULT_VMEM_LIMIT_BYTES

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402


# ---------------------------------------------------------------------------
# The hoisted metadata.
# ---------------------------------------------------------------------------


class HoistedMeta(NamedTuple):
    """Everything `streamindex_topk` derives from `(seq_lens, cu_q_lens,
    distribution)` alone, i.e. everything that is the same in all 21 indexer
    layers of one forward pass.

    A `NamedTuple` so it is a pytree and crosses a `jax.jit` boundary without
    any registration.
    """

    # i32[b_pad] -- `_effective_row_lengths` already padded to SparseCore's
    # `b_pad = align_to(num_tokens, 16)`, so both the reduce and the pad move.
    eff_lengths_padded: jax.Array
    # i32[2] x3 -- the `start_end_seq_idx_ref` scalar-prefetch operand of each
    # of the three `run_scores_kernel` calls (decode-grouped, decode-mop-up,
    # mixed).
    start_end: tuple[jax.Array, jax.Array, jax.Array]
    # i32[3] / i32[2] -- the `sem_ids` and `bo_sz` scalar-prefetch operands.
    # Loop-carried scratch the kernel writes into SMEM; the *initial* value is
    # a constant and identical everywhere.
    sem_ids: jax.Array
    bo_sz: jax.Array
    # The three grids. Dynamic (they depend on `distribution`), so they are
    # traced scalars, not Python ints.
    grid: tuple[jax.Array, jax.Array, jax.Array]


def num_positions_for(
    *,
    page_indices_len: int,
    max_num_seqs: int,
    page_size: int,
    num_kv_pages_per_blocks,
    k: int,
) -> int:
    """`scores_to_reduce.shape[1]`, without building anything.

    Mirrors `streamindex_topk.py:603-605` (`num_sublanes_total`) and the
    `scores.shape[1] < k` pad. Static: it depends only on shapes.
    """
    pages_per_seq = page_indices_len // max_num_seqs
    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)
    return max(num_sublanes_total * 128, k)


def precompute(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
    decode_req_batch_size: int = 4,
) -> HoistedMeta:
    """Compute, once per forward pass, what the incumbent computes 21 times.

    Every line here is lifted verbatim from `_common_path` /
    `_effective_row_lengths` / `sparsecore_topk`; nothing is re-derived. The
    equality of the two is asserted by
    `test_option_d_equivalence.py::test_hoisted_metadata_equals_per_layer`.
    """
    decode_batch_end = (distribution[0] // decode_req_batch_size *
                        decode_req_batch_size)

    eff = _effective_row_lengths(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        num_tokens=num_tokens,
        num_positions=num_positions,
        compression_ratio=compression_ratio,
    )
    # `sparsecore_topk`'s own pad, moved up with it (sparsecore_topk.py:321-322).
    b_pad = _sc._align_to(num_tokens, _sc.LANES)
    eff_padded = jnp.pad(eff.astype(jnp.int32), (0, b_pad - num_tokens))

    return HoistedMeta(
        eff_lengths_padded=eff_padded,
        start_end=(
            jnp.stack([jnp.int32(0), decode_batch_end.astype(jnp.int32)]),
            jnp.stack(
                [decode_batch_end.astype(jnp.int32),
                 distribution[1].astype(jnp.int32)]),
            jnp.stack([
                distribution[1].astype(jnp.int32),
                distribution[2].astype(jnp.int32)
            ]),
        ),
        sem_ids=jnp.zeros((3, ), jnp.int32),
        bo_sz=jnp.full((2, ), -1, jnp.int32),
        grid=(
            decode_batch_end // decode_req_batch_size,
            distribution[1] - decode_batch_end,
            distribution[2] - distribution[1],
        ),
    )


# ---------------------------------------------------------------------------
# SparseCore entry point that takes pre-padded lengths.
# ---------------------------------------------------------------------------


def _sc_topk_direct_pre(
    scores: jax.Array,  # f32[b, n]
    k: int,
    lengths_padded: jax.Array,  # i32[b_pad]
    *,
    write_empty: bool = True,
) -> jax.Array:
    """`sparsecore_topk._sc_topk_direct`, with the `jnp.pad` hoisted out.

    Identical to the incumbent apart from taking `lengths` already padded. The
    kernel body `_topk_kernel` is the incumbent's, imported.
    """
    b, n = scores.shape
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    words = jax.lax.bitcast_convert_type(scores, jnp.int32)
    b_pad = _sc._align_to(b, _sc.LANES)
    assert lengths_padded.shape == (b_pad, ), (lengths_padded.shape, b_pad)

    slice_len = _sc._align_to(n, _sc.LANES)
    num_waves = _sc._cdiv(b, 32)

    mesh = _sc.plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = pl.kernel(
        functools.partial(
            _sc._topk_kernel,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
        ),
        out_type=jax.ShapeDtypeStruct((b * k, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((_sc.LANES * _sc.NUM_BUCKETS + _sc.LANES, ), jnp.int32),
            pltpu.VMEM((_sc.NUM_BUCKETS + _sc.LANES, ), jnp.int32),
            pltpu.VMEM((k + _sc.LANES, ), jnp.int32),
            pltpu.VMEM((b_pad, ), jnp.int32),
            pltpu.SemaphoreType.DMA((4, )),
        ),
        mesh=mesh,
        name=f"sc_topk_direct_b{b}_n{n}_k{k}",
    )(words.reshape(-1), lengths_padded)
    return out.reshape(b, k)


def _sparsecore_topk_pre(scores, k, lengths_padded, *, write_empty_rows=True):
    """`sparsecore_topk` fed from hoisted, pre-padded row lengths.

    Two paths, matching the incumbent's own split at `sparsecore_topk.py:409`:

    * `p == 1` -- the partition `_pick_partition` returns at the nightly shape,
      and the one this option measures. Takes `lengths` already padded, so the
      `jnp.pad` inside `_sc_topk_direct` is hoisted with it.
    * `p > 1` -- reached at the small-`b` shapes the equivalence tests use. The
      two-stage path derives `lengths_p` and `cand_lengths` from the *unpadded*
      lengths and re-pads per stage at a different `b`, so there is nothing
      left to hoist past the row lengths themselves. Delegate to the
      incumbent, fed from the hoisted lengths: correctness is then the
      incumbent's, and only the hoist of `_effective_row_lengths` is retained.
    """
    b, n = scores.shape
    p = _sc._pick_partition(b, n, k)
    if p == 1:
        return _sc_topk_direct_pre(scores,
                                   k,
                                   lengths_padded,
                                   write_empty=write_empty_rows)
    return _sc.sparsecore_topk(scores,
                               k,
                               row_lengths=lengths_padded[:b],
                               write_empty_rows=write_empty_rows)


# ---------------------------------------------------------------------------
# Option D's `streamindex_topk`.
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_req_batch_size",
    ),
)
def streamindex_topk_d(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    hoisted: HoistedMeta | None = None,
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
) -> jax.Array:
    """`streamindex_topk` with the `lax.cond` removed (D1) and, when `hoisted`
    is supplied, the layer-invariant metadata taken from the caller (D2).

    `hoisted=None` is the D1-only arm: same graph, metadata still rebuilt here,
    so whatever collapses across layers collapses because XLA's CSE chose to.
    """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block] * 3
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block
    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block] * 3
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
        if (page_size * bkv_p) % 128 != 0:
            raise ValueError("bkv_sz must be a multiple of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    num_tokens = q.shape[0]
    num_positions = max(num_sublanes_total * 128, k)

    if hoisted is None:
        # D1 only: still computed here, once per layer, but no longer inside a
        # `conditional` -- so XLA is at liberty to CSE it across layers.
        meta = precompute(
            seq_lens,
            cu_q_lens,
            distribution,
            num_tokens=num_tokens,
            num_positions=num_positions,
            compression_ratio=compression_ratio,
            decode_req_batch_size=decode_req_batch_size,
        )
    else:
        meta = hoisted

    def run_scores_kernel(
        scores_init,
        start_end,
        grid_extent,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        out_dtype,
        case=MlaCase.MIXED,
    ):
        """`streamindex_topk.run_scores_kernel`, with `start_end`, `sem_ids`,
        `bo_sz` and the grid extent supplied rather than rebuilt."""
        _, num_q_heads, head_dim = q.shape
        if seq_batch_size > 1:
            assert static_q_len == 1

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        grid = (grid_extent, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        scratch_shapes = [
            pltpu.VMEM((2, seq_batch_size, bkv_sz_per_kv_packing, kv_packing,
                        lkv_dim), cache_kv.dtype),
            pltpu.VMEM((2, seq_batch_size, bq_sz, num_q_heads, head_dim),
                       q.dtype),
            pltpu.VMEM((2, seq_batch_size, bq_sz, num_q_heads),
                       prepared_indexer_weights.dtype),
            pltpu.VMEM((2, seq_batch_size * bq_sz, num_sublanes_bkv, 128),
                       out_dtype),
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            start_end,  # HOISTED (was jnp.array([start, end]))
            meta.sem_ids,  # HOISTED (was jnp.zeros((3,), jnp.int32))
            meta.bo_sz,  # HOISTED (was jnp.full((2,), -1, jnp.int32))
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

    out_dtype = jnp.float32

    # Per-layer by necessity: donated into the kernel's output
    # (`output_to_operand_aliasing`), so this is NOT hoisted.
    scores_init = jnp.full((q.shape[0], num_sublanes_total, 128),
                           -jnp.inf,
                           dtype=out_dtype)

    scores = run_scores_kernel(
        scores_init,
        meta.start_end[0],
        meta.grid[0],
        static_q_len=1,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        seq_batch_size=decode_req_batch_size,
        out_dtype=out_dtype,
        case=MlaCase.DECODE,
    )
    scores = run_scores_kernel(
        scores,
        meta.start_end[1],
        meta.grid[1],
        static_q_len=1,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        seq_batch_size=1,
        out_dtype=out_dtype,
        case=MlaCase.DECODE,
    )
    scores = run_scores_kernel(
        scores,
        meta.start_end[2],
        meta.grid[2],
        static_q_len=None,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        seq_batch_size=1,
        out_dtype=out_dtype,
        case=MlaCase.MIXED,
    )

    scores = scores.reshape(q.shape[0], -1)
    if scores.shape[1] < k:
        scores = jnp.pad(scores, ((0, 0), (0, k - scores.shape[1])),
                         constant_values=-jnp.inf)
    if scores.dtype != jnp.float32:
        scores = scores.astype(jnp.float32)

    topk_idxs = _sparsecore_topk_pre(scores,
                                     k,
                                     meta.eff_lengths_padded,
                                     write_empty_rows=True)
    return topk_idxs[:q.shape[0], :k]
