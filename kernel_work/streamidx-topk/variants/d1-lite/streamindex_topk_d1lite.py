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
"""D1-lite: keep `streamindex_topk`'s `lax.cond` and its `_fast_path`, but
move the branch-invariant metadata and scalar-prefetch constants *out* of the
branches so XLA's CSE can span the 21 indexer layers.

Why this exists
---------------
Option D measured that deleting the `lax.cond` outright ("D1") is worth
-17.05 us/layer at 21 layers, and decomposed it:

    cond removal itself   -6.766 us/layer
    metadata              -8.533 us/layer   <- cross-layer CSE
    prefetch constants    -2.221 us/layer   <- cross-layer CSE

i.e. ~63% of D1's win is *not* the branch: it is that XLA compiles each
`lax.cond` branch into its own HloComputation **per call site**, and CSE does
not span computations, so 21 layers carry 21 private copies of identical
metadata (294 `metadata` rows and 63 `prefetch` rows in the 21-layer profile,
collapsing to 34 and 3 once the cond is gone).

But D1 also deletes the fast path, and in the regime where the fast path fires
(`max(seq_lens) // compression_ratio <= k`, i.e. ctx 1024/2048 at k=2048) the
incumbent finishes in ~7.7 us with no stage A at all while D1 runs the whole
scoring path, ~7.5x slower.  D1-lite is the attempt to take the CSE-able 63%
without giving up the fast path.

What is hoisted, and why it is legal
------------------------------------
Everything hoisted is a pure function of `(seq_lens, cu_q_lens, distribution)`
plus static shape constants -- none of it depends on which branch is taken, and
none of it is a function of `q`, `indexer_weights` or `cache_kv` (which *are*
per-layer).  Concretely, matching the buckets Option D measured:

  * **metadata** -- `_effective_row_lengths(...)`
    (`streamindex_topk.py:482-506`), called from `_common_path` at
    `streamindex_topk.py:823-830`, plus `sparsecore_topk`'s own
    `jnp.pad(row_lengths, (0, b_pad - b))` (`sparsecore_topk.py:322`) and
    `decode_batch_end = distribution[0] // decode_req_batch_size *
    decode_req_batch_size` (`streamindex_topk.py:755-756`).
  * **prefetch** -- the three scalar-prefetch constants built inside
    `run_scores_kernel` at `streamindex_topk.py:694-701`:
    `jnp.array([start_seq_idx, end_seq_idx], jnp.int32)`, `jnp.zeros((3,),
    jnp.int32)` (the bq/bkv/bo semaphore indices) and `jnp.full((2,), -1,
    jnp.int32)` (the in-flight out-DMA row counts).

`sem_ids` and `bo_sz` are written by the Mosaic body
(`streamindex_topk.py:140,365,393,416`).  Sharing one hoisted HBM buffer across
call sites is nevertheless safe: scalar prefetch operands are copied into each
launch's SMEM, and Option D's `dcond21` arm already ran 63 `pallas_call`s off 3
shared buffers with its equivalence tests passing.

What is deliberately *not* hoisted, because it is genuinely per-layer:
the `-inf` `scores_init` fill (donated into the kernel output via
`input_output_aliases`), the stage-B `bitcast_convert_type`/reshape of this
layer's score matrix, and the indexer-weights layout copy + bf16->f32 convert.

Three hoist modes (`hoist=`)
----------------------------
``"none"``
    Control.  Metadata is built inside `_common_path`, exactly where the
    incumbent builds it.  Structurally the incumbent; used to show that this
    module's port is faithful (it should reproduce the `inc` arm).

``"plain"``
    The metadata is computed unconditionally at the top level of the jitted
    function and closed over by `_common_path`, so it becomes an *operand* of
    the outer `conditional` and lives in the entry computation where CSE can
    span layers.  Cost: the fast path now pays for metadata it used to skip.

``"cond"``
    The hoisted computation is itself wrapped in an inner
    `lax.cond(fast_pred, zeros, compute)`.  The fast path skips the work, and
    because the inner conditional's operands (`seq_lens`, `cu_q_lens`,
    `distribution`) and both its branch bodies are identical in every layer,
    the whole `conditional` instruction is a CSE candidate across layers.  This
    is the mode that can both keep the fast path's ~7.7 us and collapse 21
    copies to 1.

`enable_early_exit=False` deletes the outer cond entirely and is the D1 arm.

Nothing below changes a single instruction inside either Pallas kernel: the
Mosaic body `kernel`, the SparseCore body `_topk_kernel` and every pure helper
are imported from the incumbent modules by reference.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sc
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as _si

# Imported, not copied.
kernel = _si.kernel
prepare_q_inputs = _si.prepare_q_inputs
prepare_index_weights = _si.prepare_index_weights
_effective_row_lengths = _si._effective_row_lengths
align_to = _si.align_to
MlaCase = _si.MlaCase
DEFAULT_VMEM_LIMIT_BYTES = _si.DEFAULT_VMEM_LIMIT_BYTES

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

HOIST_MODES = ("none", "plain", "cond")


# ---------------------------------------------------------------------------
# The hoisted values.
# ---------------------------------------------------------------------------


class HoistedMeta(NamedTuple):
    """Everything `streamindex_topk` derives from `(seq_lens, cu_q_lens,
    distribution)` alone -- i.e. everything identical in all 21 indexer layers
    of one forward pass, and identical in both branches of the `lax.cond`.

    A `NamedTuple`, so it is a pytree and crosses `lax.cond` unregistered.
    """

    # i32[b_pad]: `_effective_row_lengths` already padded to SparseCore's
    # `b_pad = align_to(num_tokens, 16)`, so both the reduce and the pad move.
    eff_lengths_padded: jax.Array
    # i32[2] x3: the `start_end_seq_idx_ref` scalar-prefetch operand of each of
    # the three `run_scores_kernel` calls (decode-grouped, decode mop-up, mixed).
    start_end: tuple[jax.Array, jax.Array, jax.Array]
    # i32[3] / i32[2]: `sem_ids` and `bo_sz` scalar prefetch. Loop-carried SMEM
    # scratch; the *initial* value is a constant and identical everywhere.
    sem_ids: jax.Array
    bo_sz: jax.Array
    # The three pallas grid extents. Dynamic (they depend on `distribution`),
    # so traced scalars rather than Python ints.
    grid: tuple[jax.Array, jax.Array, jax.Array]


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
    """Compute once what the incumbent computes inside every branch, in every
    layer.  Every line is lifted from `_common_path` / `_effective_row_lengths`
    / `run_scores_kernel` / `sparsecore_topk`; nothing is re-derived.
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
    # `sparsecore_topk`'s own pad, moved up with it (sparsecore_topk.py:322).
    b_pad = _sc._align_to(num_tokens, _sc.LANES)
    eff_padded = jnp.pad(eff.astype(jnp.int32), (0, b_pad - num_tokens))

    return HoistedMeta(
        eff_lengths_padded=eff_padded,
        start_end=(
            jnp.stack([jnp.int32(0),
                       decode_batch_end.astype(jnp.int32)]),
            jnp.stack([
                decode_batch_end.astype(jnp.int32),
                distribution[1].astype(jnp.int32),
            ]),
            jnp.stack([
                distribution[1].astype(jnp.int32),
                distribution[2].astype(jnp.int32),
            ]),
        ),
        sem_ids=jnp.zeros((3, ), jnp.int32),
        bo_sz=jnp.full((2, ), -1, jnp.int32),
        grid=(
            (decode_batch_end // decode_req_batch_size).astype(jnp.int32),
            (distribution[1] - decode_batch_end).astype(jnp.int32),
            (distribution[2] - distribution[1]).astype(jnp.int32),
        ),
    )


def _zeros_like_struct(s) -> jax.Array:
    """A zero array with `s`'s aval, weak type included.

    `lax.cond` compares branch avals exactly, so a weakly-typed output would
    not match `jnp.zeros`' strongly-typed one.
    """
    if getattr(s, "weak_type", False):
        return jnp.full(s.shape, 0 if jnp.issubdtype(s.dtype, jnp.integer) else
                        0.0)
    return jnp.zeros(s.shape, s.dtype)


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
    """`sparsecore_topk._sc_topk_direct` with the `jnp.pad` hoisted out.

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

    Two paths, matching the incumbent's split at `sparsecore_topk.py:409`:

    * `p == 1` -- the partition `_pick_partition` returns at the production
      shape (b=64, n=10240, k=2048).  Takes lengths already padded.
    * `p > 1` -- reached at the small-`b` shapes the correctness grid uses. The
      two-stage path re-derives and re-pads lengths per stage at a different
      `b`, so nothing beyond the row lengths themselves is left to hoist.
      Delegate to the incumbent, fed the hoisted lengths.
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
# D1-lite's `streamindex_topk`.
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
        "enable_early_exit",
        "hoist",
    ),
)
def streamindex_topk_dl(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = True,
    hoist: str = "cond",
) -> jax.Array:
    """The incumbent's `streamindex_topk` with the branch-invariant metadata
    and scalar-prefetch constants hoisted out of the `lax.cond` branches.

    See the module docstring for `hoist`.
    """
    if hoist not in HOIST_MODES:
        raise ValueError(f"hoist must be one of {HOIST_MODES}, got {hoist!r}")
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
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    num_tokens = q.shape[0]
    # `scores_to_reduce.shape[1]` after the reshape and the pad-to-k, computed
    # statically instead of read off the (per-layer) score array. Identical
    # value: streamindex_topk.py:811-817.
    num_positions = max(num_sublanes_total * 128, k)

    def _make_meta() -> HoistedMeta:
        return precompute(
            seq_lens,
            cu_q_lens,
            distribution,
            num_tokens=num_tokens,
            num_positions=num_positions,
            compression_ratio=compression_ratio,
            decode_req_batch_size=decode_req_batch_size,
        )

    fast_pred = jnp.max(seq_lens) // compression_ratio <= k

    # ---- the hoist ------------------------------------------------------
    if hoist == "none" or not enable_early_exit:
        # "none": control, metadata stays inside `_common_path`.
        # no early exit: there are no branches to hoist out of; leaving it in
        # `_common_path` is exactly Option D's `d1`/`dcond` arm.
        hoisted = None if hoist == "none" else _make_meta()
    elif hoist == "plain":
        hoisted = _make_meta()
    else:  # "cond"
        spec = jax.eval_shape(_make_meta)
        hoisted = jax.lax.cond(
            fast_pred,
            lambda: jax.tree.map(_zeros_like_struct, spec),
            _make_meta,
        )

    def run_scores_kernel(
        scores_init,
        start_end,
        grid_extent,
        sem_ids,
        bo_sz,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        out_dtype,
        case=MlaCase.MIXED,
    ):
        """`streamindex_topk.run_scores_kernel` with `start_end`, `sem_ids`,
        `bo_sz` and the grid extent supplied rather than rebuilt in place."""
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
            start_end,  # HOISTED (was jnp.array([start, end], jnp.int32))
            sem_ids,  # HOISTED (was jnp.zeros((3,), jnp.int32))
            bo_sz,  # HOISTED (was jnp.full((2,), -1, jnp.int32))
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
        # `hoist="none"` rebuilds the metadata here, inside the branch, which
        # is where the incumbent builds it.
        meta = _make_meta() if hoisted is None else hoisted

        out_dtype = jnp.float32
        # Per-layer by necessity: donated into the kernel's output via
        # `input_output_aliases`, so this is NOT hoisted.
        scores_init = jnp.full((q.shape[0], num_sublanes_total, 128),
                               -jnp.inf,
                               dtype=out_dtype)

        scores = run_scores_kernel(
            scores_init,
            meta.start_end[0],
            meta.grid[0],
            meta.sem_ids,
            meta.bo_sz,
            static_q_len=1,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            seq_batch_size=decode_req_batch_size,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )
        # num_decode_seqs % decode_req_batch_size != 0 mop-up.
        scores = run_scores_kernel(
            scores,
            meta.start_end[1],
            meta.grid[1],
            meta.sem_ids,
            meta.bo_sz,
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
            meta.sem_ids,
            meta.bo_sz,
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
        assert scores.shape[1] == num_positions, (scores.shape, num_positions)

        topk_idxs = _sparsecore_topk_pre(scores,
                                         k,
                                         meta.eff_lengths_padded,
                                         write_empty_rows=True)
        return topk_idxs[:q.shape[0], :k]

    def _fast_path(_):
        # Verbatim from streamindex_topk.py:840-856.
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
    return jax.lax.cond(fast_pred, _fast_path, _common_path, None)
