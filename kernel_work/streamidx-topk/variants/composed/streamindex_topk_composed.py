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
"""The composed candidate: D1-lite's hoist + Option C gated to the MIXED launch.

This module is `variants/d1-lite/streamindex_topk_d1lite.py` with exactly one
thing added -- a per-`run_scores_kernel` `q_layout` -- and one thing removed:
the `hoist="cond"` mode (`dlc`), which D1-lite measured as worse on both axes.

Two independent switches, so all four arms of the composition test come out of
one module and one code path:

    hoist=False, head_major_mixed=False   -> `ctl`   (must reproduce `inc`)
    hoist=False, head_major_mixed=True    -> `c`     (Option C, gated)
    hoist=True,  head_major_mixed=False   -> `dl`    (D1-lite, plain hoist)
    hoist=True,  head_major_mixed=True    -> `c_dl`  (the composed candidate)

The `inc` arm is NOT this module.  It is
`vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk`, imported
unmodified from `src/`.  `ctl` exists so that the difference between "this
module with both switches off" and "the incumbent" is measured rather than
assumed.

## What is composed

**D1-lite (`hoist=True`).**  Everything that is a pure function of
`(seq_lens, cu_q_lens, distribution)` is computed once, in the entry
computation, above the `lax.cond`: `_effective_row_lengths`
(`streamindex_topk.py:482-506`), `sparsecore_topk`'s `jnp.pad` of the row
lengths (`sparsecore_topk.py:322`), `decode_batch_end`
(`streamindex_topk.py:755-756`) and the three scalar-prefetch operands
(`streamindex_topk.py:694-701`).  The `lax.cond` and `_fast_path` are kept
verbatim.  Cross-layer CSE is the point: XLA compiles each `cond` branch into
its own computation per call site, so 21 indexer layers otherwise carry 21
private copies of identical metadata.

**Option C (`head_major_mixed=True`), GATED.**  The score block's head axis is
made the outer one so `sum_h` is a whole-vreg add chain instead of a
`vrot.slane` tree.  The gate is not optional: ungated at decode this is
**+59.62%**, a 1.6x regression, because at `bq_sz=1` there is no 128-row block
to put between consecutive heads.  So only the **MIXED** launch -- the one that
serves prefill, at `bq_sz = num_queries_per_block[2] = 128` -- gets
`q_layout="head_major"`.  Both DECODE launches keep `"head_minor"`, i.e. the
incumbent's layout, byte for byte.

## What is deliberately NOT here

* the **`bkv_p` ladder** / `recommended_num_kv_pages_per_block` -- withdrawn.
  `num_kv_pages_per_block` is whatever the caller passes; production is
  `(2, 2, 2)` and every measurement uses it.
* **D1** (`enable_early_exit=False`, deleting the cond) -- withdrawn,
  superseded by D1-lite.  `enable_early_exit` is still an argument because the
  incumbent has one and the signature must match; production passes `True`.
* the **KV-cache relayout / v128** change -- a cache storage-format change, out
  of scope.
* `q_layout="head_major_hbm"` and Option F's `mm_dtype` -- reachable in
  `option-c-head-major/streamindex_topk_c.py`, not exposed here.

## Where each piece of code comes from

Not one line of Mosaic is written here.  The kernel body is imported:

* `kernel` from `option-c-head-major/streamindex_topk_c.py`, which is the
  incumbent's kernel copied verbatim plus the `q_layout` parameter.  With
  `q_layout="head_minor"` it is the incumbent's kernel; `readback.sh` prints
  the full `difflib` diff of the two function bodies so that claim is on the
  record rather than asserted.
* every other helper (`prepare_q_inputs`, `prepare_index_weights`,
  `_effective_row_lengths`, `align_to`, `MlaCase`) from the incumbent in
  `src/`, by reference.

### A defect found in `streamindex_topk_c.py`, and why it does not affect this

`streamindex_topk_c.streamindex_topk` threads `q_layout` into a
`run_scores_kernel` whose three call sites never pass it, so its own whole-op
entry point silently runs `head_minor` on all three launches.  Option C's
published numbers came from `stage_a_c.py`, which builds the `pallas_call`
itself and is unaffected.  This module does not use
`streamindex_topk_c.streamindex_topk` at all -- only its `kernel` -- and passes
`q_layout` explicitly at each of the three call sites below.  That the gate
took is checked at runtime two ways: `describe_layouts()` returns the layout of
each launch, and the Pallas scope name carries a `-hm` suffix, so the MIXED
kernel appears in `hlo_stats` as `StreamIdxTC-hm-M-bq_128-bkvp_2` and the two
DECODE kernels as plain `StreamIdxTC-D-...`.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sc
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as _si

# `streamindex_topk_c` lives in a sibling variant directory.  Add it to the
# path here rather than relying on the caller, so importing this module is
# enough.
_HERE = os.path.dirname(os.path.abspath(__file__))
_OPTION_C = os.path.abspath(os.path.join(_HERE, os.pardir,
                                         "option-c-head-major"))
if _OPTION_C not in sys.path:
    sys.path.insert(0, _OPTION_C)

import streamindex_topk_c as _c  # noqa: E402

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

# Imported, not copied.  `kernel` is Option C's (incumbent + `q_layout`);
# everything else is the incumbent's.
kernel = _c.kernel
prepare_q_inputs = _si.prepare_q_inputs
prepare_index_weights = _si.prepare_index_weights
_effective_row_lengths = _si._effective_row_lengths
align_to = _si.align_to
MlaCase = _si.MlaCase
DEFAULT_VMEM_LIMIT_BYTES = _si.DEFAULT_VMEM_LIMIT_BYTES

HEAD_MINOR = "head_minor"
HEAD_MAJOR = "head_major"

# arm -> (hoist, head_major_mixed).  `inc` is not here: it is the incumbent
# module, not this one.
ARM_SWITCHES = {
    "ctl": (False, False),
    "c": (False, True),
    "dl": (True, False),
    "c_dl": (True, True),
}


def describe_layouts(head_major_mixed: bool) -> dict[str, str]:
    """The `q_layout` each of the three launches will be built with.

    Returned rather than printed so a harness can assert on it.  The gate this
    describes is the whole reason Option C is safe to compose: ungated at
    decode it is a 1.6x regression.
    """
    return {
        "decode_grouped": HEAD_MINOR,
        "decode_mopup": HEAD_MINOR,
        "mixed": HEAD_MAJOR if head_major_mixed else HEAD_MINOR,
    }


# ---------------------------------------------------------------------------
# The hoisted values.  Verbatim from `d1-lite/streamindex_topk_d1lite.py`.
# ---------------------------------------------------------------------------


class HoistedMeta(NamedTuple):
    """Everything `streamindex_topk` derives from `(seq_lens, cu_q_lens,
    distribution)` alone -- identical in all 21 indexer layers of one forward
    pass, and identical in both branches of the `lax.cond`."""

    # i32[b_pad]: `_effective_row_lengths` already padded to SparseCore's
    # `b_pad = align_to(num_tokens, 16)`, so both the reduce and the pad move.
    eff_lengths_padded: jax.Array
    # i32[2] x3: the `start_end_seq_idx_ref` scalar-prefetch operand of each of
    # the three `run_scores_kernel` calls (decode-grouped, decode mop-up,
    # mixed).
    start_end: tuple[jax.Array, jax.Array, jax.Array]
    # i32[3] / i32[2]: `sem_ids` and `bo_sz` scalar prefetch.  Loop-carried
    # SMEM scratch; the *initial* value is a constant, identical everywhere.
    sem_ids: jax.Array
    bo_sz: jax.Array
    # The three pallas grid extents.  Dynamic (they depend on `distribution`).
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
    / `run_scores_kernel` / `sparsecore_topk`; nothing is re-derived."""
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


# ---------------------------------------------------------------------------
# SparseCore entry point that takes pre-padded lengths.  Verbatim from D1-lite.
# ---------------------------------------------------------------------------


def _sc_topk_direct_pre(
    scores: jax.Array,  # f32[b, n]
    k: int,
    lengths_padded: jax.Array,  # i32[b_pad]
    *,
    write_empty: bool = True,
) -> jax.Array:
    """`sparsecore_topk._sc_topk_direct` with the `jnp.pad` hoisted out.

    Identical to the incumbent apart from taking `lengths` already padded.  The
    kernel body `_topk_kernel` is the incumbent's, imported."""
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
    * `p > 1` -- reached at the small-`b` shapes the correctness grid uses.
      The two-stage path re-derives and re-pads lengths per stage at a
      different `b`, so nothing beyond the row lengths themselves is left to
      hoist.  Delegate to the incumbent, fed the hoisted lengths."""
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
# The composed `streamindex_topk`.
# ---------------------------------------------------------------------------

STATIC_ARGNAMES = (
    "k",
    "compression_ratio",
    "num_kv_pages_per_block",
    "num_queries_per_block",
    "vmem_limit_bytes",
    "decode_req_batch_size",
    "enable_early_exit",
    "hoist",
    "head_major_mixed",
)


@functools.partial(jax.jit, static_argnames=STATIC_ARGNAMES)
def streamindex_topk_composed(
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
    hoist: bool = True,
    head_major_mixed: bool = True,
) -> jax.Array:
    """The incumbent's `streamindex_topk`, with D1-lite's hoist and Option C's
    head-major score layout on the MIXED launch only.

    `hoist=False, head_major_mixed=False` is the port control: structurally the
    incumbent, run through this module.
    """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be "
            "specified.")
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block] * 3
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block
    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block] * 3
    else:
        num_queries_per_blocks = num_queries_per_block

    layouts = describe_layouts(head_major_mixed)

    max_num_seqs = seq_lens.shape[0]
    original_dtype = q.dtype
    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    # The incumbent's `prepare_q_inputs`: pad only, token-major.  Option C's
    # in-kernel transpose needs no HBM change, which is exactly why the gated
    # form is cheap.
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
    # statically instead of read off the (per-layer) score array.  Identical
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
    # `hoist=False` leaves the metadata inside `_common_path`, which is where
    # the incumbent builds it.  With `enable_early_exit=False` there are no
    # branches to hoist out of, so hoisting is free and unconditional.
    if hoist or not enable_early_exit:
        hoisted = _make_meta()
    else:
        hoisted = None

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
        q_layout=HEAD_MINOR,
    ):
        """`streamindex_topk.run_scores_kernel` with `start_end`, `sem_ids`,
        `bo_sz` and the grid extent supplied rather than rebuilt in place, and
        with an explicit per-launch `q_layout`.

        `q_layout` has no default that silently does the wrong thing: every
        call site below passes it explicitly.  (The `head_minor` default here
        exists only so the signature reads like the incumbent's; the defect in
        `streamindex_topk_c.streamindex_topk` is precisely a default like this
        one going unpassed.)"""
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

        # `head_major` transposes inside the kernel, so every buffer shape is
        # the incumbent's.  (`head_major_hbm`, which does change them, is not
        # reachable from this module.)
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
            start_end,  # HOISTED when hoist=True
            sem_ids,  # HOISTED when hoist=True
            bo_sz,  # HOISTED when hoist=True
        )

        # The suffix is what makes the gate visible in `hlo_stats`: the MIXED
        # kernel of a head-major arm shows up as `StreamIdxTC-hm-M-...` while
        # both DECODE kernels stay `StreamIdxTC-D-...`.  Same convention as
        # `streamindex_topk_c.py`, so the two variants' dumps are comparable.
        suffix = "-hm" if q_layout == HEAD_MAJOR else ""
        scope_name = (
            f"StreamIdxTC{suffix}-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}")
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
                q_layout=q_layout,
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
        # `hoist=False` rebuilds the metadata here, inside the branch, which is
        # where the incumbent builds it.
        meta = _make_meta() if hoisted is None else hoisted

        out_dtype = jnp.float32
        # Per-layer by necessity: donated into the kernel's output via
        # `input_output_aliases`, so this is NOT hoisted.
        scores_init = jnp.full((q.shape[0], num_sublanes_total, 128),
                               -jnp.inf,
                               dtype=out_dtype)

        # DECODE, grouped.  head_minor -- Option C is a regression here.
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
            q_layout=layouts["decode_grouped"],
        )
        # DECODE mop-up, num_decode_seqs % decode_req_batch_size != 0.
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
            q_layout=layouts["decode_mopup"],
        )
        # MIXED -- the only launch Option C is applied to.
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
            q_layout=layouts["mixed"],
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


def for_arm(arm: str):
    """`(fn, kwargs)` for one arm name, including `inc`.

    One place decides what each arm *is*, so a harness cannot disagree with the
    correctness check about what it measured.
    """
    if arm == "inc":
        return _si.streamindex_topk, {}
    if arm not in ARM_SWITCHES:
        raise ValueError(f"unknown arm {arm!r}; "
                         f"expected inc or one of {sorted(ARM_SWITCHES)}")
    hoist, hm = ARM_SWITCHES[arm]
    return streamindex_topk_composed, {"hoist": hoist, "head_major_mixed": hm}
