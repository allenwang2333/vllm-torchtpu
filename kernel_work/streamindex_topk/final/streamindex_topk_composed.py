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
"""TPU-Friendly StreamIndex Top-K kernel.

COMPOSED SHIPPING CANDIDATE -- Option B ("Right-Size") + Option A ("Narrow
Record"), built by `kernel_work/streamindex_topk/final/compose.py`. See
`kernel_work/streamindex_topk/final_report.md`.

LAYOUT OPT-OUT (discharges measurement_and_design.md section 6.5)
-----------------------------------------------------------------
`step2/sheets/layout-code-style.md` section 10 asks a kernel of this size to be
split into `api.py` / `base.py` / `config.py` / `wrapper.py` / `kernel.py` /
`compute.py` / `load_store.py` / `dma.py` / `tiling.py`, and sanctions an
OPT-OUT for a kernel that does dynamic-offset DMA, builds its metadata inside
the kernel and buffers selectively -- "say so in the module docstring and
leave". This kernel does all three: `_fetch_bkv` issues per-page DMAs at
offsets computed from `page_indices` inside the body, `_effective_row_lengths`
and the causal masks are constructed in-kernel, and the bkv / bq / bq-weights /
bo-scores buffers are double-buffered independently with a rotating semaphore
bank. **This file therefore takes the opt-out deliberately and stays one
file.** What the sheet forbids is half-using the layout, which is what an
un-annotated single file amounts to; this paragraph is the annotation. A split
should happen only as its own change, never bundled with a perf change, because
bundling makes the A/B unattributable.

WHAT THIS FILE CHANGES AGAINST THE SHIPPED KERNEL
--------------------------------------------------
Two independent changes, on two different phases. They touch disjoint code and
were measured separately in step 3; this file is the composition and is
measured as a whole.

OPTION B -- "Right-Size" (the MIXED / prefill band)
...................................................
1.  `num_sublanes_total` is sized from a caller-supplied **static bucket on the
    batch's real KV length** (`kv_len_bucket`) instead of from `max_model_len`
    via `pages_per_seq`. With `kv_len_bucket=None` this half is the incumbent.
2.  The out-DMA becomes a **masked partial-block store**: the final `bkv` slab
    writes only the sublanes that fall inside the buffer. Both extents are
    compile-time constants and the choice between them is a `pl.when` on the
    block index, so no DMA carries a traced extent. `num_bkv` is clamped to the
    number of slabs the buffer holds, so a `kv_len` above the bucket truncates
    the scan rather than writing past the end of the buffer.
3.  The `jnp.full(..., -inf)` scores prefill is **deleted**. The first band
    allocates its own Pallas output instead of aliasing an XLA-initialised one;
    the two later bands still alias onto it. Nothing initialises the lanes no
    band writes.

Why (3) is safe -- and this is the claim the poison arm tests, not an assertion
the code may rely on: `_topk_kernel` in `sparsecore_topk.py` gates every
candidate on `valid = (j < i_hi) & ...` with `i_hi = min(slice_len,
row_lengths[row])` (:151-155, :171). Positions at or past a row's
`row_lengths` are read by the resident DMA but can never be *selected*. The
kernel writes `align_to(kv_len, bkv_sz)` positions per row, which is >= that
row's `row_lengths`, so the region the prefill used to cover is exactly the
region selection cannot reach. The `-inf` the causal/validity mask writes
*inside* a written slab is unchanged and stays load-bearing -- only the
redundant prefill goes.

**`kv_len_bucket` must be a bucket, not the exact `kv_len`.** Use
`kv_len_bucket_for()` below: every rung it returns is <= `MAX_SLICE_WORDS`
(65536), which is what keeps `sparsecore_topk._pick_partition` at P=1. That
threshold, not the byte saving, is where most of B's prize comes from.

OPTION A -- "Narrow Record" (the DECODE band, reader half)
..........................................................
The *format of the compressed-KV cache* and the reads that consume it. The
arithmetic, the loop nest, the grid, the block sizes and the store are
untouched, so this half is bit-exact against the incumbent by construction.

Incumbent, one array:

    cache_kv  u8[num_pages, page_size//kv_packing, kv_packing, 256]
              byte   0..127  the 128 fp8 (e4m3) key values
              byte     128   the single e8m0 block scale
              byte 129..255  padding -- never read, always transferred

Composed, two arrays:

    cache_val    u8[num_pages, page_size//kv_packing, kv_packing, 128]
    cache_scale  u8[num_pages, page_size // 128, 128]
                 the e8m0 scale for position `p` of page `g` at
                 [g, p // 128, p % 128]

129 used bytes per compressed position instead of 256, so the indexer read of
this array falls by 49.6%.

WHY THE SCALE PLANE IS DECLARED THREE-DIMENSIONAL. `[num_pages,
page_size // 128, 128]` holds exactly the same bytes in exactly the same order
as `[num_pages, page_size]` would. The declaration is not cosmetic: Mosaic
tiles the LAST TWO dimensions of an HBM memref, so a 2-D `[num_pages, 1024]`
plane is tiled on its *page* axis and the reader's one-page DMA is a one-row
slice of a tiled dimension, which is rejected:

    Offsets along tiled dimensions must be aligned to tiles. Failed to verify
    that the index at dimension 0 is divisible by the tile dimension 32.

    -- job j-ce74a9f2-a4cd-40cb-9e87, the first build of Option A.

`tpu.assume_multiple` does NOT apply: the page index genuinely is not a
multiple of 32. With the 3-D declaration the page axis is untiled and the tiled
`(8, 128)` pair is exactly one page of 1024 scales, so the per-page DMA is a
whole number of tiles and moves 1024 B with no padding.

`load_bkv` widens e8m0 -> f32 BEFORE the regroup, so the relayout runs on an
f32 (8,128) tile rather than a packed 8-bit one. THIS IS BIT-EXACT: e8m0 is a
bare power-of-two exponent, and bf16 and f32 both have 8 exponent bits, so both
represent every e8m0 value exactly and neither rounds.

`cache_kv` is passed as a two-tuple `(cache_val, cache_scale)` so the public
signature -- and therefore the read-only instruments -- is unchanged.

**THE WRITER SIDE IS NOT IN THIS FILE.** Producing this format is a change to
`mla_attention_op.py:103-104` and to the cache allocator. It is measured
separately (3.342x at T=1024) and it is COUPLED: the split writer emits a
format only the split reader consumes, so neither half ships alone.

WHAT IS *NOT* HERE
------------------
Option E (`num_kv_pages_per_block` (2,2,2) -> (4,2,4)) is a call-site tuple in
`mla_attention_op.py`, not a change to this file. Options C, D, A1 and F are
dead on measurement; do not re-propose them.
"""

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
    bo_partial_ref,  # Shape: [2], Memory: SMEM -- 1 if that out-DMA was a tail
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim=128], Memory: HBM
    cache_scale_hbm_ref,  # Shape: [total_num_pages, page_size//128, 128], HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM
    # Scratch
    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim=128], Memory: VMEM
    bkv_scale_x2_ref,  # Shape: [2, seq_batch_size, bkv_p, 8, 128], VMEM
    bq_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads, head_dim], Memory: VMEM
    bq_weights_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads], Memory: VMEM
    scores_block_x2_ref,  # Shape: [2, seq_batch_size * bq_sz, num_sublanes_bkv, 128], Memory: VMEM
    sems,  # Shape: [5, 2, seq_batch_size], Memory: Semaphore
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

    page_size = page_size_per_kv_packing * kv_packing
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    num_sublanes_bkv = bkv_sz // 128

    # RIGHT-SIZE. The scores buffer may now be narrower than a whole number of
    # `bkv` slabs, so the last slab is partial. Both extents are static: the
    # store is full width for block indices < `n_full_slabs`, `n_tail_sublanes`
    # wide at exactly `n_full_slabs`, and there is no block after that because
    # `num_bkv` is clamped to `max_bkv` below. When the width happens to be a
    # multiple of the slab (every one of step 1's five regimes -- kv_len 8192
    # and 65536 are both multiples of bkv_sz=2048) `n_tail_sublanes` is 0, the
    # tail branch is not emitted at all, and this is the incumbent's store.
    num_sublanes_total = scores_hbm_ref.shape[1]
    n_full_slabs = num_sublanes_total // num_sublanes_bkv
    n_tail_sublanes = num_sublanes_total - n_full_slabs * num_sublanes_bkv
    max_bkv = cdiv(num_sublanes_total, num_sublanes_bkv)

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
        # The semaphore is credited with the bytes the DMA actually moved, so
        # the wait has to name the same extent the start named. Which branch
        # ran is recorded in SMEM rather than recomputed, because `bkv_idx` is
        # long gone by the time the wait for that buffer comes round.
        old_sz = bo_sz_ref[bo_sem_idx]

        if not n_tail_sublanes:
            # Byte-for-byte the incumbent's wait. When the buffer is a whole
            # number of slabs -- every one of step 1's five regimes -- nothing
            # about the out-DMA changes, not even the scalar predicate.
            @pl.when(old_sz >= 0)
            def _():
                dst = scores_block_x2_ref.at[bo_sem_idx,
                                             pl.ds(0, seq_batch_size * old_sz),
                                             pl.ds(0, num_sublanes_bkv)]
                _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)
            return

        old_partial = bo_partial_ref[bo_sem_idx]

        @pl.when((old_sz >= 0) & (old_partial == 0))
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz),
                                         pl.ds(0, num_sublanes_bkv)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

        @pl.when((old_sz >= 0) & (old_partial == 1))
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz),
                                         pl.ds(0, n_tail_sublanes)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

    def start_send_scores(bo_sem_idx, sz, token_start, bkv_idx):
        # All sequences in the batch have the same sz. Issue a single DMA for the
        # entire batch.
        bo_sz_ref[bo_sem_idx] = sz
        sublane_start = bkv_idx * num_sublanes_bkv

        if not n_tail_sublanes:
            # Byte-for-byte the incumbent's store. `bo_partial_ref` is not even
            # written, so at the five step-1 regimes the only difference from
            # the shipped kernel is the width of the buffer it writes into.
            _async_copy(
                scores_block_x2_ref.at[bo_sem_idx,
                                       pl.ds(0, seq_batch_size * sz),
                                       pl.ds(0, num_sublanes_bkv)],
                scores_hbm_ref.at[
                    pl.ds(token_start, seq_batch_size * sz),
                    pl.ds(sublane_start, num_sublanes_bkv),
                ],
                sems.at[2, bo_sem_idx, 0],
                wait=False,
            )
            return

        is_tail = bkv_idx == n_full_slabs
        bo_partial_ref[bo_sem_idx] = is_tail.astype(jnp.int32)

        @pl.when(jnp.logical_not(is_tail))
        def _():
            _async_copy(
                scores_block_x2_ref.at[bo_sem_idx,
                                       pl.ds(0, seq_batch_size * sz),
                                       pl.ds(0, num_sublanes_bkv)],
                scores_hbm_ref.at[
                    pl.ds(token_start, seq_batch_size * sz),
                    pl.ds(sublane_start, num_sublanes_bkv),
                ],
                sems.at[2, bo_sem_idx, 0],
                wait=False,
            )

        @pl.when(is_tail)
        def _():
            # The destination offset is written as the static constant rather
            # than as `sublane_start`, which is only equal to it: a traced
            # offset the compiler cannot prove aligned is the one thing that
            # reliably fails to lower (references/hardware-facts.md).
            _async_copy(
                scores_block_x2_ref.at[bo_sem_idx,
                                       pl.ds(0, seq_batch_size * sz),
                                       pl.ds(0, n_tail_sublanes)],
                scores_hbm_ref.at[
                    pl.ds(token_start, seq_batch_size * sz),
                    pl.ds(n_full_slabs * num_sublanes_bkv, n_tail_sublanes),
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
            scale_sem = sems.at[4, bkv_sem_idx, batch_idx]
            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]
            bkv_scale_vmem_ref = bkv_scale_x2_ref.at[bkv_sem_idx, batch_idx]

            kv_p_start = bkv_idx * bkv_p
            page_indices_offset = (seq_idx +
                                   batch_idx) * pages_per_seq + kv_p_start

            if not wait:
                for i in range(bkv_p):
                    sz_per_kv_packing = page_size_per_kv_packing
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    raw_page = page_indices_ref[page_idx]
                    safe_page_offset = jnp.minimum(
                        raw_page * page_size_per_kv_packing,
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

                    # NARROW RECORD: the scale plane is a second, 128x
                    # smaller array. One DMA per page moves that page's
                    # whole (8, 128) scale tile -- 1024 B, a whole number
                    # of tiles, no padding -- into slot `i` of the VMEM
                    # buffer. Slicing dimension 0 is legal precisely
                    # because the 3-D declaration leaves the page axis
                    # untiled; see the module docstring for the 2-D form
                    # Mosaic rejected.
                    safe_scale_page = jnp.minimum(
                        jnp.maximum(raw_page, 0), total_num_pages - 1)
                    _async_copy(
                        cache_scale_hbm_ref.at[pl.ds(safe_scale_page, 1)],
                        bkv_scale_vmem_ref.at[pl.ds(i, 1)],
                        scale_sem,
                        wait=False,
                    )
            else:
                dma_bkv_sz = bkv_p * page_size_per_kv_packing
                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)
                dst_sc = bkv_scale_vmem_ref.at[pl.ds(0, bkv_p)]
                _async_copy(src=dst_sc, dst=dst_sc, sem=scale_sem, wait=True)

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

            # NARROW RECORD: the DMA landed exactly the 128 value bytes,
            # so there is no slice-and-recombine here any more. The
            # `[:, :128]` slice and the `[:, 128:129].T` transpose both
            # disappear.
            flat_bkv = bkv.reshape(-1, bkv.shape[-1])
            fp8_val = pltpu.bitcast(flat_bkv, jnp.float8_e4m3fn)
            # [bkv_p, 8, 128] u8 -> e8m0 -> f32 -> [1, bkv_sz].
            # The widen happens BEFORE the regroup so the relayout runs on
            # an f32 (8,128) tile rather than a packed 8-bit one. e8m0 is
            # a bare exponent and f32 has the same 8 exponent bits, so the
            # widen is exact and this stays bit-identical to the incumbent
            # (which widens to bf16 -- also exact -- and promotes in the
            # multiply).
            raw_scale = bkv_scale_x2_ref.at[bkv_sem_idx, batch_idx][...]
            scale_val = pltpu.bitcast(
                raw_scale, jnp.float8_e8m0fnu).astype(jnp.float32).reshape(
                    1, bkv_sz)

            bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
            bkv_scales.append(scale_val)
        return bkvs, bkv_scales

    def process():
        kv_len_max = jnp.max(jnp.stack(kv_lens))
        num_bkv = jnp.maximum(1, cdiv(kv_len_max, bkv_sz))
        # A `kv_len` above the caller's bucket truncates the scan instead of
        # storing past the end of the buffer. `_effective_row_lengths` already
        # clamps `row_lengths` to the buffer width, so the two agree and the
        # result is an exact top-k over the first `num_sublanes_total * 128`
        # positions -- the same degradation the incumbent shows when a sequence
        # exceeds `max_model_len`.
        num_bkv = jnp.minimum(num_bkv, max_bkv)
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
        "kv_len_bucket",
        "poison_fill",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: tuple,  # (u8[num_pages, page_size//kv_packing, kv_packing, 128],
    #                  u8[num_pages, page_size//128, 128])
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
    kv_len_bucket: int | None = None,
    poison_fill: float | None = None,
) -> jax.Array:
    """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache, as the two-tuple (values, scales).
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
    kv_len_bucket: RIGHT-SIZE. A static upper bound on `max(seq_lens) //
      compression_ratio` for this call. When given, the scores buffer is sized
      from it rather than from `pages_per_seq` (i.e. from `max_model_len`), and
      the `-inf` prefill is dropped. It must be a *bucket*, not the exact
      length, or every distinct batch triggers a recompile; and it must be an
      upper bound, or the scan is truncated to the buffer (see `max_bkv` in the
      kernel body). `None` reproduces the incumbent exactly.
    poison_fill: TEST HOOK, not for production. Reinstates the aliased
      initialisation with this value instead of `-inf`. Used to prove that no
      consumer reads a lane the kernel does not write: a large positive poison
      that changes no margin cannot have been selected. Ignored unless
      `kv_len_bucket` is set.

  Returns:
    Top-K indices (in compressed space).
  """
    # NARROW RECORD: scale factors live in their OWN array, not inside the
    # value record. That is the whole change; see the module docstring.

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
    # NARROW RECORD: `cache_kv` arrives as the two-tuple
    #   (u8[num_pages, page_size//kv_packing, kv_packing, 128],
    #    u8[num_pages, page_size//128, 128])
    # so that the public signature is unchanged.
    cache_val, cache_scale = cache_kv
    lkv_dim = cache_val.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_val.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")

    if kv_len_bucket is None:
        # Incumbent: `align_to(pages_per_seq, bkv_p)` over the whole tuple, and
        # `pages_per_seq` comes from `max_model_len`. Slot 1 is read by no
        # executing band and still participates in the `max`
        # (measurement_and_design.md section 2.4).
        num_sublanes_total = max(
            align_to(pages_per_seq, bkv_p) * page_size // 128
            for bkv_p in num_kv_pages_per_blocks)
    else:
        # RIGHT-SIZE. Exactly the live positions, rounded up to one sublane of
        # 128 -- not up to a page and not up to a `bkv` slab. The masked store
        # in the kernel body is what makes the un-rounded width legal, and the
        # `max` over the tuple goes with the rounding: raising any `bkv_p` no
        # longer widens the buffer for every band, which is the coupling that
        # made Option E cost mixed anything (section 4.2).
        live_positions = min(kv_len_bucket, pages_per_seq * page_size)
        num_sublanes_total = max(cdiv(live_positions, 128), cdiv(k, 128))

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        cache_scale,
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

        # RIGHT-SIZE. `scores_init is None` is the deleted `-inf` prefill: this
        # band takes no scores input and Pallas hands it a fresh, uninitialised
        # HBM output. Later bands alias onto that buffer as before, so the
        # chain still accumulates -- what is gone is the XLA fill that used to
        # write the whole thing before the kernel overwrote 80-97% of it.
        aliased = scores_init is not None
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv (values only)
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_scale
        ]
        if aliased:
            in_specs.append(
                pl.BlockSpec(
                    memory_space=pltpu.HBM))  # scores_init (aliased to out)
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

        # NARROW RECORD: one extra, tiny double buffer for the scale plane.
        bkv_scale_double_buf = pltpu.VMEM(
            (2, seq_batch_size, bkv_p, page_size // 128, 128),
            cache_scale.dtype,
        )

        scratch_shapes = [
            bkv_double_buf,
            bkv_scale_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((5, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),  # (bq, bkv, bo) sem indices
            jnp.full((2, ), -1, jnp.int32),  # in-flight out DMA row counts
            jnp.zeros((2, ), jnp.int32),  # 1 if that out DMA was the tail slab
        )

        # A Pallas body is a flat positional list, so dropping the aliased
        # `scores_init` operand would silently shift every ref after it --
        # including the output and all four scratch buffers. `kernel` never
        # reads `scores_in_hbm_ref` (it exists only to name the alias), so the
        # un-aliased case splices a `None` back into that slot rather than
        # maintaining a second signature.
        # COMPOSED: `+ 4`, not B's `+ 3`. The operands before `scores_init`
        # are now q, prepared_indexer_weights, cache_kv AND cache_scale --
        # Option A inserted one. Off by one here shifts the output and all
        # five scratch refs.
        n_before_scores_in = len(scalar_prefetches) + 4

        def _body(*refs, **static):
            if aliased:
                return kernel(*refs, **static)
            return kernel(*refs[:n_before_scores_in], None,
                          *refs[n_before_scores_in:], **static)

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                _body,
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
            input_output_aliases=({
                len(scalar_prefetches) + 4: 0
            } if aliased else {}),
            name=scope_name,
        ))
        operands = [q, prepared_indexer_weights, cache_kv, cache_scale]
        if aliased:
            operands.append(scores_init)
        return pallas_kernel(*scalar_prefetches, *operands)

    def _common_path(_):
        out_dtype = jnp.float32

        if kv_len_bucket is None:
            # Incumbent: pre-fill the output with -inf and alias it, so masked
            # / unwritten columns are already -inf.
            scores_init = jnp.full(
                (q.shape[0], num_sublanes_total, 128),
                -jnp.inf,
                dtype=out_dtype,
            )
        elif poison_fill is not None:
            # TEST HOOK. Same aliasing as the incumbent, but the value is a
            # large positive rather than -inf. Any lane the kernel does not
            # write and something downstream nevertheless reads will now be the
            # largest score in its row and will be selected. Margins identical
            # to the un-poisoned run therefore prove no such lane is read; this
            # is a strict test, not an approximation (design section 4.4).
            scores_init = jnp.full(
                (q.shape[0], num_sublanes_total, 128),
                poison_fill,
                dtype=out_dtype,
            )
        else:
            # RIGHT-SIZE: no fill at all. The first band allocates.
            scores_init = None

        # TODO: we shall sort the sequences by length, so that multiple decode
        # sequences in one batch have similar lengths to reduce waste of compute.
        # With the same batch size, the longest sequence will determine number of
        # blocks to run computation for.
        decode_batch_end = (distribution[0] // decode_req_batch_size *
                            decode_req_batch_size)
        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_val,
            cache_scale,
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
            cache_val,
            cache_scale,
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
            cache_val,
            cache_scale,
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


# ---------------------------------------------------------------------------
# THE BUCKET LADDER -- shipping plumbing for Option B.
#
# Option B was MEASURED with the driver passing the exact `kv_len`, which its
# own RESULTS.md correctly calls a ceiling rather than a shippable value: a
# distinct static width per batch is a distinct jit cache entry per batch. A
# shipping version rounds up to a small fixed set of widths.
#
# The rung spacing is 8192 tokens. Measured on a six-rung ladder in job
# j-59bbfd2f-e408-4a75-a155: chunk-averaged prize at the A/B shape is -40.9%
# for this ladder against -46.2% for the exact length and -30.2% for powers of
# two, so 8192 costs little and pow2 costs a third of the prize.
#
# THE HARD CONSTRAINT IS THE CAP, NOT THE SPACING. Every rung must be
# <= `sparsecore_topk.MAX_SLICE_WORDS` (65536), because that is where
# `_pick_partition` flips from P=2 to P=1 and collapses the two-stage top-k
# into one stage -- read back directly in the same job: incumbent P=[2] at
# n=67584 at every rung, B P=[1] at all six buckets. That threshold is where
# most of B's prize comes from, so a ladder whose top rung exceeds 65536 throws
# away the large half. The incumbent's `max_model_len`-derived 67584 is stuck
# above it by 2048 columns.
# ---------------------------------------------------------------------------
BUCKET_GRANULARITY = 8192
# Mirror of sparsecore_topk.MAX_SLICE_WORDS. Imported rather than duplicated
# would be better; it is written out here so this file has no new import, and
# `_assert_bucket_cap_matches()` below refuses to run if the two disagree.
MAX_SLICE_WORDS_MIRROR = 64 * 1024


def _assert_bucket_cap_matches() -> None:
    """Refuse to size a bucket against a stale copy of MAX_SLICE_WORDS."""
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        MAX_SLICE_WORDS)
    if MAX_SLICE_WORDS != MAX_SLICE_WORDS_MIRROR:
        raise AssertionError(
            f"sparsecore_topk.MAX_SLICE_WORDS is {MAX_SLICE_WORDS} but this "
            f"file's mirror is {MAX_SLICE_WORDS_MIRROR}; the bucket ladder's "
            "P=1 guarantee no longer holds and must be re-derived.")


def kv_len_bucket_for(max_kv_len: int, max_model_len: int, k: int) -> int:
    """Round a batch's live KV length up to the next static bucket.

    Args:
      max_kv_len: an upper bound on `max(seq_lens) // compression_ratio` for
        this call. MUST be an upper bound -- see `max_bkv` in the kernel body,
        which truncates the scan rather than overrunning the buffer.
      max_model_len: the deployment flag. Caps the ladder; there is no point
        bucketing above the widest buffer the incumbent would have built.
      k: `index_topk`. The buffer can never be narrower than the top-k itself.

    Returns:
      A value from a bounded set of widths -- at most
      `cdiv(min(max_model_len, 65536), 8192) + 1` distinct results, i.e. 9 at
      the A/B's `max_model_len = 66560` -- suitable as the static
      `kv_len_bucket` kwarg.

    The count is what bounds the recompile cost, and the recompile cost is the
    reason this is a ladder rather than the exact length.
    """
    _assert_bucket_cap_matches()
    ceiling = min(max_model_len, MAX_SLICE_WORDS_MIRROR)
    b = min(align_to(max(int(max_kv_len), 1), BUCKET_GRANULARITY), ceiling)
    return max(b, k)


def bucket_ladder(max_model_len: int, k: int) -> list:
    """Every value `kv_len_bucket_for` can return. One jit entry per rung."""
    ceiling = min(max_model_len, MAX_SLICE_WORDS_MIRROR)
    rungs = []
    w = BUCKET_GRANULARITY
    while w < ceiling:
        rungs.append(max(w, k))
        w += BUCKET_GRANULARITY
    rungs.append(max(ceiling, k))
    return sorted(set(rungs))
