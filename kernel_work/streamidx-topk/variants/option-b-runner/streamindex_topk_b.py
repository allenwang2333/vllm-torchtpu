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
"""The merged StreamIndex Top-K op PLUS Option B (stage A / stage B pipelining).

A COPY of `variants/merged/streamindex_topk_merged.py` (md5
5bc0578427412d56209ef8b5d764fcae) with exactly one behavioural addition: the
static `num_chunks` argument.  Everything below the "---" is the merged
module's docstring, unchanged, describing the three changes this file
inherits.

## Option B, as a real op argument rather than a harness presentation

`variants/option-b-pipeline/` measured B by *hand-building* G pseudo-sequences
outside the op -- G separate calls whose `seq_lens` advanced by `q_len/G` and
whose `cu_q_lens` was `[0, T]`.  That is not something a caller can do:
production prefill is ONE sequence of 1024 query tokens and vLLM hands the op
`seq_lens` / `cu_q_lens` / `distribution` as traced arrays.
`option-b-pipeline/RESULTS.md` section 7 called this out as the reason B was
not adoptable.

This module makes the split internal and general.  `num_chunks=G` (static)
splits the **token** axis into G equal chunks; chunk `i` covers global tokens
`[a, b) = [i*T, (i+1)*T)`.  For each chunk the per-chunk metadata is derived
from the *unmodified* traced arrays by elementwise arithmetic:

    cu_q_lens_c[j] = clip(cu_q_lens[j], a, b) - a
    seq_lens_c[j]  = seq_lens[j] - max(0, cu_q_lens[j+1] - b)

Both are traced ops on traced values -- **nothing has to be static except G
itself**, and G derives from `q.shape[0]`, which is already static under
`jax.jit`.  The second formula is what makes the harness's trick general: it
is exactly the identity that keeps every token's absolute causal position

    seq_len - q_len + (t - cu_q_lens[seq])

unchanged when the token range is narrowed.  At the validated prefill geometry
(1 seq, q_len 1024, ctx 9216, G=4) it reproduces the harness's plan exactly:
`seq_lens_c = 8448, 8704, 8960, 9216` and `cu_q_lens_c = [0, 256]`.

Sequence-range narrowing for the MIXED launch is likewise traced:

    s = searchsorted(cu_q_lens[1:], a,   side="right")
    e = searchsorted(cu_q_lens[1:], b-1, side="right") + 1

clamped into `[distribution[1], distribution[2]]`.  A sequence outside
`[s, e)` has `q_len_c == 0` and would contribute nothing anyway; narrowing
only saves the empty grid steps.

### The launch count, and the one static guard

Naively, G chunks means 3G `run_scores_kernel` launches.  It does not have to.
Decode sequences occupy global tokens `[0, distribution[0])`, one token each,
and `distribution[0] <= max_num_seqs`.  So if

    T = num_tokens // G  >=  max_num_seqs          (both static)

every decode token is in chunk 0 and chunks `i > 0` can **statically** omit
both decode launches.  Launch count is then `2 + G`, against the incumbent's
`2 + 1`: no extra empty decode launches, only the mixed launch split G ways.
If the guard fails the module refuses to chunk (`ValueError`) rather than
silently dropping decode rows -- see `plan_chunks()`, which reports the guard
without running anything.

`num_chunks=1` takes the byte-identical unchunked path, so the G=1 arm IS the
merged module and is the honest baseline for the rest.

---

A COPY of `variants/option-c-head-major/streamindex_topk_c.py`, which is itself
a verbatim copy of the incumbent plus Option C's `q_layout` plumbing.  The
production module `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`
stays untouched so the incumbent remains measurable as a candidate.

`results.md` recommends three changes, each measured ALONE against the
incumbent.  This module is the first time they exist in one op.  Composition
was listed as an unmeasured gap in §12; `bench_merged_configs.py` closes it.

The three, and where each lands:

  (1) candidate 0 -- bkv_p.  A CALL-SITE change, not a kernel one: the decode
      slot of `num_kv_pages_per_block` should be `cdiv(max_model_len,
      page_size)` rather than the production constant 2.  Exposed here as
      `recommended_num_kv_pages_per_block()`.  Measured alone: -14.6% on the
      full decode op at p=9.  It is NOT a bare constant 9 -- fixed p=9 is
      +38.0% at ctx 4096, so it must be recomputed per config.

  (2) D1 -- delete the early-exit `lax.cond`.  Production sets
      `STREAMIDX_ENABLE_EARLY_EXIT = True` (mla_attention_op.py:57), so the
      cond IS live in production and deleting it is a real change, not a
      no-op.  Here the `_fast_path` branch and the `cond` are gone outright:
      `streamindex_topk` always takes `_common_path`.  Measured alone: -10.0%.
      XLA compiles each cond branch into its own computation per call site, so
      CSE cannot span the boundary; removing it lets the 21 redundant
      computations collapse to 1.  The *hoist* the option was named for is
      worth 0.01% on its own -- the cond boundary was the whole effect.

  (3) Option C -- head-major score layout, GATED TO PREFILL/MIXED.  The two
      DECODE `run_scores_kernel` calls get `q_layout="head_minor"` (the
      incumbent, byte-for-byte); the MIXED call gets `"head_major"`.  The gate
      is not a refinement, it is load-bearing: C measured -17.8 to -20.7% on
      prefill stage A and **+59.62% at decode**, where bq_sz=1 makes the
      permutation the identity and the compiler still fails to make it free.
      Controlled by `head_major_prefill`; set False to get C out of the way.

`head_major_hbm` is dropped.  It does not compile -- Mosaic requires sublane
tile-8 alignment the (H, T, D) layout violates -- which retired R3's -9.24%
projection.  `prepare_q_inputs` is therefore always called head-minor, and the
HBM layout of `q`, `_fetch_bq`'s DMA and `mla_attention_op.py` are untouched.

`mm_dtype` is kept only so the Option F control stays runnable from the same
module; "native" is the incumbent and the only value anyone should pass.
Option F measured +9.80% decode / +4.63% prefill and is rejected.
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

# The three q/score row orderings this module can compile.  See the module
# docstring.  `head_minor` must stay bit-identical to the incumbent -- it is
# the control every measurement in RESULTS.md is taken against.
Q_LAYOUTS = ("head_minor", "head_major", "head_major_hbm")


def bkv_double_buffer_bytes(
    bkv_p: int,
    page_size: int,
    lkv_dim: int = 256,
    decode_req_batch_size: int = 4,
) -> int:
    """VMEM held by the decode bkv double buffer alone.

    `bkv_x2_ref` is `[2, seq_batch_size, bkv_p * page_size // kv_packing,
    kv_packing, lkv_dim]` of uint8, so the `kv_packing` factor cancels and the
    size is `2 * seq_batch_size * bkv_p * page_size * lkv_dim` bytes.  At the
    production geometry (page 1024, lkv 256, decode_req_batch_size 4) that is
    exactly **2 MiB per unit of `bkv_p`**, which is what makes the decode slot
    the one parameter here that can walk off the end of VMEM.
    """
    return 2 * decode_req_batch_size * bkv_p * page_size * lkv_dim


# The TensorCore's physical VMEM on tpu7x.  A Mosaic lowering that asks for
# more fails with `Used <n> of 63.94M vmem`, and raising `vmem_limit_bytes`
# past it does NOT move the wall -- that flag caps the compiler's budget, it
# does not conjure SRAM.  Measured: bkv_p=32 (a 64 MiB double buffer) reports
# `Used 85.00M of 63.94M vmem` and will not lower at any vmem_limit_bytes.
TENSORCORE_VMEM_BYTES = 63.94 * 1024 * 1024

# How much of that the decode bkv double buffer is allowed to claim: 32 MiB,
# i.e. `bkv_p <= 16` at the production geometry.  This bound is measured at
# both ends rather than reasoned to.  `p = 32` (64 MiB) fails: `Used 85.00M of
# 63.94M vmem`, and the ~21 MiB above the buffer is the rest of the kernel's
# VMEM, which a 64 MiB buffer leaves no room for.  `p = 16` (32.00 MiB) lowers
# and runs -- confirmed on job j-f06ee0fd-8592-46cf-a2ce at pages_per_seq 64,
# where it was the arm that produced -16.1% on full decode op time.
#
# A first cut at this constant used half the physical VMEM, which is 31.97 MiB
# and excluded `p = 16` by 0.03 MiB.  That is the shape of error to watch for
# here: a budget picked for its roundness silently costs a real rung, and the
# only reason it was caught is that the sweep harness probes one rung above
# whatever this function returns and reports when the extra rung compiles.
DEFAULT_BKV_VMEM_BUDGET_BYTES = 32 * 1024 * 1024


def recommended_num_kv_pages_per_block(
    max_model_len: int,
    page_size: int,
    prefill_bkv_p: int = 2,
    *,
    lkv_dim: int = 256,
    decode_req_batch_size: int = 4,
    bkv_vmem_budget_bytes: int = DEFAULT_BKV_VMEM_BUDGET_BYTES,
) -> tuple[int, int, int]:
    """Candidate 0: the decode `bkv_p` that minimises full-op decode time.

    Returns the (decode, prefill, mixed) tuple for `num_kv_pages_per_block`.

    The decode slot wants to be `cdiv(max_model_len, page_size)` -- the number
    of pages a maximally-long sequence occupies, i.e. the whole context in one
    bkv block.  Measured -14.6% on the full decode op at the production shape,
    where that expression evaluates to 9, against the production constant 2.

    Two things this is NOT.  It is not the constant 9: the argmin tracks
    `cdiv(ctx, page_size)` and a fixed 9 costs +38.0% at ctx 4096, so a server
    that serves several context buckets wants this recomputed per bucket and
    compiled per bucket.  And it is not the `9 = 3 x 3` exactness story that
    first motivated it -- that was falsified by its own author (p=5 is inexact
    and ties p=3).  The real model is a two-regime blocks/pages trade: fewer
    bkv blocks means fewer score-matmul launches and fewer output DMAs, until
    the per-block page fetch stops fitting.

    **The VMEM clamp.**  "The whole context in one bkv block" stops being
    compilable well before it stops being desirable.  The double buffer is
    2 MiB per unit of `bkv_p` at the production geometry, so a 32k context
    (p=32) asks for 64 MiB against a 63.94 MiB physical VMEM and does not
    lower at all.  Returning that tuple is not a slow answer, it is a crash,
    and at max_model_len >= 32768 the unclamped version returned nothing else.
    So the decode slot walks the block-count ladder `cdiv(pages_per_seq, nb)`
    for nb = 1, 2, 3, ... and takes the first entry inside
    `bkv_vmem_budget_bytes`.  That ladder is the right fallback under the
    two-regime model above: it minimises the bkv block count first, and among
    tuples with equal block count it picks the smallest `p`, which overfetches
    least on the final partial block.

    The clamp is analytic and deliberately conservative -- it bounds only the
    term that scales with `bkv_p`.  It can still return a `p` that fails to
    lower for some *other* reason at an unusual geometry, so a caller sweeping
    configurations should treat it as the first rung and be willing to descend.

    **The alignment filter.**  A rung is skipped when it would widen the score
    matrix.  `num_sublanes_total = max_p align_to(pages_per_seq, p) *
    page_size // 128` is taken over the whole 3-tuple, so a decode `p` that
    does not divide the page count pads the matrix for *every* launch -- more
    `-inf` init, more positions for stage B's radix select, and a partial bkv
    block that overfetches on top.  At `pages_per_seq = 32` the ladder's next
    rung under the budget is `p = 11`, and `align_to(32, 11) = 33` inflates
    `num_sublanes_total` from 256 to 264: strictly more work than the
    incumbent, dressed as a tuning win.  Rungs are therefore filtered to those
    with `align_to(pages_per_seq, p) <= align_to(pages_per_seq,
    prefill_bkv_p)`, which at `pages_per_seq = 32` selects `p = 8`.

    `bkv_sz = page_size * bkv_p` must be a multiple of 128; the caller-side
    check in `streamindex_topk` raises if it is not.
    """
    assert max_model_len > 0 and page_size > 0
    pages_per_seq = cdiv(max_model_len, page_size)
    baseline_width = align_to(pages_per_seq, prefill_bkv_p)
    decode_bkv_p = prefill_bkv_p
    for num_blocks in range(1, pages_per_seq + 1):
        p = cdiv(pages_per_seq, num_blocks)
        if p < prefill_bkv_p:
            break
        if align_to(pages_per_seq, p) > baseline_width:
            continue
        if bkv_double_buffer_bytes(
                p, page_size, lkv_dim,
                decode_req_batch_size) <= bkv_vmem_budget_bytes:
            decode_bkv_p = p
            break
    return (decode_bkv_p, prefill_bkv_p, prefill_bkv_p)

# Option F: the dtype the score matmul's OPERANDS are widened to before the
# einsum.  "native" is the incumbent -- float8_e4m3fn x float8_e4m3fn, which
# `TpuInfo.is_matmul_supported` admits on generation 7 and which measures
# 1419.9 Tops/s against bf16's 764.8.  The other two are strictly slower on
# this chip and exist to measure how much of stage A the MXU actually owns.
MM_DTYPES = ("native", "bf16", "f32")


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
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    # ... or [num_q_heads, max_num_tokens, head_dim] when q_layout is
    # "head_major_hbm".
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim], Memory: HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM
    # Scratch
    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim], Memory: VMEM
    bq_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads, head_dim], Memory: VMEM
    # ... or [2, seq_batch_size, num_q_heads, bq_sz, head_dim] when q_layout is
    # "head_major_hbm".
    bq_weights_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads], Memory: VMEM
    scores_block_x2_ref,  # Shape: [2, seq_batch_size * bq_sz, num_sublanes_bkv, 128], Memory: VMEM
    sems,  # Shape: [4, 2, seq_batch_size], Memory: Semaphore
    *,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    q_layout: str = "head_minor",
    mm_dtype: str = "native",
):
    """Core kernel logic that operates on memory references."""

    assert q_layout in Q_LAYOUTS, q_layout
    assert mm_dtype in MM_DTYPES, mm_dtype
    head_major = q_layout in ("head_major", "head_major_hbm")
    q_hbm_head_major = q_layout == "head_major_hbm"

    if q_hbm_head_major:
        num_q_heads, _, head_dim = q_hbm_ref.shape
    else:
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

            # `head_major_hbm` puts the head axis first in both HBM and VMEM,
            # so the q copy becomes a strided one: `num_q_heads` runs of
            # `sz * head_dim` elements.  The *weights* stay token-major in HBM
            # in every layout -- a (num_q_heads, max_num_tokens) weights array
            # would put the dynamic `sz` slice on the LANE axis, which Mosaic
            # requires to be a multiple of 128.  The (bq_sz, H) -> (H, bq_sz)
            # weights transpose is done in registers instead; it is 128x32.
            if q_hbm_head_major:
                q_src = q_hbm_ref.at[:, pl.ds(q_len_start, sz)]
                q_dst = bq_vmem_ref.at[:, pl.ds(0, sz)]
            else:
                q_src = q_hbm_ref.at[pl.ds(q_len_start, sz)]
                q_dst = bq_vmem_ref.at[pl.ds(0, sz)]

            if not wait:
                _async_copy(q_src, q_dst, sem, wait=False)
                _async_copy(
                    indexer_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_weights_vmem_ref.at[pl.ds(0, sz)],
                    weights_sem,
                    wait=False,
                )
            else:
                _async_copy(q_dst, q_dst, sem, wait=True)
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
        # The einsum LHS row ordering.  `head_minor` is (bq_sz * H); Option C
        # is (H * bq_sz), which puts a whole bq_sz=128 row block -- 16 vregs --
        # between consecutive heads, so `sum_h` becomes an aligned vreg add
        # chain instead of a sublane rotate tree.
        if q_hbm_head_major:
            # already (SB, H, bq_sz, D) in VMEM; the reshape is free.
            data = bq_x2_ref.at[bq_sem_idx, :, :, :bq_sz][...].reshape(
                seq_batch_size, num_q_heads * bq_sz, head_dim)
        elif head_major:
            # (SB, bq_sz, H, D) -> (SB, H, bq_sz, D).  Paid once per bq block.
            data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].transpose(
                0, 2, 1, 3).reshape(seq_batch_size, num_q_heads * bq_sz,
                                    head_dim)
        else:
            data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].reshape(
                seq_batch_size, bq_sz * num_q_heads, head_dim)
        bqs = []
        for batch_idx in range(seq_batch_size):
            bqs.append(data[batch_idx])
        return bqs

    def load_bq_weights(bq_sem_idx):
        data = bq_weights_x2_ref.at[bq_sem_idx, :, :bq_sz][...]
        if head_major:
            # (SB, bq_sz, H) -> (SB, H, bq_sz), matching the score block's
            # new axis order.  Also once per bq block.
            data = data.transpose(0, 2, 1)
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
            )  # (seq_batch_size, bq_sz * num_q_heads, head_dim), or
            # (seq_batch_size, num_q_heads * bq_sz, head_dim) head-major.
            bkv_stacked = jnp.stack(
                bkv_vec)  # Shape: (seq_batch_size, bkv_sz, head_dim)
            scale_val_stacked = jnp.stack(
                scale_val_vec)  # Shape: (seq_batch_size, 1, bkv_sz)
            bq_weights_stacked = jnp.stack(
                bq_weights_vec
            )  # (seq_batch_size, bq_sz, num_q_heads), or
            # (seq_batch_size, num_q_heads, bq_sz) head-major.
            bq_pos_compressed_stacked = jnp.stack(
                bq_pos_compressed_vec)  # Shape: (seq_batch_size, bq_sz)

            # Compute in Registers
            if mm_dtype != "native":
                # Option F's arithmetic half.  Both operands arrive as
                # float8_e4m3fn; widening them is EXACT (3 mantissa bits fit in
                # bf16's 7 and in f32's 23), so this changes the MXU pass the
                # compiler picks and nothing else.  It does not change a single
                # byte of HBM traffic -- see RESULTS.md 10.2.
                _t = {"bf16": jnp.bfloat16, "f32": jnp.float32}[mm_dtype]
                bq_stacked = bq_stacked.astype(_t)
                bkv_stacked = bkv_stacked.astype(_t)
            s = jnp.einsum(
                "bnd,bmd->bnm",
                bq_stacked,
                bkv_stacked,
                preferred_element_type=jnp.float32,
            )  # Shape: (seq_batch_size, rows, bkv_sz)
            if head_major:
                # (SB, H, bq_sz, bkv_sz); the head axis is now the OUTER one,
                # 16 vregs apart, so the reduce is a whole-vreg add chain.
                s = s.reshape(seq_batch_size, num_q_heads, bq_sz, bkv_sz)
                s = jnp.maximum(s, 0.0)
                s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
                s_summed = s.sum(axis=1)
            else:
                s = s.reshape(seq_batch_size, bq_sz, num_q_heads, bkv_sz)
                s = jnp.maximum(s, 0.0)
                s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
                s_summed = s.sum(
                    axis=2)  # Shape: (seq_batch_size, bq_sz, bkv_sz)

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
        q_layout: str = "head_minor",
):
    """Pad q, and -- for `head_major_hbm` only -- transpose it head-major.

    The transpose is a real XLA op on a (T, H, D) fp8 array outside the Pallas
    call.  It is NOT free and it is NOT part of stage A; `bench_headmajor.py`
    times it separately so the `head_major_hbm` arm cannot silently bank a win
    it moved somewhere else.
    """
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
    if q_layout == "head_major_hbm":
        q = jnp.transpose(q, (1, 0, 2))  # (H, T, D)
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


# Named rather than inlined into the decorator so that `run_correctness_c.py`,
# which rebuilds this jit with `q_layout` already bound, cannot drift from it.
# A rebuilt jit that dropped a static argname would still run and would quietly
# retrace on every call.
STREAMINDEX_TOPK_STATIC_ARGNAMES = (
    "k",
    "compression_ratio",
    "num_kv_pages_per_block",
    "num_queries_per_block",
    "vmem_limit_bytes",
    "decode_req_batch_size",
    "head_major_prefill",
    "mm_dtype",
    "num_chunks",
)


def chunk_metadata(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    a: int,  # STATIC: chunk token range start
    b: int,  # STATIC: chunk token range end (exclusive)
):
    """Traced per-chunk `(seq_lens, cu_q_lens, mixed_start, mixed_end)`.

    `a` and `b` are static -- they come from `num_chunks` and `q.shape[0]`,
    both of which are static under `jax.jit`.  Everything returned is traced.
    That asymmetry is the finding: the plan does NOT have to be handed in as
    static metadata by the runner, which is what `option-b-pipeline/RESULTS.md`
    section 7 assumed it would.

    Module-level, not nested in `streamindex_topk`, so `probe_runner.py --arm
    plan` checks the code that actually runs rather than a transcription of it.
    """
    max_num_seqs = seq_lens.shape[0]
    ends = cu_q_lens[1:max_num_seqs + 1]
    cu_q_lens_c = jnp.clip(cu_q_lens, a, b) - a
    # Keeps every token's absolute causal position
    #   seq_len - q_len + (t - cu_q_lens[seq])
    # unchanged when the token range is narrowed to [a, b).  Clamped at 0 only
    # for sequences that start after the chunk, which are never launched and
    # never indexed.
    seq_lens_c = jnp.maximum(seq_lens - jnp.maximum(0, ends - b), 0)
    # `searchsorted(ends, x, side="right")` is what this counts, written as a
    # masked sum instead: entries past `distribution[2]` are padding and this
    # module will not assume they are sorted.  `_effective_row_lengths` guards
    # them the same way, which is where the doubt comes from.
    valid_seq = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
    s = jnp.sum(jnp.where(valid_seq, ends <= a, False).astype(jnp.int32))
    e = jnp.sum(jnp.where(valid_seq, ends <= b - 1, False).astype(jnp.int32)) + 1
    mixed_start = jnp.clip(s, distribution[1], distribution[2])
    mixed_end = jnp.clip(jnp.maximum(e, mixed_start), mixed_start,
                         distribution[2])
    return seq_lens_c, cu_q_lens_c, mixed_start, mixed_end


def plan_chunks(
    num_tokens: int,
    max_num_seqs: int,
    num_chunks: int,
) -> dict:
    """Static half of the Option B plan.  Nothing traced, nothing run.

    Returns the chunk token ranges, whether each chunk needs the two decode
    launches, the resulting launch count, and the guard's verdict.  Raises
    nothing -- `ok=False` is a reportable result, and `streamindex_topk` is
    what refuses.
    """
    if num_chunks < 1:
        return {"ok": False, "reason": f"num_chunks={num_chunks} < 1"}
    if num_chunks == 1:
        return {
            "ok": True,
            "num_chunks": 1,
            "chunk_size": num_tokens,
            "ranges": [(0, num_tokens)],
            "decode_launch": [True],
            "num_launches": 3,
            "reason": "unchunked path",
        }
    if num_tokens % num_chunks != 0:
        return {
            "ok": False,
            "reason": (f"num_tokens={num_tokens} not divisible by "
                       f"num_chunks={num_chunks}"),
        }
    t = num_tokens // num_chunks
    if t < max_num_seqs:
        return {
            "ok": False,
            "reason": (f"chunk_size={t} < max_num_seqs={max_num_seqs}: a "
                       "decode token could fall outside chunk 0, so the "
                       "decode launches cannot be statically omitted"),
        }
    return {
        "ok": True,
        "num_chunks": num_chunks,
        "chunk_size": t,
        "ranges": [(i * t, (i + 1) * t) for i in range(num_chunks)],
        "decode_launch": [i == 0 for i in range(num_chunks)],
        "num_launches": 2 + num_chunks,
        "reason": "ok",
    }


@functools.partial(jax.jit, static_argnames=STREAMINDEX_TOPK_STATIC_ARGNAMES)
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
    head_major_prefill: bool = True,
    mm_dtype: str = "native",
    num_chunks: int = 1,
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
    head_major_prefill: apply Option C's head-major score layout to the
      prefill/mixed kernel launch. Decode always stays head-minor -- see (3)
      in the module docstring. Defaults to True.
    decode_req_batch_size: maximum decode batch size per iteration.
    num_chunks: Option B.  Split the token axis into this many equal chunks so
      that chunk i's stage B (SparseCore) overlaps chunk i+1's stage A
      (TensorCore).  Static.  1 is the unchunked merged path.  See the module
      docstring for the metadata construction and the `chunk_size >=
      max_num_seqs` guard.

    There is no `enable_early_exit`: D1 deleted the cond it selected.

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

    if mm_dtype not in MM_DTYPES:
        raise ValueError(f"mm_dtype must be one of {MM_DTYPES}, got {mm_dtype}")

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype
    # Captured before `prepare_q_inputs`, which may move the token axis.
    max_num_tokens = q.shape[0]

    # Option B.  Refuse rather than silently drop rows -- see `plan_chunks`.
    chunk_plan = plan_chunks(max_num_tokens, max_num_seqs, num_chunks)
    if not chunk_plan["ok"]:
        raise ValueError(f"num_chunks={num_chunks} rejected: "
                         f"{chunk_plan['reason']}")

    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    # Always head-minor in HBM: Option C's transpose is done in registers
    # inside the kernel, once per bq block.  `head_major_hbm` is dropped (see
    # the module docstring), so nothing outside this file changes layout.
    q = prepare_q_inputs(q, "head_minor")
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
        q_layout="head_minor",
    ):
        if q_layout == "head_major_hbm":
            num_q_heads, num_tokens, head_dim = q.shape
        else:
            num_tokens, num_q_heads, head_dim = q.shape
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
        if q_layout == "head_major_hbm":
            bq_buf_shape = (2, seq_batch_size, num_q_heads, bq_sz, head_dim)
        else:
            bq_buf_shape = (2, seq_batch_size, bq_sz, num_q_heads, head_dim)
        bq_double_bufq = pltpu.VMEM(bq_buf_shape, q.dtype)
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

        # The program name is what `/xprof-llo`'s reader selects on, so the
        # layout has to be in it or the two arms' dumps are indistinguishable.
        suffix = {
            "head_minor": "",
            "head_major": "-hm",
            "head_major_hbm": "-hmhbm",
        }[q_layout] + ("" if mm_dtype == "native" else f"-{mm_dtype}")
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
                mm_dtype=mm_dtype,
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
                shape=(num_tokens, num_sublanes_total, 128),
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

    def _chunk(a: int, chunk_len: int, with_decode: bool):
        """One chunk's stage A launches + its stage B reduction.

        With `num_chunks == 1` this is the merged module's `_common_path`
        verbatim: `a == 0`, `chunk_len == max_num_tokens`, `with_decode`, and
        the `unchunked` branch below hands back the caller's own arrays.
        """
        out_dtype = jnp.float32
        b = a + chunk_len
        unchunked = (a == 0 and chunk_len == max_num_tokens)

        if unchunked:
            q_c = q
            w_c = prepared_indexer_weights
            seq_lens_c, cu_q_lens_c = seq_lens, cu_q_lens
            mixed_start, mixed_end = distribution[1], distribution[2]
        else:
            q_c = jax.lax.slice_in_dim(q, a, b, axis=0)
            w_c = jax.lax.slice_in_dim(prepared_indexer_weights, a, b, axis=0)
            (seq_lens_c, cu_q_lens_c, mixed_start,
             mixed_end) = chunk_metadata(seq_lens, cu_q_lens, distribution, a,
                                         b)

        # Pre-fill the output with -inf and alias it, so masked / unwritten
        # columns are already -inf.
        scores = jnp.full(
            (chunk_len, num_sublanes_total, 128),
            -jnp.inf,
            dtype=out_dtype,
        )

        if with_decode:
            # TODO: we shall sort the sequences by length, so that multiple
            # decode sequences in one batch have similar lengths to reduce
            # waste of compute.  With the same batch size, the longest
            # sequence will determine number of blocks to run computation for.
            decode_batch_end = (distribution[0] // decode_req_batch_size *
                                decode_req_batch_size)
            scores = run_scores_kernel(
                q_c,
                w_c,
                cache_kv,
                scores,
                seq_lens_c,
                page_indices,
                cu_q_lens_c,
                num_kv_pages_per_block=num_kv_pages_per_blocks[0],
                num_queries_per_block=num_queries_per_blocks[0],
                start_seq_idx=jnp.array(0),
                end_seq_idx=decode_batch_end,
                static_q_len=1,
                seq_batch_size=decode_req_batch_size,
                out_dtype=out_dtype,
                case=MlaCase.DECODE,
                # (3) the gate.  Decode keeps the incumbent layout: at bq_sz=1
                # the head-major permutation is the identity and still
                # measured +59.62%.
                q_layout="head_minor",
            )
            # Handle num_decode_seqs % decode_req_batch_size != 0 case.
            scores = run_scores_kernel(
                q_c,
                w_c,
                cache_kv,
                scores,
                seq_lens_c,
                page_indices,
                cu_q_lens_c,
                num_kv_pages_per_block=num_kv_pages_per_blocks[0],
                num_queries_per_block=num_queries_per_blocks[0],
                start_seq_idx=decode_batch_end,
                end_seq_idx=distribution[1],
                static_q_len=1,
                seq_batch_size=1,
                out_dtype=out_dtype,
                case=MlaCase.DECODE,
                q_layout="head_minor",
            )

        scores = run_scores_kernel(
            q_c,
            w_c,
            cache_kv,
            scores,
            seq_lens_c,
            page_indices,
            cu_q_lens_c,
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=mixed_start,
            end_seq_idx=mixed_end,
            static_q_len=None,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.MIXED,
            # (3) Option C, applied only here.  bq_sz is 128 at this call.
            q_layout="head_major" if head_major_prefill else "head_minor",
        )

        scores = scores.reshape(chunk_len, -1)
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
            seq_lens=seq_lens_c,
            cu_q_lens=cu_q_lens_c,
            distribution=distribution,
            num_tokens=chunk_len,
            num_positions=scores_to_reduce.shape[1],
            compression_ratio=compression_ratio,
        )
        topk_idxs = sparsecore_topk(
            scores_to_reduce,
            k,
            row_lengths=eff_row_lengths,
            write_empty_rows=True,
        )

        return topk_idxs[:chunk_len, :k]

    def _common_path(_):
        if num_chunks == 1:
            return _chunk(0, max_num_tokens, True)
        # Option B.  Chunk i's stage B (SparseCore) has no dependency on
        # chunk i+1's stage A (TensorCore), so XLA is free to overlap them.
        # Nothing here serialises the chunks: the concatenate is the only
        # join.
        parts = [
            _chunk(a, b - a, with_decode)
            for (a, b), with_decode in zip(chunk_plan["ranges"],
                                           chunk_plan["decode_launch"])
        ]
        return jnp.concatenate(parts, axis=0)

    # (2) D1.  The early-exit `lax.cond` and its `_fast_path` branch are
    # deleted, not disabled.  Production runs with
    # STREAMIDX_ENABLE_EARLY_EXIT = True, so the cond is live there; XLA
    # compiles each branch into its own computation per call site, which
    # blocks CSE across the boundary and leaves 21 redundant computations
    # standing.  Removing it collapses them to 1.  Measured -10.0%.
    #
    # What is given up: when `max(seq_lens) // compression_ratio <= k` every
    # row's answer is the identity permutation and `_fast_path` returned it
    # without touching the kernel.  `_common_path` computes the same answer
    # the long way.  That case does not arise at any production context
    # length -- it needs the whole compressed sequence to fit in k.
    return _common_path(None)
