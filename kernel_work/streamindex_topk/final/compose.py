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
"""Build `streamindex_topk_composed.py` = incumbent + Option B + Option A.

Run from the repo root:

    python3 kernel_work/streamindex_topk/final/compose.py

WHY A SCRIPT AND NOT A HAND-EDITED FILE
---------------------------------------
The composed candidate has to be auditable as *exactly* B's diff plus A's diff
and nothing else. A hand-merged 1100-line file is not auditable; a list of
named, exact-match string replacements is. Every replacement below asserts it
matched exactly once, so a silent no-op -- the failure mode that would let a
half-applied variant be measured as a composition -- is impossible.

Base is `variants/right-size/streamindex_topk_rs.py` (Option B, commits
9bc5f5cd / 36aedcdc / a8197f38). The edits applied on top are Option A's
(`variants/narrow-record/narrow_record.diff`, commit 6b14ae8a), transcribed
hunk by hunk with the hunk number in each step's name.

Option E is NOT here: it is a `num_kv_pages_per_block` tuple at the CALL SITE
(`variants/retune/retune.patch` touches `mla_attention_op.py`, not this file),
so it composes by rebinding `shapes.NUM_KV_PAGES_PER_BLOCK` in the driver.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, os.pardir, "variants", "right-size",
                    "streamindex_topk_rs.py")
OUT = os.path.join(HERE, "streamindex_topk_composed.py")

DOCSTRING = '''"""TPU-Friendly StreamIndex Top-K kernel.

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
"""'''

# One entry per edit: (name, old, new). Each `old` must appear exactly once.
EDITS = []


def edit(name, old, new, count=1):
    """Register a replacement that MUST match `count` times, never fewer or more.

    The count is explicit rather than "replace all" on purpose: a silently
    half-applied variant measured as a composition is the exact failure this
    script exists to make impossible.
    """
    EDITS.append((name, old, new, count))


# ---------------------------------------------------------------------------
# A hunk 1 -- module docstring. B rewrote it; the composition rewrites it again
# and adds the section 6.5 layout opt-out paragraph, which is an obligation on
# whichever option ships FIRST and is discharged here.
# ---------------------------------------------------------------------------
def _replace_docstring(src):
    start = src.index('"""TPU-Friendly StreamIndex Top-K kernel')
    end = src.index('"""', src.index("\n", start)) + 3
    return src[:start] + DOCSTRING + src[end:]


# ---------------------------------------------------------------------------
# A hunk 2 -- the `kernel` positional signature.
# ---------------------------------------------------------------------------
edit(
    "A2a: cache_scale_hbm_ref into the kernel signature",
    "    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing,"
    " kv_packing, lkv_dim], Memory: HBM\n"
    "    scores_in_hbm_ref,",
    "    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing,"
    " kv_packing, lkv_dim=128], Memory: HBM\n"
    "    cache_scale_hbm_ref,  # Shape: [total_num_pages, page_size//128, 128],"
    " HBM\n"
    "    scores_in_hbm_ref,",
)

edit(
    "A2b: bkv_scale_x2_ref scratch into the kernel signature",
    "    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing,"
    " kv_packing, lkv_dim], Memory: VMEM\n"
    "    bq_x2_ref,",
    "    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing,"
    " kv_packing, lkv_dim=128], Memory: VMEM\n"
    "    bkv_scale_x2_ref,  # Shape: [2, seq_batch_size, bkv_p, 8, 128], VMEM\n"
    "    bq_x2_ref,",
)

edit(
    "A2c: a fifth semaphore bank for the scale DMA",
    "    sems,  # Shape: [4, 2, seq_batch_size], Memory: Semaphore",
    "    sems,  # Shape: [5, 2, seq_batch_size], Memory: Semaphore",
)

# ---------------------------------------------------------------------------
# A hunk 3 -- `page_size` is needed inside the kernel body for the scale
# plane's trailing shape.
# ---------------------------------------------------------------------------
edit(
    "A3: page_size in the kernel body",
    "    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing\n"
    "    bkv_sz = bkv_sz_per_kv_packing * kv_packing",
    "    page_size = page_size_per_kv_packing * kv_packing\n"
    "    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing\n"
    "    bkv_sz = bkv_sz_per_kv_packing * kv_packing",
)

# ---------------------------------------------------------------------------
# A hunks 4-6 -- `_fetch_bkv`: one extra DMA per page for the scale tile.
# ---------------------------------------------------------------------------
edit(
    "A4: scale semaphore and VMEM ref per batch slot",
    "            sem = sems.at[0, bkv_sem_idx, batch_idx]\n"
    "            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]\n",
    "            sem = sems.at[0, bkv_sem_idx, batch_idx]\n"
    "            scale_sem = sems.at[4, bkv_sem_idx, batch_idx]\n"
    "            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]\n"
    "            bkv_scale_vmem_ref = bkv_scale_x2_ref.at[bkv_sem_idx,"
    " batch_idx]\n",
)

edit(
    "A5a: name the raw page index so the scale DMA can reuse it",
    "                    page_idx = jnp.minimum(page_indices_offset + i,\n"
    "                                           num_page_indices - 1)\n"
    "                    safe_page_offset = jnp.minimum(\n"
    "                        page_indices_ref[page_idx] *"
    " page_size_per_kv_packing,",
    "                    page_idx = jnp.minimum(page_indices_offset + i,\n"
    "                                           num_page_indices - 1)\n"
    "                    raw_page = page_indices_ref[page_idx]\n"
    "                    safe_page_offset = jnp.minimum(\n"
    "                        raw_page * page_size_per_kv_packing,",
)

edit(
    "A5b: the scale-plane DMA, and the wait for it",
    "                        sem,\n"
    "                        wait=False,\n"
    "                    )\n"
    "            else:\n"
    "                dma_bkv_sz = bkv_p * page_size_per_kv_packing\n"
    "                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]\n"
    "                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)\n",
    "                        sem,\n"
    "                        wait=False,\n"
    "                    )\n"
    "\n"
    "                    # NARROW RECORD: the scale plane is a second, 128x\n"
    "                    # smaller array. One DMA per page moves that page's\n"
    "                    # whole (8, 128) scale tile -- 1024 B, a whole number\n"
    "                    # of tiles, no padding -- into slot `i` of the VMEM\n"
    "                    # buffer. Slicing dimension 0 is legal precisely\n"
    "                    # because the 3-D declaration leaves the page axis\n"
    "                    # untiled; see the module docstring for the 2-D form\n"
    "                    # Mosaic rejected.\n"
    "                    safe_scale_page = jnp.minimum(\n"
    "                        jnp.maximum(raw_page, 0), total_num_pages - 1)\n"
    "                    _async_copy(\n"
    "                        cache_scale_hbm_ref.at[pl.ds(safe_scale_page, 1)],\n"
    "                        bkv_scale_vmem_ref.at[pl.ds(i, 1)],\n"
    "                        scale_sem,\n"
    "                        wait=False,\n"
    "                    )\n"
    "            else:\n"
    "                dma_bkv_sz = bkv_p * page_size_per_kv_packing\n"
    "                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]\n"
    "                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)\n"
    "                dst_sc = bkv_scale_vmem_ref.at[pl.ds(0, bkv_p)]\n"
    "                _async_copy(src=dst_sc, dst=dst_sc, sem=scale_sem,"
    " wait=True)\n",
)

# ---------------------------------------------------------------------------
# A hunk 7 -- `load_bkv`: the slice-and-transpose disappears.
# ---------------------------------------------------------------------------
edit(
    "A7: load_bkv reads the separate scale plane",
    "            flat_bkv = bkv.reshape(-1, bkv.shape[-1])\n"
    "            fp8_val = flat_bkv[:, :head_dim]\n"
    "            fp8_val = pltpu.bitcast(fp8_val, jnp.float8_e4m3fn)\n"
    "            scale_val = pltpu.bitcast(flat_bkv[:, head_dim:head_dim + 1].T,"
    "\n"
    "                                      jnp.float8_e8m0fnu).astype("
    "jnp.bfloat16)\n",
    "            # NARROW RECORD: the DMA landed exactly the 128 value bytes,\n"
    "            # so there is no slice-and-recombine here any more. The\n"
    "            # `[:, :128]` slice and the `[:, 128:129].T` transpose both\n"
    "            # disappear.\n"
    "            flat_bkv = bkv.reshape(-1, bkv.shape[-1])\n"
    "            fp8_val = pltpu.bitcast(flat_bkv, jnp.float8_e4m3fn)\n"
    "            # [bkv_p, 8, 128] u8 -> e8m0 -> f32 -> [1, bkv_sz].\n"
    "            # The widen happens BEFORE the regroup so the relayout runs on\n"
    "            # an f32 (8,128) tile rather than a packed 8-bit one. e8m0 is\n"
    "            # a bare exponent and f32 has the same 8 exponent bits, so the\n"
    "            # widen is exact and this stays bit-identical to the incumbent\n"
    "            # (which widens to bf16 -- also exact -- and promotes in the\n"
    "            # multiply).\n"
    "            raw_scale = bkv_scale_x2_ref.at[bkv_sem_idx, batch_idx][...]\n"
    "            scale_val = pltpu.bitcast(\n"
    "                raw_scale, jnp.float8_e8m0fnu).astype(jnp.float32).reshape("
    "\n"
    "                    1, bkv_sz)\n",
)

# ---------------------------------------------------------------------------
# A hunks 8-10 -- the public signature and the tuple unpack.
# ---------------------------------------------------------------------------
edit(
    "A8: public signature takes the two-tuple",
    "    cache_kv: jax.\n"
    "    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing,"
    " head_dim]",
    "    cache_kv: tuple,  # (u8[num_pages, page_size//kv_packing, kv_packing,"
    " 128],\n"
    "    #                  u8[num_pages, page_size//128, 128])",
)

edit(
    "A9: docstring for the tuple",
    "    cache_kv: the current kv cache.",
    "    cache_kv: the current kv cache, as the two-tuple (values, scales).",
)

edit(
    "A10: the packed-scale comment is no longer true",
    "    # Scale factors for the FP8 index cache format are packed directly"
    " inside\n"
    "    # `cache_kv` along the width dimension, keeping HBM transactions"
    " fused.",
    "    # NARROW RECORD: scale factors live in their OWN array, not inside the\n"
    "    # value record. That is the whole change; see the module docstring.",
)

edit(
    "A11: unpack the tuple",
    "    lkv_dim = cache_kv.shape[-1]\n"
    "    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape",
    "    # NARROW RECORD: `cache_kv` arrives as the two-tuple\n"
    "    #   (u8[num_pages, page_size//kv_packing, kv_packing, 128],\n"
    "    #    u8[num_pages, page_size//128, 128])\n"
    "    # so that the public signature is unchanged.\n"
    "    cache_val, cache_scale = cache_kv\n"
    "    lkv_dim = cache_val.shape[-1]\n"
    "    _, page_size_per_kv_packing, kv_packing, _ = cache_val.shape",
)

# ---------------------------------------------------------------------------
# A hunks 11-16 -- `run_scores_kernel`.
# ---------------------------------------------------------------------------
edit(
    "A12: run_scores_kernel takes cache_scale",
    "    def run_scores_kernel(\n"
    "        q,\n"
    "        prepared_indexer_weights,\n"
    "        cache_kv,\n"
    "        scores_init,",
    "    def run_scores_kernel(\n"
    "        q,\n"
    "        prepared_indexer_weights,\n"
    "        cache_kv,\n"
    "        cache_scale,\n"
    "        scores_init,",
)

edit(
    "A13: in_specs gains the scale plane",
    "            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv\n"
    "        ]\n"
    "        if aliased:",
    "            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv (values only)"
    "\n"
    "            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_scale\n"
    "        ]\n"
    "        if aliased:",
)

edit(
    "A14: the scale double buffer, and the fifth semaphore bank",
    "        scratch_shapes = [\n"
    "            bkv_double_buf,\n"
    "            bq_double_bufq,\n"
    "            bq_weights_double_buf,\n"
    "            bo_scores_double_buf,\n"
    "            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),\n"
    "        ]",
    "        # NARROW RECORD: one extra, tiny double buffer for the scale plane.\n"
    "        bkv_scale_double_buf = pltpu.VMEM(\n"
    "            (2, seq_batch_size, bkv_p, page_size // 128, 128),\n"
    "            cache_scale.dtype,\n"
    "        )\n"
    "\n"
    "        scratch_shapes = [\n"
    "            bkv_double_buf,\n"
    "            bkv_scale_double_buf,\n"
    "            bq_double_bufq,\n"
    "            bq_weights_double_buf,\n"
    "            bo_scores_double_buf,\n"
    "            pltpu.SemaphoreType.DMA((5, 2, seq_batch_size)),\n"
    "        ]",
)

# COMPOSITION POINT 1. B splices a `None` into the `scores_in_hbm_ref` slot when
# the band is un-aliased, and computes that slot's index from the operand count
# before it. A adds an operand before it, so B's `+ 3` must become `+ 4`. This
# is the one place where the two diffs are not independent, and getting it wrong
# would shift every ref after it -- silently, into a lowering error at best.
edit(
    "COMPOSE-1: B's None-splice index moves +3 -> +4 because A adds an operand",
    "        n_before_scores_in = len(scalar_prefetches) + 3",
    "        # COMPOSED: `+ 4`, not B's `+ 3`. The operands before `scores_init`\n"
    "        # are now q, prepared_indexer_weights, cache_kv AND cache_scale --\n"
    "        # Option A inserted one. Off by one here shifts the output and all\n"
    "        # five scratch refs.\n"
    "        n_before_scores_in = len(scalar_prefetches) + 4",
)

# COMPOSITION POINT 2. Same index, in the alias map. B made the map conditional;
# A moved the index. Both apply.
edit(
    "COMPOSE-2: the alias index is B's conditional map at A's +4",
    "            input_output_aliases=({\n"
    "                len(scalar_prefetches) + 3: 0\n"
    "            } if aliased else {}),",
    "            input_output_aliases=({\n"
    "                len(scalar_prefetches) + 4: 0\n"
    "            } if aliased else {}),",
)

edit(
    "A15: the scale plane joins the operand list",
    "        operands = [q, prepared_indexer_weights, cache_kv]\n"
    "        if aliased:",
    "        operands = [q, prepared_indexer_weights, cache_kv, cache_scale]\n"
    "        if aliased:",
)

# ---------------------------------------------------------------------------
# A hunks 17-19 -- the three band invocations in `_common_path`.
# ---------------------------------------------------------------------------
edit(
    "A16: DECODE batched band passes the split pair",
    "            prepared_indexer_weights,\n"
    "            cache_kv,\n"
    "            scores_init,\n"
    "            seq_lens,",
    "            prepared_indexer_weights,\n"
    "            cache_val,\n"
    "            cache_scale,\n"
    "            scores_init,\n"
    "            seq_lens,",
)

# The two remaining bands are textually identical to each other (`cache_kv,` then
# `scores,`), so they are replaced together rather than by two ambiguous edits.
edit(
    "A17: DECODE-remainder and MIXED bands pass the split pair",
    "            prepared_indexer_weights,\n"
    "            cache_kv,\n"
    "            scores,\n"
    "            seq_lens,",
    "            prepared_indexer_weights,\n"
    "            cache_val,\n"
    "            cache_scale,\n"
    "            scores,\n"
    "            seq_lens,",
    count=2)  # the DECODE-remainder and MIXED bands are textually identical

# ---------------------------------------------------------------------------
# The bucket ladder. NOT part of either variant diff -- it is the shipping
# plumbing Option B needs and does not have, appended as a module-level helper.
# ---------------------------------------------------------------------------
LADDER = '''

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
'''


def main():
    with open(BASE) as f:
        src = f.read()

    src = _replace_docstring(src)

    for name, old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            print(f"COMPOSE FAILED: [{name}] matched {n} times, expected "
                  f"{count}", file=sys.stderr)
            return 1
        src = src.replace(old, new)
        print(f"ok  {name}  (x{count})")

    src = src.rstrip("\n") + "\n" + LADDER

    with open(OUT, "w") as f:
        f.write(src)
    print(f"\nwrote {OUT} ({len(src.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
