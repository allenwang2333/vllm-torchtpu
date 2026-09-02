#!/usr/bin/env python3
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
"""Step-1 microbenchmark for `streamindex_topk` at the GLM-5.2 indexer shapes.

This is the shared yardstick for every later candidate. It is deliberately
boring: it times the incumbent as the caller invokes it, and it splits the
measurement into the three pieces that are separately attackable.

Arms, per swept shape
---------------------
1. ``total``      -- the whole ``streamindex_topk`` call, as
                     `mla_attention_op.py::_streamidx_topk_jax` invokes it.
2. ``stage_b``    -- ``sparsecore_topk`` alone, at exactly the
                     ``(num_tokens, num_positions, k)`` the op feeds it.
3. ``init``       -- the ``jnp.full(..., -inf)`` score-buffer fill alone
                     (`streamindex_topk.py:745`), a real full-size HBM write.
4. ``stage_a``    -- **derived by subtraction**: ``total - stage_b - init``.
                     It is NOT measured. Any error in the other three lands
                     here, and it also absorbs whatever XLA overlaps between
                     the three, so treat it as an estimate with a sign, not a
                     number to quote to three digits.

Everything the script prints that is not a timing is closed-form arithmetic
from the shape (score-buffer bytes, ``num_sublanes_total``, ``num_positions``,
K-cache bytes streamed). Those are exact and are printed even in ``--dry-run``.

Ratios are reported alongside every absolute, because absolutes drift between
stacks and ratios do not. The implied bandwidths exist so that an implausible
timing shows up as an implausible TB/s rather than as a plausible-looking
speedup.

Not covered here, on purpose: correctness. See
``test_streamidx_topk_correctness.py`` -- a kernel that skips work benchmarks
beautifully.

Usage (from the vllm-torchtpu checkout root)
--------------------------------------------
    # CPU dry run: geometry only, no TPU, no Pallas execution.
    JAX_PLATFORMS=cpu ./uv_venv/bin/python \\
        kernel_work/streamidx-topk/instruments/bench_streamidx_topk.py --dry-run

    # On a TPU (via /run-command-in-cdk):
    ./uv_venv/bin/python \\
        kernel_work/streamidx-topk/instruments/bench_streamidx_topk.py \\
        --json /tmp/streamidx_bench.json

The measured time per shape is milliseconds; the wall clock is Mosaic compile
time, which `compile_s` reports per arm so the two are never confused. Use
`--filter` (substring on the shape name, e.g. `--filter mml65536`) to run a
subset while iterating.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Target configuration -- GLM-5.2 indexer.
#
# Every constant below is read back from the code, with the file:line it came
# from. Do not "fix" one of these without re-reading its source.
# ---------------------------------------------------------------------------

# Indexer query heads / head dim / top-k, and the compression ratio the GLM
# caller passes. `mla_attention_op.py:107-122` fixes compression_ratio=1 and
# the two block tuples; H/D/k come from the model spec and reach the kernel as
# `q.shape` and `k=self.topk_tokens`.
H_IDX = 32
D_IDX = 128
K_IDX = 2048
COMPRESSION_RATIO = 1
NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)  # mla_attention_op.py:119
NUM_QUERIES_PER_BLOCK = (1, 128, 128)  # mla_attention_op.py:120
ENABLE_EARLY_EXIT = True  # mla_attention_op.py:57,121

# KV cache geometry.
#
# page size 1024   -- `PallasMLAttentionBackend.get_page_size` is a literal
#                     `return 1024` (attention.py:1323-1324), and the indexer
#                     backend delegates to it (attention.py:1366-1367).
# kv_packing 4     -- `get_dtype_packing(torch.uint8)` = 32 // 8
#                     (attention.py:85-91, used at attention.py:1248).
# lkv_dim 256      -- the indexer K row is `head_dim` fp8 bytes + one e8m0
#                     scale byte (`VllmTPUIndexer.rebind`,
#                     mla_attention_op.py:235-236 -> head_size = 129), and
#                     `mla_v2_kernel.get_kv_cache_shape` rounds the minor dim
#                     up to a 128 multiple (mla/v2/kernel.py:101-106).
#                     129 -> 256, so 129 of 256 bytes carry information.
PAGE_SIZE = 1024
KV_PACKING = 4
QUANT_BLOCK = D_IDX  # `rebind` asserts quant_block_size == head_dim
LKV_USEFUL_BYTES = D_IDX + D_IDX // QUANT_BLOCK  # 129
BYTES_PER_F32 = 4


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


LKV_DIM = align_to(LKV_USEFUL_BYTES, 128)  # 256

# ---------------------------------------------------------------------------
# The sweep.
#
# Axes come from the workload, not from round numbers: decode batch sizes are
# vLLM `max_num_seqs` buckets, contexts are powers of two spanning the range a
# 128K-capable deployment actually serves, and the prefill query lengths are
# chunked-prefill chunk sizes.
#
# Two lengths, deliberately separate.
#
#   `context_len`    -- what goes in `seq_lens`. It bounds the KV the kernel
#                       streams and the causal mask, and nothing else.
#   `max_model_len`  -- how wide the block table is. The runner sizes it
#                       `cdiv(max_model_len, page_size)` per request, a
#                       deployment constant that does not shrink for a short
#                       sequence (`tpu_runner.py:639`,
#                       `attention_metadata.py:290-294`), and
#                       `streamindex_topk.py:594` recovers `pages_per_seq` from
#                       exactly that width. It alone sizes the score buffer.
#
# Most rows leave `max_model_len` at `context_len`, which is the *best* case
# for the incumbent: a full request in a deployment sized to it. The
# `_mml65536` rows pin the block table at 64K while the sequence stays short,
# which is what a real 64K-capable server does for most of its requests, and
# is the case that isolates the O(max_num_tokens x max_model_len) term -- the
# score buffer grows while the KV streamed does not. If the two families
# diverge, the score buffer is the cost; if they do not, it is not.
# ---------------------------------------------------------------------------

DECODE_BATCHES = (1, 4, 8, 16, 32, 64)
DECODE_CONTEXTS = (4096, 16384, 32768, 65536)
PREFILL_Q_LENS = (1024, 2048, 4096)
PREFILL_CONTEXTS = (4096, 16384, 32768)

# The pinned block-table width for the decoupled rows, and the (batch, ctx)
# pairs to run at it. Each has a `max_model_len == context_len` twin above, so
# every one of these is a controlled comparison rather than a lone data point.
PINNED_MAX_MODEL_LEN = 65536
PINNED_DECODE = ((8, 4096), (8, 16384), (32, 4096), (32, 16384), (64, 4096))
PINNED_PREFILL = ((1024, 4096), (4096, 4096))


@dataclasses.dataclass(frozen=True)
class Shape:
    """One swept invocation shape."""

    name: str
    phase: str  # "decode" | "prefill"
    num_seqs: int
    q_len: int  # per sequence
    context_len: int  # == seq_len
    enable_early_exit: bool = ENABLE_EARLY_EXIT
    note: str = ""
    # 0 means "the deployment is sized to this request", i.e. == context_len.
    max_model_len: int = 0
    # 0 means "every slot in the batch is a real sequence of context_len".
    # Otherwise: this many slots hold context_len and the rest hold
    # `pad_seq_len`, modelling a compiled batch bucket that is mostly padding.
    num_real_seqs: int = 0
    pad_seq_len: int = 0

    @property
    def num_tokens(self) -> int:
        return self.num_seqs * self.q_len

    @property
    def block_table_len(self) -> int:
        """The length the block table is sized for, not the one in use."""
        return self.max_model_len or self.context_len


# The shape GLM-5.2 is actually served at, from the nightly perf config
# (`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh` and the matching
# recipe `vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml`):
# max_model_len 9216 (8192 in + 1024 out), max_num_seqs 64,
# max_num_batched_tokens 1024, TP=1 DP=16 so the indexer is unsharded and one
# core sees the whole thing.
#
# These rows exist because the rest of the sweep is a set of round numbers and
# the deployment is not one of them. `report.md`'s roofline was computed at
# exactly these, so they are the only rows where a modelled floor and a measured
# time are comparable -- everywhere else the comparison would be against a
# number nobody derived. Kept first in the list so a `--filter prod` run is the
# cheap way to get the load-bearing numbers.
PROD_MAX_MODEL_LEN = 9216
PROD_MAX_NUM_SEQS = 64
PROD_CHUNK = 1024  # max_num_batched_tokens


def _production_shapes() -> list[Shape]:
    mml = PROD_MAX_MODEL_LEN
    out = []
    # Decode, at three points in the output: just after prefill, mid-decode
    # (what report.md models), and at the context limit. The block table is
    # 9 pages wide throughout, so num_positions is 10240 at all three and the
    # only thing that varies is how much of it is valid.
    for ctx, when in ((8192, "first decode step"), (8704, "mid-decode; the row report.md models"),
                      (9216, "last decode step, at max_model_len")):
        out.append(
            Shape(f"prod_decode_b{PROD_MAX_NUM_SEQS}_ctx{ctx}",
                  "decode",
                  PROD_MAX_NUM_SEQS,
                  1,
                  ctx,
                  note=f"PRODUCTION: {when}",
                  max_model_len=mml))
    # Decode at a partly-full batch, since max_num_seqs is a cap and not a
    # promise. If the per-call cost is flat in num_seqs the op is dominated by
    # the fixed full-width buffer rather than by the work, which is a different
    # optimization target.
    for n in (8, 32):
        out.append(
            Shape(f"prod_decode_b{n}_ctx8704", "decode", n, 1, 8704,
                  note=f"PRODUCTION shape, batch {n} of max_num_seqs "
                  f"{PROD_MAX_NUM_SEQS}",
                  max_model_len=mml))
    # Chunked prefill: one 1024-token chunk. The first chunk has almost no
    # history, the last has nearly the full context, and they are different
    # amounts of work -- report.md models the last.
    for ctx, when in ((1024, "first chunk, no history"),
                      (5120, "middle chunk"),
                      (9216, "last chunk; the row report.md models")):
        out.append(
            Shape(f"prod_prefill_q{PROD_CHUNK}_ctx{ctx}",
                  "prefill",
                  1,
                  PROD_CHUNK,
                  ctx,
                  note=f"PRODUCTION chunked prefill: {when}",
                  max_model_len=mml))
    return out


# The *other* GLM-5.2 deployment. The nightly perf config above is not the only
# shape this model is served at: the InferenceX AgentX trajectory-replay jobs
# (`j-a71a648c-3064-471a-a131` and its three siblings, read back verbatim from
# `cdk job recipe`) serve the same checkpoint at
#
#     --max-model-len=262144  --block-size=256  --max-num-seqs=64
#     --max-num-batched-tokens=1024  --kv-cache-dtype=fp8
#
# which is a 28x wider score buffer and a 4x smaller page than the nightly.
# Those are the jobs the only available xprof captures come from, so these rows
# exist for exactly one purpose: to make the profile's per-invocation numbers
# comparable to a microbenchmark number. Without them the cross-check is
# comparing two different deployments and calling the difference a discrepancy.
#
# Run these with `--page-size 256`; PAGE_SIZE is global rather than per-shape,
# so the two families cannot share one process. `--filter agentx` selects them.
AGENTX_MAX_MODEL_LEN = 262144
AGENTX_PAGE_SIZE = 256
AGENTX_CHUNK = 1024


def _agentx_shapes() -> list[Shape]:
    mml = AGENTX_MAX_MODEL_LEN
    out = []
    # Batch 16 is what the capture actually ran at (`MLA-p_2048-bz_16-gcz_16`,
    # `router_topk` output f32[16,8]); batch 64 is the recipe's max_num_seqs.
    # The replayed traces' real context lengths are not recoverable from the
    # profile -- the compiled shapes are max_model_len-sized regardless -- so
    # sweep the context instead of guessing one, and let the sweep say whether
    # it matters.
    for num_seqs in (16, 64):
        for ctx in (8192, 32768):
            out.append(
                Shape(f"agentx_decode_b{num_seqs}_ctx{ctx}",
                      "decode",
                      num_seqs,
                      1,
                      ctx,
                      note=f"AGENTX: the profiled deployment, mml={mml}, "
                      f"page={AGENTX_PAGE_SIZE}",
                      max_model_len=mml))
    out.append(
        Shape(f"agentx_prefill_q{AGENTX_CHUNK}_ctx8192",
              "prefill",
              1,
              AGENTX_CHUNK,
              8192,
              note="AGENTX chunked prefill. No captured window contains an "
              "indexer-active prefill step, so this row has no profile twin "
              "and is a microbenchmark number only.",
              max_model_len=mml))
    return out


# The operating point the xprof capture was actually taken at. Read from the
# job's own artifacts, not assumed:
#
#   benchmark_command.txt      --concurrency 1
#   profile_export_console.txt Effective Decode Concurrency avg 1.02, p50 1.00
#   profile_export_console.txt Prompt Tokens avg 105864, p50 110897, max 169053
#
# So the profiled call is ONE real sequence of ~110k tokens sitting in a
# compiled batch-16 bucket with fifteen padding slots -- not sixteen real
# sequences of 8192, which is what `agentx_decode_b16_ctx8192` models. Those
# two differ on both axes at once and the errors run in opposite directions on
# total KV volume, which is exactly how two unlike numbers land 11% apart and
# look like agreement.
#
# What the padding slots' seq_lens entries hold in production is not yet
# established, and it is the whole question: if they hold 0 the call does one
# sequence's work, if they pay kv_len_max it does sixteen. So both are here.
# The pair brackets the truth and the gap between them IS the measurement.
PROFILE_CTX = 110592  # 432 pages of 256; nearest page-aligned to the p50 110897
PROFILE_BUCKET = 16


def _profile_shapes() -> list[Shape]:
    mml = AGENTX_MAX_MODEL_LEN
    common = dict(max_model_len=mml, num_real_seqs=1)
    return [
        Shape(f"profile_b{PROFILE_BUCKET}_1real_ctx{PROFILE_CTX}_pad0",
              "decode", PROFILE_BUCKET, 1, PROFILE_CTX,
              note="PROFILE-MATCHED, padding slots seq_len=0: the cost if the "
              "15 padding slots are guarded out",
              pad_seq_len=0, **common),
        Shape(f"profile_b{PROFILE_BUCKET}_1real_ctx{PROFILE_CTX}_padfull",
              "decode", PROFILE_BUCKET, 1, PROFILE_CTX,
              note="PROFILE-MATCHED, padding slots seq_len=context: the cost "
              "if every slot pays kv_len_max",
              pad_seq_len=PROFILE_CTX, **common),
        # The control the pair needs: one real sequence and nothing else, so
        # the bucket's own contribution is visible rather than inferred.
        Shape(f"profile_b1_ctx{PROFILE_CTX}",
              "decode", 1, 1, PROFILE_CTX,
              note="PROFILE-MATCHED control: batch 1, no padding slots at all",
              max_model_len=mml),
    ]


def _default_shapes() -> list[Shape]:
    shapes: list[Shape] = _production_shapes()
    shapes += _agentx_shapes()
    shapes += _profile_shapes()

    # `enable_early_exit` short-circuits when max(seq_lens) // compression_ratio
    # <= k. At k=2048 and compression_ratio=1 that is contexts <= 2048, so
    # these two rows are the only ones on the fast path -- and they are here as
    # a pair so the cost of the path that is skipped is visible next to it.
    shapes.append(
        Shape("decode_ee_fastpath", "decode", 8, 1, 2048, True,
              "early-exit FAST PATH fires (ctx <= k); no scores, no stage B"))
    shapes.append(
        Shape("decode_ee_fastpath_off", "decode", 8, 1, 2048, False,
              "same shape with the fast path disabled: the scoring path"))

    for num_seqs in DECODE_BATCHES:
        for ctx in DECODE_CONTEXTS:
            shapes.append(
                Shape(f"decode_b{num_seqs}_ctx{ctx}", "decode", num_seqs, 1,
                      ctx))
    for q_len in PREFILL_Q_LENS:
        for ctx in PREFILL_CONTEXTS:
            shapes.append(
                Shape(f"prefill_q{q_len}_ctx{ctx}", "prefill", 1, q_len, ctx))

    # Short sequences in a 64K-wide deployment: same KV streamed as their twin
    # above, a much larger score buffer.
    mml = PINNED_MAX_MODEL_LEN
    for num_seqs, ctx in PINNED_DECODE:
        shapes.append(
            Shape(f"decode_b{num_seqs}_ctx{ctx}_mml{mml}",
                  "decode",
                  num_seqs,
                  1,
                  ctx,
                  note=f"block table pinned at max_model_len={mml}; twin is "
                  f"decode_b{num_seqs}_ctx{ctx}",
                  max_model_len=mml))
    for q_len, ctx in PINNED_PREFILL:
        shapes.append(
            Shape(f"prefill_q{q_len}_ctx{ctx}_mml{mml}",
                  "prefill",
                  1,
                  q_len,
                  ctx,
                  note=f"block table pinned at max_model_len={mml}; twin is "
                  f"prefill_q{q_len}_ctx{ctx}",
                  max_model_len=mml))
    return shapes


SHAPES: list[Shape] = _default_shapes()

# ---------------------------------------------------------------------------
# Closed-form geometry. No JAX, no device -- this all runs under --dry-run.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Geometry:
    pages_per_seq: int
    num_sublanes_total: int
    num_positions: int
    num_tokens: int
    score_buffer_bytes: int
    kv_cache_bytes: int  # what the benchmark has to allocate
    kv_stream_bytes: int  # what the kernel DMAs out of it
    kv_stream_useful_bytes: int
    total_pages: int
    bq_sz: int
    bkv_sz: int
    num_bq: int
    num_bkv: int


def _per_seq_bkv(shape: Shape, bkv_sz: int) -> list[int]:
    """Per-slot kv-block count, matching the seq_lens the builder constructs.

    Homogeneous batches give `num_seqs` identical entries; a batch with
    `num_real_seqs` set gives that many at `context_len` and the rest at
    `pad_seq_len`. A zero-length slot still gets one block, because the loop
    runs `max(1, ...)` times.
    """
    if shape.num_real_seqs and shape.num_real_seqs < shape.num_seqs:
        lens = ([shape.context_len] * shape.num_real_seqs +
                [shape.pad_seq_len] *
                (shape.num_seqs - shape.num_real_seqs))
    else:
        lens = [shape.context_len] * shape.num_seqs
    return [max(1, cdiv(n // COMPRESSION_RATIO, bkv_sz)) for n in lens]


def geometry(shape: Shape) -> Geometry:
    """Everything derivable from a shape without running anything."""
    # streamindex_topk.py:594 -- `pages_per_seq = page_indices.shape[0] //
    # max_num_seqs`, so it is the block table's width, sized by max_model_len,
    # not by how much of the sequence exists.
    pages_per_seq = cdiv(shape.block_table_len, PAGE_SIZE)

    # streamindex_topk.py:603-605.
    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * PAGE_SIZE // 128
        for bkv_p in NUM_KV_PAGES_PER_BLOCK)
    num_positions = num_sublanes_total * 128

    num_tokens = shape.num_tokens
    # streamindex_topk.py:745-749 and :725-728 -- the init buffer and the
    # kernel output are the same shape, and the output is aliased to the init.
    score_buffer_bytes = num_tokens * num_sublanes_total * 128 * BYTES_PER_F32

    total_pages = shape.num_seqs * pages_per_seq
    kv_cache_bytes = (total_pages * (PAGE_SIZE // KV_PACKING) * KV_PACKING *
                      LKV_DIM)

    # Which of the three pallas_calls runs, and with what tiling.
    if shape.phase == "decode":
        # streamindex_topk.py:765-766 / :783-784: the decode calls use
        # num_*_per_block[0], and bq_sz = min(nqpb, static_q_len=1) = 1.
        bkv_p = NUM_KV_PAGES_PER_BLOCK[0]
        bq_sz = min(NUM_QUERIES_PER_BLOCK[0], 1)
    else:
        # streamindex_topk.py:801-802: the mixed call uses index [2] and
        # static_q_len is None, so bq_sz is the raw block size.
        bkv_p = NUM_KV_PAGES_PER_BLOCK[2]
        bq_sz = NUM_QUERIES_PER_BLOCK[2]

    bkv_sz = bkv_p * PAGE_SIZE
    kv_len = shape.context_len // COMPRESSION_RATIO
    # The loop trip count is derived from the LONGEST sequence in the batch
    # (`kv_len_max = jnp.max(jnp.stack(kv_lens))`, streamindex_topk.py:275-276),
    # so it does not shrink for a batch that is mostly padding.
    num_bkv = max(1, cdiv(kv_len, bkv_sz))
    num_bq = max(1, cdiv(shape.q_len, bq_sz))

    # `_fetch_bkv` DMAs bkv_p whole pages per (sequence, bq block, bkv block);
    # a bq block re-streams the entire KV range (streamindex_topk.py:385-418).
    #
    # For a heterogeneous batch this is a LOWER BOUND, and deliberately so. It
    # charges each slot only for its own length -- i.e. it assumes the per-slot
    # DMA is guarded. If it is not, every slot pays `num_bkv` and the true
    # figure is the homogeneous one, up to `num_seqs / num_real_seqs` higher.
    # Which of the two holds is the question these shapes exist to answer, so
    # the geometry must not quietly pick a side; `useful%` and the header say
    # which bound this is.
    per_seq_bkv = _per_seq_bkv(shape, bkv_sz)
    kv_stream_bytes = sum(per_seq_bkv) * num_bq * bkv_sz * LKV_DIM
    kv_stream_useful = sum(per_seq_bkv) * num_bq * bkv_sz * LKV_USEFUL_BYTES

    return Geometry(
        pages_per_seq=pages_per_seq,
        num_sublanes_total=num_sublanes_total,
        num_positions=num_positions,
        num_tokens=num_tokens,
        score_buffer_bytes=score_buffer_bytes,
        kv_cache_bytes=kv_cache_bytes,
        kv_stream_bytes=kv_stream_bytes,
        kv_stream_useful_bytes=kv_stream_useful,
        total_pages=total_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        num_bq=num_bq,
        num_bkv=num_bkv,
    )


def _gib(n: int) -> float:
    return n / (1 << 30)


def _mib(n: int) -> float:
    return n / (1 << 20)


# ---------------------------------------------------------------------------
# Input construction.
# ---------------------------------------------------------------------------


def _row_lengths_ref(seq_lens, cu_q_lens, num_seqs, num_tokens, num_positions,
                     compression_ratio):
    """Visible compressed KV positions per token row of the score matrix.

    Verbatim copy of `_effective_row_lengths`
    (src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py:482-506) at
    commit 0976305d, minus the `distribution` indirection (this file always
    knows `num_seqs` concretely). It is vendored rather than imported on
    purpose: this instrument is the yardstick a *replacement* kernel is judged
    by, so it must not break or silently change when the incumbent's private
    helpers do.
    """
    import jax.numpy as jnp

    max_num_seqs = seq_lens.shape[0]
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


def _make_cache(key, total_pages: int):
    """A uint8 indexer K cache with the production record layout.

    Layout, per `test_dsv4_streamindex_topk.py:264-280` and the kernel's
    `load_bkv` (streamindex_topk.py:257-272):
    ``cache[page, slot // 4, slot % 4] = [fp8 x D | one e8m0 scale byte | pad]``

    The value bytes are drawn from the PRNG and then have their 4-bit fp8
    exponent clamped into [5, 8]. That is not a Gaussian, but it is finite,
    NaN-free (e4m3fn NaN needs exp==15) and spread over +-[0.25, 3.75], which
    is what the arithmetic needs. Building it as bytes rather than quantizing
    a float array keeps the largest cache in this sweep (1 GiB) from needing a
    4x-larger f32 intermediate.
    """
    import jax
    import jax.numpy as jnp

    shape = (total_pages, PAGE_SIZE // KV_PACKING, KV_PACKING, LKV_DIM)
    raw = jax.random.bits(key, shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(LKV_DIM, dtype=jnp.int32)
    # e8m0 bits 127 == 2**0, so the dequantized key is the fp8 value itself.
    return jnp.where(lane < D_IDX, vals,
                     jnp.where(lane == D_IDX, jnp.uint8(127), jnp.uint8(0)))


def build_inputs(shape: Shape, geo: Geometry, seed: int = 0) -> dict[str, Any]:
    """Device-resident inputs for one shape, in the caller's dtypes."""
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.key(seed), 4)

    num_seqs = shape.num_seqs
    num_tokens = geo.num_tokens

    # q reaches the kernel as fp8_e4m3fn (mla_attention_op.py:106), not f32.
    q = jax.random.normal(keys[0], (num_tokens, H_IDX, D_IDX),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    # `weights` is the activation dtype at the call site; the kernel upcasts to
    # f32 in `prepare_index_weights` either way.
    weights = jax.random.uniform(keys[1], (num_tokens, H_IDX), jnp.float32,
                                 0.25, 1.75).astype(jnp.bfloat16)

    cache_kv = _make_cache(keys[2], geo.total_pages)

    # Disjoint, shuffled physical pages per sequence: a sequential block table
    # would let a page-walk bug (and the DMA engine) look better than it is.
    block_table = jax.random.permutation(
        keys[3], jnp.arange(geo.total_pages,
                            dtype=jnp.int32)).reshape(num_seqs,
                                                      geo.pages_per_seq)

    # Every row up to now has a homogeneous batch: num_seqs sequences all of
    # context_len. Real serving batches are not like that -- vLLM compiles to
    # bucket sizes (16, 32, 64, ...) and pads, so a batch-16 program can carry
    # one real sequence and fifteen padding slots. That distinction is invisible
    # in a homogeneous batch and decides whether the padding slots pay for the
    # longest sequence's KV walk, which is the single largest open question
    # about this op. `num_real_seqs` makes it askable.
    if shape.num_real_seqs and shape.num_real_seqs < num_seqs:
        seq_lens = jnp.where(
            jnp.arange(num_seqs, dtype=jnp.int32) < shape.num_real_seqs,
            shape.context_len, shape.pad_seq_len).astype(jnp.int32)
    else:
        seq_lens = jnp.full((num_seqs, ), shape.context_len, jnp.int32)
    cu_q_lens = (jnp.arange(num_seqs + 1, dtype=jnp.int32) * shape.q_len)
    num_decodes = num_seqs if shape.q_len == 1 else 0
    distribution = jnp.array([num_decodes, num_decodes, num_seqs], jnp.int32)

    return {
        "q": q,
        "indexer_weights": weights,
        "cache_kv": cache_kv,
        "seq_lens": seq_lens,
        "page_indices": block_table.reshape(-1),
        "cu_q_lens": cu_q_lens,
        "distribution": distribution,
    }


def build_stage_b_scores(shape: Shape, geo: Geometry, inputs: dict[str, Any],
                         seed: int = 1):
    """The f32 score matrix stage B consumes, at the op's exact shape.

    Values are N(0, 1) with everything at or past the row's visible length set
    to -inf, which is the shape and the sparsity pattern stage A leaves behind.
    The *distribution* is synthetic, and that matters: the SparseCore radix
    select is data dependent. It is held fixed so the arm is comparable across
    candidates, not because N(0, 1) is what the indexer emits.
    """
    import jax
    import jax.numpy as jnp

    row_lengths = _row_lengths_ref(
        inputs["seq_lens"],
        inputs["cu_q_lens"],
        shape.num_seqs,
        geo.num_tokens,
        geo.num_positions,
        COMPRESSION_RATIO,
    )
    scores = jax.random.normal(jax.random.key(seed),
                               (geo.num_tokens, geo.num_positions),
                               jnp.float32)
    col = jnp.arange(geo.num_positions, dtype=jnp.int32)
    scores = jnp.where(col[None, :] < row_lengths[:, None], scores, -jnp.inf)
    return scores, row_lengths


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------


def time_arm(fn, args, warmup: int, trials: int,
             target_s: float) -> tuple[float, int, float]:
    """Best-of-`trials` seconds per call, plus the compile+first-call cost.

    Every arm is warmed before the clock starts and `block_until_ready` closes
    every trial, so nothing here can measure dispatch instead of execution.
    Each arm's result is returned and blocked on, so nothing can be dead-code
    eliminated for being unused.
    """
    import jax

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    # One timed call sets the iteration count: short kernels get batched so the
    # host clock is not measuring its own resolution.
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    single = time.perf_counter() - t0
    iters = max(1, min(200, int(target_s / max(single, 1e-9))))

    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / iters)
    del out
    return best, iters, compile_s


def measure_dispatch_floor(args) -> float:
    """Seconds per call for a trivial jitted op, batched the same way.

    Calls are enqueued asynchronously, so a batched measurement reports
    ``max(device time, host enqueue cost)``. On the small decode shapes the
    kernel is short enough that the second term can win, and a host-bound arm
    will not move when the kernel gets faster. Printing the floor is what makes
    that failure mode visible instead of silent.

    It is a *lower* bound on the real floor: this op takes one argument, while
    `streamindex_topk` takes seven arrays and re-enters its own `jax.jit`
    wrapper on every call. `[HOST BOUND]` firing is therefore strong evidence;
    it not firing is weak evidence of the opposite.
    """
    import jax
    import jax.numpy as jnp

    fn = jax.jit(lambda x: x + 1.0)
    x = jnp.asarray(1.0, jnp.float32)
    best, _, _ = time_arm(fn, (x, ), args.warmup, args.trials, args.target_s)
    return best


def run_shape(shape: Shape, geo: Geometry, args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
        sparsecore_topk
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

    result: dict[str, Any] = {
        "name": shape.name,
        "phase": shape.phase,
        "num_seqs": shape.num_seqs,
        "q_len": shape.q_len,
        "context_len": shape.context_len,
        "block_table_len": shape.block_table_len,
        "max_model_len_pinned": bool(shape.max_model_len),
        "enable_early_exit": shape.enable_early_exit,
        "note": shape.note,
        **{f"geo_{k}": v
           for k, v in dataclasses.asdict(geo).items()},
        "status": "unknown_error",
    }

    inputs = build_inputs(shape, geo, seed=args.seed)

    def total_fn(**kw):
        return streamindex_topk(
            **kw,
            k=K_IDX,
            compression_ratio=COMPRESSION_RATIO,
            num_kv_pages_per_block=NUM_KV_PAGES_PER_BLOCK,
            num_queries_per_block=NUM_QUERIES_PER_BLOCK,
            enable_early_exit=shape.enable_early_exit,
        )

    t_total, iters_total, compile_total = time_arm(lambda: total_fn(**inputs),
                                                   (), args.warmup,
                                                   args.trials,
                                                   args.target_s)

    # --- stage B alone -----------------------------------------------------
    scores, row_lengths = build_stage_b_scores(shape, geo, inputs,
                                               seed=args.seed + 1)
    stage_b_fn = jax.jit(lambda s, rl: sparsecore_topk(
        s, K_IDX, row_lengths=rl, write_empty_rows=True))
    t_b, iters_b, compile_b = time_arm(stage_b_fn, (scores, row_lengths),
                                       args.warmup, args.trials, args.target_s)
    scores.delete()

    # --- the -inf score-buffer init alone ----------------------------------
    # Parameterized on a device scalar so XLA cannot fold the fill into a
    # constant and hand back a buffer nobody wrote.
    init_shape = (geo.num_tokens, geo.num_sublanes_total, 128)
    init_fn = jax.jit(lambda v: jnp.full(init_shape, v, jnp.float32))
    neg_inf = jnp.asarray(-jnp.inf, jnp.float32)
    t_init, iters_init, compile_init = time_arm(init_fn, (neg_inf, ),
                                                args.warmup, args.trials,
                                                args.target_s)

    for value in inputs.values():
        try:
            value.delete()
        except AttributeError:
            pass

    fast_path = (shape.enable_early_exit
                 and shape.context_len // COMPRESSION_RATIO <= K_IDX)
    t_a = None if fast_path else t_total - t_b - t_init

    score_bytes = geo.score_buffer_bytes
    result.update({
        "status": "success",
        "fast_path": fast_path,
        "iters": {
            "total": iters_total,
            "stage_b": iters_b,
            "init": iters_init
        },
        "compile_s": {
            "total": compile_total,
            "stage_b": compile_b,
            "init": compile_init
        },
        "total_ms": t_total * 1e3,
        "stage_b_ms": t_b * 1e3,
        "init_ms": t_init * 1e3,
        "stage_a_derived_ms": None if t_a is None else t_a * 1e3,
        # Ratios: absolutes drift between stacks, ratios do not.
        "stage_b_frac_of_total": t_b / t_total,
        "init_frac_of_total": t_init / t_total,
        "stage_a_frac_of_total": None if t_a is None else t_a / t_total,
        # Implied bandwidths. The init arm writes the score buffer exactly
        # once and stage B reads it exactly once, so these two are honest
        # single-pass numbers and a value above the part's HBM bandwidth means
        # the timing is wrong, not that the kernel is good.
        "init_write_TBs": score_bytes / t_init / 1e12,
        "stage_b_read_TBs": score_bytes / t_b / 1e12,
        # Stage A reads the K cache and writes the score buffer once each.
        # Derived from a derived time, so it is the softest number here.
        "stage_a_implied_TBs":
        (None if t_a is None or t_a <= 0 else
         (geo.kv_stream_bytes + score_bytes) / t_a / 1e12),
        "whole_op_score_traffic_TBs":
        # init write + kernel write + stage B read == 3 passes minimum.
        3 * score_bytes / t_total / 1e12,
    })
    return result


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

_GEO_HDR = (f"{'shape':<30} {'tok':>5} {'ctx':>7} {'blk_tbl':>8} "
            f"{'sublanes':>8} {'positions':>9} "
            f"{'score_buf':>11} {'kv_stream':>11} {'kv_alloc':>10} "
            f"{'useful%':>7}")


def print_geometry(shapes: list[Shape]) -> None:
    print("\n== geometry (closed form; no execution) ==")
    print(f"  page_size={PAGE_SIZE} kv_packing={KV_PACKING} "
          f"lkv_dim={LKV_DIM} (useful {LKV_USEFUL_BYTES}/{LKV_DIM} bytes = "
          f"{100 * LKV_USEFUL_BYTES / LKV_DIM:.1f}%)")
    print(f"  H={H_IDX} D={D_IDX} k={K_IDX} "
          f"compression_ratio={COMPRESSION_RATIO}")
    print(f"  cache_kv shape = (total_pages, {PAGE_SIZE // KV_PACKING}, "
          f"{KV_PACKING}, {LKV_DIM}) uint8")
    print(_GEO_HDR)
    print("-" * len(_GEO_HDR))
    for shape in shapes:
        geo = geometry(shape)
        print(f"{shape.name:<30} {geo.num_tokens:>5} "
              f"{shape.context_len:>7} {shape.block_table_len:>8} "
              f"{geo.num_sublanes_total:>8} {geo.num_positions:>9} "
              f"{_mib(geo.score_buffer_bytes):>9.1f}M "
              f"{_mib(geo.kv_stream_bytes):>9.1f}M "
              f"{_mib(geo.kv_cache_bytes):>8.1f}M "
              f"{100 * LKV_USEFUL_BYTES / LKV_DIM:>6.1f}%")


_RUN_HDR = (f"{'shape':<30} {'total_ms':>9} {'stageB_ms':>9} {'init_ms':>8} "
            f"{'stageA*_ms':>10} {'B/tot':>6} {'init/tot':>8} {'A*/tot':>7} "
            f"{'initBW':>8} {'stgB_BW':>8}")


def print_run_header() -> None:
    print("\n== timings ==  (stageA* is DERIVED BY SUBTRACTION, not measured; "
          "BW in TB/s)")
    print(_RUN_HDR)
    print("-" * len(_RUN_HDR))


def print_run_row(row: dict[str, Any]) -> None:
    if row["status"] != "success":
        print(f"{row['name']:<30} {row['status']}: "
              f"{row.get('error', '')[:80]}")
        return

    def _f(value, fmt):
        return "n/a".rjust(len(format(0.0, fmt))) if value is None else format(
            value, fmt)

    print(f"{row['name']:<30} {row['total_ms']:>9.3f} "
          f"{row['stage_b_ms']:>9.3f} {row['init_ms']:>8.3f} "
          f"{_f(row['stage_a_derived_ms'], '>10.3f')} "
          f"{row['stage_b_frac_of_total']:>6.3f} "
          f"{row['init_frac_of_total']:>8.3f} "
          f"{_f(row['stage_a_frac_of_total'], '>7.3f')} "
          f"{row['init_write_TBs']:>8.3f} "
          f"{row['stage_b_read_TBs']:>8.3f}"
          f"{'   <- early-exit fast path' if row['fast_path'] else ''}"
          f"{'   [HOST BOUND]' if row.get('host_bound') else ''}")


# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="print geometry and exit; safe under "
                        "JAX_PLATFORMS=cpu (no Pallas, no get_tpu_info)")
    parser.add_argument("--dry-run-build-bytes",
                        type=int,
                        default=32 << 20,
                        help="under --dry-run, also construct the inputs for "
                        "every shape whose KV cache is under this many bytes, "
                        "so the input builder is typechecked too")
    parser.add_argument("--filter",
                        default="",
                        help="substring match on the shape name")
    parser.add_argument("--page-size",
                        type=int,
                        default=PAGE_SIZE,
                        help=f"KV page size (vLLM's --block-size). Default "
                        f"{PAGE_SIZE}, the nightly perf config. The AgentX "
                        f"deployment the xprof captures come from uses "
                        f"{AGENTX_PAGE_SIZE}; pass it with `--filter agentx`. "
                        f"This is a module global rather than a per-shape "
                        f"field, so one process serves one page size.")
    parser.add_argument("--max-score-buffer-bytes",
                        type=int,
                        default=8 << 30,
                        help="skip (loudly) any shape whose score buffer is "
                        "larger than this")
    parser.add_argument("--max-kv-cache-bytes",
                        type=int,
                        default=16 << 30,
                        help="skip (loudly) any shape needing a larger cache")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--target-s",
                        type=float,
                        default=2e-3,
                        help="batch calls until a timed run is about this long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="", help="write results here")
    return parser


def main() -> int:
    args = _parser().parse_args()

    # PAGE_SIZE feeds pages_per_seq, num_positions, the cache shape and the
    # modelled byte counts, and SHAPES was built at import time under the
    # default. Rebind both before anything reads them, and say so in the header
    # -- a page-size change that is not printed is a silent reshape of every
    # number below it.
    global PAGE_SIZE, SHAPES  # noqa: PLW0603 - see above
    if args.page_size != PAGE_SIZE:
        print(f"page_size overridden: {PAGE_SIZE} -> {args.page_size}")
        PAGE_SIZE = args.page_size
        SHAPES = _default_shapes()

    shapes = [s for s in SHAPES if args.filter in s.name]
    if not shapes:
        print(f"no shape matches --filter {args.filter!r}", file=sys.stderr)
        return 2

    kept: list[Shape] = []
    for shape in shapes:
        geo = geometry(shape)
        if geo.score_buffer_bytes > args.max_score_buffer_bytes:
            print(f"SKIP {shape.name}: score buffer "
                  f"{_gib(geo.score_buffer_bytes):.2f} GiB > cap "
                  f"{_gib(args.max_score_buffer_bytes):.2f} GiB")
            continue
        if geo.kv_cache_bytes > args.max_kv_cache_bytes:
            print(f"SKIP {shape.name}: KV cache "
                  f"{_gib(geo.kv_cache_bytes):.2f} GiB > cap "
                  f"{_gib(args.max_kv_cache_bytes):.2f} GiB")
            continue
        kept.append(shape)

    print_geometry(kept)

    if args.dry_run:
        print("\n== dry run ==")
        print("Geometry above is exact. Nothing was executed: Pallas and "
              "pltpu.get_tpu_info() do not work on CPU, so a clean dry run "
              "proves only that this script is not a typo.")
        built = 0
        for shape in kept:
            geo = geometry(shape)
            if geo.kv_cache_bytes > args.dry_run_build_bytes:
                continue
            inputs = build_inputs(shape, geo, seed=args.seed)
            scores, row_lengths = build_stage_b_scores(shape, geo, inputs,
                                                       seed=args.seed + 1)
            shapes_seen = {k: tuple(v.shape) for k, v in inputs.items()}
            print(f"  built inputs for {shape.name}: {shapes_seen}, "
                  f"scores={tuple(scores.shape)} "
                  f"row_lengths={tuple(row_lengths.shape)} "
                  f"max_row_len={int(row_lengths.max())}")
            built += 1
        print(f"  input builder exercised on {built} shape(s)")
        return 0

    floor_s = measure_dispatch_floor(args)
    print(f"\nhost dispatch floor: {floor_s * 1e3:.4f} ms/call "
          "(a trivial jitted op, batched identically). Any arm within ~3x of "
          "this is host bound and will not move when the kernel does.")

    print_run_header()
    rows: list[dict[str, Any]] = []
    wall0 = time.perf_counter()
    for shape in kept:
        geo = geometry(shape)
        try:
            row = run_shape(shape, geo, args)
            row["dispatch_floor_ms"] = floor_s * 1e3
            row["host_bound"] = row["total_ms"] < 3 * floor_s * 1e3
        except Exception as error:  # a structured row beats a dead sweep
            row = {
                "name": shape.name,
                "status": ("oom" if "RESOURCE_EXHAUSTED" in str(error) else
                           "error"),
                "error": repr(error),
                **{f"geo_{k}": v
                   for k, v in dataclasses.asdict(geo).items()},
            }
        rows.append(row)
        print_run_row(row)
    print(f"\nsweep wall clock: {time.perf_counter() - wall0:.1f}s "
          f"({len(kept)} shapes, compile included)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "config": {
                        "H_IDX": H_IDX,
                        "D_IDX": D_IDX,
                        "K_IDX": K_IDX,
                        "COMPRESSION_RATIO": COMPRESSION_RATIO,
                        "PAGE_SIZE": PAGE_SIZE,
                        "KV_PACKING": KV_PACKING,
                        "LKV_DIM": LKV_DIM,
                        "LKV_USEFUL_BYTES": LKV_USEFUL_BYTES,
                        "NUM_KV_PAGES_PER_BLOCK": NUM_KV_PAGES_PER_BLOCK,
                        "NUM_QUERIES_PER_BLOCK": NUM_QUERIES_PER_BLOCK,
                    },
                    "rows": rows,
                },
                handle,
                indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
