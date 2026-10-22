# Packed FP8 paged MQA

The maintained entry point is `packed_fp8_mqa_logits` in
`src/vllm_torchtpu/kernels/deepseek_v41/index_selection/packed_fp8_mqa.py`.
It replaces the `head42_boundary_overlap` experiment as the implementation to
import and modify. Historical snapshots remain available for comparisons.
The v2 model has an opt-in native-cache route. BF16 remains the default while
model quality and end-to-end performance are being validated.

## Model integration

Set `TPU_DEEPSEEK_V41_INDEXER_BACKEND=packed_fp8` before model initialization.
The selector controls the indexer's cache, independently of the main attention
KV-cache dtype. Use `bf16` in a separate process for the existing comparison
route; a live cache cannot be reinterpreted between the two formats.

The packed route requires TPU v7, H32/D128 index queries, and 128 compressed
index keys per physical page. Set the scheduler block size to
`128 * index compression ratio`; startup rejects a mismatch. For a checkpoint
whose index compression ratio is four:

```bash
# Add the same model, parallelism and serving options used in the baseline.
TPU_DEEPSEEK_V41_IMPL=v2 TPU_DEEPSEEK_V41_INDEXER_BACKEND=packed_fp8 \
vllm serve /path/to/checkpoint --block-size 512
```

Allocation retains upstream shared-owner placements and supports one or two
native banks. The index view reserves 132 bytes per key (FP8 values plus FP32
scale), and the adapter maps this view to the transposed page format below.
Incremental writes quantize only incoming K vectors and scatter their 33 words
into the corresponding token columns. Query projection/rotation feeds the
scorer without the old FP8-to-BF16 round trip; the JAX bridge quantizes Q once
and passes its scales to the maintained kernel.

The paged model route preserves deferred writes, TP query splitting/gathering,
candidate publication and logical top-k IDs. Ragged queries are grouped by
request and restored to their original order. Score chunks use the existing
512-MiB score-element budget. The route does not materialize the BF16 candidate
K pool: it reads the original FP8 cache directly. Candidate IDs remain necessary.
Candidate consumers now build a sorted page union for each group of at most six
queries from the same request, using the existing SparseCore union implementation
when supported. Only those physical pages issue HBM DMAs. Invalid/padded tasks
issue no DMA, and the receive wait counts only the bytes actually requested.
Scores use compact page positions; query-specific candidate masks and visible
lengths are applied before top-k, then selections map back to original token IDs.
Candidate publishers still score the full visible context. The whole-batch
short-context bypass and the existing all-blocks-fit identity are preserved.

Read granularity remains a 128-token physical page: a smaller candidate block
can bring other tokens on that page into the computation. Dispersed candidates
whose union touches every page therefore still read every page. The arithmetic
pipeline retains its unrolled 32768-key tiles, including padding in a partial
tile. Sparse calls leave skipped K pages unwritten and allocate no zero buffer.
QK and both head-reduction paths operate independently on each token column;
invalid pages are masked to `-inf` before top-k or any cross-token operation.
Their arbitrary values, including NaNs, cannot contribute to valid columns.
Do not add a cross-token reduction before this mask. Compacting pages can change
which head-reduction path a valid token uses; bitwise agreement with dense
scoring has not been established. Hardware performance remains unmeasured.

To exercise the scorer, use a context exceeding `index_topk` compressed keys
and a model maximum length beyond that threshold; the existing all-indices-fit
shortcut deliberately bypasses scoring. The supported paged path also requires
the model's SWA window to be divisible by 64. The legacy eager path can decode
selected FP8 records, but uses its original FP32 scoring arithmetic and does
not exercise the optimized kernel.

Local integration checks cover incremental writes, shared ownership, empty
deferred writes, candidate publication/filtering, ragged query order, and the
Torch wrapper's dispatch arguments. CPU routing tests replace the TPU scorer
with a stand-in and are not numerical validation of the kernel. A local
sim-pjrt compile passed for the complete JAX two-bank writer, native-page
scorer, candidate-page preparation and SparseCore top-k path at P-state 7.
Metadata tests compare page coverage, per-query masks and mapped top-k IDs with
a token-level reference, including page holes, duplicates, empty candidates and
blocks crossing physical page boundaries. DMA descriptor tests verify that
three valid page tasks issue three HBM reads and wait for three pages, rather
than the full 256-page tile. Fragment tests poison skipped pages with FP8 NaNs,
infinite scales and random bits, including holes and fully invalid fragments.
They compare zero-filled and poisoned inputs through the production QK,
VPU/MXU reduction and store methods; valid outputs are identical and invalid
outputs are `-inf`. These CPU tests emulate the unsupported TPU pack/unpack/roll
primitives and are not hardware validation. No model or hardware execution has
been performed for this integration. The local check
environment lacks vLLM/torch-tpu, so real custom-op registration and a model
forward still need the full runtime; the existing startup test now includes
the packed ops. Independent kernel-reference differences remain as documented
below.

## Calling the kernel

```python
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    pack_fp8_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    packed_fp8_mqa_logits,
)

# Encode when creating cache pages, outside the repeated scorer invocation.
# k0/k1: FP8 [complete_pages * 128, 128]; s0/s1: FP32 [complete_pages * 128].
cache = (pack_fp8_pages(k0, s0), pack_fp8_pages(k1, s1))
scores = packed_fp8_mqa_logits(
    q, weights, cache, page_ids, lengths, query_offsets, q_scale=q_scale
)
```

The measured policy is TPU v7, H32/D128, six resident Q rows, 32768 keys/tile,
eight pending fragments and four K buffers. It accepts one or two cache banks;
two banks were benchmarked. Total Q rows must be at least six unless empty.
Individual groups may be empty or have fewer than six queries. Context capacity
must be a multiple of 32768. The kernel returns dense FP32 scores, including
`-inf` for page holes and tokens outside each query's visible length.

## Cache bytes and partial pages

Each page occupies int32 `[33,128]` (16,896 bytes):

- `word[d // 4, token]`, byte `d % 4`, contains the FP8 bits of `K[token,d]`.
- `word[32, token]` contains that token's FP32 scale bits.

After the native Pallas bitcast, K has shape `[dim,token]`. Two pages concatenate
to `[128,256]`, matching the RHS of the QK dot without a separate transpose.
Page IDs, bank boundaries and the order of scales are unaffected.
Ordinary `[token,dim]` packed pages have the same shape but incompatible bytes;
the caller must use the matching writer.

A page with 17 valid tokens still occupies a complete page. Pad its unused K
rows with finite zeros and scales with one before calling `pack_fp8_pages`, and
set `lengths` to the actual visible context length. The scorer still transfers
and computes the full page, then masks unused tokens. `pack_fp8_pages` encodes
complete pages. `write_fp8_records` is the model's incremental writer; it preserves
other token columns and all neighboring cache placements. Native owners are
zero-initialized at allocation. Do not convert the entire cache on every scorer
call.

## Where to change things

| File | Responsibility |
| --- | --- |
| `packed_fp8_cache.py` | Page placement, byte format, complete-page encoder and incremental writer |
| `packed_fp8_mqa.py` | Public API, metadata preparation, per-query operations and DMA descriptors |
| `packed_fp8_sparse.py` | Candidate page tasks, per-query masks and logical top-k mapping |
| `_packed_fp8_math.py` | Head permutation and BF16 products / FP32 head accumulation |
| `_packed_fp8_pipeline.py` | Buffer allocation, fragment schedule and DMA ownership |
| `tuned_block_sizes.py:PACKED_FP8_MQA_CONFIG` | Measured static policy |
| `models/vllm/deepseek_v41/v2/packed_fp8_indexer.py` | Native-owner adapter, Q quantization, ragged grouping and top-k bridge |

The schedule and numerical helpers execute at tracing time. Keep fragment loops
as Python `range` loops: all 128 fragments, including startup and drain, are
unrolled. Keep the order of operations in `emit_tile`: first next-page DMA,
QK issue, older-fragment reduction/store, second next-page DMA. Extracting Python
helpers is safe only when the generated code is checked.

Q has two buffers. Page metadata has four buffers with two-query lookahead.
Output has two tile buffers per query parity (four total), allowing the final
output transfers of Q block n to overlap Q block n+1. Reusing a parity waits for
its recorded rows first; kernel exit drains both parities. Short-context padding
first drains the current outputs. These extra buffers are VMEM/SMEM buffers,
not additional HBM banks.

Head order arranges four queries in the first 128 rows and two in the last 64.
Odd fragments before the final eight use an MXU selector reduction; other
fragments use VPU reductions. This division overlaps resources and is deliberate.
Weighted BF16 products for valid tokens must be finite: dense zero/one selectors
do not preserve independent-query NaN/Inf behavior within a token column.

## Validation and retained performance

Falcon experiment `exp-4onsyjb3t0`, P-state 7, JAX/jaxlib 0.11.2 and libtpu 0.0.50,
compared the former best with the boundary-overlap implementation:

| Seed | Former best, kernel µs | Boundary overlap, kernel µs |
| --- | ---: | ---: |
| 2026 | 395.0174 | 323.4880 |
| 2027 | 395.3746 | 323.4394 |

Each row contains 40 samples per variant in an interleaved comparison. Cache
encoding is outside timing. Kernel time excludes the separate metadata-preparation
kernels; complete module medians were about 402.6 µs and 330.8 µs respectively.
All six full-output comparisons (two seeds, normal/edge/ragged inputs) were
bitwise equal between the two scheduling variants.

Independent numerical correctness is still unresolved: both scheduling variants
have the same sampled reference failures (maximum absolute errors approximately
0.1762 and 0.1839 for the two seeds). Their masks and finite-value classification
matched. Equivalence to the previous implementation is not a claim that this
kernel passes its independent reference or model-level evaluation.

The refactor was checked with local sim-pjrt. After removing source comments,
virtual SSA identifiers and debug marker IDs, all 17,545 static scheduled bundles
matched the measured boundary-overlap snapshot exactly, including physical
registers, operands, branch targets and empty bundles. Complete compute paths,
including their entry/exit branches and verified delay slots:

| Path | Bundles |
| --- | ---: |
| Full tile, prefetch | 4093 |
| Full tile, no prefetch | 3971 |
| Masked tile, prefetch | 4127 |
| Masked tile, no prefetch | 3998 |

Each path retains 768 FP8 matrix instructions and 3072 BF16 weighting instructions.
Prefetch paths retain 512 predicated bank alternatives for 256 logical page DMAs.
These are scheduled bundle counts, not measured stall cycles. The 3500-bundle
optimization target remains unmet. The refactor has not been rerun on hardware;
the 323 µs result belongs to the equivalent pre-refactor source.

The candidate integration was also compiled at P-state 7 with identical
benchmark shapes: Q `[288,32,128]`, page table `[48,768]`, and two packed banks
of `[608256,128]` words. Its dense specialization remains instruction-identical
to the measured baseline after removing debug identities. The sparse
specialization combines the page-valid bit and bank bit into one tag; tag 2
issues bank 0 and tag 3 issues bank 1. This avoids nested predicate combinations.
Assembly-verified selected paths, including branch delay slots:

| Path | Baseline | Sparse |
| --- | ---: | ---: |
| Full tile, prefetch | 4093 | 4084 |
| Full tile, no prefetch | 3971 | 3971 |
| Masked tile, prefetch | 4127 | 4113 |
| Masked tile, no prefetch | 3998 | 3998 |

These compute bodies contain no DMA waits; the two prefetch bodies contain no
internal control-flow branches. Sparse calls have the same explicit scratch
buffers as dense calls, with no K zero-initialization or VMEM copy loops.
The entire sparse kernel contains 17,536 static bundles; the dense kernel still
matches the measured baseline's 17,545 bundles instruction-for-instruction
after debug normalization. Static totals include mutually exclusive paths and
are not execution-cycle counts. Candidate metadata preparation, DMA stalls and
model integration still require hardware timing; fewer compute bundles alone
do not establish end-to-end speedup.

## Reproduce

```bash
# Layout tests can run on CPU.
JAX_PLATFORMS=cpu pytest -q tests/kernels/test_dsv41_packed_fp8_cache.py

# On a TPU v7 host; use a fresh output directory.
LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7' \
python scripts/dsv41_bench/bench_packed_fp8_mqa.py --out /tmp/packed-fp8-mqa
```

The maintained benchmark imports the source directly and needs no experiment
source directory. It reports timing only (`correctness: null`). Use
`--compile-only` under sim-pjrt to compile the fixed shape without device data.
