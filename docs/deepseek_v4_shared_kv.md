# Experimental shared KV for DeepSeek V4 CSA

This path is disabled by default. Set the maximum shared union capacity before
JAX tracing or service startup to opt in:

```bash
export TPU_DSV4_CSA_SHARED_KV_CAPACITY=1280
```

The service accepts capacities `0`, `1024`, `1280`, `1536` and `2048`. Zero retains the original
single-query QK path. Existing gather-chunk configuration is preserved for that
path. This experiment has operator-level validation; it has not been enabled or
benchmarked in the full model service.

## Execution

Groups of four (default) or eight adjacent queries share a union of **physical** KV addresses. Top-k entries
must be unique within a query; negative entries represent padding. Physical
addresses keep different requests separate even when logical positions match.

The implementation sorts addresses once for the entire operator call and
computes exact union lengths. A SparseCore kernel compacts indices and builds
one integer owner bit mask per union entry using native indexed stores. A
bijective slot permutation spreads gathered addresses to reduce read hotspots.
Unused slots contain legal padding addresses and zero membership bits.

The smallest fitting capacity among 1024, 1280, 1536 and the configured maximum is
selected. A union larger than the maximum makes the **entire call** use the
original path. No selected key is silently dropped. Fewer than 256 padded query
rows, an incomplete query group, or a capacity that cannot reduce gathered
rows also use the original path. The API exposes
`attention_shared_kv_min_queries` for controlled operator experiments.

A successful call gathers all groups together, then each Pallas program handles
one shared query group. It computes shared QK and PV with a separate membership
mask and softmax for every query, merging SWA partials and attention sinks.
Invalid KV rows are cleared before QK/PV to prevent NaNs from overlaid cache
storage. Invalid query outputs retain their input padding values.

## Performance findings

The original generic sort/scatter prototype took about 289 microseconds to build
indices for 64 queries, top-k 512, group size four and capacity 1024. Sorting and
counting alone took about 8.5 microseconds. Moving compaction to native
SparseCore operations reduced the complete construction to about 27.4
microseconds, before the later bit-mask simplification.

Prepared-KV attention took about 48.3 microseconds with 16 queries per program
and 36.3 microseconds with four. Replacing division with approximate reciprocal
did not improve performance. Constructing metadata separately for every
64-query chunk remained expensive; whole-call construction and capacity
selection reduced that overhead. A MinHash overlap screen reduced some fallback
costs but also consumed useful-path savings, so it is not included.

Real-index replay shows benefits for the tested 256/1024-query batches, with the
largest gains in 1024-query batches. These are not universal: synthetic moderate
overlap at 256 queries and low-overlap overflow cases can still regress. Sorting
and branch overhead remain on overflow. Keep the feature opt-in and evaluate the
actual workload before using it in a service.

## Validation and measurement scope

Tests cover physical-address unions, owner masks, slot rotation, unaligned
index-vector tails, empty attention, NaN padding, cross-request groups, invalid
query padding, overflow and incomplete query groups. Pure attention and Pallas
interpreter tests run on CPU; SparseCore and full-operator tests require TPU.

Benchmarks include page lookup, shared metadata construction, `csa_gather`, QK,
softmax, PV and merging SWA results. They exclude producing indexer and SWA
inputs and exclude compilation. Real replay uses recorded layer 2/22/36 top-k
indices at query positions starting at 4096 and 8192; Q, packed FP8 KV and SWA
values are synthetic. Query counts describe one operator invocation, not the
context length. Measurements use warm-up and synchronized device loops, with
variant order rotated/reversed. Each loop feeds the preceding output as its next
Q to retain attention work. Neither model-level accuracy nor end-to-end serving
throughput is established by these tests.

## Final operator results (2026-09-07)

Real-index replay at position 4096 with 1024 query rows, H=64, D=512, top-k=512.
Baseline uses QK=1; shared KV uses groups of four.

| Layer | Baseline (μs) | Shared KV (μs) | Latency reduction |
|---|---:|---:|---:|
| 2 | 1269.47 | 928.70 | 26.8% |
| 22 | 1210.38 | 1064.87 | 12.0% |
| 36 | 1282.83 | 917.91 | 28.4% |

All 47 TPU tests passed. All 21 full-operator cases passed numerical comparison
(rtol=0.02, atol=0.02); maximum absolute difference was 0.00390625.
Disjoint synthetic 256/1024-query batches regressed by about 44%/14%.
Recorded indices were replayed with synthetic Q/KV/SWA values.

## Combining shared QK=8 with a 256-query gather chunk

The two controls are available together before JAX tracing / service startup:

```bash
export TPU_DSV4_CSA_SHARED_KV_GROUP_SIZE=8
export TPU_DSV4_CSA_SHARED_KV_CAPACITY=2048
export TPU_DSV4_CSA_GATHER_CHUNK_SIZE=256
```

Shared KV remains disabled by default (capacity 0); group size defaults to four,
and the original gather chunk defaults to 64. The combined configuration is
opt-in. A successful shared path gathers all groups in one call, so the gather
chunk setting controls the independent path and whole-call overflow fallback.
It does not split the shared union into 256-query chunks.

QK=8 can be sensitive to union capacity at small query counts: a previously
measured 256-query case regressed at capacity 1280 and improved with fixed 1536.
Increasing capacity uniformly also adds work to cases that fit in 1024. The
current capacity choice guarantees completeness; it is not a latency oracle.

The combined operator benchmark explicitly disables
`xla_enable_custom_call_region_trace`, `xla_xprof_enable_custom_call_tracing` and
`xla_xprof_register_llo_debug_info` before importing JAX. It compares the original
QK=1/chunk64, QK=1/chunk256, shared QK=8/chunk64 and shared QK=8/chunk256.

### Combined validation results

All 59 TPU tests and 18 full-operator A/B cases passed (maximum absolute error
0.00390625). The following rows use 4096 query rows and recorded indices, with
synthetic Q/KV/SWA values.

| Query start | Layer | QK1/chunk64 (μs) | QK1/chunk256 (μs) | Shared8/chunk64 (μs) | Combined (μs) |
|---|---:|---:|---:|---:|---:|
| 4096 | 2 | 5183.01 | 4475.02 | 3501.05 | 3344.45 |
| 4096 | 22 | 4888.62 | 4192.07 | 3807.05 | 3653.07 |
| 4096 | 36 | 5190.85 | 4394.61 | 2994.38 | 2838.95 |
| 8192 | 2 | 4911.83 | 4179.52 | 3458.79 | 3301.44 |
| 8192 | 22 | 4700.42 | 3989.29 | 4199.99 | 4038.94 |
| 8192 | 36 | 5096.57 | 4305.92 | 2984.52 | 2829.19 |

The combined path reduced latency versus shared8/chunk64 by about 4–5% in
these 4096-query real-index cases; 1024-query shared cases changed by at most
about 0.3%. Shared gather counts are unchanged; the additional 4096-query
improvement has not been isolated to a specific compiler/buffer effect.
At start 8192, layer 22, chunk256 alone was about 1.2% faster than the combined
path. Synthetic low-overlap overflow cases still regressed versus the original
by about 3–6%, despite improving versus shared8/chunk64.
