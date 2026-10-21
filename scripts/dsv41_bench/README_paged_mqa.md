# Paged FP8 MQA experiment

This branch checkpoints a standalone TPU v7 paged index scorer. It is not wired
into the serving model. The kernel computes weighted ReLU QK scores over a dense
paged context, or paired candidate-page unions, and returns FP32 scores.
It does not compute Top-K or block maxima.

## Interface and ownership

```text
paged_mqa_logits(
    q[T,H,128], weights[T,H], packed_pool,
    block_table[G,P], lengths[T], query_start_loc[G+1],
    page_records=128, q_scale[T,H], config, candidates=None,
) -> scores[T,P*128]
```

`query_start_loc` partitions contiguous query rows by their shared physical page
table. Group sizes can differ or be zero; six queries is only a benchmark
setting. Queries can have different visible prefix lengths. Without candidates,
K reuse is limited to a resident Q tile.

With `candidates[T,C]`, the kernel pairs adjacent queries within each interval,
merges their sorted logical page IDs, and returns `[T,C*128]` scores in the
original candidate slots. A final unpaired query works without crossing the
interval boundary. Inputs must contain unique ascending IDs followed by -1
padding; C and the page-table width must be multiples of the TC lane count.

```text
candidates + page table + query offsets
  -> paired_mqa_metadata()           # SparseCore merge-path, no K copy
     physical page / logical page / Q0 slot / Q1 slot
  -> paged_mqa_logits()              # one K tile serves both Qs
     -> write original slots in VMEM
     -> one contiguous output DMA per query
```

A union page belonging to only one query still participates in both QK products.
Its other result goes to a VMEM sink and is never published. Invalid candidate
slots and tokens beyond each query's visible context receive -inf. K DMA and
matrix computation retain tile padding. This currently merges **128-token
pages**, not the model's smaller candidate blocks; serving integration and a
smaller-block adapter are not included.

K lives in one or two global INT32 banks. Each physical page has 32 rows holding
128 FP8 D128 keys and one row holding 128 FP32 scales: `[33,128]` INT32 words.
The benchmark packs pages before timing. A production cache writer for this
layout is not included. Q uses FP8 E4M3FN and FP32 scales per head; weights and
outputs are FP32. The implementation uses native TPU accumulator refs and has
no CPU interpreter mode.

## Pipeline

`paged_mqa_pipeline.py` owns buffer allocation and scheduling;
`paged_mqa_logits.py` implements DMA, matrix operations, masking, and writes.

The dense configuration keeps six queries resident, four 16K-key DMA buffers,
two Q buffers, two output buffers, and separate QK/reduction accumulator rings.
Each fragment handles 256 keys. At issue depth eight, a static stage consumes
fragment `i-16`, reduces `i-8`, and issues QK for `i`. Loop bounds also cover fill
and drain. Page DMA for a later tile overlaps these operations.

The QK and reduction rings together reserve 896 KiB of MRB per MXU for this
configuration. Explicit VMEM and SMEM allocations are checked against device
capacity; compiler temporaries and spills still need profiling. The default
issue depth is four, so reproduce the reported result with the explicit command
below.

The candidate path keeps two queries resident. Metadata follows DeepGEMM's
[two-query merge-path](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh):
each SparseCore lane searches its merge start once, merges its partition, and
compacts it with prefix counts. Binary-search rounds and small tile operations
are statically unrolled; merge and compaction loops have static trip counts.
Candidate values, union counts, and ragged offsets remain runtime inputs.

Raw FP32 QK results are rounded to BF16 for ReLU and the second MXU operation.
Q scales fold into head weights, which are decomposed into three BF16 residual
components; K scales multiply the reduced scores. Rounding raw logits before
scaling differs from rounding dequantized logits for arbitrary scales. Native
tests check this stated operation order with non-power-of-two scales. Full-size
benchmarks use power-of-two scales. This is not a model-quality validation.

## Reproduction

Use JAX/JAXlib 0.11.2 and libtpu 0.0.50 on TPU v7x:

```bash
python -m scripts.dsv41_bench.bench_paged_mqa \
  --fp8 --requests 48 --queries 6 --width 98304 --page-records 128 \
  --key-tile 16384 --issue-depth 8 --query-group 8 --key-buffers 4 \
  --seed 2026 --repeats 21 --profile --out /tmp/tpu_logs/paged-mqa
pytest tests/kernels/test_dsv41_paged_mqa_logits.py -q
```

Repeat with seed 2027. The benchmark JSON uses host `block_until_ready` timing;
the device figures below come from XProf XLA Ops. XLA Modules additionally
include output layout conversion. Do not compare these timing scopes directly.
The `[T,1,N]` DMA output currently needs a layout conversion to the returned
`[T,N]` tensor. Packing and reference checks are outside kernel timing.

## Measurements

Falcon `exp-nwtbchddrf`, artifact `art-k78d45ckq6`, analysis `an-ci857s11w5`:
v7x-8 allocation, one device executing the kernel, 48 groups × 6 queries,
32 heads, D128, 98,304-token capacity and 98,287 visible tokens. Each figure is
the mean of two seed medians, with five device samples per seed.

| Variant | Kernel time (µs) |
| --- | ---: |
| Packed FP8 control | 943.68 |
| Six K buffers | 913.57 |
| Eight K buffers | 1021.30 |
| Explicit stages, depth four | 1065.25 |
| Explicit stages, depth eight (this kernel) | 888.34 |

All 16 native checks and 12 full-size checks passed in that experiment. The
source hashes and individual seed measurements are in
`paged_mqa_fp8_results.json`. This checkpoint removes the interpreter code and
formats the native implementation; subsequent ablation results are recorded
separately in that JSON.

Falcon `exp-f4z6frgpyy`, analysis `an-m39u778h97`, then compared epilogues at the
same shape and software versions. All 12 native and six full-size checks passed.
The native implementation at that checkpoint matches the tested `staged8` source
after excluding imports, docstrings, and exception message text from AST comparison.

| Epilogue | Kernel time (µs) | Module time (µs) |
| --- | ---: | ---: |
| Staged MXU reduction (this branch) | 897.60 | 989.40 |
| VPU reduction with BF16 rounding | 1021.04 | 1113.25 |
| VPU reduction with FP32 logits | 892.06 | 984.16 |

Removing BF16 rounding from the VPU variant reduces its latency by 12.6%, but
beats the same-run MXU variant by only 0.6%. This branch retains the staged MXU
implementation; the VPU alternatives are experimental measurements, not exposed
kernel modes. Across-run 888.34 µs and 897.60 µs should not be interpreted as a
code regression: the latter reruns the native implementation after cleanup.

The estimated minimum HBM traffic is 737,502,532 bytes. At 3.7 TB/s, the memory
roofline lower bound is 199.33 µs. A measured 888.34 µs corresponds to 22.44%
effective bandwidth. This is not measured bus occupancy; it excludes output
layout conversion and cache construction. The 400 µs target has not been met.

## Paired candidate measurements

Falcon `exp-yvq81h253l`, artifact `art-r4mz388bti`, analysis `an-5wwv0aqhal`:
same software and v7x allocation, one executing device, 48 groups × 6 queries,
32 heads, D128, 98,304-token context. Each query selects 384 pages of 128 tokens
(49,152 tokens). Both paths use 16K key tiles, issue depth eight, and four K
buffers. The independent baseline scores each query's candidate list separately;
the paired path scores each two-query union. Inputs use arbitrary positive
FP32 Q/K scales. Results average two seed medians, five device samples per seed.

| Candidate overlap | Independent scorer (µs) | Paired scorer (µs) | Scorer latency reduction | Metadata (µs) |
| --- | ---: | ---: | ---: | ---: |
| 0% | 2512.20 | 2807.44 | -11.8% | 306.22 |
| 50% | 2512.35 | 2380.70 | 5.2% | 297.84 |
| 100% | 2512.30 | 1524.85 | 39.3% | 297.99 |

Scorer time is XProf XLA Ops and excludes metadata and output conversion.
Metadata time is its standalone XLA Module. Full-module measurements, including
input preparation and output conversion, are:

| Candidate overlap | Independent module (µs) | Paired module (µs) |
| --- | ---: | ---: |
| 0% | 3601.58 | 3150.12 |
| 50% | 3602.19 | 2714.75 |
| 100% | 3602.09 | 1859.18 |

The independent module includes a large XLA page-table gather, so its module
speedup must not be attributed solely to K reuse. With no overlap, pairing
increases scorer latency. This candidate workload is not directly comparable
to the dense six-query-sharing measurements above.

All 15 native TPU tests and six full-shape comparisons passed. Tests cover
ragged, empty and odd groups, overlap, padding, page holes, and logical-position
masking. Four CPU merge-partition tests and an empty-query test also pass.
Source hashes and per-seed samples are in `paired_mqa_results.json`. After the
native run, changes were limited to documentation, a test variable rename, and
the CPU-tested empty-query early return; nonempty arithmetic is unchanged.
