# Paged FP8 MQA experiment

This branch checkpoints a standalone TPU v7 paged index scorer. It is not wired
into the serving model. The kernel computes weighted ReLU QK scores over a dense
paged context and returns FP32 `[T, N]`; it does not compute Top-K or block maxima.

## Interface and ownership

```text
paged_mqa_logits(
    q[T,H,128], weights[T,H], packed_pool,
    block_table[G,P], lengths[T], query_start_loc[G+1],
    page_records=128, q_scale[T,H], config,
) -> scores[T,P*128]
```

`query_start_loc` partitions contiguous query rows into groups sharing one page
table. Group sizes can differ or be zero; six queries is a benchmark setting,
not an interface requirement. Queries can have different visible prefix lengths.
The caller supplies this grouping. The kernel does not compare page tables or
form candidate unions: queries using `[A,B]` and `[B,C]` need separate groups.
Their shared page B is currently read twice. Reuse is limited to a resident Q
tile, so a group larger than that tile also rereads its K pages.

K lives in one or two global INT32 banks. Each physical page has 32 rows holding
128 FP8 D128 keys and one row holding 128 FP32 scales: `[33,128]` INT32 words.
The benchmark packs pages before timing. A production cache writer for this
layout is not included. Q uses FP8 E4M3FN and FP32 scales per head; weights and
outputs are FP32. The implementation uses native TPU accumulator refs and has
no CPU interpreter mode.

## Pipeline

`paged_mqa_pipeline.py` owns buffer allocation and scheduling;
`paged_mqa_logits.py` implements DMA, matrix operations, masking, and writes.

The measured configuration keeps six queries resident, four 32K-key DMA buffers,
two Q buffers, two output buffers, and separate QK/reduction accumulator rings.
Each fragment handles 256 keys. At issue depth eight, a static stage consumes
fragment `i-24`, reduces `i-16`, converts QK `i-8` into the BF16 ring, and issues
QK for `i`. Loop bounds also cover fill and drain. Page DMA for a later tile
overlaps these operations. The BF16 ring uses 0.75 MiB for this configuration.
K scales are read from the resident tile when scores are consumed; separate
scale-transfer rings are unnecessary with this schedule.

The QK and reduction rings together reserve 896 KiB of MRB per MXU for this
configuration. Explicit VMEM and SMEM allocations are checked against device
capacity; compiler temporaries and spills still need profiling. The default
issue depth is four, so reproduce the reported result with the explicit command
below.

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
  --key-tile 32768 --issue-depth 8 --query-group 8 --key-buffers 4 \
  --seed 2026 --repeats 21 --profile --out /tmp/tpu_logs/paged-mqa
pytest tests/kernels/test_dsv41_paged_mqa_logits.py -q
```

Repeat with seed 2027. The benchmark JSON uses host `block_until_ready` timing;
the device figures below come from XProf XLA Ops. XLA Modules additionally
include surrounding device operations. Do not compare these timing scopes directly.
The historical `[T,1,N]` DMA output needed conversion to the returned `[T,N]`
tensor. The current kernel writes directly to `[T,N]` using two-dimensional
single-row DMA slices; see the direct-output comparison below. Packing and
reference checks are outside kernel timing.

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
| Explicit stages, depth eight (initial checkpoint) | 888.34 |

All 16 native checks and 12 full-size checks passed in that experiment. The
source hashes and individual seed measurements are in
`paged_mqa_fp8_results.json`. The initial checkpoint removed the interpreter
code and formatted the native implementation; subsequent results are recorded
separately in that JSON.

Falcon `exp-f4z6frgpyy`, analysis `an-m39u778h97`, then compared epilogues at the
same shape and software versions. All 12 native and six full-size checks passed.
The native implementation at `920a7292a` matches the tested `staged8` source
after excluding imports, docstrings, and exception message text from AST comparison.

| Epilogue | Kernel time (µs) | Module time (µs) |
| --- | ---: | ---: |
| Staged MXU reduction (initial checkpoint) | 897.60 | 989.40 |
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

## Shared-K FP8 follow-up

The paired-candidate experiment was reverted in `8dc381676`. These measurements
use the original dense shared-K workload and preserve its precision contract:
48 groups × 6 queries, H32, D128, 98,304-token capacity, two FP8 banks, and the
same two seeds. Six resident queries still reuse each K tile. Each figure is
the mean of two seed medians, with five XProf samples per seed.

Falcon `exp-vb4z5v984g`, artifact `art-0xcwsbzip1`, analysis `an-ofm5gvzk7u`:

| Variant | Kernel time (µs) | Module time (µs) |
| --- | ---: | ---: |
| Original pipeline, 16K K tile | 887.45 | 979.32 |
| Original pipeline, 32K K tile | 861.82 | 953.96 |
| BF16 logits staging, 16K K tile | 818.39 | 910.09 |
| BF16 logits staging, 32K K tile | 794.51 | 886.43 |

The additional BF16 stage ends the FP32 QK-result lifetime before preparing the
reduction RHS. Source-attributed sim-pjrt LLO removes spills attributed to that
FP32 pop and reduces the static live spill peak from 0.63 to 0.41 MiB, while
adding a 0.75 MiB BF16 ring. Static spill counts are not dynamic traffic; the
hardware comparison establishes the benefit. All eight native checks and six
full-shape comparisons passed. The BF16-stage implementation passed these comparisons before the final
branch cleanup below.

The combined change reduces scorer latency by 10.47%. Minimum HBM traffic is
unchanged, giving 25.09% effective bandwidth at 794.51 µs. Module time includes
the existing output layout conversion. This remains a standalone scorer;
serving/model-quality validation and the 400 µs target remain outstanding.

Other measured variants were rejected: smaller math groups, repeated steady
loops instead of full unrolling, and removal of the scale rings alone were
slower. The 48K and 96K K tiles were also slower than 32K. Native tests cover
selected tile configurations; explicit `query_tile=5, key_tile=512` triggered
an output-DMA layout failure in the original pipeline under libtpu 0.0.50.
The direct-output change below also fixes that case. Detailed measurements and
source hashes are in `paged_mqa_fp8_results.json`.

A final run (`exp-o0ftn9bnzl`, artifact `art-j7t3dm8gd8`, analysis
`an-3xlth3rce0`) replaced complementary bank-selection and prefetch branches
with one conditional each. The staged 32K kernel measured 796.16 µs before
this cleanup and 795.86 µs afterward; this difference is within timing
variation. The original 16K control measured 891.97 µs in the same run.
Final module time was 887.43 µs, including the existing output conversion.
All 16 native checks and six full-shape comparisons passed. The code at
`e1054febd` matches this variant's AST after removing docstrings.

Before direct output, the approximately 92 µs module/kernel gap included a
materialized conversion from the kernel's `[T,1,N]` DMA layout to the returned
`[T,N]` layout. At
288 queries and 98,304 positions, the logical FP32 output is 108 MiB. Module
time is the relevant device duration when the caller needs the returned tensor;
the scorer-only timing excludes this extra work.

## Direct two-dimensional output

Falcon `exp-rysox43mh8`, artifact `art-l8smsanokh`, analysis `an-osmqlywm24`
compares the `e1054febd` pipeline with direct output at the same 32K K tile,
depth eight, four K buffers, 96K context, software versions, and two seeds:

| Output | Kernel time (µs) | Module time (µs) |
| --- | ---: | ---: |
| `[T,1,N]` followed by layout conversion | 794.25 | 886.20 |
| Direct `[T,N]` | 796.93 | 800.16 |

Direct output reduces complete module latency by 9.71%. The scorer itself is
2.68 µs slower, but the separate output conversion is gone: `copy.4` takes
88.60 µs in the control and is absent from the candidate's native XProf trace
and local final LLO. About 3.23 µs of input-copy and module overhead remains.
QK, weighting, precision, buffer allocation, and pipeline scheduling are unchanged.

DMA now uses size-one row slices on both refs instead of squeezing the row
axis. This permits writing directly into the final tiled two-dimensional
output without an intermediate HBM result. All 12 candidate native checks,
four baseline native checks, and two full-size comparisons passed. Tests cover
ragged groups crossing row boundaries, trailing partial tiles, empty groups,
page holes, arbitrary positive scales, one/two K banks, and two/four K buffers.
The previously failing `query_tile=5, key_tile=512` case also passes.
