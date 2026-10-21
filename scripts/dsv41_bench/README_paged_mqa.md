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

The measured configuration keeps six queries resident, four 16K-key DMA buffers,
two Q buffers, two output buffers, and separate QK/reduction accumulator rings.
Each fragment handles 256 keys. At issue depth eight, a static stage consumes
fragment `i-16`, reduces `i-8`, and issues QK for `i`. Loop bounds also cover fill
and drain. Page DMA for a later tile overlaps these operations.

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
The native implementation in this branch matches the tested `staged8` source
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
