# Fixed weight-buffer fused MoE experiments

Measurements: September 21, 2026, single-host TPU7x, EP8 / TP1.

The current implementation is
[`streaming_fused_moe`](../../src/vllm_torchtpu/kernels/experimental/streaming_fused_moe/README.md).
It uses four weight slots targeting 2 MiB each: gate/up share two slots and down
uses two slots. Scales are pipelined with the corresponding weight slices.
FFN intermediate activations stay in VMEM. Original V2 and the adaptive kernel
are unchanged. This branch is a standalone prototype, without serving integration.
The earlier `tiled_fused_moe` experiment is also retained for reproducibility;
it uses a different schedule with HBM intermediate slabs.

## Important: channel-wise results remain inconclusive

**Changing block-wise FP4 to channel-wise FP4 changes the quantized weights.
The channel-wise results are provisional and require additional tests with real
checkpoint weights. They do not establish preserved model quality.**

All results below use synthetic weights and inputs. In particular, the NVFP4
fixture draws uniform FP4 bit patterns and independent block scales, rather than
sampling a real model's weight distribution. The observed 24.7% output relative
L2 difference is reproducible for this fixture, but does not mean model accuracy
drops by 24.7%, nor predict the corresponding difference on a real checkpoint.
The measured speedup must not be treated as a quality-preserving optimization.

Before promoting channel-wise storage, repeat the comparison using the same real
NVFP4 checkpoint for both paths, measure weight and layer-output errors, then
validate perplexity or task accuracy and complete-model performance. Preserve the
original block-wise path as the reference. No serving loader is changed here.

## Conditions and interpretation

| Setting | Qwen3.5 | Qwen3.8 |
| --- | --- | --- |
| Hidden / intermediate | 4096 / 1024 | 8192 / 2048 |
| Experts / top-k | 512 / 10 | 512 / 10 |
| Input tokens per rank / total | 4096 / 32768 | 4096 / 32768 |
| Original weight path | Per-channel FP8 E4M3 | Block16 NVFP4 requantized to runtime block64 FP4 E2M1 |
| Baseline | Original V2, 3+3 buffers | Adaptive V2, 2+1 buffers |
| Current streaming capacity | 1024 | 128 |
| Output buffers | 2 | 1 |
| Gate compute rows | Full FP8 tile | 128 |
| Down compute columns | Full DMA width | 512 |

JAX/jaxlib 0.10.2, libtpu 0.0.47, numpy 2.3.5. Weights use seed42 and inputs/gating
seed43. Routing is random. Eight JAX devices are four physical TPU7x chips.
Each performance case warms up ten times, measures fifty synchronized host calls,
and profiles ten device calls. Device times below take the maximum across eight
ranks for each call, then the median across ten calls. Compilation, warmup and
correctness checking are excluded. External conversion is included where stated.
Comparisons span separate captures; small changes are not significance tests.

A complete layer includes input quantization, routing, token all-gather, fused
expert computation/transport and combine. This is not a complete model forward,
DP throughput benchmark or serving latency measurement.

Effective utilization uses useful GEMM FLOPs only:

```text
FLOPs = 6 * total_tokens * topk * hidden * intermediate
effective MFU = FLOPs / (elapsed_seconds * 4 * 4614e12)
```

Use complete-layer time for layer MFU and fused time for fused MFU. The denominator
is the [TPU7x FP8 peak](https://docs.cloud.google.com/tpu/docs/tpu7x). FP4 is widened
to FP8 for MXU computation. These values are not hardware MXU busy counters;
padded or otherwise redundant computations do not count as useful FLOPs.
FP8 paths are W8A8, with FP32 accumulation and BF16 layer output.

## Current layer and fused timings

| Model and implementation | Layer ms | Layer MFU | Fused ms | Fused MFU |
| --- | --- | --- | --- | --- |
| Qwen3.5 FP8, original V2 3+3 | 2.85135 | 15.67% | 1.85874 | 24.04% |
| Qwen3.5 FP8, streaming M1024 | 2.89246 | 15.45% | 1.89479 | 23.58% |
| Qwen3.8 NVFP4, adaptive 2+1 | 19.42689 | 9.20% | 17.75109 | 10.07% |
| Qwen3.8 NVFP4, streaming M128 | 22.91718 | 7.80% | 21.23697 | 8.42% |
| Qwen3.8 FP8, streaming M128 | 16.49131 | 10.84% | 14.81568 | 12.06% |

Qwen3.5 streaming fused time increased 1.94% against original V2. Qwen3.8 NVFP4
M128 is still slower than adaptive 2+1, despite the smaller weight scratch.
The ordinary Qwen3.8 FP8 and NVFP4 benchmark fixtures use different numerical
weights; their 6.42 ms fused-time difference is not an isolated unpack-cost
measurement. The external-conversion experiments below use a common NVFP4 source.

Both original-format streaming cases matched their respective baseline outputs
bit-for-bit over all 32768 tokens. Sampled dense-reference relative L2 was 5.4301%
for Qwen3.5 and 5.3986% for Qwen3.8 NVFP4, the same as the matching baselines.
The ordinary Qwen3.8 FP8 fixture passed its sampled dense check (5.4307%);
it has no claim of bit equality with a differently quantized NVFP4 fixture.

Earlier Qwen3.8 iterations illustrate the token-capacity tradeoff:

| Configuration | Fused ms |
| --- | --- |
| Fixed capacity1024, gate compute rows512 | 26.45751 |
| Adaptive five height rungs, gate compute rows512 | 26.60802 |
| Two height rungs, gate compute rows512 | 23.65117 |
| Fixed capacity128, gate compute rows128 | 21.23697 |

The seed's experts receive 580--702 true rows, mean640. M1024 computes 524288 rows;
M128 computes 363776 rows after tile padding. This accounts for substantial
extra work at M1024. It is not safe to size buffers assuming balanced routing:
correctness tests also cover 32K tokens selecting the same experts and idle ranks.

## VMEM and bottleneck observations

| Weight-buffer configuration | Pure weight scratch |
| --- | --- |
| Qwen3.5 FP8 original 3+3 | 36 MiB |
| Qwen3.8 packed FP4 adaptive 2+1 | 40 MiB |
| Streaming, either tested geometry | 8 MiB |

Streaming explicit scratch including scales/activation staging is approximately
33.672 MiB for Qwen3.5 M1024, 14.070 MiB for Qwen3.8 NVFP4 M128, and 13.023 MiB for
Qwen3.8 FP8 M128. These are declared-layout estimates, not compiler VMEM peaks.
Fixed weight scratch does not make total VMEM independent of hidden size,
intermediate size or token capacity. Full intermediate/output rows still reside
in VMEM. Some larger-capacity variants exceeded the compiler's VMEM budget.

A previous custom 48 MiB compiler-budget experiment produced incorrect outputs.
Its cause remains unresolved; the current streaming interface rejects reduced
budgets. Default-budget correctness does not validate that failed variant.

A detailed, instrumented Qwen3.8 NVFP4 M128 trace attributed approximately
17.93 ms of a 21.74 ms fused call to block-scaled dot regions. These regions include
FP4 widening, matmul, scale multiplication and FP32 partial-sum accumulation;
they are not pure MXU execution. Explicit weight-ready waits averaged only
0.0099 ms. Serializing weight DMA increased normal fused time from 21.237 ms to
28.575 ms. Instrumentation itself added approximately 2.37% overhead.

## Token all-gather before fused

| Configuration | SparseCore collective ms | Exposed TC wait ms |
| --- | --- | --- |
| Qwen3.5 original V2 | 0.583 | 0.369 |
| Qwen3.8 FP8 streaming | 1.150 | 0.946 |
| Qwen3.8 NVFP4 streaming | 1.150 | 0.944 |
| Qwen3.8 NVFP4 adaptive 2+1 | 1.150 | 0.946 |

The exposed time is the activation all-gather `async-done` duration on the
TensorCore stream, not the collective's entire async lifetime and not
`layer_time - fused_time`. Qwen3.8 covers about 0.207 ms with routing preparation,
then directly waits about 0.945 ms. Approximately 0.045 ms of scatter/table work
remains after the collective before fused starts. The ordinary wrapper requires
all gathered tokens before launching the fused kernel.

## External FP4-to-FP8 conversion POC

Two isolated wrappers reuse the unchanged FP8 streaming kernel:

1. **Channel storage:** at load time, requantize original block16 FP4 directly to
   one scale per output channel, using block8192 for gate/up and block2048 for
   down. Keep FP4 in HBM. Widen the FP4 values losslessly to FP8 each forward;
   apply channel scales in the existing FP8 kernel epilogue.
2. **Block64 storage:** preserve the original block64 FP4 storage. Each forward,
   apply its block scales and quantize to per-channel FP8 outside fused.

Gate/up and down require different contraction lengths. Setting the serving
loader's single shared block-size setting to 8192 would not implement case 1:
existing paths can pad the intermediate dimension to the common block size.
The POC uses separate conversions and does not modify serving behavior.

| Qwen3.8 path | Layer ms | Fused ms | External conversion ops ms |
| --- | --- | --- | --- |
| Channel FP4 storage, runtime widening | 17.48090 | 14.68663 | 2.06831 |
| Same channel weights, FP8 expanded before timing | 16.55365 | 14.86572 | Excluded |
| Block64 FP4 storage, runtime scale and FP8 conversion | 22.92244 | 14.68159 | 7.29413 |
| Same converted block64 weights, expanded before timing | 16.54366 | 14.87338 | Excluded |

Runtime channel widening overlaps approximately 1.157 ms with actual SparseCore
all-gather execution; the all-gather wait drops to 0.026 microseconds. The complete
layer still costs about 0.927 ms more than its preexpanded control. The async
collective lifetime becomes longer because its result is consumed later; actual
collective execution remains about 1.159 ms.

The channel POC is 23.72% faster than the NVFP4 streaming case and 10.02% faster
than adaptive 2+1, **with changed weight quantization and unresolved model quality**.
The current JAX block64 external converter provides no net layer speedup; this
is not a conclusion about an optimized Pallas converter's potential.

Each rank keeps 1.5 GiB of FP4 weight values and needs3 GiB of temporary expanded
FP8 weights for one layer. Channel scales use about 3 MiB/rank; block64 scales
use 192 MiB/rank. The compiler's extra temporary memory agrees with approximately
3 GiB/rank for widening. Production reuse of a single layer workspace has not
been integrated. The benchmark additionally retains a preexpanded control copy.

### Numerical quality and independent checks

| Comparison | Relative L2 |
| --- | --- |
| Channel POC vs original block64 fused output, all 32768 tokens | 24.7192% |
| Block64 external FP8 converter vs original block64 fused, all tokens | 7.3456% |
| Channel dense vs block64 dense, same 64 sampled tokens | 24.1797% |
| Channel fused vs block64 fused, same 64 sampled tokens | 24.7691% |

Both external paths match their own preexpanded control bit-for-bit over the
full output. Their fused-vs-converted-dense errors are about 5.4%, within the
existing kernel's6% aggregate /7.5% worst-token limits. These checks use the
already-converted weights and do not validate quantization quality.
The block64 external converter's error against original-weight dense is 7.5856%
aggregate and 8.1632% worst-token, exceeding those original-reference limits.

An independent NumPy lookup-table quantizer matched 102,752,256 channel-wise
FP4 values and their scales exactly, including gate/up layout boundaries.
Dense MoE without the fused kernel still reproduced 24.18% difference. Independent
CPU FFNs for two complete experts showed about 13% gate/up projection difference,
19% intermediate difference and 24% final difference. This supports quantization
as the main source of this fixture's discrepancy, without proving real-model
quality. Small block64 rounding-boundary discrepancies in the independent
quantizer caused only about 0.4% single-expert output difference.

### Zero-value investigation

Statistics sample 32 output channels per matrix in all 512 experts, retaining the
complete contraction dimension (134,217,728 W1 and 33,554,432 W2 values).

| Representation | W1 zeros | W2 zeros |
| --- | --- | --- |
| Original block16 | 12.50% | 12.51% |
| Runtime block64 | 16.07% | 16.08% |
| Channel-wise FP4 | 18.45% | 18.44% |

Only about 2.4% of all weights become additional zeros relative to block64.
They contribute about 4.6% of the weight-difference squared norm. Restoring these
new zeros to their block64 values leaves 23.4%--24.0% output difference in two CPU
expert ablations. Nonzero-to-nonzero rounding changes dominate this fixture.
The zero threshold is approximately `max(abs(channel))/24`; absolute smallness
alone is not the issue because scale changes with the channel's magnitude.

## Reproduction

Install the project's TPU dependencies and test extras on an eight-device TPU7x
host. The measured environment used the versions listed above. Run one TPU
process at a time. Supply a model directory or config JSON for its geometry;
the scripts generate synthetic weights and do not load checkpoint tensors.

```bash
export TPU_SKIP_MDS_QUERY=true JAX_PLATFORMS=tpu PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/kernels/experimental/streaming_fused_moe

python scripts/benchmark_adaptive_fused_moe.py \
  --model /path/to/qwen35/config.json --implementation v2 \
  --weight-format fp8 --capacity 128 --profile-iterations 10 \
  --output /tmp/q35_v2
python scripts/benchmark_adaptive_fused_moe.py \
  --model /path/to/qwen35/config.json --implementation streaming \
  --weight-format fp8 --capacity 1024 --output-buffers 2 \
  --profile-iterations 10 --compare-output /tmp/q35_v2/output.npy \
  --output /tmp/q35_streaming

python scripts/benchmark_adaptive_fused_moe.py \
  --model /path/to/qwen38/config.json --implementation adaptive \
  --weight-format nvfp4 --capacity 128 --profile-iterations 10 \
  --output /tmp/q38_2p1
python scripts/benchmark_adaptive_fused_moe.py \
  --model /path/to/qwen38/config.json --implementation streaming \
  --weight-format nvfp4 --capacity 128 --gate-compute-rows 128 \
  --down-compute-cols 512 --output-buffers 1 --weight-tile-mib 2 \
  --profile-iterations 10 --compare-output /tmp/q38_2p1/output.npy \
  --output /tmp/q38_streaming

python scripts/benchmark_external_fp8_moe.py \
  --storage-scale channel --compare-output /tmp/q38_2p1/output.npy \
  --output /tmp/q38_channel_poc
python scripts/benchmark_external_fp8_moe.py \
  --storage-scale block64 --compare-output /tmp/q38_2p1/output.npy \
  --output /tmp/q38_block64_poc
python scripts/analyze_fused_moe_profile.py /tmp/q38_channel_poc/external \
  --fused-name streaming_fused_moe
```

For the ordinary Qwen3.8 FP8 timing, use the streaming command with
`--weight-format fp8` and omit the NVFP4 `--compare-output` argument. Output
directories must not already exist. Device profiles, timing samples, generated
wrapper sources and numerical results are written to those directories.

[`fixed_vmem_moe_20260921.json`](fixed_vmem_moe_20260921.json) records the measured
case identifiers, device samples and numerical-audit data. Large tensor dumps
and binary profiles are not part of the Git tree. The RHS DMA/matmul investigation
scripts are retained under `scripts/benchmark_rhs_overlap.py`,
`benchmark_dma_sizes.py`, `benchmark_vmem_rhs_pair.py` and `analyze_rhs_overlap.py`.

## Commit-time validation

The unchanged streaming implementation previously passed all 11 TPU tests
(134.55 seconds), including bitwise original/adaptive baseline comparisons.
The commit-time attempt to rerun streaming and earlier tiled tests could not
initialize TPU devices: another workspace's DP4/TP2 serving workers held the
VFIO devices (`Device or resource busy`). This was an environment failure,
not a completed numerical regression run. The serving processes were left
running. Repository pre-commit checks and benchmark CLI/wrapper checks were
run for the committed files. The external-conversion benchmark was packaged
from the measured POC; its portable entry point still needs a TPU rerun when
the device is available.
