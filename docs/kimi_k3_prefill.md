# Kimi K3 prefill

`TPU_K3_SP_PREFILL=1` enables sequence-parallel prefill on a TPU7x 2×2×4
slice: TP32/EP32, PP1/DP1, 32 TensorCores. Every positive token bucket divisible
by the SP group size uses SP; other buckets retain replicated-token execution.
There is no SP-specific 8192-token scheduler cap. This targets prefill workers in prefill/decode
disaggregation; KV-cache and KDA-state outputs must remain valid for handoff.

The tuned MLA entries are submitted separately in
[PR #1016](https://github.com/vllm-project/vllm-torchtpu/pull/1016).
Apply that tuning change to reproduce the recorded performance numbers below;
it is no longer included in the SP change.

## Retained implementation

- Activations and Attention Residual history are token-sharded between layers.
  Attention gathers inputs and reduce-scatters output projections. MoE routes
  local tokens, gathers latent inputs and top-k routes, and reduces outputs in
  latent space before normalization and up-projection.
- Shared expert weights use TP2 across the two cores on each physical chip.
  Chip-local gather/reduce-scatter handles their token shards.
- Pallas rings map physical chip coordinates, mesh indices, and Torch token
  ownership explicitly. Fused gather/input-projection and output-projection/
  reduce-scatter avoid full intermediate activation buffers when their tile
  alignment permits. Other shard sizes use XLA collectives with the same token
  ownership; their use does not disable SP. Ring buffers remain bounded.
- KDA uses upstream's fused conv1d + GDN v3 path unchanged. The existing Mega
  kernel remains available, but this model adds no Mega dispatch or convolution
  adapter. The 8K comparison below found only a 1.5% operator-level benefit.
- MLA uses the existing paged attention kernel, including for queries with
  earlier KV-cache context. SP preprocessing, the TP2 gate buffer, and fused
  output projection remain; there is no separate Splash attention path.
- Packed KDA/MLA buffers refresh after checkpoint loading and preserve the
  parameter interface across token buckets. Shape, dtype, and layout guards
  retain general paths where specialized kernels cannot apply.
- Fused Attention Residual reduces intermediate memory traffic. MoE uses
  upstream's counting-sort permutation; the competing bounded-inverse kernel
  has been removed from this PR. No routes are dropped or forced balanced.

## Validation and performance

The September 14 rebase was benchmarked with the full 93-layer model, dummy
weights, and the same tuned paged MLA configuration on both revisions:

| Measurement | Main `0adbd1359` | SP PR | Main / PR |
| --- | ---: | ---: | ---: |
| HTTP median, 20 requests | 847.175 ms | 341.090 ms | 2.484× |
| Device median, slowest of 32 cores | 841.127 ms | 331.992 ms | 2.534× |
| HBM/core before KV allocation | 63.92 GiB | 78.13 GiB | — |

Two warmups precede HTTP timing; device timing uses three separate profiled
requests. Both runs passed a four-token cached-generation smoke check with
finite log probabilities. Real-checkpoint correctness and end-to-end state
transfer to a separate decode worker remain unverified for this configuration.

The matching MLA entry uses 512 queries, eight KV pages, two query splits,
and a 60 MiB VMEM limit. Upstream's unified pool produces 324-token pages
(81 packed FP8 units), despite the command-line block size of 256. The exact
8K mixed key is tuned; the subsequent sweep below adds smaller prefill keys.
The isolated operator improved from 26.227 to 1.950 ms. Random-input tests with
zero/1024 cached tokens matched valid cache writes exactly and had at most
2.69% sampled relative L2 error against FP32 attention.

The measured PR source was `b3e617cf0` with this same tuning entry applied;
main received an identical entry. Both used the same external dummy fixture,
fixed random router bias, and random prompt. Source hashes, raw timings,
profiles, tuning results, and reproduction details are in
`tpu-exp/docs/K3_PREFILL_REBASE_20260914.md` and
`tpu-exp/results/production-port/rebase-20260914-tuned/`.

A subsequent PR-only sweep added matching MLA entries for smaller prefill
buckets, using the same dummy fixture and runtime:

| Prompt | HTTP median | Device median |
| --- | ---: | ---: |
| 1K | 190.380 ms | 183.210 ms |
| 2K | 312.554 ms | 303.015 ms |
| 4K | 559.657 ms | 551.247 ms |

These historical measurements predate generalized SP: smaller buckets used
replicated-token TP and fused KDA, while SP was guarded at 8K. Thus 4K was slower
than the specialized 8K path. On a
representative core, 4K spends 155.642 ms in Attention Residual and 148.006 ms
in all-reduces, versus 9.188 ms in tuned MLA. This is an implementation limit,
not a fundamental restriction on sequence parallelism. The full report and
artifacts are in `tpu-exp/docs/K3_PREFILL_SIZES_20260914.md`.

### Generalized SP sweep (September 15)

The full 93-layer dummy-weight model now uses SP for all five sizes below.
MLA was tuned for the 64-entry page tables required by the 18,432-token model
limit. HTTP medians use 20 requests after two warmups; device medians use three
requests, taking the slowest of 32 cores per request.

| Prompt | HTTP median | Device median | Device speedup vs earlier PR |
| --- | ---: | ---: | ---: |
| 1K | 120.602 ms | 115.335 ms | 1.589× |
| 2K | 160.108 ms | 153.602 ms | 1.973× |
| 4K | 192.016 ms | 183.063 ms | 3.011× |
| 8K | 341.082 ms | 331.897 ms | 1.000× |
| 16K | 997.885 ms | 983.268 ms | — |

All sizes passed four-token cached-generation checks with finite log
probabilities. HBM before KV allocation was 78.23 GiB/core. At 16K, the current
XLA RS fallback and larger MLA workload are material costs: a representative
core records 335 ms in XLA RS events (including waits) and 131.480 ms in MLA.
This sweep used Mega KDA at 8K; the current model uses upstream fused KDA.
The full-model timings above have not been remeasured after that simplification.

Cold engine setup took 56.80 minutes, including 51.18 minutes reported as
compilation. Smaller SP buckets compile more slowly than their earlier
replicated-token graphs; each rank builds a full model executable per bucket.
A live sample found the active work in an `xla_tpu_thread`, but the exact
compiler pass has not been profiled. Persistent caching was disabled for this
run. Compilation is excluded from the runtime numbers above.

The measured source was `206589637`. Raw samples (including a first-profile
1K outlier and 16K HTTP outliers), tuning checks, compile timings, and
reproduction details are in `tpu-exp/docs/K3_SP_PREFILL_SIZES_20260915.md` and
`tpu-exp/results/production-port/sp-generalize-20260914/`.

| Historical configuration | 93-layer, 8K result | Qualification |
| --- | --- | --- |
| Mega KDA + Splash MLA | 330.335 ms HTTP median; 318.338 ms device; 78.24 GiB/core before KV cache | Dummy weights; preceding configuration, not current paged MLA |
| Earlier pipeline + Splash MLA | 338.110 ms dummy; 414.184 ms real weights | Real-checkpoint recursive-prefill smoke tests inspected: 20 correct answers |
| Original JAX prototype | 305.005 ms; 78.122 GiB/core | Dummy weights; failed comparison with a preceding implementation; not a correctness reference |

The isolated KDA operator at 8192 tokens and three local 128-wide heads measured
0.999 ms for Mega plus contiguous convolution, versus 1.260 ms for unmodified
Mega. Fresh, carried/padded, mixed, decode, and empty cases passed on four hosts.
Against an independent FP64 recurrence with BF16 activation boundaries, Mega's
output/state relative L2 errors were 0.9049%/0.6433%.

The isolated MLA operator, including projections and cache writes, measured
1.994–2.007 ms for paged MLA versus 0.777–0.784 ms for the preceding expanded
Splash implementation. This is not a measured full-model regression. Paged MLA
is retained because prefill queries can have existing cached context.

CPU regression coverage includes checkpoint layouts, token-bucket parameter
consistency, SP latent reduction order, chip-local shared experts, and fake
shapes for Torch/JAX bridges. Kernel tests cover physical topology, collective
ownership, Attention Residual, and the existing KDA implementations.

## 8K KDA microbenchmark (2026-09-15)

The complete KDA operator was compared through the production Torch/Pallas
boundary on TPU7x, with BF16 activations, three local 128-wide heads, 64 state
slots, FP32 recurrent state, and donated state buffers. Timings include
convolution, recurrence, gated output normalization, dispatch, and state writes;
linear projections and collectives are outside this operator.

| 8K case | Mega device median | Upstream fused device median | Mega speedup |
| --- | ---: | ---: | ---: |
| Fresh prefill | 483.043 µs | 490.278 µs | 1.015× |
| With existing state | 483.025 µs | 490.237 µs | 1.015× |

All 32 cores ran the comparison. Device medians use 20 invocations per variant
from each of 16 complete device-plane captures (320 samples); partial sibling
captures are excluded. Execution order alternated. Fresh-prefill host medians
were 0.838 ms and 0.844 ms respectively, across 1,280 samples per variant.
Outputs and live recurrent states differed by approximately 0.65% relative L2;
convolution states matched exactly and all outputs were finite. This is a
random-input comparison, not an independent reference or real-weight validation.

The 7.2 µs operator difference extrapolates to about 0.5 ms over 69 KDA layers,
not a measured full-model gain. This small benefit does not justify the extra
adapter code, so the PR retains upstream KDA. Mega source was `2e53b7725` and
upstream source was `0adbd1359`, using the same installed runtime. Raw results
and reproduction scripts are in the benchmark workspace under
`results/production-port/kda-mega-vs-fused-20260915/`. Further performance
iteration is limited to 8K prefill.

## Reproduction

```bash
export TPU_K3_SP_PREFILL=1
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false"
```

The preceding serving benchmarks used TP32 + EP, language-model-only,
`--max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 8`,
`--gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --block-size 256`,
64 cache blocks, compilation sizes `[1,8,8192]`, and disabled prefix caching
and async scheduling. Multihost ranks, ports, and checkpoint paths depend on
the deployment.

The extra fan-in prefill dummy initializer is an external benchmark fixture,
not production model-loading behavior. Historical timings used that fixture;
plain dummy loading after its removal is a different workload. For comparable
new dummy benchmarks, call `tools.k3_dummy_fixture.install()` from the separate
`tpu-exp` workspace in **each worker before model loading**. It preserves the
original initializer and norm scales and leaves real checkpoints unchanged.
Fixed random router bias and finite randomized MXFP4 dummy weights remain in
their existing loading hooks.

Detailed experiment history, negative findings, routing measurements, and
artifact locations are preserved in the separate `tpu-exp` workspace:

- `docs/K3_PRODUCTION_PORT_FINDINGS.md`: archived full experiment record.
- `docs/PALLAS_XLA_COLLECTIVES_2X2X4.md`: isolated Pallas/XLA collective sweep.
- `results/production-port/`: frozen source snapshots, launch scripts, raw
  timing data, device traces, and recursive-prefill responses.

Those local artifacts are supporting evidence, not files distributed by this
repository. Historical snapshots remain unchanged for reproducibility.
