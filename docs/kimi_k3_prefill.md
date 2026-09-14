# Kimi K3 prefill

`TPU_K3_SP_PREFILL=1` enables the optimized 8192-token prefill path on a TPU7x
2×2×4 slice: TP32/EP32, PP1/DP1, 32 TensorCores. The scheduler must cap
`max_num_batched_tokens` at 8192. Smaller token buckets retain the required
non-SP execution paths. This targets prefill workers in prefill/decode
disaggregation; KV-cache and KDA-state outputs must remain valid for handoff.

## Retained implementation

- Activations and Attention Residual history are token-sharded between layers.
  Attention gathers inputs and reduce-scatters output projections. MoE routes
  local tokens, gathers latent inputs and top-k routes, and reduces outputs in
  latent space before normalization and up-projection.
- Shared expert weights use TP2 across the two cores on each physical chip.
  Chip-local gather/reduce-scatter handles their token shards.
- Pallas rings map physical chip coordinates, mesh indices, and Torch token
  ownership explicitly. Fused gather/input-projection and output-projection/
  reduce-scatter avoid full intermediate activation buffers.
- KDA uses the existing Mega recurrence for an opted-in single-sequence 8K
  prefill, with contiguous convolution and FP32 tap accumulation. Other batches
  use upstream's fused conv1d + GDN v3 path. Dense caches and the unified state
  pool use the same selection. Upstream's removed manual/chunked dispatcher
  and its environment selectors are not restored.
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
8K mixed key is tuned; smaller buckets retain their existing settings.
The isolated operator improved from 26.227 to 1.950 ms. Random-input tests with
zero/1024 cached tokens matched valid cache writes exactly and had at most
2.69% sampled relative L2 error against FP32 attention.

The measured PR source was `b3e617cf0` with this same tuning entry applied;
main received an identical entry. Both used the same external dummy fixture,
fixed random router bias, and random prompt. Source hashes, raw timings,
profiles, tuning results, and reproduction details are in
`tpu-exp/docs/K3_PREFILL_REBASE_20260914.md` and
`tpu-exp/results/production-port/rebase-20260914-tuned/`.

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
