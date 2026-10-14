<!-- SPDX-License-Identifier: Apache-2.0 -->
# DeepSeek V4 routed MoE decode microbenchmark

This benchmark executes one routed MoE layer on eight TPU devices. It does not
construct a vLLM engine, load the complete model, or allocate KV cache. DP8/TP1/EP8
is represented by a single-controller JAX mesh: each device starts with its own
hidden states and logits and owns 32 of the 256 experts. This measures the actual
repository kernels and equivalent collectives, without the Torch distributed
driver/bridge overhead of a full deployment.

## Environment and reproducible commands

The tested checkout base is `2fac4ce4f355a96447f568679d09b0457465a2a5`.
Each run saves SHA256 hashes of the actual source files, including the uncommitted
benchmark. A copied checkout may carry `BENCH_SOURCE.json`; all listed hashes are
verified before running. The benchmark inserts that checkout's `src` first and
rejects mixed kernel imports. No source edits are needed in the installed package.

On the prepared `tpu2` host:

```bash
ssh tpu2
source "$HOME/.vllm-torchtpu-setup/activate.sh"
cd "$HOME/dsv4-moe-bench-20261009"
python -m pip check
python -m unittest discover -s tests -p test_benchmark_dsv4_moe_decode.py
python scripts/benchmark_dsv4_moe_decode.py --mode preflight --output results/preflight-new
python -u scripts/benchmark_dsv4_moe_decode.py --mode smoke --output results/smoke-new --trace
python -u scripts/benchmark_dsv4_moe_decode.py --mode scan \
  --model-dir /path/to/DeepSeek-V4-Flash \
  --output results/scan-new --warmup 20 --iterations 100 --rounds 3 --trace
python scripts/analyze_dsv4_moe_trace.py results/scan-new
```

Use a new output directory for every invocation. `--tokens 8 16 32 64` and
`--distributions balanced skewed` are the scan defaults. `--trace` collects five
separate profiler dispatches per path **after** official measurements; those
samples never enter timing statistics. To collect only a selected configuration,
run another scan with e.g. `--tokens 32 --distributions balanced --trace`.

Required tested runtime: Python 3.12, torch 2.13.0+cpu, torchvision 0.28.0+cpu,
torch-tpu 0.1.1+release.2026.9.22.20261003154636, JAX/jaxlib 0.11.2,
libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261008082921, vLLM 0.29.0+empty
(source `98dff2a81d747d1dba01a47f939f48c3526d4206`). The repository's
`pyproject.toml` defines the full dependency set. The existing virtual environment
is `$HOME/vllm-torchtpu/.venv`; do not replace a running service's environment.
For an isolated environment with the compatible vLLM package already available:

```bash
python3.12 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install /path/to/torch_tpu-0.1.1+release.2026.9.22.20261003154636-cp312-cp312-manylinux_2_31_x86_64.whl
# Install the matching vLLM build, then this repository's pinned dependencies.
python -m pip install -e .
python -m pip check
```

The prepared host activation file also sets the configured topology:

```bash
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_TOPOLOGY=2x2x1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_SKIP_MDS_QUERY=true
export TPU_NAME=jyacttcu-dzvd
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=10.117.82.44
```

The host address is specific to this allocation. Preflight uses privileged
`fuser` without opening TPU devices and refuses occupied devices. A cooperative
file lock prevents two copies of this benchmark from competing. Neither service
state nor device permissions are changed by the benchmark. A separate job could
still claim an idle device after preflight; reserve the slice operationally.

For host-only verification (no TPU/JAX/Torch import):

```bash
python -m unittest discover -s tests -p test_benchmark_dsv4_moe_decode.py
ruff check scripts/benchmark_dsv4_moe_decode.py scripts/analyze_dsv4_moe_trace.py tests/test_benchmark_dsv4_moe_decode.py
```

## Workload and numerical contract

Target H=4096, I=2048, E=256, top-k=6. Each DP contributes 8/16/32/64
decode tokens, so global token counts are 64/128/256/512. Context length is not
an input. Smoke uses H=1024, I=512, E=64 with the same EP8/top-k and semantics.

The first non-hash layer is selected using `num_hash_layers` (3 in this model).
Only `layers.3.ffn.experts.*.{w1,w3,w2}.{weight,scale}` and
`layers.3.ffn.gate.bias` byte ranges are read from safetensors. Every selected
tensor's shape, dtype, offsets and SHA256 are recorded. Model configuration is
saved and incompatible scoring/shape/activation fields cause an explicit error.
Checkpoint MXFP4/UE8M0 weights are decoded with the repository helper into BF16,
then requantized with its block512 native FP4 quantizer. Both paths use those same
logical weights; grouped uses packed U8 layout and fused uses native FP4 layout.

Hidden states and logits are **synthetic**, even when weights are real. Hidden
states are unit-normal BF16. Logits are constructed to obtain near-uniform expert
loads or extreme rank-0 concentration while retaining the actual score bias and
routing formula. These are controlled stress workloads, not captured model
activations and not a reproduction of real-model performance. The actual routes
after BF16 rounding, per-expert counts, per-rank counts and seeds are saved.

Scoring is `sqrt(softplus(logits))`; score bias affects selection only; selected
scores are normalized and scaled by 1.5. Activation is the repository's
`silu_and_mul_with_clamp` (gate capped at 10, up clipped to [-10,10]). The CPU
routing oracle is cross-checked against the actual Torch `select_experts` adapter.

| Property | Grouped GMM | Fused EP v2 |
|---|---|---|
| Entry/exit | local BF16 hidden/logits → local BF16 output | same |
| Dispatch | BF16 hidden/logits all-gather | FP8 activations and I32 packed metadata all-gathers |
| Weights | packed FP4, block512 FP32 scales | native FP4, same values/scales |
| Activations | dynamic FP8, block512 | rowwise FP8 input/intermediate |
| Compute | two existing `gmm_v2` calls; BF16 accumulation/output | actual v2 Pallas kernel; FP32 accumulation |
| Combine | inverse permutation, BF16 route weights, BF16 reduce-scatter | FP8 remote result DMA, FP32 route weights/weighted sum, BF16 output |

Thus both use low-bit weights but are **different numerical paths**, not
interchangeable W4A8 implementations. The normal path calls `fused_moe_func`
directly between collectives; chunk pipeline and dense-expert fallback are
disabled. Permutation selection follows the actual repository threshold.
The v2 path calls `fused_ep_moe_v2`, with `sharded_plan=False`; it never calls the
adaptive implementation. Both exclude gate projection, shared expert, mHC, norm
and attention. HLO assertions fail if expected kernels/collectives are absent.

Correctness uses a separate dense FP32 expert reference over the common decoded
**requantized** weights; it does not measure checkpoint-to-block512 requantization
error. Both paths are checked independently on all three input banks, with
relative L2, maximum absolute error and every token's error saved. Gates are
the existing FP4+FP8-transport test limits: relative L2 <0.06 and worst-token
relative L2 <0.075. Failure aborts before official timing. No tolerance is fitted
to observed results.

## Timing and evidence

Compilation, loading, input generation and H2D transfer finish before timing.
Every call takes runtime operands, so weights/inputs are not embedded constants.
There is no jitted repetition loop. Every warmup and measured dispatch calls
`jax.block_until_ready` on its result. Each configuration has 20 warmups and 100
measurements per path per round, for three rounds; AB/BA/AB measurement order.
Different deterministic input banks are used between rounds. Timings include
Python/PJRT dispatch and completion overhead; short kernels can be dominated by
that overhead. Resident weights and input buffers are reused: this does not
model cache interference from other model layers or full-service scheduling.

`rounds.json` contains all raw milliseconds and per-round statistics;
`summary.json` / `results.json` contain pooled P50/P90/P99, mean, population
standard deviation and P50(grouped)/P50(fused) speedup. Every run also saves
source/environment/hardware/config/weight provenance, inputs, outputs,
per-token correctness, lowered and optimized HLO, and optional trace/xplane files.

`analyze_dsv4_moe_trace.py` extracts per-device XLA module/op event durations.
Profiler durations are separate evidence, not official timing samples. Nested
events and async spans cannot be summed into exclusive costs. In particular,
async-start-to-done is not pure network time; the fused custom call includes
compute, remote DMA and waiting. Attribute wins to communication overlap only
when detailed device trace evidence supports it; a smaller fused event alone
cannot separate GMM, rearrangement and transport contributions.

All reported latencies concern **one routed MoE layer**, not full-model TPOT or
service throughput.

The recorded smoke/scan source is archived separately as `measured-source.tar.gz`.
Its benchmark SHA256 is `05654a922c71ea4d2733e0588ac0dfee00500f523f8c4e64dd1e2f1c0561521c`.
The delivered script additionally rejects any stderr from an unsuccessful ownership
probe (including sudo requiring a password); this preflight-only hardening does
not change timed computation. The scan source and complete source manifest remain
in the evidence archive, rather than being relabeled as the later script.
