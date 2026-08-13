# Kimi-K3 batch-1 TPOT optimization log

## Goal and constraints

- Model: `moonshotai/Kimi-K3`, revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Target: batch-1 decode TPOT at or below 15 ms without speculative decoding.
- Hardware: one TPU v7x-32 slice (4 hosts, 8 logical TPU devices per host).
- Parallel baseline: tensor parallel 32 plus expert parallel 32.
- Compile bucket: exactly `compile_sizes=[1]`.
- Weights: vLLM dummy loader with compressed-tensors MXFP4 execution.
- Benchmark: ISL=1, OSL=1000, concurrency=1, ignore EOS, 2 warmups and
  5 measured requests unless stated otherwise.

## Reproduction commands

The local environment is `.venv`. Dependencies were installed through the
repository workflow (`../download.sh` plus the local project install).

Start a fresh four-node Ray cluster inside the active Slurm allocation:

```bash
srun --jobid="$SLURM_JOB_ID" --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '
set -e
RAY=/absolute/path/to/vllm-torchtpu-4/.venv/bin/ray
$RAY stop --force >/dev/null 2>&1 || true
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,4
export TORCH_TPU_TOPOLOGY=2,2,4,2
export TPU_ACCELERATOR_TYPE=tpu7x-32
export JAX_PLATFORMS=
if [[ "$(hostname)" == *w-001* ]]; then
  exec $RAY start --head --node-ip-address=HEAD_IP --port=6379 \
    --resources='"'"'{"TPU":8}'"'"' --disable-usage-stats --block
else
  sleep 5
  exec $RAY start --address=HEAD_IP:6379 \
    --resources='"'"'{"TPU":8}'"'"' --disable-usage-stats --block
fi'
```

Launch Kimi-K3 (the script defaults to dummy weights and compile size 1):

```bash
export RAY_ADDRESS=HEAD_IP:6379
export VLLM_CACHE_ROOT=/mnt/scratch/users/$USER/vllm-runtime-EXPERIMENT
export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/torch_compile_cache"
export PROF_DIR=/mnt/scratch/kimi-k3-b1-tep32-EXPERIMENT-profile
scripts/k3/launch_kimi_k3_tp32_ep.sh
```

Run the canonical benchmark:

```bash
OUTPUT_DIR=results/kimi-k3-b1-tep32-EXPERIMENT \
NUM_PROMPTS=5 NUM_WARMUPS=2 \
scripts/k3/bench_isl1_osl1k.sh 127.0.0.1
```

The launcher explicitly sets `TPU_ACCELERATOR_TYPE=tpu7x-32`. Slurm TPU VMs
do not provide the accelerator-type metadata endpoint, and without this value
the early environment override cannot select v7 TensorCore collective
thresholds reliably.

## Baseline

Baseline result:

- Output throughput: 55.73 tokens/s.
- Mean/median TPOT: 17.93 ms.
- P99 TPOT: 17.94 ms.
- Mean TTFT: 31.85 ms.
- Result file:
  `results/kimi-k3-b1-tep32-main-dbd3/kimi-k3_isl1_osl1000_concurrency1_20260812-183535.json`.

Relevant baseline changes already present before the experimental work:

- K3 attention input projection fusion (`23bd71f0`).
- K3 MXFP4 MoE decode tiling (`dbd3b7c8`).
- One-hot MoE permute/unpermute for the batch-1 routed-row count.
- Degenerate grouped-top-k elimination.

Baseline rank-0 XPlane:

`/mnt/scratch/kimi-k3-b1-tep32-main-dbd3-profile/plugins/profile/2026_08_12_18_40_32/rank0_tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal.xplane.pb`

Average self-time per decode step from 18 profiled model iterations:

| Kernel or family | Average self-time (ms) |
| --- | ---: |
| TP all-reduce | 4.412 |
| Dense matmul | 3.478 |
| Routed MoE GMM1 (`K=3584 -> N=6144`) | 1.066 |
| Routed MoE GMM2 (`K=3072 -> N=3584`) | 0.764 |
| Softmax | 0.474 |
| Top-k | 0.440 |
| Reshape | 0.384 |
| Sort | 0.324 |
| Mean | 0.315 |
| Dot | 0.209 |
| Gather | 0.166 |
| KDA custom op | 0.154 |
| Cat | 0.154 |
| Sum | 0.127 |
| Reduce-sum | 0.081 |
| MLA | 0.066 |

These named operations total about 12.6 ms. The remaining TPOT is smaller
operations, dependency/idle gaps, and launch/runtime overhead. The dominant
actionable categories are collectives and dense matmuls, not attention.

## Major issues discovered

### Ray workers can retain stale model code

Restarting only the API server can reuse existing Ray actor processes. A first
fused-collective test returned exactly the old 17.93 ms result because the
workers still had the pre-change Python module imported. For any model-code or
worker-environment change, cancel the Ray Slurm step, start a new cluster, and
verify new actor PIDs before treating a result as valid.

### Slurm TPU metadata lookup fails

The GCE metadata request returns HTTP 404 on these hosts. The launcher now
propagates `TPU_ACCELERATOR_TYPE=tpu7x-32` to every Ray actor. This resolves
the intended libtpu arguments, including the 64 MiB SparseCore all-reduce and
all-gather offload thresholds.

### Fresh whole-model compiles are expensive and noisy

A new batch-1 executable takes roughly 2-3 minutes for the final AOT phase and
substantially longer end-to-end because 32 fresh workers initialize TPU,
construct/load the model, build KV caches, and compile auxiliary functions.
Ray log forwarding can also emit millions of repeated libtpu layout lines.
Use a distinct cache namespace for correctness, but avoid cycling workers for
changes that provably do not affect imported model code or libtpu startup.

### The profiler configuration mentions `max_num_seqs=160`

The server itself is configured with `--max-num-seqs 1` and
`--max-num-batched-tokens 1`. The custom attention tuner still reports a
batched-decode key containing `max_num_seqs=160`; attention is only about
0.22 ms total (KDA plus MLA), so this warning is not the primary TPOT limit.

## Experiments and results

### 1. Exact K3 MoE tile sizes

Change: use exact K3 MXFP4 GMM dimensions rather than generic padded tiling
(`dbd3b7c8`). This was included in the measured 17.93 ms baseline. It improves
the isolated kernels but is insufficient by itself to reach the target.

### 2. Concatenate latent and shared partials into one TP all-reduce

Change: `a154b7e8` concatenates each layer's routed latent partial and shared
expert partial, performs one TP32 all-reduce, splits it, then applies latent
RMSNorm/up-projection and adds the shared output.

This commit is the exact patch from upstream draft PR
[`vllm-project/vllm-torchtpu#410`](https://github.com/vllm-project/vllm-torchtpu/pull/410):
the local commit has the same four changed files and 373 additions / 8
deletions as PR head `528624c0eb01c9c068db12a1eb0df1dd9240ac5f`.

Valid fresh-worker result:

- Mean TPOT: 17.92 ms (baseline 17.93 ms).
- P99 TPOT: 17.92 ms.
- Output throughput: 55.78 tokens/s.
- Mean TTFT: 28.62 ms.
- Result file:
  `results/kimi-k3-b1-tep32-fused-tc/kimi-k3_isl1_osl1000_concurrency1_20260812-192404.json`.

Profile after fusion:

| Kernel or family | Average self-time (ms) |
| --- | ---: |
| TP all-reduce | 4.572 |
| Dense matmul | 3.399 |
| Routed MoE GMM1 | 1.270 |
| Routed MoE GMM2 | 0.802 |

XPlane:

`/mnt/scratch/kimi-k3-b1-tep32-fused-tc-profile/plugins/profile/2026_08_12_19_30_19/rank0_tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal.xplane.pb`

Conclusion: reducing collective count did not reduce the critical path. The
larger concatenated layout increased the program bundle and GMM/layout cost;
the net result was flat. The launcher default is therefore disabled.

### 3. Replicate shared experts per host, TP8 within each host

Design:

- Keep routed experts at EP32/TP32.
- Replicate each shared MLP four times, once per host.
- Shard each copy across the eight contiguous TP ranks on that host.
- Replace the shared expert's global TP32 reduction with a host-local TP8
  reduction, then add its fully reduced output to the globally reduced routed
  output.

The implementation is gated by
`TPU_K3_HOST_REPLICATED_SHARED_EXPERTS`. Worker placement logs verified that
global TP ranks form contiguous blocks of eight per host. vLLM linear layers
store their overridden subgroup rank/size, so every host loads the same TP8
weight shards.

Result:

- Mean TPOT: 20.24 ms.
- Median TPOT: 20.23 ms.
- P99 TPOT: 20.26 ms.
- Output throughput: 49.38 tokens/s.
- Mean TTFT: 32.72 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hostrep/kimi-k3_isl1_osl1000_concurrency1_20260812-195752.json`.

Conclusion: the shared collective became cheaper, but each rank's shared MLP
weight shard became four times wider. The additional dense-matmul cost was
larger than the communication saving. The code remains opt-in for follow-up
TP16 or heterogeneous studies; the launcher default is disabled.

### 4. Early shared-expert all-reduce outside the generic MoE runner

Design: preserve TP32 weight shards, but reduce the shared MLP output at the
end of that branch and add it outside vLLM's generic MoE runner. This exposes
the shared collective while routed computation remains independent and may
permit compiler overlap.

Result:

- Mean TPOT: 17.97 ms.
- Median TPOT: 17.94 ms.
- P99 TPOT: 18.07 ms.
- Output throughput: 55.58 tokens/s.
- Mean TTFT: 42.17 ms (with higher request-to-request variance).
- Result file:
  `results/kimi-k3-b1-tep32-early-ar/kimi-k3_isl1_osl1000_concurrency1_20260812-201736.json`.

Conclusion: XLA did not find useful enough overlap to change steady-state
TPOT. The launcher default is disabled.

### 5. Hierarchical routed-expert EP16 x TP2

Design: assign expert ownership between the 16 physical chips and shard each
chip's expert block across its two tensor cores. This changes only routed MoE
weight geometry; the surrounding model remains TP32, and the required latent
output reduction remains global. The feature is gated by
`TPU_MOE_HIERARCHICAL_EP` and includes chip-topology validation plus EP weight
filter realignment.

Unit validation: `tests/layers/vllm/test_moe_hierarchical.py` passes all 12
tests.

Result:

- Mean TPOT: 17.04 ms (0.89 ms, or 5.0%, better than baseline).
- Median TPOT: 17.03 ms.
- P99 TPOT: 17.09 ms.
- Output throughput: 58.63 tokens/s.
- Mean TTFT: 30.20 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep/kimi-k3_isl1_osl1000_concurrency1_20260812-203812.json`.

Rank-0 profile over 50 decode iterations:

| Kernel or family | Baseline (ms) | EP16 x TP2 (ms) | Change (ms) |
| --- | ---: | ---: | ---: |
| TP all-reduce | 4.412 | 4.000 | -0.412 |
| Dense matmul | 3.478 | 3.477 | -0.001 |
| Routed MoE GMM1 | 1.066 | 0.803 | -0.263 |
| Routed MoE GMM2 | 0.764 | 0.590 | -0.174 |
| Softmax | 0.474 | 0.473 | -0.001 |
| Top-k | 0.440 | 0.439 | -0.001 |

XPlane:

`/mnt/scratch/kimi-k3-b1-tep32-hier-ep-profile/plugins/profile/2026_08_12_20_42_25/rank0_tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal.xplane.pb`

Conclusion: this is the first material endpoint improvement. The 0.85 ms
reduction in the four dominant changed kernel families closely accounts for
the measured 0.89 ms TPOT improvement. Dense layers and routing operations
are essentially unchanged. This is the current best configuration, 2.04 ms
above the target.

### 6. Wider hierarchical expert tensor sharding

The EP16 x TP2 result motivates testing EP8 x TP4 and, if the trend remains
favorable, EP4 x TP8. Each rank still stores the equivalent of two full
experts, so this changes GMM geometry and expert replication without
increasing per-rank routed-expert weight memory. Groups of up to eight
contiguous ranks remain within a host. The main risk is that excessively
narrow MXFP4 output shards stop utilizing the matrix unit efficiently.

EP8 x TP4 result:

- Mean/median TPOT: 17.18 ms.
- P99 TPOT: 17.19 ms.
- Output throughput: 58.15 tokens/s.
- Mean TTFT: 30.75 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-tp4/kimi-k3_isl1_osl1000_concurrency1_20260812-210239.json`.

The worker log explicitly verified `ep_size=8, tp_size=4`. The ownership
implementation is covered by 14 focused unit tests, including exact EP/TP
coordinate tiling and rejection of partial-chip groups.

Conclusion: TP4 is 0.14 ms slower than TP2. The narrower MXFP4 GMM shards
have crossed the useful geometry point, so EP4 x TP8 is not justified by the
data and will not be compiled. EP16 x TP2 remains the best geometry.

### 7. Force batch-1 collectives onto SparseCore

Change: on the best EP16 x TP2 geometry, override both SparseCore collective
offload thresholds from the v7x default of 64 MiB to one byte. This forces the
small per-layer all-reduces onto SparseCore and directly tests whether the
launcher's TensorCore policy is leaving collective latency on the table.

Result:

- Mean TPOT: 17.13 ms.
- Median TPOT: 17.06 ms.
- P99 TPOT: 17.40 ms.
- Output throughput: 58.34 tokens/s.
- Mean TTFT: 33.66 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-sc-ar/kimi-k3_isl1_osl1000_concurrency1_20260812-212027.json`.

Conclusion: forced SparseCore is 0.09 ms slower on mean TPOT and has visibly
higher tail variance. The existing 64 MiB thresholds, which keep these
latency-oriented collectives on TensorCore, remain the default.

### 8. Batch-1 direct routed-row combine

At batch 1, all 16 routed rows belong to the same token. The generic GMM path
sorts rows by expert, computes a second inverse argsort, builds a 16 x 16
one-hot unpermutation, and multiplies it by the GMM2 output. The specialized
path retains the first forward permutation, applies it to the routing weights,
and performs one FP32-accumulating weighted row reduction. This preserves the
same expert ordering and accumulation precision while eliminating the inverse
sort and unpermutation matmul.

Focused CPU/JAX validation: six relevant fused-MoE tests pass, including a new
test that verifies forward-sort weight alignment and proves the batch-1 path
does not call the generic ragged gather-reduce combine.

A broader CPU-forced run of `test_fused_moe_gmm.py` plus
`test_moe_hierarchical.py` completed with 36 passes and six expected
TPU-backend-only failures: those six explicitly call `_require_tpu()` and
reject the deliberately selected `JAX_PLATFORMS=cpu` backend before exercising
their bodies. The TPU devices were still owned by the live serving compile, so
they cannot be run concurrently; they remain part of final validation after
the serving actors stop.

Testing issue: the first focused pytest invocation attempted to initialize a
second JAX TPU client while the serving actors owned `/dev/vfio`, so all six
tests failed at backend initialization with `Device or resource busy` before
executing test bodies. Re-running with `JAX_PLATFORMS=cpu` produced six passes;
this was an environment collision, not a functional failure.

Result:

- Mean/median TPOT: 16.80 ms (0.24 ms better than EP16 x TP2 before this
  specialization, and 1.13 ms better than baseline).
- P99 TPOT: 16.84 ms.
- Output throughput: 59.46 tokens/s.
- Mean TTFT: 30.49 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-direct/kimi-k3_isl1_osl1000_concurrency1_20260812-214134.json`.
- Rank-0 XPlane:
  `/mnt/scratch/kimi-k3-b1-tep32-hier-ep-b1-direct-profile/plugins/profile/2026_08_12_21_45_32/rank0_tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal.xplane.pb`.

Conclusion: this is the new best result. Its 0.24 ms endpoint saving is
consistent with removing one of the two approximately 0.16 ms routed-row sort
operations plus the generic unpermutation/combine work.

### 9. Unsorted router top-k

Status: measured, neutral, reverted.

K3 selects 16 experts from 896 scores. The downstream GMM consumes aligned
weight/id pairs and sorts them by expert ID, so their incoming score order is
not observable. Both normal router branches now request `torch.topk(...,
sorted=False)`, allowing the compiler to omit output-order work if its TPU
lowering supports that distinction. Eleven routing tests pass.

Result:

- Mean/median TPOT: 16.79 ms.
- P99 TPOT: 16.80 ms.
- Output throughput: 59.52 tokens/s.
- Mean TTFT: 28.29 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-unsorted/kimi-k3_isl1_osl1000_concurrency1_20260812-220601.json`.

Conclusion: the 0.01 ms difference from the 16.80 ms direct-combine result is
below measurement resolution. The TPU lowering either ignores the `sorted`
hint or hides its work behind another operation, so the source change was
reverted. The verified best remains the simpler direct-combine configuration.

### 10. Asynchronous all-reduce scheduling

Status: measured, no improvement, rejected.

All-reduce is the largest remaining profiled family at approximately 4.00 ms
per decode step. A fresh Ray cluster and executable cache are being used to
test the libtpu scheduling flag below while holding EP16 x TP2 and the direct
batch-1 combine constant:

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false \
  --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false \
  --xla_enable_async_all_reduce=true"
```

This experiment targets overlap/scheduling only; it does not change the
collective payloads or MoE geometry. It must be judged by the full endpoint
benchmark because an asynchronous lowering can move time among profiler
events without reducing the request critical path.

Result:

- Mean TPOT: 16.89 ms.
- Median TPOT: 16.78 ms.
- P99 TPOT: 17.29 ms.
- Output throughput: 59.16 tokens/s.
- Mean TTFT: 30.57 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-async-ar/kimi-k3_isl1_osl1000_concurrency1_20260812-222427.json`.

Conclusion: asynchronous lowering did not reduce the endpoint critical path
and introduced materially more tail variation than the 16.80/16.84 ms best
run. Keep `xla_enable_async_all_reduce` at its default.

### 11. Replicate dense weights: TP16 x DP2 with EP32

Status: blocked by multihost-DP Ray session ownership; no performance result.

This layout uses vLLM's combined data/expert-parallel geometry rather than
reducing the total expert sharding width:

- Dense attention/MLP weights: TP16, replicated across two DP replicas.
- Routed expert weights: EP32 across both DP replicas, then the established
  hierarchical EP16 x TP2 subdivision inside fused MoE.
- Total workers: 32; no speculative decoding; compile size remains one.

The intended benefit is that dense TP all-reduces span 16 ranks / two hosts
instead of 32 ranks / four hosts. The costs are twice-wide dense matrix shards
and duplicated dense-weight memory. The launcher now accepts `TP_SIZE` and
`DP_SIZE`, retaining 32 and 1 as its defaults; this trial uses:

```bash
TP_SIZE=16 DP_SIZE=2 scripts/k3/launch_kimi_k3_tp32_ep.sh
```

First-attempt issue: vLLM defaulted `api_server_count` to the DP size (two).
Both engine cores began constructing their 16 Ray workers, but initialization
failed before model compilation. Cleanup reported that actor handles created
in Ray jobs `01`/`02` were being used from jobs `03`/`04`:

```text
ray.exceptions.ActorHandleNotFoundError: ActorHandle objects are not valid
across Ray sessions.
```

This is a control-plane failure, not a measured model result. The next attempt
pins `--api-server-count 1` to determine whether one frontend can coordinate
both DP engines without cross-session actor ownership. If the same failure
recurs, TP16 x DP2 is blocked by the current Ray V2 multihost-DP integration
and will not be assigned a TPOT value.

The single-API retry produced the same cross-session failure: the frontend
still launches two DP engine-core processes, each with an independent Ray job
session, and actor ownership crosses those sessions. Both attempts failed
before model construction/compilation, so this layout has no valid latency
measurement. Fixing the generic Ray V2 multihost-DP control plane is a
separate systems task and is not folded into the K3 performance patch.

### 12. Latency-bound all-reduce decomposition

Status: 1 MiB compile rejected as impractical; 32 KiB measured neutral.

The next isolated libtpu trial returns to TP32 / DP1 and the verified EP16 x
TP2 MoE geometry. It asks the compiler to apply its latency-bound all-reduce
decomposer to payloads up to 1 MiB:

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false \
  --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false \
  --xla_all_reduce_latency_bound_threshold_in_bytes=1048576"
```

The threshold covers K3's small batch-1 activation reductions while remaining
well below model-weight scale. It is tested independently of async scheduling
and SparseCore offload so its endpoint effect is attributable.

Compile result: the 1 MiB setting kept all 32 compiler processes near one CPU
core each for more than nine minutes in repeated layout assignment, over three
times the normal whole-model compile duration, without producing an
executable. It was terminated rather than consuming the allocation on a
non-deployable compiler configuration. This is not a runtime benchmark result.

The retry narrows the threshold to 32768 bytes. K3's primary batch-1 bf16
reductions are about 14 KiB (`[1,7168]`) and 21 KiB for the compiler-coalesced
shared/latent tuple (`[1,7168] + [1,3584]`), so 32 KiB still targets them while
excluding larger operations that caused the pathological compile expansion.

The 32 KiB main graph compiled in 139 seconds and produced:

- Mean/median TPOT: 16.81 ms.
- P99 TPOT: 16.82 ms.
- Output throughput: 59.45 tokens/s.
- Mean TTFT: 31.13 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-latency-ar-32k/kimi-k3_isl1_osl1000_concurrency1_20260812-231146.json`.

Conclusion: the 0.01 ms delta from the 16.80 ms best is neutral. Keep the
latency-bound threshold at its libtpu default.

### 13. Exact blocked K3 router top-k

Status: runtime experiment rejected because compilation was impractical.

For K3's top-16-of-896 route selection, split the experts into fourteen
64-wide blocks, take top 16 in every block, and take the final top 16 from the
224 candidates. This is exact: any item in the global top 16 must also be in
the top 16 of its own disjoint block. Other expert counts and top-k values use
the unchanged standard path.

The blocked form replaces one large selection with fourteen vectorized small
selections plus a 224-way final selection, exposing more parallel work to TPU.
It supports both correction-bias and plain routes because it operates only on
the tensor used for expert choice; original sigmoid weights are still gathered
by the selected global IDs. Three leading-rank shapes, the fallback path, and
the existing routing suites pass: 25 tests total.

Hardware result: the whole-model XLA compile remained in layout assignment
with all 32 workers consuming roughly one CPU core each for more than ten
minutes and never produced a serving executable. The normal best-path compile
finishes in roughly two to three minutes. The trial was terminated without a
TPOT result; the exact blocked selection is therefore not deployable in this
form and will not be enabled by default.

Profile clarification: the current compiler already coalesces each layer's
independent shared-output (`bf16[1,7168]`) and routed-latent
(`bf16[1,3584]`) reductions into one variadic all-reduce. This explains the
neutral manual-concatenation experiment: there is no remaining call-count
fusion opportunity at that boundary. Further collective work must improve
algorithm choice or scheduling.

### 14. Row-sharded latent-MoE up projection

Status: global TP32 shard measured neutral; physical-chip TP2 shard measured
as a clear regression. Focused correctness tests pass (10/10).

K3 currently replicates both latent projections. The output-side projection
maps the globally reduced, normalized 3584-wide latent back to hidden width
7168 on every TP rank. This work can be row-sharded across global TP32 without
adding a collective:

1. Keep the mandatory 3584-wide global all-reduce before latent RMSNorm.
2. Split the normalized latent contracting dimension and load the matching
   `up_proj` weight shard on each rank.
3. Produce one 7168-wide projection partial per rank.
4. Add that to the still-unreduced shared-expert partial.
5. Use the existing second 7168-wide all-reduce to finish both contributions.

The experiment uses vLLM's `RowParallelLinear(input_is_parallel=False,
reduce_results=False)` and a K3-specific MoE runner that defers the shared
reduction until after the row-parallel projection. The communication count and
payloads remain the same as the current two-reduction path; only the dense
projection changes from replicated to 32-way sharded. Preconditions restrict
the path to latent MoE with norm, a shared expert, TP > 1, DP1/PCP1, and routed
scale 1.0. The earlier explicit concatenated-combine path is disabled when
this mode is selected.

The first hardware version sharded across all 32 ranks, leaving a very skinny
`[1,112] x [112,7168]` local matmul. It compiled normally in 141 seconds and
measured:

- Mean/median TPOT: 16.77 ms.
- P99 TPOT: 16.79 ms.
- Output throughput: 59.57 tokens/s.
- Mean TTFT: 30.58 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-up/kimi-k3_isl1_osl1000_concurrency1_20260812-235434.json`.

The 0.03 ms gain versus the 16.80 ms best is effectively neutral: reducing
FLOPs did not compensate for the inefficient narrow matrix shape. The
follow-up sharded only across each physical chip's TP2 group, yielding a
`[1,1792] x [1792,7168]` local matmul. Sixteen chip groups compute identical
projection partials, so each is scaled by `2/32 = 1/16` before the existing
global hidden-width all-reduce. This remains exact and adds no collective.

That wider local matmul compiled and served correctly, but measured:

- Mean/median TPOT: 17.98 ms.
- P99 TPOT: 17.99 ms.
- Output throughput: 55.58 tokens/s.
- Mean TTFT: 32.14 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-up-tp2/kimi-k3_isl1_osl1000_concurrency1_20260813-001327.json`.

This is 1.18 ms slower than the 16.80 ms best and 1.21 ms slower than the
global TP32 shard. The result is useful: at batch one, reducing arithmetic by
using a much wider weight shard does not imply lower latency. The narrower
TP32 shard better exploits aggregate bandwidth and its otherwise inefficient
shape is still preferable to repeating each TP2 shard on sixteen chips.

One launch failed before model construction because the default TorchTPU
Tier-2 cache directory in `/dev/shm` was owned by another user. The misleading
cross-Ray-session actor errors were shutdown fallout after the worker's cache
lock abort. The launcher environment propagation now includes the two Tier-2
cache variables, and the successful retry uses an experiment-specific cache
namespace.

The input-side `down_proj` is the next separate experiment. Row sharding it
would reduce the replicated 7168x3584 matmul but requires an additional
3584-wide reduction before routed expert GMM1. Unlike the repeated-compute
TP2-up design, this can use a real two-rank physical-chip group, so each chip
computes the projection once and performs a chip-local reduction. It will be
tested independently on the 16.80 ms base, without TP2 up sharding enabled.

Implementation milestone (2026-08-13): the down-only path is now wired behind
`TPU_K3_SHARD_LATENT_DOWN_PROJ`. It derives all TP subgroups from the physical
chip topology, orders each group's ranks by topology TP rank, constructs a
TP2 `RowParallelLinear(input_is_parallel=False, reduce_results=False)`, and
performs the required subgroup all-reduce in a routed-input transform. The
checkpoint-visible `routed_expert_down_proj.weight` parameter path is preserved.
Four focused construction/collective tests pass. The clean third hardware
launch constructed, loaded, compiled, served, and benchmarked successfully.

Two hardware launches failed before model construction with `/dev/vfio/*`
device-busy errors. Investigation found the precise cause: a concurrent broad
`test_kimi_k3.py` run unexpectedly initialized libtpu in the head-node pytest
process (PID 607521), racing the Ray actors for TPU ownership. The focused
four-test selection is CPU-safe and passed; the full-file suite must not run
concurrently with hardware initialization. The pytest process was terminated,
the entire Ray step is being restarted, and the down-only launch will retry
with new isolated cache namespaces. These are infrastructure retries, not
down-projection results; later cross-session actor-handle messages were cleanup
fallout.

The canonical five-request result is a new best:

- Mean TPOT: 16.56 ms.
- Median TPOT: 16.55 ms.
- P99 TPOT: 16.62 ms.
- Output throughput: 60.35 tokens/s.
- Mean TTFT: 32.01 ms.
- Result file:
  `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-tp2/kimi-k3_isl1_osl1000_concurrency1_20260813-004558.json`.

The true TP2 down shard is 0.24 ms (1.4%) faster than the prior 16.80 ms best,
despite adding one 3584-wide chip-local all-reduce per latent MoE layer. This
confirms that the saved replicated weight read/matmul is worth the extra
on-chip communication. The K3 launcher now enables down sharding by default;
the regressed TP2 up path remains disabled.

## Other avenues investigated

- Sequence parallel branches were reviewed. At batch 1 they require token
  padding/replication that is unlikely to offset communication overhead.
- KDA convolution-weight load-time fusion exists on
  `origin/gnovack/kda-conv-weight-fusion`. Its maximum possible benefit is
  small because the entire KDA custom op is only about 0.15 ms per step.
- Topology-aware rank-assignment branches mainly target PCP or rank-binding
  correctness. The observed TP ranks already form host-contiguous blocks.
- TP16/DP2 was considered, but prior TP8 measurements and the dense-matmul
  profile make a direct path to 15 ms unlikely without additional changes.
- SparseCore collective offload thresholds remain a tunable libtpu axis. The
  clean baseline and pre-explicit-accelerator run were essentially identical,
  so merely falling back to libtpu defaults is not an observed win.

## Milestones

1. Reproducible local `.venv`, dummy-weight TP32/EP32 launch, and exact
   ISL1/OSL1000 benchmark scripts established.
2. Clean 17.93 ms baseline reproduced and saved.
3. Rank-0 XPlane converted to framework operation statistics; collective and
   dense-matmul bottlenecks quantified.
4. Stale Ray actor reuse identified and eliminated from subsequent results.
5. Correct v7x environment propagation added for every Ray worker.
6. Three different shared/latent collective strategies implemented and
   measured: fused, host-replicated, and early-reduced.
7. Hierarchical physical-chip-aware MoE geometry integrated, unit-tested, and
   measured at 17.04 ms, establishing the first material 5.0% improvement.
8. Profile comparison tied the gain to 0.44 ms faster routed GMMs and 0.41 ms
   less all-reduce time; the remaining distance to the goal is 2.04 ms.
9. Generalized the hierarchy to configurable whole-chip TP groups and measured
   EP8 x TP4 at 17.18 ms; this bounded the useful sharding width at TP2.
10. Forced the batch-1 collective stream onto SparseCore and measured 17.13
    ms; TensorCore remains slightly faster and substantially steadier.
11. Specialized the one-token routed-row combine and measured a new best of
    16.80 ms, for a cumulative 1.13 ms (6.3%) gain over baseline.
12. Tested unsorted router top-k at 16.79 ms; its 0.01 ms delta was neutral, so
    the source change was reverted.
13. Tested asynchronous all-reduce scheduling at 16.89 ms mean / 17.29 ms
    p99; it was slower and noisier, so the flag was rejected.
14. Attempted TP16 x DP2 / EP32 twice (default and single API server); both
    failed before compilation on cross-Ray-session actor handles, establishing
    a current backend limitation rather than a performance result.
15. The broad 1 MiB latency-bound all-reduce setting caused pathological
    compile expansion; a targeted 32 KiB retry compiled normally but measured
    a neutral 16.81 ms, so neither setting was retained.
16. Implemented exact blocked top-k, but its graph caused more than ten minutes
    of XLA layout-assignment work and never reached serving; rejected on
    compile deployability without a runtime number.
17. Designed, CPU-validated, and hardware-tested a latent `up_proj`
    row-sharding path that reuses the existing hidden-width all-reduce. Global
    TP32 was neutral at 16.77 ms; repeated TP2 shards regressed to 17.98 ms.
18. Implemented the independent TP2 latent `down_proj` experiment with a true
    physical-chip collective; it passed focused tests and established a new
    best of 16.56 ms, 0.24 ms faster than the previous best.
19. Sharded the 7168-to-128 replicated `f_a_proj` in all 69 KDA layers across
    physical-chip TP2. The added 128-element local reduction produced a small
    new best: 16.51 ms mean/median, 16.52 ms p99, 60.51 output tokens/s, and
    27.81 ms mean TTFT. Result:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-kda-fa/kimi-k3_isl1_osl1000_concurrency1_20260813-011307.json`.
    This is 0.05 ms faster than latent-down alone and 0.29 ms faster than the
    prior direct-combine best, so the launcher now enables it by default.
20. Tested the next attention-side sharding experiment: row-shard MLA's fused
    7168-to-2112 `q_a_proj + kv_a_proj_with_mqa` input projection across each
    physical-chip TP2 group, then add one 2112-element chip-local all-reduce
    per MLA layer. The opt-in implementation uses a merged-checkpoint-aware
    row-parallel loader so the public checkpoint names and fused shard layout
    remain unchanged. Static checks, hardware construction, and full XLA
    compilation passed. The canonical result was 16.52 ms mean, 16.51 ms
    median, 16.53 ms p99, 60.49 output tokens/s, and 28.24 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-kda-fa-mla-a/kimi-k3_isl1_osl1000_concurrency1_20260813-013736.json`.
    This is 0.01 ms slower than the 16.51 ms control: halving the replicated
    matmul is almost exactly canceled by the 2112-element local reduction.
    The path remains available for experimentation but disabled by default.
21. Tested a higher-leverage attention-linear experiment: TP2 row-shard each
    replicated 7168-to-1 attention-residual scorer (the self-attention and MLP
    scorer in every decoder layer, plus the final output scorer), then reduce
    only its scalar/vector score inside the physical chip. This has a much
    stronger saved-matmul-to-communication ratio than the positive
    7168-to-128 KDA shard and the neutral 7168-to-2112 MLA shard. The opt-in
    implementation preserves checkpoint-visible parameter paths and passed
    static checks, distributed dummy-weight construction, and full XLA
    compilation. The canonical result was 17.09 ms mean, 17.08 ms median,
    17.09 ms p99, 58.49 output tokens/s, and 29.67 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-kda-fa-attn-res/kimi-k3_isl1_osl1000_concurrency1_20260813-015856.json`.
    This regresses 0.58 ms from the 16.51 ms control. Although every reduction
    carries only a scalar score at batch 1, introducing roughly 179 separate
    chip-local collectives per decode step overwhelms the saved scorer matmul
    work. The path remains available for controlled experiments but is
    rejected and disabled by default.
22. Completed an independent TP4 latent `down_proj` trial. Routed expert GMMs
    remain on the winning EP16 x TP2 hierarchy; only the replicated
    7168-to-3584 input projection is row-sharded across two adjacent physical
    chips. This isolates a second 2x reduction in projection work from the
    previously measured TP4 routed-GMM slowdown. The configurable path uses
    topology-ordered whole-chip groups behind `TPU_K3_LATENT_DOWN_TP_SIZE`,
    and preserves TP2 behavior when the value is zero. Static checks,
    fresh-worker construction, and full compilation passed. The canonical
    result was a new best: 16.19 ms mean, 16.15 ms median, 16.32 ms p99,
    61.71 output tokens/s, and 35.31 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-tp4-kda-fa/kimi-k3_isl1_osl1000_concurrency1_20260813-022250.json`.
    This is 0.32 ms faster than the 16.51 ms TP2 projection control while
    leaving routed experts at TP2. The launcher now defaults the latent-down
    projection alone to TP4; TP8 is the next width-boundary experiment.
23. Completed the independent TP8 latent-down width-boundary trial with routed
    experts still fixed at EP16 x TP2. This halves each local projection shard
    again (relative to TP4) while widening the unchanged 3584-element
    reduction from four to eight ranks within one host. It uses the same
    validated implementation and a fresh compile/cache namespace. Construction
    and compilation passed, but the canonical result regressed to 16.63 ms
    mean, 16.62 ms median, 16.70 ms p99, 60.07 output tokens/s, and 28.68 ms
    mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-tp8-kda-fa/kimi-k3_isl1_osl1000_concurrency1_20260813-023852.json`.
    TP8 is 0.44 ms slower than TP4 and 0.12 ms slower than the former TP2
    control. The full-host collective costs more than the 1792-to-896 local
    contracting-width reduction saves. TP4 remains the launcher default.
24. Completed a stacked no-new-collective trial: retain independent TP4 latent
    `down_proj`, then globally TP32-row-shard latent `up_proj` and combine its
    partial through the already-required hidden-width all-reduce. Global up
    sharding was effectively neutral (16.77 versus 16.80 ms) on the older
    direct-combine stack, but could provide a small additive gain on the new
    TP4-down best. Construction and compilation passed, but the canonical
    result regressed to 17.04 ms mean, 17.01 ms median, 17.17 ms p99, 58.64
    output tokens/s, and 32.31 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-sharded-down-tp4-up32-kda-fa/kimi-k3_isl1_osl1000_concurrency1_20260813-025614.json`.
    This is 0.85 ms slower than TP4 down alone. The `[1,112] x [112,7168]`
    local projection shape is inefficient enough that eliminating replicated
    arithmetic does not help even when it adds no collective. The up path
    remains disabled by default.
25. Selected the next communication-free experiment: preserve each
    attention-residual scorer checkpoint weight but compute its 7168-to-1 dot
    as explicit elementwise multiply plus reduction, allowing XLA to fuse the
    tiny scorer rather than dispatching a dense matmul. This targets all 179
    scorers without adding collectives. The opt-in implementation retains the
    checkpoint-visible `ReplicatedLinear` and weight loader, but uses its
    weight in an explicit multiply/reduce during forward. The first hardware
    launch exposed an implementation wiring error: `self.fuse_dot` had been
    assigned to `KimiMLP` instead of `AttentionResidual`, producing an
    `AttributeError` during Dynamo capture. After moving the assignment to
    the correct module, fresh-worker construction and full XLA compilation
    passed. The canonical corrected result was 17.03 ms mean, 16.98 ms
    median, 17.16 ms p99, 58.68 output tokens/s, and 29.52 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-fused-score/kimi-k3_isl1_osl1000_concurrency1_20260813-032043.json`.
    This regresses 0.84 ms from the TP4-down best. XLA did not lower the
    explicit multiply/reduce more efficiently than the dense scorer path;
    it also expanded main-graph compilation to roughly 140 seconds on the
    reported ranks. The experiment is rejected and
    `TPU_K3_FUSE_ATTN_RES_DOT` remains disabled by default.
26. Captured the current TP4 latent-down best with two exact ISL1/OSL1000
    profiling windows. The profiler produced two compressed Torch traces for
    each of the eight ranks local to the API-server host under
    `/mnt/scratch/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-profile`.
    Rank 0 contains 18 decode executions. Its XLA-module lane averages
    17.664 ms per execution; the leaf XLA-ops lane accounts for 15.055 ms:

    | XLA-ops family | Average time per decode (ms) |
    | --- | ---: |
    | All-reduce | 5.135 |
    | Copies | 2.741 |
    | Dense dot/convolution | 2.081 |
    | Fused elementwise kernels | 1.983 |
    | Routed GMM/custom grouped matmul | 1.516 |
    | Sort | 0.569 |
    | Gather | 0.374 |
    | Other reductions | 0.172 |
    | Other custom calls | 0.081 |
    | All-gather | 0.031 |

    The breakdown confirms that collectives remain the largest target, but
    also shows 4.16 ms in dense and fused compute plus 2.74 ms of explicit
    copies. A tooling issue was found and worked around during analysis: the
    Torch trace writer emits invalid JSON when a top-level XLA event name
    contains unescaped HLO attribute quotes, for example
    `custom_call_target="X64SplitHigh"`. A streaming parser repaired only
    unescaped quotes in top-level event-name fields before aggregation; one
    unrelated malformed gRPC metadata event was skipped. Both captures are
    otherwise complete and agree on their execution count.
27. Generalized KDA full-rank `f_a_proj` sharding to an independent whole-chip
    width behind `TPU_K3_KDA_F_A_TP_SIZE`, then tested TP4 while retaining
    EP16 x TP2 routed GMMs and TP4 latent down-projection. Static Python and
    diff checks passed. An attempted full Kimi unit-module run made no test
    progress because importing the TPU platform inside the still-active
    distributed allocation spent over 60 seconds in TPU/Transformers startup;
    it was interrupted rather than risk acquiring devices needed by Ray.
    Validation therefore proceeded through a fresh Ray cluster: all 32 ranks
    constructed the TP4 projection group and full XLA compilation passed.
    The canonical result was 16.32 ms mean, 16.31 ms median, 16.34 ms p99,
    61.22 output tokens/s, and 30.20 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-tp4/kimi-k3_isl1_osl1000_concurrency1_20260813-040322.json`.
    This is 0.13 ms slower than the TP2 KDA control. For the 7168-to-128
    projection, extending the reduction across two physical chips costs more
    than halving the local contraction. The override remains available, but
    zero continues to select the winning hierarchical TP2 width by default.
28. Tested libtpu's recommended collective-matmul V2 lowering on the unchanged
    best graph with both
    `--xla_tpu_all_gather_collective_matmul_mode=post_spmd` and
    `--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd`. A fresh cache
    and all 32 fresh workers were used; construction and compilation passed,
    with the main graph taking about 139 seconds on a reported rank. The
    canonical result was 16.27 ms mean, 16.15 ms median, 16.65 ms p99, 61.42
    output tokens/s, and 28.30 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-cmatmul-v2/kimi-k3_isl1_osl1000_concurrency1_20260813-042016.json`.
    Median matches the 16.19 ms control's 16.15 ms, but mean regresses 0.08 ms
    and p99 regresses 0.33 ms. V2 is therefore neutral-to-noisier for this
    graph and is rejected as a default.
29. Enabled libtpu's enhanced launch barrier on the unchanged best graph with
    `--xla_tpu_use_enhanced_launch_barrier=true`. Fresh construction and full
    compilation passed. The first canonical result was 16.15 ms mean, 16.15
    ms median, 16.16 ms p99, 61.87 output tokens/s, and 30.96 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier/kimi-k3_isl1_osl1000_concurrency1_20260813-043734.json`.
    Because the 0.04 ms mean gain was near measurement resolution, the same
    compiled server was immediately re-run with two warmups and five measured
    requests. The repeat confirmed a new best at 16.141 ms mean, 16.137 ms
    median, 16.150 ms p99, 61.90 output tokens/s, and 29.51 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-044145.json`.
    The launcher now appends the flag by default while preserving an explicit
    caller-supplied true or false value.
30. Captured two enhanced-barrier profile windows in
    `/mnt/scratch/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-profile`
    and parsed the second rank-0 trace with the same repaired streaming parser
    used for the control. Over 18 decode executions, the XLA-module lane fell
    from 17.664 to 16.334 ms and leaf XLA-ops time fell from 15.055 to 14.947
    ms. All-reduce accounts for essentially the entire leaf-op gain, falling
    from 5.135 to 5.030 ms (-0.105 ms). Copies (2.741 ms), dense
    dot/convolution (2.082 ms), fused elementwise work (1.981 ms), and routed
    GMM (1.516 ms) are unchanged within 0.002 ms. The profile therefore
    validates the intended mechanism rather than attributing the endpoint
    result to compute noise.
31. Tested an equivalent-math KDA scheduling rewrite behind
    `TPU_K3_OVERLAP_KDA_F_A_REDUCE`: after producing the partial TP2 `f_a`,
    place the independent output-gate projection before consuming the
    128-element subgroup reduction. This was intended to expose a large
    projection-sized overlap window. Static checks, fresh-worker construction,
    and full XLA compilation passed, but the canonical result regressed to
    16.34 ms mean, 16.33 ms median, 16.39 ms p99, 61.17 output tokens/s, and
    28.16 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-overlap/kimi-k3_isl1_osl1000_concurrency1_20260813-050418.json`.
    XLA's resulting schedule is about 0.20 ms worse despite unchanged math;
    the opt-in path remains disabled.
32. Prepared a larger latent-MoE overlap experiment behind
    `TPU_K3_OVERLAP_LATENT_DOWN_ROUTER`. `KimiRoutedInputTransform` now exposes
    separate projection and reduction phases. In the opt-in forward path, the
    TP4 row-sharded 7168-to-3584 projection is produced first, the independent
    7168-to-896 router projection is computed next, and only then is the
    3584-element subgroup reduction consumed. The already-supported
    `shared_experts_input` argument lets `MoERunner` accept the pre-transformed
    routed latent while preserving the original 7168-wide input for shared
    experts, so checkpoint layout and math remain unchanged. Python, shell,
    and diff checks passed, as did fresh-worker construction and full XLA
    compilation. The canonical result was 16.13965 ms mean, 16.13646 ms
    median, 16.15948 ms p99, 61.91 output tokens/s, and 28.51 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-latent-router-overlap/kimi-k3_isl1_osl1000_concurrency1_20260813-052434.json`.
    This differs from the enhanced-barrier repeat control by only -0.00148 ms
    mean and -0.00031 ms median, far below resolution. XLA already schedules
    the independent router work adequately; the path is neutral and remains
    disabled.
33. Combined enhanced launch barriers with the otherwise deployable 32 KiB
    latency-bound all-reduce decomposition using
    `--xla_all_reduce_latency_bound_threshold_in_bytes=32768`. Fresh-worker
    construction and compilation passed, though compilation produced a very
    large volume of repeated layout-assignment diagnostics before completing.
    The canonical result was 16.22 ms mean, 16.17 ms median, 16.35 ms p99,
    61.61 output tokens/s, and 28.53 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-latency-ar-32k/kimi-k3_isl1_osl1000_concurrency1_20260813-054123.json`.
    This regresses 0.08 ms mean and 0.20 ms p99 from enhanced barriers alone.
    The compiler mechanisms interact negatively; decomposition stays at its
    libtpu default.
34. Began an isolated large-buffer memory-placement experiment on the
    unchanged enhanced-barrier best graph using
    `--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=0`.
    The flag is present in the installed libtpu and is already used by the
    repository's Qwen3-Coder TPU configuration. This experiment targets the
    2.741 ms of explicit copies in the current-best rank-0 profile. The first
    launch attempt stopped before worker construction because the completed
    32 KiB decomposition server still held port 8000 despite its original
    terminal session no longer appearing active. The stale frontend was
    identified exactly as PID 1465297, stopped with SIGINT, and the isolated
    experiment was relaunched with fresh compile caches. Compilation and the
    canonical benchmark are currently in progress; this flag is not a
    launcher default unless the result beats and repeats against 16.141 ms.

## Forward plan

1. Capture an equivalently formatted 16.80 ms direct-combine control if a
   strict paired attribution of TP4-down's dense savings and subgroup
   collective cost is needed; the current-best profile itself is parsed and
   establishes the absolute device-time budget above.
2. Parse the paired up-projection profiles to quantify why global TP32 is
   neutral while repeated TP2 is a regression, and use the result to choose
   attention projection shard widths.
3. Use the measured projection boundary to prioritize transformations that
   reuse or fuse an existing collective: 7168-to-128 KDA sharding is slightly
   positive, 7168-to-2112 MLA sharding is neutral, while 179 independent
   scalar attention-residual reductions are collectively expensive despite
   minimal payload. Avoid adding per-layer collectives solely for small
   replicated scorers.
4. Inventory the installed libtpu binary's exposed collective and launch
   flags, then select one isolated mechanism that targets the measured
   5.030 ms all-reduce budget without reintroducing rejected decomposition,
   SparseCore offload, async-all-reduce, or collective-matmul V2 settings.
5. Use the current-best profile to prioritize collective reuse/fusion or
   dense projections with enough work to beat the now-bounded subgroup cost;
   TP8 latent-down, per-layer scorer collectives, and explicit local scorer
   reductions are rejected.
6. Preserve the best known configuration as the launcher default and run a
   final clean five-request endpoint validation once TPOT reaches 15 ms.

## Current best and gap

The best verified steady-state result is 16.14 ms from hierarchical routed
experts at EP16 x TP2, batch-1 direct routed-row combine, independent TP4
latent `down_proj`, TP2 KDA `f_a_proj` sharding, and enhanced libtpu launch
barriers. The remaining gap to the 15 ms target is 1.14 ms (7.1% of the
current result). No
speculative decoding is used in any result.
