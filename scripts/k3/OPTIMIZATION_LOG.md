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
export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_async_collective_merger=true --xla_tpu_decompose_all_reduce_bidirectional_communication=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true'
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

These named operations total about 12.6 ms, but this is a top-operations list,
not a wall-time decomposition. In particular, it omits many copies and fused
elementwise kernels and therefore must not be subtracted from TPOT to infer
host overhead. The paired rank-0 Torch trace puts the model executable across
essentially the whole decode interval. Thus the apparent roughly 5.3 ms
remainder is predominantly TPU work omitted from this short XPlane table, not
5.3 ms of input preparation.

The later comprehensive trace historically labeled `enhanced-barrier` provides the clearest
reconciliation on the optimized graph:

| Scope | Typical per decode (ms) |
| --- | ---: |
| Top-level XLA model executable, median | 15.526 |
| Union of XLA + async device-lane activity | 15.459 |
| Interval not represented on those device lanes | about 0.067 |
| Greedy endpoint TPOT on the nearby temperature control | 15.796 |

The leaf total includes 5.030 ms all-reduce, 2.741 ms copies, 2.082 ms dense
dot/convolution, 1.981 ms fused elementwise work, 1.516 ms routed GMM, 0.569
ms sort, 0.374 ms gather, 0.172 ms other reductions, 0.081 ms other custom
calls, and 0.031 ms all-gather. Those displayed rounded families sum to
14.577 ms; the additional 0.370 ms in the parser's leaf aggregation is the
long tail of sub-threshold operations. This self-time-style family aggregation
is useful for ranking kernels but is not directly subtractable from module
wall time because async work and the two TPU-core lanes have different
accounting scopes.

An interval-level audit found that the earlier 16.334 ms arithmetic module
mean was distorted by profiler-boundary outliers. Sixteen of 18 rank-0 model
executions are 15.519--15.579 ms, while two are 18.617 and 26.225 ms. The
module median is 15.526 ms. Across both TPU cores, the union of XLA Ops and
Async XLA Ops intervals is approximately 15.459 ms on a normal execution,
leaving only about 0.067 ms that is not represented by a device activity lane.
The nearby greedy endpoint is approximately 0.27 ms longer than the median
model module. Separately compiled argmax, striding, and placeholder-substitution
modules account for about 0.02 ms in the trace; the balance includes
inter-module/runtime gaps, measurement/config differences between captures,
and any non-overlapped host work. These traces therefore show no multi-millisecond
outside-model or input-preparation gap. The dominant actionable categories
remain collectives, copies, and dense/fused compute.

The previously reported 1.3--1.4 ms "in-executable dependency/idle/long-tail"
was not a separately measured trace category. It was the residual from
subtracting the 14.947 ms average leaf-family aggregation from the 16.334 ms
arithmetic mean of top-level modules: 1.387 ms. That top-level mean includes
the two profiler-boundary outliers above, while the leaf aggregation has
different two-core and async self-time accounting, so the subtraction is not
valid. Using the normal-step module median reduces the numerical residual to
0.579 ms, but even that value must not be labeled idle. Of it, approximately
0.370 ms is explicitly identified as the long tail of sub-threshold leaf
operations. The direct interval union is the appropriate idle/bubble test and
leaves only about 0.067 ms without XLA or async device-lane activity. Any
remaining difference is trace-accounting scope and rounding, not evidence for
a 1.3 ms host wait or a single hidden kernel. In practical terms, the large
optimization targets are the operations already executing during that span,
especially the roughly 5.0 ms of collectives and 2.7 ms of copies.

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

    Correction from milestone 71: distributed TorchTPU rewrote this flag to
    `false` in the worker executable, as it did in later sessions. This was not
    a valid flag-on/flag-off experiment; the small endpoint change was session
    variance. The launcher no longer appends the flag.
30. Captured two enhanced-barrier profile windows in
    `/mnt/scratch/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-profile`
    and parsed the second rank-0 trace with the same repaired streaming parser
    used for the control. The raw arithmetic XLA-module means were 17.664 ms
    for the control and 16.334 ms for enhanced barriers, but a later interval
    audit found profiler-boundary outliers in those small samples; these means
    must not be used as an endpoint wall-time comparison. The stable leaf
    XLA-op aggregation fell from 15.055 to 14.947 ms. All-reduce accounts for
    essentially the entire leaf-op gain, falling
    from 5.135 to 5.030 ms (-0.105 ms). Copies (2.741 ms), dense
    dot/convolution (2.082 ms), fused elementwise work (1.981 ms), and routed
    GMM (1.516 ms) are unchanged within 0.002 ms. The profile therefore
    validates the intended mechanism rather than attributing the endpoint
    result to compute noise.

    Correction from milestone 71: because the allegedly enhanced workers had
    the barrier forced off, this profile comparison cannot validate enhanced
    barriers. The trace remains useful for operation and occupancy accounting,
    but its between-session collective delta must not be attributed to this
    flag.
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
34. Completed an isolated large-buffer memory-placement experiment on the
    unchanged enhanced-barrier best graph using
    `--xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=0`.
    The flag is present in the installed libtpu and is already used by the
    repository's Qwen3-Coder TPU configuration. This experiment targets the
    2.741 ms of explicit copies in the current-best rank-0 profile. The first
    launch attempt stopped before worker construction because the completed
    32 KiB decomposition server still held port 8000 despite its original
    terminal session no longer appearing active. The stale frontend was
    identified exactly as PID 1465297, stopped with SIGINT, and the isolated
    experiment was relaunched with fresh compile caches. All 32 workers
    constructed and the full graph compiled. The canonical result was
    16.15640 ms mean, 16.15304 ms median, 16.16570 ms p99, 61.85 output
    tokens/s, and 27.48 ms mean TTFT:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-altmem0/kimi-k3_isl1_osl1000_concurrency1_20260813-060612.json`.
    This is 0.01527 ms slower than the enhanced-barrier repeat best, within
    run noise but with no evidence of copy-path improvement. The memory flag
    is rejected and was not added to the launcher default.
35. Surveyed [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference)
    and its current optimization PRs for mechanisms transferable to this
    batch-1 Torch/XLA graph. The most directly relevant item is open Kimi-K3
    PR [`#3292`](https://github.com/vllm-project/tpu-inference/pull/3292), whose
    JAX-native design uses composed EP4 x TP8 routed-expert sharding and
    full-device KDA head sharding. Its PR explicitly calls throughput tuning
    follow-up work and reports no batch-1 TPOT. The EP4 x TP8 expert layout is
    not a promising direct transplant here: our paired EP16 x TP2 versus EP8
    x TP4 test already regressed 0.14 ms at TP4, so the documented geometry
    trend rejects widening the same GMM contraction to TP8. Full-device KDA
    head sharding primarily distributes independent heads and model memory;
    it is distinct from widening KDA's contracting-axis `f_a_proj`, where our
    TP4 test regressed 0.13 ms from TP2.

    Other findings were triaged against the exact workload:

    - Open PR [`#3237`](https://github.com/vllm-project/tpu-inference/pull/3237)
      removes per-step RNG splitting and redundant logits casts from the
      JAX-native sampling path. The TorchTPU runner already feeds native-dtype
      logits into a separately compiled `sample_from_logits`, skips all
      sampling transforms on its `all_greedy` branch, and only executes
      `gather_logprobs` when requested, so the patch is not directly portable.
      The remaining non-greedy path still creates a vocabulary-sized
      `torch.rand_like(logits)` outside that compiled function and is worth a
      trace-level check if the canonical client resolves to sampling rather
      than greedy mode.
    - Open PR [`#3363`](https://github.com/vllm-project/tpu-inference/pull/3363)
      moves greedy argmax before BF16-to-FP32 widening. The current TorchTPU
      `sample_from_logits` already performs argmax directly on its input
      dtype and returns token IDs only; it has no equivalent full-vocabulary
      FP32 return materialization to remove.
    - Low-concurrency MoE PR
      [`#3204`](https://github.com/vllm-project/tpu-inference/pull/3204)
      explicitly accepts only more than one token, so it excludes batch 1.
      Fused EP MoE PRs
      [`#3040`](https://github.com/vllm-project/tpu-inference/pull/3040) and
      [`#3288`](https://github.com/vllm-project/tpu-inference/pull/3288) target
      larger batches where collective traffic can overlap with useful GMM
      work; their published small-batch data are neutral or slower, and one
      implementation's 4096 hidden-width ceiling excludes K3's width 7168.
    - The repository's all-gather-matmul and matmul-reduce-scatter
      microbenchmarks currently benchmark XLA baselines at token counts 256,
      1024, and 8192. They provide useful harness structure but no batch-1
      evidence. The installed collective-matmul V2 lowering was already
      tested end to end here and rejected at 16.27 ms mean.
36. Resolved the benchmark/profile sampling mismatch. The canonical
    `bench_isl1_osl1k.sh` request leaves temperature unset, which the server
    resolves to temperature-1 sampling, while `curl_isl1_osl1k.sh` had
    explicitly requested greedy decoding. A paired temperature-0 run on the
    same alternate-memory server measured 15.79632 ms mean, 15.79296 ms
    median, 15.80356 ms p99, and 63.24 output tokens/s:
    `results/kimi-k3-b1-tep32-hier-ep-b1-down-tp4-kda-fa-enh-barrier-altmem0-greedy/kimi-k3_isl1_osl1000_concurrency1_20260813-061540.json`.
    That is 0.36008 ms faster than the paired sampled run. Sampled and greedy
    rank-0 traces localized 0.341 ms of the delta to the compiled sampler:
    0.35150 versus 0.01036 ms per token. The backbone was unchanged. The
    profiling helper now accepts `TEMPERATURE` while preserving zero as its
    default, and `INCLUDE_SEED=0` supports the server's working unseeded
    non-greedy path. Seeded non-greedy raw completion requests returned HTTP
    500 with an empty `InternalServerError`; unseeded requests and the
    canonical benchmark succeeded. This API correctness issue is separate
    from TPOT optimization.
37. Profiled the temperature-1 sampler and found the principal waste:
    `apply_top_k_top_p` unconditionally materialized both binary-search
    filters. Its runtime `where` guards did not prevent 32 top-k and 33 top-p
    full-vocabulary reductions even when every request used neutral defaults
    (`top_k=0`, `top_p=1`). Added a compiled specialization selected from the
    CPU-resident active request parameters that skips both filters only when
    all are neutral. The filtered and unfiltered non-greedy variants plus the
    greedy variant are precompiled; non-neutral top-k/top-p semantics are
    unchanged. Four pure decision cases were added to
    `tests/runner/test_tpu_runner.py`. The focused pytest was interrupted
    before collection because importing the TPU plugin attempted runtime
    initialization while the distributed allocation was active; static
    compile, shell, and diff checks passed, and all 32 fresh hardware workers
    constructed and compiled both sampler specializations. The canonical
    temperature-1 result improved to 15.98475 ms mean, 15.98058 ms median,
    16.00039 ms p99, 62.51 output tokens/s, and 28.20 ms mean TTFT:
    `results/kimi-k3-b1-sampling-neutral-fastpath/kimi-k3_isl1_osl1000_concurrency1_20260813-065157.json`.
    This improves the prior 16.14113 ms best by 0.15638 ms/token. The remaining
    roughly 0.18 ms sampled-versus-greedy delta is the vocabulary-sized random
    generation, Gumbel transforms, and final reduction rather than input prep.
38. Repeated the greedy control on the *same* neutral-filter-fast-path server,
    removing configuration and capture differences from the comparison. It
    measured 15.97893 ms mean, 15.95625 ms median, 16.05147 ms p99, 62.53
    output tokens/s, and 28.32 ms mean TTFT:
    `results/kimi-k3-b1-sampling-neutral-fastpath-greedy/kimi-k3_isl1_osl1000_concurrency1_20260813-070702.json`.
    The paired temperature-1 mean is only 0.00582 ms slower, well below run
    noise, so the earlier approximately 0.18 ms residual was not a stable
    sampling cost. New temperature-1 traces show about 0.14 ms from model
    return through the separately compiled sampler/striding/placeholder
    modules and their gaps, but that tail is also present in endpoint control
    accounting and does not explain the approximately 0.98 ms target gap. An
    inverse-CDF rewrite is therefore rejected for now: it has no measured
    latency budget to recover and would add numerical/correctness risk. Work
    returns to the in-model collective and copy critical path.
39. Inventoried the installed libtpu's collective scheduling controls and
    isolated `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true`
    on the unchanged best graph. Unlike the rejected Python async-all-reduce
    rewrite, this leaves model math and collective placement unchanged and
    lets XLA schedule already-dependent collectives on the executable's layer
    critical path. A new Ray cluster and fresh Tier-2, Tier-3, and XLA caches
    were used; all 32 workers constructed and the complete graph compiled.
    The first canonical temperature-1 run measured 15.91567 ms mean, 15.91622
    ms median, 15.92265 ms p99, 62.78 output tokens/s, and 29.07 ms mean TTFT:
    `results/kimi-k3-b1-dep-collective-scheduler/kimi-k3_isl1_osl1000_concurrency1_20260813-073429.json`.
    An immediate same-server repeat measured 15.92282 ms mean, 15.92112 ms
    median, 15.93465 ms p99, 62.75 output tokens/s, and 28.32 ms mean TTFT:
    `results/kimi-k3-b1-dep-collective-scheduler-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-073810.json`.
    Both runs improve on the prior 15.98475 ms result by approximately
    0.06--0.07 ms and have tight p99, so the guarded flag is now a launcher
    default. This is a real scheduling reduction inside the executable, not
    recovery of the invalidly inferred 1.3 ms idle residual.
40. Captured two temperature-1 profile windows for the new dependent-
    collective default and compared the second rank-0 trace with the matching
    pre-scheduler neutral-sampling trace using the same repaired parser and
    accounting scope. Median model-executable duration fell from 15.69323 to
    15.63341 ms, a 0.05982 ms reduction. On the primary XLA-op lane,
    all-reduce self-time fell from 5.20205 to 5.14697 ms (-0.05507 ms) and
    copies fell from 3.70740 to 3.70142 ms (-0.00599 ms). Dense work, fused
    elementwise work, routed GMMs, sort, and custom calls were unchanged to
    within approximately 0.001 ms. This reconciles essentially the entire
    endpoint gain with collective scheduling rather than host or idle time.
    The traces are under
    `/mnt/scratch/kimi-k3-b1-dep-collective-scheduler-profile`.

    Based on that attribution, tested the broader
    `--xla_tpu_enable_latency_hiding_layer_scheduler=true` in addition to the
    default dependent-collective scheduler, with a fresh Ray cluster and fresh
    compilation caches. The two canonical runs measured 15.91772/15.91831/
    15.93611 ms and 15.91197/15.90822/15.93072 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-latency-hiding-layer-scheduler/kimi-k3_isl1_osl1000_concurrency1_20260813-080442.json`
    and
    `results/kimi-k3-b1-latency-hiding-layer-scheduler-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-080823.json`.
    The first is 0.00205 ms slower than the best dependent-scheduler control;
    the repeat is 0.00370 ms faster. Both lie inside the existing run-to-run
    spread, so the full latency-hiding scheduler is neutral and is not added
    to launcher defaults.

41. Tested the proposed additional attention projection sharding on MLA's
    fused 7168-to-2112 query/KV A projection. The implementation now accepts
    an independent `TPU_K3_MLA_A_TP_SIZE`; zero retains the hierarchical TP
    group, while the experiment used TP4. This is deliberately narrower than
    globally changing attention TP: only the contracting dimension is sharded
    and the 2112-element partial is completed by one TP4 all-reduce. Fresh
    caches and all 32 workers compiled successfully. The two canonical runs
    measured 15.84994/15.85881/15.86342 ms and
    15.81726/15.81855/15.82880 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-mla-a-tp4/kimi-k3_isl1_osl1000_concurrency1_20260813-082618.json`
    and
    `results/kimi-k3-b1-mla-a-tp4-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-082847.json`.
    Their pairwise mean is 0.08564 ms faster than the two-run 15.91925 ms
    dependent-scheduler control. The TP4 variant is therefore enabled in the
    K3 launcher; the generic environment defaults remain off so other models
    are unchanged. This also refines the projection boundary: TP2 was neutral
    for this 2112-wide output, but TP4 produces enough matmul savings to pay
    for the wider subgroup collective.
42. Investigated the proposed NVIDIA MoE top-k implementations and built a
    TPU-native exact alternative. The Inkling Triton kernel fuses scoring,
    bias, iterative top-16 selection, and renormalization; the MLJ Triton
    kernel is a stable streaming selector that takes block-local candidates
    and bitonically merges a running top-16. Neither CUDA/Triton kernel is
    directly usable on TPU, but the current-best rank-0 trace proves this is a
    material target: 92 router sorts over `[1,896]` total 0.45020 ms/step,
    plus 92 downstream expert-ID sorts over `[16]` total 0.11923 ms/step.

    The first exact blocked Torch formulation (14 blocks of 64, 16 candidates
    each, then top-16 of 224) had already been rejected because whole-model
    layout assignment ran for more than ten minutes and never served. The new
    opt-in `TPU_K3_JAX_EXACT_TOPK=1` instead calls
    `jax.lax.approx_max_k(..., recall_target=1.0)` behind a Torch/JAX custom-op
    boundary. Despite its name, recall 1.0 is exact; the repository already
    uses this form in StreamIndex with a note that it is faster than
    `lax.top_k`. Focused routing tests passed 5/5. A fresh four-host Ray
    session was mandatory after the source change; an initial launch using
    worker PIDs from the preceding experiment was explicitly discarded.

    On fresh workers, the standalone selector compiled as a 72-instruction
    `tt_jit_custom_kernel` with only two extra layout passes. The first
    profiler backbone compile took about 51 seconds and produced a 51 MB
    artifact, versus roughly 142 seconds and 120 MB for the TP4-MLA control,
    because the 92 selections no longer expand inside the monolithic graph.
    The runtime backbone also completed despite verbose layout diagnostics.
    Endpoint performance, however, was neutral/slightly slower:
    15.86269/15.84342/15.94707 ms and
    15.83474/15.83138/15.85209 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-mla-a-tp4-jax-exact-topk/kimi-k3_isl1_osl1000_concurrency1_20260813-084908.json`
    and
    `results/kimi-k3-b1-mla-a-tp4-jax-exact-topk-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-085137.json`.
    Their mean is 0.01511 ms slower than the two-run TP4-MLA control, so the
    exact custom op remains disabled.

    A subsequent streaming parse of the rank-0 TPU trace corrected the first
    interpretation of what happened inside the custom boundary. The trace has
    19 decode executions and attributes 1,748 `[1,896]` sort events directly
    to `moe_topk_op.py:30`, where `approx_max_k` is called. Dividing the raw
    totals by 19 gives 92 router sorts and 0.45000 ms per decode (4.891 us
    mean per sort), plus 92 `[16]` expert-ID sorts and 0.11900 ms per decode
    (1.293 us mean per sort). These are effectively identical to the original
    0.45020/0.11923 ms families. In this exact shape, recall-1.0
    `approx_max_k` lowers to an ordinary sort; it does **not** implement a
    faster TPU selection algorithm. The smaller monolithic artifact and
    shorter reported backbone compile came from isolating those sorts behind
    the custom-op boundary, not from removing them at runtime.

    The result therefore shows that the 0.450 ms sort family is not freely
    recoverable with a standalone exact JAX selector. A future attempt should
    fuse score activation, correction bias, selection, gather, and
    renormalization in one TPU kernel (the transferable Inkling idea), or
    integrate routing directly with MoE dispatch; another standalone exact
    selector is low priority.
    Two post-warmup profiles were captured under
    `/mnt/scratch/kimi-k3-b1-mla-a-tp4-jax-exact-topk-fresh-profile`.
43. Captured a fresh, directly comparable profile of the accepted TP4-MLA
    graph with exact JAX top-k disabled. Two endpoint runs on this fresh
    session measured 15.96797 and 15.96015 ms mean TPOT. They are tightly
    paired but about 0.13 ms slower than the earlier 15.84994/15.81726 ms
    pair, so this is recorded as cross-session variance and does not replace
    the best verified result. The results are under
    `results/kimi-k3-b1-current-best-tp4-mla-trace{,-repeat}` and the two
    15-iteration profiles are under
    `/mnt/scratch/kimi-k3-b1-current-best-tp4-mla-trace-profile`.

    The second rank-0 capture contains 19 model executions. A streaming
    parser, needed because the trace again contains unescaped quotes in HLO
    names, gives a mean/median model executable of 15.68095/15.67916 ms and
    the following *additive primary XLA-op lane* breakdown:

    | XLA category | Mean per decode (ms) |
    | --- | ---: |
    | Convolution fusion (dense matmul lowering) | 5.38134 |
    | All-reduce | 5.05100 |
    | Custom calls (including routed GMM/custom kernels) | 1.75196 |
    | Loop fusion | 1.33258 |
    | Sort | 0.56909 |
    | Data formatting | 0.31681 |
    | Custom fusion | 0.25931 |
    | All remaining leaf categories | 0.36700 |
    | **Primary-lane total** | **15.02905** |

    The 5.05100 ms all-reduce row decomposes completely by payload and group:

    | Reduction family | Calls/decode | Mean per decode (ms) |
    | --- | ---: | ---: |
    | Variadic global TP32 `(bf16[1,7168], bf16[1,3584])` | 92 | 2.48760 |
    | Standalone global TP32 `bf16[1,7168]` | 95 | 1.70837 |
    | Latent-down TP4 `bf16[3584]` | 92 | 0.67390 |
    | KDA `f_a` TP2 `bf16[128]` | 69 | 0.18113 |
    | **Total** | **348** | **5.05100** |

    The single largest dense family is the replicated latent-MoE `up_proj`:
    its 92 `3584 -> 7168` convolution-fusion matmuls total about 2.141 ms per
    decode. This made an intermediate TP width worth testing, but only if its
    dataflow preserved the existing variadic collective fusion.

    Primary-lane interval occupancy is 15.02785 ms. Including asynchronous
    device-lane intervals covers 15.62018 ms of the 15.68095 ms executable,
    leaving only 0.06077 ms genuinely uncovered. Thus the approximately
    0.59 ms between the primary leaf lane and the model interval is
    asynchronous device work that fills primary-lane gaps, not host input
    preparation or idle time. The earlier 2.7--3.7 ms "copies" figures are
    asynchronous-copy occupancy/accounting and overlap other device work;
    they must not be added to this primary-lane table as serial latency.

    Applying the identical parser to the exact-JAX-top-k trace produced the
    same 184 sorts and 0.56900 ms per decode, versus 184 and 0.56909 ms here.
    Every non-collective category also agrees within 0.006 ms. Its different
    profile-window total comes from all-reduce variance, not the selector.
    This independently confirms that recall-1.0 `approx_max_k` merely moves
    the unchanged sort lowering behind a separate executable.
44. Tested the remaining MLA fused A-projection width boundary at TP8. This
    used the accepted graph unchanged except for
    `TPU_K3_MLA_A_TP_SIZE=8`, with a fresh cache namespace
    `/mnt/scratch/kimi-k3-b1-mla-a-tp8` on Ray step `602.46`. The launch
    compiled successfully on all 32 workers. The canonical runs measured
    16.06524/16.05379/16.10537 ms and
    16.04567/16.04575/16.04602 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-mla-a-tp8/kimi-k3_isl1_osl1000_concurrency1_20260813-094834.json`
    and
    `results/kimi-k3-b1-mla-a-tp8-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-095217.json`.
    Their 16.05546 ms pair average is 0.09140 ms slower than the fresh TP4
    control pair (15.96797/15.96015 ms). TP8 is therefore rejected and the
    launcher remains at TP4. Together with the neutral TP2 result, this pins
    the useful width boundary for the 7168-to-2112 fused projection at TP4:
    halving the remaining local matmul again does not pay for widening its
    2112-element all-reduce from four to eight ranks.

    Exact reproduction after exporting the standard Ray/cache variables from
    `RUNBOOK.md` is:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-mla-a-tp8
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_MLA_A_TP_SIZE=8 scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-mla-a-tp8 \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-mla-a-tp8-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    ```

    Caveats: the first launch used cold compile caches and spent several
    minutes in model construction plus the batch-1 backbone compile; this is
    startup cost and is excluded from TPOT. The first benchmark's first warmup
    took about 19 seconds while later 1,000-token requests took about 16.1
    seconds. Only the four post-warmup requests in each saved JSON determine
    the reported TPOT. The server was stopped after the repeat; Ray step
    `602.46` and parent allocation 602 were deliberately left active at that
    milestone. Milestone 45 later replaced only that Ray step.
45. Used the detailed kernel breakdown to test an untried intermediate shard
    for latent-MoE `up_proj`. The new opt-in
    `TPU_K3_LATENT_UP_TP_SIZE` selects a topology-valid width independently of
    routed-expert TP; zero preserves the old TP2 reproduction path. The trial
    used TP4, making each local up projection `896 -> 7168`, while retaining
    TP4 latent-down, TP2 KDA `f_a`, TP4 MLA A projection, and the accepted
    scheduler flags. It reuses the existing global hidden-width reduction and
    adds no logical payload. Three focused construction/runner tests pass.

    A fresh Ray step and fresh cache namespace
    `/mnt/scratch/kimi-k3-b1-latent-up-tp4` compiled and served on all 32
    workers. The two canonical runs measured
    16.13678/16.13159/16.15579 ms and
    16.13220/16.13095/16.14028 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-latent-up-tp4/kimi-k3_isl1_osl1000_concurrency1_20260813-101638.json`
    and
    `results/kimi-k3-b1-latent-up-tp4-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-102025.json`.
    Their 16.13449 ms pair average is 0.17043 ms slower than the fresh TP4
    control pair, so the feature remains opt-in and disabled in the launcher.

    The second rank-0 trace has 18 decode executions and explains why the
    apparently high-leverage dense shard loses. Convolution-fusion time falls
    from 5.38134 to 4.12883 ms (-1.25251 ms), but all-reduce grows from
    5.05100 to 6.52982 ms (+1.47882 ms). The unchanged logical payloads are
    lowered differently:

    | Reduction family | Control calls/ms | TP4-up calls/ms |
    | --- | ---: | ---: |
    | Variadic global `(7168,3584)` | 92 / 2.48760 | 0 / 0 |
    | Global `7168` | 95 / 1.70837 | 187 / 3.55016 |
    | Global `3584` | fused above | 92 / 2.20280 |
    | TP4 latent-down `3584` | 92 / 0.67390 | 92 / 0.58993 |
    | TP2 KDA `128` | 69 / 0.18113 | 69 / 0.18693 |
    | **Total** | **348 / 5.05100** | **440 / 6.52982** |

    Thus TP4-up does not add bytes, but its dependency change prevents XLA
    from variadically fusing 92 global 7168- and 3584-wide reductions. The 92
    extra launches cost more than the sharded dense projection saves. The
    complete primary-lane total rises from 15.02905 to 15.28393 ms, consistent
    with the endpoint regression. Other major families remain close:
    custom-call 1.73598 ms, loop fusion 1.30042 ms, and sort 0.56860 ms.
    Profiles are under
    `/mnt/scratch/kimi-k3-b1-latent-up-tp4-profile`.

    Reproduction after the standard `RUNBOOK.md` environment is:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-latent-up-tp4
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_SHARD_LATENT_UP_PROJ=1 TPU_K3_LATENT_UP_TP_SIZE=4 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-latent-up-tp4 \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-latent-up-tp4-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-kimi-k3-b1-latent-up-tp4 \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-latent-up-tp4-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The cold launch took about 224 seconds in runtime backbone compilation;
    this startup cost is excluded from TPOT. The first benchmark warmup again
    took about 19 seconds, versus about 16.15 seconds after warmup. Only the
    four post-warmup requests in each JSON are used above.
46. Tested whether changing the TP4 latent-up runner's ordering could recover
    the control graph's 92 variadic global reductions. The temporary runner
    first issued the latent `[1,3584]` and shared `[1,7168]` reductions next
    to each other, then applied latent RMSNorm and the TP4 row-sharded up
    projection, reduced only that routed `[1,7168]` partial, and finally added
    the already-reduced shared output. The focused runner/construction suite
    passed 3/3. Fresh Ray step `602.49` and cache namespace
    `/mnt/scratch/kimi-k3-b1-latent-up-tp4-early-shared` compiled and served
    successfully.

    The two canonical runs measured
    16.13022/16.13083/16.13275 ms and
    16.12713/16.12420/16.13744 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-latent-up-tp4-early-shared/kimi-k3_isl1_osl1000_concurrency1_20260813-105226.json`
    and
    `results/kimi-k3-b1-latent-up-tp4-early-shared-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-105445.json`.
    Their 16.12868 ms average is only 0.00581 ms faster than the original TP4
    latent-up trial and remains 0.16462 ms slower than the fresh default
    control pair. It is rejected.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-latent-up-tp4-early-shared-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2220250.1786618613922061502.pt.trace.json.gz`.
    Excluding the first profiler-boundary outlier, 18 model executions average
    15.84788 ms. The key result is structural: XLA still emitted 440 physical
    all-reduces per decode and no variadic collective. Global `[1,3584]`
    reductions cost 2.20643 ms, 187 global `[1,7168]` reductions cost about
    3.76618 ms, TP4 latent-down reductions cost 0.59001 ms, and TP2 KDA
    reductions cost 0.18749 ms, for 6.75011 ms total. Merely placing two
    Python all-reduce calls adjacent does not make XLA coalesce them.

    Historical reproduction, while the temporary runner was present, used:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-latent-up-tp4-early-shared
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_SHARD_LATENT_UP_PROJ=1 TPU_K3_LATENT_UP_TP_SIZE=4 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-latent-up-tp4-early-shared \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-latent-up-tp4-early-shared-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-kimi-k3-b1-latent-up-tp4-early-shared \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-latent-up-tp4-early-shared-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The temporary ordering code was removed after measurement; the command is
    retained for provenance, not as a currently selectable mode.
47. Tried to express the same pair explicitly with PyTorch's functional
    `all_reduce_coalesced([latent, shared], "sum", tp_group)` operation. The
    installed TorchTPU compiler recognizes the functional op name in its
    collective graph handling, and the focused suite again passed 3/3, but a
    fresh hardware launch on Ray step `602.50` failed during backbone FX-to-
    MLIR export before compilation or serving. Every worker reported:

    ```text
    RuntimeError: Backend tpu_dist does not support allreduce_coalesced
    While executing %all_reduce_coalesced = call_function[
      target=torch.ops._c10d_functional.all_reduce_coalesced.default](
        args=([%fused_moe_kernel_0, %mm_18], sum, 3), kwargs={})
    ```

    The FX interpreter dispatches the operation through `ProcessGroupTPU`
    before the compiler can lower it, so compiler-side recognition alone is
    insufficient. The failed launch used fresh namespace
    `/mnt/scratch/kimi-k3-b1-latent-up-tp4-coalesced` and the otherwise normal
    command:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-latent-up-tp4-coalesced
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_SHARD_LATENT_UP_PROJ=1 TPU_K3_LATENT_UP_TP_SIZE=4 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh
    ```

    No endpoint result exists because the server never became healthy. The
    unsupported source path was removed. A real retry requires implementing
    `allreduce_coalesced` in the TorchTPU process group or adding a supported
    purpose-built variadic collective custom op; it is not a model-only edit.
48. Revisited current `vllm-project/tpu-inference` specifically for a backend
    workaround to milestone 47. Repository code search finds no
    `all_reduce_coalesced` or `variadic` implementation. The most relevant
    current work is merged PR
    [`#2849`](https://github.com/vllm-project/tpu-inference/pull/2849), which
    defers both shared-expert and routed-expert reductions, adds the same-shape
    local outputs, and issues one ordinary `jax.lax.psum`; release correctness
    fix [`#3155`](https://github.com/vllm-project/tpu-inference/pull/3155)
    wires the defer flag through the unquantized linear path. This is a useful
    confirmation of the preferred pattern when outputs have the same shape,
    but it does not solve K3 latent MoE: the `[1,3584]` routed latent must be
    globally reduced before RMSNorm/up-projection, while the shared output is
    `[1,7168]`. They cannot be locally added before reduction. This makes
    backend support for a mixed-shape variadic reduction, or a more deeply
    fused latent-MoE kernel, the necessary next boundary.
49. Implemented the transferable part of the proposed NVIDIA MoE router
    designs as a real TPU Mosaic/Pallas kernel. The relevant references are
    the fused iterative selector in
    [`vllm/models/inkling/nvidia/moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/inkling/nvidia/moe.py#L81)
    and the streaming top-k merge in
    [`gate_topk.py`](https://github.com/Inferact/mlj/blob/7219aae117ae9a174b102acc7f39e3c4c0269e7f/vllm/models/tml/nvidia/ops/gate_topk.py).
    The new opt-in boundary initially performed sigmoid, correction-bias add,
    16 exact argmax/remove passes over 896 experts, selected-score gather, and
    renormalization in one kernel. It is deliberately restricted to the exact
    K3 decode case: logits `[1,896]`, top-k 16, sigmoid scoring, correction
    bias, renormalization, and routed scaling factor 1.0.

    The first TPU attempt on fresh Ray step `602.52` failed during FX-to-MLIR
    compilation. CPU Pallas interpretation had matched the JAX reference
    exactly, but Mosaic rejected the dynamic gather across the seven
    128-column source vector registers:

    ```text
    Not implemented: Multiple source vregs along gather dimension
    %190 = "tpu.dynamic_gather"(...) :
      (vector<8x896xf32>, vector<8x1xi32>) -> vector<8x1xf32>
    ```

    Replacing that gather with an equivalent one-hot masked reduction kept all
    score access statically tiled. Three random-seed interpreter tests still
    matched exact expert IDs and normalized weights bit-for-bit. Fresh Ray
    step `602.53`, cache namespace
    `/mnt/scratch/kimi-k3-b1-pallas-router-topk-mask`, and the masked kernel
    then compiled and served successfully on all 32 chips. The profile
    backbone compiled in 46.65 seconds to a 42 MB artifact; the runtime
    backbone compile took 133.92 seconds, and the full model compile stage
    took 215.15 seconds. These are startup costs, not TPOT.

    The two canonical runs measured
    15.44748/15.42422/15.52063 ms and
    15.42870/15.43001/15.43442 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-pallas-router-topk-mask/kimi-k3_isl1_osl1000_concurrency1_20260813-114456.json`
    and
    `results/kimi-k3-b1-pallas-router-topk-mask-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-114722.json`.
    The 15.43809 ms pair average is 0.39551 ms faster than the former best
    15.83360 ms pair and 0.52597 ms faster than the fresh 15.96406 ms control
    pair. This is the new best, so `TPU_K3_PALLAS_ROUTER_TOPK=1` is now the K3
    launcher default; the generic environment default remains false.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-pallas-router-topk-mask-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2411951.1786621907237364489.pt.trace.json.gz`.
    Excluding the first profiler-boundary outlier, 18 model executions average
    15.13673 ms (15.13633 ms median). The additive primary XLA-op lane is:

    | XLA category | Mean per decode (ms) |
    | --- | ---: |
    | All fusion operations | 6.84388 |
    | All-reduce | 5.14840 |
    | Other custom calls | 1.69143 |
    | Fused K3 router Pallas calls | 0.27328 |
    | Data formatting | 0.38780 |
    | Remaining expert-ID sorts | 0.15532 |
    | All other leaf operations | 0.22719 |
    | **Primary-lane total** | **14.72730** |

    There are 92 fused router calls per decode and **zero** remaining
    `[1,896]` router sorts; the prior graph had 92 such sorts costing about
    0.450 ms. The remaining 92 sorts are only the downstream 16-route
    expert-order permutation used to group rows for GMM. Primary-lane interval
    occupancy is 14.72613 ms. Its union with asynchronous device work is
    15.07414 ms, leaving 0.06259 ms uncovered inside the 15.13673 ms model
    executable. Thus there is again no 1.3 ms in-executable idle or host-input
    bubble: nearly the whole interval is accounted for device work.

    Exact reproduction after starting fresh Ray workers and exporting the
    standard runbook cache variables is:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-pallas-router-topk-mask
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_PALLAS_ROUTER_TOPK=1 scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-pallas-router-topk-mask \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-pallas-router-topk-mask-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-kimi-k3-b1-pallas-router-topk-mask \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-pallas-router-topk-mask-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    Caveats: real logits are finite and random finite tests match the current
    exact reference bit-for-bit. Exact ties follow repeated `argmax`'s
    first-index rule; `top_k` does not specify a portable tie order. Dummy
    weights make endpoint quality checks meaningless, so correctness evidence
    is the exact interpreter comparison plus successful full-model execution.
    The kernel keeps eight padded token rows to satisfy TPU tiling; only row
    zero is returned. Do not restore the cross-vreg dynamic gather without a
    Mosaic backend change.
50. Removed the second and final TPU sort family with an exact fixed-size
    rank network. After the fused Pallas router, every batch-1 routed MoE
    layer still called a general stable `argsort` on its 16 selected expert
    IDs to group rows for GMM. The new `_rank16_argsort` computes each item's
    stable rank from all 16x16 `(expert_id, original_position)` comparisons,
    then inverts those unique ranks with a one-hot reduction. It is enabled
    only for the exact batch-1/top-16 path; other token counts and top-k values
    retain the general `jnp.argsort`. Duplicate-ID tests include the EP-local
    invalid-route sentinel and match NumPy's stable reference exactly.

    Fresh Ray step `602.54`, cache namespace
    `/mnt/scratch/kimi-k3-b1-rank16-route-permute`, and
    `TPU_K3_RANK16_ROUTE_PERMUTE=1` compiled and served on all 32 chips. The
    profile backbone compiled in 47.56 seconds, its full AOT stage took 125.08
    seconds, and the runtime backbone compiled in 134.12 seconds. These are
    startup costs, not steady-state latency.

    The two canonical endpoint runs measured
    15.27481/15.27821/15.28103 ms and
    15.27295/15.27367/15.27611 ms mean/median/p99 TPOT:
    `results/kimi-k3-b1-rank16-route-permute/kimi-k3_isl1_osl1000_concurrency1_20260813-122123.json`
    and
    `results/kimi-k3-b1-rank16-route-permute-repeat/kimi-k3_isl1_osl1000_concurrency1_20260813-122503.json`.
    Their 15.27388 ms pair average is 0.16421 ms faster than the fused-router
    baseline pair and 0.55972 ms faster than the former 15.83360 ms best. This
    is accepted, so `TPU_K3_RANK16_ROUTE_PERMUTE=1` is now the K3 launcher
    default; the generic environment default remains false.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-rank16-route-permute-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2503254.1786624163924546668.pt.trace.json.gz`.
    Excluding one profiler-boundary outlier, 18 model executables average
    14.98202 ms (14.98424 ms median, 14.96134--14.99963 ms). A paired parse
    with the same operation-family rules as milestone 49 gives:

    | XLA category | Mean per decode (ms) |
    | --- | ---: |
    | All fusion operations | 6.86487 |
    | All-reduce | 5.09829 |
    | Other custom calls | 1.72813 |
    | Fused K3 router Pallas calls | 0.27816 |
    | Data formatting | 0.39019 |
    | Sort | **0.00000** |
    | All other leaf operations | 0.20665 |
    | **Primary-lane total** | **14.56628** |

    The trace contains 92 fused router calls and zero sort operations of any
    shape per decode. The fixed-rank work raised fusion/custom/router time by
    about 0.063 ms while removing 0.15532 ms of sort and coinciding with about
    0.071 ms less other/collective work; net primary-lane time fell about
    0.161 ms. Stable primary-lane occupancy is 14.56567 ms. Its union with
    asynchronous device work is 14.91656 ms, leaving only 0.06545 ms uncovered
    inside the 14.98202 ms model executable. As in the previous profile, there
    is no 1.3 ms in-executable idle or host-input interval.

    Exact launch, benchmark, and profile commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-rank16-route-permute
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_RANK16_ROUTE_PERMUTE=1 scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-rank16-route-permute \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-rank16-route-permute-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-kimi-k3-b1-rank16-route-permute \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-rank16-route-permute-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    Caveats: the optimization relies on exactly 16 route entries and stable
    integer-key ordering. It does not replace the 896-way router selector,
    whose exact tie semantics remain in the fused Pallas kernel. Dummy weights
    make output-quality checks meaningless; correctness evidence is stable
    duplicate/sentinel unit coverage plus successful full-model execution.
    The profile's single 15.94352 ms executable is profiler-boundary overhead
    and is excluded from the stable 18-execution statistic. Raw benchmark and
    profile-response directories remain intentionally untracked. Before the
    distributed launch, focused CPU tests passed (three tests, 27 deselected).
    A later repeat while the server was live was interrupted during plugin
    import because this environment initializes the TPU platform even with
    `JAX_PLATFORMS=cpu`; it did not reach a test failure, and server health
    remained HTTP 200. Do not run that import-heavy test again against a live
    distributed server.
51. Tested an MLJ-inspired two-stage Pallas router selector and rejected it as
    a default. The relevant NVIDIA reference is
    `Inferact/mlj@7219aae117ae9a174b102acc7f39e3c4c0269e7f`,
    `vllm/models/tml/nvidia/ops/gate_topk.py`: it streams 32-entry blocks,
    keeps a local top 16, then bitonically merges candidates. The Inkling
    reference in `vllm/models/inkling/nvidia/moe.py` instead uses the same
    iterative full-width selection structure as the already accepted K3
    Pallas kernel, so it did not expose a distinct algorithm to port.

    The TPU experiment uses seven native 128-column blocks because Mosaic's
    vector layout and the 896-expert shape naturally map to `7 x 128`. It
    computes each block's exact local top 16 in parallel, flattens the 112
    candidates in global expert-index order, and performs an exact global top
    16 pass. This preserves first-index tie behavior and avoids the
    cross-vector-register dynamic gather that failed in milestone 49. The
    implementation is retained behind
    `TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=1`; both its generic and K3 launcher
    defaults are zero.

    The focused interpreter test command was:

    ```bash
    JAX_PLATFORMS=cpu TPU_ACCELERATOR_TYPE= \
      .venv/bin/pytest -q tests/kernels/test_kimi_k3_router_topk.py
    ```

    All six cases passed in 7.31 seconds: three random seeds for both the
    accepted full-width and experimental blockwise implementations. An
    earlier invocation omitted `JAX_PLATFORMS=cpu`, initialized the TPU/VFIO
    plugin, and was terminated without reaching a test result. Always force
    the CPU platform for this interpreter-only test, and never run the
    accidental form while a distributed server is starting.

    Fresh Ray step `602.55`, cache namespace
    `/mnt/scratch/kimi-k3-b1-pallas-block-router-topk`, and the explicit
    blockwise flag compiled and served successfully on all 32 chips. The
    profile backbone compiled in 47.65 seconds, its full AOT stage took 126.05
    seconds, and the runtime backbone compiled in 134.77 seconds. Exact launch
    and measurement commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-pallas-block-router-topk
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/kimi-k3-b1-pallas-block-router-topk \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/kimi-k3-b1-pallas-block-router-topk-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-kimi-k3-b1-pallas-block-router-topk \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-pallas-block-router-topk-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The endpoint runs measured 15.24950 and 15.25911 ms mean TPOT, averaging
    15.25431 ms. That is nominally 0.01958 ms faster than the accepted pair,
    but the paired kernel evidence rejects a causal router improvement. The
    latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-pallas-block-router-topk-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2599376.1786626871906653164.pt.trace.json.gz`.
    Excluding one 21.50944 ms profiler-boundary outlier, 18 model executables
    average 14.97524 ms (14.97548 ms median). Primary-lane occupancy is
    14.55725 ms; its union with asynchronous device work is 14.91150 ms,
    leaving only 0.06375 ms uncovered. There is no host-input or executable
    idle bubble to explain the endpoint movement.

    More importantly, 92 block-router calls cost 0.30741 ms/decode, 0.02925 ms
    more than the accepted full-width router's 0.27816 ms. Stable model time
    improved by only 0.00678 ms, within run-to-run variation, while all-reduce
    time moved independently. The MLJ-style decomposition is therefore a
    measured router regression on TPU, despite the favorable raw endpoint
    sample. TPU lacks the reference kernel's cheap packed-key bitonic merge;
    repeated local reductions plus a second 112-way selection add work. Do
    not promote this flag without a new primitive or a direct trace showing
    the router family itself below 0.27816 ms. A later handoff-time repeat of
    the documented CPU-only pytest command did not emit a test result: two
    pytest instances remained in import/initialization for more than a minute
    at roughly 1 GB RSS each even though no distributed server was live. Only
    those exact PIDs were terminated. This is an environment/plugin hang, not
    a failed assertion, and does not replace the earlier six-case pass or the
    successful full-model executions. Do not start parallel repeats; first
    diagnose JAX plugin discovery in a standalone process if another unit-test
    rerun is needed.
52. Reconstructed the exact routed-GMM breakdown and swept the current
    hierarchical 56-group geometry. The accepted rank-16 trace records 1,729
    timed calls for each of router, GMM1, and GMM2. This is not 91 calls across
    19 equivalent decodes: the window contains 18 complete 92-layer MoE
    decodes plus a 73-layer profiler-boundary fragment (`18 * 92 + 73 =
    1729`). Dividing by 19 undercounts these families. Using mean call time
    times 92 gives the complete-decode breakdown:

    | Routed kernel | Exact traced geometry | Mean per complete decode |
    | --- | --- | ---: |
    | Router | fused exact top-16 | 0.27849 ms |
    | GMM1 gate/up | `g=56,m=16,k=3584,n=3072,tn=1536` | 0.79947 ms |
    | GMM2 down | `g=56,m=16,k=1536,n=3584,tn=3584` | 0.59325 ms |

    Thus the two routed GMMs account for about 1.39272 ms/decode. GMM1 is the
    larger surface, while GMM2 already uses one full-width N tile. These are
    `device_duration_ps` custom-kernel times, not an additive endpoint/host
    remainder; the primary-lane interval-union accounting in milestone 50 is
    still authoritative for wall time.

    Before the sweep, the rejected block-router server was stopped cleanly
    and only Ray step `602.55` was canceled. Allocation job 602 remained
    active. A first one-host microbenchmark attempt, temporary step `602.56`,
    was canceled after JAX correctly warned that TPU backend initialization
    required all hosts of the v7x-32 slice. The valid harness starts one
    coordinated process on each of the four hosts:

    ```bash
    srun --jobid=602 --overlap -N4 -n4 --ntasks-per-node=1 \
      --unbuffered --export=ALL bash -lc '
        cd /mnt/lustre/homes-x86/woosuk/workspace/vllm-torchtpu-4
        export PATH="$PWD/.venv/bin:$PATH"
        export TPU_ACCELERATOR_TYPE=tpu7x-32
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_mxfp4_gmm.py \
          --weight-experts 56 --active-experts 1 --iterations 100 \
          --gmm1-tile 16,3584,1024 --gmm2-tile 16,3072,1792
      '
    ```

    Unlike the older 28-group EP32 sweep, `--weight-experts 56` reproduces the
    current EP16 x TP2 per-rank weight geometry. The authoritative metric is
    the five-sample dependent `pair_device_loop_us`, not separately queued
    GMM timings, which contain a large host-dispatch floor. Four-host medians
    for current `GMM1 tn=1536/GMM2 tn=3584` versus the best low-occupancy
    `GMM1 tn=1024/GMM2 tn=1792` candidate were:

    | Active local routes | Current pair | Split candidate | Change |
    | ---: | ---: | ---: | ---: |
    | 1 | 21.362 us | 20.526 us | -0.837 us |
    | 2 | 31.341 us | 31.019 us | -0.322 us |
    | 4 | 50.840 us | 52.377 us | +1.537 us |
    | 8 | 89.123 us | 94.865 us | +5.742 us |
    | 16 | 165.659 us | 179.747 us | +14.088 us |

    Intermediate GMM1-only candidates at one active route bounded the tuning
    surface: `tn=1280` was about 23.0 us/pair, `tn=768` about 21.4 us, and
    `tn=1024` with full-width GMM2 about 21.2 us, versus about 21.4 us for an
    explicit current-tile control in the same warm session. The GMM1-only
    opportunity is therefore only roughly 0.2 us/layer (about 0.02 ms/decode)
    before occupancy risk and run variance. Splitting GMM2 creates a larger
    one-route win but crosses over by four routes and regresses the legal
    16-route case by 8.5%.

    Conclusion: do not compile the combined split in the full model and do
    not change production tiles. Dynamic top-16 routing provides no static
    guarantee of at most two local routes, and an input-dependent tile choice
    would require the same costly conditional/custom-kernel boundaries that
    were rejected in the original GMM investigation. A future GMM attempt
    needs either a kernel that pipelines full-width GMM2 without occupancy
    regression or a static routing bound established outside the operator.
53. Audited current `vllm-project/tpu-inference` at head
    `7dddc17a98b17fa9aed8d477e4268022045a3212` (2026-08-13) after the router and
    GMM bounds above. No source was ported blindly. The useful findings and
    dispositions are:

    - Its recent GMM improvements are already in this branch: stride-zero
      scale replication ([#3135](https://github.com/vllm-project/tpu-inference/pull/3135)),
      fused gate/up lane interleaving
      ([#3161](https://github.com/vllm-project/tpu-inference/pull/3161)), and
      bucketized M execution
      ([#3219](https://github.com/vllm-project/tpu-inference/pull/3219)).
      The current trace names include the resulting bucket/tile geometry, so
      copying these commits would duplicate existing code.
    - StreamIndex uses `jax.lax.approx_max_k(..., recall_target=1.0)` as its
      exact faster-than-`top_k` selector. That exact primitive is already the
      implementation of `TPU_K3_JAX_EXACT_TOPK`. Milestone 47 measured it
      0.015 ms slower at the endpoint, and its trace still contained the same
      896-way TPU sort. It is not a new alternative to the accepted Pallas
      router.
    - Shared-expert reduction deferral
      ([#2849](https://github.com/vllm-project/tpu-inference/pull/2849)) is the
      same one-collective direction as local PR #410 and the earlier K3 fused
      combine experiments. The exact PR #410 patch is included here and was
      endpoint-neutral; the later latent/shared sharding variants are also
      measured in this log.
    - Dense SparseCore gather-reduce and its tree reduction
      ([#2979](https://github.com/vllm-project/tpu-inference/pull/2979),
      [#3018](https://github.com/vllm-project/tpu-inference/pull/3018)) target
      generic/high-row unpermute. K3 batch 1 already uses a direct 16-row FP32
      weighted reduction and has zero route-permutation sorts. Paying an HBM
      round trip to SparseCore is not justified for this shape.
    - `VLLM_MOE_CHUNK_SIZE` pipelines unpermute with collectives only when a
      token batch can be divided into multiple chunks. One decode token has no
      such pipeline depth. Hierarchical reduce-scatter likewise requires a
      scatterable token/sequence dimension and does not fit a one-token
      all-reduce.
    - The repository's monolithic `fused_ep_moe` kernel is not a viable K3
      batch-1 port. It requires `num_tokens % ep_size == 0`, while this workload
      has one token and EP16. Its subchannel-quantized weight path also requires
      scale blocks aligned to 256, while the lossless checkpoint is MXFP4
      block 32. Padding to 16 tokens and lossy block-256 requantization would
      change both the work and model semantics before any performance claim.
    - The new stacked-RPA work targets conventional paged attention. K3's
      accepted trace instead has specialized KDA and MLA custom calls totaling
      only about 0.32 ms/decode, and the local tree already contains its KDA v3
      and MLA v2 paths. A large stacked-RPA port cannot close the current gap
      on the measured kernel mix.

    Conclusion: the current upstream audit yields no safe untested patch. The
    remaining actionable ideas must reduce the measured TP32 collective path,
    the 0.137 ms inter-executable launch gap, or fuse work across an existing
    K3-specific boundary. Re-check upstream after a direct-VMEM SparseCore
    top-k or a batch-1/EP-capable fused MoE kernel lands.
54. Quantified the time outside the stable model executable. For the accepted
    rank-16 profile, 18 non-outlier model executables average 14.98202 ms. Their
    start-to-start spacing averages 15.11888 ms (15.12171 ms median), and the
    gap from each stable executable end to the next executable start averages
    0.13740 ms (0.13733 ms median, 0.13637--0.13845 ms range). This is the real
    device inter-launch/dependency interval; it is neither 1.3 ms nor 4 ms.

    The accepted endpoint pair averages 15.27388 ms TPOT. Comparing that with
    the profile gives an approximate 0.29187 ms endpoint-minus-executable
    envelope: 0.13740 ms is visible between device executables, and the
    remaining approximately 0.15500 ms lies between device start spacing and
    client-observed TPOT. The endpoint and profiler are separate requests, so
    the latter subtraction includes run variance, streaming/async-output
    accounting, and client timing; it is not evidence of 0.155 ms of input
    preparation. Inside the executable, primary plus asynchronous device work
    already leaves only 0.06545 ms uncovered.

    This also explains the old baseline's apparent four-to-five-millisecond
    remainder: the 12.6 ms table was only a selected list of named operations,
    not the complete 17--18 ms device timeline. The current interval-union
    table is the detailed wall-time accounting; do not subtract a top-N kernel
    list from endpoint TPOT and label the difference host overhead.
55. Tested libtpu async collective fusion on fresh Ray workers and fresh
    caches. This is a compiler fusion pass, distinct from the rejected
    `xla_enable_async_all_reduce` scheduling experiment in section 10. The
    exact launch override was:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-async-collective-fusion
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_async_collective_fusion=true'
    TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=0 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh
    ```

    The flag compiled and served on this TPU v7x-32/libtpu build, but libtpu
    itself warns that the pass is supported/recommended only on Viperlite and
    may otherwise cause incorrect results, hangs, or SDC. Its cold compile was
    also exceptionally verbose (hundreds of thousands of layout-assignment
    messages); log volume is not a readiness signal. The canonical benchmark
    pair and profile were run with:

    ```bash
    OUTPUT_DIR="$PWD/results/kimi-k3-b1-async-collective-fusion" \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR="$PWD/results/kimi-k3-b1-async-collective-fusion-repeat" \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR="$PWD/results/profile-kimi-k3-b1-async-collective-fusion" \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-async-collective-fusion-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The endpoint means were 15.27772 and 15.27654 ms, averaging 15.27713 ms,
    which is 0.00325 ms slower than the accepted 15.27388 ms pair. The latest
    rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-async-collective-fusion-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2731456.1786629679643870728.pt.trace.json.gz`.
    Sixteen normal model executables average 14.98988 ms (14.98926 ms median),
    versus 14.98202 ms for the accepted profile. Their start-to-start spacing
    averages 15.12862 ms and the end-to-next-start gap averages 0.13917 ms,
    versus 15.11888 and 0.13740 ms accepted. The primary plus async device-lane
    union is 14.92637 ms, leaving only 0.06351 ms uncovered.

    Most importantly, the trace has the same 623 matched collective events per
    normal decode as the accepted trace. On the same primary-lane accounting,
    collective duration increased from 5.22903 to 5.24140 ms/decode; the async
    collective lane remained 0.2064 ms. This pass neither merged collectives
    nor shortened their critical path. Reject it, restore
    `xla_tpu_enable_async_collective_fusion=false`, and do not expose this
    platform-risky flag as a launcher option.
56. Tested the separate libtpu async collective merger on fresh Ray workers
    and caches, while holding the accepted model graph and other launcher flags
    fixed. Unlike async collective fusion, this pass emitted no non-Viperlite
    safety warning. The exact launch addition was:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-async-collective-merger
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_async_collective_merger=true'
    TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=0 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh
    ```

    The benchmark and profile commands are the milestone-55 commands with
    `async-collective-fusion` replaced by `async-collective-merger`. Endpoint
    means are 15.20766 and 15.21559 ms, averaging 15.21162 ms. This improves
    the previous accepted 15.27388 ms pair by 0.06226 ms and leaves a 0.21162
    ms gap to the 15 ms objective.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-async-collective-merger-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2812978.1786631597999023926.pt.trace.json.gz`.
    Eighteen stable executables average 14.91814 ms (14.91951 ms median), down
    0.06387 ms from the accepted control. Stable starts are 15.05362 ms apart,
    and the post-executable gap is 0.13612 ms. The primary plus async device
    union falls from 14.91656 to 14.85122 ms, while uncovered time remains
    essentially unchanged at 0.06692 ms.

    The HLO-visible matched collective count remains 623 events/decode (615
    primary-lane and eight async-lane events), so do not describe this as
    removing high-level all-reduces. On identical primary-lane accounting,
    their duration falls from 5.22903 to 5.16166 ms (-0.06736 ms); the async
    lane stays 0.2062 ms. The collective reduction and executable reduction
    agree with the endpoint pair, establishing a causal backend schedule win.
    Accept the flag as a launcher default while preserving explicit `=false`
    for controls.
57. Tested TensorCore compute/collective overlap on top of the accepted async
    collective merger, using fresh Ray workers and caches. This is distinct
    from both async collective fusion and merging. The exact launch was:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-overlap-compute-collective-tc
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_async_collective_merger=true --xla_tpu_overlap_compute_collective_tc=true'
    TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=0 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh
    ```

    The endpoint and profile commands were:

    ```bash
    OUTPUT_DIR="$PWD/results/kimi-k3-b1-overlap-compute-collective-tc" \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR="$PWD/results/profile-kimi-k3-b1-overlap-compute-collective-tc" \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-overlap-compute-collective-tc-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The single canonical endpoint run measured 15.25797 ms mean TPOT, 0.04635
    ms slower than the accepted merger pair. A second endpoint run was not
    justified because the trace independently attributes a larger regression:
    15 stable executables average 14.97352 ms (14.97319 ms median), versus
    14.91814 ms accepted. Primary-lane matched collective duration rises from
    5.16166 to 5.21436 ms/decode, while the count remains 615 primary plus
    eight async events. The primary plus async interval union rises from
    14.85122 to 14.90776 ms; uncovered time remains only 0.06576 ms. Normal
    post-executable gaps have a 0.13702 ms median; the raw mean is invalidated
    by a profiler-window boundary outlier and must not be used as dispatch
    time.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-overlap-compute-collective-tc-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2888689.1786633296525064381.pt.trace.json.gz`.
    Reject `xla_tpu_overlap_compute_collective_tc`; it makes the already
    dominant collective path slower on this workload. Ray step `602.65` and
    server PID 2885827 were stopped after profiling, while allocation job 602
    was deliberately preserved.
58. Tested bidirectional all-reduce decomposition on top of the accepted
    async collective merger, using fresh Ray workers and fresh caches. The
    exact launch addition was:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-bidirectional-allreduce
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_async_collective_merger=true --xla_tpu_decompose_all_reduce_bidirectional_communication=true'
    TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=0 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh
    ```

    The canonical validation and profile commands were:

    ```bash
    OUTPUT_DIR="$PWD/results/kimi-k3-b1-bidirectional-allreduce" \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR="$PWD/results/kimi-k3-b1-bidirectional-allreduce-repeat" \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR="$PWD/results/profile-kimi-k3-b1-bidirectional-allreduce" \
      PROF_DIR=/mnt/scratch/kimi-k3-b1-bidirectional-allreduce-profile \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    Endpoint means are 15.11826 and 15.11931 ms, averaging 15.11879 ms. This
    improves the accepted merger pair by 0.09284 ms and leaves a 0.11879 ms
    gap to the 15 ms objective. The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-bidirectional-allreduce-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2952330.1786635017233596519.pt.trace.json.gz`.
    Nineteen stable model executables average 14.83475 ms (14.83529 ms
    median), 0.08339 ms below the accepted merger profile. Stable starts are
    14.96886 ms apart and their post-executable gap averages 0.13437 ms.
    Primary plus async device-lane occupancy is 14.76383 ms, leaving only
    0.07093 ms uncovered.

    HLO-visible all-reduce count remains exactly 348/decode. Their primary-lane
    duration falls from 5.03057 to 4.94817 ms (-0.08240 ms), which explains
    essentially the entire executable win. The per-decode shape breakdown is:

    | All-reduce result shape | Calls | Merger control | Bidirectional | Change |
    | --- | ---: | ---: | ---: | ---: |
    | tuple `bf16[1,7168] + bf16[1,3584]` | 92 | 2.37616 ms | 2.39450 ms | +0.01834 ms |
    | `bf16[1,7168]` | 95 | 1.81723 ms | 1.81566 ms | -0.00157 ms |
    | `bf16[3584]` | 92 | 0.65925 ms | 0.56034 ms | -0.09891 ms |
    | `bf16[128]` | 69 | 0.17793 ms | 0.17767 ms | -0.00026 ms |

    Thus this is not a generic reduction of every collective. The win comes
    from the 92 TP4-width `bf16[3584]` reductions, partially offset by the
    variadic tuple family. No safety warning was emitted by this libtpu build.
    Accept the flag as a launcher default, while preserving explicit `=false`
    for controls.
59. Audited the installed libtpu implementation before spending a full-model
    compile on `xla_tpu_inference_short_all_reduce_max_tensor_chunks`. The
    flag exists and defaults to zero (disabled), but this workload cannot
    select its emitter. The installed binary is
    `/opt/tpu-info-venv/lib/python3.12/site-packages/libtpu/libtpu.so`; its
    retained local symbol
    `RingSumEmitter::ShouldUseInferenceShortLatencyEmitter` first checks
    `inference_short_ring_sum_emitter_utils::IsSupported`, the configured
    tensor-size limit in target chunks, and exact chunk divisibility. Its
    successful return path then requires both the collective's relevant
    device-count argument and `TpuTopology::LogicalDeviceCount(...)` to equal
    exactly eight. The emitter's retained action builders likewise name only
    `2x2x2` and `4x2` eight-device layouts.

    The K3 v7x-32 executable has 32 logical TPU devices and its measured
    all-reduce replica groups have sizes 32, four, or two, not eight. Raising
    the chunk limit therefore cannot make any of the current 348 all-reduces
    eligible. A full-model trial would only pay another cold compile and could
    not exercise the candidate path, so this is a static rejection rather
    than an unmeasured performance claim. The audit commands are:

    ```bash
    LIBTPU=/opt/tpu-info-venv/lib/python3.12/site-packages/libtpu/libtpu.so
    readelf -Ws "$LIBTPU" | c++filt | \
      rg 'FUNC.*RingSumEmitter::ShouldUseInferenceShortLatencyEmitter'
    objdump -d --demangle --start-address=0x10502750 \
      --stop-address=0x10502a50 "$LIBTPU"
    rg -n 'xla_tpu_inference_short_all_reduce' \
      /mnt/scratch/sc-probe-s1/module_0045*.{flagfile,tpu_comp_env.txt}
    ```

    Caveat: the function address is specific to this installed libtpu build;
    rediscover it with `readelf` after an upgrade. Revisit the flag only on an
    eight-device slice/group or after a libtpu build broadens the selector.
    No source or launcher setting changed for this rejected idea.
60. Added a fast, four-host screening benchmark for the next PR #410
    interaction:
    `benchmarks/kernels/benchmark_kimi_k3_allreduce_layout.py`. It compares a
    dependent loop of adjacent `[1,7168]` and `[1,3584]` BF16 psums against one
    concatenated `[1,10752]` BF16 psum on the exact TP32 device order emitted
    by the current K3 executable. The loop lives inside one compiled program,
    amortizing Python/runtime dispatch, and reports seven device-loop samples
    plus their median difference.

    The probe refuses to time by default unless optimized HLO contains exactly
    one all-reduce with the expected operand shape(s). This matters because two
    unmerged reductions would not model the current compiler-created variadic
    tuple, and a backend-promoted FP32 reduction would not model the K3 BF16
    operation. A one-device CPU mesh confirms the variadic and concatenated
    functions return identical arrays; syntax compilation passes. CPU HLO
    promotes the reduction to FP32, as expected, so CPU timing is deliberately
    not reported as TPU evidence.

    Allocation 602 later expired at its 24-hour limit. A replacement four-host
    v7x-32 allocation, job 664, ran the guarded TPU screen successfully. Both
    programs printed `hlo_matches_expected=true`; optimized HLO contained one
    BF16 variadic all-reduce for the tuple and one BF16 array all-reduce for
    the flat operand. Both selected `RotatedPincerShortEmitter` with
    `StrategyND`, so flattening did not unlock a different backend emitter.

    Seven 1,000-iteration device-loop samples measured a 19.331 us variadic
    median and a 19.429 us flat median. The flat operand was 0.097 us slower
    per adjacent pair, projecting to about +0.0089 ms over 92 decoder layers.
    This is a small negative screen rather than grounds for an early static
    rejection: PR #410 also changes whole-model scheduling and fusion, so its
    already-planned full-model interaction run was launched. The exact screen
    command was:

    ```bash
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_async_collective_merger=true --xla_tpu_decompose_all_reduce_bidirectional_communication=true'
    srun --jobid=664 --overlap -N4 -n4 --ntasks-per-node=1 \
      --unbuffered --export=ALL bash -lc '
        cd /mnt/lustre/homes-x86/woosuk/workspace/vllm-torchtpu-4
        export PATH="$PWD/.venv/bin:$PATH"
        export TPU_ACCELERATOR_TYPE=tpu7x-32
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export JAX_PLATFORMS=
        .venv/bin/python \
          benchmarks/kernels/benchmark_kimi_k3_allreduce_layout.py \
          --iterations 1000 --samples 7
      '
    ```

    This is a screening result only. Even a favorable flat-layout median must
    be followed by the planned full-model two-run benchmark and trace because
    PR #410 also changes layout, scheduling, RMSNorm/up-projection placement,
    and program size. Never use `--skip-hlo-validation` for an acceptance run.
61. Re-tested the exact PR #410 fused MoE combine on top of async collective
    merging and bidirectional all-reduce decomposition. Fresh allocation 664,
    Ray workers, cache namespace
    `/mnt/scratch/kimi-k3-b1-pr410-fused-combine-bidirectional`, and
    `TPU_K3_FUSED_MOE_COMBINE=1` were used. The two endpoint runs measured:

    | Run | Mean TPOT | Median TPOT | P99 TPOT |
    |---|---:|---:|---:|
    | first | 15.19703 ms | 15.19633 ms | 15.20928 ms |
    | repeat | 15.19804 ms | 15.19868 ms | 15.20708 ms |

    Their 15.19753 ms pair average regresses 0.07875 ms from the accepted
    15.11879 ms pair. The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-pr410-fused-combine-bidirectional-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3054790.1786638279481968652.pt.trace.json.gz`.
    After excluding its one 19.78574 ms profiler-boundary/warmup executable,
    18 stable model executables average 14.90591 ms. That is 0.07116 ms slower
    than the accepted bidirectional trace's 14.83475 ms and explains nearly
    all of the endpoint regression. Stable executable starts are 15.04232 ms
    apart and the mean end-to-next-start gap is 0.13628 ms. Primary plus async
    device intervals cover 14.83790 ms, leaving only 0.06801 ms genuinely
    uncovered; the regression is device work, not input preparation or idle.

    The HLO-visible all-reduce count remains 348/decode, but total primary-lane
    all-reduce duration rises from 4.94817 to 5.05646 ms (+0.10829 ms):

    | Shape family | Calls/decode | accepted | PR #410 | Change |
    |---|---:|---:|---:|---:|
    | `[1,7168]` | 95 | 1.81566 ms | 1.81818 ms | +0.00252 ms |
    | tuple `[1,7168] + [1,3584]` / flat `[1,10752]` | 92 | 2.39450 ms | 2.30574 ms | -0.08876 ms |
    | `[3584]` | 92 | 0.56034 ms | 0.73288 ms | +0.17254 ms |
    | `[128]` | 69 | 0.17767 ms | 0.19966 ms | +0.02199 ms |

    Flattening helps its intended TP32 family in the full graph, but its
    scheduling/layout interaction makes the independent TP4 latent-down and
    TP2 KDA reductions slower by more, producing a net regression. This also
    explains why the isolated probe was only a screen: it selected the same
    emitter and predicted a tiny flat penalty, while the full graph changed
    the surrounding schedule. Keep `TPU_K3_FUSED_MOE_COMBINE=0`; the
    implementation remains available only for future compiler/hardware
    interaction tests. The launch, benchmark, profile, cache, sampling, and
    trace-repair caveats are recorded in `RUNBOOK.md`.
62. Screened additional installed-libtpu all-reduce controls with the guarded
    four-host layout benchmark from milestone 60 before considering another
    full-model compile. Native bidirectional decomposition remained the same
    `RotatedPincerShortEmitter`/`StrategyND` path (19.378 us variadic versus
    19.476 us flat), so it does not fix the flat-layout interaction in PR
    #410. `--xla_tpu_use_routing_table_indices_in_all_reduce=true` and
    `--xla_tpu_use_tree_reduce=true` were accepted but did not change the
    selected emitter/configuration or materially move the control timings.
    `--xla_tpu_small_all_reduce_binomial_threshold_bytes=32768` is not a
    registered flag in this libtpu build. Finally,
    `--xla_tpu_force_1d_allreduce_at_chunk_count=1` forced `StrategyRing` and
    regressed the isolated medians to 94.817/95.183 us. None justified a
    full-model run; do not add them to the launcher.

    Each registered flag was tested by appending it to the accepted
    `LIBTPU_INIT_ARGS` from milestone 60 and rerunning the same guarded `srun`
    command. The benchmark's optimized-HLO shape assertion remained enabled.
    Driver argv/config evidence is under `/tmp/tpu_logs_woosuk` for allocation
    664; `/tmp` is ephemeral, so the disposition above is the durable record.
63. Investigated the approximately 0.134 ms interval between accepted model
    executables and tested the first suspected removal mechanism. An initial
    host/device overlay appeared to put 23 serialized `aten::copy_` calls in
    that interval. Milestone 66 corrects that interpretation: this Torch trace
    does not align the host and TPU clocks at microsecond resolution. The host
    calls are real and milestone 65 maps them to the tail of a 162-input
    AOTAutograd replay, but their host timestamps cannot be used to claim that
    they occupy the TPU interval. The device-only breakdown instead finds
    about 0.105 ms of auxiliary TPU modules and only about 0.031 ms uncovered.

    A temporary, cache-keyed compiler diagnostic changed TorchTPU's
    `keep_inference_input_mutations=False` to true only when
    `TPU_KEEP_INFERENCE_INPUT_MUTATIONS=1`. Two focused compiler tests passed,
    and a fresh Ray cluster/cache confirmed all ranks compiled the alternate
    path. The exact one-request result was 29.99 ms mean TPOT for 1,000 output
    tokens, versus the accepted 15.11879 ms pair. The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-keep-input-mutations-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3169093.1786640594153779741.pt.trace.json.gz`.
    Excluding one 32.22376 ms profiler-boundary execution, 17 stable model
    executions average 29.72604 ms and start 29.83863 ms apart, leaving a
    0.11200 ms post-executable gap with zero `aten::copy_` calls. The graph
    instead contains 24 large BF16 state copies totaling 14.6079 ms on the
    primary TPU lane: 23 copies of `[37440,8,2,640]` plus one
    `[38656,8,2,640]` copy. Thus moving mutation replay into the executable
    materializes full device copies and explains essentially the whole
    14.89 ms regression.

    The diagnostic source and its tests were removed after the rejection;
    the default compiler is unchanged. An alternative direct `set_` mutation
    was also rejected before a TPU compile because Dynamo fullgraph export
    reports `Getting an inplace view on a graph input is not supported`.
    Some workers transiently logged a 17.25 GiB program-allocation failure
    while loading this much larger executable before the server recovered,
    another reason not to retain it. A future attack on the gap must batch or
    rebind the runtime writebacks without turning them into HLO tensor copies.
    The exact launch/evaluation commands and profiler-flush caveat are in
    `RUNBOOK.md`.
64. Tested the donation-preserving runtime-rebind version of the KDA mutation
    idea and rejected it. The diagnostic rewrote AOTAutograd's generated
    mutation epilogue from `copy_` to `set_`, outside the XLA graph, and armed
    the rewrite around the compiled callable's lazy first invocation. Two
    focused CPU tests verified that the patch was scoped/restored and that
    the updated tensor storage was aliased after the first lazy invocation.
    The first implementation armed the rewrite only while constructing the
    AOT wrapper, so it never observed the lazily generated epilogue; that
    launch was stopped after distributed memory profiling made no progress
    for more than six minutes. Only its exact server process and Ray step were
    canceled; allocation 664 was preserved. A fresh 32-resource Ray session
    was required after correcting the hook.

    The corrected launch logged **324 generated `copy_` sites per rank**, not
    the expected 23. That is 162 data-mutation inputs with leaf/non-leaf branch
    variants, so a textual epilogue rewrite is much broader than the 23 tail
    calls visible in one accepted post-executable gap. The server nevertheless
    constructed and completed all requests. Its one canonical four-request
    result was 15.22068 ms mean TPOT (15.21427 median, 15.24210 p99), a
    +0.10190 ms regression from the accepted 15.11879 ms pair average. The
    result is under `results/kimi-k3-b1-runtime-state-rebind-v2`.

    The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-runtime-state-rebind-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3357610.1786644450540916711.pt.trace.json.gz`.
    Nineteen stable `region.268435455` device intervals have a 14.92480 ms
    median, 0.10071 ms slower than the accepted trace's 14.82409 ms median.
    The trace also contains 3,097 `aten::set_` calls (163/decode, 0.21854
    ms/decode aggregate host duration) and 399 remaining `aten::copy_` calls
    (21/decode, 0.22857 ms/decode aggregate). Thus broad storage rebinding
    neither removes all runtime copies nor shortens the device interval, and
    its host dispatch is itself larger than the original 0.13437 ms gap.
    Because the candidate is causally slower and changes storage for many
    non-KDA mutations, no second endpoint run was warranted. The diagnostic
    compiler/platform/test code was removed after profiling; it is not a
    launcher option. The exact commands, lazy-codegen trap, cache-format
    warning, and trace-parser caveat are in `RUNBOOK.md`.
65. Mapped every AOT mutation input by instrumenting the lazily generated
    mutation epilogue without changing its behavior. The rank-0 audit recorded
    exactly 162 data mutations in four shape families:

    | Shape / dtype | Count | Interpretation |
    |---|---:|---|
    | `[9,3,1152]` BF16 | 69 | KDA convolution state |
    | `[9,3,128,128]` FP32 | 69 | KDA recurrent state |
    | `[37440,8,2,640]` BF16 | 23 | ordinary attention KV pools |
    | `[38656,8,2,640]` BF16 | 1 | final attention KV pool |

    The 23 ordinary-KV mutations are updated-input indices 6, 13, 20, ...,
    160. They are interleaved with six KDA-state mutations per group and are
    followed by the final KV mutation at index 161. Thus the 23
    `aten::copy_` calls observed after a normal executable are simply the last
    approximately 23 dispatches of the ordered 162-call epilogue; earlier
    calls overlap the asynchronous device interval. This reconciles the broad
    `set_` trace's 163 calls/decode with the 23 normal copies in the visible
    gap, and invalidates selecting the 23 KV-pool shape as a gap remedy.

    The read-only audit passed its focused test and logged once per generated
    epilogue on every rank. Its successful fresh-cache launch used
    `/mnt/scratch/kimi-k3-b1-runtime-state-audit`. An earlier launch
    accidentally pointed `VLLM_CACHE_ROOT` at the accepted cache and loaded a
    zero-artifact serialized AOT entry, failing with `TypeError: 'NoneType'
    object is not callable`. A unique `VLLM_CACHE_ROOT`, not merely a unique
    XLA/profile path, is mandatory for compiler-wrapper experiments. Exact
    commands and the rank-0 log path are in `RUNBOOK.md`.

    A semantics-preserving follow-up replaced all 162 Python `Tensor.copy_`
    dispatches with one mixed-shape/mixed-dtype `aten::_foreach_copy_` call
    under a default-off diagnostic. Two focused tests passed, and code
    generation refused the rewrite unless it saw exactly two inference
    leaf-copy branches for every mutation. All ranks logged exactly 162
    batched copies and the server completed all requests.

    The canonical endpoint runs measured 15.14086 and 15.11524 ms mean TPOT,
    averaging 15.12805 ms: 0.00927 ms slower than the accepted 15.11879 ms
    pair. The latest rank-0 trace is
    `/mnt/scratch/kimi-k3-b1-foreach-runtime-copies-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3517963.1786648147416799827.pt.trace.json.gz`.
    Its 18 stable model regions average 14.85159 ms and start 14.98840 ms
    apart, versus 14.83462 and 14.97048 ms in the comparable accepted capture.
    Most decisively, the end-to-next-start gap is unchanged at 0.13570 ms
    versus 0.13608 ms accepted. The trace has one 0.21790 ms
    `aten::_foreach_copy_` event per decode plus 21 remaining `aten::copy_`
    calls/decode. CompositeExplicitAutograd reduces the visible child dispatch
    count but does not shorten the critical gap; its own call is larger than
    that gap and is overlapped with device execution. Reject it and leave the
    default compiler unchanged. The exact audit/foreach diagnostic and tests
    are archived in commit `75922a6f`; commit `63bae361` removes them from the
    branch tip while retaining this durable record.
66. Reparsed the accepted trace using only TPU-device timestamps and resolved
    the remaining inter-backbone interval. Across 17 stable gaps, consecutive
    `region.268435455` backbone executions start 14.97048 ms apart and leave a
    0.13608 ms mean interval after the preceding backbone finishes. The union
    of top-level TPU modules on both cores fills 0.10517 ms of that interval;
    only 0.03091 ms is not covered by a top-level device module. Mean module
    contributions are:

    | Post-backbone TPU work | Mean per token |
    |---|---:|
    | LM-head local projection + TP32 vocabulary all-gather | 0.06839 ms |
    | Vocabulary-sized random tensor | 0.00409 ms |
    | RNG-state update | 0.00253 ms |
    | Neutral-sampling Gumbel argmax | 0.01881 ms |
    | Token view / `as_strided` | 0.00503 ms |
    | Async placeholder-token scatter | 0.00462 ms |
    | Other tiny/overlapping device work | about 0.00170 ms |
    | Inter-module interval not covered by a top-level TPU module | 0.03091 ms |

    This is the detailed answer to the earlier “where is the remaining 4 ms?”
    and “what is the 1.3 ms idle?” questions: neither residual exists in a
    stable decode. The 14.83475 ms backbone plus 0.13608 ms post-backbone tail
    explains the 14.97048 ms device start spacing. The endpoint pair is
    15.11879 ms, leaving about 0.14831 ms from device start spacing to client
    TPOT; that last cross-request difference includes server/client streaming
    and measurement variance and is not a hidden four-ms model stage.

    The correction matters for causality. Chrome-trace host events (`pid` of
    the Ray worker) and TPU events (`pid` 2/3) have a several-millisecond clock
    offset in this capture, so an `aten::copy_` timestamp must not be overlaid
    directly on a TPU gap. The generic foreach experiment remains a valid
    negative endpoint result, but its unchanged device gap does not prove that
    mutation replay occupies that gap. Reproduce the device-only accounting
    with `scripts/k3/analyze_trace_tail.py`; the exact command is in
    `RUNBOOK.md`.
67. Implemented and measured an opt-in selected-hidden -> LM-head -> sampler
    fusion with `TPU_K3_FUSE_LM_HEAD_SAMPLER=1`. It keeps the existing TP32
    full-vocabulary all-gather and exact Gumbel-max sampling algorithm, but
    places the local projection, all-gather, and argmax in one compiled TPU
    module. The flag is restricted to non-speculative generation without
    grammar or logprobs; unsupported requests fail explicitly. It remains off
    by default because it is a narrow batch-1 specialization, not a general
    replacement for the sampler.

    Two fused endpoint runs measured 15.1334594 and 15.1313557 ms mean TPOT
    (15.1324076 ms pair). Those absolute values are slower than the older
    accepted 15.1187865 ms pair, but the environment rewrote the enhanced
    launch barrier to false in this allocation, so the old pair is not a fair
    microsecond-scale control. A fresh same-allocation, same-source flag-off
    control measured 15.1419436 and 15.1396092 ms (15.1407764 ms pair). The
    fusion therefore wins the matched endpoint comparison by **0.0083688 ms
    per token**.

    Device-only traces agree. The two fused captures leave 0.12800 and
    0.12836 ms mean post-backbone gaps versus 0.13590 and 0.13609 ms for the
    matched controls. The fused LM-head/all-gather/argmax module costs
    0.081285 ms/decode versus 0.06835 + 0.01877 = 0.08712 ms for the two
    separate control modules. The vocabulary RNG remains a separate launch
    (0.005996 ms fused versus 0.00409 ms control); including it, named device
    work falls by about 0.0039 ms while eliminating one launch reduces the
    whole gap by about 0.0079 ms and endpoint TPOT by about 0.0084 ms. The
    clean second fused capture starts backbones 14.95671 ms
    apart, but short profiles contain occasional slow backbone outliers, so
    use the endpoint pair and gap/module decomposition rather than treating
    one start-spacing mean as a new absolute best.

    Exact launch, benchmark, profiling, analysis commands, cache paths, and
    trace paths are in `RUNBOOK.md`. `scripts/k3/analyze_trace_tail.py` now
    reports backbone duration, start spacing, and optional top device-module
    names in addition to the detailed tail accounting. `git diff --check`,
    shell/Python syntax checks, and all 80 `tests/runner/test_tpu_runner.py`
    tests pass. Running that suite with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
    causes 25 fixture setup errors because `torch.device("tpu")` is then
    unregistered; rerun normally rather than interpreting those as candidate
    failures.

68. Identified the next exact candidate after the fusion screen. The installed
    vLLM `LogitsProcessor.get_top_tokens()` already implements vocab-parallel
    greedy argmax by gathering only one `(max_value, global_token_id)` pair
    per TP rank instead of all-gathering the vocabulary. Neutral Gumbel-max
    sampling admits the same exact reduction: apply temperature/noise to each
    local vocab shard, compute its local noisy maximum, all-gather 32 pairs,
    and select the global maximum. The first implementation should still draw
    the current full-vocabulary `u` tensor on every rank and slice the local
    shard. That preserves existing generator advancement and deterministic
    random-number mapping while removing the logits all-gather; optimizing the
    random draw can follow only after proving generator equivalence. Gate the
    path to no grammar, no logprobs, and neutral top-k/top-p, falling back
    otherwise. This is an analyzed future candidate, not yet implemented or
    benchmarked.

69. Implemented the milestone-68 candidate behind the default-off
    `TPU_K3_DISTRIBUTED_NEUTRAL_SAMPLER=1` flag and rejected this first
    standalone-collective architecture. The implementation preserves the
    current full-vocabulary random draw on every rank, slices the local vocab
    interval, applies the local LM head plus exact temperature/Gumbel transform,
    and all-gathers only one fp32 `(max_value, global_token_id)` pair per TP
    rank. Greedy selection uses the same reduction without noise. It is gated
    to non-speculative requests with neutral top-k/top-p and no grammar or
    logprobs. This is the same pair-reduction primitive as installed vLLM's
    `LogitsProcessor.get_top_tokens()`; the NVIDIA Inkling and MLJ router
    sources concern the separate 896-way MoE top-16 problem already resolved
    and measured in milestones 49 and 51.

    A clean four-host launch used fresh Ray step `664.18` after cancelling
    only stale step `664.17`; parent allocation `664` remained running. XLA
    successfully compiled `sample_from_sharded_hidden_states` on all 32 ranks.
    The candidate server became healthy, but the first canonical benchmark
    warmup produced no token for 176 seconds. Worker logs show the live sampler
    executable compiling at 20:44:19--20:44:20 UTC, followed by repeated
    `No available shared memory broadcast block found in 60 seconds` messages
    and no request completion. `/health` continued returning HTTP 200, so
    endpoint health alone was misleading. The benchmark was interrupted at
    `0/2` warmups; no result JSON or usable trace was produced. The server was
    then stopped, while Ray step `664.18` and the parent allocation were
    preserved.

    The evidence is consistent with a collective-order/participation stall in
    the new standalone sampling executable, not a compile rejection. Do not
    benchmark this flag as-is. A justified retry must put the pair reduction
    inside an executable already known to run in TP lockstep, or first prove a
    synchronized eager reduction in a small multihost harness. It must also
    make precompile arguments exactly match live dtypes so the first request
    does not build another sampler variant. Keep the code default-off as a
    reproducible prototype and warning; the accepted configuration and
    15.1187865 ms best pair are unchanged. Static checks and the normal-device
    `tests/runner/test_tpu_runner.py` suite pass (80 tests).

70. Re-audited current
    [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference)
    at head `5fa7bb815bf63afec509e27f4bc7f68922cdd233` (2026-08-13),
    used its lockstep design to repair the distributed sampler deadlock, and
    rejected the repaired sampler on matched performance evidence.

    The current upstream MoE conclusions are unchanged from milestone 53.
    Its stride-zero GMM scale load, fused gate/up lane interleave, bucketized-M
    execution, exact `approx_max_k`, shared-expert reduction deferral, and
    SparseCore gather/reduce paths are already present locally or already
    measured/rejected for K3 batch 1. No newer K3-applicable MoE kernel was
    found. The new useful architectural reference is
    [`tpu_inference/runner/decode_loop.py`](https://github.com/vllm-project/tpu-inference/blob/5fa7bb815bf63afec509e27f4bc7f68922cdd233/tpu_inference/runner/decode_loop.py):
    `model_fn -> compute_logits_fn -> sample_fn` runs inside one top-level
    `jax.jit`/`lax.while_loop`, with donated KV cache. PR
    [#3374](https://github.com/vllm-project/tpu-inference/pull/3374) also shows
    that this wrapper must inherit the model step's compiler options rather
    than nest independently compiled functions. Its normal sampling path still
    explicitly unshards the full vocabulary before sampling and carries a TODO
    for sharded-vocab sampling, so it contains no ready-made pair-reduction
    sampler. Its compilation manager does provide an important operational
    rule: dummy logits and sampling metadata must exactly match live dtype and
    sharding (`float32` logits/temperature/top-p and `int32` top-k), or the
    first request recompiles.

    A direct port of that fused loop is not a small change in this TorchTPU
    runner. K3's 92-layer decoder is compiled by vLLM's `support_torch_compile`
    wrapper as one AOT backbone, while input prep, sampling, async placeholder
    state, and KDA mutation replay live outside it. Wrapping the already
    compiled backbone plus a sampler cannot create the single SPMD compilation
    boundary used by JAX; a faithful port would recompile the entire decoder,
    thread/donate KV and KDA state through it, and change scheduler ownership.
    That is disproportionate for a measured 0.119 ms target gap and risks
    changing the canonical one-host-step workload.

    Instead, the milestone-69 sampler was split at the risky boundary. The
    local LM head, exact temperature/Gumbel transform, and local maximum now
    run in a collective-free compiled graph; every worker then calls the
    ordinary process-group two-value all-gather from the common runner path;
    a second tiny compiled graph selects the winning global token. Precompile
    now builds only the two collective-free graphs using live-matching shapes
    and dtypes. Fresh Ray step `664.30` had 32/32 TPUs. A four-token liveness
    request returned HTTP 200 with four tokens in 1.6 seconds, proving that the
    split removes the previous first-request deadlock.

    The canonical endpoint result rejects it. Two candidate means were
    15.2901584 and 15.2938704 ms; two same-source, same-Ray, independently
    clean-compiled flag-off controls were 15.2733866 and 15.2872469 ms. Pair
    averages are 15.2920144 versus 15.2803168 ms: the repaired sampler is
    **0.0116976 ms/token slower**. Cross-session comparison to the 15.1187865
    ms absolute best is invalid here because both fresh source hashes produced
    a roughly 14.99 ms backbone; the matched control is the attribution.

    Two candidate traces give a 152.83--153.32 us post-backbone gap, versus
    140.28--140.38 us for two controls. Candidate device components are about
    38.8 us local LM-head/Gumbel/max, 49.7--50.1 us across the two emitted
    pair-allgather modules, 9.3--9.4 us final selection, 6.1 us vocabulary RNG,
    5.1 us token view, and 4.6--4.8 us placeholder scatter, leaving about
    31.1 us uncovered. Controls spend about 72.7 us on LM head/full-vocab
    all-gather, 18.8 us on sampling/argmax, 6.7 us on the two RNG modules,
    5.0 us on the token view, and 4.7 us on placeholder scatter, leaving about
    31.0 us uncovered. Thus shrinking the communication payload does work, but
    the extra executable plus process-group rendezvous gives the time back and
    adds about 12 us, matching the endpoint regression.

    `analyze_trace_tail.py` initially misreported 34 half-step gaps and a
    7.5 ms "backbone" because the tiny final-select executable reused the same
    root `region.268435455` name as the decoder. It now ignores same-named
    regions shorter than 1000 us by default; override with
    `--min-backbone-duration-us` only for a genuinely different model. This is
    a trace-tool correction, not a performance change. `git diff --check`,
    Python and shell syntax checks, and `tests/runner/test_tpu_runner.py` pass
    (80 tests, 15 warnings). Keep the repaired code default-off for
    reproducibility, but do not pursue another sampler-only executable split.

71. Accepted libtpu's multi-compute layer-scheduler overlap after a matched,
    fresh-Ray candidate/control experiment. The exact candidate-only delta was
    `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`. It does
    not change model math, sharding, or HLO-visible collective count; it lets
    the dependent-collective layer scheduler overlap multiple compute regions.

    The first attempted session (`664.31`) was invalidated before benchmarking.
    Ray was started with only the candidate flag, while the server launcher
    appended the previously accepted scheduler/merger/bidirectional flags.
    Inspection of a remote `tpu_driver` command line showed that actors had
    inherited only the flag present at Ray-node startup. This establishes an
    important operational rule: changing `LIBTPU_INIT_ARGS` on a later vLLM
    launcher does not reliably reconfigure persistent Ray actors. The complete
    flag set must be exported before the Ray `srun`, and every libtpu change
    requires fresh Ray workers as well as fresh compilation caches.

    A second audit found that TorchTPU rewrites
    `--xla_tpu_use_enhanced_launch_barrier=true` to `false` in distributed
    actors, with a warning that `true` can cause distributed hangs. Historical
    worker logs show the same rewrite in the older accepted sessions. Enhanced
    barriers therefore were never part of the effective accepted executable;
    the launcher default has been removed and the historical 15.1187865 ms
    result remains valid under the effective `false` setting.

    Fresh candidate Ray step `664.32` received this complete startup value
    (the enhanced-barrier argument is shown only to record the discovery; the
    worker rewrote it to `false`):

    ```bash
    export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_async_collective_merger=true --xla_tpu_decompose_all_reduce_bidirectional_communication=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true'
    export K3_REPO_DIR=$PWD K3_JOB_ID=664 K3_HEAD_IP=10.13.232.26 K3_RAY_PORT=6379
    srun --jobid="$K3_JOB_ID" --overlap -N4 -n4 --ntasks-per-node=1 \
      --unbuffered --export=ALL bash -lc '
        export PATH="$K3_REPO_DIR/.venv/bin:$PATH"
        node_ip=$(hostname -I | cut -d" " -f1)
        if [[ $SLURM_PROCID -eq 0 ]]; then
          exec ray start --head --node-ip-address="$node_ip" \
            --port="$K3_RAY_PORT" --resources='"'"'{"TPU":8}'"'"' \
            --disable-usage-stats --block
        fi
        until timeout 1 bash -c \
            "</dev/tcp/$K3_HEAD_IP/$K3_RAY_PORT" 2>/dev/null; do sleep 1; done
        exec ray start --address="$K3_HEAD_IP:$K3_RAY_PORT" \
          --node-ip-address="$node_ip" --resources='"'"'{"TPU":8}'"'"' \
          --disable-usage-stats --block
      '
    ```

    After verifying 32 TPU resources, the exact candidate server, benchmark,
    and profile commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-multi-compute-layer-scheduler-v2
    export RAY_ADDRESS=10.13.232.26:6379
    export HF_HOME=/mnt/scratch/hf-hub
    export HF_HUB_CACHE=$HF_HOME/hub HF_MODULES_CACHE=$HF_HOME/modules HF_HUB_OFFLINE=1
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME PROF_DIR=$PROF_DIR \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    The control required another fresh Ray step (`664.33`), because libtpu was
    already initialized in the candidate actors. It used the same startup
    string with only the multi-compute flag removed, then the same commands
    with `K3_RUN_NAME=kimi-k3-b1-multi-compute-layer-scheduler-control-v2`.
    Actor command lines were inspected in both sessions: candidate workers had
    dependent scheduling, async merging, bidirectional decomposition, and
    multi-compute overlap effective; controls had the first three but not the
    candidate. Enhanced barriers were effective `false` in both.

    | Measurement | Candidate | Matched control | Candidate change |
    | --- | ---: | ---: | ---: |
    | Endpoint run 1 mean TPOT | 15.1348461 ms | 15.2038554 ms | -0.0690093 ms |
    | Endpoint run 2 mean TPOT | 15.1260408 ms | 15.2060887 ms | -0.0800479 ms |
    | Endpoint pair average | **15.1304435 ms** | **15.2049720 ms** | **-0.0745286 ms** |
    | Stable trace backbone | 14.8395028 ms | 14.9154054 ms | -0.0759026 ms |
    | Post-backbone device gap | 0.1360355 ms | 0.1378425 ms | -0.0018071 ms |
    | Device start spacing | 14.9755383 ms | 15.0532480 ms | -0.0777097 ms |

    The trace localizes essentially the entire endpoint win to the model
    executable. It is not input preparation, sampling, or the roughly 0.136 ms
    post-backbone tail. The candidate's absolute pair is 0.01166 ms slower
    than the historical best because fresh compilation/session baselines
    shifted; use the matched -0.07453 ms delta for attribution. Applying that
    delta to the historical 15.1187865 ms pair projects 15.0442579 ms, but that
    number is **not** a verified endpoint result. The strict best remains
    15.1187865 ms until a clean accepted-stack validation reproduces the gain.

    Candidate results are in
    `results/kimi-k3-b1-multi-compute-layer-scheduler-v2{,-repeat}`; controls
    are in
    `results/kimi-k3-b1-multi-compute-layer-scheduler-control-v2{,-repeat}`.
    The stable rank-0 traces used above are:

    ```text
    /mnt/scratch/kimi-k3-b1-multi-compute-layer-scheduler-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_4002722.1786659616108759429.pt.trace.json.gz
    /mnt/scratch/kimi-k3-b1-multi-compute-layer-scheduler-control-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_4066076.1786661087127912117.pt.trace.json.gz
    ```

    Analyze either with
    `.venv/bin/python scripts/k3/analyze_trace_tail.py --top-modules 15 TRACE`.
    The launcher and Slurm wrapper now default the accepted flag on, while
    preserving explicit caller settings. The Slurm wrapper supplies the full
    accepted set before Ray startup. The server and exact step `664.33` were
    stopped after capture; parent allocation `664` remains running.

72. Screened the next narrow extension of milestone 71 and rejected it on
    trace causality. Installed libtpu exposes the autotune range
    `--xla_tpu_layer_scheduler_min_compute_latency={0.0,1.0}`; the default is
    `auto`. Fresh Ray step `664.34` used the complete accepted startup string
    plus `--xla_tpu_layer_scheduler_min_compute_latency=0.0`, with unique
    cache/profile namespace
    `/mnt/scratch/kimi-k3-b1-layer-scheduler-min-compute-zero-v1`.

    Two canonical endpoint means were 15.1222852 and 15.1167043 ms, averaging
    15.1194947 ms. That is superficially 0.0109487 ms faster than the
    milestone-71 multi-compute candidate pair, but the stable second trace
    does not support that attribution:

    | Device interval | min-compute `0.0` | milestone-71 candidate | Change |
    | --- | ---: | ---: | ---: |
    | Backbone | 14.8346349 ms | 14.8395028 ms | -0.0048679 ms |
    | Post-backbone gap | 0.1402548 ms | 0.1360355 ms | +0.0042193 ms |
    | Start spacing | 14.9748897 ms | 14.9755383 ms | -0.0006486 ms |

    The candidate merely trades about 4--5 us from the backbone into the tail,
    leaving device start spacing neutral. A fresh flag-off compile is not
    justified for a sub-microsecond device delta. Keep the threshold at
    `auto`; do not add it to launcher defaults. Result directories are
    `results/kimi-k3-b1-layer-scheduler-min-compute-zero-v1{,-repeat}`. The
    analyzed rank-0 trace is:

    ```text
    /mnt/scratch/kimi-k3-b1-layer-scheduler-min-compute-zero-v1-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_4141542.1786663173234711254.pt.trace.json.gz
    ```

    The exact launch was the milestone-71 accepted command with only this
    appended before Ray startup:
    `--xla_tpu_layer_scheduler_min_compute_latency=0.0`; benchmark/profile and
    analyzer commands were unchanged. The server and exact Ray step `664.34`
    were stopped; allocation `664` remains running. Static inspection also
    established that `xla_tpu_scheduler_using_real_cost_model` and
    `xla_tpu_enable_scheduler_memory_pressure_tracking` default true in this
    libtpu, so explicitly enabling either is not a new experiment.

73. Tested the complementary short-communication threshold and rejected it on
    a tight endpoint pair. Fresh Ray step `664.35` used the complete accepted
    milestone-71 flag set plus
    `--xla_tpu_layer_scheduler_min_async_communication_latency=0.0`, supplied
    before Ray startup. Its unique cache/profile namespace was
    `/mnt/scratch/kimi-k3-b1-layer-scheduler-min-async-zero-v1`.

    Canonical means were 15.1316907 and 15.1331638 ms, averaging 15.1324273
    ms. This is 0.0019838 ms slower than the milestone-71 multi-compute pair
    average of 15.1304435 ms and is entirely within run noise. Because there
    is no endpoint improvement to attribute, no profile or fresh flag-off
    compile was taken. Keep the threshold at `auto`; the result directories
    are `results/kimi-k3-b1-layer-scheduler-min-async-zero-v1{,-repeat}`.
    The server and exact Ray step `664.35` were stopped; allocation `664`
    remains running.

    A source audit also bounded the next substantive tail idea. The existing
    `TPU_K3_FUSE_LM_HEAD_SAMPLER` graph begins after
    `KimiModel.forward`'s `support_torch_compile` boundary, which explains why
    it removes only one auxiliary launch and saves 0.00837 ms. A true
    backbone/LM-head/sampler fusion cannot be obtained by wrapping that
    already-compiled function. It requires moving the compilation boundary to
    `KimiLinearForCausalLM`, passing fixed-shape temperature/top-k/top-p/RNG
    tensors into the model step, returning the selected token through
    `execute_model`, and bypassing the later sampler only for the strict
    no-speculation/no-grammar/no-logprobs path. This is the next credible
    design capable of attacking most of the measured 0.136 ms tail; it must
    preserve full-vocabulary Gumbel-max and generator advancement exactly.

74. Implemented and served the milestone-73 whole-step boundary experiment:
    opt-in `TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1` moves
    `support_torch_compile` from `KimiModel` to
    `KimiLinearForCausalLM`, selects the active hidden row, executes the TP32
    LM head, applies the existing temperature-scaled full-vocabulary
    Gumbel-max sampler, and returns the token from one outer executable. The
    runner supplies the existing rank-synchronized random stream and bypasses
    the later LM-head/sampler path. It is deliberately restricted to
    `max_num_seqs=max_num_batched_tokens=1`, no speculation, PCP, grammar,
    logprobs, greedy requests, or non-neutral top-k/top-p. It is off by
    default and mutually exclusive with the two earlier post-backbone sampler
    experiments.

    The first `v1` compile proved the 70,922/81,830-instruction graph viable
    (151 overlays, 140.56 MiB program, 63.94/64.00 MiB VMEM, about 217 seconds
    for final capture), but its first real endpoint request failed with
    `RuntimeProgramInputMismatch`: dummy capture supplied sampling noise
    `[1,163840]` while the generic request padding path supplied
    `[8,163840]`. The server correctly returned HTTP 500 and died, so this
    produced no benchmark result. `v2` passes only the one real request row
    through the fused LM head and sampler; it compiled and served normally.

    The two canonical `v2` endpoint means were 15.1354325 and 15.1069760 ms,
    averaging 15.1212043 ms. Both detailed result files show four successful,
    zero failed requests, four input tokens, and 4,000 generated tokens, with
    every input length exactly 1 and output length exactly 1,000. Stable
    second-trace device accounting against the fresh-Ray, same-source flag-off
    control is:

    | Device interval | Outer fused graph | Matched flag-off control | Change |
    | --- | ---: | ---: | ---: |
    | Compiled executable | 14.9032681 ms | 14.9237891 ms | -0.0205211 ms |
    | Post-executable gap | 0.0427384 ms | 0.1377510 ms | -0.0950126 ms |
    | Start spacing | 14.9460064 ms | 15.0615401 ms | -0.1155337 ms |

    The matched control endpoint means were 15.2135721 and 15.2168965 ms,
    averaging 15.2152343 ms. The fused candidate therefore wins by
    0.0940301 ms/token end to end. The stable device trace indicates a
    0.1155337 ms/token start-spacing win: it eliminates the control's separate
    69.934 us LM-head/all-gather and 18.805 us argmax modules and associated
    launch intervals. The outer executable is itself 20.521 us shorter in
    this compile rather than longer, which is why the earlier comparison to
    the older milestone-71 binary understated the gain. The remaining 42.74
    us gap contains
    about 4.54 us full-vocabulary uniform generation, 2.88 us seed/uniform
    setup, 5.20 us token view, 4.85 us placeholder scatter, and 24.81 us not
    attributed to named device modules. It does not erase sampling work, and
    random-number generation still occurs outside the outer graph before its
    input can be consumed.

    Candidate and control detailed JSONs both completed 4/4 requests per run
    with no errors and the exact expected 1-token inputs/1,000-token outputs.
    More strongly, the four detokenized outputs are byte-for-byte identical
    between candidate and control in each corresponding run. Their SHA-256
    hashes (over the JSON-encoded output list) are
    `7028e05098bf2cc993b9a27608577eeba2157c6b13283b5296521b08c830f6b7`
    and
    `b6138c70a6259103bf7bc0a84bc3347e376a35754b176f49032f92ab8b4356b0`.
    This is useful equivalence evidence for the dummy-weight sampled path,
    not a substitute for real-checkpoint quality validation.

    Exact candidate launch on fresh Ray step `664.36`:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-backbone-lm-sampler-v2
    export RAY_ADDRESS=10.13.232.26:6379
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-backbone-lm-sampler-v1
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v1/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
      PROF_DIR=$PROF_DIR \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    Analyze the stable trace with:

    ```bash
    .venv/bin/python scripts/k3/analyze_trace_tail.py --top-modules 20 \
      /mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_89892.1786667257350005705.pt.trace.json.gz
    ```

    The matched control used exact Ray step `664.45`, the same complete
    `LIBTPU_INIT_ARGS`, source checkout, and Tier-2/Tier-3 lineage, but a fresh
    flag-off runtime namespace:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-backbone-lm-sampler-control-v2
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=0 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
      PROF_DIR=$PROF_DIR \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26

    .venv/bin/python scripts/k3/analyze_trace_tail.py --top-modules 30 \
      /mnt/scratch/kimi-k3-b1-backbone-lm-sampler-control-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_201194.1786668732781632626.pt.trace.json.gz
    ```

    The canonical workload is real serving execution with synthetic content:
    one probe, two unmeasured warmups, then four measured sequential requests;
    each has one reproducible random non-special prefix token (`seed=42`) and
    exactly 1,000 requested outputs with EOS ignored. The launcher uses dummy
    weights, so tensor shapes, kernels, collectives, scheduling, sampling, and
    HTTP streaming are real, but generated text is random and is not quality
    evidence. `bench_isl1_osl1k.sh` now adds `--save-detailed`, preserving
    per-request lengths/timings/errors and generated text; historical JSONs
    retain only aggregates. The literal prompt text is not saved. Also expect
    a 20--30 second benchmark-client startup while it imports TorchTPU and
    emits the benign metadata 404. `VLLM_PLUGINS=""` breaks this installed CLI
    with an unresolved `current_platform`; it is not a client-startup fix.

75. Eliminated the remaining view and placeholder-scatter modules from the
    opt-in whole-step path, but did not establish a new absolute endpoint
    best. The outer graph now returns its sampled token as one-dimensional
    `int32`; the async runner parks that tensor without `view(-1)` and, for the
    admitted exact batch-1 continuation, reuses it directly as the next
    step's `input_ids` rather than staging two index tensors and launching
    `_substitute_placeholder_token`.

    The first live `v3` attempt compiled successfully but exposed a host-side
    shape assumption before any timed request completed. The async public
    output path already normalizes a 1-D copied token tensor to
    `[num_reqs, 1]`, while `_modify_prev_results` did not; its `.tolist()`
    therefore produced an integer and failed at `len(sampled_ids)` with
    `TypeError: object of type 'int' has no len()`. The benchmark correctly
    reported 0 successful/4 failed requests and HTTP 500, and the repeat was
    interrupted before it ran. `_modify_prev_results` now performs the same
    1-D-to-2-D normalization. Fresh Ray step `664.47` then completed a smoke
    request with exactly one prompt and 1,000 completion tokens, followed by
    three detailed 4/4 endpoint runs with zero errors and exact
    `input_lens=[1,1,1,1]` / `output_lens=[1000,1000,1000,1000]`.

    The endpoint results were:

    | Run | Mean TPOT | Median TPOT | Interpretation |
    | --- | ---: | ---: | --- |
    | `direct-token-v3-fix` | 15.1744385 ms | 15.1711596 ms | clean |
    | `direct-token-v3-fix-repeat` | 15.3153087 ms | 15.1784383 ms | one request contained a 138.19 ms ITL stall |
    | `direct-token-v3-fix-repeat2` | 15.2000188 ms | 15.1678214 ms | one request contained a 79.09 ms ITL stall |

    Excluding those isolated stalls, individual request means cluster at
    15.1672--15.1881 ms. This is slower than the earlier whole-step `v2`
    pair (15.1212043 ms average), so it is not a new absolute best and does
    not close the 15 ms objective. The stalls are why detailed per-request
    ITLs and a repeat are mandatory; never average the all-failed attempt or
    interpret a single long ITL as TPU kernel time.

    The first stable rank-0 capture is:

    ```text
    /mnt/scratch/kimi-k3-b1-backbone-lm-sampler-direct-token-v3-fix-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_332681.1786670922652223808.pt.trace.json.gz
    ```

    Its device accounting versus whole-step `v2` is:

    | Device interval | Direct-token v3 | Whole-step v2 | Change |
    | --- | ---: | ---: | ---: |
    | Compiled executable | 14.9991191 ms | 14.9032681 ms | +0.0958510 ms |
    | Post-executable gap | 0.0275971 ms | 0.0427384 ms | -0.0151413 ms |
    | Start spacing | 15.0267162 ms | 14.9460064 ms | +0.0807098 ms |
    | Token view | 0 | 0.0051971 ms | -0.0051971 ms |
    | Placeholder scatter | 0 | 0.0048511 ms | -0.0048511 ms |

    The micro-optimization is real in its intended scope: it removes both
    named modules and another roughly 5 us of launch/interval overhead. The
    new outer binary's unrelated 95.9 us executable regression dominates the
    absolute comparison, illustrating again why cross-compile endpoint
    attribution is unsafe. Milestone 77 subsequently removes the direct
    handoff after a fresh-Ray, fresh-cache endpoint recheck found no benefit;
    do not restore it without a same-binary/matched control method.

    Exact repair validation and profile commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-backbone-lm-sampler-direct-token-v3-fix
    export RAY_ADDRESS=10.13.232.26:6379
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-direct-token-v3
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-backbone-lm-sampler-v1
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v1/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh

    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/smoke-$K3_RUN_NAME \
      scripts/k3/curl_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat2 \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
      PROF_DIR=$PROF_DIR \
      scripts/k3/profile_isl1_osl1k.sh 10.13.232.26
    ```

    Reusing the failed server's vLLM runtime cache did not avoid all host AOT
    work: both profile and serving wrappers logged `a bytes-like object is
    required, not 'BundledAOTAutogradResult'`, regenerated their wrappers,
    and then loaded the cached TPU executable. Treat a saved XLA artifact and
    a reusable vLLM AOT wrapper as distinct cache layers.

    A leaf-timeline audit also rejects the tempting temperature-1 follow-up:
    the outer executable's final full-vocabulary `broadcast_divide_fusion`
    costs only 0.061 us in the representative step, while the fused
    Gumbel/argmax reduction costs 10.612 us. Hard-coding temperature 1 would
    narrow semantics for effectively no device gain. The remaining visible
    tail is about 4.15 us vocabulary RNG, 2.58 us RNG setup, and 20.74 us of
    inter-module intervals. Moving RNG into the compiled graph is the only
    remaining tail fusion with even single-digit-microsecond kernel leverage,
    but it must preserve the rank-synchronized generator stream and exact
    random-number-to-token mapping. A separate host-side hygiene experiment
    may launch the same binary with `--disable-log-stats` to see whether it
    removes the isolated 79/138 ms ITL stalls; do not project it as a stable
    TPOT or TPU-kernel win unless clean paired endpoint data supports that.

76. **Disabling periodic vLLM request statistics removes the observed large
    endpoint stalls, but does not lower the stable device/TPOT floor.** The
    repaired direct-token v3 server was restarted on the same Ray step with
    the same source, model flags, libtpu flags, cache roots, and canonical
    request, adding only `--disable-log-stats`. Two detailed endpoint runs
    completed all eight measured requests with exact one-token inputs and
    1,000-token outputs:

    | Run | Mean TPOT | Median TPOT | P99 TPOT | Worst individual ITL |
    | --- | ---: | ---: | ---: | ---: |
    | `direct-token-v3-disable-log-stats` | 15.1726597 ms | 15.1700076 ms | 15.1848790 ms | 26.6853 ms |
    | `direct-token-v3-disable-log-stats-repeat` | 15.1690104 ms | 15.1662968 ms | 15.1774271 ms | 26.4205 ms |

    This is much cleaner than the flag-on predecessor repeats that contained
    isolated 79.09 and 138.19 ms ITLs. It is useful host-side benchmark
    hygiene and should be used for future endpoint measurements. It is not a
    kernel optimization: central TPOT remains about 15.17 ms, consistent with
    the direct-token v3 executable floor, and no trace was needed to establish
    that the device critical path did not improve. Eight clean requests also
    do not prove that request-stat logging alone caused every rare stall;
    longer soak testing would be needed for a tail-SLO claim.

    Exact launch and benchmark commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-backbone-lm-sampler-direct-token-v3-disable-log-stats
    export RAY_ADDRESS=10.13.232.26:6379
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-direct-token-v3
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-backbone-lm-sampler-v1
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v1/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh --disable-log-stats

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    ```

    Cache caveat: this restart reused the prior cache paths and resolved the
    same graph/config cache directories, but vLLM nevertheless regenerated
    both profile and serving AOT wrappers and ran fresh TPU compilation on all
    ranks. Startup took roughly 12 minutes. Consequently the comparison is
    same-source/same-flags but not a zero-recompile matched binary experiment;
    do not attribute sub-0.1-ms central-latency movement to
    `--disable-log-stats`. The detailed JSONs contain four generated-text
    strings and empty errors in each run, but dummy weights make those strings
    random rather than quality evidence.

77. **Removed the direct-token handoff specialization after a clean recheck
    found no endpoint benefit.** Milestone 75 established that returning a
    1-D `int32` token and bypassing normal placeholder substitution saved
    15.141 us in the post-executable tail, but its newly compiled outer
    executable lost 95.851 us. Keeping a narrow, shape-sensitive runner path
    for an unaccepted result was therefore not justified. The outer sampler
    once again returns the normal `[1,1]` argmax tensor, async scheduling uses
    `view(-1)` and `_substitute_placeholder_token`, and the special direct
    reuse branch is gone. The independent 1-D-to-2-D normalization in
    `_modify_prev_results` remains as a harmless correctness hardening.

    This was validated on fresh exact Ray step `664.48`, after the source edit
    and with a unique cold cache namespace, all accepted libtpu flags,
    `TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1`, and `--disable-log-stats`.
    Compilation confirmed the restored outer HLO module
    `tt_jit_model_L332C0_forward_argmax` (70,922 initial instructions) and the
    placeholder-scatter program. Both detailed serving benchmarks completed
    four of four measured requests with no errors:

    | Run | Mean TPOT | Median TPOT | P99 TPOT | Worst individual ITL |
    | --- | ---: | ---: | ---: | ---: |
    | `v2-restored-disable-log-stats` | 15.1711375 ms | 15.1697700 ms | 15.1834833 ms | 23.2942 ms |
    | `v2-restored-disable-log-stats-repeat` | 15.1680589 ms | 15.1678781 ms | 15.1733049 ms | 21.2040 ms |

    Their 15.1695982 ms average is only 0.0012368 ms below the milestone-76
    logging-disabled v3 average (15.1708350 ms), far below attribution noise.
    This is a source-simplification/correctness decision, not a claimed
    performance win and not a new absolute best. It also reinforces that the
    earlier 15.1212043 ms v2 average versus 15.17 ms later binaries contains
    substantial cross-compile/session variance.

    Exact server and benchmark commands were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-backbone-lm-sampler-v2-restored-disable-log-stats
    export RAY_ADDRESS=10.13.232.26:6379
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-backbone-lm-sampler-v1
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v1/torch_tpu_tier3
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh --disable-log-stats

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    ```

    The two JSON files each contain `completed=4`, `failed=0`,
    `input_lens=[1,1,1,1]`, `output_lens=[1000,1000,1000,1000]`, four empty
    error strings, and four detokenized generated-text strings. As documented
    in milestone 74 and the runbook, this is real OpenAI-serving-path work
    over reproducible synthetic one-token content and dummy-shaped weights;
    the outputs are random and are not model-quality data. The probe and two
    warmups per invocation are real requests but are excluded from the saved
    four-request performance aggregate.

    A same-server profile then supplied the current detailed runtime
    reconciliation. Because `--disable-log-stats` removes the
    `vllm:iteration_tokens_total_count` metric, the normal
    `profile_isl1_osl1k.sh` helper completed its two warmups but exited before
    capture when its iteration-counter poll failed. The working method used
    the server's already configured `delay_iterations=3` and
    `max_iterations=15` directly:

    ```bash
    scripts/k3/start_profile.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_FILE=$PWD/results/profile-$K3_RUN_NAME/manual-capture-1-response.json \
      scripts/k3/curl_isl1_osl1k.sh 10.13.232.26
    scripts/k3/stop_profile.sh 10.13.232.26

    .venv/bin/python scripts/k3/analyze_trace_tail.py \
      --region 'tt_jit_model_L332C0_forward_argmax(1115740701171761723)' \
      --top-modules 40 \
      /mnt/scratch/kimi-k3-b1-backbone-lm-sampler-v2-restored-disable-log-stats-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_448724.1786674414491634297.pt.trace.json.gz
    ```

    The manual capture response again contains exactly one prompt token and
    1,000 completion tokens. Distributed `/stop_profile` flushing took about
    90 seconds; never use that request for endpoint latency. The trace has 80
    usable decode-to-decode gaps:

    | Current rank-0 interval | Mean | Median |
    | --- | ---: | ---: |
    | Outer executable | 14.966610 ms | 14.941079 ms |
    | Post-executable gap | 0.042732 ms | 0.042728 ms |
    | Decode start spacing | 15.009342 ms | 14.983568 ms |
    | Named device modules in the gap | 0.015346 ms | 0.015332 ms |
    | Uncovered gap interval | 0.027386 ms | 0.027378 ms |

    The gap's named means are 4.757 us token view, 4.277 us placeholder
    scatter, 3.962 us vocabulary RNG, and 2.351 us RNG setup. A separate
    opcode-based, exclusive aggregation of the primary XLA-op lane over 61
    fully leaf-covered stable modules gives:

    | In-executable leaf family | Self time/decode |
    | --- | ---: |
    | Fusion | 6.864951 ms |
    | All-reduce | 4.906397 ms |
    | Custom call | 2.004175 ms |
    | Other | 0.232389 ms |
    | Reshape | 0.181888 ms |
    | Copy | 0.130153 ms |
    | Copy-done | 0.052311 ms |
    | All-gather | 0.030696 ms |
    | Copy-start | 0.023394 ms |

    Those exclusive leaf opcodes sum to 14.426354 ms versus a 14.937575 ms
    enclosing-module median on the same leaf-covered subset. The 0.511 ms
    arithmetic difference is not a measured idle category: leaf self time,
    async/core interval coverage, and enclosing module wall time have
    different scopes. Together with the directly measured 0.0427 ms outer
    gap, the current trace again rules out both a 4 ms outside-model/input-prep
    remainder and a 1.3 ms hidden idle kernel. The dominant targets remain
    in-executable fusions, all-reduces, and custom calls. Explicit `copy`,
    `copy-start`, and `copy-done` opcodes total only 0.205858 ms on this lane;
    asynchronous copy activity can have a different lane/accounting scope.

    Correction: an earlier draft of this table classified raw HLO event names
    with substring tests. Many all-reduce and fusion event names embed operand
    text such as `%copy-done`, so that parser incorrectly reported 3.628891 ms
    as `Copy` and correspondingly understated fusion/custom compute. The total
    leaf time and the interval-level conclusion were unchanged; the opcode
    column above is the authoritative current kernel-family breakdown.

78. **Row-sharding the `7168 -> 896` MoE router gate at TP2 is implemented,
    hardware-valid, and rejected.** This tested the remaining intermediate
    projection suggested by the broader up/down-projection sharding idea. The
    implementation uses a `RowParallelLinear` over topology-local two-rank
    groups, performs one 896-element subgroup all-reduce per MoE layer, then
    casts the complete logits to FP32 before the existing exact Pallas top-k.
    Checkpoint parameter names remain `gate.weight` and
    `gate.e_score_correction_bias`. The opt-in controls are
    `TPU_K3_SHARD_ROUTER_GATE=1` and `TPU_K3_ROUTER_GATE_TP_SIZE=2`; launcher
    defaults remain zero because the result regressed.

    Focused construction/forward validation passed before hardware launch:

    ```bash
    .venv/bin/python -m py_compile \
      src/vllm_torchtpu/envs.py \
      src/vllm_torchtpu/models/vllm/kimi_k3/moe.py \
      tests/models/vllm/test_kimi_k3.py
    .venv/bin/pytest -q tests/models/vllm/test_kimi_k3.py \
      -k 'sharded_router_gate or sharded_latent_down or sharded_latent_up'
    # 6 passed, 27 deselected
    ```

    Fresh exact Ray step `664.49` imported the changed source and initialized
    all 32 TPU devices. The complete server and measured benchmark commands
    were:

    ```bash
    export K3_RUN_NAME=kimi-k3-b1-router-gate-tp2-v1
    export RAY_ADDRESS=10.13.232.26:6379
    export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
    export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
    export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
    export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
    TPU_K3_SHARD_ROUTER_GATE=1 \
    TPU_K3_ROUTER_GATE_TP_SIZE=2 \
    TPU_K3_FUSE_BACKBONE_LM_HEAD_SAMPLER=1 \
      scripts/k3/launch_kimi_k3_tp32_ep.sh --disable-log-stats

    OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
      scripts/k3/bench_isl1_osl1k.sh 10.13.232.26
    ```

    The detailed result completed all four requests with no errors and gives
    15.4423879892 ms mean TPOT, 15.4422537848 ms median TPOT,
    15.4467336855 ms P99 TPOT, 27.9053335 ms mean TTFT, and 23.9260670 ms
    worst individual ITL. Relative to the fresh milestone-77 flag-off pair's
    15.1695982096 ms average, this is a clear **+0.2727897800 ms/token
    regression**. A repeat was deliberately skipped: the loss is more than
    an order of magnitude larger than run noise and the trace below identifies
    the expected mechanism.

    The saved JSON is also direct evidence about benchmark data. It contains
    `completed=4`, `failed=0`, `total_input_tokens=4`,
    `total_output_tokens=4000`, `input_lens=[1,1,1,1]`,
    `output_lens=[1000,1000,1000,1000]`, four empty error strings, and four
    1,000-token detokenized output strings. The client generated each prompt
    as one reproducible random non-special prefix token (`dataset=random`,
    `seed=42`, `random_input_len=0`, `random_prefix_len=1`) and did not save
    the prompt text/token ID. These are real HTTP requests and real serving,
    model, collective, sampler, streaming, and detokenization operations, but
    `LOAD_FORMAT=dummy` makes the output strings random byte-like text. The
    run is valid performance data, not real-checkpoint correctness or quality
    data.

    The matched profile used the manual procedure required by
    `--disable-log-stats`:

    ```bash
    scripts/k3/start_profile.sh 10.13.232.26
    TEMPERATURE=1 INCLUDE_SEED=0 \
      OUTPUT_FILE=$PWD/results/profile-$K3_RUN_NAME/manual-capture-1-response.json \
      scripts/k3/curl_isl1_osl1k.sh 10.13.232.26
    scripts/k3/stop_profile.sh 10.13.232.26

    .venv/bin/python scripts/k3/analyze_trace_tail.py \
      --region 'tt_jit_model_L332C0_forward_argmax(3774355073135375805)' \
      --top-modules 40 \
      /mnt/scratch/kimi-k3-b1-router-gate-tp2-v1-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_553481.1786676821518314463.pt.trace.json.gz

    .venv/bin/python scripts/k3/analyze_trace_kernels.py --show-counts \
      --region 'tt_jit_model_L332C0_forward_argmax(3774355073135375805)' \
      /mnt/scratch/kimi-k3-b1-router-gate-tp2-v1-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_553481.1786676821518314463.pt.trace.json.gz
    ```

    The candidate trace has 80 usable gaps: its outer executable is
    15.234860 ms mean / 15.242764 ms median, its post-executable tail remains
    0.042723 ms mean, and decode-start spacing is 15.277583 ms mean. Thus the
    regression is wholly inside the model executable, not input preparation
    or the runtime tail. A same opcode-based event-self-time audit against the
    milestone-77 trace shows the cause:

    | Primary XLA-op family | Flag off | Router TP2 | Delta |
    | --- | ---: | ---: | ---: |
    | All-reduce | 4.930564 ms | 5.283180 ms | +0.352616 ms |
    | Fusion | 6.901265 ms | 6.929807 ms | +0.028542 ms |
    | Custom call | 2.014523 ms | 1.967932 ms | -0.046590 ms |
    | Reshape | 0.183012 ms | 0.161336 ms | -0.021677 ms |
    | Explicit copy/start/done | 0.207490 ms | 0.205817 ms | -0.001673 ms |
    | All-gather | 0.031209 ms | 0.035018 ms | +0.003809 ms |
    | Other opcodes | 0.233826 ms | 0.225196 ms | -0.008630 ms |
    | Total attributed event self time | 14.501890 ms | 14.808286 ms | +0.306396 ms |

    The all-reduce event count rises exactly from 348 to 440 per decode: one
    new 896-element reduction in each of 92 MoE layers. The approximately
    0.353 ms collective increase overwhelms the saved local router matmul.
    Therefore TP4 is not justified, and this gate should remain replicated.
    This also sharpens the projection boundary: sharding 7168-to-128 KDA can
    win, 7168-to-2112 MLA is near neutral, but even 7168-to-896 router compute
    cannot pay for 92 new serialized subgroup collectives.

    `analyze_trace_kernels.py` is now the reproducible opcode parser used for
    pairwise audits. It extracts the assigned HLO opcode after ` = ` and never
    classifies operand substrings such as `%copy-done`; it also subtracts
    nested child scopes when reporting event self time. Absolute totals remain
    self-time accounting rather than wall time, so use
    `analyze_trace_tail.py` for interval/idle questions. Real-checkpoint
    numerical equivalence remains untested; the opt-in path reduces BF16
    partial logits before FP32 conversion and may have different rounding.

## Forward plan

1. Preserve both accepted exact routing optimizations in the K3 launcher. The
   896-way selector and 16-route GMM permutation now contain zero TPU sort
   operations; do not reintroduce a general sort into either batch-1 path.
2. Keep the MLJ-inspired `7 x 128` blockwise selector disabled. It preserves
   the successful score/bias/selection/normalization fusion and exact order,
   but its router family costs 0.30741 ms/decode versus 0.27816 ms for the
   accepted full-width kernel. Revisit only with a TPU-native bitonic/merge
   primitive; avoid cross-vreg dynamic gather, which Mosaic rejects.
   The Inkling iterative selector and MLJ streaming/block-merge references
   have both already been ported and measured; neither remains an unexplored
   top-k suggestion.
3. Treat TP4 as the measured optimum for MLA's fused A projection: TP2 was
   neutral, TP4 won about 0.086 ms pairwise, and TP8 regressed about 0.091 ms
   against its fresh TP4 control. Do not widen this collective further.
4. Use the measured projection boundary to prioritize transformations that
    reuse or fuse an existing collective: 7168-to-128 KDA sharding is slightly
    positive, 7168-to-2112 MLA sharding is neutral, and milestone 78 proves
    that 7168-to-896 router sharding loses 0.273 ms/token by adding 92 new
    reductions. The 179 independent scalar attention-residual reductions are
    likewise collectively expensive despite minimal payload. Avoid adding
    per-layer collectives solely for replicated scorers; a remaining linear
    shard needs to reuse or fuse a reduction already on the critical path.
5. Preserve the now-enabled dependent-collective layer scheduler, multi-compute
   layer-scheduler overlap, async collective merger, and bidirectional
   all-reduce decomposition. The matched milestone-71 result attributes a
   further 0.07453 ms/token win to multi-compute overlap, almost entirely in
   the model executable. The merger
   reduces the collective critical path by 0.06736 ms, and bidirectional
   decomposition removes another 0.08240 ms of all-reduce time without
   changing HLO-visible counts. Async collective fusion is rejected and must
   remain off because it regressed and carries an explicit non-Viperlite
   safety warning. TensorCore compute/collective overlap is also rejected
   because it adds 0.05270 ms to primary collective duration. Further compiler
   work must target a different measured mechanism and retain explicit control
   runs. Do not try the installed inference-short all-reduce chunk flag on
   v7x-32: milestone 59 proves its emitter is gated to exactly eight logical
   devices and an eight-device collective.
6. Use the current-best profile to prioritize collective reuse/fusion or
   dense projections with enough work to beat the now-bounded subgroup cost;
   TP8 latent-down, TP4 latent-up when it breaks variadic reduction fusion,
   per-layer scorer collectives, and explicit local scorer reductions are
   rejected. Another model-only reordering is low priority: adjacent ordinary
   reductions remained separate, and functional `all_reduce_coalesced` is not
   supported by `ProcessGroupTPU`. A future up-projection attempt is justified
   only after adding a supported mixed-shape variadic collective or fusing the
   latent reduction, norm, projection, and shared combine more deeply.
7. Keep the current routed-GMM tiles. The exact 56-group sweep found only an
   approximately 0.02 ms/decode GMM1-only opportunity, while splitting GMM2
   regressed at four or more active local routes. Revisit only with a static
   local-occupancy bound or a new full-width/pipelined kernel design.
8. Treat the original 0.13608 ms post-backbone tail as mostly device work,
   not runtime idle. Milestone 74's outer backbone/LM-head/sampler graph cuts
   the matched tail to 0.04274 ms and wins 0.09403 ms/token end to end; retain
   it as an opt-in canonical temperature-1 path because its request contract
   is deliberately narrow. Milestone 75's direct token handoff cut the tail
   to 0.02760 ms and removed both view/scatter modules, but its fresh outer
   binary was 0.09585 ms slower; milestone 77 removes that unaccepted
   specialization after a clean recheck found no endpoint benefit.
   Hard-coding temperature 1 is statically rejected because the traced divide
   costs only 0.061 us. The only remaining tail fusion is moving the roughly
   6.7 us RNG work into the executable with exact generator semantics; larger
   gains must instead improve the backbone or avoid/shrink the full-vocabulary
   TP32 all-gather.
9. Preserve the broadly applicable best configuration as the launcher default
   and keep the narrow whole-step path opt-in. The combined matched deltas now
   project 14.95023 ms, but no absolute whole-step binary has verified that
   number. The next acceptance gate is an absolute clean run at or below
   15 ms with all intended flags present before Ray startup; once reached,
   repeat it with at least five measured requests and no long-ITL outlier.
10. Treat sampling as bounded: the exact-server temperature-1 versus greedy
   delta is only 0.006 ms. Revisit it only if a future trace shows a stable
   endpoint delta; do not adopt inverse-CDF sampling without both a measurable
   budget and exact mixed-parameter/deterministic-generator semantics.
11. Keep the re-tested PR #410 fused MoE combine disabled. Although its flat
   `[1,10752]` TP32 reduction saves 0.08876 ms/decode versus the variadic
   tuple, other all-reduce families regress by more: total all-reduce time
   rises 0.10829 ms, stable executable time rises 0.07116 ms, and the endpoint
   pair rises 0.07875 ms. Revisit only if a compiler change removes the TP4
   `[3584]` scheduling regression; milestone 61 is the required control.
12. Do not re-enable `keep_inference_input_mutations` for KDA state. Milestone
   63 proves it doubles TPOT. If runtime mutation replay cannot be batched
   without copies, return to the device critical path: the remaining credible
    opportunities are fusing/reusing an existing collective, or a projection
    shard whose compute saving exceeds its already measured subgroup cost.
13. Do not recreate a global AOT mutation-epilogue `copy_` to `set_` textual
    rewrite. Milestone 64 shows it reaches 162 mutation inputs rather than the
    23 visible tail calls, adds about 0.219 ms/decode of host `set_`
    dispatch, leaves 21 `copy_` calls/decode, and regresses endpoint TPOT by
    0.102 ms. Milestone 65 proves there is no isolated 23-buffer KDA family in
    the gap. A future runtime attempt must batch the complete replay while
    retaining copy semantics; per-tensor Python/ATen dispatch cannot fit the
    remaining 0.119 ms budget.
14. Do not replace the mutation epilogue with `aten::_foreach_copy_` on this
    TorchTPU build. Milestone 65 shows its composite implementation costs about
    0.218 ms/decode and leaves the 0.136 ms inter-executable gap unchanged.
    The remaining gap is therefore not removable merely by reducing Python
    call count; a future runtime attempt needs native asynchronous batching or
    a dependency change, not the generic foreach composite.
15. Keep the distributed neutral sampler disabled. Milestone 70 repairs the
    first-request deadlock by placing the pair all-gather at an explicit
    process-group rendezvous between collective-free compiled graphs, but its
    matched endpoint pair regresses by 0.01170 ms/token and its post-backbone
    tail grows by about 12 us. Do not try another sampler-only split. The only
    remaining version of this idea with a credible cost model would place the
    local head, pair reduction, and final selection in the same already-needed
    backbone executable, analogous to `tpu-inference`'s whole-loop JAX design;
    that requires a full runner/state compilation redesign, not another
    wrapper. Preserve full RNG advancement and do not generalize to top-k/top-p,
    penalties, grammar, or logprobs without exact semantics.

## Current best and gap

The best strict absolute steady-state result remains 15.11826--15.11931 ms
from the fused
exact Pallas router plus the exact rank-16 route permutation, hierarchical
routed experts at EP16 x TP2, batch-1 direct routed-row combine, independent TP4
latent `down_proj`, TP2 KDA `f_a_proj` sharding, dependent-collective layer
scheduling, async collective merging,
bidirectional all-reduce decomposition, TP4 MLA fused A-projection sharding,
plus the neutral top-k/top-p sampling specialization. Milestone 71 additionally
accepts multi-compute layer-scheduler overlap from a matched -0.07453 ms/token
comparison. Milestone 74 measures a further -0.09403 ms/token matched delta
for the opt-in whole-step graph on top of that flag set. Arithmetically
stacking both matched deltas with the strict 15.11879 ms pair projects about
14.95023 ms, but this is not a verified absolute result: the separately
compiled whole-step binaries measured 15.12120 ms (`v2`) and 15.17444 ms
(clean direct-token `v3`) in their own sessions. The strict verified gap
therefore remains 0.119 ms even though the delta projection crosses the goal.
Enhanced launch barriers are not part of any effective configuration because
TorchTPU forces them off in distributed actors. No speculative decoding is
used in any result.
