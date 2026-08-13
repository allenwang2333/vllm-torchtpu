# Kimi-K3 runbook

Run these commands from the repository root. Replace `<SERVER_IP>` with the IP
of the host serving the OpenAI-compatible endpoint.

## Optimization handoff (2026-08-13)

The active objective is batch-1 TPOT at or below 15 ms on one TPU v7x-32
slice, without speculative decoding. All performance runs use dummy weights,
`compile_sizes=[1]`, ISL=1, OSL=1000, and concurrency 1. Read
`OPTIMIZATION_LOG.md` before selecting another experiment; it records rejected
ideas as well as results. The best repeated configuration currently measures
15.11826--15.11931 ms mean TPOT. It is backed up on the non-main branch
`agent/k3-b1-tpot-optimization-backup`.

The launcher defaults contain the current winning model configuration:

- hierarchical MoE EP16 x TP2;
- exact fused Pallas router top-k for the batch-1 K3 shape;
- exact stable rank-16 route permutation with no downstream TPU sort;
- direct batch-1 routed-row combine;
- TP4 latent-MoE down projection;
- TP2 KDA `f_a` projection;
- TP4 MLA fused query/KV A projection;
- neutral top-k/top-p sampler specialization;
- enhanced TPU launch barriers;
- XLA dependent-collective layer scheduling;
- XLA asynchronous collective merging; and
- bidirectional all-reduce decomposition.

Do not enable opt-in experimental flags just because their implementation is
present. Many are retained for reproducibility after measuring neutral or
slower. The optimization log is authoritative for their disposition.

The accepted Pallas router is the launcher default. Set
`TPU_K3_PALLAS_ROUTER_TOPK=0` only to obtain a control. It fuses sigmoid,
correction bias, exact top-16 selection, selected-score collection, and
renormalization. The rank-16 route permutation is also a launcher default;
set `TPU_K3_RANK16_ROUTE_PERMUTE=0` only for its control. Together their latest
two acceptance runs average 15.27388 ms mean TPOT, and the paired profile has
zero TPU sort operations per decode. The MLJ-inspired two-stage selector is
reproduced with `TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=1`. It passed exact CPU
interpreter tests and served successfully, but its traced router cost
regressed from 0.27816 to 0.30741 ms/decode. Leave it zero; its superficially
faster endpoint pair was collective/session variance rather than a router
improvement. The
older exact TPU/JAX router experiment is reproduced by adding
`TPU_K3_JAX_EXACT_TOPK=1` before launch. It compiles and is exact, but its two
runs averaged 0.015 ms slower than the launcher default, so leave it unset.
After any source change, verify that worker PIDs are newer than the edit or,
preferably, restart the exact Ray Slurm step. A new cache path alone does not
refresh Python imports in persistent Ray workers.

### Resume an existing four-host Slurm allocation

The commands below show the exact manual workflow used during optimization.
Set the job ID and head IP from the allocation rather than copying the example
values. Resolve hosts and the head IP first:

```bash
export K3_JOB_ID="<allocation-job-id>"
mapfile -t K3_HOSTS < <(scontrol show hostnames "$(squeue -j "$K3_JOB_ID" -h -o %N)")
export K3_HEAD_HOST="${K3_HOSTS[0]}"
export K3_HEAD_IP="$(srun --jobid="$K3_JOB_ID" --overlap -N1 -n1 -w "$K3_HEAD_HOST" hostname -I | awk '{print $1}')"
export K3_RAY_PORT=6379
```

Start a fresh Ray step in its own terminal. Fresh workers are required after
every source-code change or `LIBTPU_INIT_ARGS` change; restarting only vLLM
can silently reuse actors with old imports or an old libtpu runtime.

```bash
export K3_REPO_DIR="$PWD"
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

Verify that all 32 TPU resources joined before launching a server:

```bash
RAY_ADDRESS="$K3_HEAD_IP:$K3_RAY_PORT" .venv/bin/python - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
resources = ray.cluster_resources()
print(resources)
assert int(resources.get("TPU", 0)) == 32
PY
```

Start the current default server in another terminal with a unique cache and
profile namespace. Reusing cache paths across experiments can make attribution
ambiguous; changing libtpu flags requires both fresh workers and fresh caches.

```bash
export K3_RUN_NAME="kimi-k3-b1-<experiment>"
export RAY_ADDRESS="$K3_HEAD_IP:$K3_RAY_PORT"
export HF_HOME=/mnt/scratch/hf-hub
export HF_HUB_CACHE=/mnt/scratch/hf-hub/hub
export HF_MODULES_CACHE=/mnt/scratch/hf-hub/modules
export HF_HUB_OFFLINE=1
export VLLM_XLA_CACHE_PATH="/mnt/scratch/$K3_RUN_NAME"
export VLLM_CACHE_ROOT="$VLLM_XLA_CACHE_PATH/vllm-runtime"
export VLLM_CONFIG_ROOT="$VLLM_XLA_CACHE_PATH/vllm-config"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${USER}_${K3_RUN_NAME}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="$VLLM_XLA_CACHE_PATH/torch_tpu_tier3"
export PROF_DIR="/mnt/scratch/${K3_RUN_NAME}-profile"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false"
scripts/k3/launch_kimi_k3_tp32_ep.sh
```

Set model experiment overrides on the launcher invocation, not on the Ray
step. For example, the rejected TP8 MLA A-projection boundary was launched as:

```bash
TPU_K3_MLA_A_TP_SIZE=8 \
  scripts/k3/launch_kimi_k3_tp32_ep.sh
```

The rejected intermediate latent-up shard is reproduced with:

```bash
TPU_K3_SHARD_LATENT_UP_PROJ=1 TPU_K3_LATENT_UP_TP_SIZE=4 \
  scripts/k3/launch_kimi_k3_tp32_ep.sh
```

Do not substitute PyTorch's functional `all_reduce_coalesced` in this path.
The installed `tpu_dist` process group rejects it during FX-to-MLIR export
with `Backend tpu_dist does not support allreduce_coalesced`; the server never
reaches compilation or health. Putting two ordinary all-reduces next to one
another was also measured and did not make XLA variadically combine them.

Unset the override on the next launch to recover the accepted launcher default
of TP4 MLA A sharding with latent-up disabled. A model-only environment
override needs a new vLLM server and unique caches, but not a new Ray step when
neither source nor `LIBTPU_INIT_ARGS` changed.

The launcher appends the winning enhanced-barrier, dependent-collective
scheduler, async-collective-merger, and bidirectional-all-reduce flags unless
the caller explicitly supplies a setting with `=true` or `=false`. For an
isolated compiler experiment, append only that experiment's flag to
`LIBTPU_INIT_ARGS`; do not bundle unrelated tuning flags.

Wait for health before benchmarking:

```bash
until curl --fail --silent "http://$K3_HEAD_IP:8000/health" >/dev/null; do
  sleep 10
done
```

Run the canonical benchmark twice when the delta is near measurement noise:

```bash
OUTPUT_DIR="$PWD/results/$K3_RUN_NAME" \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
OUTPUT_DIR="$PWD/results/${K3_RUN_NAME}-repeat" \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
```

`bench_isl1_osl1k.sh` intentionally leaves temperature unset. With the current
server this resolves to temperature-1 sampling, not greedy decoding. Use the
exact request helper for controls:

```bash
TEMPERATURE=0 INCLUDE_SEED=1 \
  OUTPUT_DIR="$PWD/results/${K3_RUN_NAME}-greedy" \
  scripts/k3/curl_isl1_osl1k.sh "$K3_HEAD_IP"
TEMPERATURE=1 INCLUDE_SEED=0 \
  OUTPUT_DIR="$PWD/results/${K3_RUN_NAME}-sampled" \
  scripts/k3/curl_isl1_osl1k.sh "$K3_HEAD_IP"
```

Seeded non-greedy completion requests have returned HTTP 500 in this setup;
use `INCLUDE_SEED=0` for temperature-1 profiling. This is an API correctness
issue separate from the steady-state TPOT benchmark.

Capture two post-warmup, 15-iteration temperature-1 profile windows:

```bash
TEMPERATURE=1 INCLUDE_SEED=0 \
  OUTPUT_DIR="$PWD/results/profile-$K3_RUN_NAME" \
  PROF_DIR="$PROF_DIR" \
  scripts/k3/profile_isl1_osl1k.sh "$K3_HEAD_IP"
```

Run GSM8K only when correctness/quality validation is needed. The default
server is configured for batch 1, so keep evaluation concurrency at one unless
the server is intentionally relaunched with larger sequence/token limits:

```bash
NUM_CONCURRENT=1 OUTPUT_DIR="$PWD/results/${K3_RUN_NAME}-gsm8k" \
  scripts/k3/eval_gsm8k.sh "$K3_HEAD_IP"
```

### Safe shutdown and experiment boundaries

Stop the vLLM PID with SIGINT, wait for it to exit, then cancel only the exact
Ray step (for example `602.41`), never the parent allocation job. Resolve the
step with `squeue -s -j "$K3_JOB_ID"`; do not guess it. Preserve the allocation
while iterating.

```bash
kill -INT "<vllm-pid>"
scancel "<exact-ray-step-id>"
```

Do not run broad TPU-importing pytest suites while the distributed server is
initializing: a head-node test process previously acquired libtpu devices and
caused `/dev/vfio/*` device-busy failures. Focused static tests are safe only
when they do not initialize the TPU plugin.

### Profiling and result caveats

- The Slurm TPU metadata endpoint returns HTTP 404. This is expected here;
  the launcher explicitly exports `TPU_ACCELERATOR_TYPE=tpu7x-32`. Confirm the
  subsequent TPU-info log reports `tpu7x-32` rather than treating the 404 as a
  failed launch.
- Cold starts can take more than ten minutes. Check the frontend, EngineCore,
  all 32 `RayWorkerProc` actors, and compilation logs before declaring a hang.
- The Torch profiler can emit invalid JSON because quotes inside XLA event
  names such as `custom_call_target="X64SplitHigh"` are not escaped. It can
  also emit one malformed gRPC `event_args` value. Repair only those string
  fields in a streaming parser; do not drop device events.
- Stopping a distributed Torch profile can take roughly 30--60 seconds after
  the requested 15 iterations while all ranks merge and flush traces. During
  that time the in-flight completion can stall. Do not infer its endpoint
  TPOT from that profiled request; use the separate canonical benchmark.
- Do not add durations from the two TPU-core lanes or nested async-copy events
  as wall time. Use interval unions for occupancy and one consistent XLA-op
  lane for paired family comparisons. The former 1.387 ms "idle" residual was
  invalid arithmetic across incompatible scopes; normal uncovered device-lane
  time is only about 0.067 ms.
- `results/` and `uv.lock` are intentionally untracked. Commit source and
  documentation only unless explicitly asked to version raw benchmark data.
- Push only `agent/k3-b1-tpot-optimization-backup`, never `main`. If the SSH
  agent is unavailable, `gh auth setup-git` followed by an HTTPS push is the
  working fallback.

### Live state at this handoff

Allocation 602 expired at its 24-hour limit. Replacement allocation 664 is
running on `tpu-rack1-w-[001-004]` through 2026-08-14 16:00:28 UTC, with head
IP `10.13.232.26`. A four-host Ray cluster is live in overlapping step
`664.8`, with temporary directory `/tmp/ray-k3-job664-mut`. No vLLM server is
currently running. Those workers imported the now-removed input-mutation
diagnostic, so they are stale: cancel exactly `664.8`, start fresh Ray workers,
and verify all 32 TPU resources before the next server. Preserve parent
allocation 664. The rejected diagnostic used cache
`/mnt/scratch/kimi-k3-b1-keep-input-mutations` and profile namespace with the
same name plus `-profile`.

Current status checks are:

```bash
timeout 12s squeue -j 664 -o '%.18i %.9T %.10M %.6D %R'
timeout 12s squeue -s -j 664
curl --max-time 3 -sS -o /dev/null -w '%{http_code}\n' \
  http://10.13.232.26:8000/health
ps -eo pid,ppid,stat,etime,args | \
  rg 'ray start|vllm serve|api_server|EngineCore|RayWorkerProc'
```

Do not repeat a broad process kill. If stale workers recur, resolve the exact
new step/PIDs first. The bidirectional trial cache remains
`/mnt/scratch/kimi-k3-b1-bidirectional-allreduce` and its profile directory is
`/mnt/scratch/kimi-k3-b1-bidirectional-allreduce-profile`.

The router's CPU-interpreter pytest command has a current environment caveat:
two handoff-time invocations hung during import/initialization despite
`JAX_PLATFORMS=cpu TPU_ACCELERATOR_TYPE=` and were stopped by their exact PIDs.
They produced no test result. Do not launch parallel retries; milestone 51
retains the prior six-case pass and describes the measured full-model evidence.

The accepted result files are under
`results/kimi-k3-b1-rank16-route-permute{,-repeat}` and measure 15.27481 and
15.27295 ms mean TPOT. The two profile requests are under
`results/profile-kimi-k3-b1-rank16-route-permute`. The latest rank-0 trace
is
`/mnt/scratch/kimi-k3-b1-rank16-route-permute-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2503254.1786624163924546668.pt.trace.json.gz`.
It contains 18 stable decode executables averaging 14.98202 ms, 92 fused
router calls totaling 0.27816 ms/decode, and zero sort operations of any shape.
Primary plus asynchronous device work covers 14.91656 ms, leaving only
0.06545 ms uncovered.

The last experimental server's endpoint result files are under
`results/kimi-k3-b1-pallas-block-router-topk{,-repeat}` and measure 15.24950
and 15.25911 ms mean TPOT. Its latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-pallas-block-router-topk-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2599376.1786626871906653164.pt.trace.json.gz`.
The stable executable average is 14.97524 ms, but the block-router family is
0.30741 ms/decode and therefore slower than the accepted router. Restore the
accepted configuration by starting fresh Ray workers after source edits and
launching with the default `TPU_K3_PALLAS_BLOCK_ROUTER_TOPK=0`.

The accepted trace's 1,729 timed router/GMM events are 18 complete 92-layer
decodes plus one 73-layer profile-boundary fragment. Do not divide them by 19.
Mean call time times 92 gives 0.27849 ms router, 0.79947 ms GMM1, and 0.59325
ms GMM2 per complete decode. The exact current 56-group tile sweep and its
four-host `srun` command are milestone 52 in `OPTIMIZATION_LOG.md`. Its
low-occupancy split candidate was rejected because it regressed from four
active local routes onward; no production GMM tile changed.

For the same accepted profile, stable model executables average 14.98202 ms,
start 15.11888 ms apart, and leave a 0.13740 ms mean end-to-next-start launch
gap. The accepted endpoint pair averages 15.27388 ms, so the approximate
endpoint-minus-executable envelope is 0.29187 ms: 0.13740 ms between device
executables plus about 0.15500 ms from device start spacing to client TPOT.
These are separate profile and benchmark requests, so the last component also
contains run variance and client/streaming accounting. It is not a four-ms
input-preparation cost. See milestones 50 and 54 for the complete device-lane
scope rules.

The current `vllm-project/tpu-inference` audit is milestone 53. Its recent GMM
scale-load, lane-interleave, and bucketized-M improvements are already present;
its exact `approx_max_k` path is the already rejected K3 JAX experiment; and
its fused MoE kernel cannot accept one token under EP16 or lossless block-32
MXFP4. No upstream code was ported from that audit.

The most recent rejected full-model trial is milestone 55. Its endpoint pair
is under `results/kimi-k3-b1-async-collective-fusion{,-repeat}` and measures
15.27772 and 15.27654 ms mean TPOT. Its latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-async-collective-fusion-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2731456.1786629679643870728.pt.trace.json.gz`.
Stable executable time is 14.98988 ms and primary-lane collective time is
5.24140 ms, both slightly slower than the accepted profile. Do not enable
`xla_tpu_enable_async_collective_fusion`: besides being non-improving, the
installed libtpu warns that it is unsupported/not recommended off Viperlite
and could cause incorrect results, hangs, or SDC. It also emits extremely
large compiler logs; use `/health`, process state, and per-rank artifact saves
rather than log volume to judge cold-start progress.

The previous accepted full-model trial is milestone 56. Its endpoint results under
`results/kimi-k3-b1-async-collective-merger{,-repeat}` are 15.20766 and
15.21559 ms mean TPOT (15.21162 ms pair average). Its latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-async-collective-merger-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2812978.1786631597999023926.pt.trace.json.gz`.
Stable executable time is 14.91814 ms, primary-lane collective time is
5.16166 ms, and the primary-plus-async device union is 14.85122 ms. The flag
does not reduce the 623 HLO-visible matched collective events, but it shortens
their backend critical path by 0.06736 ms/decode. It is now a launcher default;
set `--xla_tpu_enable_async_collective_merger=false` explicitly in
`LIBTPU_INIT_ARGS` only for a control.

The immediately preceding experiment is the rejected TensorCore collective-overlap
trial in milestone 57. Its sole endpoint result under
`results/kimi-k3-b1-overlap-compute-collective-tc` is 15.25797 ms mean TPOT.
Its latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-overlap-compute-collective-tc-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2888689.1786633296525064381.pt.trace.json.gz`.
Stable executable time rose to 14.97352 ms and primary-lane collective time to
5.21436 ms, so leave `xla_tpu_overlap_compute_collective_tc` disabled. Only
one endpoint run was taken because the trace causally confirmed the
regression; this result must not replace the two-run accepted merger pair.

The current best trial is milestone 58. Endpoint results under
`results/kimi-k3-b1-bidirectional-allreduce{,-repeat}` are 15.11826 and
15.11931 ms mean TPOT (15.11879 ms pair average). Its latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-bidirectional-allreduce-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_2952330.1786635017233596519.pt.trace.json.gz`.
Stable executable time is 14.83475 ms. All-reduce count remains 348/decode,
but duration falls from 5.03057 to 4.94817 ms; the 92 `bf16[3584]` TP4
reductions alone improve by 0.09891 ms. The launcher now enables
`xla_tpu_decompose_all_reduce_bidirectional_communication`; set it explicitly
to `false` in `LIBTPU_INIT_ARGS` only for a control.

The next collective idea was statically closed without a full-model launch.
Installed libtpu's inference-short all-reduce emitter requires exactly eight
logical TPU devices and an eight-device collective; this v7x-32 graph has 32
logical devices and replica groups of 32, four, or two. Leave
`xla_tpu_inference_short_all_reduce_max_tensor_chunks=0`. Milestone 59 records
the `readelf`, `objdump`, and flagfile commands and the upgrade/address caveat.

The exact PR #410 fused MoE combine interaction is complete and rejected. The
patch remains present but defaults off. Its prerequisite collective-layout
screen can be reproduced with:

```bash
export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_async_collective_merger=true --xla_tpu_decompose_all_reduce_bidirectional_communication=true'
srun --jobid="$K3_JOB_ID" --overlap -N4 -n4 --ntasks-per-node=1 \
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

The completed job-664 probe printed `hlo_matches_expected=true` for both paths.
Both selected `RotatedPincerShortEmitter`/`StrategyND`; medians were 19.331 us
for the variadic tuple and 19.429 us for flat, a projected +0.0089 ms/decode
for flat across 92 layers. Never use `--skip-hlo-validation` for an acceptance
screen. The small negative result still required endpoint and trace validation
because PR #410 changes more than collective layout. The full-model candidate
launch and evaluation commands used were:

```bash
export K3_RUN_NAME=kimi-k3-b1-pr410-fused-combine-bidirectional
export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
TPU_K3_FUSED_MOE_COMBINE=1 scripts/k3/launch_kimi_k3_tp32_ep.sh

OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
TEMPERATURE=1 INCLUDE_SEED=0 \
  OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
  PROF_DIR=$PROF_DIR scripts/k3/profile_isl1_osl1k.sh "$K3_HEAD_IP"
```

The reason to revisit a previously neutral flag is narrow: the accepted
bidirectional trace's compiler-created variadic TP32 tuple regressed by
0.01834 ms, while the explicit fused path exposes one flat `[1,10752]`
operand. Compare all-reduce shape-family time as well as endpoint TPOT. Restore
`TPU_K3_FUSED_MOE_COMBINE=0` afterward unless the result is causally positive.

The result was causally negative: the endpoint means were 15.19703 and
15.19804 ms, and the stable executable mean was 14.90591 ms. The flat TP32
family improved by 0.08876 ms/decode, but total all-reduce time regressed by
0.10829 ms because the separate `[3584]` and `[128]` families slowed down.
Milestone 61 has the exact breakdown and latest trace path. Keep the default
zero.

The next runtime diagnostic tested whether KDA input-state writebacks could be
kept inside the compiled executable. It is rejected and its temporary source
was removed, but these are the exact launch and measurement commands for the
record. They assume the standard cache/profile exports above plus fresh Ray
workers:

```bash
export K3_RUN_NAME=kimi-k3-b1-keep-input-mutations
export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=${USER}_${K3_RUN_NAME}
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
TPU_KEEP_INFERENCE_INPUT_MUTATIONS=1 \
  scripts/k3/launch_kimi_k3_tp32_ep.sh

NUM_PROMPTS=1 NUM_WARMUPS=0 \
  OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-one \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
```

The single 1,000-token result was 29.99 ms mean TPOT. A focused trace was
captured by starting one `curl_isl1_osl1k.sh` request, waiting for three engine
iterations, calling `start_profile.sh`, waiting for 15 more iterations, and
calling `stop_profile.sh`; this avoids the canonical profiler helper's two
warmups and second capture when a candidate is already decisively negative.
The rank-0 trace is
`/mnt/scratch/kimi-k3-b1-keep-input-mutations-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3169093.1786640594153779741.pt.trace.json.gz`.
It shows 17 stable 29.726 ms executables and 24 full-state HLO copies totaling
14.6079 ms. The accepted trace's 0.13437 ms gap instead has 23 host
`aten::copy_` mutation replays. Do not recreate this flag: retaining mutations
in XLA trades approximately 0.134 ms of host dispatch for approximately 14.6
ms of device copies and transiently stressed program memory. Future work must
batch or alias the runtime writebacks without materializing tensor copies.

The follow-up runtime-storage-rebind diagnostic is also rejected and its
temporary source was removed. It replaced generated mutation-epilogue
`copy_` calls with `set_` only while the compiled callable performed its lazy
first invocation. The exact default-off diagnostic is archived in commit
`0e96e808`; the branch tip removes it again in commit `80be4ba3`. Reproduce
the historical launch from a detached worktree at `0e96e808`, on fresh Ray
workers, with:

```bash
export TPU_K3_RUNTIME_STATE_REBIND=1
export K3_RUN_NAME=kimi-k3-b1-runtime-state-rebind-v2
export VLLM_XLA_CACHE_PATH=/mnt/scratch/kimi-k3-b1-bidirectional-allreduce
export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-bidirectional-allreduce
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/torch_tpu_tier3
export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
scripts/k3/launch_kimi_k3_tp32_ep.sh

OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
  scripts/k3/bench_isl1_osl1k.sh "$K3_HEAD_IP"
TEMPERATURE=1 INCLUDE_SEED=0 \
  OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
  PROF_DIR=$PROF_DIR scripts/k3/profile_isl1_osl1k.sh "$K3_HEAD_IP"
```

The focused pre-launch check was:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q \
  tests/compilation/test_tpu_compiler.py -k runtime_state_rebind
```

It passed 2/2, but the full model logged 324 generated branch sites and the
trace measured 163 `aten::set_` calls/decode. The 15.22068 ms endpoint mean
regressed by 0.10190 ms; do not add this environment variable to the launcher.
The latest trace is
`/mnt/scratch/kimi-k3-b1-runtime-state-rebind-v2-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3357610.1786644450540916711.pt.trace.json.gz`.

Caveats for any successor:

- AOT's mutation epilogue is generated lazily on the compiled callable's
  first invocation. A context around compiler/wrapper construction silently
  misses it; require an explicit rewrite-count log before benchmarking.
- Existing serialized vLLM AOT entries may fail to load with
  `a bytes-like object is required, not 'BundledAOTAutogradResult'` after a
  compiler-source change. vLLM then recompiles; this is startup/cache-format
  behavior, not steady-state TPOT.
- Torch traces contain unescaped quotes inside some HLO names, so generic JSON
  parsers can fail. Use the line-oriented streaming parser described in the
  optimization log, and compare the same rank, capture position, category,
  and outlier rule.
- A source or actor-environment change requires fresh Ray workers. Cancel only
  the exact Ray/server Slurm step, never the parent allocation.

The shape audit that resolved what those mutations are is archived as
default-off source in commit `75922a6f`; the clean branch tip removes it in
commit `63bae361`. Reproduce from a detached worktree at the archive commit.
Its focused check and exact successful launch were:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q \
  tests/compilation/test_tpu_compiler.py -k runtime_state_mutation_audit

export TPU_K3_RUNTIME_STATE_AUDIT=1
export K3_RUN_NAME=kimi-k3-b1-runtime-state-audit
export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-bidirectional-allreduce
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-bidirectional-allreduce/torch_tpu_tier3
export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
scripts/k3/launch_kimi_k3_tp32_ep.sh
```

Rank 0 logged 162 records in
`/tmp/ray-k3-job664-state-audit/session_latest/logs/worker-686b13f5806ed393f52e2275d1833bcd560b5bee4396667ae7bdb706-ffffffff-3438328.out`.
That `/tmp` path is allocation-local and ephemeral; milestone 65 preserves the
four aggregate families. To re-aggregate a live log:

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter

path = "<rank-0-worker-log>"
for line in open(path):
    if "K3 mutation audit: " in line:
        records = json.loads(line.split("K3 mutation audit: ", 1)[1])
        break
counts = Counter((tuple(x["shape"]), x["dtype"], x["numel"])
                 for x in records)
print(len(records), counts)
PY
```

Do not reuse the accepted `VLLM_CACHE_ROOT` for this audit. The failed first
attempt did so and loaded a zero-artifact AOT entry from
`.../torch_aot_compile/dcd446.../rank_0_0/model`, then failed with
`TypeError: 'NoneType' object is not callable`. A fresh XLA path does not
override an explicitly stale `VLLM_CACHE_ROOT`.

The rejected semantics-preserving batching screen is in the same archive
commit and uses the same launch with a new
`K3_RUN_NAME=kimi-k3-b1-foreach-runtime-copies`, audit zero, and:

```bash
export TPU_K3_RUNTIME_STATE_AUDIT=0
export TPU_K3_BATCH_RUNTIME_STATE_COPIES=1
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q \
  tests/compilation/test_tpu_compiler.py -k runtime_state_mutation
scripts/k3/launch_kimi_k3_tp32_ep.sh
```

It batches only if generated source contains exactly two leaf `copy_` branches
for every mutation. All ranks logged 162, but endpoint runs were 15.14086 and
15.11524 ms mean TPOT (15.12805 ms pair), 0.00927 ms slower than accepted.
The latest rank-0 trace is
`/mnt/scratch/kimi-k3-b1-foreach-runtime-copies-profile/rank_0/tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_3517963.1786648147416799827.pt.trace.json.gz`.
It shows one 0.21790 ms `aten::_foreach_copy_` per decode and an unchanged
0.13570 ms end-to-next-start gap. Do not enable this flag; the composite call
does not provide native TPU batching.

### Device-only post-backbone tail accounting

Do not align host `aten::*` events with TPU kernels by their raw Chrome-trace
timestamps. In this profiler output the Ray-worker clock and TPU-device clock
have a several-millisecond offset, which is large relative to the 0.136 ms
tail. Use only TPU `pid` 2/3 timestamps for this breakdown:

```bash
.venv/bin/python scripts/k3/analyze_trace_tail.py \
  /mnt/scratch/kimi-k3-b1-bidirectional-allreduce-profile/rank_0/\
tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_\
2952330.1786635017233596519.pt.trace.json.gz
```

Expected accepted-capture means in microseconds are `gap=136.08002`,
`device_module_union=105.17285`, `uncovered=30.90717`, `lm_head=68.38793`,
`uniform_vocab=4.09355`, `uniform_seed=2.52651`,
`sample_argmax=18.81089`, `token_view=5.03490`, and
`placeholder_scatter=4.62236`. The parser skips the first profiler-boundary
backbone region by default and forms 17 stable consecutive gaps. It parses
one trace object at a time because some HLO names contain unescaped quotes.

This corrects the earlier claim that the tail was filled by AOT mutation
copies. Those calls and the 162-input shape audit are real, but their host
timestamps do not establish occupancy of this device interval. The current
next target is the separate 68.4 us LM-head projection/all-gather boundary,
followed by the approximately 31 us of inter-module launch intervals. See
milestone 66 for the full accounting and endpoint reconciliation.

### Opt-in LM-head/sampler fusion experiment

Milestone 67 tests the separate 68 us LM-head boundary with
`TPU_K3_FUSE_LM_HEAD_SAMPLER=1`. The flag is deliberately off by default and
supports only non-speculative generation without grammar or logprobs. It does
not change the full-vocabulary Gumbel-max algorithm or remove the TP32 vocab
all-gather. Use a fresh vLLM runtime/config cache for each flag value, while
reusing the accepted Tier-2/Tier-3 compiler lineage:

```bash
export K3_HEAD_IP=10.13.232.26
export K3_RUN_NAME=kimi-k3-b1-fused-lm-head-sampler-v2
export RAY_ADDRESS=$K3_HEAD_IP:6379
export HF_HOME=/mnt/scratch/hf-hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_XLA_CACHE_PATH=/mnt/scratch/$K3_RUN_NAME
export VLLM_CACHE_ROOT=$VLLM_XLA_CACHE_PATH/vllm-runtime
export VLLM_CONFIG_ROOT=$VLLM_XLA_CACHE_PATH/vllm-config
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=woosuk_kimi-k3-b1-bidirectional-allreduce
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=/mnt/scratch/kimi-k3-b1-bidirectional-allreduce/torch_tpu_tier3
export PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile
export LIBTPU_INIT_ARGS='--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false'
export TPU_K3_FUSE_LM_HEAD_SAMPLER=1
scripts/k3/launch_kimi_k3_tp32_ep.sh
```

The launcher appends the accepted dependent-collective scheduler, async
collective merger, bidirectional all-reduce, and enhanced-barrier flags.
Inspect the startup argv: on allocation 664 the distributed safety layer
rewrote `xla_tpu_use_enhanced_launch_barrier=true` to false. Do not compare
this run to an older allocation at microsecond resolution. For the matched
control, stop only the vLLM server, retain the Ray allocation, set
`K3_RUN_NAME=kimi-k3-b1-fused-control-v2` and
`TPU_K3_FUSE_LM_HEAD_SAMPLER=0`, recompute the three vLLM cache variables and
`PROF_DIR`, then relaunch. A source or actor-environment change needs fresh
actors; never cancel the parent allocation merely to restart vLLM.

Run each endpoint configuration twice:

```bash
OUTPUT_DIR=$PWD/results/$K3_RUN_NAME \
  scripts/k3/bench_isl1_osl1k.sh $K3_HEAD_IP
OUTPUT_DIR=$PWD/results/${K3_RUN_NAME}-repeat \
  scripts/k3/bench_isl1_osl1k.sh $K3_HEAD_IP
```

Capture two short traces per configuration with the same non-greedy request
shape used by the candidate:

```bash
TEMPERATURE=1 INCLUDE_SEED=0 \
OUTPUT_DIR=$PWD/results/profile-$K3_RUN_NAME \
PROF_DIR=/mnt/scratch/${K3_RUN_NAME}-profile \
  scripts/k3/profile_isl1_osl1k.sh $K3_HEAD_IP
```

Analyze rank 0 using TPU timestamps only. Add `--top-modules 8` to discover
the fused module name rather than relying on source-line numbers embedded in
HLO names:

```bash
.venv/bin/python scripts/k3/analyze_trace_tail.py --top-modules 8 \
  /mnt/scratch/kimi-k3-b1-fused-lm-head-sampler-v2-profile/rank_0/\
tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_\
3610275.1786650623170757310.pt.trace.json.gz

.venv/bin/python scripts/k3/analyze_trace_tail.py --top-modules 8 \
  /mnt/scratch/kimi-k3-b1-fused-control-v2-profile/rank_0/\
tpu-rack1-w-001.us-central1-c.c.inferact-vllm-tpu.internal_\
3610281.1786651991968150704.pt.trace.json.gz
```

Fused endpoint means were 15.1334594 and 15.1313557 ms; matched controls were
15.1419436 and 15.1396092 ms, a 0.0083688 ms/token fused win. Fused gaps were
127.999 and 128.363 us; control gaps were 135.901 and 136.094 us. The fused
LM-head/all-gather/argmax module was 81.285 us versus 87.12 us for the two
separate control modules. The vocabulary RNG remains separate (5.996 us fused
versus 4.09 us control); the larger whole-gap saving comes partly from
eliminating a launch interval. Short traces can include a slow backbone
outlier, so report means and medians and do not infer a new absolute best from
a single capture.

The immediately preceding Pallas attempt on Ray step `602.52` failed because
Mosaic cannot dynamically gather across the seven source vector registers of
an `8x896` score tile. The current code uses a masked reduction instead. Do
not restore the dynamic gather. Leave `TPU_K3_SHARD_LATENT_UP_PROJ=0`.

Trace caveat for the earlier exact-JAX experiment: its rank-0 trace proves the
selector still lowers to the same 92 `[1,896]` sorts (0.45000 ms/decode) and
92 `[16]` expert-ID sorts (0.11900 ms/decode). The custom boundary reduced the
size/compile time of the monolithic backbone artifact but did not replace the
device sort algorithm. Do not treat the 0.450 ms family as removed.

## Non-Slurm cluster

This assumes the Ray cluster is already running. Start the server in a
dedicated terminal, replacing `<RAY_HEAD_IP>` with the Ray head IP:

```bash
RAY_ADDRESS="<RAY_HEAD_IP>:6379" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/launch_kimi_k3_tp32_ep.sh
```

After the server is healthy, run the performance benchmark:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/benchmark" scripts/k3/bench_isl1_osl1k.sh "<SERVER_IP>"
```

Capture two post-warmup profiles:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/profile-requests" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/profile_isl1_osl1k.sh "<SERVER_IP>"
```

Run the full GSM8K five-shot evaluation:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/gsm8k" scripts/k3/eval_gsm8k.sh "<SERVER_IP>"
```

## Slurm cluster

Submit the server wrapper, replacing `<SLURM_PARTITION>` with the partition to
use:

```bash
sbatch --partition="<SLURM_PARTITION>" --export=ALL,KIMI_K3_LOG_ROOT="$PWD/results/kimi-k3/server",PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/slurm_launch_k3_tp32_tp.sbatch
```

The Slurm log prints the server endpoint. After it is healthy, run the
performance benchmark from a host that can reach that endpoint:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/benchmark" scripts/k3/bench_isl1_osl1k.sh "<SERVER_IP>"
```

Capture two post-warmup profiles:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/profile-requests" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/profile_isl1_osl1k.sh "<SERVER_IP>"
```

Run the full GSM8K five-shot evaluation:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/gsm8k" scripts/k3/eval_gsm8k.sh "<SERVER_IP>"
```
