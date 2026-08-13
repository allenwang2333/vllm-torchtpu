# Kimi-K3 runbook

Run these commands from the repository root. Replace `<SERVER_IP>` with the IP
of the host serving the OpenAI-compatible endpoint.

## Optimization handoff (2026-08-13)

The active objective is batch-1 TPOT at or below 15 ms on one TPU v7x-32
slice, without speculative decoding. All performance runs use dummy weights,
`compile_sizes=[1]`, ISL=1, OSL=1000, and concurrency 1. Read
`OPTIMIZATION_LOG.md` before selecting another experiment; it records rejected
ideas as well as results. The best repeated configuration currently measures
15.27295--15.27481 ms mean TPOT. It is backed up on the non-main branch
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
- enhanced TPU launch barriers; and
- XLA dependent-collective layer scheduling.

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

The launcher appends the winning enhanced-barrier and dependent-collective
flags unless the caller explicitly supplies either flag with `=true` or
`=false`. For an isolated compiler experiment, append only that experiment's
flag to `LIBTPU_INIT_ARGS`; do not bundle unrelated tuning flags.

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

Allocation job 602 is intentionally still active. There is currently no Ray
step and no vLLM server: async-collective-fusion server PID 2744587 was stopped
cleanly and only Ray step `602.62` was canceled. Start fresh Ray workers with
the four-host command above before the next experiment; a fresh Ray step is
mandatory because the last trial changed `LIBTPU_INIT_ARGS`. The next planned
isolated trial is `--xla_tpu_enable_async_collective_merger=true` on top of the
normal launcher flags, with async collective fusion omitted/false.

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

The immediately preceding Pallas attempt on Ray step `602.52` failed because
Mosaic cannot dynamically gather across the seven source vector registers of
an `8x896` score tile. The current code uses a masked reduction instead. Do
not restore the dynamic gather. The next experiment requires fresh Ray workers
because no Ray step is active. Never cancel allocation job 602 while
continuing this work. Leave `TPU_K3_SHARD_LATENT_UP_PROJ=0`.

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
