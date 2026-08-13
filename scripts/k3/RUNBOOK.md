# Kimi-K3 runbook

Run these commands from the repository root. Replace `<SERVER_IP>` with the IP
of the host serving the OpenAI-compatible endpoint.

## Optimization handoff (2026-08-13)

The active objective is batch-1 TPOT at or below 15 ms on one TPU v7x-32
slice, without speculative decoding. All performance runs use dummy weights,
`compile_sizes=[1]`, ISL=1, OSL=1000, and concurrency 1. Read
`OPTIMIZATION_LOG.md` before selecting another experiment; it records rejected
ideas as well as results. The best repeated configuration currently measures
15.82--15.85 ms mean TPOT. It is backed up on the non-main branch
`agent/k3-b1-tpot-optimization-backup`.

The launcher defaults contain the current winning model configuration:

- hierarchical MoE EP16 x TP2;
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

The exact TPU/JAX router experiment is reproduced by adding
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

Allocation job 602 is intentionally still active. Ray step `602.50` is active
on all four hosts with head `10.13.232.26`; its persistent terminal/session is
`10531`. No vLLM server is active. The most recent launch used fresh namespace
`/mnt/scratch/kimi-k3-b1-latent-up-tp4-coalesced` but failed before serving
because `ProcessGroupTPU` does not support `allreduce_coalesced`; its former
API PID 2301781 exited. The source was restored to the compile-safe TP4
latent-up implementation after that failure.

The preceding early-shared-ordering server on Ray step `602.49` was stopped
after two endpoint runs and profiles. Its caches and profiles remain under
`/mnt/scratch/kimi-k3-b1-latent-up-tp4-early-shared{,-profile}`, and results
are under `results/kimi-k3-b1-latent-up-tp4-early-shared{,-repeat}`. The pair
measured 16.13022 and 16.12713 ms mean TPOT, 0.16462 ms slower than the fresh
default control. Its trace still contains 440 physical all-reduces and no
variadic collective, so the temporary ordering code was removed.

Step `602.50` has imported the now-removed coalesced source and must not be
reused. Cancel only step `602.50`, then start fresh Ray workers before the next
source, model, or `LIBTPU_INIT_ARGS` experiment. Never cancel allocation job
602 while continuing this work. Leave `TPU_K3_SHARD_LATENT_UP_PROJ=0` for the
accepted configuration.

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
