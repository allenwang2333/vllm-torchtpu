#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3-verify
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# Minimal Slurm wrapper for launch_kimi_k3_tp32_ep.sh. It starts Ray on the
# allocated hosts, waits for all 32 TPU resources, then runs the standalone
# server launcher on the Ray head. Submit this file from the repository root.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-56}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-300}"
RAY_RESOURCE_TIMEOUT="${RAY_RESOURCE_TIMEOUT:-600}"
PORT="${PORT:-8000}"
RUN_USER="$(id -un)"

# This job starts the server and holds it up until cancelled or the time limit.
# It deliberately runs NO eval: bringing K3 up at TP32 costs ~40 minutes, and
# coupling a measurement to that load means every mistake in the measurement --
# a wrong dataset path, a bad flag -- costs another 40 minutes to retry. Run
# evals and benchmarks against the endpoint separately, as many times as needed.
MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
# Off by default so this file keeps behaving as it always has; the hierarchical
# arm sets it to 1.
export TPU_MOE_HIERARCHICAL_EP="${TPU_MOE_HIERARCHICAL_EP:-0}"
# Pin this. vLLM auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for K3's
# architecture whenever the variable is *absent* from the environment
# (vllm/config/vllm.py, "Auto-enabling"), and that disables the torch.compile
# pipeline outright -- an ~18x decode slowdown, 344 ms TPOT instead of 19 ms.
# Job 669 measured exactly that because the variable happened to be exported in
# the shell that submitted the earlier runs and not in the one that submitted
# it. A benchmark must not depend on the submitting shell.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# Independent of the flag above: shards the latent-MoE down/up projections
# across a chip's two cores instead of replicating them.
export TPU_LATENT_PROJ_INTRA_CHIP_TP="${TPU_LATENT_PROJ_INTRA_CHIP_TP:-0}"
# Fused KDA core (conv1d + GDN v3) vs the manual unfused path. Default 1
# matches kda_attention_op.py's own default; pinned here so an arm's value
# comes from the job record and not from the submitting shell.
export VLLM_TORCHTPU_KDA_FUSED="${VLLM_TORCHTPU_KDA_FUSED:-1}"
# Default ON here, unlike the launcher, which defaults it off. With async
# scheduling off the host scheduler does not overlap device execution and adds
# ~8.4 ms to every decode step -- half again the step time -- which swamps any
# effect being measured. Job 688 recorded 16.1 ms/step of device work and a
# 24.5 ms TPOT for exactly this reason, and looked like a 29% regression next to
# a baseline that had the flag on. Comparisons across this flag are meaningless.
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
# Pinned for the same reason as the flags above: job 643 ran with 16 (exported
# interactively) while the launcher's own default is 8, so its concurrency sweep
# is not comparable to a default run above c=8 -- at c=16 a max_num_seqs=8 server
# queues half the requests and reports a flattering TPOT over a shorter batch.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Static compile buckets. The launcher defaults to 2,16,512, under which every
# decode batch from 2 to 16 pads up to the bucket-16 executable and a
# concurrency sweep measures bucketing, not scaling. Pinned here so the bucket
# set comes from the job record, not the submitting shell; must contain the
# --max-num-batched-tokens bucket (512) or the launcher refuses to start.
export COMPILE_SIZES="${COMPILE_SIZES:-2,16,512}"
# Max tokens per step. Default 512 (chunked prefill) is what the launcher has
# always passed. Raising it changes which MLA tuning key a prefill hits: the
# Kimi-K3 tuned mixed-prefill entries key on max_num_tokens=8192, so chunked
# prefill at 512 never reaches them. Must have a matching COMPILE_SIZES bucket.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
# Route TP all-reduces whose payload sits inside the measured crossover to the
# one-shot Pallas kernel instead of XLA's ring. Pinned here for the same reason
# as everything above: it changes decode speed, so it must come from the batch
# script and not from whatever shell happened to submit the job.
# The unified attention-shaped KV pool, and prefix caching on top of it.
# Pinned here for the same reason as everything above: both change the block
# size, the memory layout and the kernels that run, so they must come from this
# file and not from whatever shell submitted the job.
#
# TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL is an *optional* bool in envs.py: unset
# means "let the architecture decide" (POOLED_GDN_ARCHITECTURES), which for
# Kimi-K3 today means off. Exporting 0 explicitly says so in the job record and
# keeps a control arm a control arm even if K3 later joins that set.
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-0}"
# Prefix caching is rejected up front on hybrid models unless the pool is on
# (tpu_platform.py), so the two travel together.
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && "${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}" != "1" ]]; then
  echo "ENABLE_PREFIX_CACHING=1 needs TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 on a hybrid model" >&2
  exit 2
fi
export TPU_LOW_LATENCY_ALLREDUCE="${TPU_LOW_LATENCY_ALLREDUCE:-0}"
export TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT="${TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT:-0}"
LOG_DIR="${KIMI_K3_LOG_ROOT:-${REPO_DIR}/results/kimi-k3-verify-${SLURM_JOB_ID}}"
# Set PROFILE_DIR to make /start_profile work. Both halves are required and
# neither can be turned on after the server is up: the phased profiler is read
# at worker init, and the trace directory is a server CLI flag. Without them
# /start_profile answers 404 and the arm script cheerfully reports "profiled
# tokens: 128" having captured nothing -- which is how job 672 finished its
# whole sweep before the missing profile surfaced.
PROFILE_ARGS=()
if [[ -n "${PROFILE_DIR:-}" ]]; then
  mkdir -p "${PROFILE_DIR}"
  export USE_PHASED_PROFILER=true
  # `profiler=torch` is not optional: vLLM's ProfilerConfig validator rejects a
  # torch_profiler_dir without it ("only applicable when profiler is set to
  # 'torch'") and the server exits before Ray even starts. Three flags, all
  # required, none of them settable after startup.
  PROFILE_ARGS=(--profiler-config.profiler=torch
                --profiler-config.torch_profiler_dir="${PROFILE_DIR}")
fi
RAY_LOG="${LOG_DIR}/ray.log"
LAUNCHER="${REPO_DIR}/scripts/k3/launch_kimi_k3_tp32_ep.sh"

[[ -x "${LAUNCHER}" ]] || { echo "Launcher not found: ${LAUNCHER}" >&2; exit 2; }
[[ -x "${REPO_DIR}/.venv/bin/ray" ]] || { echo "Ray not found in ${REPO_DIR}/.venv" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HEAD_HOST="${HOSTS[0]}"
HEAD_IP="${KIMI_K3_HEAD_IP:-$(srun --overlap -N1 -n1 -w "${HEAD_HOST}" hostname -I | awk '{print $1}')}"
[[ -n "${HEAD_IP}" ]] || { echo "Could not resolve ${HEAD_HOST}" >&2; exit 2; }

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_USAGE_STATS_ENABLED=0
export TPU_MULTIHOST_BACKEND=ray
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export JAX_PLATFORMS=""
export PORT

# Checkpoint location. The shared model store is /mnt/models (mounted on
# every host, verified at submit time below) in HF hub layout. Set
# HF_HUB_CACHE explicitly, never derive it from HF_HOME -- an inherited value
# silently wins over HF_HOME and has redirected loads to a partial copy before.
MODEL_DIR="models--${MODEL//\//--}"
# Existence of snapshots/ is NOT evidence the snapshot is usable. Job 174 died
# five minutes into bring-up on a /mnt/scratch copy that held all 96 safetensors
# shards but was missing kimi_k3_vision_processing.py: vLLM builds ModelConfig
# before any weight is read, and that probes multimodal support -- which imports
# the vision remote code even under --language-model-only. The weights are the
# slow half of the cache and the metadata is the fragile half, so check the
# fragile half explicitly and check it on EVERY host, since /mnt/scratch is
# node-local and the four copies differ.
REQUIRED_SNAPSHOT_FILES="${REQUIRED_SNAPSHOT_FILES:-config.json configuration_kimi_k3.py modeling_kimi_k3.py modeling_kimi_linear.py kimi_k3_processor.py kimi_k3_vision_processing.py media_utils.py encoding_k3.py tokenization_kimi.py tokenizer_config.json tiktoken.model model.safetensors.index.json}"
all_hosts_have() {
  srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c '
      snap=$(ls -d "'"$1"'/'"${MODEL_DIR}"'/snapshots/"*/ 2>/dev/null | head -1)
      [ -n "${snap}" ] || exit 1
      for f in '"${REQUIRED_SNAPSHOT_FILES}"'; do
        [ -e "${snap}${f}" ] || { echo "$(hostname -s): ${snap} lacks ${f}" >&2; exit 1; }
      done
      ls "${snap}"*.safetensors >/dev/null 2>&1 || exit 1
    ' >/dev/null 2>&1
}
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  echo "Using caller-supplied HF_HUB_CACHE=${HF_HUB_CACHE}"
elif all_hosts_have /mnt/models; then
  export HF_HUB_CACHE=/mnt/models
else
  echo "${MODEL} not found under /mnt/models (the shared model store)" >&2
  exit 2
fi
# /mnt is root-owned on the compute nodes, so do not derive these from the
# model store's parent: HF_HOME=/mnt made HF_MODULES_CACHE=/mnt/modules and the
# mkdir below died on it (job 28). Keep the writable aux caches in the repo.
export HF_HOME="${HF_HOME:-${REPO_DIR}/.runtime-cache/hf-home}"
echo "Checkpoint: ${HF_HUB_CACHE}/${MODEL_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# The cache holds *compiled executables*, so any flag that changes the graph
# must change the path. TPU_LOW_LATENCY_ALLREDUCE adds two workspace tensors as
# graph inputs; sharing a cache across that flag made a treatment run pick up
# the control's executable and die at execution with
#   execution supplied 2135 arguments but compiled program expected 2133
# -- off by exactly the two workspaces. The Tier-2/Tier-3 keys do not capture
# it, so keep the arms in separate trees.
_LLAR_TAG=""
[[ "${TPU_LOW_LATENCY_ALLREDUCE:-0}" == "1" ]] && _LLAR_TAG="-llar"
# Same hazard, different flag: the unified pool moves the KDA state into the
# attention pages and lifts block_size from 16 to the state-fit floor, so every
# executable differs from the per-layer arm's. Prefix caching adds the seed-copy
# graphs on top. Neither is captured by the Tier-2/Tier-3 keys, so give each arm
# its own tree rather than debug an argument-count mismatch at execution time.
_POOL_TAG=""
[[ "${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL}" == "1" ]] && _POOL_TAG="-pool"
[[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && _POOL_TAG="${_POOL_TAG}-apc"
# Same hazard again: fused vs unfused KDA produces different graphs, so the
# unfused arm gets its own tree. Fused is the code's default and keeps the
# untagged path so existing caches stay usable.
_KDA_TAG=""
[[ "${VLLM_TORCHTPU_KDA_FUSED}" != "1" ]] && _KDA_TAG="-kda-unfused"
# Same hazard, fourth flag: the bucket set changes which executables exist, so
# a non-default COMPILE_SIZES gets its own cache tree too.
_SIZES_TAG=""
[[ "${COMPILE_SIZES}" != "2,16,512" ]] && _SIZES_TAG="-cs$(echo "${COMPILE_SIZES}" | tr ', ' '--')"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/mnt/scratch/users/${RUN_USER}/vllm-xla-cache/kimi-k3${_LLAR_TAG}${_POOL_TAG}${_KDA_TAG}${_SIZES_TAG}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-runtime}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-config}"
# torch_tpu >= 0.1.1.dev20260813 reads the non-INTERNAL names; the INTERNAL
# exports below are inert on this build (workers fall back to tier2 name
# "default" and crash on another user's /dev/shm/torch_tpu_cache/default).
# Export both so older builds keep working too.
export TORCH_TPU_TIER2_COMPILATION_CACHE="${KIMI_K3_TIER2_CACHE_NAMESPACE:-${RUN_USER}_kimi_k3_${SLURM_JOB_ID}}"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${TORCH_TPU_TIER2_COMPILATION_CACHE}"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"

mkdir -p "${LOG_DIR}"
srun --overlap -N4 -n4 --ntasks-per-node=1 mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT}"

RAY_STEP_PID=
SERVER_STEP_PID=
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n ${SERVER_STEP_PID} ]] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
  # Workers outlive the launcher often enough to strand the TPUs for the next
  # job, and at TP32 that is four hosts held hostage rather than one.
  timeout 60 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    pkill -9 -f 'VLLM::Worker' >/dev/null 2>&1 || true
  if [[ -n ${RAY_STEP_PID} ]] && kill -0 "${RAY_STEP_PID}" 2>/dev/null; then
    kill "${RAY_STEP_PID}" 2>/dev/null || true
    wait "${RAY_STEP_PID}" 2>/dev/null || true
  fi
  timeout 30 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    ray stop --force >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export KIMI_K3_RAY_HEAD_IP="${HEAD_IP}"
export KIMI_K3_RAY_PORT="${RAY_PORT}"
export KIMI_K3_RAY_NUM_CPUS="${RAY_NUM_CPUS}"

# Start one persistent Ray node per host. Task zero becomes the head; the
# remaining tasks wait for its port and then join as workers.
# shellcheck disable=SC2016
srun --overlap -N4 -n4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    node_ip=$(hostname -I | awk "{print \$1}")
    if [[ ${SLURM_PROCID} -eq 0 ]]; then
      exec ray start --block --head \
        --node-ip-address="${node_ip}" \
        --port="${KIMI_K3_RAY_PORT}" \
        --include-dashboard=false \
        --num-cpus="${KIMI_K3_RAY_NUM_CPUS}"
    fi
    until timeout 1 bash -c \
        "</dev/tcp/${KIMI_K3_RAY_HEAD_IP}/${KIMI_K3_RAY_PORT}" 2>/dev/null; do
      sleep 1
    done
    exec ray start --block \
      --address="${KIMI_K3_RAY_HEAD_IP}:${KIMI_K3_RAY_PORT}" \
      --node-ip-address="${node_ip}" \
      --num-cpus="${KIMI_K3_RAY_NUM_CPUS}"
  ' >"${RAY_LOG}" 2>&1 &
RAY_STEP_PID=$!

echo "Waiting up to ${RAY_START_TIMEOUT}s for Ray at ${RAY_ADDRESS}"
deadline=$(( $(date +%s) + RAY_START_TIMEOUT ))
until timeout 1 bash -c "</dev/tcp/${HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
  if ! kill -0 "${RAY_STEP_PID}" 2>/dev/null; then
    echo "Ray exited during startup; see ${RAY_LOG}" >&2
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out waiting for Ray; see ${RAY_LOG}" >&2
    exit 1
  fi
  sleep 1
done

echo "Waiting up to ${RAY_RESOURCE_TIMEOUT}s for 32 Ray TPU resources"
deadline=$(( $(date +%s) + RAY_RESOURCE_TIMEOUT ))
tpu_count=0
while (( $(date +%s) < deadline )); do
  tpu_count=$(timeout 30 "${REPO_DIR}/.venv/bin/python" -c '
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("TPU", 0)))
' 2>/dev/null | tail -n 1 || true)
  [[ ${tpu_count:-0} == 32 ]] && break
  echo "  Ray TPU resources: ${tpu_count:-0}/32"
  sleep 10
done
if [[ ${tpu_count:-0} != 32 ]]; then
  echo "Only ${tpu_count:-0}/32 TPU resources joined; see ${RAY_LOG}" >&2
  exit 1
fi

echo "Ray is ready on ${HOSTS[*]} with 32 TPU resources"
echo "Kimi-K3 endpoint will be http://${HEAD_IP}:${PORT}/v1"
echo "Ray log: ${RAY_LOG}"

echo "Hierarchical MoE parallelism: TPU_MOE_HIERARCHICAL_EP=${TPU_MOE_HIERARCHICAL_EP}"
echo "KDA path: VLLM_TORCHTPU_KDA_FUSED=${VLLM_TORCHTPU_KDA_FUSED} (1=fused, 0=unfused)"

# The server runs as a background job step so this script can detect readiness,
# report the endpoint, and then simply hold the allocation. Extra arguments
# passed to sbatch after the script name are forwarded to vLLM.
SERVER_LOG="${LOG_DIR}/server.log"
srun --overlap -N1 -n1 -w "${HEAD_HOST}" --unbuffered --export=ALL \
  --chdir="${REPO_DIR}" "${LAUNCHER}" "${PROFILE_ARGS[@]}" "$@" >"${SERVER_LOG}" 2>&1 &
SERVER_STEP_PID=$!

echo "Waiting up to ${READY_TIMEOUT}s for the endpoint (log: ${SERVER_LOG})"
start=$(date +%s)
ready=0
while (( $(date +%s) - start < READY_TIMEOUT )); do
  if curl -sf "http://${HEAD_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server exited during startup; tail:" >&2
    tail -60 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 10
done
if (( ready != 1 )); then
  echo "Endpoint never came up within ${READY_TIMEOUT}s; tail:" >&2
  tail -60 "${SERVER_LOG}" >&2
  exit 1
fi
echo "SERVER_READY after $(( $(date +%s) - start ))s"

# Record the endpoint before any diagnostics run: a reporting failure must not
# leave callers without the URL of a server that is up and serving.
ENDPOINT="http://${HEAD_IP}:${PORT}"
printf '%s\n' "${ENDPOINT}" > "${LOG_DIR}/endpoint.txt"

# Whether the chip split actually engaged. Ray de-duplicates repeated worker log
# lines, so the raw count understates badly -- count the [repeated Nx] suffixes
# too or a healthy TP32 run looks like it only patched four layers.
#
# This whole block is diagnostics: it must never be able to end the job. Job 610
# lost a healthy server 1933s in because one grep child was killed (the head node
# is also loading K3's weights, so the kernel reclaims whatever it can) and
# `set -e` turned that into a teardown. Hence report_split() ... || true, and
# line-based greps rather than a greedy `.*` across a 38 MB log.
report_split() {
  local marker="Hierarchical MoE parallelism: EP between chips"
  local raw suppressed fallback
  # `|| true`, never `|| echo 0`: grep -c already prints "0" when it finds
  # nothing and *also* exits 1, so `|| echo 0` yields the two-line value
  # "0\n0" and the arithmetic below dies with a syntax error.
  raw=$(grep -Fc "${marker}" "${SERVER_LOG}" 2>/dev/null || true)
  suppressed=$(grep -F "${marker}" "${SERVER_LOG}" 2>/dev/null | cut -c1-400 \
    | grep -oE "repeated [0-9]+x" | grep -oE "[0-9]+" \
    | awk '{n+=$1} END {print n+0}' || true)
  fallback=$(grep -c "falling back to flat expert parallelism" "${SERVER_LOG}" 2>/dev/null || true)
  echo "Split check: config lines=$(( ${raw:-0} + ${suppressed:-0} )) (raw ${raw:-0} + suppressed ${suppressed:-0}), fallbacks=${fallback:-0}"
  grep -m1 -o "Chip topology: .*" "${SERVER_LOG}" 2>/dev/null | head -1 || true
  # The filter logs the CALLER's rank-derived ep_size/ep_rank, not the patched
  # ones, so those numbers look wrong even when it is working. The expert COUNT
  # is the honest signal: 56/896 is a chip block (896/16), 28/896 is unpatched.
  grep -o "loading [0-9]*/[0-9]* experts" "${SERVER_LOG}" 2>/dev/null | sort | uniq -c || true
  # Same idea for the latent projections: the honest signal is the per-core
  # column count. 1792/3584 means the down-proj really is split across a chip;
  # absent lines mean it silently stayed replicated.
  grep -o "Intra-chip group ready: .*" "${SERVER_LOG}" 2>/dev/null | head -1 || true
  grep -o "Latent MoE projection \[[0-9, ]*\] sharded across [0-9]* cores per chip ([0-9]* columns each)" \
    "${SERVER_LOG}" 2>/dev/null | sort -u || true
  # A server with torch.compile disabled still serves correct tokens at ~1/18th
  # the decode speed, so nothing downstream notices -- the sweep just reports a
  # terrible number. Say so here, next to the endpoint, where it cannot be missed.
  if grep -qF "disabling vLLM's torch.compile pipeline" "${SERVER_LOG}" 2>/dev/null; then
    echo "WARNING: torch.compile is DISABLED (breakable cudagraph). Benchmarks from"
    echo "         this server are not comparable to compiled runs."
  else
    echo "torch.compile: enabled"
  fi
  echo "profiling: ${PROFILE_DIR:-off (set PROFILE_DIR to enable /start_profile)}"
  # Read it back from the server's own resolved config rather than echoing the
  # variable: the launcher, not this file, decides what reaches vLLM.
  echo "async scheduling: $(grep -m1 -oE "async_scheduling': *[A-Za-z]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  echo "max_num_seqs: $(grep -m1 -oE "max_num_seqs': *[0-9]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  echo "compile_sizes: requested=${COMPILE_SIZES} resolved=$(
    grep -m1 -oE "compile_sizes': *\[[0-9, ]*\]" "${SERVER_LOG}" 2>/dev/null \
    || echo 'unknown')"
  echo "max_num_batched_tokens: $(grep -m1 -oE "max_num_batched_tokens': *[0-9]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  # MLA tuned-params misses are loud in the worker logs ("No tuned parameters
  # found ..."). List which mixed-prefill keys missed so a Kimi-K3 8k miss
  # cannot hide; hits are silent by design.
  echo "MLA tuned-param misses (mixed):"
  grep -ao "No tuned parameters found for the given tuning key: TuningKey(case='mixed'[^)]*)" \
    "${SERVER_LOG}" 2>/dev/null | sed -E "s/.*(max_num_tokens=[0-9]+).*(actual_num_q_heads=[0-9]+).*/  miss: \1 \2/" \
    | sort | uniq -c | head -10 || true
  # Requested vs resolved, again: the pool is selected inside the platform from
  # the env var AND the architecture, and prefix caching is silently a no-op on
  # a server that refused it. Read both back off the server's own log.
  echo "unified KV pool: requested=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL} derivation=$(
    grep -m1 -c "TPU block_size derivation path" "${SERVER_LOG}" 2>/dev/null || echo 0
    )x (1 = pool path ran)"
  echo "prefix caching: requested=${ENABLE_PREFIX_CACHING} resolved=$(
    grep -m1 -oE "enable_prefix_caching=[A-Za-z]+" "${SERVER_LOG}" 2>/dev/null \
    || grep -m1 -oE "enable_prefix_caching': *[A-Za-z]+" "${SERVER_LOG}" 2>/dev/null \
    || echo 'unknown')"
  # block_size is the single number that says whether the KDA state moved into
  # the attention pages: 16 is the per-layer arm, the state-fit floor the pool.
  echo "block_size: $(grep -m1 -oE "Adjusting TPU FA block_size for [^:]*: [0-9]+ -> [0-9]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unchanged (16)')"
  echo "KV cache: $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  local latent_fallback
  latent_fallback=$(grep -c "leaving the latent projections replicated\|leaving it replicated" \
    "${SERVER_LOG}" 2>/dev/null || true)
  echo "Latent projection fallbacks: ${latent_fallback:-0}"
  # Read back what the dispatcher actually did, not what was requested: it
  # falls back to XLA silently whenever a shape is outside the measured
  # crossover or an op fails to build, and "the flag was set" is not evidence
  # that a single all-reduce changed.
  echo "Low-latency all-reduce: TPU_LOW_LATENCY_ALLREDUCE=${TPU_LOW_LATENCY_ALLREDUCE}"
  local ll_one ll_two ll_xla
  ll_one=$(grep -c -- "-> one_shot" "${SERVER_LOG}" 2>/dev/null || true)
  ll_two=$(grep -c -- "-> two_shot" "${SERVER_LOG}" 2>/dev/null || true)
  ll_xla=$(grep -c "Low-latency all-reduce \[.*-> xla" "${SERVER_LOG}" 2>/dev/null || true)
  echo "  routed: one_shot=${ll_one:-0} two_shot=${ll_two:-0} xla=${ll_xla:-0} (per rank, per shape)"
  grep -h "Low-latency all-reduce \[" "${SERVER_LOG}" 2>/dev/null \
    | sed -E 's/.*(shape=[^ ]+ .*)/    \1/' | sort -u | head -8 || true
}
report_split || true

cat <<EOF

============================================================
 ENDPOINT   ${ENDPOINT}/v1
 model      ${MODEL}
 job        ${SLURM_JOB_ID} on ${HOSTS[*]}
 log        ${SERVER_LOG}

 Run evals against it, e.g.
   MODEL="${MODEL}" BASE_URL="${ENDPOINT}" \\
   OUTPUT_DIR="${LOG_DIR}/gsm8k" \\
   HF_HUB_CACHE=/mnt/lustre/hf-models/hub \\
   HF_DATASETS_CACHE="${REPO_DIR}/.runtime-cache/hf-datasets" \\
     ${REPO_DIR}/scripts/k3/eval_gsm8k.sh

 Holding the allocation until the time limit or scancel ${SLURM_JOB_ID}.
============================================================

EOF

wait "${SERVER_STEP_PID}"
