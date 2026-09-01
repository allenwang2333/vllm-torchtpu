#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3-dp4tp8
#SBATCH --partition=ici-2x2x4-b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# Kimi-K3 at DP=4 x TP=8 + EP=32 (Batch Q). Unlike the TP=32 sbatch there is
# NO Ray: the mp multihost DP path runs one `vllm serve` per host (head +
# --headless followers), each engine owning its host's 8 TPU cores as a TP=8
# group, while expert parallelism spans the unified 32-device slice.
#
# This job starts the server and holds it up until cancelled or the time
# limit. It deliberately runs NO eval. Extra args after the script name are
# forwarded to every host's `vllm serve` (e.g. --load-format dummy).
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PORT="${PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"
DP_MASTER_PORT="${DP_MASTER_PORT:-29511}"
RUN_USER="$(id -un)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL
READY_TIMEOUT="${READY_TIMEOUT:-7200}"

# --- job-288 parity pins (a benchmark must not depend on the submitting shell)
export TPU_MOE_HIERARCHICAL_EP="${TPU_MOE_HIERARCHICAL_EP:-1}"
export TPU_LATENT_PROJ_INTRA_CHIP_TP="${TPU_LATENT_PROJ_INTRA_CHIP_TP:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_TORCHTPU_KDA_FUSED="${VLLM_TORCHTPU_KDA_FUSED:-1}"
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-0}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export TPU_LOW_LATENCY_ALLREDUCE="${TPU_LOW_LATENCY_ALLREDUCE:-0}"
export TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT="${TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT:-0}"
# Per-rank sizing: job 288's global values (seqs 32, tokens 8192) divided by
# DP=4. MAX_MODEL_LEN stays 10240: KV is partitioned by sequence across DP
# groups, so each engine still needs room for full-length sequences.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
export COMPILE_SIZES="${COMPILE_SIZES:-1,2,4,8,2048}"
# Job 288's LIBTPU_INIT_ARGS: the diagnostic quieting plus hierarchical
# all-reduce offload.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true}"

LOG_DIR="${KIMI_K3_LOG_ROOT:-${REPO_DIR}/results/kimi-k3-dp4tp8-${SLURM_JOB_ID}}"
PROFILE_ARGS=()
if [[ -n "${PROFILE_DIR:-}" ]]; then
  mkdir -p "${PROFILE_DIR}"
  export USE_PHASED_PROFILER=true
  # profiler=torch is not optional: vLLM's ProfilerConfig validator rejects a
  # torch_profiler_dir without it.
  PROFILE_ARGS=(--profiler-config.profiler=torch
                --profiler-config.torch_profiler_dir="${PROFILE_DIR}")
fi
LAUNCHER="${REPO_DIR}/scripts/k3/launch_kimi_k3_dp4tp8_ep.sh"

[[ -x "${LAUNCHER}" ]] || { echo "Launcher not found: ${LAUNCHER}" >&2; exit 2; }

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
export PORT

# Checkpoint metadata probe (same as the TP=32 sbatch): even with
# --load-format dummy the server reads config + remote code from the
# snapshot, so verify the fragile files on EVERY host.
MODEL_DIR="models--${MODEL//\//--}"
REQUIRED_SNAPSHOT_FILES="${REQUIRED_SNAPSHOT_FILES:-config.json configuration_kimi_k3.py modeling_kimi_k3.py modeling_kimi_linear.py kimi_k3_processor.py kimi_k3_vision_processing.py media_utils.py encoding_k3.py tokenization_kimi.py tokenizer_config.json tiktoken.model model.safetensors.index.json}"
all_hosts_have() {
  srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c '
      snap=$(ls -d "'"$1"'/'"${MODEL_DIR}"'/snapshots/"*/ 2>/dev/null | head -1)
      [ -n "${snap}" ] || exit 1
      for f in '"${REQUIRED_SNAPSHOT_FILES}"'; do
        [ -e "${snap}${f}" ] || { echo "$(hostname -s): ${snap} lacks ${f}" >&2; exit 1; }
      done
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
export HF_HOME="${HF_HOME:-${REPO_DIR}/.runtime-cache/hf-home}"
echo "Checkpoint: ${HF_HUB_CACHE}/${MODEL_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# The cache holds *compiled executables*, so the geometry goes in the path:
# a TP=32 cache must never serve a DP=4 x TP=8 graph or vice versa.
_SIZES_TAG=""
[[ "${COMPILE_SIZES}" != "1,2,4,8,2048" ]] && _SIZES_TAG="-cs$(echo "${COMPILE_SIZES}" | tr ', ' '--')"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/mnt/scratch/users/${RUN_USER}/vllm-xla-cache/kimi-k3-dp4tp8${_SIZES_TAG}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-runtime}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-config}"
export TORCH_TPU_TIER2_COMPILATION_CACHE="${KIMI_K3_TIER2_CACHE_NAMESPACE:-${RUN_USER}_kimi_k3_${SLURM_JOB_ID}}"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${TORCH_TPU_TIER2_COMPILATION_CACHE}"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"

mkdir -p "${LOG_DIR}"
srun --overlap -N4 -n4 --ntasks-per-node=1 mkdir -p \
  "${HF_HOME}" \
  "${HF_MODULES_CACHE}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT}"

SERVER_STEP_PIDS=()
cleanup() {
  status=$?
  trap - EXIT
  for pid in "${SERVER_STEP_PIDS[@]:-}"; do
    if [[ -n ${pid} ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  # vLLM engine cores and workers outlive the launcher often enough to
  # strand the TPUs for the next job.
  timeout 60 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c "pkill -9 -f 'VLLM::Worker' 2>/dev/null; pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'EngineCore' 2>/dev/null; true" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Hierarchical MoE parallelism: TPU_MOE_HIERARCHICAL_EP=${TPU_MOE_HIERARCHICAL_EP}"
echo "KDA path: VLLM_TORCHTPU_KDA_FUSED=${VLLM_TORCHTPU_KDA_FUSED} (1=fused, 0=unfused)"
echo "Geometry: DP=4 x TP=8 attention, EP=32 experts; head=${HEAD_HOST} (${HEAD_IP})"
echo "Per-rank sizing: max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_model_len=${MAX_MODEL_LEN} compile_sizes=${COMPILE_SIZES}"

# One server per host. Host i owns DP ranks [i] (data-parallel-size-local=1);
# host 0 additionally serves the API on ${PORT}.
SERVER_LOGS=()
for i in 0 1 2 3; do
  SERVER_LOGS+=("${LOG_DIR}/server_host${i}.log")
  srun --overlap -N1 -n1 -w "${HOSTS[$i]}" --unbuffered --export=ALL \
    --chdir="${REPO_DIR}" \
    env DP_START_RANK="$i" DP_MASTER_ADDR="${HEAD_IP}" \
        DP_RPC_PORT="${DP_RPC_PORT}" DP_MASTER_PORT="${DP_MASTER_PORT}" \
      "${LAUNCHER}" "${PROFILE_ARGS[@]}" "$@" \
      >"${SERVER_LOGS[$i]}" 2>&1 &
  SERVER_STEP_PIDS+=($!)
done

echo "Waiting up to ${READY_TIMEOUT}s for the endpoint (logs: ${LOG_DIR}/server_host*.log)"
start=$(date +%s)
ready=0
while (( $(date +%s) - start < READY_TIMEOUT )); do
  if curl -sf "http://${HEAD_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  for i in 0 1 2 3; do
    if ! kill -0 "${SERVER_STEP_PIDS[$i]}" 2>/dev/null; then
      echo "Server on ${HOSTS[$i]} exited during startup; tail:" >&2
      tail -60 "${SERVER_LOGS[$i]}" >&2
      exit 1
    fi
  done
  sleep 10
done
if (( ready != 1 )); then
  echo "Endpoint never came up within ${READY_TIMEOUT}s; tails:" >&2
  for i in 0 1 2 3; do
    echo "== ${HOSTS[$i]} ==" >&2
    tail -30 "${SERVER_LOGS[$i]}" >&2
  done
  exit 1
fi
echo "SERVER_READY after $(( $(date +%s) - start ))s"

ENDPOINT="http://${HEAD_IP}:${PORT}"
printf '%s\n' "${ENDPOINT}" > "${LOG_DIR}/endpoint.txt"

# Diagnostics. Read from ALL four host logs: each host's workers log their
# own expert split, banner and kernel-tuning misses, and Ray's line
# de-duplication is not in play here (no Ray). This whole block must never
# be able to end the job (job 610 lesson), hence the || true guards.
report_split() {
  local marker="Hierarchical MoE parallelism: EP between chips"
  local raw fallback
  raw=$(cat "${SERVER_LOGS[@]}" 2>/dev/null | grep -Fc "${marker}" || true)
  fallback=$(cat "${SERVER_LOGS[@]}" 2>/dev/null | grep -c "falling back to flat expert parallelism" || true)
  echo "Split check: hierarchical-EP banner lines=${raw:-0}, fallbacks=${fallback:-0}"
  grep -h -m1 -o "Chip topology: .*" "${SERVER_LOGS[@]}" 2>/dev/null | head -1 || true
  # The expert COUNT is the honest signal: 28/896 is flat EP=32
  # (896/32 per rank); 56/896 is a hierarchical chip block (896/16, two
  # cores share it); 112/896 would mean EP wrongly collapsed to the TP=8
  # group.
  grep -h -o "loading [0-9]*/[0-9]* experts" "${SERVER_LOGS[@]}" 2>/dev/null | sort | uniq -c || true
  grep -h -o "Intra-chip group ready: .*" "${SERVER_LOGS[@]}" 2>/dev/null | sort -u | head -2 || true
  grep -h -o "Latent MoE projection \[[0-9, ]*\] sharded across [0-9]* cores per chip ([0-9]* columns each)" \
    "${SERVER_LOGS[@]}" 2>/dev/null | sort -u || true
  if grep -qF "disabling vLLM's torch.compile pipeline" "${SERVER_LOGS[0]}" 2>/dev/null; then
    echo "WARNING: torch.compile is DISABLED (breakable cudagraph). Benchmarks from"
    echo "         this server are not comparable to compiled runs."
  else
    echo "torch.compile: enabled"
  fi
  echo "profiling: ${PROFILE_DIR:-off (set PROFILE_DIR to enable /start_profile)}"
  # Read back the resolved config from the head's own log, not the shell.
  echo "async scheduling: $(grep -m1 -oE "async_scheduling': *[A-Za-z]+" \
    "${SERVER_LOGS[0]}" 2>/dev/null || echo 'unknown')"
  echo "max_num_seqs: $(grep -m1 -oE "max_num_seqs': *[0-9]+" \
    "${SERVER_LOGS[0]}" 2>/dev/null || echo 'unknown')"
  echo "compile_sizes: requested=${COMPILE_SIZES} resolved=$(
    grep -m1 -oE "compile_sizes': *\[[0-9, ]*\]" "${SERVER_LOGS[0]}" 2>/dev/null \
    || echo 'unknown')"
  echo "max_num_batched_tokens: $(grep -m1 -oE "max_num_batched_tokens': *[0-9]+" \
    "${SERVER_LOGS[0]}" 2>/dev/null || echo 'unknown')"
  echo "data_parallel_size: $(grep -m1 -oE "data_parallel_size': *[0-9]+" \
    "${SERVER_LOGS[0]}" 2>/dev/null || echo 'unknown')"
  # MLA tuned-params misses are loud in the worker logs. At TP=8 the per-rank
  # actual_num_q_heads is 12 (96/8) while every tuned entry is keyed at 3
  # (96/32), so 100% misses are EXPECTED in stage 1 and are what stage 2
  # fixes.
  echo "MLA tuned-param misses (all cases):"
  grep -ah -o "No tuned parameters found for the given tuning key: TuningKey(case='[a-z_]*'[^)]*)" \
    "${SERVER_LOGS[@]}" 2>/dev/null \
    | sed -E "s/.*case='([a-z_]+)'.*(max_num_tokens=[0-9]+).*(actual_num_q_heads=[0-9]+).*/  miss: case=\1 \2 \3/" \
    | sort | uniq -c | head -20 || true
  echo "KV cache: $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" \
    "${SERVER_LOGS[0]}" 2>/dev/null || echo 'unknown')"
  local latent_fallback
  latent_fallback=$(cat "${SERVER_LOGS[@]}" 2>/dev/null \
    | grep -c "leaving the latent projections replicated\|leaving it replicated" || true)
  echo "Latent projection fallbacks: ${latent_fallback:-0}"
}
report_split || true

cat <<EOF

============================================================
 ENDPOINT   ${ENDPOINT}/v1
 model      ${MODEL}
 geometry   DP=4 x TP=8 attention, EP=32 experts
 job        ${SLURM_JOB_ID} on ${HOSTS[*]}
 logs       ${LOG_DIR}/server_host{0..3}.log

 Holding the allocation until the time limit or scancel ${SLURM_JOB_ID}.
============================================================

EOF

wait "${SERVER_STEP_PIDS[0]}"
