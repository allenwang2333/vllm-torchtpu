#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3-verify
#SBATCH --partition=ici-2x2x4-b
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
LOG_DIR="${KIMI_K3_LOG_ROOT:-${REPO_DIR}/results/kimi-k3-verify-${SLURM_JOB_ID}}"
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

# These cache defaults match the original full Kimi-K3 Slurm job. The model
# checkpoint must already exist in the node-local HF cache on every host when
# HF_HUB_OFFLINE=1.
export HF_HOME="${HF_HOME:-/mnt/scratch/hf-hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/mnt/scratch/users/${RUN_USER}/vllm-xla-cache/kimi-k3}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-runtime}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-config}"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${KIMI_K3_TIER2_CACHE_NAMESPACE:-${RUN_USER}_kimi_k3_${SLURM_JOB_ID}}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3}"
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
cleanup() {
  status=$?
  trap - EXIT
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

# This foreground job step is the only place the standalone launcher is used.
# Extra arguments passed to sbatch after the script name are forwarded to vLLM.
srun --overlap -N1 -n1 -w "${HEAD_HOST}" --unbuffered --export=ALL \
  --chdir="${REPO_DIR}" "${LAUNCHER}" "$@"
