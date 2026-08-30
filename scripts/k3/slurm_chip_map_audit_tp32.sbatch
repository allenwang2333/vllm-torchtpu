#!/usr/bin/env bash
#SBATCH --job-name=chip-map-audit-tp32
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out
#
# Is the 32-rank chip map a valid partition?
#
# Hierarchical MoE parallelism gives each *chip* a block of experts and splits
# each expert's width between the chip's two cores. Every rank derives its own
# (ep_rank, tp_rank) independently, from its own `jax.devices()` and its own TP
# rank, and then trusts it. Whether the 32 independent answers add up to a
# partition -- every chip claimed exactly twice, once per core -- is a property
# no single worker can see, and nothing in the code checks it.
#
# It matters because the failure is silent. If two ranks computed the same
# ep_rank while a chip went unclaimed, that chip's experts would never be read
# from the checkpoint by anybody: --enable-ep-weight-filter faithfully skips
# weights no rank asked for, the tensors keep their initialized values, nothing
# raises, and the model returns a corrupted distribution. That is exactly the
# TP32 signature in K3_HIERARCHICAL_EP_PLAN.md -- right answers followed by
# repetition loops and unrelated CJK tokens.
#
# THE PART THAT NEEDS 4 HOSTS. Device ids come from the physical slice position.
# vLLM ranks come from Ray's worker ordering -- `ray_distributed_executor`
# assigns rank i over IP-sorted worker metadata. Those two orderings are
# independent facts that happen to coincide on one host, which is why the single
# host TP8 runs (pruned K3, and Kimi-Linear in job 532) cannot answer this.
#
# WHY DUMMY WEIGHTS AND NOT K3. The rank->device map is a property of the
# distributed bring-up, not of the model, and the split is computed during MoE
# *construction*, before any weight is read. So this uses Kimi-Linear (bf16, so
# --load-format=dummy works, unlike mxfp4 K3) purely as a vehicle to instantiate
# 32 real vLLM TPU workers under the real Ray path, and stops as soon as all 32
# ranks have written their audit file. Minutes, not the ~1 h a K3 load costs.
#
# The audit files land on lustre so every host writes to the same directory.
#
# Usage
#   sbatch scripts/k3/slurm_chip_map_audit_tp32.sbatch
#   sbatch --partition=ici-2x2x4-c scripts/k3/slurm_chip_map_audit_tp32.sbatch
set -uo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"   # Kimi-Linear; K3 is 896, pruned75 is 224
RAY_PORT="${RAY_PORT:-6379}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-56}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-300}"
RAY_RESOURCE_TIMEOUT="${RAY_RESOURCE_TIMEOUT:-600}"
AUDIT_TIMEOUT="${AUDIT_TIMEOUT:-2400}"
PORT="${PORT:-8020}"
RUN_USER="$(id -un)"

LOG_DIR="${REPO_DIR}/results/chip-map-audit-${SLURM_JOB_ID}"
AUDIT_DIR="${LOG_DIR}/chip-map"
RAY_LOG="${LOG_DIR}/ray.log"
SERVER_LOG="${LOG_DIR}/server.log"
mkdir -p "${AUDIT_DIR}"

[[ -x "${REPO_DIR}/.venv/bin/ray" ]] || { echo "Ray not found in ${REPO_DIR}/.venv" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HEAD_HOST="${HOSTS[0]}"
HEAD_IP="$(srun --overlap -N1 -n1 -w "${HEAD_HOST}" hostname -I | awk '{print $1}')"
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
export VLLM_TARGET_DEVICE=tpu
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false

# The whole point of the job.
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_CHIP_MAP_AUDIT_DIR="${AUDIT_DIR}"

# $HOME is not writable on compute nodes; see .claude/ENVIRONMENT.md.
export HOME="${LOG_DIR}/home"
export XDG_CACHE_HOME="${LOG_DIR}/xdg-cache"
export HF_HOME="${HF_HOME:-/mnt/lustre/hf-models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/lustre/hf-models/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_MODULES_CACHE="${LOG_DIR}/hf_modules"
export VLLM_CACHE_ROOT="${LOG_DIR}/vllm"
export VLLM_CONFIG_ROOT="${LOG_DIR}/vllm-config"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export TF_CPP_MIN_LOG_LEVEL=1
export TPU_STDERR_LOG_LEVEL=3
export TPU_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"

srun --overlap -N4 -n4 --ntasks-per-node=1 mkdir -p \
  "${HOME}" "${XDG_CACHE_HOME}" "${HF_MODULES_CACHE}" \
  "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${AUDIT_DIR}"

RAY_STEP_PID=
SERVER_PID=
cleanup() {
  status=$?
  trap - EXIT
  [[ -n ${SERVER_PID} ]] && kill "${SERVER_PID}" 2>/dev/null
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

echo "== waiting up to ${RAY_START_TIMEOUT}s for Ray at ${RAY_ADDRESS}"
deadline=$(( $(date +%s) + RAY_START_TIMEOUT ))
until timeout 1 bash -c "</dev/tcp/${HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
  kill -0 "${RAY_STEP_PID}" 2>/dev/null || { echo "Ray exited; see ${RAY_LOG}" >&2; exit 1; }
  (( $(date +%s) >= deadline )) && { echo "Ray timeout; see ${RAY_LOG}" >&2; exit 1; }
  sleep 1
done

echo "== waiting up to ${RAY_RESOURCE_TIMEOUT}s for 32 Ray TPU resources"
deadline=$(( $(date +%s) + RAY_RESOURCE_TIMEOUT ))
tpu_count=0
while (( $(date +%s) < deadline )); do
  tpu_count=$(timeout 30 "${REPO_DIR}/.venv/bin/python" -c '
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("TPU", 0)))
' 2>/dev/null | tail -n 1 || true)
  [[ ${tpu_count:-0} == 32 ]] && break
  echo "   Ray TPU resources: ${tpu_count:-0}/32"
  sleep 10
done
[[ ${tpu_count:-0} == 32 ]] || { echo "Only ${tpu_count:-0}/32 TPUs joined" >&2; exit 1; }

echo "== Ray ready on ${HOSTS[*]}"
echo "== hosts in nodelist order: ${HOSTS[*]}"
echo "== audit dir: ${AUDIT_DIR}"
echo "== git: $(git -C "${REPO_DIR}" rev-parse --short HEAD) $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"

# Dummy weights: nothing is read from the checkpoint, but the config and
# tokenizer still are, so the model must exist in the HF cache.
srun --overlap -N1 -n1 -w "${HEAD_HOST}" --unbuffered --export=ALL \
  --chdir="${REPO_DIR}" \
  "${REPO_DIR}/.venv/bin/vllm" serve "${MODEL}" \
    -tp 32 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --load-format dummy \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 4 \
    --block-size 16 \
    --num-gpu-blocks-override 1024 \
    --attention-backend CUSTOM \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --gpu-memory-utilization 0.80 \
    --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# The audit files are written during MoE construction, well before the engine
# is ready, so poll for the files rather than for /v1/models. That is what makes
# this job minutes instead of an hour, and it also means a later startup failure
# -- which is entirely possible with dummy weights on an untested TP32 config --
# does not cost us the measurement.
echo "== waiting up to ${AUDIT_TIMEOUT}s for 32 rank audit files"
start=$(date +%s)
while :; do
  count=$(find "${AUDIT_DIR}" -name 'rank-*.json' 2>/dev/null | wc -l)
  (( count >= 32 )) && { echo "== all 32 ranks reported after $(( $(date +%s) - start ))s"; break; }
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "== server exited with ${count}/32 rank files; tail:"
    tail -40 "${SERVER_LOG}"
    break
  fi
  if (( $(date +%s) - start > AUDIT_TIMEOUT )); then
    echo "!! timeout with ${count}/32 rank files; tail:"
    tail -40 "${SERVER_LOG}"
    break
  fi
  sleep 10
done

kill "${SERVER_PID}" 2>/dev/null
SERVER_PID=

echo
echo "== which ranks the executor put on which host:"
grep -m1 "node_workers=" "${SERVER_LOG}" || echo "   (not logged)"
grep -m1 "Chip topology:" "${SERVER_LOG}" || echo "   (chip topology not logged)"
echo

"${REPO_DIR}/.venv/bin/python" \
  "${REPO_DIR}/scripts/kernels/collect_chip_map_audit.py" \
  "${AUDIT_DIR}" --num-experts "${NUM_EXPERTS}"
VERDICT=$?
echo "== collector exit ${VERDICT}   (audit files kept in ${AUDIT_DIR})"
exit "${VERDICT}"
