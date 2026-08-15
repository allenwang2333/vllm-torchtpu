#!/usr/bin/env bash
# XLA vs one-shot Pallas all-reduce across a full ici-2x2x4 slice:
# 4 hosts x 4 chips x 2 cores = 16 chips = 32 JAX devices.
#
# Override the partition at submission time to use another free slice:
#   sbatch --partition=ici-2x2x4-b scripts/slurm_allreduce_32.sbatch
#SBATCH --job-name=ll-allreduce-32
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${LL_ALLREDUCE_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
# /mnt/scratch is node-local NVMe, so the venv must come off lustre.
PYTHON="${LL_ALLREDUCE_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
BENCH="${REPO_DIR}/scripts/allreduce_latency_bench.py"
RUN_DIR="${LL_ALLREDUCE_LOG_ROOT:-${REPO_DIR}/results/ll-allreduce-32-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${BENCH}" ]] || { echo "benchmark not found: ${BENCH}" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi

# Compute nodes have no writable home directory; keep every cache in the
# shared job directory.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# two_shot scatters along rows, so it needs rows % num_devices == 0; at 32
# devices that puts its smallest payload at rows=32.
export LL_IMPLS="${LL_IMPLS:-xla,oneshot}"
export LL_ROWS="${LL_ROWS:-1,2,4,8,16,32,64,128,256}"
export LL_COLS="${LL_COLS:-1024}"
export REPO_DIR PYTHON BENCH RUN_DIR LL_IMPLS LL_ROWS LL_COLS
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== hosts: ${HOSTS[*]}"
echo "== python: ${PYTHON}"
echo "== results: ${RUN_DIR}"
echo "== 4 JAX processes x 8 local devices = 32 devices (16 chips)"

# One task per host: on TPU, JAX leaves every local device visible to its task.
# jax.distributed.initialize() discovers the slice from SLURM_STEP_NODELIST /
# SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID, which this job step provides.
#
# --sc-offload-bytes 64MiB matches what env_override.py sets in production, so
# the XLA baseline here is the one vLLM actually runs rather than the default
# that falls off the SparseCore cliff at 64KiB.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    exec "${PYTHON}" "${BENCH}" \
      --multihost \
      --devices 32 \
      --impls "${LL_IMPLS}" \
      --dtype bfloat16 \
      --rows "${LL_ROWS}" \
      --cols "${LL_COLS}" \
      --sc-offload-bytes 67108864 \
      --csv "${RUN_DIR}/allreduce_32.csv"
  '
