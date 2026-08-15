#!/usr/bin/env bash
# Correctness + latency for the low-latency all-reduce integration on 8 cores.
#
#   sbatch scripts/debug/slurm_test_ll_allreduce_tp8.sbatch
#SBATCH --job-name=ll-ar-tp8
#SBATCH --partition=dev-2x2x1
#SBATCH --nodelist=tpu-rack1-w-013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="${LL_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${REPO_DIR}/results/ll-ar-tp8-${SLURM_JOB_ID}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# $HOME is not writable on compute nodes.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export VLLM_CACHE_ROOT="${RUN_DIR}/vllm-cache"
export VLLM_CONFIG_ROOT="${RUN_DIR}/vllm-config"
export HF_MODULES_CACHE="${RUN_DIR}/hf-modules"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
         "${VLLM_CONFIG_ROOT}" "${HF_MODULES_CACHE}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export PATH="${REPO_DIR}/.venv/bin:${PATH}"

# shellcheck disable=SC2317  # invoked via trap
cleanup() { pkill -9 -f 'test_low_latency_all_reduce' 2>/dev/null || true; }
trap cleanup EXIT TERM INT

cd "${REPO_DIR}"

eval "$("${PYTHON}" -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

echo "== node: $(hostname)  WORLD_SIZE=${WORLD_SIZE}"

"${REPO_DIR}/.venv/bin/torchrun" --nproc_per_node="${WORLD_SIZE}" \
  scripts/debug/test_low_latency_all_reduce.py "$@"
status=$?
echo "== test exit status: ${status}"
exit "${status}"
