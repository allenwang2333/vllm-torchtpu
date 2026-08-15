#!/usr/bin/env bash
# Probe whether a cross-device Pallas collective is callable from vLLM's
# one-process-per-core layout. Single host, 8 devices, ~2 min.
#
#   sbatch scripts/debug/slurm_probe_pallas_collective.sbatch
#SBATCH --job-name=pallas-collective-probe
#SBATCH --partition=dev-2x2x1
#SBATCH --nodelist=tpu-rack1-w-013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="${PROBE_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${REPO_DIR}/results/pallas-probe-${SLURM_JOB_ID}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# $HOME is not writable on compute nodes; redirect everything that might write.
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
cleanup() { pkill -9 -f "probe_dynamic_shape_guard" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

cd "${REPO_DIR}"

# singlehost_wrapper picks free ports, detects the topology, and prints the
# exports the tpu_dist backend needs. Same recipe as scripts/torchtpu_allreduce_bench.py.
eval "$("${PYTHON}" -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

echo "== node: $(hostname)"
echo "== WORLD_SIZE=${WORLD_SIZE}"
echo "== TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY}"

"${REPO_DIR}/.venv/bin/torchrun" --nproc_per_node="${WORLD_SIZE}" \
  scripts/debug/probe_dynamic_shape_guard.py
status=$?
echo "== probe exit status: ${status}"
exit "${status}"
