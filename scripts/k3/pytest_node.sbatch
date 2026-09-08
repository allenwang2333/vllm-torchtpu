#!/usr/bin/env bash
# Run a pytest node on the slice (single-device tests run per-task on each
# node's local devices; 4 tasks by house slice rule, all run the same tests).
#
# Submit from the repository root:
#   PYTEST_TARGET=tests/kernels/test_kimi_decode_kda.py::test_decode_kda_wprefetch_matches_decode_kda \
#     sbatch --partition=ici-2x2x4-e scripts/k3/pytest_node.sbatch
#SBATCH --job-name=k3-pytest-node
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${PYTEST_TARGET:?Set PYTEST_TARGET to a pytest node id}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${TEST_PYTHON:-${REPO_DIR}/.venv/bin/python}"
RUN_DIR="${PYTEST_ROOT:-${REPO_DIR}/results/pytest-node-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# The venv's vllm_torchtpu install points at the main worktree; this
# worktree's src must win.
export PYTHONPATH="${REPO_DIR}/src"
# Single-device tests: no multislice boot, diagnostic coordinator off.
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "== hosts: ${HOSTS[*]}"
echo "== target: ${PYTEST_TARGET}"

timeout 1200 srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered \
  --export=ALL \
  "${PYTHON}" -m pytest -x -q "${REPO_DIR}/${PYTEST_TARGET}" 2>&1 | tail -20

echo "== done -> ${RUN_DIR}"
