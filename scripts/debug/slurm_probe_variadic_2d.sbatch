#!/usr/bin/env bash
# Can one kernel reduce K3's two MoE tensors without concatenating them?
#   sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_variadic_2d.sbatch
#SBATCH --job-name=var2d
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${V2_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${V2_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
PROBE="${REPO_DIR}/scripts/debug/probe_variadic_2d.py"
RUN_DIR="${V2_LOG_ROOT:-${REPO_DIR}/results/var2d-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${PROBE}" ]]  || { echo "probe not found: ${PROBE}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# K3's latent 3584 + hidden 7168. The extra widths check the trend holds and
# that a both-lane-aligned pair behaves the same.
export V2_WIDTHS="${V2_WIDTHS:-3584x7168,4096x8192,3584x3584}"
export V2_ARMS="${V2_ARMS:-prod,two_calls,variadic}"
export REPO_DIR PYTHON PROBE RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== results: ${RUN_DIR}"
echo "== widths=${V2_WIDTHS} arms=${V2_ARMS}"

# SC2016 off: expanded by the remote shell srun starts, via --export=ALL.
# shellcheck disable=SC2016
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    exec "${PYTHON}" -u "${PROBE}" \
      --multihost --arms "${V2_ARMS}" --widths "${V2_WIDTHS}" \
      --n-lo 200 --n-hi 800 --repeats 7
  '
