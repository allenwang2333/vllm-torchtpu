#!/usr/bin/env bash
# Correctness stress for the variadic (two-operand) workspace all-reduce at 32
# devices, in both barrier modes.
#   sbatch --partition=ici-2x2x4-a scripts/debug/slurm_stress_variadic_32.sbatch
#SBATCH --job-name=stress-var
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${SV_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${SV_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
STRESS="${REPO_DIR}/scripts/allreduce_variadic_stress.py"
RUN_DIR="${SV_LOG_ROOT:-${REPO_DIR}/results/stress-var-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${STRESS}" ]] || { echo "stress not found: ${STRESS}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
export SV_WIDTHS="${SV_WIDTHS:-3584x7168,3584x3584,7168x3584}"
export SV_ROUNDS="${SV_ROUNDS:-1,2,3,8,64,200}"
export SV_REPEATS="${SV_REPEATS:-20}"
export REPO_DIR PYTHON STRESS RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== results: ${RUN_DIR}"

rc_all=0
for mode in control nosteady; do
  flag=""
  [[ "${mode}" == nosteady ]] && flag="--no-steady-barrier"
  echo ""
  echo "##### variadic mode=${mode} #####"
  export SV_FLAG="${flag}"
  # SC2016 off: expanded by the remote shell srun starts, via --export=ALL.
  # shellcheck disable=SC2016
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
    bash -c '
      exec "${PYTHON}" -u "${STRESS}" \
        --multihost --devices 32 \
        --widths "${SV_WIDTHS}" --rounds "${SV_ROUNDS}" \
        --repeats "${SV_REPEATS}" ${SV_FLAG}
    ' 2>&1 | tee "${RUN_DIR}/variadic-${mode}.log" | tail -24
  rc=${PIPESTATUS[0]}
  echo "##### variadic mode=${mode} rc=${rc} #####"
  [[ ${rc} -ne 0 ]] && rc_all=1
done

echo ""
echo "##### ALL DONE overall_rc=${rc_all} #####"
exit "${rc_all}"
