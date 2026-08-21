#!/usr/bin/env bash
# Correctness stress for the barrier-free (steady_barrier=False) _ws kernels at
# 32 devices, plus the barriered control, for both one-shot and two-shot.
#
#   sbatch --partition=ici-2x2x4-a scripts/debug/slurm_stress_barrier_free_32.sbatch
#
# 32 devices is the case that matters. The 8-device validation is single-host,
# so it has no inter-host arrival skew -- and absorbing skew is precisely what
# the barrier was doing. A barrier-free kernel that passes on one host proves
# only that the slot rotation is self-consistent, not that the data-dependency
# argument holds under real skew.
#SBATCH --job-name=stress-nobar
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${SB_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${SB_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
STRESS="${REPO_DIR}/scripts/allreduce_ws_stress.py"
RUN_DIR="${SB_LOG_ROOT:-${REPO_DIR}/results/stress-nobar-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${STRESS}" ]] || { echo "stress not found: ${STRESS}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# two_shot scatters along rows, so rows % 32 == 0.
export SB_ROWS="${SB_ROWS:-32,64,128}"
export SB_COLS="${SB_COLS:-1024}"
export SB_ROUNDS="${SB_ROUNDS:-1,2,3,8,64,200}"
export SB_REPEATS="${SB_REPEATS:-20}"
export REPO_DIR PYTHON STRESS RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== results: ${RUN_DIR}"
echo "== rows=${SB_ROWS} cols=${SB_COLS} rounds=${SB_ROUNDS} repeats=${SB_REPEATS}"

rc_all=0
# The kernel loop is gone with the one_shot kernel: two_shot_all_reduce_ws is
# the only workspace kernel left, so only the barrier mode varies.
for mode in control nosteady; do
  flag=""
  [[ "${mode}" == nosteady ]] && flag="--no-steady-barrier"
  export SB_FLAG="${flag}"
  echo ""
  echo "##### mode=${mode} #####"
  # SC2016 off: the ${...} below are expanded by the remote shell srun starts,
  # not by this one. They reach it via --export=ALL.
  # shellcheck disable=SC2016
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
    bash -c '
      exec "${PYTHON}" -u "${STRESS}" \
        --multihost \
        --devices 32 \
        --rows "${SB_ROWS}" \
        --cols "${SB_COLS}" \
        --rounds "${SB_ROUNDS}" \
        --repeats "${SB_REPEATS}" \
        ${SB_FLAG}
    ' 2>&1 | tee "${RUN_DIR}/${mode}.log" | tail -24
  rc=${PIPESTATUS[0]}
  echo "##### mode=${mode} rc=${rc} #####"
  [[ ${rc} -ne 0 ]] && rc_all=1
done

echo ""
echo "##### ALL DONE overall_rc=${rc_all} #####"
exit "${rc_all}"
