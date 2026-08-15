#!/usr/bin/env bash
# Diagnose why one_shot_all_reduce_ws fails at 32 devices while working at 8.
#
# Bisects along three independent axes, each probe in its own srun so a crash
# or hang costs only that probe:
#
#   A. rounds=1 vs rounds>=2  -- is it the basic machinery, or a cross-round
#      race? A single round never reuses a slot or a DMA semaphore, so if
#      rounds=1 already fails the double-buffering is not the culprit.
#   B. barrier_first  -- barrier ahead of the sends, giving up the overlap but
#      keeping the workspace, slots, semaphore arrays and SMEM counter. If this
#      passes where the default fails, the overlap is what breaks at 32
#      devices, not the machinery around it.
#   C. device count -- 8 vs 32 in the *same* multi-host job, so "more devices"
#      is separated from "multi-host". All four processes still take part in
#      every collective; only the mesh is smaller.
#
#   sbatch --partition=ici-2x2x4-c scripts/slurm_ws_diagnose_32.sbatch
#SBATCH --job-name=ws-diag-32
#SBATCH --partition=ici-2x2x4-c
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail   # not -e: a failing probe is data, not a reason to stop

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${LL_ALLREDUCE_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${LL_ALLREDUCE_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
STRESS="${REPO_DIR}/scripts/allreduce_ws_stress.py"
RUN_DIR="${LL_ALLREDUCE_LOG_ROOT:-${REPO_DIR}/results/ws-diag-32-${SLURM_JOB_ID}}"

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
export REPO_DIR PYTHON STRESS RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== hosts: $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')"

# A hung probe would otherwise burn the whole time limit; 240 s is far beyond
# what a passing probe of this size needs.
probe() {   # $1=label  $2..=stress args
  local label="$1"; shift
  echo "---- ${label}"
  timeout --signal=KILL 240 \
    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
      --kill-on-bad-exit=0 \
      "${PYTHON}" -u "${STRESS}" --cols 1024 --repeats 5 "$@" 2>&1 \
    | grep -E "rows=|executions checked|SLICE_FAILURE|Timed out waiting" \
    | sed "s/^/  /"
  local rc=${PIPESTATUS[0]}
  [[ ${rc} -eq 137 ]] && echo "  >> HUNG (killed at 240s)" || echo "  >> exit ${rc}"
}

# Control: the same probe at 8 devices inside this multi-host job. If this
# passes and the 32-device one does not, device count is the variable, not
# multi-host dispatch skew.
probe "C1  8dev  rounds=1"        --devices 8  --rows 1 --rounds 1
probe "C2  8dev  rounds=1,2,8"    --devices 8  --rows 1 --rounds 1,2,8

# Axis A: does a single round already fail at 32 devices?
probe "A1 32dev  rounds=1"        --devices 32 --rows 1 --rounds 1
probe "A2 32dev  rounds=2"        --devices 32 --rows 1 --rounds 2
probe "A3 32dev  rounds=1,2,8"    --devices 32 --rows 1 --rounds 1,2,8

# Axis B: same as A but with the barrier ahead of the sends.
probe "B1 32dev  rounds=1    bf"  --devices 32 --rows 1 --rounds 1     --barrier-first
probe "B2 32dev  rounds=1,2,8 bf" --devices 32 --rows 1 --rounds 1,2,8 --barrier-first

# Axis C: intermediate device counts, to find where it starts failing.
probe "C3 16dev  rounds=1,2,8"    --devices 16 --rows 1 --rounds 1,2,8
probe "C4 12dev  rounds=1,2,8"    --devices 12 --rows 1 --rounds 1,2,8
probe "C5 10dev  rounds=1,2,8"    --devices 10 --rows 1 --rounds 1,2,8

echo "== done"
