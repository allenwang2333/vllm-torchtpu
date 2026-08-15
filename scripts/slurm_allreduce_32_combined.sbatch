#!/usr/bin/env bash
# XLA vs one_shot vs one_shot_ws vs two_shot across a full ici-2x2x4 slice:
# 4 hosts x 4 chips x 2 cores = 16 chips = 32 JAX devices.
#
#   sbatch --partition=ici-2x2x4-b scripts/slurm_allreduce_32_combined.sbatch
#SBATCH --job-name=ll-allreduce-32c
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail   # deliberately not -e: a crashing step must not kill the run

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${LL_ALLREDUCE_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${LL_ALLREDUCE_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
BENCH="${REPO_DIR}/scripts/allreduce_latency_bench.py"
STRESS="${REPO_DIR}/scripts/allreduce_ws_stress.py"
RUN_DIR="${LL_ALLREDUCE_LOG_ROOT:-${REPO_DIR}/results/ll-allreduce-32c-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${BENCH}"  ]] || { echo "benchmark not found: ${BENCH}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1   # block-buffered stdout previously hid which config crashed
export REPO_DIR PYTHON BENCH STRESS RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== hosts: $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')"
echo "== results: ${RUN_DIR}"

# One shape per step. one_shot_ws takes down the slice above some payload size
# (128 KiB per shard at 8 devices; its scratch grows with device count, so the
# 32-device threshold is unknown), and a crash kills the whole process -- so
# isolating each shape keeps one bad size from costing every other row.
run_step() {   # $1=label  $2=rows  $3=impls  $4..=extra args
  local label="$1" rows="$2" impls="$3"; shift 3
  echo "---- ${label}: rows=${rows} impls=${impls}"
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
    --kill-on-bad-exit=0 \
    "${PYTHON}" -u "${BENCH}" --multihost --devices 32 --dtype bfloat16 \
      --impls "${impls}" --rows "${rows}" --cols 1024 \
      --sc-offload-bytes 67108864 "$@" 2>&1 \
    | grep -E "^ *${rows} +[0-9]|!!|SLICE_FAILURE|Timed out waiting" \
    | sed "s/^/  /"
  echo "  (step exit ${PIPESTATUS[0]})"
}

# 1. Correctness first: never trust a _ws timing number without it.
echo "=== stress: one_shot_ws correctness at 32 devices ==="
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  --kill-on-bad-exit=0 \
  "${PYTHON}" -u "${STRESS}" --devices 32 --rows 1,4,16 --rounds 1,2,3,8,32 \
    --repeats 10 2>&1 \
  | grep -E "rows=|executions checked|SLICE_FAILURE|Timed out waiting" | sed "s/^/  /"

# 2. Timing sweep. two_shot scatters along rows, so at 32 devices it needs
#    rows % 32 == 0 -- hence the split at rows=32.
echo "=== timing sweep ==="
for R in 1 2 4 8 16; do
  run_step "small" "${R}" "xla,oneshot,oneshot_ws"
done
for R in 32 64 128 256 512; do
  run_step "large" "${R}" "xla,oneshot,twoshot"
done
# _ws separately at the two-shot sizes: if it survives, it belongs in the table
# there too; if it takes the slice down, only this step is lost.
for R in 32 64; do
  run_step "ws-probe" "${R}" "xla,oneshot_ws"
done

echo "== done"
