#!/usr/bin/env bash
# Batch R2 bisection: why is the natural kernel ~3us/call slower than the
# flat kernel at identical chunking (17.05 vs 14.02 at 56 KiB, job 405/409)?
#
# Arms, all at 32 devices bf16:
#   - flat twoshot at 32x896 / 32x1792        (reference, chunk = row)
#   - twoshot_nat at 32x896 / 32x1792         (natural code path, per_row=1 --
#                                             chunk-identical to flat)
#   - twoshot_nat / twoshot_sl / twoshot_slconst at the real [T,7168] shapes
#                                             (ws provenance: threaded values
#                                             vs staged inputs vs constants)
# Reading the grid: nat@32x896 ~ flat@32x896 => the kernel path is free and
# the gap is workspace/x provenance; slconst ~ sl tells whether in-body
# constant workspaces beat staged inputs.
#
#   sbatch --partition=ici-2x2x4-e scripts/k3/llar_bisect.sbatch
#SBATCH --job-name=llar-bisect
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${REPO_DIR}/results/llar-bisect-${SLURM_JOB_ID}"

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== repo: ${REPO_DIR}"
echo "== results: ${RUN_DIR}"

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  "${PYTHON}" "${REPO_DIR}/scripts/allreduce_latency_bench.py" \
    --multihost --devices 32 \
    --impls twoshot,twoshot_nat,twoshot_sl,twoshot_slconst \
    --dtype bfloat16 \
    --shapes 32x896,32x1792,4x7168,8x7168,16x7168 \
    --sc-offload-bytes 67108864 \
    --csv "${RUN_DIR}/bisect_32_bfloat16.csv" || true

echo "== done -> ${RUN_DIR}"
