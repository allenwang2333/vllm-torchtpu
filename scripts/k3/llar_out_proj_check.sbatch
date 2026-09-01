#!/usr/bin/env bash
# On-device check for the llar out-proj op (Batch O, arm 1), on a full
# ici-2x2x4 slice: 4 hosts x 4 chips x 2 cores = 32 JAX devices.
#
# Re-measures XLA's ring against the two-shot kernel at K3's four decode
# out-proj payloads -- T=4, 8, 16, 32 tokens of [T, 7168] bf16, flattened to
# rows x cols = 32 x {896, 1792, 3584, 7168} (the kernel's own view; the rows
# sweep cannot express them) -- and verifies the kernel against psum in both
# bf16 (production dtype, scaled tolerance) and float32 (exact to ~1e-7).
# T=16 (224 KiB) is the crossover tie Batch N measured and the threshold this
# arm ships; it is re-measured here rather than inherited.
#
# Submit from the repository root, overriding the partition to a free slice:
#   sbatch --partition=ici-2x2x4-b scripts/k3/llar_out_proj_check.sbatch
#SBATCH --job-name=llar-op-check
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
# /mnt/scratch is node-local NVMe, so the venv must come off lustre.
PYTHON="${REPO_DIR}/.venv/bin/python"
BENCH="${REPO_DIR}/scripts/allreduce_latency_bench.py"
RUN_DIR="${LLAR_CHECK_LOG_ROOT:-${REPO_DIR}/results/llar-op-check-${SLURM_JOB_ID}}"

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
# The payloads as the kernel sees them: (n_devices, numel/n_devices). T tokens
# of [T, 7168] bf16 is T*14336 bytes.
export LL_SHAPES="${LL_SHAPES:-32x896,32x1792,32x3584,32x7168}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== hosts: ${HOSTS[*]}"
echo "== python: ${PYTHON}"
echo "== results: ${RUN_DIR}"
echo "== shapes: ${LL_SHAPES} (= T=4, 8, 16, 32 tokens at width 7168)"

# One task per host: on TPU, JAX leaves every local device visible to its
# task. jax.distributed.initialize() discovers the slice from
# SLURM_STEP_NODELIST / SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID, which this
# job step provides.
#
# --sc-offload-bytes 64MiB matches what env_override.py sets in production on
# v7 (the per-tensor-core VMEM capacity), so the XLA baseline here is the one
# vLLM actually runs rather than the libtpu default.
for DT in bfloat16 float32; do
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
    "${PYTHON}" "${BENCH}" \
      --multihost \
      --devices 32 \
      --impls xla,twoshot \
      --dtype "${DT}" \
      --shapes "${LL_SHAPES}" \
      --sc-offload-bytes 67108864 \
      --csv "${RUN_DIR}/allreduce_32_${DT}.csv"
done
