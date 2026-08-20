#!/usr/bin/env bash
# Simulated two-level all-reduce DMA patterns across a full ici-2x2x4 slice:
# 4 hosts x 4 chips x 2 cores = 16 chips = 32 JAX devices.
#
#   sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_hierarchical_dma.sbatch
#
# 32 devices is the only interesting case. At 8 devices (4 chips) the egress cut
# is small and 14 KiB is not yet bandwidth-bound, and hierarchy measured 0.54 us
# slower than flat one-shot / 1.53 us slower than flat two-shot.
#SBATCH --job-name=hier-dma
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${HD_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${HD_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
PROBE="${REPO_DIR}/scripts/debug/probe_hierarchical_dma.py"
RUN_DIR="${HD_LOG_ROOT:-${REPO_DIR}/results/hier-dma-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${PROBE}" ]]  || { echo "probe not found: ${PROBE}" >&2; exit 2; }

export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# 224 -> 14 KiB (K3 post-attention), 336 -> 21 KiB (K3 latent-MoE pair),
# 448 -> 28 KiB, 896 -> 56 KiB to see whether the trend continues past K3.
export HD_COLS="${HD_COLS:-224,336,448,896}"
export HD_ARMS="${HD_ARMS:-flat_oneshot,flat_twoshot,hier_oneshot,hier_oneshot_noc,hier_twoshot}"
export HD_NLO="${HD_NLO:-200}"
export HD_NHI="${HD_NHI:-800}"
export HD_REPEATS="${HD_REPEATS:-7}"
export REPO_DIR PYTHON PROBE RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== results: ${RUN_DIR}"
echo "== cols=${HD_COLS} arms=${HD_ARMS}"

# SC2016 off: the ${...} below are expanded by the remote shell srun starts, not
# by this one. They reach it via --export=ALL.
# shellcheck disable=SC2016
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    exec "${PYTHON}" -u "${PROBE}" \
      --multihost \
      --arms "${HD_ARMS}" \
      --cols "${HD_COLS}" \
      --n-lo "${HD_NLO}" \
      --n-hi "${HD_NHI}" \
      --repeats "${HD_REPEATS}" \
      --csv "${RUN_DIR}/hier_dma.csv"
  '
