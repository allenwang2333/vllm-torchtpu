#!/usr/bin/env bash
# Pairwise ICI latency across a full ici-2x2x4 slice:
# 4 hosts x 4 chips x 2 cores = 16 chips = 32 JAX devices.
#
#   sbatch --partition=ici-2x2x4-a scripts/debug/slurm_probe_pairwise_latency.sbatch
#
# Knobs (env): PL_MODE (fanout|matrix|both), PL_ROOT, PL_ROWS, PL_COLS,
#              PL_BARRIER (first|every), PL_NLO, PL_NHI, PL_REPEATS.
#SBATCH --job-name=pairwise-lat
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${PL_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
# /mnt/scratch is node-local NVMe, so the venv must come off lustre.
PYTHON="${PL_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/.venv/bin/python}"
PROBE="${REPO_DIR}/scripts/debug/probe_pairwise_latency.py"
RUN_DIR="${PL_LOG_ROOT:-${REPO_DIR}/results/pairwise-lat-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }
[[ -f "${PROBE}" ]]  || { echo "probe not found: ${PROBE}" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi

# Compute nodes have no writable home; keep every cache in the job directory.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1

export PL_MODE="${PL_MODE:-fanout,allpairs,matrix}"
export PL_ROOT="${PL_ROOT:-0}"
export PL_ROWS="${PL_ROWS:-8}"
export PL_COLS="${PL_COLS:-128}"
export PL_BARRIER="${PL_BARRIER:-first}"
export PL_NLO="${PL_NLO:-400}"
export PL_NHI="${PL_NHI:-1600}"
export PL_REPEATS="${PL_REPEATS:-7}"
export REPO_DIR PYTHON PROBE RUN_DIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== hosts:   ${HOSTS[*]}"
echo "== python:  ${PYTHON}"
echo "== results: ${RUN_DIR}"
echo "== mode=${PL_MODE} root=${PL_ROOT} payload=${PL_ROWS}x${PL_COLS} barrier=${PL_BARRIER}"

# One task per host: JAX leaves every local device visible to its task, and
# jax.distributed.initialize() discovers the slice from SLURM_STEP_NODELIST /
# SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID.
#
# No --sc-offload-bytes here: this probe never emits an XLA collective, so the
# SparseCore threshold cannot affect it, and leaving it unset keeps the run
# independent of that knob.
# SC2016 off: the ${...} below are meant to be expanded by the remote shell
# srun starts, not by this one. They reach it via --export=ALL.
# shellcheck disable=SC2016
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    exec "${PYTHON}" -u "${PROBE}" \
      --multihost \
      --mode "${PL_MODE}" \
      --root "${PL_ROOT}" \
      --rows "${PL_ROWS}" \
      --cols "${PL_COLS}" \
      --cold-barrier "${PL_BARRIER}" \
      --n-lo "${PL_NLO}" \
      --n-hi "${PL_NHI}" \
      --repeats "${PL_REPEATS}" \
      --csv "${RUN_DIR}/pairwise.csv"
  '
