#!/usr/bin/env bash
# Shape sensitivity of the one-shot kernel, 8 cores, single host.
#
# The cheap half of step 0. The production A/B measured ~26 us per one-shot
# call at 2x3584; the microbenchmark measured 15.14 us at 16x1024 -- the same
# bytes, eight times the rows. This asks whether that shape difference alone
# accounts for it, which needs no multi-host bring-up: if the kernel is much
# worse at 2xW than at 16x1024 here, the mechanism is confirmed and the
# 32-device run only has to pin down the magnitude.
#
#   sbatch scripts/debug/slurm_profile_ll_allreduce_tp8.sbatch
#SBATCH --job-name=ll-ar-shapes-tp8
#SBATCH --partition=dev-2x2x1
#SBATCH --nodelist=tpu-rack1-w-013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="${LL_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${REPO_DIR}/results/ll-ar-shapes-tp8-${SLURM_JOB_ID}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# $HOME is not writable on compute nodes.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export VLLM_CACHE_ROOT="${RUN_DIR}/vllm-cache"
export VLLM_CONFIG_ROOT="${RUN_DIR}/vllm-config"
export HF_MODULES_CACHE="${RUN_DIR}/hf-modules"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
         "${VLLM_CONFIG_ROOT}" "${HF_MODULES_CACHE}" "${RUN_DIR}/traces"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export PATH="${REPO_DIR}/.venv/bin:${PATH}"

# The default Tier-2 cache namespace in /dev/shm is shared across users on the
# node; whoever creates it first owns it and everyone else gets "Permission
# denied" on the lock file. Namespace it per user and job.
RUN_USER="$(id -un)"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${RUN_USER}_llar_${SLURM_JOB_ID}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${RUN_DIR}/torch_tpu_tier3"

# shellcheck disable=SC2317  # invoked via trap
cleanup() { pkill -9 -f 'profile_ll_allreduce_shapes' 2>/dev/null || true; }
trap cleanup EXIT TERM INT

cd "${REPO_DIR}"

eval "$("${PYTHON}" -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

echo "== node: $(hostname)  WORLD_SIZE=${WORLD_SIZE}"
echo "== results: ${RUN_DIR}"

# --force-kernel so the wide shapes are measurable: at 8 devices the gate lets
# one-shot take up to 32 KiB, and 16x7168 (224 KiB) is far past that.
"${REPO_DIR}/.venv/bin/torchrun" --nproc_per_node="${WORLD_SIZE}" \
  scripts/debug/profile_ll_allreduce_shapes.py \
  --shapes "${LL_SHAPES:-16x1024,2x3584,2x7168,16x3584}" \
  --force-kernel \
  --profile-dir "${RUN_DIR}/traces" \
  "$@"
status=$?

echo "== per-occurrence device self time:"
"${PYTHON}" "${REPO_DIR}/scripts/allreduce_profile_stats.py" \
  "${RUN_DIR}/traces" || true

echo "== exit status: ${status}"
exit "${status}"
