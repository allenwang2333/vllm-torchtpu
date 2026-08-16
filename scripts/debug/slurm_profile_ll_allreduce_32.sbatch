#!/usr/bin/env bash
# Per-occurrence device self time for the all-reduce at K3's shapes, 32 cores.
#
# The other half of step 0. The 8-core job isolates *shape*; this one produces
# the number that is directly comparable to the K3 production trace, at the
# device count that trace was taken at. The comparison that decides everything:
#
#   production, rank 0, decode-only   XLA   2x7168   20.1 us/occ
#   production, rank 0, decode-only   kernel 2x3584  ~26   us/occ
#
# If this harness reproduces the XLA figure, it is faithful, and the kernel's
# 26 us is a property of the shape rather than of the production context --
# which means no amount of fusion-preserving or padding work rescues it, and
# the whole approach should stop. If XLA reads well under 20 us here, then
# production carries a per-collective tax this harness cannot see, and the
# finding is that the kernel is more exposed to that tax than XLA's ring.
#
# NOTE: unlike every other 4-host job in this repo, this uses torchrun rather
# than Ray -- there is no model to serve, and Ray would only add a bring-up
# path to debug. torch_tpu's multihost_wrapper wants TPU_WORKER_HOSTNAMES and
# JOB_COMPLETION_INDEX, which Slurm supplies as the nodelist and SLURM_NODEID.
#
#   sbatch scripts/debug/slurm_profile_ll_allreduce_32.sbatch
#   sbatch --partition=ici-2x2x4-b scripts/debug/slurm_profile_ll_allreduce_32.sbatch
#SBATCH --job-name=ll-ar-shapes-32
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${LL_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${REPO_DIR}/results/ll-ar-shapes-32-${SLURM_JOB_ID}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HOSTLIST="$(IFS=,; echo "${HOSTS[*]}")"

# Caches must be per-job and on lustre: compute nodes have no writable $HOME,
# and all four hosts need to see the same trace directory.
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

# The Tier-2 cache lives in /dev/shm under a *namespace* directory, and the
# default namespace is shared by every user on the node -- whoever creates it
# first owns it, and everyone else dies with
#   Check failed: lock_fd_.valid() ... Permission denied
# So namespace it per user and job, exactly as the K3 launcher does. Tier-3 is
# on-disk and gets a per-job directory for the same reason.
RUN_USER="$(id -un)"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${RUN_USER}_llar_${SLURM_JOB_ID}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${RUN_DIR}/torch_tpu_tier3"

# ici-2x2x4 = 2x2x4 chips x 2 cores, the same value the K3 launcher pins.
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export TPU_WORKER_HOSTNAMES="${HOSTLIST}"
# One session id for the whole slice; the wrapper would otherwise mint a
# different one per host and split the trace across four runs.
export TORCH_TPU_XPROF_SESSION_ID="${SLURM_JOB_ID}"
export LL_SHAPES="${LL_SHAPES:-16x1024,2x3584,2x7168,16x3584,16x7168}"
# e.g. LL_EXTRA_ARGS="--gap-dim 4096" to separate the collectives by compute
# instead of chaining them back-to-back.
export LL_EXTRA_ARGS="${LL_EXTRA_ARGS:-}"
export REPO_DIR PYTHON RUN_DIR

echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"
echo "== shapes: ${LL_SHAPES}"
echo "== 4 hosts x 8 cores = 32 ranks (torchrun, not Ray)"

cd "${REPO_DIR}"

# shellcheck disable=SC2317  # invoked via trap
cleanup() {
  scancel --signal=KILL --batch "${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup TERM INT

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    set -euo pipefail
    export JOB_COMPLETION_INDEX="${SLURM_NODEID}"
    eval "$("${PYTHON}" -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES \
           TORCH_TPU_XPROF_SESSION_ID NNODES NODE_RANK MASTER_ADDR MASTER_PORT
    echo "== $(hostname) node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
    exec "${REPO_DIR}/.venv/bin/torchrun" \
      --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
      --nproc_per_node=8 \
      scripts/debug/profile_ll_allreduce_shapes.py \
      --shapes "${LL_SHAPES}" \
      --force-kernel \
      --profile-dir "${RUN_DIR}/traces" \
      ${LL_EXTRA_ARGS}
  '
status=$?

echo "== per-occurrence device self time (comparable to a production trace):"
"${PYTHON}" "${REPO_DIR}/scripts/allreduce_profile_stats.py" \
  "${RUN_DIR}/traces" || true

echo "== exit status: ${status}"
exit "${status}"
