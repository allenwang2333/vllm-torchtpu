#!/usr/bin/env bash
# Decompose the two-shot all-reduce's 12.67 us/occ at 2x7168, 32 cores.
#
# A line fit across 14/28/42 KiB says ~11.9 us of that 12.67 is fixed cost and
# ~0.8 us is payload, so the question "is the kernel any good?" reduces to "is
# ~11.9 us a reasonable floor for two synchronized phases across 32 devices?".
# This runs the ladder that answers it; see the script's docstring.
#
#   sbatch --partition=ici-2x2x4-b scripts/debug/two_shot_ablation.sbatch
#SBATCH --job-name=ts-ablation-32
#SBATCH --partition=ici-2x2x4-b
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
RUN_DIR="${REPO_DIR}/results/ts-ablation-${SLURM_JOB_ID}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HOSTLIST="$(IFS=,; echo "${HOSTS[*]}")"

# Compute nodes have no writable $HOME and all four hosts must see the same
# trace directory, so every cache goes to a per-job directory on lustre.
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

# The Tier-2 cache namespace directory in /dev/shm is shared by every user on
# the node unless namespaced, and whoever creates it first owns it.
RUN_USER="$(id -un)"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${RUN_USER}_tsabl_${SLURM_JOB_ID}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${RUN_DIR}/torch_tpu_tier3"

export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export TPU_WORKER_HOSTNAMES="${HOSTLIST}"
export TORCH_TPU_XPROF_SESSION_ID="${SLURM_JOB_ID}"
export TS_SHAPE="${TS_SHAPE:-2x7168}"
export TS_MODES="${TS_MODES:-noop,barrier1,barrier2,p1,full,full_1bar,full_bf}"
export REPO_DIR PYTHON RUN_DIR

echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"
echo "== shape: ${TS_SHAPE}  modes: ${TS_MODES}"

cd "${REPO_DIR}"

# shellcheck disable=SC2317  # invoked via trap
cleanup() {
  scancel --signal=KILL --batch "${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup TERM INT

# shellcheck disable=SC2016  # single quotes are deliberate: the body runs on
# the remote node, where these expand against that node's environment.
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
      scripts/debug/two_shot_ablation.py \
      --shape "${TS_SHAPE}" \
      --modes "${TS_MODES}" \
      --profile-dir "${RUN_DIR}/traces"
  '
status=$?

echo "== per-occurrence device self time by rung:"
"${PYTHON}" "${REPO_DIR}/scripts/allreduce_profile_stats.py" \
  "${RUN_DIR}/traces" || true

echo "== exit status: ${status}"
exit "${status}"
