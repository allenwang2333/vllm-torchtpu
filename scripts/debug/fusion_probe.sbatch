#!/usr/bin/env bash
# K3's collective topology at 32 cores: standalone reduction + a fusable pair.
#
# Exists because every earlier microbenchmark measured a collective with no
# neighbours, and so mispredicted production by exactly the effects neighbours
# cause -- XLA's combiner (18 us/layer) and the two K3 sites sharing the shape
# [2,7168]. Both need more than one reduction in the graph to appear at all.
#
# Replaces a 45-minute K3 bring-up with a two-minute job for any question about
# fusion or call-site routing. See the script docstring for the arms.
#
#   sbatch --partition=ici-2x2x4-b scripts/debug/fusion_probe.sbatch
#SBATCH --job-name=fusion-probe-32
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
RUN_DIR="${REPO_DIR}/results/fusion-probe-${SLURM_JOB_ID}"

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

# Per user and per job. The Tier-2 namespace directory in /dev/shm is otherwise
# shared by every user on the node, and a stale entry from a structurally
# different graph is not a hypothetical: it killed a smoke test in this series
# with RuntimeProgramInputMismatch on an unrelated layernorm module.
RUN_USER="$(id -un)"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${RUN_USER}_fusprobe_${SLURM_JOB_ID}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${RUN_DIR}/torch_tpu_tier3"

export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export TPU_WORKER_HOSTNAMES="${HOSTLIST}"
export TORCH_TPU_XPROF_SESSION_ID="${SLURM_JOB_ID}"
# Empty by default so the arm list lives in ONE place, the script's ARMS
# tuple. Hardcoding it here instead silently dropped a newly added arm from
# a run that otherwise looked complete.
export FP_ARMS="${FP_ARMS:-}"
export FP_EXTRA_ARGS="${FP_EXTRA_ARGS:-}"
export REPO_DIR PYTHON RUN_DIR

echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"
echo "== arms: ${FP_ARMS:-<script default>}"

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
      scripts/debug/fusion_probe.py \
      ${FP_ARMS:+--arms "${FP_ARMS}"} \
      --profile-dir "${RUN_DIR}/traces" \
      ${FP_EXTRA_ARGS}
  '
status=$?

# The collective *counts* are the point here, not just the times: the xla arm
# must show one fused [2,3584]+[2,7168] op or the probe is not exercising the
# combiner. The wrapper section below is new -- it reports what the copy_
# writeback costs, which the collective rows do not contain.
echo "== per-occurrence device self time, and collective counts, by arm:"
"${PYTHON}" "${REPO_DIR}/scripts/allreduce_profile_stats.py" \
  "${RUN_DIR}/traces" || true

echo "== exit status: ${status}"
exit "${status}"
