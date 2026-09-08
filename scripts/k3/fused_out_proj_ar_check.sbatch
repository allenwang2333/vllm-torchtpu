#!/usr/bin/env bash
# On-device check for the shipped fused o_proj+all-reduce kernel (T=1), on a
# full ici-2x2x4 slice: 4 hosts x 4 chips x 2 cores = 32 JAX devices. See
# fused_out_proj_ar_check.py for what is under test (slot rotation across
# repeated calls, execution-boundary barrier_first, fp32-matmul parity).
#
# Submit from the repository root, overriding the partition to a free slice:
#   sbatch --partition=ici-2x2x4-d scripts/k3/fused_out_proj_ar_check.sbatch
#SBATCH --job-name=fused-out-proj-ar-check
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${FUSED_CHECK_PYTHON:-${REPO_DIR}/.venv/bin/python}"
RUN_DIR="${FUSED_CHECK_ROOT:-${REPO_DIR}/results/fused-out-proj-ar-check-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# Compute nodes have no writable home directory; keep every cache local.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
# The venv's vllm_torchtpu install points at the main worktree; this
# worktree's src must win.
export PYTHONPATH="${REPO_DIR}/src"
# House LIBTPU_INIT_ARGS (diagnostic coordinator / telemetry off).
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false ${EXTRA_LIBTPU_FLAGS:-}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"

# The nodes may still run a previous job's processes (a step killed
# mid-collective can wedge libtpu init for the next job). Pre-clean; the
# pkills are no-ops when clean.
srun --overlap -N4 -n4 --ntasks-per-node=1 \
  bash -c 'pkill -9 -f "fused_out_proj_ar_chec[k]" 2>/dev/null; true' || true
sleep 5

timeout 600 srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered \
  --export=ALL \
  bash -c '
    export XLA_FLAGS="--xla_dump_to='"${RUN_DIR}"'/hlo-r${SLURM_PROCID} --xla_dump_hlo_as_text"
    exec "'"${PYTHON}"'" "'"${REPO_DIR}"'/scripts/k3/'"${CHECK_SCRIPT:-fused_out_proj_ar_check.py}"'"
  '

echo "== done -> ${RUN_DIR}"
