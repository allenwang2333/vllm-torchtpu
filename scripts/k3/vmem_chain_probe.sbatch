#!/usr/bin/env bash
# VMEM chaining / w-prefetch probe: can a pallas VMEM out_spec feed the next
# kernel without an HBM roundtrip, and can w staging be scheduled early
# enough to overlap unrelated compute? Single-device per rank; see
# vmem_chain_probe.py for the questions under test.
#
# Submit from the repository root, overriding the partition to a free slice:
#   sbatch --partition=ici-2x2x4-d scripts/k3/vmem_chain_probe.sbatch
#SBATCH --job-name=vmem-chain-probe
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
# The LLO tracing flags segfault the production venv's jax 0.10.2 / libtpu
# 0.0.44.1 at capture start (measured, job 461); use the probe venv.
PYTHON="${PROBE_PYTHON:-/mnt/lustre/home/george/ws-torchtpu/probe-jax311/bin/python}"
RUN_DIR="${VMEM_PROBE_ROOT:-${REPO_DIR}/results/vmem-chain-probe-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# Compute nodes have no writable home directory; keep every cache local.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"

# The nodes may still run a previous job's processes. Pre-clean; the pkills
# are no-ops when clean.
srun --overlap -N4 -n4 --ntasks-per-node=1 \
  bash -c 'pkill -9 -f "vmem_chain_prob[e]" 2>/dev/null; true' || true
sleep 5

# House LIBTPU_INIT_ARGS (diagnostic coordinator / telemetry off) plus the
# xprof LLO tracing flags (VMEM_PROBE_TRACE=0 to disable).
_LLO_FLAGS=""
if [[ "${VMEM_PROBE_TRACE:-1}" == "1" ]]; then
  _LLO_FLAGS="--xla_xprof_enable_custom_call_tracing=true --xla_xprof_register_llo_debug_info=true"
  export VMEM_PROBE_PROFILE="${RUN_DIR}/traces"
fi
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false ${_LLO_FLAGS} ${EXTRA_LIBTPU_FLAGS:-}"

timeout 900 srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered \
  --export=ALL \
  bash -c '
    export XLA_FLAGS="--xla_dump_to='"${RUN_DIR}"'/hlo-r${SLURM_PROCID} --xla_dump_hlo_as_text '"${EXTRA_XLA_FLAGS:-}"'"
    exec "'"${PYTHON}"'" "'"${REPO_DIR}"'/scripts/k3/vmem_chain_probe.py"
  '

echo "== done -> ${RUN_DIR}"
