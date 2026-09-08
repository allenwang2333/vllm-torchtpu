#!/usr/bin/env bash
# Packed-layout probe (Batch W follow-up): what layout does XLA assign a
# custom-call result whose shape carries a packing dim in the second-minor
# position -- read straight off the after_optimizations dump. Single-device
# compile question; the slice is 4 nodes by house rule but only node 0 runs.
#
# Submit from the repository root, overriding the partition to a free slice:
#   sbatch --partition=ici-2x2x4-d scripts/k3/packed_layout_probe.sbatch
#SBATCH --job-name=packed-layout-probe
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
# The LLO tracing flags segfault the production venv's jax 0.10.2 / libtpu
# 0.0.44.1 at capture start (measured, job 461); the guide requires
# jax>=0.11 / libtpu>=0.0.46. PROBE_PYTHON points at a standalone venv for
# that; the probe script itself only needs jax+numpy.
PYTHON="${PROBE_PYTHON:-${REPO_DIR}/.venv/bin/python}"
RUN_DIR="${PACKED_PROBE_ROOT:-${REPO_DIR}/results/packed-layout-probe-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# Compute nodes have no writable home directory; keep every cache local.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "== hosts: ${HOSTS[*]}"
echo "== results: ${RUN_DIR}"

# The nodes may still run a previous job's processes (a step killed
# mid-collective can wedge libtpu init for the next job). Pre-clean; the
# pkills are no-ops when clean.
srun --overlap -N4 -n4 --ntasks-per-node=1 \
  bash -c 'pkill -9 -f "packed_layout_prob[e]" 2>/dev/null; true' || true
sleep 5

# House LIBTPU_INIT_ARGS (diagnostic coordinator / telemetry off). The
# xprof LLO tracing flags are optional (PACKED_PROBE_TRACE=0 to disable):
# they segfault the production venv at capture start (job 461) and, on the
# no-entry-barrier kernel shape, trip a bundle-instrumentation
# RESOURCE_EXHAUSTED at compile (job 463). BUNDLE_OPTS appends tuning flags.
# PACKED_PROBE_DIAG=1 leaves the diagnostic coordinator ON so a hung kernel
# gets a stall-artifact harvest.
if [[ "${PACKED_PROBE_DIAG:-0}" == "1" ]]; then
  export LIBTPU_INIT_ARGS="${BUNDLE_OPTS:-}"
else
  _LLO_FLAGS=""
  if [[ "${PACKED_PROBE_TRACE:-1}" == "1" ]]; then
    _LLO_FLAGS="--xla_xprof_enable_custom_call_tracing=true --xla_xprof_register_llo_debug_info=true"
    export PACKED_PROBE_PROFILE="${RUN_DIR}/traces"
  fi
  export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false ${_LLO_FLAGS} ${BUNDLE_OPTS:-}"
fi

# One variant per srun step (PACKED_PROBE_VARIANTS is a comma list), with a
# per-step timeout so a hung variant kills its own step instead of the job,
# and a pre-clean so a killed-mid-collective step can't wedge the next one.
IFS=',' read -ra VLIST <<< "${PACKED_PROBE_VARIANTS:-sequential,pipelined}"
for V in "${VLIST[@]}"; do
  echo "== variant: ${V}"
  srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c 'pkill -9 -f "packed_layout_prob[e]" 2>/dev/null; true' || true
  sleep 5
  timeout "${PACKED_PROBE_STEP_TIMEOUT:-420}" srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered \
    --export=ALL \
    bash -c '
      export XLA_FLAGS="--xla_dump_to='"${RUN_DIR}"'/hlo-'"${V}"'-r${SLURM_PROCID} --xla_dump_hlo_as_text"
      export PACKED_PROBE_VARIANTS="'"${V}"'"
      exec "'"${PYTHON}"'" "'"${REPO_DIR}"'/scripts/k3/packed_layout_probe.py"
    ' || echo "== variant ${V} FAILED/HUNG (exit $?)"
done

echo "== dumps:"
ls -d "${RUN_DIR}"/hlo-* 2>/dev/null | head || true
echo "== done -> ${RUN_DIR}"
