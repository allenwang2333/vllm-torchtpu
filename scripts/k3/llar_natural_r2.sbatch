#!/usr/bin/env bash
# Batch R2 on-device job: natural-shape (glue-free) two-shot out-proj AR.
#
# Three phases on one 4-node slice, sequential:
#   1. Write-only-workspace semantics: chained kernel calls on persistent
#      buffers, numerics vs psum per call, and the step-counter readback that
#      proves XLA hands the custom-call the operand buffers uncopied. The
#      compiled module is dumped (--xla_dump_to) for the copy/slice audit.
#   2. Crossover re-measurement at the REAL shapes [T,7168] (T=1..32), plus
#      the chunk-alignment probe for the old flat kernel (cols 224/448/896).
#   3. Server arm (only if phase 1 passes): full K3 with the new op, c-sweep
#      and a decode trace, for the devpct comparison against job 377.
#
# Submit from the repository root, overriding the partition to a free slice:
#   sbatch --partition=ici-2x2x4-d scripts/k3/llar_natural_r2.sbatch
#SBATCH --job-name=llar-r2
#SBATCH --partition=ici-2x2x4-a
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_DIR}/.venv/bin/python"
RUN_DIR="${LLAR_R2_ROOT:-${REPO_DIR}/results/llar-r2-${SLURM_JOB_ID}}"

[[ -x "${PYTHON}" ]] || { echo "python not found: ${PYTHON}" >&2; exit 2; }

# Compute nodes have no writable home directory; keep every cache in the
# shared job directory.
export HOME="${RUN_DIR}/home"
export XDG_CACHE_HOME="${RUN_DIR}/xdg-cache"
export PYTHONUNBUFFERED=1
export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

echo "== repo: ${REPO_DIR}"
echo "== results: ${RUN_DIR}"
echo "== hosts: $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')"

# The nodes may still run a previous job's Ray cluster (its teardown races the
# next job's start), which kills engine bring-up with "ActorHandle objects are
# not valid across Ray sessions". Pre-clean; both pkills are no-ops when clean.
srun --overlap -N4 -n4 --ntasks-per-node=1 \
  bash -c 'pkill -9 -f "VLLM::Worker" 2>/dev/null; true' || true
srun --overlap -N4 -n4 --ntasks-per-node=1 \
  "${REPO_DIR}/.venv/bin/ray" stop --force >/dev/null 2>&1 || true

# ---------------------------------------------------------------- phase 1
# Write-only semantics + numerics + HLO dump.
if [[ "${LLAR_R2_SKIP_BENCH:-0}" == "1" ]]; then
  echo "== phases 1-2 SKIPPED (LLAR_R2_SKIP_BENCH=1)"
  phase1_ok=1
else
echo "== phase 1: natural-kernel semantics check"
phase1_ok=1
for DT in bfloat16 float32; do
  for T in 4 8; do
    echo "-- dtype=${DT} T=${T}"
    XLA_FLAGS="--xla_dump_to=${RUN_DIR}/hlo-t${T}-${DT} --xla_dump_hlo_as_text" \
    LLAR_CHECK_TOKENS="${T}" LLAR_CHECK_DTYPE="${DT}" \
      srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
        "${PYTHON}" "${REPO_DIR}/scripts/k3/llar_natural_check.py" \
      || phase1_ok=0
  done
done
echo "== phase 1 ok=${phase1_ok}"

# ---------------------------------------------------------------- phase 2
# Crossover at the real shapes PLUS the flat-kernel bisection: 32x896 and
# 32x1792 are chunk-identical between the flat and natural paths, so they
# separate "kernel code path" from "workspace provenance" for the ~3us/call
# gap the first run measured.
echo "== phase 2: crossover sweep at real shapes + flat bisection"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  "${PYTHON}" "${REPO_DIR}/scripts/allreduce_latency_bench.py" \
    --multihost --devices 32 \
    --impls xla,twoshot,twoshot_nat,twoshot_sl,twoshot_slconst,twoshot_static \
    --dtype bfloat16 \
    --shapes 1x7168,2x7168,4x7168,8x7168,16x7168,32x7168,32x896,32x1792 \
    --sc-offload-bytes 67108864 \
    --csv "${RUN_DIR}/natural_32_bfloat16.csv" || true

# Exactness of the natural kernel in f32 (psum and the kernel agree to ~1e-7).
echo "== phase 2c: f32 numerics"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --unbuffered --export=ALL \
  "${PYTHON}" "${REPO_DIR}/scripts/allreduce_latency_bench.py" \
    --multihost --devices 32 \
    --impls xla,twoshot_nat,twoshot_sl,twoshot_static \
    --dtype float32 \
    --shapes 4x7168,8x7168 \
    --sc-offload-bytes 67108864 \
    --csv "${RUN_DIR}/natural_32_float32.csv" || true

fi  # LLAR_R2_SKIP_BENCH

# ---------------------------------------------------------------- phase 3
if [[ "${phase1_ok}" != "1" ]]; then
  echo "== phase 3 SKIPPED: semantics check failed; the op would corrupt "
  echo "   workspace state. Fix the mechanism before serving."
  exit 1
fi

echo "== phase 3: server arm"
# A caller/exported HF_HUB_CACHE pointing at node-local /mnt/scratch breaks
# bring-up on the API host (LocalEntryNotFoundError); the launcher derives
# the shared model store itself when it is unset.
unset HF_HUB_CACHE
export TPU_LOW_LATENCY_ALLREDUCE=1
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export MAX_NUM_SEQS=32
export COMPILE_SIZES=4,8,16,32,8192
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_MODEL_LEN=10240
export PROFILE_DIR="${RUN_DIR}/traces"
export KIMI_K3_LOG_ROOT="${RUN_DIR}/server-log"
# Fresh cache tree: the tier keys do not capture this code change, and a
# stale arm-1 executable would silently become the arm (see the cache-tag
# comment in slurm_launch_k3_tp32_tp.sbatch).
export VLLM_XLA_CACHE_PATH="/mnt/scratch/users/$(id -un)/vllm-xla-cache/kimi-k3-r2-${SLURM_JOB_ID}"

bash "${REPO_DIR}/scripts/k3/slurm_launch_k3_tp32_tp.sbatch" \
  --load-format dummy --kv-cache-dtype fp8 &
LAUNCH_PID=$!

endpoint=""
deadline=$(( $(date +%s) + 4200 ))
while (( $(date +%s) < deadline )); do
  if [[ -f "${KIMI_K3_LOG_ROOT}/endpoint.txt" ]]; then
    endpoint="$(cat "${KIMI_K3_LOG_ROOT}/endpoint.txt")"
    break
  fi
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "launcher exited before the endpoint came up" >&2
    break
  fi
  sleep 15
done

if [[ -z "${endpoint}" ]]; then
  echo "server never came up; giving up" >&2
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
  exit 1
fi
echo "== endpoint: ${endpoint}"

ARM="llar-r2" ENDPOINT="${endpoint}" OUTPUT_DIR="${RUN_DIR}/sweep" \
CONCURRENCIES="1 4 8 16 32" \
  "${REPO_DIR}/scripts/k3/concurrency_sweep.sh" || true

# Decode trace, same capture as jobs 376/377 (phased profiler, one c=1
# request, which pads to the T=4 bucket).
echo "== profile capture"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${endpoint}/start_profile")
echo "start_profile: ${code}"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":128,"temperature":0,"ignore_eos":true}' \
  > "${RUN_DIR}/profile-req.json"
curl -sf "${endpoint}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${RUN_DIR}/profile-req.json" >/dev/null || echo "profile request FAILED"
code=$(curl -s -o /dev/null --max-time 900 -w '%{http_code}' -X POST "${endpoint}/stop_profile")
echo "stop_profile: ${code}"

kill "${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true
echo "== done -> ${RUN_DIR}"
