#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3-verify
#SBATCH --partition=ici-2x2x4-b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# Minimal Slurm wrapper for launch_kimi_k3_tp32_ep.sh. It starts Ray on the
# allocated hosts, waits for all 32 TPU resources, then runs the standalone
# server launcher on the Ray head. Submit this file from the repository root.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-56}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-300}"
RAY_RESOURCE_TIMEOUT="${RAY_RESOURCE_TIMEOUT:-600}"
PORT="${PORT:-8000}"
RUN_USER="$(id -un)"

# This job starts the server and holds it up until cancelled or the time limit.
# It deliberately runs NO eval: bringing K3 up at TP32 costs ~40 minutes, and
# coupling a measurement to that load means every mistake in the measurement --
# a wrong dataset path, a bad flag -- costs another 40 minutes to retry. Run
# evals and benchmarks against the endpoint separately, as many times as needed.
MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
# Off by default so this file keeps behaving as it always has; the hierarchical
# arm sets it to 1.
export TPU_MOE_HIERARCHICAL_EP="${TPU_MOE_HIERARCHICAL_EP:-0}"
# Pin this. vLLM auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for K3's
# architecture whenever the variable is *absent* from the environment
# (vllm/config/vllm.py, "Auto-enabling"), and that disables the torch.compile
# pipeline outright -- an ~18x decode slowdown, 344 ms TPOT instead of 19 ms.
# Job 669 measured exactly that because the variable happened to be exported in
# the shell that submitted the earlier runs and not in the one that submitted
# it. A benchmark must not depend on the submitting shell.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# Independent of the flag above: shards the latent-MoE down/up projections
# across a chip's two cores instead of replicating them.
export TPU_LATENT_PROJ_INTRA_CHIP_TP="${TPU_LATENT_PROJ_INTRA_CHIP_TP:-0}"

# Routed-row count at/below which the MoE permute/combine run as TensorCore
# one-hot matmuls instead of the SparseCore gather/reduce kernels. Dispatch is
# `if token_indices_sorted.shape[0] <= threshold`, and the count is
# num_tokens * topk.
#
# This cannot be a blanket default in either direction, which is the whole
# reason for the branch below.
#
# 0 (upstream's default since #485 reverted the auto-derivation in #371) breaks
# a single-token decode bucket. Below one SparseCore block ragged_gather_v2 pads
# to a whole block, and at bucket 1 that padding path fails in Mosaic:
#
#   ragged_gather_v2.py:182 -> emit_pipeline -> _make_pipeline_allocations
#   ValueError: Expected input and output shapes are the same after
#              multiplying the second-minor dimension by the bitwidths.
#
# 64 breaks a *multi*-bucket config. At topk=16 it puts bucket 2 (32 routed
# rows) on the one-hot path while buckets 16 (256) and 512 (8192) stay on
# SparseCore, so buckets sharing a compile range get different subgraph
# structure and compilation refuses:
#
#   vllm_torchtpu.compilation.shape_variants.ShapeSpecializationError:
#   artifact_compile_range_2_2_subgraph_0 was compiled by another trace of this
#   model whose inputs differ
#
# Measured: [2,16,512] never comes up at 64 (jobs 305/306), and [1] needs a
# non-zero threshold (302/304 vs earlier). That the one-hot path cannot be
# *partially* enabled across buckets is the likely reason #371 was reverted at
# all, so treat 64 as a decode-only setting and not as a fix to carry everywhere.
#
# CAUTION: threshold 0 is necessary but *not* sufficient for a multi-bucket
# config. Job 307 ran [2,16,512] at 0, on a fresh cache, and still died with the
# same ShapeSpecializationError at artifact_compile_range_2_2_subgraph_0. An
# earlier version of this comment claimed 293 proved [2,16,512] boots at 0;
# 293 did boot, but with a different flag set, so it never licensed that
# generalisation. Verified for 307 vs 293: identical src (no src/ commits
# between them), identical max_model_len/max_num_batched_tokens/max_num_seqs/
# compile_sizes, and identical threshold -- explicit 0 is indistinguishable from
# unset, since envs.py reads `int(os.getenv(...) or "0")`. The failure is
# deterministic (6 occurrences in each of 305/306/307, 0 in 293) and its cause is
# still unidentified. GSM8K on a multi-bucket config is therefore UNVALIDATED.
#
# So: single bucket -> 64, several buckets -> 0. 64 is what the reverted
# auto-derivation computed on TPU7x: SparseCoreInfo(num_cores=2,
# num_subcores=16, num_lanes=16) gives block_rows = 512, min(512-1, cap 64) = 64.
# Set the variable explicitly to override the choice.
if [[ -z "${ONEHOT_MOE_PERMUTE_THRESHOLD:-}" ]]; then
  if [[ -n "${COMPILE_SIZES:-}" && "${COMPILE_SIZES}" != *,* ]]; then
    ONEHOT_MOE_PERMUTE_THRESHOLD=64
  else
    ONEHOT_MOE_PERMUTE_THRESHOLD=0
  fi
fi
export ONEHOT_MOE_PERMUTE_THRESHOLD
# Default ON here, unlike the launcher, which defaults it off. With async
# scheduling off the host scheduler does not overlap device execution and adds
# ~8.4 ms to every decode step -- half again the step time -- which swamps any
# effect being measured. Job 688 recorded 16.1 ms/step of device work and a
# 24.5 ms TPOT for exactly this reason, and looked like a 29% regression next to
# a baseline that had the flag on. Comparisons across this flag are meaningless.
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
# Pinned for the same reason as the flags above: job 643 ran with 16 (exported
# interactively) while the launcher's own default is 8, so its concurrency sweep
# is not comparable to a default run above c=8 -- at c=16 a max_num_seqs=8 server
# queues half the requests and reports a flattering TPOT over a shorter batch.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Route TP all-reduces whose payload sits inside the measured crossover to the
# one-shot Pallas kernel instead of XLA's ring. Pinned here for the same reason
# as everything above: it changes decode speed, so it must come from the batch
# script and not from whatever shell happened to submit the job.
export TPU_LOW_LATENCY_ALLREDUCE="${TPU_LOW_LATENCY_ALLREDUCE:-0}"
# Default 1: two-shot is the only kernel left, so 0 makes LLAR a no-op.
export TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT="${TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT:-1}"
# Reduce K3's latent-MoE pair as one collective instead of upstream's two.
export TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT="${TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT:-0}"
# Do it with the two-operand kernel, dropping the torch.cat and both reshapes.
# Supersedes MOE_CONCAT at the same call site rather than composing with it, but
# still requires it, since that is what patches the MoE tail at all.
export TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC="${TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC:-0}"
# Remove the steady-state barrier from the _ws kernels (round 0 keeps its own).
export TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER="${TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER:-0}"
# Format is ROWSxCOLS, not a bare width: the parser does
# tuple(int(v) for v in value.split("x")) and compares against the plan's
# (rows, cols), so "7168" builds a 1-tuple that matches nothing and falls back
# silently-ish. Use 1x7168 for K3 at compile size 1.
#
# Which single shape gets the communicator's entry op. Without it the selector
# takes the *smallest* payload that takes a kernel, which on K3 is the 7 KiB
# (1,3584) latent reduce -- leaving the 14 KiB (1,7168) post-attention reduce on
# XLA, where it costs 17.1 us against two_shot's ~12.5. That is ~433 us/step over
# 95 layers, and it is why an arm run without this measured 15.73 ms against
# 15.31 for the same nominal config. Set it to the hidden size to put the
# post-attention reduce on the kernel.
export TPU_LOW_LATENCY_ALLREDUCE_SHAPE="${TPU_LOW_LATENCY_ALLREDUCE_SHAPE:-}"

# MoE routing renormalize: fold vs .sum(dim=-1). Nothing to do with the
# collectives, but it is A/B'd through the same harness, so it gets the same
# treatment -- an explicit default and its own cache tag.
export TPU_MOE_RENORMALIZE_FOLD="${TPU_MOE_RENORMALIZE_FOLD:-1}"

# Which token buckets get compiled, and the scheduler's token budget. Both are
# consumed by launch_kimi_k3_tp32_ep.sh, which already builds the
# --compilation-config and --max-num-batched-tokens flags from them and, at its
# lines 28-31, refuses to start unless COMPILE_SIZES contains a bucket for
# MAX_NUM_BATCHED_TOKENS.
#
# They are re-exported here, rather than left to the submitting shell, because
# they change decode speed and the graph -- the same argument the flags above
# make. Getting there took four launches of the MoE-fold A/B:
#
#   1-3. compile_sizes left at the inner default [2, 16, 512] while the run
#        being reproduced had [1]. A different decode regime, and at 224/336 KiB
#        past the all-reduce gate's ceiling, so the kernel under test was not
#        exercised at all. Rounds 2 and 3 also proved buckets do not follow
#        max_num_seqs.
#   4.   Passed as a trailing --compilation-config argument instead. That works
#        -- it reached vLLM as compile_sizes [1] -- but it *bypassed the guard*,
#        which then validated the inner default 512 against the inner default
#        bucket list and passed, while the program was actually compiled for 1.
#        The server came up and died on its first request in _prepare_inputs,
#        writing 12 prompt positions into a buffer sized for one token.
#
# So route through the variables, never a trailing argument: the guard is the
# thing that catches an incoherent pair, and only the variables reach it.
export COMPILE_SIZES="${COMPILE_SIZES:-}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
LOG_DIR="${KIMI_K3_LOG_ROOT:-${REPO_DIR}/results/kimi-k3-verify-${SLURM_JOB_ID}}"
# Set PROFILE_DIR to make /start_profile work. Both halves are required and
# neither can be turned on after the server is up: the phased profiler is read
# at worker init, and the trace directory is a server CLI flag. Without them
# /start_profile answers 404 and the arm script cheerfully reports "profiled
# tokens: 128" having captured nothing -- which is how job 672 finished its
# whole sweep before the missing profile surfaced.
PROFILE_ARGS=()
if [[ -n "${PROFILE_DIR:-}" ]]; then
  mkdir -p "${PROFILE_DIR}"
  export USE_PHASED_PROFILER=true
  # `profiler=torch` is not optional: vLLM's ProfilerConfig validator rejects a
  # torch_profiler_dir without it ("only applicable when profiler is set to
  # 'torch'") and the server exits before Ray even starts. Three flags, all
  # required, none of them settable after startup.
  PROFILE_ARGS=(--profiler-config.profiler=torch
                --profiler-config.torch_profiler_dir="${PROFILE_DIR}")
fi
RAY_LOG="${LOG_DIR}/ray.log"
LAUNCHER="${REPO_DIR}/scripts/k3/launch_kimi_k3_tp32_ep.sh"

[[ -x "${LAUNCHER}" ]] || { echo "Launcher not found: ${LAUNCHER}" >&2; exit 2; }
[[ -x "${REPO_DIR}/.venv/bin/ray" ]] || { echo "Ray not found in ${REPO_DIR}/.venv" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HEAD_HOST="${HOSTS[0]}"
HEAD_IP="${KIMI_K3_HEAD_IP:-$(srun --overlap -N1 -n1 -w "${HEAD_HOST}" hostname -I | awk '{print $1}')}"
[[ -n "${HEAD_IP}" ]] || { echo "Could not resolve ${HEAD_HOST}" >&2; exit 2; }

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_USAGE_STATS_ENABLED=0
export TPU_MULTIHOST_BACKEND=ray
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export JAX_PLATFORMS=""
export PORT

# Checkpoint location. /mnt/scratch is node-local NVMe and its contents differ
# per host, so "the node-local cache" is only usable if EVERY host has the
# snapshot -- one host missing it fails the job an hour in, after Ray, worker
# bring-up and a partial load. Probe all four and fall back to shared lustre.
#
# The two mounts also use different layouts: models--* sits at the root of
# /mnt/scratch/hf-hub but under hub/ on lustre. Set HF_HUB_CACHE explicitly,
# never derive it from HF_HOME -- an inherited value silently wins over HF_HOME
# and has redirected loads to a partial copy before.
MODEL_DIR="models--${MODEL//\//--}"
all_hosts_have() {
  srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c "[ -d '$1/${MODEL_DIR}/snapshots' ]" >/dev/null 2>&1
}
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  echo "Using caller-supplied HF_HUB_CACHE=${HF_HUB_CACHE}"
elif all_hosts_have /mnt/scratch/hf-hub; then
  export HF_HUB_CACHE=/mnt/scratch/hf-hub
elif all_hosts_have /mnt/scratch/hf-hub/hub; then
  export HF_HUB_CACHE=/mnt/scratch/hf-hub/hub
elif [[ -d "/mnt/lustre/hf-models/hub/${MODEL_DIR}/snapshots" ]]; then
  export HF_HUB_CACHE=/mnt/lustre/hf-models/hub
  echo "Checkpoint served from lustre -- slower load, but shared by all hosts"
else
  echo "${MODEL} not found on any host's cache root" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-$(dirname "${HF_HUB_CACHE}")}"
echo "Checkpoint: ${HF_HUB_CACHE}/${MODEL_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# The cache holds *compiled executables*, so any flag that changes the graph
# must change the path. TPU_LOW_LATENCY_ALLREDUCE adds two workspace tensors as
# graph inputs; sharing a cache across that flag made a treatment run pick up
# the control's executable and die at execution with
#   execution supplied 2135 arguments but compiled program expected 2133
# -- off by exactly the two workspaces. The Tier-2/Tier-3 keys do not capture
# it, so keep the arms in separate trees.
#
# The same applies to the MoE pair flags, and more sharply. The concat op takes
# three workspace tensors (rs, ag, step); the variadic op takes five (rsa, rsb,
# aga, agb, step). Those are different argument counts on the same call site, so
# sharing a cache across these flags reproduces the arity crash above exactly.
_LLAR_TAG=""
[[ "${TPU_LOW_LATENCY_ALLREDUCE:-0}" == "1" ]] && _LLAR_TAG="-llar"
[[ "${TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT:-0}" == "1" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-cat"
[[ "${TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC:-0}" == "1" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-var"
# This one is the most dangerous to share. Barrier-free changes the kernel *body*
# and not the op signature, so unlike the workspace-count flags above it produces
# no arity mismatch to fail on -- a shared cache would silently execute the
# barriered program and the A/B would measure nothing at all.
[[ "${TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER:-0}" == "1" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-nobar"
# The override decides *which* shape carries the kernel, so it changes the
# workspace shapes that are graph inputs. Sharing a cache across it is the same
# arity hazard as the flags above.
[[ -n "${TPU_LOW_LATENCY_ALLREDUCE_SHAPE:-}" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-s${TPU_LOW_LATENCY_ALLREDUCE_SHAPE}"
# The fold changes the emitted HLO for every MoE layer, so it needs its own
# cache tree for the same reason the flags above do: a shared cache would
# replay whichever program was compiled first and the A/B would measure noise.
[[ "${TPU_MOE_RENORMALIZE_FOLD:-1}" == "0" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-nofold"
# Different buckets are different programs, so they must not share a cache tree
# for the same reason the flags above must not.
[[ -n "${COMPILE_SIZES:-}" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-cs${COMPILE_SIZES//,/_}"
# Selects between the one-hot matmul and SparseCore gather kernels, so the
# emitted programs differ and the cache trees must too.
[[ "${ONEHOT_MOE_PERMUTE_THRESHOLD}" != "0" ]] && \
  _LLAR_TAG="${_LLAR_TAG}-oh${ONEHOT_MOE_PERMUTE_THRESHOLD}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/mnt/scratch/users/${RUN_USER}/vllm-xla-cache/kimi-k3${_LLAR_TAG}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-runtime}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${VLLM_XLA_CACHE_PATH}/vllm-config}"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${KIMI_K3_TIER2_CACHE_NAMESPACE:-${RUN_USER}_kimi_k3_${SLURM_JOB_ID}}"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT:-${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"

mkdir -p "${LOG_DIR}"
srun --overlap -N4 -n4 --ntasks-per-node=1 mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT}"

RAY_STEP_PID=
SERVER_STEP_PID=
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n ${SERVER_STEP_PID} ]] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
  # Workers outlive the launcher often enough to strand the TPUs for the next
  # job, and at TP32 that is four hosts held hostage rather than one.
  timeout 60 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    pkill -9 -f 'VLLM::Worker' >/dev/null 2>&1 || true
  if [[ -n ${RAY_STEP_PID} ]] && kill -0 "${RAY_STEP_PID}" 2>/dev/null; then
    kill "${RAY_STEP_PID}" 2>/dev/null || true
    wait "${RAY_STEP_PID}" 2>/dev/null || true
  fi
  timeout 30 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    ray stop --force >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export KIMI_K3_RAY_HEAD_IP="${HEAD_IP}"
export KIMI_K3_RAY_PORT="${RAY_PORT}"
export KIMI_K3_RAY_NUM_CPUS="${RAY_NUM_CPUS}"

# Start one persistent Ray node per host. Task zero becomes the head; the
# remaining tasks wait for its port and then join as workers.
# shellcheck disable=SC2016
srun --overlap -N4 -n4 --ntasks-per-node=1 --unbuffered --export=ALL \
  bash -c '
    node_ip=$(hostname -I | awk "{print \$1}")
    if [[ ${SLURM_PROCID} -eq 0 ]]; then
      exec ray start --block --head \
        --node-ip-address="${node_ip}" \
        --port="${KIMI_K3_RAY_PORT}" \
        --include-dashboard=false \
        --num-cpus="${KIMI_K3_RAY_NUM_CPUS}"
    fi
    until timeout 1 bash -c \
        "</dev/tcp/${KIMI_K3_RAY_HEAD_IP}/${KIMI_K3_RAY_PORT}" 2>/dev/null; do
      sleep 1
    done
    exec ray start --block \
      --address="${KIMI_K3_RAY_HEAD_IP}:${KIMI_K3_RAY_PORT}" \
      --node-ip-address="${node_ip}" \
      --num-cpus="${KIMI_K3_RAY_NUM_CPUS}"
  ' >"${RAY_LOG}" 2>&1 &
RAY_STEP_PID=$!

echo "Waiting up to ${RAY_START_TIMEOUT}s for Ray at ${RAY_ADDRESS}"
deadline=$(( $(date +%s) + RAY_START_TIMEOUT ))
until timeout 1 bash -c "</dev/tcp/${HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
  if ! kill -0 "${RAY_STEP_PID}" 2>/dev/null; then
    echo "Ray exited during startup; see ${RAY_LOG}" >&2
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out waiting for Ray; see ${RAY_LOG}" >&2
    exit 1
  fi
  sleep 1
done

echo "Waiting up to ${RAY_RESOURCE_TIMEOUT}s for 32 Ray TPU resources"
deadline=$(( $(date +%s) + RAY_RESOURCE_TIMEOUT ))
tpu_count=0
while (( $(date +%s) < deadline )); do
  tpu_count=$(timeout 30 "${REPO_DIR}/.venv/bin/python" -c '
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("TPU", 0)))
' 2>/dev/null | tail -n 1 || true)
  [[ ${tpu_count:-0} == 32 ]] && break
  echo "  Ray TPU resources: ${tpu_count:-0}/32"
  sleep 10
done
if [[ ${tpu_count:-0} != 32 ]]; then
  echo "Only ${tpu_count:-0}/32 TPU resources joined; see ${RAY_LOG}" >&2
  exit 1
fi

echo "Ray is ready on ${HOSTS[*]} with 32 TPU resources"
echo "Kimi-K3 endpoint will be http://${HEAD_IP}:${PORT}/v1"
echo "Ray log: ${RAY_LOG}"

echo "Hierarchical MoE parallelism: TPU_MOE_HIERARCHICAL_EP=${TPU_MOE_HIERARCHICAL_EP}"

# Echo the requested configuration *before* the server starts, rather than only
# in report_split below. These are two different things and only one of them
# needs a running server: this is the set of env vars the job was submitted
# with, known already, whereas report_split greps SERVER_LOG for what the
# dispatcher actually did.
#
# Keeping them together cost three slices on 2026-08-21. Jobs 305, 306 and 307
# all died during engine-core startup, so report_split never ran and each run
# discarded the only record of its own configuration. Diagnosing 307 then had to
# proceed by inference from a hand-set cache-path name, which produced two wrong
# root causes in a row -- the failure was blamed on
# ONEHOT_MOE_PERMUTE_THRESHOLD, which is in fact identical (0) in both the
# failing run and the passing one it was compared against. A failed startup is
# when this information is most needed, so it must be printed unconditionally.
echo "Requested config:" \
     "moe_concat=${TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT}" \
     "moe_variadic=${TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC}" \
     "no_steady_barrier=${TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER}" \
     "shape=${TPU_LOW_LATENCY_ALLREDUCE_SHAPE:-auto}" \
     "two_shot=${TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT:-unset}" \
     "moe_renormalize_fold=${TPU_MOE_RENORMALIZE_FOLD}" \
     "latent_proj_intra_chip=${TPU_LATENT_PROJ_INTRA_CHIP_TP:-unset}" \
     "compile_sizes=[${COMPILE_SIZES:-launcher default}]" \
     "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-launcher default}" \
     "max_num_seqs=${MAX_NUM_SEQS}" \
     "async_scheduling=${ASYNC_SCHEDULING}" \
     "onehot_moe_permute_threshold=${ONEHOT_MOE_PERMUTE_THRESHOLD}" \
     "hierarchical_ep=${TPU_MOE_HIERARCHICAL_EP}" \
     "xla_cache=${VLLM_XLA_CACHE_PATH}"

# The server runs as a background job step so this script can detect readiness,
# report the endpoint, and then simply hold the allocation. Extra arguments
# passed to sbatch after the script name are forwarded to vLLM.
SERVER_LOG="${LOG_DIR}/server.log"
srun --overlap -N1 -n1 -w "${HEAD_HOST}" --unbuffered --export=ALL \
  --chdir="${REPO_DIR}" "${LAUNCHER}" "${PROFILE_ARGS[@]}" "$@" >"${SERVER_LOG}" 2>&1 &
SERVER_STEP_PID=$!

echo "Waiting up to ${READY_TIMEOUT}s for the endpoint (log: ${SERVER_LOG})"
start=$(date +%s)
ready=0
while (( $(date +%s) - start < READY_TIMEOUT )); do
  if curl -sf "http://${HEAD_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server exited during startup; tail:" >&2
    tail -60 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 10
done
if (( ready != 1 )); then
  echo "Endpoint never came up within ${READY_TIMEOUT}s; tail:" >&2
  tail -60 "${SERVER_LOG}" >&2
  exit 1
fi
echo "SERVER_READY after $(( $(date +%s) - start ))s"

# Record the endpoint before any diagnostics run: a reporting failure must not
# leave callers without the URL of a server that is up and serving.
ENDPOINT="http://${HEAD_IP}:${PORT}"
printf '%s\n' "${ENDPOINT}" > "${LOG_DIR}/endpoint.txt"

# Whether the chip split actually engaged. Ray de-duplicates repeated worker log
# lines, so the raw count understates badly -- count the [repeated Nx] suffixes
# too or a healthy TP32 run looks like it only patched four layers.
#
# This whole block is diagnostics: it must never be able to end the job. Job 610
# lost a healthy server 1933s in because one grep child was killed (the head node
# is also loading K3's weights, so the kernel reclaims whatever it can) and
# `set -e` turned that into a teardown. Hence report_split() ... || true, and
# line-based greps rather than a greedy `.*` across a 38 MB log.
report_split() {
  local marker="Hierarchical MoE parallelism: EP between chips"
  local raw suppressed fallback
  # `|| true`, never `|| echo 0`: grep -c already prints "0" when it finds
  # nothing and *also* exits 1, so `|| echo 0` yields the two-line value
  # "0\n0" and the arithmetic below dies with a syntax error.
  raw=$(grep -Fc "${marker}" "${SERVER_LOG}" 2>/dev/null || true)
  suppressed=$(grep -F "${marker}" "${SERVER_LOG}" 2>/dev/null | cut -c1-400 \
    | grep -oE "repeated [0-9]+x" | grep -oE "[0-9]+" \
    | awk '{n+=$1} END {print n+0}' || true)
  fallback=$(grep -c "falling back to flat expert parallelism" "${SERVER_LOG}" 2>/dev/null || true)
  echo "Split check: config lines=$(( ${raw:-0} + ${suppressed:-0} )) (raw ${raw:-0} + suppressed ${suppressed:-0}), fallbacks=${fallback:-0}"
  grep -m1 -o "Chip topology: .*" "${SERVER_LOG}" 2>/dev/null | head -1 || true
  # The filter logs the CALLER's rank-derived ep_size/ep_rank, not the patched
  # ones, so those numbers look wrong even when it is working. The expert COUNT
  # is the honest signal: 56/896 is a chip block (896/16), 28/896 is unpatched.
  grep -o "loading [0-9]*/[0-9]* experts" "${SERVER_LOG}" 2>/dev/null | sort | uniq -c || true
  # Same idea for the latent projections: the honest signal is the per-core
  # column count. 1792/3584 means the down-proj really is split across a chip;
  # absent lines mean it silently stayed replicated.
  grep -o "Intra-chip group ready: .*" "${SERVER_LOG}" 2>/dev/null | head -1 || true
  grep -o "Latent MoE projection \[[0-9, ]*\] sharded across [0-9]* cores per chip ([0-9]* columns each)" \
    "${SERVER_LOG}" 2>/dev/null | sort -u || true
  # A server with torch.compile disabled still serves correct tokens at ~1/18th
  # the decode speed, so nothing downstream notices -- the sweep just reports a
  # terrible number. Say so here, next to the endpoint, where it cannot be missed.
  if grep -qF "disabling vLLM's torch.compile pipeline" "${SERVER_LOG}" 2>/dev/null; then
    echo "WARNING: torch.compile is DISABLED (breakable cudagraph). Benchmarks from"
    echo "         this server are not comparable to compiled runs."
  else
    echo "torch.compile: enabled"
  fi
  echo "profiling: ${PROFILE_DIR:-off (set PROFILE_DIR to enable /start_profile)}"
  # Read it back from the server's own resolved config rather than echoing the
  # variable: the launcher, not this file, decides what reaches vLLM.
  echo "async scheduling: $(grep -m1 -oE "async_scheduling': *[A-Za-z]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  echo "max_num_seqs: $(grep -m1 -oE "max_num_seqs': *[0-9]+" \
    "${SERVER_LOG}" 2>/dev/null || echo 'unknown')"
  local latent_fallback
  latent_fallback=$(grep -c "leaving the latent projections replicated\|leaving it replicated" \
    "${SERVER_LOG}" 2>/dev/null || true)
  echo "Latent projection fallbacks: ${latent_fallback:-0}"
  # Read back what the dispatcher actually did, not what was requested: it
  # falls back to XLA silently whenever a shape is outside the measured
  # crossover or an op fails to build, and "the flag was set" is not evidence
  # that a single all-reduce changed.
  echo "Low-latency all-reduce: TPU_LOW_LATENCY_ALLREDUCE=${TPU_LOW_LATENCY_ALLREDUCE}"
  echo "  moe_concat=${TPU_LOW_LATENCY_ALLREDUCE_MOE_CONCAT}" \
       "moe_variadic=${TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC}" \
       "no_steady_barrier=${TPU_LOW_LATENCY_ALLREDUCE_NO_STEADY_BARRIER}" \
       "shape=${TPU_LOW_LATENCY_ALLREDUCE_SHAPE:-auto}" \
       "moe_renormalize_fold=${TPU_MOE_RENORMALIZE_FOLD}" \
       "compile_sizes=[${COMPILE_SIZES:-launcher default}]" \
       "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-launcher default}" \
       "onehot_moe_permute_threshold=${ONEHOT_MOE_PERMUTE_THRESHOLD}" \
       "xla_cache=${VLLM_XLA_CACHE_PATH}"
  local ll_one ll_two ll_xla
  ll_one=$(grep -c -- "-> one_shot" "${SERVER_LOG}" 2>/dev/null || true)
  ll_two=$(grep -c -- "-> two_shot" "${SERVER_LOG}" 2>/dev/null || true)
  ll_xla=$(grep -c "Low-latency all-reduce \[.*-> xla" "${SERVER_LOG}" 2>/dev/null || true)
  echo "  routed: one_shot=${ll_one:-0} two_shot=${ll_two:-0} xla=${ll_xla:-0} (per rank, per shape)"
  grep -h "Low-latency all-reduce \[" "${SERVER_LOG}" 2>/dev/null \
    | sed -E 's/.*(shape=[^ ]+ .*)/    \1/' | sort -u | head -8 || true
}
report_split || true

cat <<EOF

============================================================
 ENDPOINT   ${ENDPOINT}/v1
 model      ${MODEL}
 job        ${SLURM_JOB_ID} on ${HOSTS[*]}
 log        ${SERVER_LOG}

 Run evals against it, e.g.
   MODEL="${MODEL}" BASE_URL="${ENDPOINT}" \\
   OUTPUT_DIR="${LOG_DIR}/gsm8k" \\
   HF_HUB_CACHE=/mnt/lustre/hf-models/hub \\
   HF_DATASETS_CACHE="${REPO_DIR}/.runtime-cache/hf-datasets" \\
     ${REPO_DIR}/scripts/k3/eval_gsm8k.sh

 Holding the allocation until the time limit or scancel ${SLURM_JOB_ID}.
============================================================

EOF

# POST_READY_CMD runs once the endpoint is up, then the job exits and releases
# the nodes. Without it the job holds the allocation for its whole time limit,
# which is right for interactive use and wrong for an A/B: driving the client by
# hand from a login shell means the two arms can drift in what they actually ran,
# and arm_check.sh already makes that argument ("a hand-assembled sequence
# drifted between arms is not a comparison"). ENDPOINT and OUT are exported for
# it. A non-zero exit is reported but still releases the nodes.
# Refuse to measure a wiring we did not ask for. Both failure modes here have
# already cost a run: a bare width silently falls back to the smallest shape,
# which leaves K3's post-attention reduce on XLA at 17.1 us against two_shot's
# 12.5 -- ~433 us/step over 95 layers, enough to swamp anything under test.
if [[ -n "${TPU_LOW_LATENCY_ALLREDUCE_SHAPE:-}" ]]; then
  if grep -q "does not match any shape that takes a kernel" \
      "${LOG_DIR}/server.log" 2>/dev/null; then
    echo "FATAL: TPU_LOW_LATENCY_ALLREDUCE_SHAPE=${TPU_LOW_LATENCY_ALLREDUCE_SHAPE}" \
         "was rejected by the dispatcher (expects ROWSxCOLS, e.g. 1x7168)." >&2
    grep -oE "does not match any shape[^.]*" "${LOG_DIR}/server.log" \
      | head -1 >&2
    scancel "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 3
  fi
  echo "shape override accepted: $(grep -hoE 'wired to \(1, [0-9]+\) \([a-z_]+\) at site=[a-z]+' "${LOG_DIR}/server.log" | head -1)"
fi

if [[ -n "${POST_READY_CMD:-}" ]]; then
  export ENDPOINT="http://${HEAD_IP}:${PORT}"
  export OUT="${LOG_DIR}"
  echo "POST_READY_CMD: ${POST_READY_CMD}"
  echo "  ENDPOINT=${ENDPOINT}"
  set +e
  bash -c "${POST_READY_CMD}"
  _post_rc=$?
  set -e
  echo "POST_READY_CMD exited ${_post_rc}; releasing the allocation."
  scancel "${SLURM_JOB_ID}" 2>/dev/null || true
  exit "${_post_rc}"
fi

wait "${SERVER_STEP_PID}"
