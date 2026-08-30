#!/usr/bin/env bash
#SBATCH --job-name=k3-compile-flag-sweep
#SBATCH --partition=ici-2x2x4-c
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# XLA/libtpu compile-flag sweep for Kimi-K3 TP32 (task 10): ONE 4-node
# allocation, N sequential arms. Each arm brings the server up cold with a
# short COMPILE_SIZES list (decode buckets 1 and 4 plus the 8192 prefill
# bucket -- every decode batch pads UP to the next bucket, so c=4 perf with
# only 1,8192 would run its decode on the 8192 graph; job 154 measured
# 5.6 tok/s at c=4 for exactly that reason), records compile timings, runs a
# short c=1/c=4 perf check, tears the server (and Ray) down, and DELETES its
# compile-cache directory before the next arm starts.
#
# The FIRST arm is a control (no extra flags). It must reproduce job 148's
# per-graph anchors ((1,1)=107.56 s, (8192,8192)=316.09/374.35 s, rank TP24)
# within 10%; if it does not, the sweep stops -- something else changed and
# every later arm would be invalid.
#
# The runtime checkout is pinned to ws-torchtpu (the same commit job 148
# used); this script lives in the k3-dev checkout but must not serve from it.
#
# Arms: name|flags pairs in ARM_SPECS below (override with ARMS_SPEC, arms
# separated by ';;'). A flags field of '@combo_flags.txt' reads the flags from
# ${LOG_DIR}/combo_flags.txt AT ARM START, so the combination arm can be
# decided from the login node while the single-flag arms are still running;
# the arm is skipped if the file is absent or empty.
#
# One retry maximum on a failing arm is the brief's allowance; this script
# does NOT retry -- a failed arm is recorded and the sweep moves on.

set -uo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

# --- Runtime checkout: pinned, per the task brief. ---------------------------
REPO_DIR="${KIMI_K3_REPO_DIR:-/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu}"
RUN_USER="$(id -un)"

# --- Recipe pins: exactly tp32_8k1k_sweep_tpu_seqs32.yaml, except the -------
# --- experimental variables (COMPILE_SIZES and the arm's libtpu flags). -----
MODEL="${MODEL:-moonshotai/Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
export MODEL REVISION
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Buckets: 1 and 4 for the c=1/c=4 decode measurement, 8192 = the prefill
# bucket (must contain --max-num-batched-tokens). Same list for every arm so
# the comparison stays single-variable; costs ~1100 s of compile per arm.
export COMPILE_SIZES="${COMPILE_SIZES:-1,4,8192}"
export ASYNC_SCHEDULING=1
# vLLM auto-enables breakable cudagraph for K3 when the var is ABSENT, which
# disables torch.compile (344 ms TPOT instead of ~19). Pinned 0, always.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_TORCHTPU_KDA_FUSED=1
export TPU_MOE_HIERARCHICAL_EP=0
export TPU_LATENT_PROJ_INTRA_CHIP_TP=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0
export ENABLE_PREFIX_CACHING=0
export TPU_LOW_LATENCY_ALLREDUCE=0
export TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=0
export RAY_memory_usage_threshold=0.99
# Dummy weights + fp8 KV, stated explicitly (serving perf only, not
# correctness); passed to the launcher below.
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-56}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-300}"
RAY_RESOURCE_TIMEOUT="${RAY_RESOURCE_TIMEOUT:-600}"
# Per-arm health-OK budget. Job 148 took 2410 s cold INCLUDING Ray formation;
# an arm compiles 2 buckets instead of 7 (~800 s vs ~1460 s of graph compiles),
# so ~25-30 min is expected. 3600 s is generous margin, not a target.
ARM_READY_TIMEOUT="${ARM_READY_TIMEOUT:-3600}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-420}"
# Stop starting new arms once this much of the 6 h allocation is gone.
SWEEP_TIME_BUDGET_S="${SWEEP_TIME_BUDGET_S:-18600}"

LOG_DIR="${KIMI_K3_LOG_ROOT:-${SLURM_SUBMIT_DIR}/results/compile-flag-sweep-${SLURM_JOB_ID}}"
CSV="${LOG_DIR}/compile_flag_sweep.csv"
mkdir -p "${LOG_DIR}"

# Base libtpu args every arm carries (the same diagnostics-off trio the
# launcher/wrapper always set); the arm's flags are appended.
BASE_LIBTPU_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false"

# --- Arms --------------------------------------------------------------------
# One variable per arm; combinations only after singles. Order matters:
# control first (validity anchor), then singles, then the hlo_profile
# diagnostic, combo last so it can be injected from the singles' results via
# combo_flags.txt.
#
# Spec format: "name|libtpu_flags|xla_flags" (the third field is optional).
# libtpu_flags go into LIBTPU_INIT_ARGS; xla_flags go into XLA_FLAGS. Both
# must be supported: this libtpu build registers only libtpu's own flag set --
# generic XLA DebugOptions flags (xla_backend_optimization_level,
# xla_disable_hlo_passes, xla_hlo_profile) are REJECTED with "Unknown command
# line flag" in LIBTPU_INIT_ARGS (job 158) and only parse via XLA_FLAGS.
# '@file' in either field reads that file from LOG_DIR at arm start, so a
# combination can be decided from the login node mid-run; the arm is skipped
# if a referenced file is absent or empty.
ARM_SPECS_DEFAULT=(
  "control|"
  "optlevel1||--xla_backend_optimization_level=1"
  "nolhs|--xla_tpu_enable_latency_hiding_scheduler=false"
  "nopasses||--xla_disable_hlo_passes=constant_folding,multi_output_fusion"
  "noasyncfusion|--xla_tpu_enable_async_collective_fusion=false"
  "hloprof||--xla_hlo_profile=true"
  "combo|@combo_flags.txt|@combo_xla_flags.txt"
)
ARM_SPECS=()
# Arm list precedence: ARMS_FILE (one spec per line, '#' comments allowed;
# comma-safe, unlike sbatch --export=...,VAR=val which splits values at
# commas) > ARMS_SPEC (';;'-separated) > defaults above.
if [[ -n "${ARMS_FILE:-}" && -f "${ARMS_FILE}" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] && ARM_SPECS+=("${line}")
  done < "${ARMS_FILE}"
elif [[ -n "${ARMS_SPEC:-}" ]]; then
  IFS=';;' read -ra _parts <<< "${ARMS_SPEC}"
  for p in "${_parts[@]}"; do [[ -n "${p}" ]] && ARM_SPECS+=("${p}"); done
else
  ARM_SPECS=("${ARM_SPECS_DEFAULT[@]}")
fi

LAUNCHER="${REPO_DIR}/scripts/k3/launch_kimi_k3_tp32_ep.sh"
[[ -x "${LAUNCHER}" ]] || { echo "Launcher not found: ${LAUNCHER}" >&2; exit 2; }
[[ -x "${REPO_DIR}/.venv/bin/ray" ]] || { echo "Ray not found in ${REPO_DIR}/.venv" >&2; exit 2; }
[[ -x "${REPO_DIR}/vllm-bench" ]] || { echo "vllm-bench not found in ${REPO_DIR}" >&2; exit 2; }

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#HOSTS[@]} -ne 4 ]]; then
  echo "Expected four hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
HEAD_HOST="${HOSTS[0]}"
HEAD_IP="$(srun --overlap -N1 -n1 -w "${HEAD_HOST}" hostname -I | awk '{print $1}')"
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

# Checkpoint metadata (dummy weights still need config/tokenizer/remote code).
# HF_HUB_CACHE is PINNED, not defaulted: on this cluster an explicit
# `sbatch --export=ALL,...` injects HF_HUB_CACHE=/mnt/scratch/hf-hub into the
# batch env (job 174 envtest; plain sbatch does not, job 173). That path is
# node-local and lacks the snapshot metadata, which cost job 163 two arms to
# LocalEntryNotFoundError. Override via KIMI_K3_HF_HUB_CACHE if ever needed.
MODEL_DIR="models--${MODEL//\//--}"
export HF_HUB_CACHE="${KIMI_K3_HF_HUB_CACHE:-/mnt/models}"
export HF_HOME="${KIMI_K3_HF_HOME:-${REPO_DIR}/.runtime-cache/hf-home}"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HUB_OFFLINE=1
# The same injected-env hazard: per-arm vars must not leak in from the
# submitting context; the arm loop sets them itself.
unset XLA_FLAGS LIBTPU_INIT_ARGS VLLM_XLA_CACHE_PATH

# Forensics: job 163 lost two arms to a server-side LocalEntryNotFoundError
# (offline HF snapshot lookup) that neither reproduces nor matches the env the
# script sets. Record what we export and verify the snapshot is visible from
# EVERY node before any arm runs -- a stale /mnt/models mount on one node
# otherwise costs a ~35-minute arm to discover.
echo "env: HF_HUB_CACHE=${HF_HUB_CACHE} HF_HOME=${HF_HOME} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
SNAPSHOT_DIR="${HF_HUB_CACHE}/${MODEL_DIR}/snapshots/${REVISION}"
srun --overlap -N4 -n4 --ntasks-per-node=1 bash -c '
  snap="'"${SNAPSHOT_DIR}"'"
  for f in config.json tokenizer_config.json tiktoken.model model.safetensors.index.json; do
    [[ -e "${snap}/${f}" ]] || { echo "$(hostname -s): MISSING ${snap}/${f}" >&2; exit 1; }
  done
  compgen -G "${snap}/*.safetensors" >/dev/null || { echo "$(hostname -s): no safetensors in ${snap}" >&2; exit 1; }
  echo "$(hostname -s): snapshot OK"
' || { echo "model snapshot preflight failed; aborting" >&2; exit 2; }

# --- Guardrails ---------------------------------------------------------------
LUSTRE_FREE_START_GB="$(df -BG --output=avail /mnt/lustre | tail -1 | tr -dc '0-9')"
echo "Lustre free at start: ${LUSTRE_FREE_START_GB} GB"

check_disks() {
  # /mnt/scratch: watch by percentage (512 GB NVMe per node, node-local).
  local worst
  # Keep the newline: stripping it concatenates the four nodes' outputs into
  # one bogus number (job 153 read 23% as "23232323" and aborted).
  worst=$(srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c "df --output=pcent /mnt/scratch | tail -1 | tr -d ' %'" 2>/dev/null \
    | sort -rn | head -1)
  echo "disk: /mnt/scratch worst node ${worst}% used"
  if (( ${worst:-100} > 88 )); then
    echo "ABORT: /mnt/scratch above 88% used on a node" >&2
    return 1
  fi
  # /mnt/lustre: absolutes only. A "100% full" reading with hundreds of GB
  # free is normal for this mount; abort below 200 GB free or a >100 GB fall.
  local free_gb
  free_gb=$(df -BG --output=avail /mnt/lustre | tail -1 | tr -dc '0-9')
  echo "disk: /mnt/lustre ${free_gb} GB free (start ${LUSTRE_FREE_START_GB} GB)"
  if (( ${free_gb:-0} < 200 )) || (( ${LUSTRE_FREE_START_GB} - ${free_gb:-0} > 100 )); then
    echo "ABORT: /mnt/lustre free-space guardrail tripped" >&2
    return 1
  fi
  return 0
}

# --- Ray lifecycle (per arm: Ray workers inherit the raylet's env, so a -----
# --- per-arm LIBTPU_INIT_ARGS only reaches workers through a fresh Ray). ----
RAY_STEP_PID=
SERVER_STEP_PID=

start_ray() {
  local ray_log="$1"
  # Ray workers inherit the raylet's env, so pin the per-arm compile flags on
  # the ray start step explicitly rather than relying on --export=ALL alone.
  local ray_pin=(LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS}")
  [[ -n "${xflags:-}" ]] && ray_pin+=(XLA_FLAGS="${xflags}")
  srun --overlap -N4 -n4 --ntasks-per-node=1 --unbuffered --export=ALL \
    env "${ray_pin[@]}" \
    bash -c '
      node_ip=$(hostname -I | awk "{print \$1}")
      if [[ ${SLURM_PROCID} -eq 0 ]]; then
        exec ray start --block --head \
          --node-ip-address="${node_ip}" \
          --port="'"${RAY_PORT}"'" \
          --include-dashboard=false \
          --num-cpus="'"${RAY_NUM_CPUS}"'"
      fi
      until timeout 1 bash -c \
          "</dev/tcp/'"${HEAD_IP}"'/'"${RAY_PORT}"'" 2>/dev/null; do
        sleep 1
      done
      exec ray start --block \
        --address="'"${HEAD_IP}"':'"${RAY_PORT}"'" \
        --node-ip-address="${node_ip}" \
        --num-cpus="'"${RAY_NUM_CPUS}"'"
    ' >"${ray_log}" 2>&1 &
  RAY_STEP_PID=$!

  local deadline=$(( $(date +%s) + RAY_START_TIMEOUT ))
  until timeout 1 bash -c "</dev/tcp/${HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
    kill -0 "${RAY_STEP_PID}" 2>/dev/null || { echo "Ray exited during startup; see ${ray_log}" >&2; return 1; }
    (( $(date +%s) < deadline )) || { echo "Timed out waiting for Ray head; see ${ray_log}" >&2; return 1; }
    sleep 1
  done

  deadline=$(( $(date +%s) + RAY_RESOURCE_TIMEOUT ))
  local tpu_count=0
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
  [[ ${tpu_count:-0} == 32 ]] || { echo "Only ${tpu_count:-0}/32 TPU resources joined; see ${ray_log}" >&2; return 1; }
  echo "Ray ready on ${HOSTS[*]} (32 TPU resources)"
  return 0
}

stop_ray() {
  if [[ -n ${SERVER_STEP_PID} ]] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
  SERVER_STEP_PID=
  # Workers outlive the launcher often enough to strand the TPUs for the next
  # arm; at TP32 that is four hosts held hostage rather than one.
  timeout 60 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    pkill -9 -f 'VLLM::Worker' >/dev/null 2>&1 || true
  if [[ -n ${RAY_STEP_PID} ]] && kill -0 "${RAY_STEP_PID}" 2>/dev/null; then
    kill "${RAY_STEP_PID}" 2>/dev/null || true
    wait "${RAY_STEP_PID}" 2>/dev/null || true
  fi
  RAY_STEP_PID=
  timeout 30 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    ray stop --force >/dev/null 2>&1 || true
  # Drop this job's tier-2 namespaces: /dev/shm is tmpfs, so they count
  # against host RAM until removed.
  timeout 30 srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c "rm -rf /dev/shm/torch_tpu_cache/${RUN_USER}_kimi_k3_${SLURM_JOB_ID}_* " >/dev/null 2>&1 || true
}

cleanup() {
  status=$?
  trap - EXIT
  stop_ray
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Drop stale tier-2 namespaces from dead jobs of ours before anything starts.
timeout 30 srun --overlap -N4 -n4 --ntasks-per-node=1 \
  bash -c "rm -rf /dev/shm/torch_tpu_cache/${RUN_USER}_kimi_k3_*" >/dev/null 2>&1 || true

# --- Preflight: does the arm's flag set fatal at libtpu init? -----------------
# A trivial single-host jax compile parses LIBTPU_INIT_ARGS; an unknown flag
# dies here in ~90 s instead of ~15 min into a server bring-up. Also prints
# the assembled LIBTPU_INIT_ARGS exactly as env_override.py will hand it to
# the workers (env_override prepends its own flags; a user-supplied flag wins
# via its tokenized _user_has_flag check).
preflight_probe() {
  local probe_log="$1"
  timeout "${PROBE_TIMEOUT}" srun --overlap -N1 -n1 -w "${HEAD_HOST}" \
    --unbuffered --export=ALL \
    env TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
        TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT= \
        TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT= \
    "${REPO_DIR}/.venv/bin/python" -c '
import os
try:
    import vllm_torchtpu.env_override  # noqa: F401  (assembles LIBTPU_INIT_ARGS)
except Exception as exc:
    print("env_override import failed:", exc)
print("ASSEMBLED_LIBTPU_INIT_ARGS=" + os.environ.get("LIBTPU_INIT_ARGS", ""))
import jax
import jax.numpy as jnp
x = jnp.ones((64, 64))
print("PROBE_DEVICE=" + str(jax.devices()[0]))
print("PROBE_OK=" + str(float((x @ x).sum())))
' >"${probe_log}" 2>&1
}

# --- Per-arm metrics -> one CSV row -------------------------------------------
record_arm() {
  local arm="$1" flags="$2" assembled="$3" echo_count="$4" probe_ok="$5" \
        ready_wall="$6" server_log="$7" perf_dir="$8" cache_gb="$9" status="${10}"
  ARM="${arm}" FLAGS="${flags}" ASSEMBLED="${assembled}" ECHO_COUNT="${echo_count}" \
  PROBE_OK="${probe_ok}" READY_WALL="${ready_wall}" SERVER_LOG="${server_log}" \
  PERF_DIR="${perf_dir}" CACHE_GB="${cache_gb}" STATUS="${status}" \
  "${REPO_DIR}/.venv/bin/python" - <<'PYEOF' >> "${CSV}"
import os, re, csv, sys

def f(pat, text):
    return [float(m) for m in re.findall(pat, text)]

log = ""
try:
    with open(os.environ["SERVER_LOG"], errors="replace") as fh:
        log = fh.read()
except OSError:
    pass

# Per-graph compiles: "Compiling a graph for compile range (X, Y) takes N s".
# Ray de-duplicates lines across ranks, so keep every distinct nonzero value.
ranges = {}
for x, y, secs in re.findall(
        r"compile range \((\d+), (\d+)\) takes ([0-9.]+) s", log):
    v = float(secs)
    if v > 0:
        ranges.setdefault(f"({x},{y})", set()).add(v)
range_s = ";".join(f"{k}={'+'.join(f'{v:.2f}' for v in sorted(vs))}"
                   for k, vs in sorted(ranges.items()))
totals = sorted(set(f(r"torch\.compile took ([0-9.]+) s in total", log)))
m = re.search(r"init engine \(profile, create kv cache, warmup model\) took "
              r"([0-9.]+) s \(compilation: ([0-9.]+) s\)", log)
init_s, init_comp_s = (m.group(1), m.group(2)) if m else ("", "")

def perf(c):
    import json
    try:
        d = json.load(open(os.path.join(os.environ["PERF_DIR"], f"c{c}.json")))
        return (f"{d['median_tpot_ms']:.2f}", f"{d['median_ttft_ms']:.1f}",
                f"{d['output_throughput']:.1f}")
    except Exception:
        return ("", "", "")

c1, c4 = perf(1), perf(4)
row = [os.environ["ARM"], os.environ["FLAGS"], os.environ["ASSEMBLED"],
       os.environ["ECHO_COUNT"], os.environ["PROBE_OK"],
       os.environ["READY_WALL"], range_s,
       ";".join(f"{v:.2f}" for v in totals), init_s, init_comp_s,
       os.environ["CACHE_GB"], *c1, *c4, os.environ["STATUS"]]
csv.writer(sys.stdout).writerow(row)
PYEOF
}

write_csv_header() {
  [[ -f "${CSV}" ]] || cat > "${CSV}" <<'EOF'
arm,flags,assembled_libtpu_init_args,flag_echo_count,probe_ok,ready_wall_s,graph_compile_s,torch_compile_totals_s,init_engine_s,init_engine_compilation_s,cache_gb,c1_tpot_ms,c1_ttft_ms,c1_tok_s,c4_tpot_ms,c4_ttft_ms,c4_tok_s,status
EOF
}
write_csv_header

echo "=============================================================="
echo "compile-flag sweep: job ${SLURM_JOB_ID} on ${HOSTS[*]}"
echo "repo: ${REPO_DIR}"
echo "log dir: ${LOG_DIR}"
echo "arms: ${ARM_SPECS[*]}"
echo "=============================================================="

# --- Arm loop ------------------------------------------------------------------
for spec in "${ARM_SPECS[@]}"; do
  arm="${spec%%|*}"
  rest="${spec#*|}"
  flags="${rest%%|*}"
  [[ "${rest}" == *"|"* ]] && xflags="${rest#*|}" || xflags=""
  # '@file' fields are read from LOG_DIR at arm start (combo injection).
  for which in flags xflags; do
    val="${!which}"
    if [[ "${val}" == @* ]]; then
      f="${LOG_DIR}/${val#@}"
      if [[ -s "${f}" ]]; then
        printf -v "${which}" '%s' "$(tr '\n' ' ' < "${f}")"
        echo "${arm} ${which} from ${f}: ${!which}"
      else
        echo "SKIPPING ${arm}: ${f} absent or empty"
        continue 2
      fi
    fi
  done

  remaining=$(( SWEEP_TIME_BUDGET_S - SECONDS ))
  if (( remaining < 3300 )); then
    echo "TIME BUDGET: ${remaining}s left of sweep budget; stopping before arm ${arm}"
    break
  fi

  echo ""
  echo "==================== ARM ${arm}: LIBTPU+='${flags}' XLA_FLAGS='${xflags}' ===================="
  check_disks || break

  # Per-arm compile caches on NVMe; DELETED at the end of the arm.
  export VLLM_XLA_CACHE_PATH="/mnt/scratch/users/${RUN_USER}/vllm-xla-cache/k3-cfs-${SLURM_JOB_ID}-${arm}"
  export VLLM_CACHE_ROOT="${VLLM_XLA_CACHE_PATH}/vllm-runtime"
  export VLLM_CONFIG_ROOT="${VLLM_XLA_CACHE_PATH}/vllm-config"
  export TORCH_TPU_TIER2_COMPILATION_CACHE="${RUN_USER}_kimi_k3_${SLURM_JOB_ID}_${arm}"
  export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${TORCH_TPU_TIER2_COMPILATION_CACHE}"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3"
  export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"
  export LIBTPU_INIT_ARGS="${BASE_LIBTPU_ARGS} ${flags}"
  # XLA_FLAGS only when the arm asks for it: absent vs empty changes the
  # compile-cache key, and arms without XLA flags must stay identical to the
  # job-148/158 control environment.
  if [[ -n "${xflags}" ]]; then export XLA_FLAGS="${xflags}"; else unset XLA_FLAGS; fi

  srun --overlap -N4 -n4 --ntasks-per-node=1 mkdir -p \
    "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" \
    "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"

  server_log="${LOG_DIR}/server-${arm}.log"
  probe_log="${LOG_DIR}/probe-${arm}.log"
  perf_dir="${LOG_DIR}/perf_${arm}"
  probe_ok=0 ready_wall="" assembled="" echo_count=0 status=""

  # Preflight: trivial compile with the arm's flags.
  if preflight_probe "${probe_log}"; then
    probe_ok=1
    assembled=$(grep -m1 -oE "ASSEMBLED_LIBTPU_INIT_ARGS=.*" "${probe_log}" \
      | sed 's/^ASSEMBLED_LIBTPU_INIT_ARGS=//' || true)
    echo "preflight OK; assembled LIBTPU_INIT_ARGS: ${assembled:-<unrecorded>}"
    # For XLA_FLAGS arms the argv-echo check below cannot see the flags (that
    # echo covers LIBTPU_INIT_ARGS only). Confirm effect directly where the
    # flag has an observable probe signature.
    if [[ "${xflags}" == *xla_hlo_profile* ]]; then
      if grep -q "Total runtime (ms) of HLO passes" "${probe_log}"; then
        echo "preflight effect check: xla_hlo_profile per-pass dump PRESENT in probe"
      else
        echo "preflight effect check: xla_hlo_profile dump NOT seen in probe (parsed, effect unconfirmed)"
      fi
    fi
  else
    echo "preflight FAILED for arm ${arm}; see ${probe_log}" >&2
    tail -20 "${probe_log}" >&2 || true
    status="FAIL_PREFLIGHT"
  fi

  if [[ ${probe_ok} == 1 ]]; then
    if start_ray "${LOG_DIR}/ray-${arm}.log"; then
      # env-pin the variables this arm depends on rather than trusting
      # --export=ALL alone (job 163 lost two arms to a server whose HF cache
      # env never arrived or whose /mnt/models view was stale).
      env_pin=(HF_HUB_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
               HF_HOME="${HF_HOME}" HF_MODULES_CACHE="${HF_MODULES_CACHE}"
               LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS}")
      [[ -n "${xflags}" ]] && env_pin+=(XLA_FLAGS="${xflags}")
      srun --overlap -N1 -n1 -w "${HEAD_HOST}" --unbuffered --export=ALL \
        --chdir="${REPO_DIR}" \
        env "${env_pin[@]}" "${LAUNCHER}" \
        --load-format "${LOAD_FORMAT}" --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        >"${server_log}" 2>&1 &
      SERVER_STEP_PID=$!

      start=$(date +%s)
      ready=0
      while (( $(date +%s) - start < ARM_READY_TIMEOUT )); do
        if curl -sf "http://${HEAD_IP}:${PORT}/health" >/dev/null 2>&1; then
          ready=1
          break
        fi
        if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
          echo "Server exited during startup; tail:" >&2
          tail -40 "${server_log}" >&2 || true
          break
        fi
        sleep 10
      done

      if (( ready == 1 )); then
        ready_wall=$(( $(date +%s) - start ))
        echo "ARM ${arm}: SERVER_READY after ${ready_wall}s"
        # libtpu echoes each parsed flag as "argv[N]: '--flag=value'"; confirm
        # the arm's flags actually reached the workers.
        for tok in ${flags}; do
          n=$(grep -F "argv[" "${server_log}" 2>/dev/null | grep -cF "'${tok}'" || true)
          echo_count=$(( echo_count + ${n:-0} ))
          echo "flag echo: '${tok}' seen ${n:-0}x in worker log"
        done

        # Short perf check: c=1 and c=4 at 8192/1024. This is what turns a
        # compile-time saving into a tradeoff decision; do not cut it.
        ARM="${arm}" ENDPOINT="http://${HEAD_IP}:${PORT}" \
          OUTPUT_DIR="${perf_dir}" \
          CONCURRENCIES="1 4" INPUT_LEN=8192 OUTPUT_LEN=1024 \
          PROMPTS_PER_SLOT=6 PROMPT_TOKEN_IDS=1 \
          RANDOM_PREFIX_LEN=0 RANDOM_RANGE_RATIO=1 RESET_PREFIX_CACHE=0 \
          MODEL="${MODEL}" \
          "${REPO_DIR}/scripts/k3/concurrency_sweep.sh" \
          > "${LOG_DIR}/perf_${arm}.log" 2>&1 \
          && status="OK" || status="FAIL_PERF"
        tail -5 "${LOG_DIR}/perf_${arm}.log" || true
      else
        status="FAIL_BRINGUP"
        echo "ARM ${arm}: endpoint never came up; tail:" >&2
        tail -40 "${server_log}" >&2 || true
      fi
    else
      status="FAIL_RAY"
    fi
  fi

  # Cache size BEFORE teardown, then delete the cache tree on all nodes.
  cache_gb=$(srun --overlap -N4 -n4 --ntasks-per-node=1 \
    bash -c "du -s --block-size=1M '${VLLM_XLA_CACHE_PATH}' 2>/dev/null | cut -f1" \
    2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1024}')
  echo "ARM ${arm}: cache ${cache_gb} GB (4 nodes total)"

  stop_ray
  srun --overlap -N4 -n4 --ntasks-per-node=1 \
    rm -rf "${VLLM_XLA_CACHE_PATH}" || true

  record_arm "${arm}" "${flags}${xflags:+ XLA_FLAGS=${xflags}}" "${assembled}" "${echo_count}" "${probe_ok}" \
    "${ready_wall}" "${server_log}" "${perf_dir}" "${cache_gb:-0}" "${status}"
  echo "ARM ${arm}: status=${status:-FAIL_UNKNOWN} -> ${CSV}"

  # Control-arm validity check (per the brief: if A0 does not reproduce the
  # job-148 per-graph numbers within tolerance, the whole sweep is invalid).
  if [[ "${arm}" == "control" && "${status}" == "OK" ]]; then
    if ! SERVER_LOG="${server_log}" "${REPO_DIR}/.venv/bin/python" - <<'PYEOF'
import os, re, sys
log = open(os.environ["SERVER_LOG"], errors="replace").read()
g11 = [float(x) for x in re.findall(r"compile range \(1, 1\) takes ([0-9.]+) s", log)]
g8192 = [float(x) for x in re.findall(r"compile range \(8192, 8192\) takes ([0-9.]+) s", log)]
TOL = 0.10
A11, APF = 107.56, 374.35  # job-148 rank-TP24 anchors (prefill phase 2 is the max)
m11, mpf = (max(g11) if g11 else 0.0), (max(g8192) if g8192 else 0.0)
ok = g11 and g8192 and abs(m11 - A11) <= TOL * A11 and abs(mpf - APF) <= TOL * APF
print(f"control anchor check: max (1,1)={m11:.2f}s vs {A11}, max (8192,8192)={mpf:.2f}s vs {APF}, tol={TOL} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PYEOF
    then
      echo "CONTROL MISMATCH -- stopping sweep; see ${server_log}" >&2
      touch "${LOG_DIR}/CONTROL_MISMATCH"
      exit 3
    fi
  fi
done

echo ""
echo "sweep complete -> ${CSV}"
cat "${CSV}"
