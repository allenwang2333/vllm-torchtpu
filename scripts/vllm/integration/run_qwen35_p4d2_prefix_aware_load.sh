#!/usr/bin/env bash

# Qwen3.5 P/D prefix-aware Stage-3 load test (#487). Same pair and settings as
# run_qwen35_p4d2_disagg_correctness.sh (prefill PCP4, decode DP4 on one 2x2x1
# host) with TPU_RAIDEN_PREFIX_AWARE_LOAD=1, and decode on the Raiden transfer
# connector alone (DECODE_KV_CONNECTOR=raiden): with the default
# transfer+offload MultiConnector, vLLM never lets the hybrid model reuse
# decode-local pages, so the feature cannot take effect. Every request goes
# through the proxy. When decode already has the start of a prompt cached, it
# must skip those pages and pull only the rest from prefill.
# Stage 1  warmup             smoke_prefix_cache_correctness.py
#   One short chat request. Pass: the answer is right.
# Stage 2  same_prompt_twice  smoke_prefix_cache_e2e_divergence.py
#   Each prompt sent twice to decode replica PA_DP_RANK. The second send reuses
#   the first send's cached pages and pulls only the rest. Pass: both sends
#   return the same text and token ids.
# Stage 3  shared_start       smoke_prefix_cache_correctness.py
#   Prompts sharing a long start with different questions, on the same replica.
#   Each reuses the shared pages and pulls only its own question. Pass: every
#   answer is exactly its code.
# Stage 4  counters           /metrics of both engines
#   Stages 2-3 also pass if decode never reused a page. Pass: decode reused
#   prompt tokens from its cache, pulled the rest, computed at most
#   MAX_DECODE_COMPUTE_PCT percent itself, and neither engine preempted.
# Stage 5  decode_log         check_prefix_aware_load_log.py
#   Stages 1-4 read the same if decode pulled everything. Pass: decode's log
#   shows at least PA_MIN_SUFFIX_LOADS loads with skip_tokens > 0, each skipping
#   whole pages, with one bytes-per-token clip throughout.
# All stages run even if one fails. Exit status: 0 every stage passed, 1 a stage
# failed, 2 no stage was run. Run with TPU_RAIDEN_PREFIX_AWARE_LOAD=0 as a
# negative control: stage 5 must then fail.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${QWEN35_A3B_FP8_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_p4d2_prefix_aware_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
P4D2_BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
VLLM_SRC="${VLLM_SRC:-${repo_root}}"
TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
PREFILL_TP="${PREFILL_TP:-1}"
PREFILL_PCP="${PREFILL_PCP:-4}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
DECODE_TP="${DECODE_TP:-1}"
DECODE_DP="${DECODE_DP:-4}"
PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-768}"
DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-2304}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
DECODE_KV_CONNECTOR="${DECODE_KV_CONNECTOR:-raiden}"
# P4D4 uses a smaller per-worker premapped pool than the plugin-wide Raiden
# default to reduce libtpu startup reservation time for this bounded case.
TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-8589934592}"
TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
# Match the large-cache PCP prefill compilation geometry. An empty override
# makes vLLM derive the block count from available HBM.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256}"
default_prefill_speculative_config='{"method":"mtp","num_speculative_tokens":1}'
default_decode_speculative_config='{"method":"mtp","num_speculative_tokens":3}'
PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-${default_prefill_speculative_config}}"
DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-${default_decode_speculative_config}}"

cleanup() {
  local pid_file pid
  if [[ -d "${RUN_DIR}" ]]; then
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          kill -TERM -- "-${pid}" 2>/dev/null \
            || kill -TERM "${pid}" 2>/dev/null \
            || true
        fi
      fi
    done
    sleep 2
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          if kill -0 -- "-${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
          elif kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
          fi
        fi
      fi
    done
  fi
}
trap cleanup EXIT

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}" "${RUN_DIR}/logs"

# Dynamically extract the exact vLLM version installed in the container
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')}"
export EXPECTED_VLLM_VERSION

export MODEL_PATH
export SERVED_MODEL_NAME
export RUN_ROOT
export RUN_DIR
export P4D2_BIND_HOST
export PREFILL_PORT
export DECODE_PORT
export PROXY_PORT
export VLLM_SRC
export TORCHTPU_VLLM_SRC
export USE_CURRENT_PY_ENV
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
export ASYNC_SCHEDULING
export PREFILL_TP
export PREFILL_PCP
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE
export DECODE_TP
export DECODE_DP
export PREFILL_BLOCK_SIZE
export DECODE_BLOCK_SIZE
export ENABLE_PREFIX_CACHING
export MAMBA_CACHE_MODE
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL
export TPU_RAIDEN_PREFIX_AWARE_LOAD
export DECODE_KV_CONNECTOR
export TPU_PREMAPPED_BUFFER_SIZE
export TPU_PARALLEL_PRECOMPILE
export NUM_GPU_BLOCKS_OVERRIDE
export GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN
export MAX_NUM_SEQS
export PREFILL_COMPILE_SIZES
export DECODE_COMPILE_SIZES
export PREFILL_SPECULATIVE_CONFIG
export DECODE_SPECULATIVE_CONFIG

cd "${repo_root}"

bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh \
  2>&1 | tee "${RUN_DIR}/logs/launch.log" || exit 2

if ! grep -Fqx \
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}" \
    "${RUN_DIR}/launch_params.txt"; then
  echo "Expected VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL} in launch_params.txt" >&2
  exit 2
fi
echo "PREFIX_CACHE_RETENTION_INTERVAL_OK ${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}"

if ! grep -Fqx \
    "TPU_RAIDEN_PREFIX_AWARE_LOAD=${TPU_RAIDEN_PREFIX_AWARE_LOAD}" \
    "${RUN_DIR}/launch_params.txt"; then
  echo "Expected TPU_RAIDEN_PREFIX_AWARE_LOAD=${TPU_RAIDEN_PREFIX_AWARE_LOAD} in launch_params.txt" >&2
  exit 2
fi
echo "RAIDEN_PREFIX_AWARE_LOAD_OK ${TPU_RAIDEN_PREFIX_AWARE_LOAD}"

if ! grep -Fqx "DECODE_KV_CONNECTOR=${DECODE_KV_CONNECTOR}" \
    "${RUN_DIR}/launch_params.txt"; then
  echo "Expected DECODE_KV_CONNECTOR=${DECODE_KV_CONNECTOR} in launch_params.txt" >&2
  exit 2
fi
echo "DECODE_KV_CONNECTOR_OK ${DECODE_KV_CONNECTOR}"

expected_prefill_scheduler_block_size=$((PREFILL_BLOCK_SIZE * PREFILL_PCP))
expected_decode_scheduler_block_size="${DECODE_BLOCK_SIZE}"
prefill_resolution="scheduler_block_size=${expected_prefill_scheduler_block_size} hash_block_size=${expected_prefill_scheduler_block_size}"

if ! grep -Fq "${prefill_resolution}" "${RUN_DIR}/logs/prefill.log"; then
  echo "Expected Prefill ${prefill_resolution}, but it was not found in the server log" >&2
  exit 2
fi
if ! grep -Fq "Using KV cache block size: ${DECODE_BLOCK_SIZE}" \
    "${RUN_DIR}/logs/decode.log"; then
  echo "Expected Decode physical/scheduler block size ${DECODE_BLOCK_SIZE}, but it was not found in the server log" >&2
  exit 2
fi
echo "SCHEDULER_BLOCK_SIZE_OK prefill=${expected_prefill_scheduler_block_size} decode=${expected_decode_scheduler_block_size}"

PREFILL_URL="http://${P4D2_BIND_HOST}:${PREFILL_PORT}"
DECODE_URL="http://${P4D2_BIND_HOST}:${DECODE_PORT}"
# Each decode DP replica has its own prefix cache, so a repeat only reuses
# cached pages when it reaches the replica that cached them. Rank 0 is the only
# rank valid on both engines: prefill runs one replica (PCP4), decode four.
PA_DP_RANK="${PA_DP_RANK:-0}"
# About 32 tokens a line with the Qwen3.5 chat template: 160 lines is ~5k
# tokens, so every repeat covers at least one full 2304-token decode page.
PA_PROMPT_LINES="${PA_PROMPT_LINES:-160}"
PA_DIVERGENCE_ATTEMPTS="${PA_DIVERGENCE_ATTEMPTS:-3}"
PA_MIN_SUFFIX_LOADS="${PA_MIN_SUFFIX_LOADS:-3}"
MAX_DECODE_COMPUTE_PCT="${MAX_DECODE_COMPUTE_PCT:-5}"

# metric <base url> <name> [label]: a counter summed over its series. -1 when
# the engine does not answer or has no such series, so that a counter nobody
# could read is never taken for one that reads 0.
metric() {
  local body
  if ! body="$(curl -sf --max-time 30 "$1/metrics")"; then
    echo -1
    return
  fi
  awk -v name="$2" -v label="${3:-}" '
    index($0, name) == 1 && (label == "" || index($0, label)) {s += $NF; n++}
    END {if (n) printf "%d", s; else printf "-1"}' <<<"${body}"
}

# Stages 2 and 3 also pass if decode never reused a page, so require that it
# did, that it pulled the rest, and that it computed almost nothing itself.
check_counters() {
  local rc=0 name value
  local by_source=vllm:prompt_tokens_by_source_total
  local -A must_be_positive=(
    [decode_local_cache_hit_tokens]="$(metric "${DECODE_URL}" \
      "${by_source}" 'source="local_cache_hit"')"
    [decode_external_kv_transfer_tokens]="$(metric "${DECODE_URL}" \
      "${by_source}" 'source="external_kv_transfer"')"
  )
  local -A must_be_zero=(
    [prefill_preemptions]="$(metric "${PREFILL_URL}" vllm:num_preemptions_total)"
    [decode_preemptions]="$(metric "${DECODE_URL}" vllm:num_preemptions_total)"
  )
  local computed
  computed="$(metric "${DECODE_URL}" "${by_source}" 'source="local_compute"')"

  for name in "${!must_be_positive[@]}" "${!must_be_zero[@]}"; do
    value="${must_be_positive[${name}]:-${must_be_zero[${name}]:-}}"
    echo "COUNTER ${name} ${value}"
    if [[ "${value}" -lt 0 ]]; then
      echo "COUNTER_FAIL ${name} could not be read"
      rc=1
    fi
  done
  echo "COUNTER decode_local_compute_tokens ${computed}"
  if [[ "${computed}" -lt 0 ]]; then
    echo "COUNTER_FAIL decode_local_compute_tokens could not be read"
    rc=1
  fi
  [[ "${rc}" -eq 0 ]] || return 1

  for name in "${!must_be_positive[@]}"; do
    if [[ "${must_be_positive[${name}]}" -eq 0 ]]; then
      echo "COUNTER_FAIL ${name} is 0: no request took that path"
      rc=1
    fi
  done
  for name in "${!must_be_zero[@]}"; do
    if [[ "${must_be_zero[${name}]}" -ne 0 ]]; then
      echo "COUNTER_FAIL ${name} is ${must_be_zero[${name}]}, expected 0"
      rc=1
    fi
  done
  local reused transferred share
  reused="${must_be_positive[decode_local_cache_hit_tokens]}"
  transferred="${must_be_positive[decode_external_kv_transfer_tokens]}"
  share="$(awk -v c="${computed}" -v t="$((computed + transferred + reused))" \
    'BEGIN {printf "%d", (t > 0) ? 100 * c / t : 100}')"
  echo "decode computed ${share}% of its prompt tokens itself"
  if [[ "${share}" -gt "${MAX_DECODE_COMPUTE_PCT}" ]]; then
    echo "COUNTER_FAIL that is over ${MAX_DECODE_COMPUTE_PCT}%"
    rc=1
  fi
  return "${rc}"
}

declare -A stage_rc=()
stage() {
  local name="$1" rc=0 start="${SECONDS}"
  shift
  echo "--- stage ${name}"
  "$@" 2>&1 | tee "${RUN_DIR}/logs/${name}.log" || rc=$?
  stage_rc["${name}"]="${rc}"
  echo "STAGE ${name} rc=${rc} in $((SECONDS - start))s"
}

target=(--host "${P4D2_BIND_HOST}" --port "${PROXY_PORT}" --model "${SERVED_MODEL_NAME}")
pinned=(--data-parallel-rank "${PA_DP_RANK}")
correctness="${script_dir}/smoke_prefix_cache_correctness.py"

stage warmup python3 "${correctness}" "${target[@]}" --quick-probe-only
# --no-logprobs: TPU speculative decoding (MTP on decode) returns none.
stage same_prompt_twice python3 \
  "${script_dir}/smoke_prefix_cache_e2e_divergence.py" "${target[@]}" \
  "${pinned[@]}" --attempts "${PA_DIVERGENCE_ATTEMPTS}" \
  --line-count "${PA_PROMPT_LINES}" --max-tokens 16 --no-logprobs
stage shared_start python3 "${correctness}" "${target[@]}" "${pinned[@]}" \
  --namespace "prefix-aware-$(date +%s)" --skip-short-qa \
  --long-repeat-lines "${PA_PROMPT_LINES}" --long-repeat-count 3 \
  --long-shared-lines "${PA_PROMPT_LINES}" --long-shared-rounds 2 \
  --mixed-lines "${PA_PROMPT_LINES}" --mixed-rounds 2
stage counters check_counters
stage decode_log python3 "${script_dir}/check_prefix_aware_load_log.py" \
  "${RUN_DIR}/logs/decode.log" --page-tokens "${expected_decode_scheduler_block_size}" \
  --min-suffix-loads "${PA_MIN_SUFFIX_LOADS}"

echo "=== QWEN35 PREFIX-AWARE LOAD SUMMARY ==="
failed=0
for name in warmup same_prompt_twice shared_start counters decode_log; do
  echo "STAGE_RESULT ${name} ${stage_rc[${name}]}"
  [[ "${stage_rc[${name}]}" -eq 0 ]] || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "QWEN35_PREFIX_AWARE_LOAD_FAIL (logs in ${RUN_DIR}/logs)"
  exit 1
fi
echo "QWEN35_PREFIX_AWARE_LOAD_OK"
