#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${QWEN35_A3B_FP8_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B-FP8}"

RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_pcp4_dp4_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_KV_PORT_BASE="${PREFILL_KV_PORT_BASE:-9100}"
DECODE_KV_PORT_BASE="${DECODE_KV_PORT_BASE:-9200}"
SIDE_CHANNEL_PORT_BASE="${SIDE_CHANNEL_PORT_BASE:-9600}"
STRIDED_PORT_OFFSET=10000

# This CI topology consumes all eight chips of an 8-chip TPU host.
PREFILL_TP=1
PREFILL_PCP=4
DECODE_TP=1
DECODE_DP=4
PREFILL_CHIPS=$((PREFILL_TP * PREFILL_PCP))
DECODE_CHIPS=$((DECODE_TP * DECODE_DP))
REQUIRED_CHIPS=$((PREFILL_CHIPS + DECODE_CHIPS))
PREFILL_TPU_LOCAL_RANK_OFFSET=0
DECODE_TPU_LOCAL_RANK_OFFSET="${PREFILL_CHIPS}"

BLOCK_SIZE_PREFILL="${BLOCK_SIZE_PREFILL:-4096}"
BLOCK_SIZE_DECODE="${BLOCK_SIZE_DECODE:-4096}"
NUM_GPU_BLOCKS_OVERRIDE_PREFILL="${NUM_GPU_BLOCKS_OVERRIDE_PREFILL:-16}"
NUM_GPU_BLOCKS_OVERRIDE_DECODE="${NUM_GPU_BLOCKS_OVERRIDE_DECODE:-72}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
CORRECTNESS_TIMEOUT_S="${CORRECTNESS_TIMEOUT_S:-300}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

process_is_alive() {
  local pid_file="$1"
  local pid
  [[ -s "${pid_file}" ]] || return 1
  pid="$(<"${pid_file}")"
  kill -0 "${pid}" 2>/dev/null
}

stop_process_group() {
  local pid_file="$1"
  local pid
  [[ -s "${pid_file}" ]] || return 0
  pid="$(<"${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || true
}

kill_process_group() {
  local pid_file="$1"
  local pid
  [[ -s "${pid_file}" ]] || return 0
  pid="$(<"${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  stop_process_group "${RUN_DIR}/proxy.pid"
  stop_process_group "${RUN_DIR}/prefill.pid"
  stop_process_group "${RUN_DIR}/decode.pid"
  sleep 2
  kill_process_group "${RUN_DIR}/proxy.pid"
  kill_process_group "${RUN_DIR}/prefill.pid"
  kill_process_group "${RUN_DIR}/decode.pid"

  if ((exit_code != 0)) && [[ -d "${RUN_DIR}/logs" ]]; then
    echo "Test failed; showing the last 100 lines from each service log:" >&2
    local log
    for log in proxy prefill decode; do
      if [[ -f "${RUN_DIR}/logs/${log}.log" ]]; then
        echo "===== ${log}.log =====" >&2
        tail -n 100 "${RUN_DIR}/logs/${log}.log" >&2 || true
      fi
    done
  fi

  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_http() {
  local name="$1"
  local url="$2"
  local pid_file="$3"
  local log_file="$4"
  local start_time="${SECONDS}"
  local status

  echo "Waiting up to ${VLLM_ENGINE_READY_TIMEOUT_S}s for ${name}: ${url}"
  while ((SECONDS - start_time < VLLM_ENGINE_READY_TIMEOUT_S)); do
    if ! process_is_alive "${pid_file}"; then
      echo "${name} exited before becoming ready" >&2
      tail -n 100 "${log_file}" >&2 || true
      return 1
    fi
    status="$(curl -sS -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
    if [[ "${status}" == "200" ]]; then
      echo "${name} is ready"
      return 0
    fi
    sleep 10
  done

  echo "Timed out waiting for ${name}" >&2
  tail -n 100 "${log_file}" >&2 || true
  return 1
}

wait_for_prefill_and_decode() {
  local start_time="${SECONDS}"
  local prefill_status
  local decode_status

  echo "Waiting up to ${VLLM_ENGINE_READY_TIMEOUT_S}s for prefill and decode"
  while ((SECONDS - start_time < VLLM_ENGINE_READY_TIMEOUT_S)); do
    if ! process_is_alive "${RUN_DIR}/prefill.pid"; then
      echo "Prefill server exited before becoming ready" >&2
      tail -n 100 "${RUN_DIR}/logs/prefill.log" >&2 || true
      return 1
    fi
    if ! process_is_alive "${RUN_DIR}/decode.pid"; then
      echo "Decode server exited before becoming ready" >&2
      tail -n 100 "${RUN_DIR}/logs/decode.log" >&2 || true
      return 1
    fi

    prefill_status="$(
      curl -sS -o /dev/null -w '%{http_code}' \
        "http://${SERVE_HOST}:${PREFILL_PORT}/health" 2>/dev/null || true
    )"
    decode_status="$(
      curl -sS -o /dev/null -w '%{http_code}' \
        "http://${SERVE_HOST}:${DECODE_PORT}/health" 2>/dev/null || true
    )"
    if [[ "${prefill_status}" == "200" && "${decode_status}" == "200" ]]; then
      echo "Prefill and decode servers are ready"
      return 0
    fi
    sleep 10
  done

  echo "Timed out waiting for prefill and decode servers" >&2
  return 1
}

require_free_ports() {
  local ports=("$@")
  if ! python - "${SERVE_HOST}" "${ports[@]}" <<'PY'
import socket
import sys

host = sys.argv[1]
busy = []
for raw_port in sys.argv[2:]:
    port = int(raw_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as error:
            busy.append((port, str(error)))

if busy:
    for port, error in busy:
        print(f"Port {port} is not available: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    die "One or more required ports are already in use; stop the existing service or select different ports"
  fi
}

for command in python curl tpu-info setsid timeout; do
  command -v "${command}" >/dev/null 2>&1 ||
    die "Required command is not available on PATH: ${command}"
done

mapfile -t TPU_DEVICE_PATHS < <(
  tpu-info 2>/dev/null |
    grep -oE '/dev/vfio/[0-9]+' |
    sort -Vu
)
AVAILABLE_CHIPS="${#TPU_DEVICE_PATHS[@]}"
if ((AVAILABLE_CHIPS < REQUIRED_CHIPS)); then
  die "This test requires ${REQUIRED_CHIPS} TPU chips, but only ${AVAILABLE_CHIPS} were found"
fi

if [[ "${MODEL_PATH}" == /* ||
      "${MODEL_PATH}" == ./* ||
      "${MODEL_PATH}" == ../* ]]; then
  [[ -d "${MODEL_PATH}" ]] || die "MODEL_PATH does not exist: ${MODEL_PATH}"
fi

managed_ports=(
  "${PREFILL_PORT}"
  "${DECODE_PORT}"
  "${PROXY_PORT}"
)
for ((rank = 0; rank < PREFILL_CHIPS; rank++)); do
  managed_ports+=("$((PREFILL_KV_PORT_BASE + rank))")
  managed_ports+=(
    "$((PREFILL_KV_PORT_BASE + STRIDED_PORT_OFFSET + rank))"
  )
done
for ((rank = 0; rank < DECODE_CHIPS; rank++)); do
  managed_ports+=("$((DECODE_KV_PORT_BASE + rank))")
  managed_ports+=(
    "$((DECODE_KV_PORT_BASE + STRIDED_PORT_OFFSET + rank))"
  )
  managed_ports+=("$((SIDE_CHANNEL_PORT_BASE + rank))")
done
require_free_ports "${managed_ports[@]}"

mkdir -p "${RUN_DIR}/logs"
cd "${repo_root}"

echo "Model: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "KV transport: TPUConnectorV2/ZMQ strided transfer"
echo "Prefill topology: TP=${PREFILL_TP} PCP=${PREFILL_PCP}"
echo "Decode topology: TP=${DECODE_TP} DP=${DECODE_DP}"
echo "Required TPU chips: ${REQUIRED_CHIPS}"
echo "Available TPU chips: ${AVAILABLE_CHIPS}"
echo "Run directory: ${RUN_DIR}"

prefill_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
decode_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[256],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
prefill_kv_config='{"kv_connector":"TPUConnectorV2","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector","kv_role":"kv_producer"}'
decode_kv_config='{"kv_connector":"TPUConnectorV2","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector","kv_role":"kv_consumer"}'

common_args=(
  --host "${SERVE_HOST}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --seed 42
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype fp8
  --language-model-only
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --no-disable-hybrid-kv-cache-manager
  --attention-backend "${ATTENTION_BACKEND}"
  --mamba-cache-mode "${MAMBA_CACHE_MODE}"
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --no-enable-prefix-caching
  --no-async-scheduling
)

prefill_args=(
  --port "${PREFILL_PORT}"
  "${common_args[@]}"
  --compilation-config "${prefill_compilation_config}"
  --block-size "${BLOCK_SIZE_PREFILL}"
  --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE_PREFILL}"
  --tensor-parallel-size "${PREFILL_TP}"
  --prefill-context-parallel-size "${PREFILL_PCP}"
  --cp-kv-cache-interleave-size 256
  --kv-transfer-config "${prefill_kv_config}"
)

decode_args=(
  --port "${DECODE_PORT}"
  "${common_args[@]}"
  --compilation-config "${decode_compilation_config}"
  --block-size "${BLOCK_SIZE_DECODE}"
  --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE_DECODE}"
  --tensor-parallel-size "${DECODE_TP}"
  --prefill-context-parallel-size 1
  --data-parallel-size "${DECODE_DP}"
  --data-parallel-size-local "${DECODE_DP}"
  --kv-transfer-config "${decode_kv_config}"
)

common_env=(
  PYTHONUNBUFFERED=1
  "JAX_PLATFORMS=tpu,cpu"
  PJRT_DEVICE=TPU
  TPU_BACKEND_TYPE=jax
  VLLM_TARGET_DEVICE=tpu
  SKIP_JAX_PRECOMPILE=1
  VLLM_XLA_CHECK_RECOMPILATION=0
  USE_MOE_SPARSE_CORE=1
  TPUMISC_HMA_CHECKSUM_TRACE=0
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  "LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-} --xla_tpu_scoped_vmem_limit_kib=65536"
  ONEHOT_MOE_PERMUTE_THRESHOLD=32768
  TPU_RAGGED_GATHER_REDUCE_IMPL=fallback
  TPU_RAGGED_GATHER_IMPL=fallback
  DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0
  "TPU_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT_BASE}"
  TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  TPU_KV_RESHARD_TRANSPORT=zmq
  TPU_KV_RESHARD_DST_PAGE_TOKENS=4096
  TPU_GDN_CONV_STATE_TILE_PAD=1
)

run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
prefill_namespace="prefill_pcp4_dp4_${run_stamp}"
decode_namespace="decode_pcp4_dp4_${run_stamp}"

{
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
  echo "CONNECTOR=TPUConnectorV2"
  echo "CONNECTOR_MODULE=vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector"
  echo "TPU_KV_RESHARD_TRANSPORT=zmq"
  echo "PREFILL_TP=${PREFILL_TP}"
  echo "PREFILL_PCP=${PREFILL_PCP}"
  echo "DECODE_TP=${DECODE_TP}"
  echo "DECODE_DP=${DECODE_DP}"
  echo "PREFILL_TPU_LOCAL_RANK_OFFSET=${PREFILL_TPU_LOCAL_RANK_OFFSET}"
  echo "DECODE_TPU_LOCAL_RANK_OFFSET=${DECODE_TPU_LOCAL_RANK_OFFSET}"
  echo "PREFILL_PORT=${PREFILL_PORT}"
  echo "DECODE_PORT=${DECODE_PORT}"
  echo "PROXY_PORT=${PROXY_PORT}"
  echo "PREFILL_KV_PORT_BASE=${PREFILL_KV_PORT_BASE}"
  echo "DECODE_KV_PORT_BASE=${DECODE_KV_PORT_BASE}"
  echo "SIDE_CHANNEL_PORT_BASE=${SIDE_CHANNEL_PORT_BASE}"
} >"${RUN_DIR}/launch_params.txt"

echo "Starting prefill server on ${SERVE_HOST}:${PREFILL_PORT}"
setsid env \
  -u TPU_USE_RAIDEN_CONNECTOR \
  -u TPU_USE_RAIDEN_KV_CACHE_MANAGER \
  -u TPU_RAIDEN_QWEN35_ADMISSION \
  -u TPU_RAIDEN_JOB_NAME \
  -u TPU_RAIDEN_ENGINE_ID \
  -u TPU_RAIDEN_CONTROLLER_ADDRESS \
  -u TPU_RAIDEN_TRANSFER_PARALLELISM \
  -u TPU_KV_RESHARD_TRANSFER_TAGS \
  -u VLLM_DISABLE_COMPILE_CACHE \
  -u VLLM_CACHE_ROOT \
  -u TORCHINDUCTOR_CACHE_DIR \
  "${common_env[@]}" \
  "DEBUG_TPU_LOCAL_RANK_OFFSET=${PREFILL_TPU_LOCAL_RANK_OFFSET}" \
  "TPU_LOCAL_RANK_OFFSET=${PREFILL_TPU_LOCAL_RANK_OFFSET}" \
  "TPU_KV_TRANSFER_NAMESPACE=${prefill_namespace}" \
  "TPU_KV_TRANSFER_PORT=${PREFILL_KV_PORT_BASE}" \
  python -m vllm.entrypoints.openai.api_server "${prefill_args[@]}" \
  >"${RUN_DIR}/logs/prefill.log" 2>&1 &
echo "$!" >"${RUN_DIR}/prefill.pid"

echo "Starting decode server on ${SERVE_HOST}:${DECODE_PORT}"
setsid env \
  -u TPU_USE_RAIDEN_CONNECTOR \
  -u TPU_USE_RAIDEN_KV_CACHE_MANAGER \
  -u TPU_RAIDEN_QWEN35_ADMISSION \
  -u TPU_RAIDEN_JOB_NAME \
  -u TPU_RAIDEN_ENGINE_ID \
  -u TPU_RAIDEN_CONTROLLER_ADDRESS \
  -u TPU_RAIDEN_TRANSFER_PARALLELISM \
  -u TPU_KV_RESHARD_TRANSFER_TAGS \
  -u VLLM_DISABLE_COMPILE_CACHE \
  -u VLLM_CACHE_ROOT \
  -u TORCHINDUCTOR_CACHE_DIR \
  "${common_env[@]}" \
  "DEBUG_TPU_LOCAL_RANK_OFFSET=${DECODE_TPU_LOCAL_RANK_OFFSET}" \
  "TPU_LOCAL_RANK_OFFSET=${DECODE_TPU_LOCAL_RANK_OFFSET}" \
  "TPU_KV_TRANSFER_NAMESPACE=${decode_namespace}" \
  "TPU_KV_TRANSFER_PORT=${DECODE_KV_PORT_BASE}" \
  python -m vllm.entrypoints.openai.api_server "${decode_args[@]}" \
  >"${RUN_DIR}/logs/decode.log" 2>&1 &
echo "$!" >"${RUN_DIR}/decode.pid"

wait_for_prefill_and_decode

echo "Starting P/D proxy on ${SERVE_HOST}:${PROXY_PORT}"
setsid python -u examples/disagg/toy_proxy_server.py \
  --host "${SERVE_HOST}" \
  --port "${PROXY_PORT}" \
  --prefiller-hosts "${SERVE_HOST}" \
  --prefiller-ports "${PREFILL_PORT}" \
  --decoder-hosts "${SERVE_HOST}" \
  --decoder-ports "${DECODE_PORT}" \
  >"${RUN_DIR}/logs/proxy.log" 2>&1 &
echo "$!" >"${RUN_DIR}/proxy.pid"

wait_for_http \
  "P/D proxy" \
  "http://${SERVE_HOST}:${PROXY_PORT}/healthcheck" \
  "${RUN_DIR}/proxy.pid" \
  "${RUN_DIR}/logs/proxy.log"

echo "Sending deterministic correctness request through the P/D proxy"
timeout "${CORRECTNESS_TIMEOUT_S}" \
  python scripts/vllm/integration/smoke_prefix_cache_correctness.py \
    --host "${SERVE_HOST}" \
    --port "${PROXY_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --quick-probe-only \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"

grep -q '^QUICK_PROBE_OK 1$' "${RUN_DIR}/logs/correctness.log" ||
  die "The P/D correctness probe did not report success"
grep -Eq '"cached_tokens": [1-9][0-9]*' \
  "${RUN_DIR}/logs/correctness.log" ||
  die "The decode response did not report transferred prompt tokens"

echo "PCP4_DP4_E2E_OK"
