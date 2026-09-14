#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kimi-K3 1P1D smoke on two single-host roles (Raiden Stage-3, PR #857):
#   ROLE=head   -> launch the prefill server locally, wait for both roles,
#                  start the toy proxy, run a small `vllm bench serve`, and
#                  keep every log under RUN_DIR.
#   ROLE=decode -> launch the decode server and block on it.
# Driven by .buildkite/scripts/run_disagg_multihost.sh (DISAGG_ROLE_SCRIPT);
# every knob arrives through the environment.

set -euo pipefail
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

ROLE="${ROLE:-head}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required (HF id, local dir, or gs:// URI)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"
TOKENIZER="${TOKENIZER:-${SERVED_MODEL_NAME}}"
RUN_ROOT="${RUN_ROOT:-/tmp/kimi_k3_pd_smoke}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"
PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"

# Kimi Stage-3 topology: PCP1, producer DP1, producer TP == transfer
# parallelism, decode TP == TPU_RAIDEN_DST_SHARDS, and dst | src.
export RAIDEN_ADMISSION="${RAIDEN_ADMISSION:-kimik3}"
export PREFILL_PCP="${PREFILL_PCP:-1}"
export PREFILL_TP="${PREFILL_TP:-8}"
export DECODE_DP="${DECODE_DP:-1}"
export DECODE_TP="${DECODE_TP:-8}"
export TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-${PREFILL_TP}}"
export TPU_RAIDEN_DST_SHARDS="${TPU_RAIDEN_DST_SHARDS:-${DECODE_TP}}"
# Kimi pool admission needs bf16 caches; PR #857's validated serve settings.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
export PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-16}"
export DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-1,8,512}"
export DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-1,8,512}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
# No MTP draft for Kimi-K3: explicitly empty disables the Qwen defaults.
export PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-}"
export DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-}"
# Raiden knobs PR #857 validated with.
export TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
export TPU_RAIDEN_TRANSFER_NUM_SLOTS="${TPU_RAIDEN_TRANSFER_NUM_SLOTS:-1}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS="${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
export MODEL_PATH SERVED_MODEL_NAME RUN_ROOT RUN_DIR PREFILL_HOST DECODE_HOST
export PREFILL_PORT DECODE_PORT STARTUP_TIMEOUT_S

# Bench client (small by design: this is a smoke, not a benchmark).
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-2}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-64}"

mkdir -p "${RUN_ROOT}" "${RUN_DIR}/logs"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: no python3 in PATH" >&2; exit 2; }

cleanup() {
  local pid_file pid
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 3
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

env | grep -E "^(TPU_|VLLM_|RAIDEN_|PREFILL_|DECODE_|MODEL_|SERVED_|KV_|LOAD_|ENABLE_|MAX_|BENCH_|EXTRA_)" | sort \
  > "${RUN_DIR}/effective_env.txt" || true

wait_health() {  # host port label
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if [[ "$(curl -fsS -o /dev/null -w '%{http_code}' "http://$1:$2/health" 2>/dev/null || true)" == "200" ]]; then
      echo "$3 healthy at $1:$2"; return 0
    fi
    sleep 10
  done
  echo "ERROR: $3 not healthy within ${STARTUP_TIMEOUT_S}s" >&2; return 1
}

cd "${repo_root}"
case "${ROLE}" in
  decode)
    echo "--- Launching role=decode (TP=${DECODE_TP}, DP=${DECODE_DP}, PORT=${DECODE_PORT})"
    ROLE=decode bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
      2>&1 | tee "${RUN_DIR}/logs/launch_decode.log"
    decode_pid="$(cat "${RUN_DIR}/decode.pid" 2>/dev/null || true)"
    [[ -n "${decode_pid}" ]] || { echo "ERROR: decode launch wrote no pid" >&2; exit 1; }
    echo "Decode server running (PID ${decode_pid}); blocking on it"
    while kill -0 "${decode_pid}" 2>/dev/null; do sleep 5; done
    ;;
  head)
    echo "--- Launching role=prefill (TP=${PREFILL_TP}, PCP=${PREFILL_PCP}, PORT=${PREFILL_PORT})"
    ROLE=prefill bash "${repo_root}/examples/disagg/launch_qwen35_disagg_multihost.sh" \
      2>&1 | tee "${RUN_DIR}/logs/launch_prefill.log"
    wait_health "${PREFILL_HOST}" "${PREFILL_PORT}" "prefill"
    wait_health "${DECODE_HOST}" "${DECODE_PORT}" "decode"

    echo "--- Starting toy proxy on :${PROXY_PORT}"
    setsid "${PYTHON_BIN}" "${repo_root}/examples/disagg/toy_proxy_server.py" \
      --host 0.0.0.0 --port "${PROXY_PORT}" \
      --prefiller-hosts "${PREFILL_HOST}" --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${DECODE_HOST}" --decoder-ports "${DECODE_PORT}" \
      >"${RUN_DIR}/logs/proxy.log" 2>&1 &
    echo $! >"${RUN_DIR}/proxy.pid"
    for _ in $(seq 1 30); do
      [[ "$(curl -fsS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PROXY_PORT}/healthcheck" 2>/dev/null || true)" == "200" ]] && break
      sleep 2
    done

    echo "--- One warm-up request through the proxy"
    curl -sS -m 600 "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"The TPU is\",\"max_tokens\":16,\"temperature\":0}" \
      | tee "${RUN_DIR}/warmup_response.json"; echo

    echo "--- vllm bench serve: ${BENCH_NUM_PROMPTS} prompts, conc ${BENCH_MAX_CONCURRENCY}, ${BENCH_INPUT_LEN}/${BENCH_OUTPUT_LEN}"
    set +e
    vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:${PROXY_PORT}" \
      --model "${SERVED_MODEL_NAME}" \
      --tokenizer "${TOKENIZER}" \
      --trust-remote-code \
      --dataset-name random \
      --random-input-len "${BENCH_INPUT_LEN}" \
      --random-output-len "${BENCH_OUTPUT_LEN}" \
      --random-range-ratio 0 \
      --num-prompts "${BENCH_NUM_PROMPTS}" \
      --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
      --request-rate inf \
      --ignore-eos \
      --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result --result-dir "${RUN_DIR}" \
      --result-filename "bench_isl${BENCH_INPUT_LEN}_osl${BENCH_OUTPUT_LEN}_c${BENCH_MAX_CONCURRENCY}.json" \
      2>&1 | tee "${RUN_DIR}/logs/bench.log"
    BENCH_EXIT=${PIPESTATUS[0]}
    set -e

    echo "--- Raiden transfer lines from the prefill log (first 40)"
    grep -iE "raiden|reshard|stage-3|stage3|transfer" "${RUN_DIR}/logs/prefill.log" | head -40 || true
    echo "--- bench exit ${BENCH_EXIT}"
    exit "${BENCH_EXIT}"
    ;;
  *)
    echo "ERROR: ROLE must be head or decode: ${ROLE}" >&2; exit 2 ;;
esac
