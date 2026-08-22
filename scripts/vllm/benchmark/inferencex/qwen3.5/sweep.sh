#!/usr/bin/env bash
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

# Sweep the measured-best 8k1k configurations. TP8 c4-c64 intentionally share
# one server; DP8 c128 and c256 each use their own measured scheduler shape.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

# "ISL:OSL" pairs to sweep, and the concurrencies to run for each.
# Both are overridable via env so per-point jobs can each own a slice.
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256}"
# Every table is keyed by "ISL,OSL,CONC". Points with the same complete server
# configuration reuse the running server instead of recompiling per tier.
declare -A SHARDING_TABLE=(
  [8192,1024,4]=TP8_EP
  [8192,1024,8]=TP8_EP
  [8192,1024,16]=TP8_EP
  [8192,1024,32]=TP8_EP
  [8192,1024,64]=TP8_EP
  [8192,1024,128]=DP8_EP
  [8192,1024,256]=DP8_EP
)
declare -A MAX_BATCHED_TOKENS_TABLE=(
  [8192,1024,4]=16384
  [8192,1024,8]=16384
  [8192,1024,16]=16384
  [8192,1024,32]=16384
  [8192,1024,64]=16384
  [8192,1024,128]=256
  [8192,1024,256]=512
)
declare -A MAX_SEQS_TABLE=(
  [8192,1024,4]=256
  [8192,1024,8]=256
  [8192,1024,16]=256
  [8192,1024,32]=256
  [8192,1024,64]=256
  [8192,1024,128]=32
  [8192,1024,256]=64
)
declare -A PREFILL_INTERVAL_TABLE=(
  [8192,1024,4]=1
  [8192,1024,8]=1
  [8192,1024,16]=1
  [8192,1024,32]=1
  [8192,1024,64]=1
  [8192,1024,128]=1
  [8192,1024,256]=1
)
declare -A FUSED_EP_TABLE=(
  [8192,1024,4]=0
  [8192,1024,8]=0
  [8192,1024,16]=0
  [8192,1024,32]=0
  [8192,1024,64]=0
  [8192,1024,128]=1
  [8192,1024,256]=1
)

VLLM_PROCS='vllm serve|VLLM::'

stop_server() {
  pkill -TERM -f "$VLLM_PROCS" 2>/dev/null || true
  for _ in $(seq 1 15); do
    pgrep -f "$VLLM_PROCS" >/dev/null 2>&1 || break
    sleep 2
  done
  pkill -KILL -f "$VLLM_PROCS" 2>/dev/null || true
  # Do not proceed while anything vllm-shaped still exists.
  for _ in $(seq 1 30); do
    pgrep -f "$VLLM_PROCS" >/dev/null 2>&1 || break
    sleep 2
  done
  if pgrep -f "$VLLM_PROCS" >/dev/null 2>&1; then
    echo "WARN: vllm processes survived SIGKILL:" >&2
    pgrep -af "$VLLM_PROCS" >&2 || true
  fi
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://0.0.0.0:${PORT}/health" 2>/dev/null)" = "200" ] || break
    sleep 2
  done
  sleep "${TPU_RELEASE_WAIT:-20}"
}

start_server() {
  local sharding="$1"
  local max_batched_tokens="$2"
  local max_seqs="$3"
  local prefill_interval="$4"
  local fused_ep="$5"
  local -a server_env=(SHARDING="$sharding" CONC="$CONC")
  [ -n "$max_batched_tokens" ] && server_env+=(MAX_NUM_BATCHED_TOKENS="$max_batched_tokens")
  [ -n "$max_seqs" ] && server_env+=(MAX_NUM_SEQS="$max_seqs")
  [ -n "$prefill_interval" ] && server_env+=(PREFILL_SCHEDULE_INTERVAL="$prefill_interval")
  [ -n "$fused_ep" ] && server_env+=(USE_MOE_FUSED_EP_KERNEL="$fused_ep")

  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_${sharding}_conc${CONC}_$(date +%m%d-%H%M%S)_$$.log"
  echo "--- starting server SHARDING=$sharding ISL=$ISL OSL=$OSL CONC=$CONC MAX_NUM_BATCHED_TOKENS=${max_batched_tokens:-default} MAX_NUM_SEQS=${max_seqs:-default} PREFILL_SCHEDULE_INTERVAL=${prefill_interval:-default} USE_MOE_FUSED_EP_KERNEL=${fused_ep:-default} (log: $SERVER_LOG) ---"
  env "${server_env[@]}" bash "${SCRIPT_DIR}/server.sh" > >(tee "$SERVER_LOG") 2>&1 &
  SERVER_PID=$!
  local waited=0
  until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited during startup (see $SERVER_LOG)" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge "$READY_TIMEOUT" ] && { echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; return 1; }
  done
  echo "--- server ready ---"
}

trap stop_server EXIT

current_server_key=""
for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for CONC in "${CONCS[@]}"; do
    export CONC
    point="$ISL,$OSL,$CONC"
    if [ -n "${SHARDING_OVERRIDE:-}" ]; then
      # Preserve the one-off override workflow: let server.sh derive or inherit
      # its settings and restart for each concurrency.
      sharding="$SHARDING_OVERRIDE"
      max_batched_tokens=""
      max_seqs=""
      prefill_interval=""
      fused_ep=""
      server_key="override:$sharding:$point"
    else
      sharding="${SHARDING_TABLE[$point]:-}"
      max_batched_tokens="${MAX_BATCHED_TOKENS_TABLE[$point]:-}"
      max_seqs="${MAX_SEQS_TABLE[$point]:-}"
      prefill_interval="${PREFILL_INTERVAL_TABLE[$point]:-}"
      fused_ep="${FUSED_EP_TABLE[$point]:-}"
      server_key="$ISL:$OSL:$sharding:$max_batched_tokens:$max_seqs:$prefill_interval:$fused_ep"
    fi
    [ -n "$sharding" ] || { echo "ERROR: no sharding in SHARDING_TABLE for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    if [ "$server_key" != "$current_server_key" ]; then
      stop_server
      started=0
      for attempt in 1 2; do
        if start_server "$sharding" "$max_batched_tokens" "$max_seqs" "$prefill_interval" "$fused_ep"; then
          started=1
          break
        fi
        echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
        TPU_RELEASE_WAIT=60 stop_server
      done
      if [ "$started" -ne 1 ]; then
        echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
        current_server_key=""
        continue
      fi
      current_server_key="$server_key"
    else
      echo "--- reusing server for ISL=$ISL OSL=$OSL CONC=$CONC ---"
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$sharding ##########"
    BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M%S)_$$.log"
    echo "--- benchmark (log: $BENCH_LOG) ---"
    if ! bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee "$BENCH_LOG"; then
      echo "ERROR: benchmark failed for ISL=$ISL OSL=$OSL CONC=$CONC" >&2
    fi
  done
done
echo "########## sweep complete ##########"
