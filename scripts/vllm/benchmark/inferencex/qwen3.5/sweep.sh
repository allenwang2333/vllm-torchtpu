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

# Unified Sweep Benchmark Runner
# Supports:
#   1. Standalone Single-Host VM mode (default): runs server in background and executes benchmarks locally.
#   2. Dual-Pod / Cloud-Native mode:
#      - Server Pod: bash sweep.sh --server
#      - Client Pod: bash sweep.sh --client
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_URL="${VLLM_URL:-http://${VLLM_HOST}:${PORT}}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers cold compiles)

# "ISL:OSL" pairs to sweep, and the concurrencies to run for each.
# Both are overridable via env so per-point jobs can each own a slice.
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256}"

# Sharding per swept point, keyed by "ISL,OSL,CONC" (every point listed).
# Low concurrency runs TP8_EP, higher concurrency runs DP4TP2_EP.
declare -A SHARDING_TABLE=(
  [8192,1024,4]=TP8_EP
  [8192,1024,8]=TP8_EP
  [8192,1024,16]=TP8_EP
  [8192,1024,32]=DP4TP2_EP
  [8192,1024,64]=DP4TP2_EP
  [8192,1024,128]=DP4TP2_EP
  [8192,1024,256]=DP4TP2_EP
)

ROLE="${1:-all}"

# Query point configuration from sharding table
if [ "$ROLE" = "--point-config" ]; then
  key="${2:-8192},${3:-1024},${4:-64}"
  echo "${SHARDING_TABLE[$key]:-}"
  exit 0
fi

# =============================================================================
# Role Mode 1: Server Pod (Dedicated TPU Pod in Dual-Pod Architecture)
# =============================================================================
if [ "$ROLE" = "--server" ]; then
  echo "============================================================"
  echo "=== Running in Server Mode (TPU Pod) ==="
  echo "=== Host: $HOST, Port: $PORT, Sharding: ${SHARDING:-TP8_EP} ==="
  echo "============================================================"
  export HOST="$HOST"
  export PORT="$PORT"
  exec bash "${SCRIPT_DIR}/server.sh"
fi

# =============================================================================
# Role Mode 2: Client Pod (Dedicated CPU Benchmark Pod in Dual-Pod Architecture)
# =============================================================================
if [ "$ROLE" = "--client" ]; then
  echo "============================================================"
  echo "=== Running in Client Mode (CPU Benchmark Pod) ==="
  echo "=== Target URL: $VLLM_URL ==="
  echo "=== Workloads: ${WORKLOADS[*]} ==="
  echo "=== Concurrencies: ${CONCS[*]} ==="
  echo "============================================================"

  # Wait for remote server to become healthy
  echo "Waiting for vLLM server to become healthy at ${VLLM_URL}/health..."
  waited=0
  until curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$READY_TIMEOUT" ]; then
      echo "ERROR: vLLM server not ready at ${VLLM_URL} after ${READY_TIMEOUT}s" >&2
      exit 1
    fi
  done
  echo "--- vLLM server is healthy and ready! Starting client sweeps ---"

  overall_client_exit=0
  for wl in "${WORKLOADS[@]}"; do
    export ISL="${wl%%:*}" OSL="${wl#*:}"
    for CONC in "${CONCS[@]}"; do
      export CONC
      sharding="${SHARDING_OVERRIDE:-${SHARDING_TABLE[$ISL,$OSL,$CONC]:-}}"
      echo "########## Client Benchmark: ISL=$ISL OSL=$OSL CONC=$CONC (Configured Sharding: ${sharding:-TP8_EP}) ##########"
      BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
      echo "--- benchmark (log: $BENCH_LOG) ---"
      if ! VLLM_URL="$VLLM_URL" bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee -a "$BENCH_LOG"; then
        echo "ERROR: benchmark failed for ISL=$ISL OSL=$OSL CONC=$CONC" >&2
        overall_client_exit=1
      fi
    done
  done
  echo "########## Client sweep complete (exit: $overall_client_exit) ##########"
  exit "$overall_client_exit"
fi

# =============================================================================
# Role Mode 3: Standalone Single-Host VM Mode (Default backward-compatible)
# =============================================================================
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
  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
  echo "--- starting server SHARDING=$1 ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  HOST="$HOST" PORT="$PORT" SHARDING="$1" bash "${SCRIPT_DIR}/server.sh" > >(tee -a "$SERVER_LOG") 2>&1 &
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

for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for CONC in "${CONCS[@]}"; do
    export CONC
    sharding="${SHARDING_OVERRIDE:-${SHARDING_TABLE[$ISL,$OSL,$CONC]:-}}"
    [ -n "$sharding" ] || { echo "ERROR: no sharding in SHARDING_TABLE for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    stop_server
    started=0
    for attempt in 1 2; do
      if start_server "$sharding"; then started=1; break; fi
      echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
      TPU_RELEASE_WAIT=60 stop_server
    done
    if [ "$started" -ne 1 ]; then
      echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
      continue
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$sharding ##########"
    BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
    echo "--- benchmark (log: $BENCH_LOG) ---"
    bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee -a "$BENCH_LOG"
  done
done
echo "########## sweep complete ##########"
