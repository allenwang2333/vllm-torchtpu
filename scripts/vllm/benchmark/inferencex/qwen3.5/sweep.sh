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

# Sweep: restart the server per concurrency.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

# "ISL:OSL" pairs to sweep, and the concurrencies to run for each.
# Both are overridable via env so per-point jobs can each own a slice.
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256 512 1024}"
# Per-point sharding, followed by DP8 slots, global token budget and
# release group.
declare -A POINT_TABLE=(
  [8192,1024,4]="TP8_EP"
  [8192,1024,8]="TP8_EP"
  [8192,1024,16]="TP8_EP"
  [8192,1024,32]="TP8_EP"
  [8192,1024,64]="DP8_EP 16 8192 8"
  [8192,1024,128]="DP8_EP 20 8192 16"
  [8192,1024,256]="DP8_EP 40 3072 8"
  [8192,1024,512]="DP8_EP 80 6144 16"
  [8192,1024,1024]="DP8_EP 144 16384 16"
)

# The attention kernel's decode batch per point (the TP8 points' placement
# with 128 expert copies is set in start_server).
declare -A POINT_FLAGS=(
  [8192,1024,4]="TPU_RPA_DECODE_BATCH=4"
  [8192,1024,8]="TPU_RPA_DECODE_BATCH=8"
  [8192,1024,16]="TPU_RPA_DECODE_BATCH=8"
  [8192,1024,32]="TPU_RPA_DECODE_BATCH=8"
)

sweep_status=0

VLLM_PROCS='vllm serve|VLLM::'

# A point's settings are passed to server.sh explicitly (below); everything
# else comes from the caller's environment, as upstream's sweep does.

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
    return 1
  fi
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://0.0.0.0:${PORT}/health" 2>/dev/null)" = "200" ] || break
    sleep 2
  done
  # The next server cannot open the accelerators until the last one has let go
  # of them: wait for the device files to have no holders rather than trusting
  # a fixed sleep.
  for _ in $(seq 1 60); do
    holders=0
    for dev in /dev/vfio/* /dev/accel*; do
      [ -e "$dev" ] || continue
      n="$(fuser "$dev" 2>/dev/null | wc -w)"
      holders=$((holders + n))
    done
    [ "$holders" -eq 0 ] && break
    sleep 5
  done
  sleep "${TPU_RELEASE_WAIT:-20}"
}

start_server() {
  # $1 = the point's settings: the sharding, then (DP8_EP points) requests
  # per engine, tokens per step across the engines, release group, head.
  local sharding seqs budget group
  read -r sharding seqs budget group <<< "$1"
  # The point carries its own workload and concurrency rather than relying on
  # the sweep's exports reaching the server through the environment.
  local point_env=(SHARDING="$sharding" ISL="$ISL" OSL="$OSL" CONC="$CONC")
  if [ -n "${seqs:-}" ]; then
    point_env+=(MAX_NUM_SEQS="$seqs" GLOBAL_MAX_NUM_BATCHED_TOKENS="$budget"
                DP_SCHED_BUFFER_THRESHOLD="$group")
  fi
  local point_flags="${POINT_FLAGS[$ISL,$OSL,$CONC]:-}"
  local flags=()
  read -r -a flags <<< "$point_flags"
  point_env+=("${flags[@]}")
  if [ "$ISL,$OSL" = "8192,1024" ] && [ "$sharding" = "TP8_EP" ]; then
    case "$CONC" in
      4|8|16|32) point_env+=("TPU_EXPERT_PLACEMENT=${SCRIPT_DIR}/expert_placement_dp8_ep8_copies128.json") ;;
    esac
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    env "${point_env[@]}" bash "${SCRIPT_DIR}/server.sh"
    return
  fi
  SERVER_LOG="$(mktemp "/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_conc${CONC}_XXXXXX.log")" || return 1
  echo "--- starting server ${point_env[*]} ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  env "${point_env[@]}" bash "${SCRIPT_DIR}/server.sh" > >(tee -a "$SERVER_LOG") 2>&1 &
  SERVER_PID=$!
  local waited=0
  until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited during startup (see $SERVER_LOG)" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge "$READY_TIMEOUT" ] && { echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; return 1; }
  done
  echo "--- server ready ---"
}

if [ "${DRY_RUN:-0}" != "1" ]; then trap stop_server EXIT; fi

for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for CONC in "${CONCS[@]}"; do
    export CONC
    # SHARDING_OVERRIDE keeps its meaning: one sharding for every point.
    point="${POINT_OVERRIDE:-${SHARDING_OVERRIDE:-${POINT_TABLE[$ISL,$OSL,$CONC]:-}}}"
    [ -n "$point" ] || { echo "ERROR: no settings in POINT_TABLE for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    if [ "${DRY_RUN:-0}" = "1" ]; then
      start_server "$point" || exit 1
      continue
    fi
    stop_server || exit 1
    started=0
    for attempt in 1 2; do
      if start_server "$point"; then started=1; break; fi
      echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
      TPU_RELEASE_WAIT=60 stop_server || exit 1
    done
    if [ "$started" -ne 1 ]; then
      echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
      sweep_status=1
      continue
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC POINT=$point ##########"
    BENCH_LOG="$(mktemp "/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_XXXXXX.log")" || exit 1
    echo "--- benchmark (log: $BENCH_LOG) ---"
    # CLIENT runs against each point's server: bench.sh (the InferenceX
    # client), or eval.sh for the same points scored on GSM8K.
    if ! bash "${CLIENT:-${SCRIPT_DIR}/bench.sh}" 2>&1 | tee -a "$BENCH_LOG"; then
      echo "ERROR: benchmark failed for concurrency $CONC (see $BENCH_LOG)" >&2
      sweep_status=1
    fi
  done
done
echo "########## sweep complete (status $sweep_status) ##########"
exit "$sweep_status"
