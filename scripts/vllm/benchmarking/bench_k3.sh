#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Ephemeral-server benchmark runner for Kimi-K3 across 4 hosts (TP=32).
#
# run_benchmarks.sh launches ONE local server and has no concept of headless
# ranks, so it cannot drive a model that spans 4 hosts. This runner keeps its
# disciplines -- max-concurrency, pinned temperature, discarded warmup, JSON
# per cell, fixed seed -- and adds the multi-host launch.
#
# The server is EPHEMERAL: serve -> wait -> benchmark -> evaluate -> exit.
# The process ends, so the pods terminate and the pool frees itself. No
# watchdog, no manual delete.
#
# Rank 0 serves and measures. Ranks 1..N-1 run `vllm serve --headless` with
# identical parallel args and rendezvous over torch.distributed.
#
# Usage: bench_k3.sh --config <name> [--rank N] [--nnodes N] [--leader HOST]
#                    [--results-dir DIR] [--record-identity] [--skip-eval]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME=""; RANK="${JOB_COMPLETION_INDEX:-0}"; NNODES=4; LEADER=""
LEADER_ADDR=""
RESULTS_DIR="/tmp/out/bench"; PORT=8000
RECORD_IDENTITY=0; SKIP_EVAL=0; EXTRA_SERVE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)        CONFIG_NAME="$2"; shift 2 ;;
    --rank)          RANK="$2"; shift 2 ;;
    --nnodes)        NNODES="$2"; shift 2 ;;
    --leader)        LEADER="$2"; shift 2 ;;
    --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --record-identity) RECORD_IDENTITY=1; shift ;;
    --skip-eval)     SKIP_EVAL=1; shift ;;
    --extra-serve)   EXTRA_SERVE="$2"; shift 2 ;;   # the knob under test
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$CONFIG_NAME" ] || { echo "ERROR: --config is required"; exit 2; }
LEADER_ADDR="${LEADER:-127.0.0.1}"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
mkdir -p "$RESULTS_DIR"

SERVE=(--served-model-name "$MODEL"
  --tensor-parallel-size "$TENSOR_PARALLELISM"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --attention-backend "${ATTENTION_BACKEND:-CUSTOM}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.7}"
  --trust-remote-code)
[ "${ENABLE_EP:-false}" = "true" ] && SERVE+=(--enable-expert-parallel)
[ -n "${KV_CACHE_DTYPE:-}" ]       && SERVE+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "${COMPILE_SIZES:-}" ]        && SERVE+=(--compilation-config "{\"compile_sizes\":${COMPILE_SIZES}}")
[ -n "${CHAT_TEMPLATE:-}" ]        && SERVE+=(--chat-template "$CHAT_TEMPLATE")
if [ "$NNODES" -gt 1 ]; then
  SERVE+=(--nnodes "$NNODES" --node-rank "$RANK" --data-parallel-backend mp
          --master-addr "$LEADER" --master-port 29500)
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_SERVE_ARGS:-}" ] && SERVE+=($EXTRA_SERVE_ARGS)
# shellcheck disable=SC2206
[ -n "$EXTRA_SERVE" ] && SERVE+=($EXTRA_SERVE)

TARGET="${MODEL_URI:-$MODEL}"
[ -n "${MODEL_URI:-}" ] && SERVE+=(--load-format runai_streamer)

# ---- headless ranks: serve, then follow the leader out.
#
# `vllm serve --headless` never returns, so an exec here leaves the rank running
# after the leader finishes. With completions=N the JobSet then never completes
# and the slice keeps its chips until somebody deletes it. Measured once: rank 0
# Completed while ranks 1-3 held 16 chips for 20 minutes.
#
# Instead: run headless in the background and poll the leader. The leader is up
# once its port answers; when it stops answering after having been up, the run
# is over and this rank exits too.
if [ "$RANK" != "0" ]; then
  echo "rank $RANK: headless, following leader $LEADER_ADDR"
  vllm serve "$TARGET" "${SERVE[@]}" --headless > "$RESULTS_DIR/worker.log" 2>&1 &
  WORKER_PID=$!
  LEADER_WAS_UP=0
  while true; do
    sleep 20
    kill -0 "$WORKER_PID" 2>/dev/null || { echo "rank $RANK: worker exited"; break; }
    if curl -sf --max-time 5 "http://${LEADER_ADDR}:${PORT}/health" >/dev/null 2>&1; then
      LEADER_WAS_UP=1
    elif [ "$LEADER_WAS_UP" = "1" ]; then
      echo "rank $RANK: leader went away, shutting down"
      kill -TERM "$WORKER_PID" 2>/dev/null || true
      sleep 10; kill -9 "$WORKER_PID" 2>/dev/null || true
      break
    fi
  done
  echo "rank $RANK: done"
  exit 0
fi

# ---- leader
# The server is a background child holding this script's stdout. Without an
# explicit kill it survives the script, `tail`/`grep` downstream never see EOF,
# and a caller running arms in sequence wedges forever. Measured once: arm A
# finished its cells, then the sweep sat for 30 minutes with arm B never
# starting. Trap covers the error paths too.
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM

echo "[cmd] vllm serve $TARGET ${SERVE[*]} --host 0.0.0.0 --port $PORT"
vllm serve "$TARGET" "${SERVE[@]}" --host 0.0.0.0 --port "$PORT" \
  > "$RESULTS_DIR/server.log" 2>&1 &
  > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!

T0=$SECONDS
MAXWAIT=$(( ${SERVER_READY_WAIT_MIN:-40} * 60 ))
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  (( SECONDS - T0 > MAXWAIT )) && { echo "TIMEOUT after $((SECONDS-T0))s"; tail -120 "$RESULTS_DIR/server.log"; exit 1; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server died"; tail -120 "$RESULTS_DIR/server.log"; exit 1; }
  (( (SECONDS - T0) % 180 == 0 )) && echo "[$((SECONDS-T0))s] booting..."
  sleep 10
done
READY_S=$((SECONDS-T0)); echo "SERVER READY after ${READY_S}s"

bench () {   # $1=isl $2=osl $3=concurrency $4=label
  local isl=$1 osl=$2 conc=$3 label=$4
  vllm bench serve --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --trust-remote-code \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio 0.0 \
    --num-prompts $(( conc * ${PROMPTS_PER_CONC:-5} )) \
    --max-concurrency "$conc" --request-rate inf \
    --temperature "${BENCHMARK_TEMPERATURE:-0}" --ignore-eos --seed 42 \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99" \
    --save-result --result-filename "$RESULTS_DIR/${label}.json" 2>&1 | tail -26
}

for cell in ${ISL_OSL_CONFIGS}; do
  isl="${cell%%:*}"; osl="${cell##*:}"
  for conc in ${CONCURRENCY_OPTIONS}; do
    echo "########## WARMUP isl=$isl osl=$osl c=$conc (discarded) ##########"
    bench "$isl" "$osl" "$conc" "warmup_isl${isl}_osl${osl}_c${conc}" >/dev/null 2>&1
    echo "########## BENCH isl=$isl osl=$osl c=$conc ##########"
    bench "$isl" "$osl" "$conc" "isl${isl}_osl${osl}_c${conc}"
  done
done

# ---- Tier 0: output identity
echo "########## TIER 0 — output identity ##########"
for f in "$RESULTS_DIR"/isl*_osl*_c*.json; do
  base="$SCRIPT_DIR/baselines/identity/${CONFIG_NAME}.$(basename "$f" .json).json"
  if [ "$RECORD_IDENTITY" = "1" ]; then
    python3 "$SCRIPT_DIR/check_identity.py" --record "$f" --baseline "$base"
  else
    python3 "$SCRIPT_DIR/check_identity.py" --result "$f" --baseline "$base" || IDENTITY_FAIL=1
  fi
done

# ---- Tier 1: accuracy, against the same server
if [ "$SKIP_EVAL" != "1" ] && [ -n "${EVAL_TASKS:-}" ]; then
  echo "########## TIER 1 — accuracy (${EVAL_TASKS}) ##########"
  bash "$SCRIPT_DIR/run_eval_flow.sh" --config "$CONFIG_NAME" \
       --results-dir "$RESULTS_DIR" --host 127.0.0.1 --port "$PORT" --run-lm-eval \
       2>&1 | tail -40
fi

echo "########## SUMMARY ##########"
echo "ready_s=$READY_S"
for f in "$RESULTS_DIR"/isl*_osl*_c*.json; do
  python3 -c "
import json,sys
d=json.load(open('$f'))
print('$(basename "$f" .json)', '|',
      'out_tok_s=%.2f' % d.get('output_throughput',0),
      'tpot_ms=%.2f' % d.get('median_tpot_ms',0),
      'ttft_ms=%.2f' % d.get('median_ttft_ms',0),
      'completed=%s' % d.get('completed'))"
done
echo "identity_fail=${IDENTITY_FAIL:-0}"
cleanup_server
echo "########## BENCH_K3_DONE ##########"
