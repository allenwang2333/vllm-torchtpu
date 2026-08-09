#!/bin/bash
# Start a DeepSeek-V4-Flash vLLM server on this node and wait until it is
# healthy. Leaves the server running (use `dsv4_kill_server` or pkill to stop).
#
#   scripts/dsv4/serve_dsv4.sh [server.log]
#
# Tunables come from scripts/dsv4/dsv4_env.sh and can be overridden per-run:
#   MAX_MODEL_LEN=1024 GPU_MEM_UTIL=0.45 scripts/dsv4/serve_dsv4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./dsv4_env.sh disable=SC1091
source "$SCRIPT_DIR/dsv4_env.sh"

SERVER_LOG="${1:-$WORKSPACE/eval_results/dsv4_server.log}"
mkdir -p "$(dirname "$SERVER_LOG")"

dsv4_kill_server

ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --no-enable-prefix-caching
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size "$TP_SIZE"
    --enable-expert-parallel
    --port "$PORT"
)
[[ -n "$BLOCK_SIZE" ]] && ARGS+=(--block-size "$BLOCK_SIZE")
# LOAD_FORMAT=dummy skips the safetensors load AND the MoE FP4 stage (~11 of
# the ~14 min startup), giving random weights. Valid ONLY for questions that
# cannot depend on weight values -- compile failures, buffer/VMEM allocation,
# KV sizing, shape and sharding errors. Never for accuracy: this model has
# already shipped a bug (all-zero MoE experts) that produced fluent, plausible
# output for weeks, and random weights are a stronger version of that trap.
[[ -n "${LOAD_FORMAT:-}" ]] && ARGS+=(--load-format "$LOAD_FORMAT")
[[ "$ASYNC_SCHEDULING" == "1" ]] && ARGS+=(--async-scheduling)
# EXTRA_ARGS: space-separated extra flags, for experiments (e.g.
# EXTRA_ARGS="--disable-hybrid-kv-cache-manager").
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=(${EXTRA_ARGS})
[[ "$VLLM_ENFORCE_EAGER" == "1" ]] && ARGS+=(--enforce-eager)

echo "Starting vLLM server -> $SERVER_LOG"
printf '  %s\n' "${ARGS[@]}"
"$VENV/bin/vllm" serve "${ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

# Weight load + requantization + compile is slow on a cold FP4 cache
# (~20 min observed); a warm cache is a few minutes.
DEADLINE=$((SECONDS + ${SERVER_TIMEOUT:-3600}))
while (( SECONDS < DEADLINE )); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "server healthy after ${SECONDS}s (pid $SERVER_PID)"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died. Log tail:" >&2
        tail -n 60 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 10
done

echo "ERROR: server did not become healthy within ${SERVER_TIMEOUT:-3600}s" >&2
tail -n 60 "$SERVER_LOG" >&2
exit 1
