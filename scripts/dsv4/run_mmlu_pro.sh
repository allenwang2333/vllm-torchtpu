#!/bin/bash
# Run the MMLU-Pro lm_eval benchmark against DeepSeek-V4-Flash on this node:
# starts the server, waits for health, runs lm_eval, tears the server down.
#
#   scripts/dsv4/run_mmlu_pro.sh [run_name]
#
# Useful overrides:
#   LIMIT=10             # lm_eval --limit (smoke tests; omit for the full run)
#   NUM_CONCURRENT=128   # request concurrency
#   NUM_FEWSHOT=5
#   GPU_MEM_UTIL=0.7 MAX_MODEL_LEN=9216 ...   (see dsv4_env.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./dsv4_env.sh disable=SC1091
source "$SCRIPT_DIR/dsv4_env.sh"

RUN_NAME="${1:-mmlu_pro_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/eval_results/$RUN_NAME}"
mkdir -p "$RESULTS_DIR"
SERVER_LOG="$RESULTS_DIR/vllm_server.log"
EVAL_LOG="$RESULTS_DIR/lm_eval.log"

trap dsv4_kill_server EXIT INT TERM

"$SCRIPT_DIR/serve_dsv4.sh" "$SERVER_LOG"

EVAL_ARGS=(
    --model local-completions
    --model_args "model=$SERVED_MODEL_NAME,base_url=http://localhost:$PORT/v1/completions,num_concurrent=${NUM_CONCURRENT:-128},max_gen_toks=${MAX_GEN_TOKS:-4096},timeout=72000,max_length=${MAX_LENGTH:-8192},tokenizer_backend=huggingface,tokenizer=$MODEL_PATH"
    --tasks mmlu_pro
    --num_fewshot "${NUM_FEWSHOT:-5}"
    --batch_size auto
    --seed 42
    --output_path "$RESULTS_DIR"
)
[[ -n "${LIMIT:-}" ]] && EVAL_ARGS+=(--limit "$LIMIT")
[[ "${LOG_SAMPLES:-0}" == "1" ]] && EVAL_ARGS+=(--log_samples)

echo "Running lm_eval -> $EVAL_LOG"
printf '  %s\n' "${EVAL_ARGS[@]}"
API_KEY=EMPTY "$VENV/bin/lm_eval" "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_LOG"

echo "Done. Results in $RESULTS_DIR"
