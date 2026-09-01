#!/bin/bash
# One server lifetime, three measurements for Kimi-K3.
#
# run_eval_flow.sh only scores through /v1/chat/completions, and the Kimi-K3
# checkpoint ships no chat template, so every accuracy number it could produce
# depends on a hand-written template being right. This driver puts the
# template-free measurement first:
#
#   1. run_benchmarks.sh --keep-alive   -> server up, one perf row
#   2. gsm8k 5-shot over /v1/completions -> accuracy with NO template involved
#   3. mmlu_llama over /v1/chat/completions -> also a test of the template
#
# Usage: k3_accuracy.sh --config <name> --results-dir <dir> [--port 8000]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME=""; RESULTS_DIR=""; PORT=8000
GSM8K_LIMIT="${GSM8K_LIMIT:-200}"; MMLU_LIMIT="${MMLU_LIMIT:-100}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)      CONFIG_NAME="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$CONFIG_NAME" ] || { echo "ERROR: --config is required"; exit 2; }
RESULTS_DIR="${RESULTS_DIR:-/tmp/k3_accuracy_$CONFIG_NAME}"
mkdir -p "$RESULTS_DIR"

trap 'echo "--- cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM
bash "$SCRIPT_DIR/cleanup_server.sh"

echo "--- 1/3 server + perf row ($CONFIG_NAME)"
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" \
  --keep-alive --port "$PORT" || { echo "::error::server/bench leg failed"; exit 1; }

MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
BASE="http://localhost:$PORT"
fail=0

echo "--- 2/3 gsm8k 5-shot via /v1/completions (no chat template)"
lm_eval --model local-completions \
  --model_args "model=${MODEL},base_url=${BASE}/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=16,timeout=5000,max_length=8192,max_gen_toks=256" \
  --tasks gsm8k --num_fewshot 5 --limit "$GSM8K_LIMIT" \
  --output_path "$RESULTS_DIR/gsm8k" --log_samples 2>&1 | tee "$RESULTS_DIR/gsm8k.log" || fail=1

echo "--- 3/3 mmlu_llama via /v1/chat/completions (also tests the K3 chat template)"
lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE}/v1/chat/completions,num_concurrent=16" \
  --apply_chat_template --gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --tasks mmlu_llama --limit "$MMLU_LIMIT" \
  --output_path "$RESULTS_DIR/mmlu" --log_samples 2>&1 | tee "$RESULTS_DIR/mmlu.log" || fail=1

echo "--- summary"
python3 - "$RESULTS_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
for f in sorted(glob.glob(os.path.join(root, '**', 'results_*.json'), recursive=True)):
    try:
        res = json.load(open(f)).get('results', {})
    except Exception as e:
        print('K3EVAL_ERR', f, e); continue
    for task, metrics in res.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k != 'alias':
                print(f'K3EVAL {task} {k}={v}')
PY
exit "$fail"
