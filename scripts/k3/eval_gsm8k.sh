#!/usr/bin/env bash
# gsm8k (5-shot) accuracy eval against a vLLM OpenAI-compatible endpoint
# using lm-eval.
#
# Override any parameter below through the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
COMPLETIONS_URL="${BASE_URL%/}/v1/completions"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-256}"
TIMEOUT="${TIMEOUT:-5000}"
LM_EVAL="${LM_EVAL:-$REPO_ROOT/.venv/bin/lm_eval}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

mkdir -p "$OUTPUT_DIR"

echo "gsm8k ${NUM_FEWSHOT}-shot: model=${MODEL} base_url=${COMPLETIONS_URL} -> ${OUTPUT_DIR}"

"$LM_EVAL" --model local-completions \
  --model_args "model=${MODEL},base_url=${COMPLETIONS_URL},tokenized_requests=False,tokenizer_backend=None,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT},max_length=${MAX_LENGTH},max_gen_toks=${MAX_GEN_TOKS}" \
  --tasks gsm8k \
  --num_fewshot "$NUM_FEWSHOT" \
  --output_path "$OUTPUT_DIR" \
  --log_samples
