#!/usr/bin/env bash
# Benchmark ISL=1 / OSL=1000 against an existing vLLM server using the
# benchmark client bundled with the repository's upstream vLLM installation.
#
# Pass the server IP as the only command-line argument. Override benchmark
# parameters through the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
if (( $# != 1 )); then
  echo "Usage: $0 SERVER_IP" >&2
  exit 2
fi
SERVER_IP="$1"
SERVER_PORT="${SERVER_PORT:-8000}"
BASE_URL="http://${SERVER_IP}:${SERVER_PORT}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/perf_isl1_osl1k}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
NUM_WARMUPS="${NUM_WARMUPS:-$((2 * MAX_CONCURRENCY))}"
SEED="${SEED:-42}"
VLLM="${VLLM:-$REPO_ROOT/.venv/bin/vllm}"

# Use the K3 model cache by default. Kimi-specific environment overrides keep
# the client standalone and deterministic.
export HF_HOME="${KIMI_K3_HF_HOME:-/mnt/scratch/hf-hub}"
export HF_HUB_CACHE="${KIMI_K3_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${KIMI_K3_HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

[[ -x "$VLLM" ]] || { echo "vLLM CLI not found: $VLLM" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
result_filename="kimi-k3_isl1_osl1000_concurrency${MAX_CONCURRENCY}_${timestamp}.json"

echo "Performance benchmark: model=${MODEL} ISL=1 OSL=1000 concurrency=${MAX_CONCURRENCY}"
echo "Endpoint: ${BASE_URL}"
echo "Result: ${OUTPUT_DIR}/${result_filename}"

# A one-token fixed prefix keeps the total ISL exactly one even when the
# tokenizer reports special tokens that vLLM subtracts from random-input-len.
"$VLLM" bench serve \
  --backend vllm \
  --model "$MODEL" \
  --trust-remote-code \
  --base-url "$BASE_URL" \
  --dataset-name random \
  --random-input-len 0 \
  --random-prefix-len 1 \
  --random-output-len 1000 \
  --random-range-ratio 0 \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups "$NUM_WARMUPS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate inf \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --save-detailed \
  --result-dir "$OUTPUT_DIR" \
  --result-filename "$result_filename" \
  --seed "$SEED"
