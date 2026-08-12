#!/usr/bin/env bash
# Send one exact ISL=1 / OSL=1000 request to an existing vLLM server.
# A token-id prompt avoids tokenizer-dependent string lengths.
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
PROMPT_TOKEN_ID="${PROMPT_TOKEN_ID:-1000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1000}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/curl_isl1_osl1k}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/response_$(date +%Y%m%d-%H%M%S).json}"

for value_name in PROMPT_TOKEN_ID OUTPUT_TOKENS SEED; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done

command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_FILE")"

payload="$(jq --compact-output --null-input \
  --arg model "$MODEL" \
  --argjson prompt_token_id "$PROMPT_TOKEN_ID" \
  --argjson output_tokens "$OUTPUT_TOKENS" \
  --argjson seed "$SEED" \
  '{
    model: $model,
    prompt: [$prompt_token_id],
    max_tokens: $output_tokens,
    min_tokens: $output_tokens,
    ignore_eos: true,
    temperature: 0,
    seed: $seed
  }')"

echo "Request: ISL=1 OSL=${OUTPUT_TOKENS} endpoint=${BASE_URL%/}/v1/completions"
echo "Response: $OUTPUT_FILE"

curl --fail-with-body --silent --show-error \
  --header 'Content-Type: application/json' \
  --request POST \
  --data "$payload" \
  --output "$OUTPUT_FILE" \
  "${BASE_URL%/}/v1/completions"

if ! jq --exit-status \
    --argjson output_tokens "$OUTPUT_TOKENS" \
    '.usage.prompt_tokens == 1 and
     .usage.completion_tokens == $output_tokens' \
    "$OUTPUT_FILE" >/dev/null; then
  echo "Response did not contain exactly 1 input and $OUTPUT_TOKENS output tokens" >&2
  jq '{error, usage}' "$OUTPUT_FILE" >&2 || true
  exit 1
fi

jq --compact-output '{id, usage}' "$OUTPUT_FILE"
