#!/usr/bin/env bash
# Prefill-latency sweep across exact input lengths against a live K3 endpoint.
#
#   ARM=kda-fused ENDPOINT=http://10.13.232.34:8000 OUTPUT_DIR=results/... \
#     scripts/k3/prefill_sweep.sh
#
# Same client and same exact-length mechanism as concurrency_sweep.sh
# (--prompt-token-ids), pointed at TTFT instead of TPOT: output length is 1,
# so everything the number contains is prefill. Both arms of a comparison must
# run this same script with the same INPUT_LENS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENDPOINT="${ENDPOINT:?set ENDPOINT, e.g. http://10.13.232.34:8000}"
ARM="${ARM:?set ARM, e.g. kda-fused}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/k3-prefill-${ARM}}"
INPUT_LENS="${INPUT_LENS:-1024 4096 8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
# c=1: one prompt in flight, so TTFT is pure prefill of exactly one request.
CONCURRENCY="${CONCURRENCY:-1}"
# Exact input length, for the same reason as concurrency_sweep.sh: the random
# dataset otherwise decodes its prompt to text and the server's re-tokenization
# drifts, which at 8192 vs a 8192 compile bucket is the difference between
# hitting the tuned MLA key and padding past it.
PROMPT_TOKEN_IDS="${PROMPT_TOKEN_IDS:-1}"
BENCH="${BENCH:-${REPO_ROOT}/vllm-bench}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
TOKENIZER="${TOKENIZER:-/mnt/models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42}"

mkdir -p "${OUTPUT_DIR}"

# A benchmark never inspects tokens; record what the model says next to the
# numbers so garbage output cannot masquerade as a fast arm.
printf '{"model":"%s","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":40,"temperature":0}' \
  "${MODEL}" > "${OUTPUT_DIR}/canary-request.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUTPUT_DIR}/canary-request.json" > "${OUTPUT_DIR}/canary.json" 2>&1 || true
echo "canary: $(python3 -c "
import json
try:
    print(repr(json.load(open('${OUTPUT_DIR}/canary.json'))['choices'][0]['text'][:120]))
except Exception as exc:
    print('unreadable:', exc)
" 2>&1 | tail -1)"

for isl in ${INPUT_LENS}; do
  out="${OUTPUT_DIR}/isl${isl}.json"
  echo "=== ${ARM}: isl=${isl} osl=${OUTPUT_LEN} prompts=${NUM_PROMPTS} c=${CONCURRENCY} ==="
  extra=()
  [ "${PROMPT_TOKEN_IDS}" = "1" ] && extra+=(--prompt-token-ids)
  "${BENCH}" \
    --backend openai \
    --random-range-ratio 1 \
    "${extra[@]}" \
    --base-url "${ENDPOINT}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --dataset-name random \
    --random-input-len "${isl}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --ignore-eos \
    --save-result \
    --result-dir "${OUTPUT_DIR}" \
    --result-filename "isl${isl}.json" \
    > "${OUTPUT_DIR}/isl${isl}.log" 2>&1 \
    || { echo "  FAILED (see ${OUTPUT_DIR}/isl${isl}.log)"; continue; }
  python3 -c "
import json
d = json.load(open('${out}'))
print('  median_ttft_ms=%.1f  mean_ttft_ms=%.1f  prefill_tok_s=%.1f  completed=%s' % (
    d['median_ttft_ms'], d['mean_ttft_ms'],
    ${isl} / (d['median_ttft_ms'] / 1000.0), d['completed']))
"
done

echo "prefill sweep complete -> ${OUTPUT_DIR}"
