#!/usr/bin/env bash
# Decode-speed sweep across concurrency levels against a live K3 endpoint.
#
#   ARM=hier ENDPOINT=http://10.13.232.34:8000 OUTPUT_DIR=results/... \
#     scripts/k3/concurrency_sweep.sh
#
# Uses ./vllm-bench (the standalone Rust client) rather than `vllm bench serve`:
# the latter imports torch and dies on the login node under vLLM 0.27 with
# `RuntimeError: ... device type at start of device string: tpu`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENDPOINT="${ENDPOINT:?set ENDPOINT, e.g. http://10.13.232.34:8000}"
ARM="${ARM:?set ARM, e.g. hier or flat}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/k3-sweep-${ARM}}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
INPUT_LEN="${INPUT_LEN:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
# Prompts per level scale with concurrency so every level runs a comparable
# number of full decode rounds (and comparable wall time), instead of level 16
# finishing in a fraction of level 1's.
PROMPTS_PER_SLOT="${PROMPTS_PER_SLOT:-4}"
# Exact prompt lengths. The random dataset builds prompts with whatever
# tokenizer it was handed and then DECODES them to text, so the server
# re-tokenizes and the realised input length drifts from --random-input-len.
# --prompt-token-ids sends the ID array instead, which makes isl exact -- and
# exactness is the whole point once the measurement is about prefill or about
# how many tokens a prefix cache can match.
PROMPT_TOKEN_IDS="${PROMPT_TOKEN_IDS:-0}"
# Tokens every prompt in the run shares. 0 (the default) gives independent
# prompts, which is the honest measurement of pooled-path overhead because the
# prefix cache cannot hit. A positive value is the workload prefix caching
# exists for.
RANDOM_PREFIX_LEN="${RANDOM_PREFIX_LEN:-0}"
# 1 means every prompt is exactly --random-input-len long.
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"
# Off by default so a run whose prompts share a prefix is not silently served
# from a cache warmed by an earlier level of the same sweep.
RESET_PREFIX_CACHE="${RESET_PREFIX_CACHE:-0}"
BENCH="${BENCH:-${REPO_ROOT}/vllm-bench}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
# K3 ships tiktoken.model with a custom Python tokenizer class and no
# tokenizer.json, which the Rust tokenizers crate cannot load. The random
# dataset needs *a* tokenizer only to build the prompt; at INPUT_LEN=1 the
# choice cannot affect decode speed, so borrow a small local one.
TOKENIZER="${TOKENIZER:-/mnt/models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42}"

mkdir -p "${OUTPUT_DIR}"

# A benchmark with --ignore-eos never inspects a token, so it reports a healthy
# tok/s for a model emitting garbage. Record what the model actually says next
# to the numbers, so nobody reads this sweep as evidence of correctness.
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

for c in ${CONCURRENCIES}; do
  n=$(( c * PROMPTS_PER_SLOT ))
  [ "${n}" -lt 5 ] && n=5
  out="${OUTPUT_DIR}/c${c}.json"
  echo "=== ${ARM}: concurrency=${c} prompts=${n} isl=${INPUT_LEN} osl=${OUTPUT_LEN} shared_prefix=${RANDOM_PREFIX_LEN} ==="
  if [ "${RESET_PREFIX_CACHE}" = "1" ]; then
    echo "  reset_prefix_cache: $(curl -s -o /dev/null -w '%{http_code}' \
      -X POST "${ENDPOINT}/reset_prefix_cache")"
  fi
  # vllm:prefix_cache_{hits,queries}_total are counters since server boot, so
  # printing them raw reports the server's lifetime hit rate and attributes it
  # to this benchmark. Snapshot and difference them instead: the first version
  # of this claimed 20% hits for a run that turned out to hit 0 of 10,012.
  _pc_before=$(curl -sf "${ENDPOINT}/metrics" 2>/dev/null | awk '
    /^vllm:prefix_cache_queries_total/ {q += $2}
    /^vllm:prefix_cache_hits_total/    {h += $2}
    END {printf "%d %d", h, q}')
  extra=()
  [ "${PROMPT_TOKEN_IDS}" = "1" ] && extra+=(--prompt-token-ids)
  [ "${RANDOM_PREFIX_LEN}" -gt 0 ] && extra+=(--random-prefix-len "${RANDOM_PREFIX_LEN}")
  "${BENCH}" \
    --backend openai \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    "${extra[@]}" \
    --base-url "${ENDPOINT}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${n}" \
    --max-concurrency "${c}" \
    --ignore-eos \
    --save-result \
    --result-dir "${OUTPUT_DIR}" \
    --result-filename "c${c}.json" \
    > "${OUTPUT_DIR}/c${c}.log" 2>&1 \
    || { echo "  FAILED (see ${OUTPUT_DIR}/c${c}.log)"; continue; }
  python3 -c "
import json
d = json.load(open('${out}'))
print('  median_tpot_ms=%.2f  output_tok_s=%.1f  median_ttft_ms=%.1f  completed=%s' % (
    d['median_tpot_ms'], d['output_throughput'], d['median_ttft_ms'], d['completed']))
"
  curl -sf "${ENDPOINT}/metrics" 2>/dev/null | awk -v before="${_pc_before}" '
    /^vllm:prefix_cache_queries_total/ {q += $2}
    /^vllm:prefix_cache_hits_total/    {h += $2}
    END {split(before, b, " ");
         dh = h - b[1]; dq = q - b[2];
         if (dq > 0) printf "  prefix_cache: %d/%d blocks this run (%.1f%%)\n", dh, dq, 100*dh/dq;
         else printf "  prefix_cache: no queries this run\n"}'

done

echo "sweep complete -> ${OUTPUT_DIR}"
