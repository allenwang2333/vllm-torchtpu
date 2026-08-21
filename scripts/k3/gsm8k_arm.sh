#!/usr/bin/env bash
# Correctness arm: canary, then gsm8k 5-shot, then report the score.
#
# Deliberately separate from barrier_arm.sh. That arm measures decode latency
# and so runs at COMPILE_SIZES=1 / MAX_NUM_BATCHED_TOKENS=1, where prefill
# advances one token per step -- a 900-token 5-shot gsm8k prompt costs ~12.6 s
# of prefill alone at 14 ms/step, serialized at max_num_seqs=1. Scoring on that
# config would take hours and tell us nothing extra.
#
# Run this against a normal serving config instead. That is also the stronger
# test for the MoE renormalize fold: the fold is gated at
# TPU_MOE_RENORMALIZE_FOLD_MAX_TOKENS (64 routed rows), so a prefill bucket
# above it exercises the plain .sum() path while decode exercises the fold, and
# both are covered in one run. The decode-only A/B never touches the boundary.
#
#   ENDPOINT and OUT are exported by the launcher; ARM names the output paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
OUT="${OUT:?set OUT}"
ARM="${ARM:-gsm8k}"
# 100 matches the existing reference runs (results/k3-concat{2,3}/gsm8k-100),
# both of which scored exact_match 1.000. Saturated, so this detects breakage
# but cannot measure small regressions -- do not quote it as a gsm8k score.
GSM8K_LIMIT="${GSM8K_LIMIT:-100}"

mkdir -p "${OUT}"
echo "### [${ARM}] endpoint ${ENDPOINT}"

echo "### [${ARM}] 1/2 canary"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":40,"temperature":0}' \
  > "${OUT}/canary-req.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/canary-req.json" > "${OUT}/canary.json" || true
if ! python3 -c "
import json
print('canary:', repr(json.load(open('${OUT}/canary.json'))['choices'][0]['text'][:160]))
"; then
  echo "### [${ARM}] FATAL: canary did not return a completion." >&2
  exit 4
fi

echo "### [${ARM}] 2/2 gsm8k 5-shot, limit=${GSM8K_LIMIT}"
MODEL="${MODEL:-moonshotai/Kimi-K3}" \
  BASE_URL="${ENDPOINT}" \
  OUTPUT_DIR="${OUT}/gsm8k-${GSM8K_LIMIT}" \
  LIMIT="${GSM8K_LIMIT}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.runtime-cache/hf-datasets}" \
  "${REPO_ROOT}/scripts/k3/eval_gsm8k.sh" 2>&1 | tail -25
rc=${PIPESTATUS[0]}

# lm_eval can exit 0 having scored nothing, so read the artifact rather than
# trusting the status -- the same failure barrier_arm.sh's sweep guard exists
# for.
shopt -s nullglob
found=0
for res in "${OUT}/gsm8k-${GSM8K_LIMIT}"/*/results_*.json; do
  found=1
  python3 -c "
import json
d = json.load(open('${res}'))
r = d.get('results', {}).get('gsm8k', {})
n = r.get('sample_len')
strict = r.get('exact_match,strict-match')
flex = r.get('exact_match,flexible-extract')
print(f'  n={n} strict={strict} flexible={flex}')
print('  reference (results/k3-concat{2,3}/gsm8k-100): strict=1.0 flexible=1.0')
"
done
if (( found == 0 )); then
  echo "### [${ARM}] FATAL: gsm8k produced no results file (lm_eval rc=${rc})." >&2
  exit 6
fi
echo "### [${ARM}] done -> ${OUT}"
