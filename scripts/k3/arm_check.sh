#!/usr/bin/env bash
# One arm of a hier-vs-flat comparison: correctness, then the concurrency
# sweep, then a profile capture at concurrency 1.
#
#   ENDPOINT=http://10.13.232.34:8000 ARM=hier OUT=results/k3-hier-v3 \
#     scripts/k3/arm_check.sh
#
# Both arms must run this same script: a hand-assembled sequence drifted
# between arms is not a comparison.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
ARM="${ARM:?set ARM (hier|flat)}"
OUT="${OUT:?set OUT}"
GSM8K_LIMIT="${GSM8K_LIMIT:-100}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"

mkdir -p "${OUT}"

echo "### [${ARM}] 1/4 canary"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":40,"temperature":0}' \
  > "${OUT}/canary-req.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/canary-req.json" > "${OUT}/canary.json" || true
python3 -c "
import json
try:
    t = json.load(open('${OUT}/canary.json'))['choices'][0]['text']
    print('canary:', repr(t[:160]))
except Exception as exc:
    print('canary UNREADABLE:', exc)
"

echo "### [${ARM}] 2/4 gsm8k (limit=${GSM8K_LIMIT})"
# Not offline-forced: the current datasets/lm-eval stack fails hub resolution
# of openai/gsm8k under HF_HUB_OFFLINE=1 even with the snapshot cached
# (ConnectionError in dataset_module_factory), while the hub IS reachable
# from the cluster. HF_HUB_CACHE still keeps the fetch local after the first
# run.
MODEL="moonshotai/Kimi-K3" BASE_URL="${ENDPOINT}/v1/completions" \
  OUTPUT_DIR="${OUT}/gsm8k-${GSM8K_LIMIT}" \
  HF_HUB_CACHE=/mnt/models \
  HF_DATASETS_CACHE="${REPO_ROOT}/.runtime-cache/hf-datasets" \
  LIMIT="${GSM8K_LIMIT}" \
  "${REPO_ROOT}/scripts/k3/eval_gsm8k.sh" > "${OUT}/gsm8k.log" 2>&1 \
  && grep -E "^\|gsm8k" "${OUT}/gsm8k.log" || echo "gsm8k FAILED (see ${OUT}/gsm8k.log)"

echo "### [${ARM}] 3/4 concurrency sweep"
ARM="${ARM}" ENDPOINT="${ENDPOINT}" OUTPUT_DIR="${OUT}/sweep" \
  "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh"

# Profiling last: it leaves the server's profiler session closed and writes
# gigabytes, so nothing downstream depends on it.
echo "### [${ARM}] 4/4 profile capture at concurrency 1"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${ENDPOINT}/start_profile")
echo "start_profile: ${code}"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":%s,"temperature":0,"ignore_eos":true}' \
  "${PROFILE_TOKENS}" > "${OUT}/profile-req.json"
t0=$(date +%s)
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/profile-req.json" \
  | python3 -c "import json,sys; print('profiled tokens:', json.load(sys.stdin)['usage']['completion_tokens'])" \
  || echo "profile request FAILED"
echo "profiled decode window: $(( $(date +%s) - t0 ))s (profiler overhead makes this far slower than the sweep)"
code=$(curl -s -o /dev/null --max-time 900 -w '%{http_code}' -X POST "${ENDPOINT}/stop_profile")
echo "stop_profile: ${code}"

echo "### [${ARM}] done -> ${OUT}"
