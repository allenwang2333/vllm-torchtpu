#!/usr/bin/env bash
# One arm of a prefill-side comparison: canary, then an exact-length prefill
# sweep (TTFT at 1k/4k/8k), then a profile capture of a single 8k prefill.
#
#   ENDPOINT=http://10.13.232.34:8000 ARM=kda-fused OUT=results/... \
#     scripts/k3/arm_check_prefill.sh
#
# GSM8K is deliberately absent: both KDA paths passed it at 100/100 in the
# decode rounds on this same code, and this arm exists to time prefill. The
# canary stays -- a benchmark with --ignore-eos never inspects a token.
# Both arms must run this same script: a hand-assembled sequence drifted
# between arms is not a comparison.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
ARM="${ARM:?set ARM (kda-fused|kda-unfused)}"
OUT="${OUT:?set OUT}"
INPUT_LENS="${INPUT_LENS:-1024 4096 8192}"
PROFILE_ISL="${PROFILE_ISL:-8192}"

mkdir -p "${OUT}"

echo "### [${ARM}] 1/3 prefill sweep (${INPUT_LENS})"
ARM="${ARM}" ENDPOINT="${ENDPOINT}" OUTPUT_DIR="${OUT}/prefill" \
  INPUT_LENS="${INPUT_LENS}" \
  "${REPO_ROOT}/scripts/k3/prefill_sweep.sh"

echo "### [${ARM}] 2/3 MLA tuned-params check (mixed case, from server log)"
# get_tuned_params logs a WARNING on every miss and stays silent on a hit, so
# the tuned Kimi-K3 8k key is proven by its ABSENCE here. Lookups happen at
# trace time, so every compile bucket appears exactly once per rank.
SERVER_LOG="${SERVER_LOG:-${OUT}-log/server.log}"
grep -ao "No tuned parameters found for the given tuning key: TuningKey(case='mixed'[^)]*)" \
  "${SERVER_LOG}" 2>/dev/null \
  | sed -E "s/.*(max_num_tokens=[0-9]+).*(actual_num_q_heads=[0-9]+).*/  miss: \1 \2/" \
  | sort | uniq -c || echo "  (no mixed-case misses logged)"

echo "### [${ARM}] 3/3 profile capture of one ${PROFILE_ISL}-token prefill"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${ENDPOINT}/start_profile")
echo "start_profile: ${code}"
python3 - "${PROFILE_ISL}" "${OUT}/profile-req.json" <<'EOF'
import json, sys
isl, path = int(sys.argv[1]), sys.argv[2]
# A filler token repeated isl times: content is irrelevant to a profiled
# prefill, but the length must be exact to hit the 8192 bucket.
json.dump({"model": "moonshotai/Kimi-K3", "prompt": "hello " * isl,
           "max_tokens": 1, "temperature": 0, "ignore_eos": True},
          open(path, "w"))
EOF
t0=$(date +%s)
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/profile-req.json" \
  | python3 -c "import json,sys; print('profiled prefill, tokens out:', json.load(sys.stdin)['usage']['completion_tokens'])" \
  || echo "profile request FAILED"
echo "profiled prefill window: $(( $(date +%s) - t0 ))s (profiler overhead inflates this)"
code=$(curl -s -o /dev/null --max-time 900 -w '%{http_code}' -X POST "${ENDPOINT}/stop_profile")
echo "stop_profile: ${code}"

echo "### [${ARM}] done -> ${OUT}"
