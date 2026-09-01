#!/usr/bin/env bash
# Batch U driver: decode sweep + c=1 decode profile against the live
# DP=4 x TP=8 endpoint running TPU_MOE_DP_A2A. Mirrors batchq2_measure.sh
# (isl=8192, osl=1024, 6 prompts/slot, exact prompt token ids) but with a
# shorter level list to fit the remaining slice time.
#
#   ENDPOINT=http://10.13.232.87:8000 OUT=results/batchU-a2a-ragged \
#     scripts/k3/batchu_measure.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
OUT="${OUT:?set OUT}"
LEVEL_TIMEOUT="${LEVEL_TIMEOUT:-1500}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"
LEVELS="${LEVELS:-1 4 16 32}"
mkdir -p "${OUT}/sweep"

echo "=== sweep c=${LEVELS// /,} (isl=8192 osl=1024, job-288 settings) ==="
for c in ${LEVELS}; do
  timeout "${LEVEL_TIMEOUT}" env \
    ARM=dp4tp8-a2a ENDPOINT="${ENDPOINT}" \
    OUTPUT_DIR="${OUT}/sweep" \
    CONCURRENCIES="${c}" INPUT_LEN=8192 OUTPUT_LEN=1024 \
    PROMPTS_PER_SLOT=6 PROMPT_TOKEN_IDS=1 \
    RANDOM_PREFIX_LEN=0 RANDOM_RANGE_RATIO=1 RESET_PREFIX_CACHE=0 \
    "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh" \
    > "${OUT}/perf_c${c}.log" 2>&1
  rc=$?
  if [[ ${rc} == 124 ]]; then
    echo "c=${c}: TIMEOUT"
  elif [[ ${rc} != 0 ]]; then
    echo "c=${c}: FAILED rc=${rc} (see ${OUT}/perf_c${c}.log)"
  else
    tail -2 "${OUT}/perf_c${c}.log" | head -1
  fi
done

echo ""
echo "=== decode profile capture (c=1, ${PROFILE_TOKENS} tokens) ==="
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${ENDPOINT}/start_profile")
echo "start_profile: ${code}"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":%s,"temperature":0,"ignore_eos":true}' \
  "${PROFILE_TOKENS}" > "${OUT}/profile-req.json"
t0=$(date +%s)
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/profile-req.json" \
  | python3 -c "import json,sys; print('profiled tokens:', json.load(sys.stdin)['usage']['completion_tokens'])" \
  || echo "profile request FAILED"
echo "profiled window: $(( $(date +%s) - t0 ))s"
code=$(curl -s -o /dev/null --max-time 900 -w '%{http_code}' -X POST "${ENDPOINT}/stop_profile")
echo "stop_profile: ${code}"

echo ""
echo "=== summary vs job 396 (dp4tp8 seqs=8 AG/RS control) ==="
PERF_DIR="${OUT}/sweep" python3 - <<'PYEOF'
import json, os
# job 396 (DP=4 x TP=8 seqs=8, AG/RS dispatch) and job 288 (TP=32) numbers
# from the Batch U brief; not re-measured.
j396 = {1: (31.49, 29.8), 4: (24.69, 147.3), 16: (33.09, 430.4),
        32: (42.10, 691.1)}
j288 = {1: (15.58, 61.1), 32: (51.25, 586.3)}
print(f"{'lvl':>4} {'TPOT ms':>9} {'tok/s':>9}   vs job396 (tpot / tok/s)")
for c in (1, 4, 16, 32):
    try:
        d = json.load(open(os.path.join(os.environ["PERF_DIR"], f"c{c}.json")))
        p, s_ = d["median_tpot_ms"], d["output_throughput"]
        p0, s0 = j396[c]
        print(f"c={c:<3} {p:8.2f} {s_:9.1f}   {100*(p-p0)/p0:+.1f}% / {100*(s_-s0)/s0:+.1f}%")
    except Exception as e:
        print(f"c={c}: MISSING ({e})")
PYEOF
echo "done -> ${OUT}"
