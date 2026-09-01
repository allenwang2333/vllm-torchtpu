#!/usr/bin/env bash
# Batch Q2 stage 3 driver: sweep + decode profile against a live DP=4 x TP=8
# endpoint. Runs on the login node while slurm_launch_k3_dp4tp8.sbatch holds
# the allocation.
#
#   ENDPOINT=http://10.13.232.87:8000 OUT=results/batchQ2-dp4tp8 \
#     scripts/k3/batchq2_measure.sh
#
# Sweep settings mirror job 288 exactly (isl=8192, osl=1024, 6 prompts/slot,
# exact prompt token ids) so the c=1..32 table compares against the TP=32
# baseline without a control run. The profile mirrors the TP=32 control
# capture (llar-control): one c=1 request, 128 tokens, profiler around it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
OUT="${OUT:?set OUT}"
LEVEL_TIMEOUT="${LEVEL_TIMEOUT:-1800}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"
mkdir -p "${OUT}/sweep"

echo "=== sweep c=1,2,4,8,16,32 (isl=8192 osl=1024, job-288 settings) ==="
for c in 1 2 4 8 16 32; do
  timeout "${LEVEL_TIMEOUT}" env \
    ARM=dp4tp8 ENDPOINT="${ENDPOINT}" \
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
echo "=== comparison vs job 288 (TP=32 baseline) ==="
PERF_DIR="${OUT}/sweep" python3 - <<'PYEOF'
import json, os
base = {1: (806.0, 15.58, 61.1), 2: (904.6, 16.68, 113.8),
        4: (2389.1, 17.24, 204.6), 8: (3166.3, 22.52, 313.5),
        16: (3173.9, 31.70, 454.1), 32: (3185.2, 51.25, 586.3)}
print(f"{'lvl':>4} {'TTFT ms (delta%)':>22} {'TPOT ms (delta%)':>22} {'tok/s (delta%)':>20}")
for c, (ttft0, tpot0, tps0) in base.items():
    try:
        d = json.load(open(os.path.join(os.environ["PERF_DIR"], f"c{c}.json")))
        t, p, s_ = d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"]
        print(f"c={c:<3} {t:9.1f} ({100*(t-ttft0)/ttft0:+5.1f}%) "
              f"{p:9.2f} ({100*(p-tpot0)/tpot0:+5.1f}%) "
              f"{s_:9.1f} ({100*(s_-tps0)/tps0:+5.1f}%)")
    except Exception as e:
        print(f"c={c}: MISSING ({e})")
PYEOF

echo "done -> ${OUT}"
