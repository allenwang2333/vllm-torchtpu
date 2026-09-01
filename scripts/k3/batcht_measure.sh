#!/usr/bin/env bash
# Batch T (lead D) driver: high-concurrency sweep + decode profile against a
# live DP=4 x TP=8 endpoint running MAX_NUM_SEQS=32 per rank
# (COMPILE_SIZES=1,2,4,8,16,32,2048). Runs on the login node while
# slurm_launch_k3_dp4tp8.sbatch holds the allocation.
#
#   ENDPOINT=http://10.13.232.87:8000 OUT=results/batchT-seqs32 \
#     scripts/k3/batcht_measure.sh
#
# Sweep settings mirror job 288/396 (isl=8192, osl=1024, 6 prompts/slot,
# exact prompt token ids); c=32 repeats as the continuity point against job
# 396's 691.1 tok/s, c=64/128 are the new regime. The profile mirrors the
# llar-control capture: one c=1 request, 128 tokens, profiler around it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
OUT="${OUT:?set OUT}"
LEVEL_TIMEOUT="${LEVEL_TIMEOUT:-1800}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"
LEVELS="${LEVELS:-32 64 128}"
mkdir -p "${OUT}/sweep"

echo "=== sweep c=${LEVELS// /,} (isl=8192 osl=1024, job-288 settings) ==="
for c in ${LEVELS}; do
  timeout "${LEVEL_TIMEOUT}" env \
    ARM=dp4tp8-seqs32 ENDPOINT="${ENDPOINT}" \
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
if [[ "${SKIP_PROFILE:-0}" == "1" ]]; then
  echo "=== decode profile capture skipped (SKIP_PROFILE=1) ==="
else
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
fi

echo ""
echo "=== summary ==="
PERF_DIR="${OUT}/sweep" python3 - <<'PYEOF'
import json, os
base = {32: (3185.2, 51.25, 586.3)}
print(f"{'lvl':>4} {'TTFT ms':>10} {'TPOT ms':>10} {'tok/s':>10}")
for c in (32, 64, 128):
    try:
        d = json.load(open(os.path.join(os.environ["PERF_DIR"], f"c{c}.json")))
        t, p, s_ = d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"]
        extra = ""
        if c in base:
            t0, p0, s0 = base[c]
            extra = (f"  vs TP=32 job288: ttft {100*(t-t0)/t0:+.1f}% "
                     f"tpot {100*(p-p0)/p0:+.1f}% tok/s {100*(s_-s0)/s0:+.1f}%")
        print(f"c={c:<3} {t:9.1f} {p:9.2f} {s_:9.1f}{extra}")
    except Exception as e:
        print(f"c={c}: MISSING ({e})")
PYEOF

echo "done -> ${OUT}"
