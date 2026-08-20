#!/usr/bin/env bash
# One arm of the barrier-free A/B: canary, decode speed at concurrency 1, then a
# profile capture. Invoked by the launcher's POST_READY_CMD so both arms run a
# byte-identical sequence -- the point arm_check.sh makes, that a hand-assembled
# sequence drifted between arms is not a comparison.
#
#   ENDPOINT and OUT are exported by the launcher.
#   ARM names the arm in the output paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
OUT="${OUT:?set OUT}"
ARM="${ARM:-arm}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"
# Concurrency 1 only: this A/B is about per-step decode latency at batch size 1,
# which is the regime the kernels are gated for.
CONCURRENCIES="${CONCURRENCIES:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

mkdir -p "${OUT}"
echo "### [${ARM}] endpoint ${ENDPOINT}"

echo "### [${ARM}] 1/3 canary"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":40,"temperature":0}' \
  > "${OUT}/canary-req.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/canary-req.json" > "${OUT}/canary.json" || true
python3 -c "
import json
try:
    print('canary:', repr(json.load(open('${OUT}/canary.json'))['choices'][0]['text'][:160]))
except Exception as exc:
    print('canary UNREADABLE:', exc)
"

# Decode speed. Repeated so the reported TPOT carries a spread rather than a
# single sample -- the job-154 baseline was stable to +/-0.01 ms over three runs,
# and the effect under test is ~0.06-0.28 ms, so run-to-run noise has to be
# visible or the comparison cannot be read.
echo "### [${ARM}] 2/3 decode speed, ${CONCURRENCIES} x3"
for rep in 1 2 3; do
  echo "--- rep ${rep}"
  ARM="${ARM}-r${rep}" ENDPOINT="${ENDPOINT}" \
    OUTPUT_DIR="${OUT}/sweep-r${rep}" \
    CONCURRENCIES="${CONCURRENCIES}" OUTPUT_LEN="${OUTPUT_LEN}" \
    "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh" 2>&1 | tail -12
done

# Profiling last: it leaves the profiler session closed and writes gigabytes, so
# nothing downstream depends on it.
echo "### [${ARM}] 3/3 profile capture at concurrency 1"
echo "start_profile: $(curl -s -o /dev/null -w '%{http_code}' -X POST "${ENDPOINT}/start_profile")"
printf '{"model":"moonshotai/Kimi-K3","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":%s,"temperature":0,"ignore_eos":true}' \
  "${PROFILE_TOKENS}" > "${OUT}/profile-req.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/profile-req.json" \
  | python3 -c "import json,sys; print('profiled tokens:', json.load(sys.stdin)['usage']['completion_tokens'])" \
  || echo "profile request FAILED"
echo "stop_profile: $(curl -s -o /dev/null --max-time 900 -w '%{http_code}' -X POST "${ENDPOINT}/stop_profile")"

echo "### [${ARM}] done -> ${OUT}"
