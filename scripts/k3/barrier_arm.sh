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
if ! python3 -c "
import json
print('canary:', repr(json.load(open('${OUT}/canary.json'))['choices'][0]['text'][:160]))
"; then
  # Abort rather than continue. A dead endpoint makes every number below
  # meaningless, and this script's exit status is what the launcher reports --
  # job 297 printed '### done' and 'POST_READY_CMD exited 0' having measured
  # nothing at all, because the canary failure was swallowed and the sweep's
  # own 'completed=0 failed=5' was never inspected.
  echo "### [${ARM}] FATAL: canary did not return a completion; endpoint is not serving." >&2
  exit 4
fi

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
  # The sweep tool exits 0 with "completed":0 when the endpoint refuses the
  # connection, so its exit status proves nothing. Read the result instead.
  # Glob c<digits>.json, not *.json: the sweep directory also holds
  # canary.json and canary-request.json, which have no "completed" key, so a
  # bare *.json glob reads 0 out of them and fails a passing run. That is
  # exactly what it did to job 302, which had completed=5 failed=0
  # tpot_med=14.6425 and was aborted anyway.
  shopt -s nullglob
  for res in "${OUT}/sweep-r${rep}"/c[0-9]*.json; do
    python3 -c "
import json, sys
d = json.load(open('${res}'))
if 'completed' not in d:
    print(f'  ${res##*/}: no completed key, skipping')
    sys.exit(0)
done_, failed = d['completed'], d.get('failed', 0)
print(f'  ${res##*/}: completed={done_} failed={failed} '
      f'tpot_med={d.get(\"median_tpot_ms\")}')
sys.exit(0 if done_ > 0 and failed == 0 else 5)
" || {
      echo "### [${ARM}] FATAL: sweep rep ${rep} completed no requests." >&2
      exit 5
    }
  done
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
