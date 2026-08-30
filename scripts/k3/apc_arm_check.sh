#!/usr/bin/env bash
# One arm of the pooled-prefix-caching comparison on Kimi-K3 at TP=32.
#
#   ENDPOINT=$(cat results/<arm>-log/endpoint.txt) \
#   ARM=<arm> OUT=results/<arm> APC=0|1 scripts/k3/apc_arm_check.sh
#
# Every arm runs this same script; APC only gates the smokes that need a cache
# to exist. A hand-assembled sequence that drifts between arms is not a
# comparison.
#
# Order is chosen to fail fast and to keep the expensive server up for the
# cheap checks: correctness before speed, and profiling last because it leaves
# the profiler session closed and writes gigabytes.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ENDPOINT:?set ENDPOINT}"
ARM="${ARM:?set ARM}"
OUT="${OUT:?set OUT}"
APC="${APC:-0}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
GSM8K_LIMIT="${GSM8K_LIMIT:-100}"
# The workload under test: 1k in, 1k out, one request at a time.
ISL="${ISL:-1000}"
OSL="${OSL:-1000}"
# Tokens the shared-prefix sweep gives every prompt in common. 800 of 1000
# leaves a 200-token tail, so a hit saves most of the prefill without making
# the request trivial.
SHARED_PREFIX="${SHARED_PREFIX:-800}"
# The pool block size the probe aims its boundary prompts at. K3 at TP=32
# derives 162 tokens (77 padded SSM rows + 4 conv rows, 2 tokens per packed
# MLA row); the no-pool arms run block_size 16 and simply see no boundary
# effects, which is the point of running the identical probe in both.
BLOCK_SIZE="${BLOCK_SIZE:-162}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"

mkdir -p "${OUT}"
echo "### [${ARM}] APC=${APC} isl=${ISL} osl=${OSL} -> ${OUT}"

echo
echo "### [${ARM}] 1/7 canary"
printf '{"model":"%s","prompt":"Question: What is 7 plus 5?\nAnswer:","max_tokens":40,"temperature":0}' \
  "${MODEL}" > "${OUT}/canary-req.json"
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/canary-req.json" > "${OUT}/canary.json" || true
python3 -c "
import json
try:
    print('canary:', repr(json.load(open('${OUT}/canary.json'))['choices'][0]['text'][:160]))
except Exception as exc:
    print('canary UNREADABLE:', exc)
"

echo
echo "### [${ARM}] 2/7 gsm8k (limit=${GSM8K_LIMIT})"
MODEL="${MODEL}" BASE_URL="${ENDPOINT}/v1/completions" \
  OUTPUT_DIR="${OUT}/gsm8k-${GSM8K_LIMIT}" \
  HF_HUB_CACHE=/mnt/lustre/hf-models/hub \
  HF_DATASETS_CACHE="${REPO_ROOT}/.runtime-cache/hf-datasets" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 LIMIT="${GSM8K_LIMIT}" \
  "${REPO_ROOT}/scripts/k3/eval_gsm8k.sh" > "${OUT}/gsm8k.log" 2>&1 \
  && grep -E "^\|gsm8k" "${OUT}/gsm8k.log" || echo "gsm8k FAILED (see ${OUT}/gsm8k.log)"

# Before the benchmarks: a corrupted state restore is a correctness bug, and a
# benchmark run against a broken server is wasted server time.
echo
echo "### [${ARM}] 3/7 fact-retrieval prefix probe"
# The correctness gate. Scored on whether the answer is retrievable from the
# cached document, which noise cannot move -- unlike byte equality, which on
# this server is not a test: two identical runs with prefix caching switched
# OFF entirely agreed on 11 of 24 completions and on 24 of 24 answers.
ENDPOINT="${ENDPOINT}" MODEL="${MODEL}" \
  python3 "${REPO_ROOT}/scripts/k3/apc_text_probe.py" \
    --out "${OUT}/text-probe.json" 2>&1 | tee "${OUT}/text-probe.log" | tail -12

echo
echo "### [${ARM}] 3b/7 token-id boundary probe (sensitivity, not pass/fail)"
# Random-ID prompts on block boundaries: maximally sensitive to a corrupted
# state, and maximally sensitive to noise too. Reported, never gated on.
ENDPOINT="${ENDPOINT}" MODEL="${MODEL}" \
  python3 "${REPO_ROOT}/scripts/k3/apc_probe.py" \
    --out "${OUT}/apc-probe.json" --block-size "${BLOCK_SIZE}" \
    2>&1 | tee "${OUT}/apc-probe.log" | tail -12 || true

echo
echo "### [${ARM}] 4/7 benchmark c=1, independent prompts"
# No shared prefix: the prefix cache cannot hit, so this is the honest
# measurement of what the pooled path itself costs.
ARM="${ARM}" ENDPOINT="${ENDPOINT}" OUTPUT_DIR="${OUT}/bench-nocache" \
  CONCURRENCIES=1 INPUT_LEN="${ISL}" OUTPUT_LEN="${OSL}" \
  PROMPT_TOKEN_IDS=1 RANDOM_PREFIX_LEN=0 PROMPTS_PER_SLOT=10 \
  RESET_PREFIX_CACHE=1 MODEL="${MODEL}" \
  "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh"

echo
echo "### [${ARM}] 5/7 benchmark c=1, ${ISL} tokens of which ${SHARED_PREFIX} shared"
# The workload prefix caching exists for. Runs in every arm so the no-cache
# arms provide the baseline the saving is measured against.
# vllm-bench ADDS --random-prefix-len on top of --random-input-len, so the
# unique tail is what goes in INPUT_LEN. Both benchmarks then send exactly ISL
# tokens and differ only in how much of it is shared.
ARM="${ARM}" ENDPOINT="${ENDPOINT}" OUTPUT_DIR="${OUT}/bench-sharedprefix" \
  CONCURRENCIES=1 INPUT_LEN="$(( ISL - SHARED_PREFIX ))" OUTPUT_LEN="${OSL}" \
  PROMPT_TOKEN_IDS=1 RANDOM_PREFIX_LEN="${SHARED_PREFIX}" PROMPTS_PER_SLOT=10 \
  RESET_PREFIX_CACHE=1 MODEL="${MODEL}" \
  "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh"

echo
echo "### [${ARM}] 6/7 legacy decode reference (isl=1, osl=1024)"
# The historical concurrency-sweep configuration, one level of it. It exists so
# this arm can be placed against the recorded main-branch arms (k3-llar-control
# and friends, median TPOT ~16.8 ms at c=1) without spending a second
# allocation on a main-branch control -- the cluster is shared and these runs
# are four hosts each. It is a weaker comparison than a same-session A/B and is
# reported as such: different day, different job, same nodes and same pinned
# flags.
ARM="${ARM}" ENDPOINT="${ENDPOINT}" OUTPUT_DIR="${OUT}/bench-legacy" \
  CONCURRENCIES=1 INPUT_LEN=1 OUTPUT_LEN=1024 PROMPTS_PER_SLOT=4 \
  RESET_PREFIX_CACHE=1 MODEL="${MODEL}" \
  "${REPO_ROOT}/scripts/k3/concurrency_sweep.sh"

if [ "${APC}" = "1" ]; then
  echo
  echo "### [${ARM}] 6b/7 prefix-cache smokes"
  # The contamination smoke is the decisive one: it interleaves two prefixes so
  # a resumed request's slot last held the other, which is exactly what a cache
  # hit restoring attention KV without the KDA recurrent state gets wrong.
  python3 "${REPO_ROOT}/scripts/vllm/integration/smoke_mamba_state_contamination.py" \
    --host "$(echo "${ENDPOINT}" | sed -E 's#https?://##; s#:.*##')" \
    --port "$(echo "${ENDPOINT}" | sed -E 's#.*:##')" \
    --model "${MODEL}" --lines 200 > "${OUT}/smoke-contamination.log" 2>&1
  echo "contamination smoke: exit=$? (see ${OUT}/smoke-contamination.log)"
  tail -5 "${OUT}/smoke-contamination.log"

  python3 "${REPO_ROOT}/scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py" \
    --host "$(echo "${ENDPOINT}" | sed -E 's#https?://##; s#:.*##')" \
    --port "$(echo "${ENDPOINT}" | sed -E 's#.*:##')" \
    --model "${MODEL}" > "${OUT}/smoke-divergence.log" 2>&1
  echo "divergence smoke: exit=$? (see ${OUT}/smoke-divergence.log)"
  tail -5 "${OUT}/smoke-divergence.log"

  # A run that never hit the cache would pass both smokes vacuously.
  curl -sf "${ENDPOINT}/metrics" | awk '
    /^vllm:prefix_cache_queries_total/ {q += $2}
    /^vllm:prefix_cache_hits_total/    {h += $2}
    END {printf "prefix_cache_totals: hits=%d queries=%d\n", h, q}'
fi

echo
echo "### [${ARM}] 7/7 profile capture at concurrency 1"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${ENDPOINT}/start_profile")
echo "start_profile: ${code}"
# Profile the workload under test, not a toy prompt: prefill cost is what the
# 1k-token input changes, and the phased profiler splits the capture into
# prefill_only and decode_only for exactly this reason. Output is capped well
# below OSL because the profiler inflates step time ~50x and decode steps are
# homogeneous.
python3 -c "
import json, random
# A seed of its own. apc_probe.py draws its token stream from Random(20260819),
# so sharing it would make the profiled prompt a prefix of prompts the probe
# had already cached -- and with no /reset_prefix_cache in this build, the
# capture would silently profile a partial prefill and report it as a full one.
rng = random.Random(778899)
ids = [rng.randrange(1000, 20000) for _ in range(${ISL})]
json.dump({'model': '${MODEL}', 'prompt': ids, 'max_tokens': ${PROFILE_TOKENS},
           'temperature': 0, 'ignore_eos': True},
          open('${OUT}/profile-req.json', 'w'))
"
if [ "${APC}" = "1" ]; then
  # Profile a cold prefill: a cache hit would skip the very prefill work the
  # capture exists to show.
  curl -s -o /dev/null -X POST "${ENDPOINT}/reset_prefix_cache"
fi
t0=$(date +%s)
curl -sf "${ENDPOINT}/v1/completions" -H 'Content-Type: application/json' \
  --data @"${OUT}/profile-req.json" \
  | python3 -c "import json,sys; u=json.load(sys.stdin)['usage']; print('profiled: prompt=%s completion=%s' % (u['prompt_tokens'], u['completion_tokens']))" \
  || echo "profile request FAILED"
echo "profiled window: $(( $(date +%s) - t0 ))s (profiler overhead makes this far slower than the benchmark)"
code=$(curl -s -o /dev/null --max-time 1200 -w '%{http_code}' -X POST "${ENDPOINT}/stop_profile")
echo "stop_profile: ${code}"

echo
echo "### [${ARM}] done -> ${OUT}"
