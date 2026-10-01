#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An xprof capture of the Kimi-K3 1P1D pair under a benchmark, run on the
# prefill head by k3_1p1d_dp.sh with K3_DP_PROFILE=1.
#
#   k3_1p1d_dp_profile.sh <run_dir> <engine port> <proxy port> <decode host> <model> <model uri>
#
# One warm pass of the benchmark through the proxy, then the same pass with
# both engines' profilers running. Each engine's head host writes its eight
# ranks' xplanes under <run_dir>/profile; the decode head packs and serves its
# own (k3_1p1d_dp.sh), and this fetches it. Everything lands under
# ARTIFACTS_DIR/xprof/{prefill,decode}, and in GCS when the pod may write there.
#
# The capture is bounded by the benchmark, not by the profiler: a short output
# keeps the decode trace to a few dozen steps per rank.
set -uo pipefail

run_dir="${1:?run dir}"; PORT="${2:?engine port}"; PROXY_PORT="${3:?proxy port}"
DECODE_HOST="${4:?decode host}"; MODEL="${5:?model}"; MODEL_URI="${6:-}"
ISL="${K3_DP_PROFILE_ISL:-8192}"
OSL="${K3_DP_PROFILE_OSL:-32}"
NUM_PROMPTS="${K3_DP_PROFILE_PROMPTS:-32}"
CONCURRENCY="${K3_DP_PROFILE_CONCURRENCY:-32}"
GCS_BASE="${K3_DP_PROFILE_GCS_BASE:-gs://tpu-commons-ci/xprof/kimi-k3/torchtpu/1p1d-tp8-dp4-ep32}"
out="${ARTIFACTS_DIR:-$run_dir}/xprof"
mkdir -p "$out"

prefill="http://127.0.0.1:${PORT}"
decode="http://${DECODE_HOST}:${PORT}"

# The client tokenizes from the directory the server pulled the model into,
# as run_benchmarks.sh does, rather than resolving the model on the hub.
tokenizer_arg=()
if [[ -n "$MODEL_URI" ]]; then
  pulled=$(python3 -c 'import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)' "$MODEL_URI" 2>/dev/null | tail -1)
  [[ -n "$pulled" && -f "$pulled/config.json" ]] && tokenizer_arg=(--tokenizer "$pulled")
fi

bench() {
  local name="$1"
  vllm bench serve \
    --backend vllm --model "$MODEL" "${tokenizer_arg[@]}" --trust-remote-code \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0 --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
    --request-rate inf --ignore-eos --temperature 0 --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$out/${name}.json" 2>&1 | tee "$out/${name}.log"
  return "${PIPESTATUS[0]}"
}

profile() {
  local action="$1" url rc=0
  for url in "$prefill" "$decode"; do
    if ! curl -sf -X POST --max-time 120 "${url}/${action}" >/dev/null; then
      echo "ERROR: ${url}/${action} failed"
      rc=1
    fi
  done
  return "$rc"
}

echo "--- Warm pass: ${NUM_PROMPTS} x ${ISL}/${OSL} at concurrency ${CONCURRENCY}"
bench warm || { echo "ERROR: the warm pass failed"; exit 1; }

echo "--- Profiled pass"
profile start_profile || exit 1
bench profiled
bench_rc=$?
profile stop_profile || exit 1
[[ "$bench_rc" -eq 0 ]] || { echo "ERROR: the profiled pass failed"; exit 1; }

# Each rank writes its capture on stop and merges it under plugins/profile/<ts>/.
wait_xplanes() {
  local root="$1" what="$2" deadline=$((SECONDS + 600)) n
  until n="$(find "$root" -path '*/plugins/profile/*' -name '*.xplane.pb' 2>/dev/null | wc -l)" && [[ "$n" -ge 8 ]]; do
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "ERROR: ${what} has ${n} xplane files after 10 minutes, wanted 8"
      return 1
    fi
    sleep 10
  done
  echo "${what}: ${n} xplane files"
}
wait_xplanes "$run_dir/profile" "prefill capture" || exit 1
mkdir -p "$out/prefill" && cp -r "$run_dir/profile/plugins" "$out/prefill/"

echo "--- Fetching the decode capture"
deadline=$((SECONDS + 900))
until curl -sf -o "$out/decode.tgz" "http://${DECODE_HOST}:8200/profile.tgz"; do
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "ERROR: the decode head did not publish its capture within 15 minutes"
    exit 1
  fi
  sleep 20
done
mkdir -p "$out/decode" && tar -C "$out/decode" -xzf "$out/decode.tgz" && rm "$out/decode.tgz"
wait_xplanes "$out/decode" "decode capture" || exit 1

{
  echo "Kimi-K3 1P1D, prefill TP8 x DP4 x EP32 -> decode TP8 x DP4 x EP32, one 2x2x4 slice per role"
  echo "capture: head host of each role (slice-global ranks 0-7 = DP replica 0, TP8)"
  echo "workload: vllm bench serve through the toy proxy, random ${ISL}/${OSL}, ${NUM_PROMPTS} prompts, concurrency ${CONCURRENCY} (${CONCURRENCY}/4 per replica)"
  echo "build: ${BUILDKITE_BUILD_URL:-?} commit ${BUILDKITE_COMMIT:-?}"
  echo "kv dtype: bf16, prefix caching on, max-num-batched-tokens 8192, max-num-seqs 8"
} | tee "$out/README.txt"

if [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
  dest="${GCS_BASE}/$((ISL / 1024))k-${OSL}/${BUILDKITE_BUILD_NUMBER}"
  echo "--- Uploading to ${dest}"
  gcloud storage cp -r "$out"/* "$dest/" || echo "WARNING: GCS upload failed; the Buildkite artifacts still carry the traces"
fi
du -sh "$out"
