#!/bin/bash
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

# Compiled-graph cache carryover test for the gsm8k accuracy collapse seen at
# eval concurrency 32 (but not 128) on the per-layer KV path.
#
# The arm matrix showed ONEHOT_MOE_PERMUTE_THRESHOLD=32768 is the load-bearing
# knob: repro_ctrl scored 0.00 at conc 32 while every other arm scored ~0.85.
# But an earlier two-phase run of this script saw its no-env phase B come back
# garbage too, which that result cannot explain. Suspicion: the threshold is
# baked into the pallas kernel closure but is likely NOT part of the compile
# cache key, so phase B silently reused phase A's poisoned compiled graphs —
# the default vLLM cache dir lives inside the container and survives across
# server lifecycles within one job.
#
# Three sequential server lifecycles in ONE job, same cache dir throughout
# (deliberately NOT cleared between them — reuse is the thing under test):
#   1. noenv_first  no MoE envs      -> expect clean  (~0.85)
#   2. onehot_mid   threshold=32768  -> expect garbage (~0.00)
#   3. noenv_again  no MoE envs      -> the verdict
#
# If noenv_again is garbage, it was poisoned by lifecycle 2's cache and the
# carryover theory holds. If noenv_again is clean, the carryover theory dies
# and the earlier phase-B garbage needs a different explanation.
#
# Runs inside the CI docker image via .buildkite/scripts/run_in_docker.sh.
set -uo pipefail

MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
RESULTS_DIR="/perf_eval_results"
READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
mkdir -p "$RESULTS_DIR"
FAIL=0

# shellcheck disable=SC2317  # invoked indirectly via trap
cleanup() {
  echo "=== MOE_DIAG cleanup: killing vllm server ==="
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  local tag="$1"; shift
  local server_log="$RESULTS_DIR/server_${tag}.log"
  echo "=== MOE_DIAG SERVER START tag=${tag} extra_env=[$*] ==="
  env "$@" vllm serve "$MODEL" \
    --max-model-len=9216 \
    --max-num-batched-tokens=8192 \
    --no-enable-prefix-caching \
    --max-num-seqs=512 \
    --gpu-memory-utilization=0.9 \
    --block-size=256 \
    --async-scheduling \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder \
    --reasoning-parser=qwen3 \
    --limit-mm-per-prompt='{"image": 0, "video": 0}' \
    --kv-cache-dtype=fp8 \
    --tensor-parallel-size=8 \
    --enable-expert-parallel \
    --attention-backend=CUSTOM \
    --port=$PORT > "$server_log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "=== MOE_DIAG ERROR: server ${tag} died during startup ==="
      tail -100 "$server_log"
      exit 1
    fi
    if [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
      echo "=== MOE_DIAG ERROR: server ${tag} not ready after ${READY_TIMEOUT_S}s ==="
      tail -100 "$server_log"
      exit 1
    fi
    sleep 15; waited=$((waited + 15))
  done
  echo "=== MOE_DIAG SERVER READY tag=${tag} after ~${waited}s ==="
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 10
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 10
}

probe() {
  local tag="$1"
  echo "=== MOE_DIAG PROBE ${tag} ==="
  curl -s "http://localhost:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:","max_tokens":64,"temperature":0}' \
    | tee -a "$RESULTS_DIR/probes_${tag}.json"
  echo ""
  echo "=== MOE_DIAG PROBE ${tag} END ==="
}

run_gsm8k() {
  local tag="$1" conc="$2" limit="$3"
  local limit_args=()
  if [ "$limit" != "full" ]; then
    limit_args=(--limit "$limit")
  fi
  echo "=== MOE_DIAG GSM8K START tag=${tag} conc=${conc} limit=${limit} ==="
  HF_HUB_OFFLINE=0 lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=${conc},max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --seed "0,1234,None,1234" \
    "${limit_args[@]}" \
    --log_samples \
    --output_path "$RESULTS_DIR/gsm8k_${tag}" \
    2>&1 | tee "$RESULTS_DIR/gsm8k_${tag}.log" || FAIL=1
  echo "=== MOE_DIAG GSM8K END tag=${tag} ==="
}

# Snapshot of the compile cache so the log shows whether lifecycle N+1
# actually reused what lifecycle N wrote (entry count + newest entries).
cache_stat() {
  local tag="$1"
  local cache_root="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
  echo "=== MOE_DIAG CACHE ${tag} root=${cache_root} ==="
  if [ -d "$cache_root" ]; then
    echo "[cache] entries: $(find "$cache_root" -type f 2>/dev/null | wc -l)"
    echo "[cache] bytes:   $(du -sb "$cache_root" 2>/dev/null | cut -f1)"
    echo "[cache] newest:"
    find "$cache_root" -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -5 | cut -d' ' -f2-
  else
    echo "[cache] root does not exist yet"
  fi
  echo "=== MOE_DIAG CACHE ${tag} END ==="
}

# ------------------------------------------------------------------
# 1. noenv_first: no MoE envs. Establishes the clean baseline for this
#    job on whatever cache state the image ships with.
# ------------------------------------------------------------------
cache_stat before_noenv_first
start_server noenv_first
probe noenv_first_fresh
run_gsm8k noenv_first 32 250
stop_server

# ------------------------------------------------------------------
# 2. onehot_mid: the load-bearing threshold ON. Expected to garble, and
#    to write the suspect compiled graphs into the shared cache.
# ------------------------------------------------------------------
cache_stat before_onehot_mid
start_server onehot_mid ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1
probe onehot_mid_fresh
run_gsm8k onehot_mid 32 250
stop_server

# ------------------------------------------------------------------
# 3. noenv_again: byte-identical server config to lifecycle 1. The cache
#    is deliberately NOT cleared, so garbage here means lifecycle 2's
#    graphs were reused despite the env being gone — i.e. the threshold
#    is missing from the compile cache key.
# ------------------------------------------------------------------
cache_stat before_noenv_again
start_server noenv_again
probe noenv_again_fresh
run_gsm8k noenv_again 32 250
stop_server
cache_stat final

# ------------------------------------------------------------------
# Summary: compare noenv_first vs noenv_again — identical server configs,
# so any delta between them is cache carryover from lifecycle 2.
# ------------------------------------------------------------------
echo "=== MOE_DIAG SUMMARY ==="
grep -H -E "strict-match|flexible-extract" "$RESULTS_DIR"/gsm8k_*.log | grep exact_match || true
echo "=== MOE_DIAG DONE FAIL=${FAIL} ==="
exit "$FAIL"
