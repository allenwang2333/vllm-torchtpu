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

# MoE-focused diagnosis of the gsm8k accuracy collapse seen at eval
# concurrency 32 (but not 128) on the per-layer KV path.
#
# Phase A: server with USE_MOE_SPARSE_CORE=1 + ONEHOT_MOE_PERMUTE_THRESHOLD
#   (the config under discussion) — sweep eval concurrency on a FRESH server
#   to (a) check whether bench traffic is required to trigger the garbage and
#   (b) map which decode batch shapes garble.
# Phase B: identical server WITHOUT the MoE sparse-core envs — if conc-32
#   recovers, the SC MoE path is implicated.
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

# ------------------------------------------------------------------
# Phase A: MoE sparse-core ON (config under discussion), fresh server,
# concurrency sweep. conc32/conc128 use the full 1319 questions; the
# other points are 250-question spot checks.
# ------------------------------------------------------------------
start_server sc_on ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1
probe sc_on_fresh
run_gsm8k sc_on_conc32 32 full
run_gsm8k sc_on_conc128 128 full
run_gsm8k sc_on_conc64 64 250
run_gsm8k sc_on_conc16 16 250
run_gsm8k sc_on_conc8 8 250
stop_server

# ------------------------------------------------------------------
# Phase B: identical server WITHOUT the MoE sparse-core envs.
# ------------------------------------------------------------------
start_server sc_off
probe sc_off_fresh
run_gsm8k sc_off_conc32 32 full
run_gsm8k sc_off_conc128 128 250
stop_server

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "=== MOE_DIAG SUMMARY ==="
grep -H -E "strict-match|flexible-extract" "$RESULTS_DIR"/gsm8k_*.log | grep exact_match || true
echo "=== MOE_DIAG DONE FAIL=${FAIL} ==="
exit "$FAIL"
