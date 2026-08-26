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

# A/B evidence run for PR #170 (unified KV pool default) on Qwen3.5-397B tp8.
# One server lifecycle: serve -> benchmark_serving.py (perf) -> lm_eval gsm8k
# (accuracy) against the SAME server instance, so quality and perf are
# measured on identical serving state.
#
# Usage: run_ab_test.sh <arm-name>
# Runs inside the CI docker image via .buildkite/scripts/run_in_docker.sh.
# Results land in /perf_eval_results (uploaded as Buildkite artifacts).
set -uo pipefail

ARM="${1:?usage: run_ab_test.sh <arm-name>}"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
RESULTS_DIR="/perf_eval_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

mkdir -p "$RESULTS_DIR"
COMMIT="${BUILDKITE_COMMIT:-unknown}"

echo "=== PR170_AB ARM=${ARM} COMMIT=${COMMIT} POOL_ENV=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-unset} ==="

# shellcheck disable=SC2317  # invoked indirectly via trap
cleanup() {
  echo "=== PR170_AB cleanup: killing vllm server ==="
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 1. Server: the command under discussion, verbatim except
#    machine-local cache/download dirs (CI agents use the mounted
#    HF cache) and --attention-config.backend spelled as
#    --attention-backend for the vllm 0.26.0 pinned here.
# ------------------------------------------------------------------
SERVER_LOG="$RESULTS_DIR/server_${ARM}.log"
ONEHOT_MOE_PERMUTE_THRESHOLD=32768 \
USE_MOE_SPARSE_CORE=1 \
vllm serve "$MODEL" \
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
  --port=$PORT > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[pr170_ab] server pid=$SERVER_PID, waiting up to ${READY_TIMEOUT_S}s for readiness"

WAITED=0
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[pr170_ab] ERROR: server process died during startup; last 100 log lines:"
    tail -100 "$SERVER_LOG"
    exit 1
  fi
  if [ "$WAITED" -ge "$READY_TIMEOUT_S" ]; then
    echo "[pr170_ab] ERROR: server not ready after ${READY_TIMEOUT_S}s; last 100 log lines:"
    tail -100 "$SERVER_LOG"
    exit 1
  fi
  sleep 15
  WAITED=$((WAITED + 15))
  if [ $((WAITED % 300)) -eq 0 ]; then
    echo "[pr170_ab] still waiting for server... ${WAITED}s"
  fi
done
echo "[pr170_ab] server ready after ~${WAITED}s"

FAIL=0

# Fixed-prompt greedy probe so raw completions are visible in the log at
# known points of the server lifecycle (fresh, post-bench).
probe() {
  local tag="$1"
  echo "=== PR170_AB PROBE ${tag} ARM=${ARM} ==="
  curl -s "http://localhost:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:","max_tokens":64,"temperature":0}' \
    | tee -a "$RESULTS_DIR/probes_${ARM}.json"
  echo ""
  echo "=== PR170_AB PROBE ${tag} END ==="
}

# Full GSM8K (5-shot) via lm_eval against the live server, with per-sample
# outputs logged so wrong answers can be inspected, not just counted.
run_gsm8k() {
  local phase="$1"
  echo "=== PR170_AB GSM8K ${phase} START ARM=${ARM} ==="
  HF_HUB_OFFLINE=0 lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --seed "0,1234,None,1234" \
    --log_samples \
    --output_path "$RESULTS_DIR/gsm8k_${phase}_${ARM}" \
    2>&1 | tee "$RESULTS_DIR/gsm8k_${phase}_${ARM}.log" || FAIL=1
  echo "=== PR170_AB GSM8K ${phase} END ARM=${ARM} ==="
}

probe fresh

# GSM8K on the fresh server, before any bench traffic: separates
# "path is wrong from the start" from "bench traffic corrupts state".
run_gsm8k pre

# ------------------------------------------------------------------
# 2. Perf: the golden benchmark_serving.py client, verbatim args.
# ------------------------------------------------------------------
echo "=== PR170_AB BENCHMARK START ARM=${ARM} ==="
python3 "$SCRIPT_DIR/bench_serving/benchmark_serving.py" \
  --ignore-eos \
  --model="$MODEL" \
  --backend=vllm \
  --port=$PORT \
  --dataset-name=random \
  --random-input-len=8192 \
  --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=320 \
  --num-warmups=64 \
  --max-concurrency=32 \
  --request-rate=inf \
  --percentile-metrics='ttft,tpot,itl,e2el' \
  --save-result \
  --result-dir "$RESULTS_DIR" \
  --result-filename "bench_${ARM}.json" \
  2>&1 | tee "$RESULTS_DIR/bench_${ARM}.log" || FAIL=1
echo "=== PR170_AB BENCHMARK END ARM=${ARM} ==="

# ------------------------------------------------------------------
# 3. Accuracy: full GSM8K (5-shot) on the SAME live server, after the
#    bench traffic (plus a fixed-prompt probe in between).
# ------------------------------------------------------------------
probe post_bench
run_gsm8k post

# ------------------------------------------------------------------
# 4. Compact summary for log scraping.
# ------------------------------------------------------------------
echo "=== PR170_AB SUMMARY ARM=${ARM} COMMIT=${COMMIT} POOL_ENV=${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-unset} ==="
if [ -f "$RESULTS_DIR/bench_${ARM}.json" ]; then
  python3 - "$RESULTS_DIR/bench_${ARM}.json" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in d if any(s in k for s in (
    "throughput", "ttft", "tpot", "itl", "e2el", "duration", "completed"))]
for k in sorted(keys):
    print(f"[metric] {k}={d[k]}")
PYEOF
fi
grep -E "exact_match" "$RESULTS_DIR"/gsm8k_*_"${ARM}".log | tail -10 || true
echo "=== PR170_AB DONE ARM=${ARM} FAIL=${FAIL} ==="
exit "$FAIL"
