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

# One-knob-at-a-time matrix for the gsm8k accuracy collapse seen at eval
# concurrency 32 (but not 128) on the per-layer KV path.
#
# Each arm runs the SAME server command and the SAME client sequence
# (probe -> gsm8k pre -> benchmark_serving -> probe -> gsm8k post) against a
# single server lifecycle, varying exactly one serving knob:
#
#   repro_ctrl       the config under discussion, unchanged (control)
#   onehot_default   threshold back to its default 0 -> one-hot permute
#                    branch never taken (sparse core stays on)
#   sc_off           USE_MOE_SPARSE_CORE=0 -> sparse-core MoE truly off
#   no_ep            control minus --enable-expert-parallel
#   flash_attn       control with --attention-backend=FLASH_ATTN
#   skip_padded      control plus TPU_MOE_SKIP_PADDED_TOKENS=1
#   bf16_kv          control minus --kv-cache-dtype=fp8
#
# Comparing an arm against repro_ctrl attributes (or exonerates) that knob.
#
# bf16_kv tests the uninitialized-memory hypothesis rather than a MoE knob:
# kv_cache_materializer.py allocates the raw KV cache with torch.empty for fp8
# dtypes and torch.zeros otherwise, and the failing config sets
# --kv-cache-dtype=fp8. Dropping the flag takes the zeros branch naturally, so
# a clean bf16_kv points at reads of uninitialized KV memory -- under which the
# other "rescuing" arms above would only be moving buffer assignments so those
# reads land on benign memory, not fixing anything.
#
# Note on the MoE env defaults (src/vllm_torchtpu/envs.py): USE_MOE_SPARSE_CORE
# defaults to TRUE, so leaving it unset does NOT disable sparse core -- it must
# be set to 0 explicitly (sc_off). ONEHOT_MOE_PERMUTE_THRESHOLD defaults to 0.
# The sparse-core path in layers/common/fused_moe_gmm.py is only reached with
# expert parallelism on, which is what no_ep isolates.
#
# Usage: run_matrix_arm.sh <arm-name>
# Runs inside the CI docker image via .buildkite/scripts/run_in_docker.sh.
# Results land in /perf_eval_results (uploaded as Buildkite artifacts).
set -uo pipefail

ARM="${1:-}"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
RESULTS_DIR="/perf_eval_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

usage() {
  echo "usage: run_matrix_arm.sh <arm-name>"
  echo "valid arms: repro_ctrl onehot_default sc_off no_ep flash_attn skip_padded bf16_kv"
}

# ------------------------------------------------------------------
# Arm definition: SERVE_ENV holds the extra env assignments prepended to
# the serve invocation; EP_FLAG / ATTN_FLAG / KV_DTYPE_FLAG hold the varied
# flags (an empty EP_FLAG or KV_DTYPE_FLAG means the flag is dropped from the
# command line).
# ------------------------------------------------------------------
SERVE_ENV=()
EP_FLAG="--enable-expert-parallel"
ATTN_FLAG="--attention-backend=CUSTOM"
KV_DTYPE_FLAG="--kv-cache-dtype=fp8"

case "$ARM" in
  repro_ctrl)
    # USE_MOE_SPARSE_CORE=1 is redundant (it is the default) but kept so the
    # control mirrors the original command verbatim.
    SERVE_ENV=(ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1)
    ;;
  onehot_default)
    SERVE_ENV=()
    ;;
  sc_off)
    SERVE_ENV=(USE_MOE_SPARSE_CORE=0)
    ;;
  no_ep)
    SERVE_ENV=(ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1)
    EP_FLAG=""
    ;;
  flash_attn)
    SERVE_ENV=(ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1)
    ATTN_FLAG="--attention-backend=FLASH_ATTN"
    ;;
  skip_padded)
    SERVE_ENV=(
      ONEHOT_MOE_PERMUTE_THRESHOLD=32768
      USE_MOE_SPARSE_CORE=1
      TPU_MOE_SKIP_PADDED_TOKENS=1
    )
    ;;
  bf16_kv)
    SERVE_ENV=(ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1)
    KV_DTYPE_FLAG=""
    ;;
  *)
    echo "=== MATRIX ERROR: unknown arm '${ARM}' ==="
    usage
    exit 1
    ;;
esac

mkdir -p "$RESULTS_DIR"
COMMIT="${BUILDKITE_COMMIT:-unknown}"
FAIL=0

echo "=== MATRIX ARM=${ARM} COMMIT=${COMMIT} ENV=[${SERVE_ENV[*]-}] EP=[${EP_FLAG}] ATTN=[${ATTN_FLAG}] KV=[${KV_DTYPE_FLAG}] ==="

# shellcheck disable=SC2317  # invoked indirectly via trap
cleanup() {
  echo "=== MATRIX cleanup: killing vllm server ARM=${ARM} ==="
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  local server_log="$RESULTS_DIR/server_${ARM}.log"
  local serve_args=(
    --max-model-len=9216
    --max-num-batched-tokens=8192
    --no-enable-prefix-caching
    --max-num-seqs=512
    --gpu-memory-utilization=0.9
    --block-size=256
    --async-scheduling
    --language-model-only
    --enable-auto-tool-choice
    --tool-call-parser=qwen3_coder
    --reasoning-parser=qwen3
    '--limit-mm-per-prompt={"image": 0, "video": 0}'
  )
  if [ -n "$KV_DTYPE_FLAG" ]; then
    serve_args+=("$KV_DTYPE_FLAG")
  fi
  serve_args+=(--tensor-parallel-size=8)
  if [ -n "$EP_FLAG" ]; then
    serve_args+=("$EP_FLAG")
  fi
  serve_args+=("$ATTN_FLAG" "--port=$PORT")

  echo "=== MATRIX SERVER START ARM=${ARM} ==="
  env "${SERVE_ENV[@]+"${SERVE_ENV[@]}"}" \
    vllm serve "$MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
  SERVER_PID=$!
  echo "[matrix] server pid=$SERVER_PID, waiting up to ${READY_TIMEOUT_S}s for readiness"

  local waited=0
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "=== MATRIX ERROR: server died during startup ARM=${ARM} ==="
      tail -100 "$server_log"
      exit 1
    fi
    if [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
      echo "=== MATRIX ERROR: server not ready after ${READY_TIMEOUT_S}s ARM=${ARM} ==="
      tail -100 "$server_log"
      exit 1
    fi
    sleep 15
    waited=$((waited + 15))
    if [ $((waited % 300)) -eq 0 ]; then
      echo "[matrix] still waiting for server... ${waited}s"
    fi
  done
  echo "=== MATRIX SERVER READY ARM=${ARM} after ~${waited}s ==="
}

# Fixed-prompt greedy probe so raw completions are visible in the log at
# known points of the server lifecycle (fresh, post-bench).
probe() {
  local tag="$1"
  echo "=== MATRIX PROBE ${tag} ARM=${ARM} ==="
  curl -s "http://localhost:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:","max_tokens":64,"temperature":0}' \
    | tee -a "$RESULTS_DIR/probes_${ARM}.json"
  echo ""
  echo "=== MATRIX PROBE ${tag} END ARM=${ARM} ==="
}

# GSM8K at the failing client concurrency (32), 250-question spot check,
# with per-sample outputs logged so wrong answers can be inspected.
run_gsm8k() {
  local phase="$1"
  echo "=== MATRIX GSM8K ${phase} START ARM=${ARM} ==="
  HF_HUB_OFFLINE=0 lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --seed "0,1234,None,1234" \
    --limit 250 \
    --log_samples \
    --output_path "$RESULTS_DIR/gsm8k_${phase}_${ARM}" \
    2>&1 | tee "$RESULTS_DIR/gsm8k_${phase}_${ARM}.log" || FAIL=1
  echo "=== MATRIX GSM8K ${phase} END ARM=${ARM} ==="
}

start_server

# 1. Fresh-server probe, before any traffic.
probe fresh

# 2. GSM8K on the fresh server: separates "path is wrong from the start"
#    from "bench traffic corrupts state".
run_gsm8k pre

# 3. Perf: the golden benchmark_serving.py client, verbatim args.
echo "=== MATRIX BENCHMARK START ARM=${ARM} ==="
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
echo "=== MATRIX BENCHMARK END ARM=${ARM} ==="

# 4. Post-bench probe + GSM8K on the SAME live server.
probe post_bench
run_gsm8k post

# 5. Compact summary for log scraping.
echo "=== MATRIX SUMMARY ARM=${ARM} COMMIT=${COMMIT} ==="
grep -H -E "exact_match" "$RESULTS_DIR"/gsm8k_*_"${ARM}".log || true
echo "=== MATRIX DONE ARM=${ARM} FAIL=${FAIL} ==="
exit "$FAIL"
