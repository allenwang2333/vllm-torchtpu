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

# Expected FAILURE (repro) config for Qwen3-0.6B with PCP=8 and compile_sizes:[256] (b531897683#comment6).
# Standalone execution: starts vllm serve, waits for health, runs vllm bench serve.

set -e

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-128}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1}
NUM_WARMUPS=${NUM_WARMUPS:-1}

echo "--- Starting vllm serve in background ---"
vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.8 \
  --attention-backend CUSTOM \
  --trust-remote-code \
  --dtype bfloat16 \
  --distributed-executor-backend mp \
  --cp-kv-cache-interleave-size 32 \
  --prefill-context-parallel-size 8 \
  --disable-custom-all-reduce \
  --block-size 32 \
  --no-async-scheduling \
  --compilation-config '{"compile_sizes":[256]}' \
  > /tmp/vllm_serve_repro.log 2>&1 &
SERVER_PID=$!

echo "--- Waiting for server to be healthy ---"
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Server is healthy after ${i}0s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process died. Logs:"
    cat /tmp/vllm_serve_repro.log
    exit 1
  fi
  sleep 10
done

echo "--- Running vllm bench serve ---"
vllm bench serve \
  --model "${MODEL}" \
  --host localhost \
  --port "${PORT}" \
  --dataset-name random \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --num-warmups "${NUM_WARMUPS}" \
  --ignore-eos

echo "--- Stopping server ---"
kill "${SERVER_PID}" || true
