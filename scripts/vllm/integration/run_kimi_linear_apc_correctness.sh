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
#
# Single-instance prefix caching for Kimi-Linear (KDA) on the unified KV
# pool. Mirrors run_qwen35_apc_correctness.sh; the contamination smoke is the
# one that catches a cache hit restoring attention KV without the KDA state.
set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-2700}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kimi-Linear defaults to the unified pool (POOLED_GDN_ARCHITECTURES).
# max_num_seqs=1 removes batching nondeterminism, so a cold/warm difference is
# the cache and nothing else.
vllm serve "$MODEL" \
  --tensor-parallel-size "${TP:-8}" \
  --enable-expert-parallel \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --port "$PORT" \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --safetensors-load-strategy prefetch &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "localhost:$PORT/health" >/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "APC_CORRECTNESS_FAIL: server exited during startup"
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "APC_CORRECTNESS_FAIL: server did not become healthy in time"
    exit 1
  fi
  sleep 10
done

python3 "$HERE/smoke_prefix_cache_e2e_divergence.py" \
  --host localhost --port "$PORT" --model "$MODEL"
# The contamination smoke interleaves two prefixes so a resumed request's
# slot last held the other one: it is the decisive state-restore check, so it
# runs before the correctness smoke (whose uncached strict-string cases, e.g.
# translate_cn's trailing period, fail on content grounds for this model and
# exit non-zero).
python3 "$HERE/smoke_mamba_state_contamination.py" \
  --host localhost --port "$PORT" --model "$MODEL" --lines 200
python3 "$HERE/smoke_prefix_cache_correctness.py" \
  --host localhost --port "$PORT" --model "$MODEL" || true

# A run that never hit the cache would pass both smokes vacuously.
hits=$(curl -sf "localhost:$PORT/metrics" |
  awk '/^vllm:prefix_cache_hits_total/ {s += $2} END {printf "%d", s}')
echo "prefix_cache_hits_total=$hits"
if [ "$hits" -le 0 ]; then
  echo "APC_CORRECTNESS_FAIL: prefix cache never hit"
  exit 1
fi

echo "APC_CORRECTNESS_OK"
