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
# Baseline: does GLM-5.2-NVFP4 load and serve on one 2x2x1 node?
#
# CLI equivalent of the reference invocation, adapted for cdk:
#
#   vllm serve nvidia/GLM-5.2-NVFP4 --tensor-parallel-size 8 \
#     --kv-cache-dtype fp8 --tool-call-parser glm47 --enable-auto-tool-choice \
#     --reasoning-parser glm45 --safetensors-load-strategy=prefetch \
#     --max_model_len=32768 --enable_expert_parallel
#
# Adaptations, all forced by the environment rather than by preference:
#   * weights are pre-fetched once, single-process, into the SSD-backed
#     $HF_HOME PVC so they survive the job and are reused by the next one;
#   * the server is health-polled and torn down in a trap, because a container
#     that does not exit holds the four chips until TTL cleanup;
#   * a smoke completion runs against the live server -- "/health returned
#     200" only proves the HTTP layer came up, not that the sparse MLA path
#     produces tokens.
#
#   cdk job create glm52-serve --ttl 3600 \
#     SCRIPT=../../.claude/worktrees/glm-kv-probe/scripts/cdk/glm_serve
set -uo pipefail

MODEL="${MODEL_ID:-nvidia/GLM-5.2-NVFP4}"
TP="${TP_SIZE:-8}"                     # 2x2x1 = 4 chips x 2 devices
MAX_LEN="${MAX_MODEL_LEN:-32768}"
KV_DTYPE="${KV_CACHE_DTYPE:-fp8}"
LOAD_STRATEGY="${LOAD_STRATEGY:-prefetch}"
PORT="${SERVE_PORT:-8000}"
MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
FETCH_ONLY="${FETCH_ONLY:-0}"
SERVER_LOG=/tmp/glm_server.log

echo "### model=$MODEL tp=$TP max_len=$MAX_LEN kv_dtype=$KV_DTYPE"
echo "### load_strategy=$LOAD_STRATEGY mem_util=$MEM_UTIL fetch_only=$FETCH_ONLY"
echo "### HF_TOKEN present: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"

export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

# This branch's source must win over the main checkout that the recipe put on
# PYTHONPATH ahead of us -- the fix for the positional MLAAttention super()
# call lives here, and without it construction dies against this vLLM.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HERE}/../../src:${PYTHONPATH:-}"
echo "### PYTHONPATH=$PYTHONPATH"

df -h /hf-cache 2>&1 | sed 's/^/### /'

# ------------------------------------------------------------------ fetch ---
# One process, with retries. Concurrent snapshot_download calls from each TP
# worker would multiply 433 GiB of egress and HF starts dropping connections
# mid-file. Already-present files are skipped, so a re-run is nearly free.
if [ "${MODEL#/}" = "$MODEL" ]; then   # not an absolute path -> HF repo id
  echo "### pre-fetching $MODEL into ${HF_HOME:-~/.cache/huggingface}"
  MODEL="$MODEL" python3 - <<'FETCHEOF'
import os, time
from huggingface_hub import snapshot_download
repo = os.environ["MODEL"]
t0 = time.time()
for attempt in range(1, 31):
    try:
        p = snapshot_download(repo, max_workers=16,
                              token=os.environ.get("HF_TOKEN") or None)
        print(f"### fetched to {p} in {time.time()-t0:.0f}s", flush=True)
        break
    except Exception as e:
        print(f"### fetch attempt {attempt} failed: {type(e).__name__}: "
              f"{str(e)[:200]}", flush=True)
        time.sleep(min(30, 5 * attempt))
else:
    raise SystemExit("### FATAL: could not fetch weights")
FETCHEOF
  rc=$?
  [ $rc -eq 0 ] || { echo "### FETCH_EXIT=$rc"; exit $rc; }
fi
du -sh "${HF_HOME:-/hf-cache}" 2>/dev/null | sed 's/^/### cache size: /'

if [ "$FETCH_ONLY" = "1" ]; then
  echo "### FETCH_ONLY=1, weights are on the PVC; stopping before serve"
  exit 0
fi

# ----------------------------------------------------------------- server ---
stop_server() {
  echo "### stopping vLLM server"
  [ -n "${SERVER_PID:-}" ] && kill -TERM "$SERVER_PID" 2>/dev/null
  command -v pkill >/dev/null 2>&1 && pkill -TERM -f "vllm serve" 2>/dev/null
  sleep 5
  [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null
  if command -v pkill >/dev/null 2>&1; then
    pkill -9 -f "vllm serve" 2>/dev/null
    pkill -9 -f "VLLM::" 2>/dev/null
  fi
  sleep 2
  rm -f /tmp/libtpu_lockfile*
}
trap stop_server EXIT INT TERM

command -v vllm >/dev/null 2>&1 || { echo "### FATAL: no 'vllm' on PATH"; exit 4; }

SERVER_CMD=(vllm serve "$MODEL"
  --tensor-parallel-size "$TP"
  --kv-cache-dtype "$KV_DTYPE"
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --safetensors-load-strategy="$LOAD_STRATEGY"
  --max-model-len "$MAX_LEN"
  --enable-expert-parallel
  --gpu-memory-utilization "$MEM_UTIL"
  --trust-remote-code
  --port "$PORT")
# shellcheck disable=SC2206  # deliberate word split
[ -n "${EXTRA_SERVE_ARGS:-}" ] && SERVER_CMD+=(${EXTRA_SERVE_ARGS})

echo "### [cmd] ${SERVER_CMD[*]}"
T0=$(date +%s)
"${SERVER_CMD[@]}" > >(tee -a "$SERVER_LOG") 2>&1 &
SERVER_PID=$!
echo "### server pid=$SERVER_PID"

waited=0
READY=0
while [ "$waited" -lt "$READY_TIMEOUT" ]; do
  if curl -s -o /dev/null --connect-timeout 1 "http://localhost:$PORT/health"; then
    echo "### SERVER READY after $(( $(date +%s) - T0 ))s"
    READY=1
    break
  fi
  # kill -0 on the pid, not pgrep: pgrep is absent from this image and its
  # non-zero exit reads as "the server died".
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "### server died during startup after ${waited}s"
    break
  fi
  sleep 10
  waited=$((waited + 10))
  [ $((waited % 300)) -eq 0 ] && echo "### waiting for /health ... ${waited}s"
done

if [ "$READY" != "1" ]; then
  echo "========== last 120 lines of server.log =========="
  tail -120 "$SERVER_LOG"
  echo "=================================================="
  echo "### LOAD_RESULT=FAIL"
  exit 10
fi

# ------------------------------------------------------------ smoke test ----
# /health only proves the HTTP layer is alive. The sparse MLA + indexer path
# is not exercised until a request actually decodes.
echo "### smoke completion"
curl -s -X POST "http://localhost:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 24, \"temperature\": 0}" \
  | head -c 2000
echo
SMOKE_RC=$?

echo "### --- capacity / config lines from the server log ---"
grep -hoE "Maximum concurrency[^\"]*|GPU KV cache size[^\"]*|# GPU blocks[^\"]*|Available KV cache memory[^\"]*|Model loading took[^\"]*|available_memory[^\"]*" \
  "$SERVER_LOG" | tail -12 | sed 's/^/### /'
echo "### --- sparse / indexer evidence ---"
grep -hiE "sparse|indexer|topk|fp8_ds_mla|kv_cache_dtype" "$SERVER_LOG" | tail -15 | sed 's/^/### /'

echo "### LOAD_RESULT=OK smoke_rc=$SMOKE_RC"

# ----------------------------------------------------------- post-ready -----
# Anything that needs a live server (lm_eval, benchmarks) runs here rather
# than in its own job: TPU quota is the scarce resource, and a second job
# would pay another admission wait plus another weight load. Unset by default,
# so a plain load check is unaffected.
POST_RC=0
if [ -n "${POST_READY_CMD:-}" ]; then
  echo "### POST_READY_CMD: $POST_READY_CMD"
  bash -c "$POST_READY_CMD"
  POST_RC=$?
  echo "### POST_READY_RC=$POST_RC"
fi

exit $POST_RC
