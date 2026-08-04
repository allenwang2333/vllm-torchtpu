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

# Re-exec from a local copy before doing anything else. bash reads a script
# incrementally, so editing this file on the NFS export while a job is part
# way through it invalidates the handle and the run dies at the next read
# with "error reading input file: Stale file handle" -- which is exactly what
# killed the first successful load, after the server was already up. The NFS
# fast loop is worth keeping; this just makes it safe to edit mid-run.
if [ "${GLM_SERVE_LOCAL:-0}" != "1" ]; then
  export GLM_SERVE_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cp "${BASH_SOURCE[0]}" /tmp/glm_serve.local.sh || {
    echo "### FATAL: could not stage the script locally"; exit 6; }
  export GLM_SERVE_LOCAL=1
  exec bash /tmp/glm_serve.local.sh "$@"
fi

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

# XLA compilation of this model takes ~38 min across the eight shape buckets,
# and vLLM caches it under $VLLM_CACHE_ROOT -- which defaults to ~/.cache/vllm
# inside the container and dies with the pod. Point it at the weights PVC so
# the next run reuses the graphs instead of recompiling. Keyed by config hash,
# so a change that matters still recompiles.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/hf-cache/vllm_cache}"
mkdir -p "$VLLM_CACHE_ROOT"
echo "### VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"

# Required, not tuning. Left unset, the NVFP4 MoE keeps the checkpoint's
# native block-16 scales and dequantizes inside the gmm; because 16 < the
# MXU's 128 columns, `should_dequantize_before_matmul` takes the branch that
# broadcasts the scale across sublanes with a zero-stride load, and Mosaic
# only implements that when the base memref's lane dim is 128. GLM's is
# tile_n = 2048, so compilation dies with
#   "Not Implemented: The last dim size is not 128 in original base memref"
# in WeightsRef.get_scale. Requantizing to a block >= 128 removes the
# in-kernel dequant entirely and the broadcast with it. GLM's hidden (6144)
# and moe_intermediate (2048) are both divisible by 512, so this takes the
# fast JAX requant path rather than the torch pad-and-requant fallback.
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-512}"
echo "### MOE_REQUANTIZE_BLOCK_SIZE=$MOE_REQUANTIZE_BLOCK_SIZE"

# Mosaic ships inside libtpu, so the compiler that rejects the zero-stride
# scale broadcast is a property of the libtpu build, not of vLLM or the
# kernels. The only recipe anyone has GLM-5.2 serving from
# (multihost/torchtpu/vllm-torchtpu-glm-5.2.yml) does not use the image's
# libtpu at all -- it pulls a pinned build from GCS and points
# TPU_LIBRARY_PATH at it. Set LIBTPU_BUILD to a date under
# gs://cloud-devkit/libs/libtpu/ to do the same; unset uses the image's.
LIBTPU_BUILD="${LIBTPU_BUILD:-}"
if [ -n "$LIBTPU_BUILD" ]; then
  LIBTPU_DEST="/root/cloud-devkit/base/bin/libtpu.so"
  echo "### pinning libtpu to $LIBTPU_BUILD"
  mkdir -p "$(dirname "$LIBTPU_DEST")"
  if gcloud storage cp "gs://cloud-devkit/libs/libtpu/${LIBTPU_BUILD}/libtpu.so" \
       "$LIBTPU_DEST" 2>&1 | tail -2; then
    export TPU_LIBRARY_PATH="$LIBTPU_DEST"
    ls -la "$LIBTPU_DEST" | sed 's/^/### /'
  else
    echo "### FATAL: could not fetch libtpu build $LIBTPU_BUILD"
    exit 5
  fi
fi
echo "### TPU_LIBRARY_PATH=${TPU_LIBRARY_PATH:-<image default>}"
echo "### TPU_GMM_SCALE_BCAST_FALLBACK=${TPU_GMM_SCALE_BCAST_FALLBACK:-1}"

# This branch's source must win over the main checkout that the recipe put on
# PYTHONPATH ahead of us -- the fix for the positional MLAAttention super()
# call lives here, and without it construction dies against this vLLM.
HERE="${GLM_SERVE_SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
  --host 0.0.0.0
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

# ------------------------------------------------------------ serve wait ----
# Split-pod mode: a client in another pod drives this server, so it has to
# stay up. The JobSet's successPolicy targets the client job, so when the
# client finishes JobSet tears this pod down and releases the chips -- which
# is the only thing that stops SERVE_WAIT running to the deadline.
if [ "${SERVE_WAIT:-0}" = "1" ]; then
  DEADLINE="${SERVE_WAIT_TIMEOUT:-14400}"
  echo "### SERVE_WAIT=1: holding the server up for up to ${DEADLINE}s"
  waited=0
  while [ "$waited" -lt "$DEADLINE" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "### server exited on its own after ${waited}s"
      break
    fi
    sleep 30
    waited=$((waited + 30))
    [ $((waited % 600)) -eq 0 ] && echo "### still serving ... ${waited}s"
  done
  echo "### SERVE_WAIT done after ${waited}s"
fi

exit $POST_RC
