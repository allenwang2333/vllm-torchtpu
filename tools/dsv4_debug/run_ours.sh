#!/bin/bash
# Run DSv4 on vllm-torchtpu (ours). Same LLM(...) args as the tpu-inference
# reference runner, so the two are directly comparable.
#
# Usage: run_ours.sh <logfile> [VAR=value ...]
#   e.g. run_ours.sh /tmp/out.log DSV4_MAX_TOKENS=150 DSV4_PROMPT_FILES=...

set -u
LOG="$1"; shift
# Resolve the log path before cd'ing: a relative path would otherwise land in
# the repo we switch to, or fail outright if the directory isn't writable.
case "$LOG" in /*) ;; *) LOG="$PWD/$LOG" ;; esac
mkdir -p "$(dirname "$LOG")"
cd ${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}

env \
  VLLM_FP4_CACHE_DIR="" \
  MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
  MOE_REQUANTIZE_BLOCK_SIZE=512 \
  NEW_MODEL_DESIGN=1 \
  MODEL_IMPL_TYPE=vllm \
  VLLM_TP_SIZE=8 \
  VLLM_EP_SIZE=8 \
  VLLM_NO_USAGE_STATS=1 \
  VLLM_PLUGINS=torchtpu,torchtpu_general \
  VLLM_ENFORCE_EAGER=1 \
  DSV4_ENABLE_PREFIX_CACHING=0 \
  GPU_MEMORY_UTILIZATION=0.45 \
  DSV4_MAX_MODEL_LEN=1024 \
  DSV4_MAX_TOKENS=16 \
  DSV4_MAX_NUM_SEQS=16 \
  PYTHONPATH=${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}/src \
  HF_HOME=${WORKSPACE:-$HOME/workspace}/hf_cache \
  DSV4_MODEL_PATH="${WORKSPACE:-$HOME/workspace}/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1" \
  DSV4_PROMPT_FILES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prompts/short.txt" \
  "$@" \
  ${WORKSPACE:-$HOME/workspace}/uv_venv_dsv4/bin/python3 \
  examples/run_dsv4_inference.py > "$LOG" 2>&1

echo "=== exit $? ==="
grep -E "^\[SWEEP\]|^Generated token IDs" "$LOG"
