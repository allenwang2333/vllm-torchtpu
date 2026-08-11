#!/bin/bash
# vllm-torchtpu needle sweep. Same LLM(...) args as tpu-inference/run_dsv4_needle_ref.py
# so the two runs are directly comparable.
#
# Usage: run_ours_needle.sh <logfile> [extra env assignments...]
#   e.g. run_ours_needle.sh runB.log DSV4_CSA_FILTER=1
set -u
LOG="$1"; shift
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
  DSV4_PROMPT_FILES="${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}/scratchpad/short.txt,${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}/scratchpad/needle_250.txt" \
  "$@" \
  ${WORKSPACE:-$HOME/workspace}/uv_venv_dsv4/bin/python3 \
  examples/run_dsv4_inference.py > "$LOG" 2>&1

echo "=== exit $? ==="
grep -E "^\[SWEEP\]|^Generated token IDs" "$LOG"
