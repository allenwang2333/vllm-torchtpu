#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k decode with DP=2 x TP=4 + EP=8 and DSpark K=5
# speculative decoding (EXT-1B).
# Under OFFLOAD=1 (40 GiB 2-bank KV cache per rank), each TP=4 DP replica
# supports up to ~172 concurrent 100k requests (~344 total across DP=2),
# unlocking C_total=128..224 (C_local=64..112 per DP replica).
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${OFFLOAD:-0}" == "1" ]]; then
  export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-1}"
fi
if [[ "${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-0}" == "1" ]]; then
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
  KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-42949672960}"
  CONCURRENCY_OPTIONS="${CONCURRENCY_OPTIONS:-16 32 48 64 96 128 192}"
fi
if [[ "${MAX_NUM_SEQS:-16}" -gt 96 ]]; then
  COMPILE_SIZES="${COMPILE_SIZES:-[4,8,16,24,32,48,64,96,112,144,192,288,384,576,672,2048]}"
elif [[ "${MAX_NUM_SEQS:-16}" -gt 16 ]]; then
  COMPILE_SIZES="${COMPILE_SIZES:-[4,8,16,24,32,48,64,96,144,192,288,384,576,2048]}"
else
  COMPILE_SIZES="${COMPILE_SIZES:-[4,8,16,24,48,96,2048]}"
fi
# shellcheck source=/dev/null
source "$CONFIG_DIR/deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh"
DSPARK_K="${DSPARK_K:-5}"
# With MODEL_URI the target's model path becomes a local config-only cache
# dir; the drafter inherits it and runai_streamer finds no safetensors there,
# so point the draft at the object-storage URI explicitly.
DSPARK_MODEL="${MODEL_URI:+,\"model\":\"${MODEL_URI}\"}"
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --speculative-config={\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K}${DSPARK_MODEL}}"
