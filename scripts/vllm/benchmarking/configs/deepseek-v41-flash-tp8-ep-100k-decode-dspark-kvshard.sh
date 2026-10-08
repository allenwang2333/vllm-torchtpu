#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k decode with DSpark K=5 and owner-sharded KV
# (TPU_DEEPSEEK_V41_KV_SHARD): each TP rank keeps the MQA KV and Indexer K
# blocks of the requests it owns, so the default 20 GiB KV budget can hold
# C=224, where replicated 100k requests fill it near C=86. With
# TPU_DEEPSEEK_V41_KV_SHARD=0 this is the replicated baseline on the same
# buckets (OFFLOAD=1 above C=86).
# KV_SHARD pads a step to a bucket that is a multiple of the TP size and holds
# TP x the tokens of its busiest owner, and rejects buckets the TP size does
# not divide, so the DSpark config's 4 is dropped. A balanced verify step at
# C=96/128/192/224 is C x (1 + K) = 576/768/1152/1344 tokens; 1536 covers a
# busier owner. Each owner computes at most 2048 / TP = 256 tokens a step.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TPU_DEEPSEEK_V41_KV_SHARD="${TPU_DEEPSEEK_V41_KV_SHARD:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-224}"
CONCURRENCY_OPTIONS="${CONCURRENCY_OPTIONS:-96 128 192 224}"
COMPILE_SIZES="${COMPILE_SIZES:-[8,16,24,32,48,64,96,144,192,288,384,576,768,1152,1344,1536,2048]}"
# shellcheck source=/dev/null
source "$CONFIG_DIR/deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh"
