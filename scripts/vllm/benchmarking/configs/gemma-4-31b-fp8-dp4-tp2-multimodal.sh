#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: RedHatAI/gemma-4-31B-it-FP8-Dynamic (Dense, TP=2 on v7x-8).
MODEL="RedHatAI/gemma-4-31B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="128"
# Per DP engine (vLLM does not divide max_num_seqs by data_parallel_size), but
# nothing here puts more than 128 requests in flight server-wide: the benchmark
# runs 128, lm-eval num_concurrent=128, evalscope eval-batch-size 128. So 128
# holds even if one engine got every request; 512 compiled the 256 and 512
# sequence buckets for nothing.
MAX_NUM_SEQS=128
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.03"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --mm-processor-kwargs {\"size\":{\"longest_edge\":262144,\"shortest_edge\":3136}} --disable-chunked-mm-input --additional_config {\"mm-encoder-tp-mode\":\"data\"} --compilation-config {\"cudagraph_mm_encoder\":true,\"encoder_cudagraph_max_vision_items_per_batch\":8,\"encoder_cudagraph_max_frames_per_batch\":8}"
