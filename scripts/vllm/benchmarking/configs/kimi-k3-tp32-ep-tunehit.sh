#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# The real-model recipe that can SELECT a Kimi-K3 row from the MLA v2 tuning
# table. `kimi-k3-tp32-ep` cannot: it ships max-num-batched-tokens 512 and
# KV dtype auto, and both K3 TP32 rows require 8192 and float8_e4m3fn.
#
# Target row, `kernels/mla/v2/tuned_params.py:1169`:
#   case='mixed' max_num_tokens=8192 actual_num_q_heads=3 actual_lkv_dim=512
#   actual_r_dim=64 kv_dtype='float8_e4m3fn' q_dtype='bfloat16'
#   page_size_per_kv_packing=64 kv_packing=4 max_num_seqs=8 pages_per_seq=40
#     -> q_split=8, num_kv_pages_per_block=16, num_queries_per_block=512
#
# and how each field is forced (`layers/common/attention_interface.py:540`):
#   max_num_tokens           = the compiled bucket   <- mnbt 8192 + compile 8192
#   actual_num_q_heads       = 96 / TP               <- TP=32 gives 3
#   kv_dtype                 = KV cache dtype        <- --kv-cache-dtype fp8
#   kv_packing               = 4 for fp8, 2 for bf16 <- implied by fp8
#   page_size_per_kv_packing = block_size/kv_packing <- --block-size 256 -> 64
#   max_num_seqs             = --max-num-seqs        <- 8
#   pages_per_seq            = ceil(mml/block_size)  <- 10240/256 = 40
#
# The sibling row `:1143` is NOT reachable at this workload: it needs
# pages_per_seq=520 at block 16, i.e. max_model_len ~8320 < 8192+1024.
#
# Only `case='mixed'` can hit. No batched_decode row in the table has 3 (or
# 12) q-heads, so K3 decode runs untuned at every TP. This is a TTFT lever.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 5, not the nightly's 20: this job pays for four server lifetimes, and the
# noise band (lane D, BK #305) says 3 repeats of 5 prompts already resolve
# 1% of TTFT once the arms are interleaved.
NUM_PROMPTS=5
RANDOM_RANGE_RATIO="0.0"

# 10240 -> pages_per_seq 40 (row hits). 10496 -> 41 (identical workload,
# fallback). Still covers 8192 + 1024.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[16,8192]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
