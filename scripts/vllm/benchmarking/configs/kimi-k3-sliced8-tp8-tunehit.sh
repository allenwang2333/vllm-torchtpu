#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# PROBE: the only serving config that can SELECT a Kimi-K3 row from the MLA v2
# tuning table. Every K3 recipe in the repo runs the untuned fallback.
#
# `kernels/mla/v2/tuned_params.py:1122` is the sole K3 row reachable at TP8:
#
#   case='mixed' max_num_tokens=4096 actual_num_q_heads=12 actual_lkv_dim=512
#   actual_r_dim=64 kv_dtype='bfloat16' q_dtype='bfloat16'
#   page_size_per_kv_packing=128 kv_packing=2 max_num_seqs=8 pages_per_seq=128
#
# The lookup is exact equality over all of those. The runtime key is read off
# the arrays the kernel receives (`layers/common/attention_interface.py:540`),
# which pins each field to a serve flag:
#
#   max_num_tokens          <- the compiled token bucket  => 4096
#   actual_num_q_heads      <- 96 heads / TP              => TP=8
#   kv_dtype, q_dtype       <- KV cache dtype             => auto (bf16)
#   page_size_per_kv_packing<- block_size / kv_packing    => --block-size 256
#   max_num_seqs            <- --max-num-seqs             => 8
#   pages_per_seq           <- ceil(max_model_len/block)  => 32768/256 = 128
#
# The shipped `kimi-k3-sliced8-tp8-ep-conc` misses the same row on exactly
# three of those (2048, 16, 36) -- BK #306 logged `6 distinct keys MISSED,
# 0 hit` at four block sizes. If this config still misses, the miss is not a
# configuration gap and that is the more interesting answer.
#
# Read the verdict from the server log:
#   grep 'MLA v2 tuned params:' server.log
MODEL="moonshotai/Kimi-K3-sliced8"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=5
RANDOM_RANGE_RATIO="0.0"

# Env-overridable so an A/B can move ONE key field and leave the rest alone:
# 32768 -> pages_per_seq 128 (hit), 33024 -> 129 (miss), same workload.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[16,4096]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
