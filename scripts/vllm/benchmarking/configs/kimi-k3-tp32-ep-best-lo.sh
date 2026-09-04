#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 on the tpu7x-32 pod (TP32 + EP32, 4 hosts), 8k in / 1k out, REAL
# weights, at the best flag set found by the round 1-4 A/B campaign. Runs on
# stock main: `git diff origin/main -- src/` is empty for this branch, which is
# only true since #539 landed `runai_repull` (before it, every off-head rank
# died resolving the head-local streamer cache dir -- dev build 362).
#
# Same harness and same shape as main's own `kimi-k3-tp32-ep` nightly, so the
# committed baseline is the comparison:
#   baselines/perf/kimi-k3-tp32-ep.baseline.json -> TTFT 4038.1 ms, TPOT 30.899 ms
#
# Flag provenance (dev builds, full 1561 GB checkpoint, same queue and URI):
#   --block-size 256                     -42.6% TTFT   (#311 vs #289)
#   MLA tuned-row geometry               -30.4% TTFT   (#317)
#   --async-scheduling                   -24.2% TPOT   (#327)
#   compile_sizes[0] = 1 (decode pad)    -21.0% TPOT   (#338)
#
# The MLA tuned row is the fragile part. tuned_params.py:1169 is keyed on the
# exact tuple below and every field is set by a flag here:
#   max_num_tokens 8192      <- MAX_NUM_BATCHED_TOKENS 8192 + compile size 8192
#   actual_num_q_heads 3     <- 96 / TP32
#   kv_dtype float8_e4m3fn   <- KV_CACHE_DTYPE fp8
#   kv_packing 4             <- implied by fp8
#   page_size_per_kv_packing <- block_size 256 / 4 = 64
#   max_num_seqs 8           <- MAX_NUM_SEQS 8
#   pages_per_seq 40         <- ceil(MAX_MODEL_LEN 10240 / 256)
# Change any one of those and the row stops matching; the kernel falls back
# silently and TTFT returns to ~2050 ms.
#
# NOT set: TPU_MOE_HIERARCHICAL_EP=1 (-7.3%..-9.6% TPOT, #329). On the
# runai_streamer path it serves a model whose expert weights are never
# initialised unless the loader realignment fix is applied, and that fix is
# still not on main.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the checkpoint self-describes. Forcing it made every rank
# re-resolve the quant config from the model path (dev builds 356/357).
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
RANDOM_RANGE_RATIO="0.0"
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
KV_CACHE_DTYPE="fp8"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"

# --- sweep cell LO: c = 1, 2, 4, 8 -------------------------------------------
# max_num_seqs stays 8, so this cell hits the MLA tuned row and CANNOT go above
# c=8: the table holds exactly two K3 TP32 rows and both are keyed mns=8, and
# concurrency above max_num_seqs only queues. c=16/32 live in the -hi config.
#
# compile_sizes carries every decode batch size this cell can produce.
# `_get_padded_token_len` is a bisect_left, so a 4-token decode step against
# [1,8,8192] would pad to 8 (survivable) and a 16-token one to 8192 (a
# prefill-shaped decode step: ITL 3.28 ms at c=1 -> 63.89 ms at c=8, measured).
#
# NUM_PROMPTS 16 gives c=8 two full waves. Build 360 sent 6 prompts at every
# concurrency and its c>=8 rows all returned one number: 6 in flight, 3 times.
CONCURRENCY_OPTIONS="1 2 4 8"
NUM_PROMPTS=16
MAX_NUM_SEQS=8
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,2,4,8,8192]}"
