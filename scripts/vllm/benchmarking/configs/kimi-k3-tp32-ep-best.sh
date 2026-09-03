#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 on the tpu7x-32 pod (TP32 + EP32, 4 hosts), 8k in / 1k out, at the
# best flag set found by the round 1-4 A/B campaign. NO SOURCE CHANGES: every
# lever here is a serve flag, so this config measures what stock main can do
# today.
#
# Provenance of each lever (dev pipeline build numbers, full 1561 GB
# checkpoint, real weights, same queue and MODEL_URI):
#
#   --block-size 256                     -42.6% TTFT   (#311 vs #289)
#   MLA tuned-row geometry               -30.4% TTFT   (#317, 6-vs-6, no overlap)
#   --async-scheduling                   -24.2% TPOT   (#327)
#   compile_sizes[0] = 1 (decode pad)    -21.0% TPOT   (#338)
#
#   composed: TTFT 4488.3 -> 1435.9 ms, TPOT 32.09 -> 17.785 ms vs main as
#   found (#289); -69.1% / -50.4% against the PR #539 baseline of 08-20.
#
# The MLA tuned row is the fragile one. tuned_params.py:1169 is keyed on the
# exact tuple below, and every field is set by a flag here:
#   max_num_tokens 8192      <- MAX_NUM_BATCHED_TOKENS 8192 + compile size 8192
#   actual_num_q_heads 3     <- 96 / TP32
#   kv_dtype float8_e4m3fn   <- KV_CACHE_DTYPE fp8
#   kv_packing 4             <- implied by fp8
#   page_size_per_kv_packing <- block_size 256 / 4 = 64
#   max_num_seqs 8           <- MAX_NUM_SEQS 8
#   pages_per_seq 40         <- ceil(MAX_MODEL_LEN 10240 / 256)
# Change any one of max-model-len, block-size, kv-cache-dtype or max-num-seqs
# and the row stops matching; the kernel silently falls back and TTFT returns
# to ~2050 ms.
#
# Deliberately NOT set: TPU_MOE_HIERARCHICAL_EP=1 (worth -7.3%..-9.6% TPOT,
# #329). On the runai_streamer path it loads a model whose expert weights are
# never initialised unless the loader realignment fix is applied, and that fix
# is not on main. A fast number from a broken model is the one outcome worth
# less than no number (see the sanity probe at the end of run_benchmarks.sh,
# and dev build 299).
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the checkpoint self-describes. Forcing it made every rank
# re-resolve the quant config from the model path (#356/#357).
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 10 prompts at c=1, matching .buildkite/pipeline_dev_k3_gcs.yml so the number
# is directly comparable to every historical K3 figure on this rig.
NUM_PROMPTS=10
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
KV_CACHE_DTYPE="fp8"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# compile_sizes: 1 is the decode bucket. Without it the smallest compiled shape
# is 16 and a single-sequence decode step pads 1 token to 16 (-21.0% TPOT,
# #338). 8 covers max-num-seqs, 8192 is the prefill bucket the MLA row needs.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,8,8192]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
