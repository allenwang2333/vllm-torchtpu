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
# --- ARMS -------------------------------------------------------------------
# Same checkpoint, same lm_eval invocation, matched max_num_seqs: the delta
# between arms is the result, and nothing else may move. Arms are selected by
# env because both container drivers forward BUILDKITE_* verbatim and neither
# forwards anything else we could name.
#
#   K1 (lossy): fp8 the two latent projections. bf16 -> float8_e4m3fn is 3
#   mantissa bits against 7, on 184 instructions per step in the latent
#   attention path, so it cannot be recommended on ladder + probe alone.
#
#   kv auto: our perf recipe has run --kv-cache-dtype fp8 from the start with
#   NO K3 accuracy number under it. Without this arm, K1 would stack a second
#   unverified numerics change on an unverified first one, and neither could be
#   attributed. It costs a whole server lifetime; skip it and note the gap
#   rather than dropping an arm that has a hypothesis behind it.
export K3_FP8_LATENT="${K3_FP8_LATENT:-${BUILDKITE_K3_FP8_LATENT:-0}}"
KV_CACHE_DTYPE="${K3_KV_DTYPE:-${BUILDKITE_K3_KV_DTYPE:-fp8}}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"
# --- ACCURACY variant -------------------------------------------------------
# Same serve flags as the perf recipe -- that is the entire point: the question
# is whether the flag set that made K3 3.9x faster on TTFT also changed what the
# model says. The suspect is --kv-cache-dtype fp8 (KV quantisation is the one
# flag here that touches numerics); block size, compile buckets and async
# scheduling should not, and this measures whether "should not" holds.
#
# The perf cell is deliberately tiny (4 prompts at c=1) -- run_eval_flow always
# benches before it evals, and here that pass only needs to prove the server is
# healthy and roughly on its known number (~995 ms / ~19 ms).
#
# lm_eval then runs mmlu_llama and mmlu_pro, --limit 100 each, num_concurrent
# 128 against max_num_seqs 8, so the client queues and the server stays in the
# regime the perf numbers were measured in.
#
# MMLU_PRO_DISABLE_MULTITURN_ARGS=true takes the enable_thinking:false path
# instead of fewshot_as_multiturn + continue_final_message. K3 renders chat
# through the checkpoint's own encoder (auto_map in config.json), and
# continue_final_message is exactly the argument that encoder drops -- the same
# gap #665 fixed for DeepSeek-V4. Until someone verifies K3's encoder honours
# it, multiturn few-shot here would measure the template, not the model.
#
# There is no prior K3 accuracy number at TP32 to compare against: the 0.8179
# mmlu_pro figure in the survey is Kimi-K2.6, a different model. The nearest
# reference is a colleague's GSM8K 0.9644 +/- 0.0051 on real K3 at this
# topology. That is why this build runs a control arm on main's own
# kimi-k3-tp32-ep config: the delta between the two arms is the result, not the
# absolute score.
MMLU_PRO_DISABLE_MULTITURN_ARGS="true"
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
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
MAX_NUM_SEQS=8
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,8,8192]}"

# The run's own environment is the evidence for what it ran with.
echo "[env-audit] eval arm: K3_FP8_LATENT=${K3_FP8_LATENT} KV_CACHE_DTYPE=${KV_CACHE_DTYPE} MAX_NUM_SEQS=${MAX_NUM_SEQS} MODEL=${MODEL}"
