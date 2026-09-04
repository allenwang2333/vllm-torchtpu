#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf benchmark for Kimi-K3 on the multihost TPU v7x-32 pod
# (4 hosts x 8 chips): TP=32 with EP. Single-request latency probe at
# 8k in / 1k out. Baseline calibrated on the pod from two dev-pipeline
# runs with run-to-run spread <= 0.3% on every gated metric — the
# default 5% perf tolerance applies.
# Serve flags below are the set measured by a TP32 A/B campaign on this exact
# rig; each is a flag, none is a source change. Measured on the full 1561 GB
# checkpoint, real weights, 8192/1024, c=1, against this config's own previous
# settings:
#
#   --block-size 256                  -42.6% TTFT   (dev build 311 vs 289)
#   MLA tuned-row geometry            -30.4% TTFT   (dev build 317)
#   --async-scheduling                -24.2% TPOT   (dev build 327)
#   compile_sizes[0] = 1              -21.0% TPOT   (dev build 338)
#
#   together: TTFT 4038.1 -> 995.2 ms (-75.4%), TPOT 30.899 -> 18.954 ms
#   (-38.7%), output 28.7 -> 50.2 tok/s (1.75x). Dev build 365, 3 repeats,
#   spread 0.26% TTFT / 0.01% TPOT.
#
# The MLA tuned row is the fragile part. kernels/mla/v2/tuned_params.py:1169 is
# keyed on the exact tuple below, and four flags here set it:
#   max_num_tokens 8192      <- MAX_NUM_BATCHED_TOKENS 8192 + compile size 8192
#   actual_num_q_heads 3     <- 96 / TP32
#   kv_dtype float8_e4m3fn   <- KV_CACHE_DTYPE fp8
#   page_size_per_kv_packing <- block_size 256 / kv_packing 4 = 64
#   max_num_seqs 8           <- MAX_NUM_SEQS 8
#   pages_per_seq 40         <- ceil(MAX_MODEL_LEN 10240 / 256)
# Change any one of max-model-len, block-size, kv-cache-dtype or max-num-seqs
# and the row stops matching: the kernel falls back silently and TTFT returns
# to ~2050 ms. There are exactly two K3 TP32 rows in that table and both are
# keyed max_num_seqs=8, which is also why this config cannot be run above c=8.
#
# NOT enabled: TPU_MOE_HIERARCHICAL_EP=1, worth a further -7.3%..-9.6% TPOT.
# On the runai_streamer path it serves a model whose expert weights are never
# initialised unless the loader realignment fix is applied, and that fix is not
# on main. A fast number from a broken model is worth less than no number.
MODEL="moonshotai/Kimi-K3"
# v7x-32 hosts have no data disk that fits the 1561 GB checkpoint, so
# weights stream directly from GCS via --load-format runai_streamer.
# --enable-ep-weight-filter (model_loader_patches.py, #373) keeps that
# tractable: each rank streams only the byte ranges of its own experts
# (28/896 at EP32) plus dense weights, instead of the full checkpoint —
# full-checkpoint streaming accumulated >1 TB of host anonymous memory
# and drew Ray OOM kills (dev builds 82/83/85).
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes (compressed-tensors).
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 64 sequential bs=1 requests (~42 s each, ~45 min of bench time) give a
# stabler median for the gated metrics; checkpoint stream + compile still
# dominate the step's wall clock. check_regression.py fails a cell when
# completed < 0.95 x baseline completed, so the floor is 60.8 and up to 3
# transient request failures do not fail the nightly on the single
# tpu_v7x_32 agent.
NUM_PROMPTS=64

# Disable random length jitter: benchmark requests use exact nominal
# 8192:1024 lengths, so TTFT/TPOT medians are comparable run to run.
RANDOM_RANGE_RATIO="0.0"

# Server configuration (mirrors the proven multihost bring-up recipe):
# - max-model-len=9216: exact nominal length for 8192 in + 1024 out.
# - max-num-batched-tokens=512 / max-num-seqs=1: bs=1 latency regime.
# 10240, not 9216: pages_per_seq = ceil(max_model_len / block_size) must be 40
# for the tuned MLA row below. Still covers 8192 in + 1024 out.
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

# Startup budget covers the GCS stream plus a cold XLA compile
# (VLLM_DISABLE_COMPILE_CACHE=1 in the multihost containers).
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
# The proven recipe serves with vllm's default KV cache dtype and
# synchronous scheduling; both are recorded in the run's config.json.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# Kimi-K3 is an MLA model; ATTENTION_BACKEND default CUSTOM applies.

# Serve flags:
# --trust-remote-code: the HF repo ships custom code (TikTokenTokenizer).
# --language-model-only + --limit-mm-per-prompt: K3 is image-text-to-text;
#   serve just the LM (the vision tower has no TPU path here).
# --model-loader-extra-config memory_limit=16GiB: caps the runai streamer's
#   per-rank host buffer so 32 ranks fit host RAM.
# --compilation-config compile_sizes [16,512]: the only token-count
#   buckets this bench reaches — prefill chunks are exactly 512 (8192 is
#   a multiple of max-num-batched-tokens, no remainder) and bs=1 decode
#   pads to 16. 64/256 were compiled cold every run (~5.6 min on the pod)
#   and never executed.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,8,8192]}"

# The bench client tokenizes from the dir the server pulled from GCS
# (run_benchmarks.sh derives --tokenizer from MODEL_URI); the harness
# passes --trust-remote-code to the client unconditionally, which covers
# the custom tokenizer code in that dir.

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
