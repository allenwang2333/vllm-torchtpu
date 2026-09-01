#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# gnovack/k3-dev's own K3 TP32 recipe, transcribed for this harness so their
# reported table (TPOT 15.58 ms / TTFT 806 ms at c=1, 8192/1024) can be
# reproduced on our pod.
#
# SOURCE, not guesswork: scripts/k3/compile_flag_sweep.sbatch (their file, on
# this branch) pins MAX_MODEL_LEN 10240 / MAX_NUM_SEQS 32 / MNBT 8192 /
# ASYNC_SCHEDULING 1 / VLLM_USE_BREAKABLE_CUDAGRAPH 0 / KDA_FUSED 1 /
# UNIFIED_BLOCK_POOL 0 / ENABLE_PREFIX_CACHING 0 / RAY_memory_usage_threshold
# 0.99 / LOAD_FORMAT dummy / KV_CACHE_DTYPE fp8, and drives
# scripts/k3/launch_kimi_k3_tp32_ep.sh, which supplies the rest of the serve
# line. Their client is scripts/k3/concurrency_sweep.sh at INPUT_LEN=8192
# OUTPUT_LEN=1024 PROMPTS_PER_SLOT=6 RANDOM_RANGE_RATIO=1 (exact lengths).
#
# --block-size 16 IS THE POINT, NOT AN OVERSIGHT. Their launcher hardcodes it
# and their 647 tuned MLA rows are keyed on pages_per_seq=640 = 10240/16. Our
# --block-size 256 lever (-42.6% TTFT) would move every key to 40 and make
# their own tuned rows unreachable, i.e. it would silently measure a different
# experiment. Q2 is "does their number reproduce on their code and their
# config"; the page-size lever belongs to the composition lane, not here.
#
# DELIBERATE DEVIATIONS from their serve line, all of them measurement-neutral
# and all of them listed so nobody has to diff two files to find them:
#   --revision                    we serve a staged copy, not an HF snapshot
#   --chat-template               K3 ships none; passing one corrupts the
#                                 prompt (r1) and /v1/completions ignores it
#   --safetensors-load-strategy   inert under runai_streamer and under dummy
#   --enable-prompt-tokens-details  metrics only
#   +--limit-mm-per-prompt        --language-model-only alone was not enough
#                                 to keep K3's vision modality out of the
#                                 multimodal token budget
#   +--model-loader-extra-config  runai host-memory cap; K3 streaming OOMs the
#                                 host without it, and it is inert under dummy
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
# dummy => stage only the checkpoint's metadata and let vLLM randomise the
# weights; runai_streamer (empty) => stream all 1561 GB. THE Q3 VARIABLE.
LOAD_FORMAT="${LOAD_FORMAT:-}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# config.json declares mxfp4-pack-quantized under compressed-tensors, so their
# explicit --quantization compressed-tensors and vLLM's auto-detection resolve
# to the same method. Passed explicitly anyway: fidelity is free here.
QUANTIZATION="compressed-tensors"

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="${K3_CONCURRENCY:-1 8}"
NUM_PROMPTS="${NUM_PROMPTS:-6}"
# Their PROMPTS_PER_SLOT=6 with a floor of 5. At c>=8 that is many minutes of
# decode per rep for no extra precision, so cap each level at one full batch
# (>=5 prompts, which is what resolved 1% in BK #305/#317).
NUM_PROMPTS_BY_CONCURRENCY="${NUM_PROMPTS_BY_CONCURRENCY:-1:6 2:8 4:8 8:8 16:16 32:32}"
# Exact 8192-token prompts, as their --random-range-ratio 1 gives them.
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# tpu_platform.py:908's min_token_size=16 floor applies ONLY when compile_sizes
# is None; an explicit list is taken as given (`sorted(compile_sizes)`), and
# tpu_runner.py:632 makes the smallest entry the decode batch. So their 4 is a
# real pad-4 decode floor on this branch, without our decode-bucket fix.
K3_COMPILE_SIZES="${K3_COMPILE_SIZES:-[4,8,16,32,8192]}"
K3_BLOCK_SIZE="${K3_BLOCK_SIZE:-16}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size $K3_BLOCK_SIZE --compilation-config {\"compile_sizes\":$K3_COMPILE_SIZES}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
