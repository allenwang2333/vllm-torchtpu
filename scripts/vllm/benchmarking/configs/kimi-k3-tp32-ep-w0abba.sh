#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Round-3 W0 leg 3: ABBA A/B of the decode token bucket at the fixed config.
#
# Driven by k3_knob_matrix.sh --arms bucket16,bucket8,bucket8,bucket16 --reps 3,
# so both arms run in ONE job on ONE agent, interleaved. That is what buys 1.0%
# resolution instead of the 2.05% a one-cell-per-arm comparison can reach.
#
# The two arms differ in K3_COMPILE_SIZES and nothing else. Everything else is
# the round's best-known config, so the bucket8 cells' absolute TTFT/TPOT are
# quotable against the #289 nightly baseline (4488.3 ms / 32.09 ms) rather than
# being only a delta.
#
# `kimi-k3-tp32-ep-profile` (BK #316, the only full-model TP32 decode xprof)
# was taken at the nightly config: page 16, KV auto, mnbt 512, mns 1, sync
# scheduling. Three knobs have since each been measured worth double digits,
# so that capture's per-category table is stale as a prioritisation input.
# This is the same capture at the config round 2 ended on.
#
# Differences from kimi-k3-tp32-ep-profile, and what each is expected to move:
#   --block-size 256        pages_per_seq 576 -> 40, page_size_per_kv_packing
#                           8 -> 64. Aims at the MLA decode kernel, which
#                           #316 measured at 108.15 us/layer walking 512 pages
#                           in blocks of num_kv_pages_per_block=3.
#   --kv-cache-dtype fp8    halves the MLA latent cache read; also the field
#                           the MLA v2 tuning table keys on (kv_packing 2 -> 4).
#   mnbt 8192 / mns 8       the prefill-side half of the tuned-row key. Decode
#                           still pads 1 token to 16, so the decode shape is
#                           byte-identical to #316's and the per-category
#                           table stays comparable.
#   --max-model-len 10240   load-bearing: pages_per_seq = 10240/256 = 40 is
#                           what selects MLA tuned row tuned_params.py:1169.
#                           10496 gives 41 and misses.
#   --async-scheduling      NEVER combined with the above on the pod. Changes
#                           WHEN the host issues work, not what the device
#                           does: per-category device us stay comparable to
#                           #316, device-busy/ITL does NOT, because async
#                           shrinks the denominator by design.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# Four prompts, as in #316: the unprofiled warmup run is this build's own ITL
# denominator (4 x 1023 inter-token samples) and the profiled run follows it on
# the same server. The trace is the product, not the throughput number.
# 5 prompts x 3 reps per cell, as lane D's noise study sized it.
NUM_PROMPTS=5
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
# The arm. tpu_runner.py:632 makes the smallest entry the decode batch size.
K3_COMPILE_SIZES="${K3_COMPILE_SIZES:-[16,8192]}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# kv_len_threshold 8200 waits 8 decode steps past the 8192-token prompt so the
# window is steady state, not the first step out of prefill. Both keys are in
# HASH_IGNORED_ADDITIONAL_CONFIG_KEYS, so neither busts the compile cache.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":$K3_COMPILE_SIZES}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-1}"