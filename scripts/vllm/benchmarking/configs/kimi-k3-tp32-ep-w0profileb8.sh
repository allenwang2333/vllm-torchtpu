#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Round-3 W0: the decode capture at the fixed config + async + a decode token
# bucket of 8 instead of 16. Fires only if BK #325 shows the bucket pays.
#
# `tpu_runner.py:632` sets num_tokens_paddings = compile_sizes, so the decode
# batch pads to the smallest compiled size. `--max-num-seqs 8` makes 8 the
# honest floor on this rig: a step with 8 running sequences needs a bucket that
# can hold 8 tokens, so [1,8192] would be a correctness bug, not a faster
# config. [8,8192] is shippable as it stands.
#
# THREE things now differ from BK #316, not two -- page 256 + fp8 KV, async,
# and the bucket -- so every category movement has to name which one moved it.
# The bucket is the only one that touches the token-proportional categories
# (routing/permute, AttentionResidual, KDA and MLA projections, the routed
# gmm's m); page 256 is the only one that touches the MLA decode kernel; async
# touches none of them and only shrinks the ITL denominator.
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
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# kv_len_threshold 8200 waits 8 decode steps past the 8192-token prompt so the
# window is steady state, not the first step out of prefill. Both keys are in
# HASH_IGNORED_ADDITIONAL_CONFIG_KEYS, so neither busts the compile cache.
PHASED_PROFILER_ARGS='--additional-config={"phased_profiler_phases":"decode_only","phased_profiler_decode_only_kv_len_threshold":8200} --profiler-config.torch_profiler_dir /perf_eval_results/kimi-k3-tp32-ep-w0profileb8/profile --profiler-config.max_iterations 8'
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[8,8192]} ${PHASED_PROFILER_ARGS}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-1}"
