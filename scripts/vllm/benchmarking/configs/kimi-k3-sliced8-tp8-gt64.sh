#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# IN-SITU per-op cost of the routed gmm at TPU_GMM_DECODE_BUCKET_ROWS=64.
#
# W6 measured the m-bucket curve in a kernel microbenchmark and found buckets
# 32 and 64 sitting ~25% above the floor that 16, 128 and 256 share, with no
# mechanism. If that step is an artifact of back-to-back identical calls in a
# microbenchmark it will not appear in a served decode step, and the lever is
# worth nothing. Three profiled server arms in one build settle it: the routed
# gmm's per-op self time is read straight out of the trace, in situ, at K3's
# real per-rank shapes. The gmm's m-axis geometry is identical at TP8 and TP32
# (`m_256 ... tm_256` in both, only `g` differs, per W2's HLO read), so the
# surrogate is the right rig and it is free.
#
# Rehearsal of kimi-k3-tp32-ep-w0profile on the 8-layer surrogate.
#
# Its numbers describe the surrogate and gate nothing. Its job is to prove
# three mechanisms before the single v7x-32 agent is spent:
#   (a) the phased profiler still classifies DECODE_ONLY under
#       --async-scheduling. Nobody has run that combination: the classifier
#       reads the scheduler's own batch composition, and async changes when
#       the scheduler commits a step. If DECODE_ONLY stops being decode-only
#       by construction, the pod capture would be worthless.
#   (b) --block-size 256 + fp8 KV + mnbt 8192 survive the profiler wiring.
#   (c) with the expert-load dump OFF, ops/step/core equals the count from a
#       build that does not contain the dump code at all.
MODEL="moonshotai/Kimi-K3-sliced8"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

PHASED_PROFILER_ARGS='--additional-config={"phased_profiler_phases":"decode_only","phased_profiler_decode_only_kv_len_threshold":8200} --profiler-config.torch_profiler_dir /perf_eval_results/kimi-k3-sliced8-tp8-gt64/profile --profiler-config.max_iterations 8'
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[16,8192]} ${PHASED_PROFILER_ARGS}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-1}"
