#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 DECODE-ONLY device profile: `kimi-k3-tp32-ep-best-lo` verbatim plus
# the in-tree phase-based profiler. Every profile this campaign has taken so
# far is a prefill profile, and prefill is 5% of wall clock at c=1
# (e2e = 0.995 s + 1024 x 18.95 ms). This config produces the other 95%.
#
# Zero source changes: `git diff <main> -- src/` is empty on this branch.
#
# ---------------------------------------------------------------------------
# WHY THE PROFILER IS TURNED ON FROM *HERE* AND NOT FROM THE BUILDKITE STEP
# ---------------------------------------------------------------------------
# `.buildkite/scripts/run_multihost.sh` forwards a fixed `-e` allowlist plus
# BUILDKITE_*; USE_PHASED_PROFILER and CAPTURE_PROFILE are in neither, so a
# build-level `env:` never reaches the container. A config file does, because
# run_benchmarks.sh sources it *inside* the container --
# `configs/deepseek-v4-flash-dp8-ep-profile.sh:32-33` already relies on this.
# From here the exports land in the environment of `vllm serve`, which is the
# Ray driver; on multihost TPU the platform forces RayDistributedExecutorV2
# (tpu_platform.py:1081), whose `get_driver_env_vars` copies the whole driver
# environment onto every worker actor, so all 32 ranks are armed, not just the
# head. The [env-audit] line below is what turns that from an argument into
# evidence in the job log.
#
# ---------------------------------------------------------------------------
# ONE FILE, TWO RIGS
# ---------------------------------------------------------------------------
# K3_RIG=tp32     the real thing: 1561 GB checkpoint, TP32+EP32, 4 hosts,
#                 tpu_v7x_32_queue. This is the measurement.
# K3_RIG=sliced8  the 8-layer surrogate on ONE v7x-8 host. Same profiler
#                 wiring, same phase protocol, same artifact layout, same
#                 analysis script -- a rehearsal on free hardware that proves
#                 the mechanism before the single v7x-32 agent is spent.
#                 Its LEVELS describe the surrogate and gate nothing; what it
#                 validates is the pipeline, not the number.
# Selected by env because both container drivers forward BUILDKITE_* verbatim,
# so the pipeline sets BUILDKITE_K3_RIG and a local gate sets K3_RIG.
K3_RIG="${K3_RIG:-${BUILDKITE_K3_RIG:-tp32}}"
case "$K3_RIG" in
  tp32)
    MODEL="moonshotai/Kimi-K3"
    MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
    TENSOR_PARALLELISM=32
    SERVER_READY_WAIT_MIN=240
    ;;
  sliced8)
    MODEL="moonshotai/Kimi-K3-sliced8"
    MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
    TENSOR_PARALLELISM=8
    SERVER_READY_WAIT_MIN=90
    ;;
  *)
    echo "ERROR: K3_RIG must be 'tp32' or 'sliced8' (got '${K3_RIG}')" >&2
    exit 1
    ;;
esac

DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the checkpoint self-describes. Forcing it made every rank
# re-resolve the quant config from the model path (dev builds 356/357).
QUANTIZATION=""

# --- the -lo recipe, unchanged ------------------------------------------------
# The MLA tuned row is the fragile part. tuned_params.py:1169 is keyed on the
# exact tuple below and every field is set by a flag here:
#   max_num_tokens 8192      <- MAX_NUM_BATCHED_TOKENS 8192 + compile size 8192
#   actual_num_q_heads 3     <- 96 / TP32   (sliced8 runs TP8 -> 12 -> no row;
#                               expected, and the reason its levels are not
#                               comparable to the pod's)
#   kv_dtype float8_e4m3fn   <- KV_CACHE_DTYPE fp8
#   kv_packing 4             <- implied by fp8
#   page_size_per_kv_packing <- block_size 256 / 4 = 64
#   max_num_seqs 8           <- MAX_NUM_SEQS 8
#   pages_per_seq 40         <- ceil(MAX_MODEL_LEN 10240 / 256)
ISL_OSL_CONFIGS="8192:1024"
RANDOM_RANGE_RATIO="0.0"
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.82
KV_CACHE_DTYPE="fp8"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
MAX_NUM_SEQS=8
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

# --- what differs from -lo, and only this -------------------------------------
# c = 1 and c = 8 only. Both are wanted, and run_benchmarks.sh arms the profiler
# once per *main* (non-warmup) run, while tpu_runner.start_phased_profiling
# builds a FRESH PhaseBasedProfiler each time it is armed -- so its
# `inference_phase_seen` map resets and one server lifetime yields a decode-only
# window at c=1 AND one at c=8. c=2/4 are dropped: they add two more bring-up-free
# minutes of pod time and no phase this brief reads.
CONCURRENCY_OPTIONS="1 8"
# 8 prompts: c=8 gets exactly one full wave (all eight in flight, then a long
# pure-decode tail, which is the window), and c=1 gets 8 x 1023 inter-token
# samples for the ITL denominator. NUM_PROMPTS is shared across cells, so this
# is the smallest value that makes the c=8 cell honest.
NUM_PROMPTS=8
# Pass A unprofiled (the denominator and a clean /metrics timeline), pass B with
# the profiler armed: `--profile` costs ~+11% TTFT / -4% throughput, so the
# profiled pass must never be the number anyone quotes.
BENCHMARK_WARMUP_RUNS="${K3_WARMUP_RUNS:-1}"

export USE_PHASED_PROFILER=1
export CAPTURE_PROFILE=1
# Pinned, not inherited. #448 (IntraChipColumnParallelLinear) has merged and is
# dormant by default (envs.py `TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False`,
# latent_proj_intra_chip.py early-returns when unset), and round 4 measured it
# as a REGRESSION at two hosts (+3.90%). Pinning it here puts the value in the
# driver environment, so it is copied to all 32 worker actors and can be read
# back per rank instead of assumed. Set to 1 only in a deliberate A/B arm.
export TPU_LATENT_PROJ_INTRA_CHIP_TP="${TPU_LATENT_PROJ_INTRA_CHIP_TP:-0}"

# WHERE THE TRACES GO, AND WHY IT IS NOT /perf_eval_results ON THE POD.
# On a multihost slice only the HEAD container gets
# `-v ${PERSIST_ROOT}/perf_eval_results:/perf_eval_results`; the worker hosts
# write into a container that run_multihost.sh's cleanup() deletes on exit, so
# a capture under /perf_eval_results silently loses 24 of 32 ranks. The HF cache
# *is* bind-mounted on every host of a Ray slice (run_multihost_ray.sh passes
# `$HOME/hf_home` to run_cluster.sh, which mounts it at /root/.cache/huggingface),
# so pointing the profiler there leaves all four hosts' traces on their own host
# disks, where k3r5_decodeprof_run.sh collects them after the slice is torn down.
# The single-host driver mounts a different HF path and no /root/.cache/huggingface
# at all, but it does bind and copy back /perf_eval_results -- so the surrogate
# keeps the simple path. Per-build subdirectory: nothing cleans the HF cache
# between builds, and two builds' captures in one directory is a wrong answer
# that looks right.
case "$K3_RIG" in
  tp32)    K3_PROFILE_ROOT="/root/.cache/huggingface/k3r5_profile" ;;
  sliced8) K3_PROFILE_ROOT="/perf_eval_results/k3r5_profile" ;;
esac
K3_PROFILE_DIR="${K3_PROFILE_DIR:-${K3_PROFILE_ROOT}/${K3_RIG}/${BUILDKITE_BUILD_NUMBER:-local}}"
# max_iterations is the per-PHASE step budget. 8 decode steps is ~0.15 s, two
# orders under the ~4 s on-device buffer overflow that silently returns a
# capture with zero device events. The same 8 steps of the c=8 prefill wave is
# ~8 s and will drop events -- that window is a by-product, not the deliverable;
# report its dropped_traces rather than reading it. `phased_profiler_phases` does
# NOT exist in this tree (only the two kv_len_threshold keys are in
# HASH_IGNORED_ADDITIONAL_CONFIG_KEYS), so passing it would both do nothing and
# bust the TPU compile cache key.
# decode_only_kv_len_threshold 8200: 8 decode steps past the 8192-token prompt,
# so the window is steady state rather than the first step out of prefill. It is
# hash-ignored, so it does not bust the compile cache.
PHASED_PROFILER_ARGS="--profiler-config.torch_profiler_dir ${K3_PROFILE_DIR} --profiler-config.max_iterations ${K3_PROFILE_MAX_ITERS:-8} --additional-config={\"phased_profiler_decode_only_kv_len_threshold\":8200}"

# Optional passenger arm (default OFF). These two libtpu flags print which
# all-reduce implementation is selected for a 14 KiB 32-way reduction -- the
# single fact that decides whether the collective row is a latency floor or a
# bandwidth cost. They are OFF in the measured arm on purpose: an unknown libtpu
# flag aborts at TPU init, and a flag that changes the compile changes the thing
# being measured. /dev/shm/torch_tpu_cache is NOT keyed on LIBTPU_INIT_ARGS, so
# an arm that sets this must clear it first or it silently runs the untraced
# executable.
if [ "${K3_ALLREDUCE_TRACING:-${BUILDKITE_K3_ALLREDUCE_TRACING:-0}}" = "1" ]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_inference_short_all_reduce_enable_tracing --xla_tpu_enable_all_reduce_tracing"
fi

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,2,4,8,8192]} ${PHASED_PROFILER_ARGS}"

# Printed at source time, inside the container, into the job log: the run's own
# environment is the evidence for what it ran with. Two of us have already been
# wrong about TPU_LATENT_PROJ_INTRA_CHIP_TP by reading the source instead of the
# run.
echo "[env-audit] K3_RIG=${K3_RIG} TP=${TENSOR_PARALLELISM} USE_PHASED_PROFILER=${USE_PHASED_PROFILER} CAPTURE_PROFILE=${CAPTURE_PROFILE} TPU_LATENT_PROJ_INTRA_CHIP_TP=${TPU_LATENT_PROJ_INTRA_CHIP_TP} LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-<unset>} PROFILE_DIR=${K3_PROFILE_DIR}"
