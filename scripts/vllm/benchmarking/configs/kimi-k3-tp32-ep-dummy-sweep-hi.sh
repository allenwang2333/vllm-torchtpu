#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# A/B twin of kimi-k3-tp32-ep-best: same hardware, same image, same serve
# flags, same benchmark — RANDOM WEIGHTS instead of the 1561 GB checkpoint.
#
# Why run it: every "dummy weights" perf number in this repo's history is
# quoted as if it were a stand-in for the real thing, and lane E's v7x-8
# measurement says the opposite at max-num-batched-tokens 8192 (dummy was
# SLOWER on prefill). This pins the size and sign of that bias at TP32 on the
# pod, so future dummy-weight results can be corrected instead of discarded.
#
# Exactly two flags differ from the real arm, both forced by vLLM:
#
#   1. --load-format dummy                        (the variable under test)
#   2. --model-loader-extra-config is DROPPED     DummyModelLoader.__init__
#      raises ValueError("Model loader extra config is not supported for load
#      format dummy") — vllm/model_executor/model_loader/dummy_loader.py:27.
#      It only sized the streamer's host buffer, which no longer exists.
#
# The model is served by HF repo id, not by the gs:// URI, for a hard reason:
# VllmConfig.__post_init__ rejects dummy against object storage —
#   "To load a model from object storage (S3/GCS/Azure), 'load_format' must be
#    'modelexpress', 'runai_streamer' or 'runai_streamer_sharded'"
# (vllm/config/vllm.py, v0.27.0). Staging the metadata to a local directory is
# the other way round it, and that is what lost all seven cells of dev build
# 352: staging runs in the head container only, so the worker hosts have no
# such path and HF reads it as a repo id. The hub id exists in every container
# and needs no local path at all. moonshotai/Kimi-K3's config.json on the hub
# is byte-identical to the copy in gs://tpu-commons-ci/moonshootai/kimi/k3
# (verified 2026-09-03, 7006 bytes both), so both arms build the same model.
# SWEEP VARIANT (c=16,32) of kimi-k3-tp32-ep-best-dummy.
#
# max_num_seqs 8 -> 32, and that is a REAL COST, not a free knob: the MLA tuned
# row (tuned_params.py:1169) is keyed on max_num_seqs=8 and no K3 TP32 row
# exists for any other value -- the table holds exactly two, both mns=8. These
# two cells therefore run the fallback tiling, and their TTFT is NOT comparable
# to the c<=8 cells from the -lo config (a tuned-row miss measured ~2050 ms
# against ~1435 ms hit at c=1). They are comparable to each other.
#
# Concurrency above max_num_seqs does not raise concurrency, it just queues, so
# measuring c=16/32 requires this. compile_sizes covers every decode batch size
# up to 32, for the bisect_left reason spelled out in the -lo config.
MODEL="moonshotai/Kimi-K3"
# Deliberately empty — see above. Not an oversight.
MODEL_URI=""
LOAD_FORMAT="dummy"
# No runai pull happens, so the client tokenizes from the hub copy.
BENCH_TOKENIZER="moonshotai/Kimi-K3"
# Random weights are degenerate by construction; the probe exists to catch
# UNINTENDED garbage (dev build 299) and would only ever fire here.
BENCHMARK_SANITY_PROBE=0

TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="16 32"
NUM_PROMPTS=64
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=32
# No 1561 GB stream, so bring-up is compile-bound rather than IO-bound.
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
KV_CACHE_DTYPE="fp8"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --block-size 256 --compilation-config {\"compile_sizes\":[1,2,4,8,16,32,8192]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
