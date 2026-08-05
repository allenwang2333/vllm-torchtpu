#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf sweep for GLM-5.2-FP8 with TP=16, EP on TPU v7x-16.
# Serve flags follow scripts/vllm/benchmarking/glm_cmds.yaml (glm branch):
# TP=16, EP, fp8 KV cache, safetensors prefetch. The perf cell is 8k in /
# 1k out at
# concurrency 64/256/512 to track latency (TTFT/TPOT) and throughput.
# Throughput/chip = throughput / 16 (TPxDP recorded in config.json).
MODEL="zai-org/GLM-5.2-FP8"
# v7x-16 hosts have no attached disk, so weights stream from GCS via
# --load-format runai_streamer. The checkpoint lives in tpu-commons-ci
# (same bucket tpu-inference streams DeepSeek/Kimi from — the CI compute SA
# has the storage.buckets.get permission runai's GCS client needs there).
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/glm/GLM-5.2-FP8}"
TENSOR_PARALLELISM=16
DATA_PARALLELISM=1
# PROBE(build 55): EP off — multihost EP sub-group creation crashes the TPU
# program ("continuator halted") at init_model_parallel_group; TP-only
# isolates it. Restore to true once EP-on-multihost is fixed.
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes (glm_cmds.yaml passes
# none either).
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="64 256 512"
NUM_PROMPTS=64
# 0.8 in benchmark_serving.py/inferenceX semantics: sample lengths in
# [0.8*len, len]. RANGE_RATIO_STYLE=min makes the runner translate this for
# vllm bench serve (whose native 0.8 would mean [0.2*len, 1.8*len] and
# overflow max-model-len). Max sampled input+output = nominal 9216 tokens,
# so every request fits and runs complete fully.
RANDOM_RANGE_RATIO="0.8"
RANGE_RATIO_STYLE="min"
# glm_cmds.yaml serves with 32768 (sized for lm_eval's 30k gen toks); the
# perf cell needs only 9216, so use the suite-standard 10240 to maximize KV
# blocks.
MAX_MODEL_LEN=10240
# 8192 does not compile: the 8192-token graph alone needs more HBM for XLA
# temporaries than the chip has (98.84G vs 94.74G on v7x, build 66) — no
# gpu-memory-utilization value can fix that. 4096 halves the activation
# working set; chunked prefill covers the 8k benchmark prompts in 2 chunks.
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=512
SERVER_READY_WAIT_MIN=180
# 0.95 leaves too little HBM for XLA compile temporaries: capture_model
# fails with "HLO temporaries (95.10G) exceeds available HBM (94.74G)"
# on the 8192-token compile size (v7x-16, build 64 retry). 0.92 frees
# ~3G/chip of KV-cache budget for the compiler's working set.
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE="fp8"

# GLM 5.2 is an MLA model: tpu_platform.get_attn_backend_cls forces
# FLASH_ATTN_MLA whenever use_mla is set, so the runner's default
# --attention-backend CUSTOM is inert here (glm_cmds.yaml passes no backend
# flag).

# Parser flags only affect the chat endpoint (unused by vllm bench serve's
# /v1/completions path) but keep the server identical to glm_cmds.yaml and
# ready for future lm-eval runs.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45"

# Greedy decoding for bench requests: vllm bench serve otherwise defers to the
# server-side generation_config sampling, measurably slower on decode-heavy
# cells.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
