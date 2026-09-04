#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# [branch-local, not for merge] Kimi-K3 on the v7x-32 pod, server geometry
# for the InferenceX AgentX trace replay (multi-turn agentic-coding
# sessions, 100k+ token contexts, heavy prefix reuse). Derived from
# kimi-k3-tp32-ep.sh; differences:
#   - long context: MAX_MODEL_LEN default 131072 (env-overridable by the
#     driver); the aiperf client filters traces above it
#   - prefix caching via ENABLE_PREFIX_CACHING (driver --apc). On the hybrid
#     KDA+MLA model it needs --mamba-cache-mode align AND the unified KV pool
#     (TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1: K3 is not in
#     POOLED_GDN_ARCHITECTURES, and #546's pooled Kimi KDA op is not wired
#     into kimi_k3/attention.py yet) — expected to fail, kept as the probe
#   - throughput regime: max-num-seqs = 2 x concurrency, batched tokens 8192
#   - gpu-memory-utilization 0.85 (0.7 leaves ~3 GiB/rank of KV = ~100k
#     tokens, far too little for concurrent 128k contexts)
#   - tool-call + reasoning parsers so the chat endpoint accepts the
#     tool_calls / tool-result messages in the Claude Code traces
#   - the built-in harness bench is a tiny warmup (conc 1, 4 prompts at
#     8k/1k); the real replay is driven by k3_agentx.sh against the
#     kept-alive server
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
# Prefix caching on the hybrid needs the align cache mode (tpu_platform.py
# gate); it is meaningless without prefix caching, so add it conditionally.
APC_ARGS=""
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then APC_ARGS="--mamba-cache-mode align "; fi

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} ${APC_ARGS}--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 --compilation-config {\"compile_sizes\":[16,32,64,128,256,512,1024,2048,4096,8192]}"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
