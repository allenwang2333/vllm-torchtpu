#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Accuracy-shaped sibling of kimi-k3-tp32-ep: same TP=32/EP serve recipe and
# the same GCS EP-sharded stream, but batched for lm_eval instead of the
# bs=1 latency probe. Perf numbers from this config are NOT comparable to
# the kimi-k3-tp32-ep baseline -- it exists so run_eval_flow.sh can hold one
# server up long enough to score it.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

# One short bench row only: the point of this config is the server, and every
# extra row costs another 8k-token prefill on the single v7x-32 agent.
ISL_OSL_CONFIGS="1024:128"
CONCURRENCY_OPTIONS="16"
NUM_PROMPTS=32
RANDOM_RANGE_RATIO="0.0"

# mmlu_llama few-shot prompts run ~2-3k tokens; mmlu_pro CoT answers need
# room to finish. 8192 covers both with the 16-way batch below.
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=32

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

# Do NOT add --chat-template. The checkpoint ships no Jinja template on
# purpose: vLLM maps KimiK3ForConditionalGeneration to
# tokenizer_mode="kimi_k3" (vllm/config/model.py), whose renderer builds chat
# prompts with the checkpoint's own XTML encoder --
# TikTokenTokenizer.apply_chat_template -> encoding_k3.build_chat_segments --
# so structural markers keep their special-token ids. A Jinja override would
# re-tokenize those markers as ordinary text. --trust-remote-code is what
# makes that encoder importable.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184}"

EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

# build_chat_segments takes **kwargs and ignores what it does not know, so
# lm_eval's `continue_final_message` is dropped silently and the
# few-shot-as-multiturn form renders with the final assistant turn CLOSED --
# nothing for the model to continue. Take the single-turn path instead.
#
# thinking defaults to True with thinking_effort="max", so every eval request
# would emit a reasoning block; run_eval_flow.sh / k3_accuracy.sh pass
# chat_template_kwargs {"enable_thinking": false}, which the K3 renderer maps
# onto thinking=False.
MMLU_PRO_DISABLE_MULTITURN_ARGS="true"
LM_EVAL_DISABLE_MULTITURN="true"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
