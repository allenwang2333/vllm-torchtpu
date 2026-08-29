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

# The Kimi-K3 checkpoint ships no chat_template (tokenizer_config.json has
# tokenizer_class=TikTokenTokenizer and no template key), so the OpenAI
# /v1/chat/completions route lm_eval drives has nothing to render with.
# Supply the text-only template explicitly.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --chat-template scripts/kimi_k3_text_chat_template.jinja"

EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

# This template renders each turn as a closed <|open|>message ... <|end_of_msg|>
# block, so lm_eval's default `continue_final_message` few-shot form has no
# open assistant turn to continue. Take the single-turn path instead.
MMLU_PRO_DISABLE_MULTITURN_ARGS="true"
LM_EVAL_DISABLE_MULTITURN="true"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
