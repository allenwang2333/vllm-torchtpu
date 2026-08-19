#!/bin/bash
# shellcheck disable=SC2034  # sourced by bench_k3.sh and run_eval_flow.sh
#
# Kimi-K3, TP=32 + EP across 4 hosts (16 chips / 32 devices, topology 2x2x4).
#
# The only single-replica configuration that exists: the 1.42 TiB checkpoint
# needs >=16 chips, and 96 attention heads force 96 % TP == 0, so TP=32 is the
# sole intersection. TP=64 and TP=128 do not divide 96.
#
# run_benchmarks.sh cannot drive this -- it launches one local server. Use
# bench_k3.sh, which adds the multi-host launch and keeps the disciplines.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="gs://torchtitan-assets/moonshootai/kimi/3"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true

# Match the published shape so numbers are comparable: vLLM's K3 blog reports
# batch 1, 8K in / 1K out, on a served endpoint.
RANDOM_RANGE_RATIO="0.0"        # run_eval_flow.sh requires it; 0.0 pins exact lengths
RANGE_RATIO_STYLE="symmetric"
ISL_OSL_CONFIGS="8192:1024"
# --max-num-seqs is a CORRECTNESS ceiling on this model, not a tuning knob:
# above 4 concurrent sequences a stress test produced repeated-token
# degeneration at high KV occupancy. Root cause unknown.
CONCURRENCY_OPTIONS="1 4"
MAX_NUM_SEQS=4
PROMPTS_PER_CONC=5                 # num-prompts = 5 x concurrency, as the other configs use

MAX_MODEL_LEN=9216                 # must hold 8192 in + 1024 out
MAX_NUM_BATCHED_TOKENS=512
COMPILE_SIZES="[16,64,256,512]"
SERVER_READY_WAIT_MIN=40           # weights stream ~5 min, then 4 compile shapes; ready ~1250 s
GPU_MEMORY_UTILIZATION=0.7
ATTENTION_BACKEND="CUSTOM"         # KDA + MLA both need it; leave VLLM_TPU_USE_NAIVE_KDA unset
BENCHMARK_TEMPERATURE="0"          # greedy; otherwise the client inherits server-side sampling
CHAT_TEMPLATE="/tmp/k3_chat.jinja" # not in the repo; staged by the entrypoint

# K3 always builds the MoonViT tower, so text-only serving needs BOTH flags.
# --enable-ep-weight-filter makes each rank fetch only its 28/896 experts.
EXTRA_SERVE_ARGS="--enable-ep-weight-filter --language-model-only --limit-mm-per-prompt={\"image\":0,\"video\":0} --no-enable-prefix-caching"

# Tier 1 accuracy. Protocol is FIXED before calibration; changing it later
# invalidates the baseline silently. run_eval_flow.sh passes
# enable_thinking:false, which we keep for the routine gate -- it is cheaper
# and matches every other config here. Run thinking-on once for the final
# recipe's headline number.
#
# We gate against a baseline we calibrate ourselves. The published 0.976 is
# NOT a target: the blog names no harness and no prompting, so it is not
# reproducible. Treat it as a sanity band.
EVAL_TASKS="gsm8k"
EVAL_TOLERANCE="0.015"
