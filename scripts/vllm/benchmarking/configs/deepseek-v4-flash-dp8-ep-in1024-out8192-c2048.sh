#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# deepseek-v4-flash-dp8-ep at one shape: 1024 tokens in, 8192 out, 2048 requests at
# once, 2048 requests in total. Made by the forge workload picker.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/deepseek-v4-flash-dp8-ep.sh"
ISL_OSL_CONFIGS="1024:8192"
CONCURRENCY_OPTIONS="2048"
NUM_PROMPTS=2048
