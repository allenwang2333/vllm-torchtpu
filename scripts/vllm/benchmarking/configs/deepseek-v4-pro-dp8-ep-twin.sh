#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# The nightly DeepSeek-V4-Pro perf config (TP=1 x DP=16 with EP on a v7x-16) as
# a twin for one v7x-8 host: half the engines and half the routed experts, so
# each chip holds the same 24 experts and each expert receives the same share
# of tokens, at the same batch per engine. Meant to be served cut to the
# model's leading layers and its first 192 routed experts. What a twin cannot
# keep is the all-to-all's span: eight chips on one host, not sixteen across
# two.
# shellcheck source=deepseek-v4-pro-dp16-ep.sh
source "$(dirname "${BASH_SOURCE[0]}")/deepseek-v4-pro-dp16-ep.sh"
DATA_PARALLELISM=8
DATA_PARALLELISM_LOCAL=8
# 64 sequences per engine, as the nightly runs, over 8 engines.
CONCURRENCY_OPTIONS="512"
NUM_PROMPTS=1024
