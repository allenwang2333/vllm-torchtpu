#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# The nightly GLM-5.2-FP8 perf config (TP=1 x DP=16 with EP on a v7x-16) as a
# twin for one v7x-8 host: half the engines and half the routed experts, so
# each chip holds the same 16 experts and each expert receives the same share
# of tokens, at the same batch per engine. It serves the checkpoint
# `perfkit layers build --experts 128` writes from MODEL_URI (a layer run of
# perfkit's serve_e2e builds it): the first decoder layers, each MoE layer's
# first 128 experts, and its router cut to match. What a twin cannot keep is
# the all-to-all's span: eight chips on one host, not sixteen across two.

# shellcheck source=glm-5.2-fp8-dp16-ep.sh
source "$(dirname "${BASH_SOURCE[0]}")/glm-5.2-fp8-dp16-ep.sh"

DATA_PARALLELISM=8
# 64 sequences per engine, as the nightly runs, over 8 engines.
CONCURRENCY_OPTIONS="512"
NUM_PROMPTS=1024
# The server serves a local checkpoint, so the client takes the tokenizer
# from the model's name rather than from a directory the server pulled.
TOKENIZER="${TOKENIZER:-$MODEL}"
