#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# deepseek-v2-tp8-ep (MLA attention, 8 chips, single host) with a large batch,
# for the concurrency probe (issue #845): same model and flags as the nightly
# config, a 2-request smoke bench instead of the 64-way sweep, and the batch /
# context knobs the probe driver exports.
source "$(dirname "${BASH_SOURCE[0]}")/deepseek-v2-tp8-ep.sh"
ISL_OSL_CONFIGS="1024:256"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"
MAX_MODEL_LEN="${K3_SWEEP_CTX:-8192}"
MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--compilation-config {\"compile_sizes\":[1,8,16,32,64,128,256,1024,2048,4096]}${K3_EXTRA_SERVE_ARGS:+ $K3_EXTRA_SERVE_ARGS}"
