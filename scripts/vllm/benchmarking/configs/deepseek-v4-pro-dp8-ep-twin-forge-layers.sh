#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Written by forge (scorers/serve_e2e.py): deepseek-v4-pro-dp8-ep-twin on its leading layers,
# with the phased profiler on, for one scorer arm.
# shellcheck source=deepseek-v4-pro-dp8-ep-twin.sh
source "$(dirname "${BASH_SOURCE[0]}")/deepseek-v4-pro-dp8-ep-twin.sh"
GPU_MEMORY_UTILIZATION="0.5"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--hf-overrides {\"n_routed_experts\":192,\"num_hidden_layers\":5} --profiler-config.max_iterations 20"
export USE_PHASED_PROFILER=true
CAPTURE_PROFILE=1
