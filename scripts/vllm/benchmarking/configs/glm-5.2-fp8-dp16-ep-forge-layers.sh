#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Written by forge (scorers/serve_e2e.py): glm-5.2-fp8-dp16-ep on its leading layers,
# with the phased profiler on, for one scorer arm.
# shellcheck source=glm-5.2-fp8-dp16-ep.sh
source "$(dirname "${BASH_SOURCE[0]}")/glm-5.2-fp8-dp16-ep.sh"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--hf-overrides {\"indexer_types\":[\"full\",\"full\",\"full\",\"shared\",\"shared\",\"shared\",\"full\"],\"mlp_layer_types\":[\"dense\",\"dense\",\"dense\",\"sparse\",\"sparse\",\"sparse\",\"sparse\"],\"num_hidden_layers\":7} --profiler-config.max_iterations 20 --num-gpu-blocks-override 443"
export USE_PHASED_PROFILER=true
CAPTURE_PROFILE=1
