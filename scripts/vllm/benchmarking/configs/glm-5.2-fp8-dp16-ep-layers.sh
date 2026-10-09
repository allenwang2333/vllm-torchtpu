#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# A layer run of the nightly GLM-5.2-FP8 perf config: the same server, layout
# and 8k-in/1k-out workload, serving the first 7 of the 78 decoder layers with
# real weights. Seven layers hold every kind of layer the model has: the three
# dense ones, then one period of the MoE region's DSA indexer pattern (three
# layers that reuse the indexer's top-k and one that computes it). The loader
# streams only those layers' tensors: model_loader_patches.py skips a layer
# past the num_hidden_layers override, the MTP layer included.
#
# The phased profiler captures op time per step for each kind of step, and
# scripts/vllm/benchmarking/with_metrics.sh records the server's histogram of
# tokens per engine step, which says how many steps of each kind ran.

# shellcheck source=glm-5.2-fp8-dp16-ep.sh
source "$(dirname "${BASH_SOURCE[0]}")/glm-5.2-fp8-dp16-ep.sh"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--hf-overrides {\"num_hidden_layers\":7,\"indexer_types\":[\"full\",\"full\",\"full\",\"shared\",\"shared\",\"shared\",\"full\"],\"mlp_layer_types\":[\"dense\",\"dense\",\"dense\",\"sparse\",\"sparse\",\"sparse\",\"sparse\"]} --profiler-config.max_iterations 20"
export USE_PHASED_PROFILER=true
CAPTURE_PROFILE=1
