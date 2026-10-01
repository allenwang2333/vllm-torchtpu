#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_1p1d_dp.sh
# One engine of the Kimi-K3 1P1D pair with data parallelism on both roles:
# TP8 x DP4 x EP32 on each 2x2x4 slice, one DP replica per host, served by
# vLLM's own multi-node DP (mp backend) rather than Ray. Everything else is the
# TP32 pair's (kimi-k3-tp32-ep-1p1d.sh): bf16, prefix caching, chunk == block,
# the Raiden Stage-3 environment and the per-role identities and ports.
#
# K3_1P1D_ROLE, prefill or decode, selects the engine. Sourced by
# .buildkite/kubernetes/k3_1p1d_dp.sh on every host before its `vllm serve`.
# What only a running pod knows stays there: VLLM_HOST_IP and
# TPU_RAIDEN_ADVERTISE_HOST, which here is the role's four host addresses in
# DP-rank order (one Raiden reshard store per DP replica, on its own host).
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp32-ep-1p1d.sh" || return 1

TENSOR_PARALLELISM=8
DATA_PARALLELISM=4
# As the prefill-DP validation of the connector change (PR #1271).
GPU_MEMORY_UTILIZATION="${K3_DP_GPU_MEMORY_UTILIZATION:-0.80}"

# At TP8 a rank holds four times the KDA state of a TP32 rank, and the engine
# raises the attention block until a block's pages hold it. One prefill chunk
# must still be one block (see kimi-k3-tp32-ep-1p1d.sh); the stages script
# checks the block the engine chose against this and refuses to run otherwise.
K3_DP_BLOCK_TOKENS="${K3_DP_BLOCK_TOKENS:-648}"
MAX_NUM_BATCHED_TOKENS="$K3_DP_BLOCK_TOKENS"
# K3_DP_PROFILE=1: an xprof capture of a benchmark rather than the correctness
# stages, so the prefill chunk is the nightly's 8192 and the server carries
# the profiler, with its traces under this role's run directory.
if [ "${K3_DP_PROFILE:-0}" = "1" ]; then
    MAX_NUM_BATCHED_TOKENS=8192
fi
_before="$EXTRA_SERVE_ARGS"
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed "s/\"compile_sizes\":\[1,8,256\]/\"compile_sizes\":[1,8,${MAX_NUM_BATCHED_TOKENS}]/")
if [ "$EXTRA_SERVE_ARGS" = "$_before" ]; then
    echo "ERROR: kimi-k3-tp8-dp4-ep-1p1d.sh: the TP32 pair no longer sets compile_sizes [1,8,256]; update the substitution above" >&2
    return 1
fi
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --distributed-executor-backend mp"
if [ "${K3_DP_PROFILE:-0}" = "1" ]; then
    # As run_benchmarks.sh arms it; ignore_frontend skips the AsyncLLM CPU
    # profiler, only the TPU xplane is read.
    EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --profiler-config.profiler torch --profiler-config.torch_profiler_dir /tmp/k3_1p1d/${K3_1P1D_ROLE}/profile --profiler-config.ignore_frontend true"
fi

# Kimi admission: producer TP here, decode TP here, on both engines.
export TPU_RAIDEN_TRANSFER_PARALLELISM="$TENSOR_PARALLELISM"
export TPU_RAIDEN_DST_SHARDS="$TENSOR_PARALLELISM"

# The rest of the environment the connector change was validated with.
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_RAIDEN_CLIENT_IMPL=cpp
export TPU_RAIDEN_TRANSFER_NUM_SLOTS=1
export TPU_P2P_WAIT_PULL_TIMEOUT=900
export TPU_PARALLEL_PRECOMPILE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false

# Sample branch only: /reset_prefix_cache for k3_dp_pair_matrix.py.
export VLLM_SERVER_DEV_MODE=1
