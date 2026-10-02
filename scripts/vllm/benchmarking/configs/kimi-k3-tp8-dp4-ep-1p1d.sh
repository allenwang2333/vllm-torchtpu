#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_1p1d_dp.sh
# One engine of the Kimi-K3 1P1D pair with data parallelism on both roles:
# TP8 x DP4 x EP32 on each 2x2x4 slice, one DP replica per host, over vLLM's
# own multi-node DP (the mp backend) rather than Ray. Everything else is the
# TP32 pair's (kimi-k3-tp32-ep-1p1d.sh): the cache dtype, prefix caching, one
# prefill chunk to a KV block, the Raiden Stage-3 environment and the per-role
# identities and ports.
#
# K3_1P1D_ROLE, prefill or decode, selects the engine. Sourced by
# .buildkite/kubernetes/k3_1p1d_dp.sh on every host before its `vllm serve`.
# What only a running pod knows stays there: VLLM_HOST_IP, and
# TPU_RAIDEN_ADVERTISE_HOST, which is the role's four host addresses in DP-rank
# order - one Raiden store per DP replica, on that replica's host (PR #1271).
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp32-ep-1p1d.sh" || return 1

TENSOR_PARALLELISM=8
DATA_PARALLELISM=4
# A TP8 rank holds four times the weights of a TP32 rank; what the connector
# change was validated with.
GPU_MEMORY_UTILIZATION=0.80

# A TP8 rank also holds four times the KDA state, and the engine raises the
# attention block until a block's pages hold it: 648 tokens in bf16, as
# observed. fp8 halves the bytes a token takes in a page, so 1296 (derived; the
# stages script checks the block the engine chose and refuses to run otherwise).
_tp32_chunk="$MAX_NUM_BATCHED_TOKENS"
if [ "$KV_CACHE_DTYPE" = "fp8" ]; then
    MAX_NUM_BATCHED_TOKENS=1296
else
    MAX_NUM_BATCHED_TOKENS=648
fi
_before="$EXTRA_SERVE_ARGS"
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed "s/\"compile_sizes\":\[1,8,${_tp32_chunk}\]/\"compile_sizes\":[1,8,${MAX_NUM_BATCHED_TOKENS}]/")
if [ "$EXTRA_SERVE_ARGS" = "$_before" ]; then
    echo "ERROR: kimi-k3-tp8-dp4-ep-1p1d.sh: the TP32 pair no longer sets compile_sizes [1,8,${_tp32_chunk}]; update the substitution above" >&2
    return 1
fi
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --distributed-executor-backend mp"

# Kimi admission: producer TP here and decode TP here, on both engines.
export TPU_RAIDEN_TRANSFER_PARALLELISM="$TENSOR_PARALLELISM"
export TPU_RAIDEN_DST_SHARDS="$TENSOR_PARALLELISM"

# The rest of what the connector change was validated with (PR #1271).
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_RAIDEN_TRANSFER_NUM_SLOTS=1
export TPU_PARALLEL_PRECOMPILE=1
export TPU_P2P_WAIT_PULL_TIMEOUT=900
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
