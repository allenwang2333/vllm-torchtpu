#!/usr/bin/env bash
# Launch Kimi-K3 with TP32 + expert parallelism on an existing Ray cluster.
# The caller is responsible for starting Ray and provisioning the TPU hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
COMPILE_SIZES="${COMPILE_SIZES:-1}"
PROF_DIR="${PROF_DIR:-${REPO_DIR}/results/kimi-k3-profile}"
TP_SIZE="${TP_SIZE:-32}"
DP_SIZE="${DP_SIZE:-1}"

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export TPU_MULTIHOST_BACKEND=ray
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
# Slurm TPU VMs do not expose the GCE accelerator-type metadata endpoint.
# Make the v7x family explicit so env_override.py selects the latency-oriented
# TensorCore collective thresholds instead of silently keeping libtpu defaults.
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x-32}"
export JAX_PLATFORMS=""
export VLLM_TARGET_DEVICE=tpu
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-10800}"
# vLLM 0.27 auto-enables breakable CUDA graphs for Kimi and, as a side
# effect, disables the whole-model compile path. TPU needs the latter for the
# static compile_sizes=[1] decode executable.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# K3 can concatenate the routed-expert latent and shared-expert partials, then
# pay for one cross-host all-reduce instead of two per MoE layer.
export TPU_K3_FUSED_MOE_COMBINE="${TPU_K3_FUSED_MOE_COMBINE:-0}"
export TPU_K3_HOST_REPLICATED_SHARED_EXPERTS="${TPU_K3_HOST_REPLICATED_SHARED_EXPERTS:-0}"
export TPU_K3_EARLY_SHARED_ALL_REDUCE="${TPU_K3_EARLY_SHARED_ALL_REDUCE:-0}"
export TPU_MOE_HIERARCHICAL_EP="${TPU_MOE_HIERARCHICAL_EP:-1}"
export TPU_MOE_HIERARCHICAL_TP_SIZE="${TPU_MOE_HIERARCHICAL_TP_SIZE:-2}"
export TPU_K3_BLOCKED_TOPK="${TPU_K3_BLOCKED_TOPK:-0}"
export TPU_K3_JAX_EXACT_TOPK="${TPU_K3_JAX_EXACT_TOPK:-0}"
export TPU_K3_PALLAS_ROUTER_TOPK="${TPU_K3_PALLAS_ROUTER_TOPK:-1}"
export TPU_K3_PALLAS_BLOCK_ROUTER_TOPK="${TPU_K3_PALLAS_BLOCK_ROUTER_TOPK:-0}"
export TPU_K3_RANK16_ROUTE_PERMUTE="${TPU_K3_RANK16_ROUTE_PERMUTE:-1}"
export TPU_K3_SHARD_LATENT_UP_PROJ="${TPU_K3_SHARD_LATENT_UP_PROJ:-0}"
export TPU_K3_LATENT_UP_TP_SIZE="${TPU_K3_LATENT_UP_TP_SIZE:-0}"
export TPU_K3_SHARD_LATENT_DOWN_PROJ="${TPU_K3_SHARD_LATENT_DOWN_PROJ:-1}"
export TPU_K3_LATENT_DOWN_TP_SIZE="${TPU_K3_LATENT_DOWN_TP_SIZE:-4}"
export TPU_K3_SHARD_KDA_F_A_PROJ="${TPU_K3_SHARD_KDA_F_A_PROJ:-1}"
export TPU_K3_KDA_F_A_TP_SIZE="${TPU_K3_KDA_F_A_TP_SIZE:-0}"
export TPU_K3_OVERLAP_KDA_F_A_REDUCE="${TPU_K3_OVERLAP_KDA_F_A_REDUCE:-0}"
export TPU_K3_OVERLAP_LATENT_DOWN_ROUTER="${TPU_K3_OVERLAP_LATENT_DOWN_ROUTER:-0}"
export TPU_K3_SHARD_MLA_A_PROJ="${TPU_K3_SHARD_MLA_A_PROJ:-1}"
export TPU_K3_MLA_A_TP_SIZE="${TPU_K3_MLA_A_TP_SIZE:-4}"
export TPU_K3_SHARD_ATTN_RES_PROJ="${TPU_K3_SHARD_ATTN_RES_PROJ:-0}"
export TPU_K3_FUSE_ATTN_RES_DOT="${TPU_K3_FUSE_ATTN_RES_DOT:-0}"
export TPU_K3_FUSE_LM_HEAD_SAMPLER="${TPU_K3_FUSE_LM_HEAD_SAMPLER:-0}"
export TPU_K3_DISTRIBUTED_NEUTRAL_SAMPLER="${TPU_K3_DISTRIBUTED_NEUTRAL_SAMPLER:-0}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
# Reduce launch synchronization overhead on this collective-heavy decode
# graph. Preserve an explicit caller choice (including `=false`).
if [[ " ${LIBTPU_INIT_ARGS:-} " != *" --xla_tpu_use_enhanced_launch_barrier="* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_use_enhanced_launch_barrier=true"
fi
# Let XLA schedule serial collective dependencies as a layer-level critical
# path. This is a small but repeatable batch-1 win on K3's latent-MoE graph.
# Preserve an explicit caller choice for isolation/control experiments.
if [[ " ${LIBTPU_INIT_ARGS:-} " != *" --xla_tpu_enable_layer_scheduler_for_dependent_collectives="* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
fi
# Merge compatible asynchronous collective work in the backend schedule. This
# lowers the batch-1 K3 collective critical path without changing HLO-visible
# collective count. Preserve an explicit caller choice for control runs.
if [[ " ${LIBTPU_INIT_ARGS:-} " != *" --xla_tpu_enable_async_collective_merger="* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_enable_async_collective_merger=true"
fi
# Use both directions of the TPU interconnect for eligible all-reduces. This
# specifically shortens K3's repeated TP4 bf16[3584] reductions at batch 1.
# Preserve an explicit caller choice for isolation/control experiments.
if [[ " ${LIBTPU_INIT_ARGS:-} " != *" --xla_tpu_decompose_all_reduce_bidirectional_communication="* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_decompose_all_reduce_bidirectional_communication=true"
fi

mkdir -p "${PROF_DIR}"

exec "${REPO_DIR}/.venv/bin/vllm" serve "${MODEL}" \
  --revision "${REVISION}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --load-format "${LOAD_FORMAT}" \
  --quantization compressed-tensors \
  --safetensors-load-strategy lazy \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size 16 \
  --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES}]}" \
  --attention-backend CUSTOM \
  --trust-remote-code \
  --language-model-only \
  --no-enable-prefix-caching \
  --async-scheduling \
  --gpu-memory-utilization 0.82 \
  --port "${PORT}" \
  --profiler-config.profiler=torch \
  "--profiler-config.torch_profiler_dir=$PROF_DIR" \
  --profiler-config.delay_iterations=3 \
  --profiler-config.max_iterations=15 \
  --profiler-config.ignore_frontend=true \
  "$@"
