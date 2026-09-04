#!/usr/bin/env bash
# Bring up the official Qwen3.8-2.4T-A95B config with eight transformer
# layers, dummy weights, and single-host PCP=8. The GDN head geometry is
# reduced to K/Q=8 and V=32 so each PCP rank gets K/Q=1 and V=4, matching the
# worst-case rank-local head allocation planned for the 32-rank deployment.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

MODEL_DIR="${MODEL_DIR:-/home/liubinhao.lbh/models/Qwen3.8-2.4T-A95B-FP8}"
SERVER_BIN="${VLLM_BIN:-vllm}"
# Avoid exposing the launcher-only VLLM_BIN variable to vLLM's strict
# environment-variable scanner.
unset VLLM_BIN
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"

# Keep the Hugging Face files pristine: all smoke-test pruning and GDN head
# overrides are applied at runtime. Eight layers select exactly two
# [linear_attention x3, full_attention] cycles. Full-attention heads and MoE
# experts keep the official model geometry.
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"
LINEAR_NUM_KEY_HEADS="${LINEAR_NUM_KEY_HEADS:-8}"
LINEAR_NUM_VALUE_HEADS="${LINEAR_NUM_VALUE_HEADS:-32}"
if [[ -z "${HF_OVERRIDES:-}" ]]; then
  HF_OVERRIDES="{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"linear_num_key_heads\":${LINEAR_NUM_KEY_HEADS},\"linear_num_value_heads\":${LINEAR_NUM_VALUE_HEADS}}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
# This is the global scheduler budget. PCP8 compiles the corresponding
# rank-local 512-token bucket (4096 global tokens).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# This is a KV-cache page size, not the PCP attention query tile. 2048 is a
# conservative smoke-test default that matches max_model_len and the split
# Mamba manager page; it is not a model or kernel requirement. It must remain
# divisible by the configured PCP KV interleave (768 is also legal here).
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
# Keep the VMEM bring-up tuning scoped to this launch. With a 256-token query
# block the kernel's default q_compute_size is also 256.
PCP_Q_BLOCK_SIZE="${PCP_Q_BLOCK_SIZE:-256}"
# Keep the public PCP token ownership/layout at the production-like value.
PCP_KV_INTERLEAVE_SIZE="${PCP_KV_INTERLEAVE_SIZE:-256}"
# With the reduced rank-local GDN geometry, the fused projection/A2A/GDN
# staging chunk fits at the full public interleave size.
PCP_GDN_COMM_CHUNK_SIZE="${PCP_GDN_COMM_CHUNK_SIZE:-256}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"pcp_streaming_rpa_q_block_size\":${PCP_Q_BLOCK_SIZE},\"pcp_gdn_comm_chunk_size\":${PCP_GDN_COMM_CHUNK_SIZE}}}"
# The default directory is the official serialized-FP8 metadata, matching the
# Qwen3.5 daily launch while dummy loading avoids all weight downloads. Its
# config still declares BF16 model/activation dtype and controls exactly which
# modules contain FP8 weights. To exercise online quantization instead, point
# MODEL_DIR at the BF16 metadata directory and set WEIGHT_QUANTIZATION=fp8.
WEIGHT_QUANTIZATION="${WEIGHT_QUANTIZATION:-config}"
QUANTIZATION_ARGS=()
if [[ "$WEIGHT_QUANTIZATION" != "config" && "$WEIGHT_QUANTIZATION" != "none" ]]; then
  QUANTIZATION_ARGS+=(--quantization "$WEIGHT_QUANTIZATION")
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: official Hugging Face config not found: $MODEL_DIR/config.json" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: official Hugging Face tokenizer not found: $MODEL_DIR/tokenizer.json" >&2
  exit 1
fi

# TPU_SKIP_MDS_QUERY is intentionally unconditional. Supplying the metadata
# values explicitly keeps startup independent of the GCE metadata service.
export TPU_SKIP_MDS_QUERY="true"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export TPU_NAME="${TPU_NAME:-$(hostname -s)}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-torchtpu}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# All model metadata is local. No weight shard is needed by the dummy loader.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/qwen38-pcp8}"
export TPU_LOG_DIR="${TPU_LOG_DIR:-$RUNTIME_ROOT/tpu-logs}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RUNTIME_ROOT/cache/vllm}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$RUNTIME_ROOT/cache/xla}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUNTIME_ROOT/cache/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/cache/xdg}"
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TPU_LOG_DIR" "$VLLM_CACHE_ROOT" "$VLLM_XLA_CACHE_PATH" \
  "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
# The TPU driver writes logs as its service account.
chmod 1777 "$TPU_LOG_DIR"

export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-0}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG="${TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG:-1}"
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-0}"
# Keep the original GMM MoE path for this correctness-first bring-up.
export USE_MOE_FUSED_EP_KERNEL=0
export TPU_MOE_COLLECTION_CHUNK_SIZE="${TPU_MOE_COLLECTION_CHUNK_SIZE:-16384}"
# The first request lazily compiles the PCP Pallas programs. Give that compile
# more than vLLM's five-minute default RPC deadline.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS="${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}"
unset TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT
unset TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT
export TORCH_TPU_TIER2_COMPILATION_CACHE=disabled
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled

# Keep exercising the attention-shaped unified pool required by deployment.
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1

exec "$SERVER_BIN" serve "$MODEL_DIR" \
  --hf-overrides "$HF_OVERRIDES" \
  --served-model-name Qwen3.8-2.4T-A95B-8L-FP8-dummy \
  --generation-config vllm \
  --seed 42 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size "$BLOCK_SIZE" \
  --additional-config "$ADDITIONAL_CONFIG" \
  --compilation-config '{"compile_sizes":[512]}' \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size "$PCP_KV_INTERLEAVE_SIZE" \
  --attention-backend CUSTOM \
  --kv-cache-dtype fp8 \
  "${QUANTIZATION_ARGS[@]}" \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --language-model-only \
  --load-format dummy \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
