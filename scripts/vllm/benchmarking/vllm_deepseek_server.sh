#!/bin/bash
set -ex

export VLLM_PLUGINS="torchtpu,torchtpu_general"
export VLLM_TP_SIZE="8"
export VLLM_EP_SIZE="8"
export VLLM_FP4_CACHE_DIR="/dev/shm/dsv4-flash-real-fp4"
export MOE_REQUANTIZE_WEIGHT_DTYPE="fp4"
export MOE_REQUANTIZE_BLOCK_SIZE="512"
export REQUANTIZE_BLOCK_SIZE="128"
export NEW_MODEL_DESIGN="1"
export MODEL_IMPL_TYPE="vllm"
export ENABLE_QUANTIZED_MATMUL_KERNEL="1"
export USE_MOE_SPARSE_CORE="0"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export JAX_COMPILATION_CACHE_DIR="/home/patemotter_google_com/workspace/jax_cache"
export TPU_LIBRARY_PATH="/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so"
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1000000000 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1000000000"
export TORCH_DIST_TIMEOUT="1200"
export VLLM_RPC_TIMEOUT="1200000"
export VLLM_NO_USAGE_STATS="1"
export PYTHONPATH="/home/patemotter_google_com/workspace/vllm-torchtpu/src"

MODEL_PATH="/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
PORT=8000

# Clean up any lingering processes
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
rm -rf /tmp/libtpu_lockfile* /tmp/ipc* /tmp/psm* /dev/shm/psm_* 2>/dev/null || true

# Start native vLLM server
/home/patemotter_google_com/uv_venv/bin/vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --block-size 1024 \
    --gpu-memory-utilization 0.58 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --port $PORT > /home/patemotter_google_com/workspace/vllm-torchtpu/scratch/vllm_server.log 2>&1
