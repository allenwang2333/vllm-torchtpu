#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Runs one host of PP4 x PCP8 prefill or TP32 + DCP8 decode, with real MTP.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
role="${1:?expected prefill, decode or client}"
case "$role" in prefill|decode|client) ;; *) exit 2 ;; esac
MODEL=/ckpt/Qwen3.8-2.4T-A95B-FP8
MODEL_URI=gs://tpu-commons-ci/qwen/Qwen3.8-2.4T-A95B-FP8

# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "$HERE/multihost_lib.sh"
pod_setup

if [[ "$role" == client ]]; then
  : "${PREFILL_HOST:?}" "${DECODE_HOST:?}"
  export TOKENIZERS_PARALLELISM=false
  ARTIFACTS_DIR="${ARTIFACTS_DIR:-perf_eval_results}/qwen38_mtp"
  export ARTIFACTS_DIR
  mkdir -p "$ARTIFACTS_DIR"
  proxy_pid=""
  # shellcheck disable=SC2317  # Invoked by the EXIT trap.
  cleanup() {
    local rc=$?
    trap - EXIT
    [[ -z "$proxy_pid" ]] || kill "$proxy_pid" 2>/dev/null || true
    if ! publish_results && [[ "$rc" == 0 ]]; then rc=1; fi
    exit "$rc"
  }
  trap cleanup EXIT
  python3 -c 'from lm_eval import evaluator; from lm_eval.api.registry import get_model; get_model("local-completions")'
  python3 -c \
    'import runpy; exec(runpy.run_path("scripts/vllm/integration/check_qwen38_mtp.py")["BENCHMARK_MAIN"])' \
    --help >/dev/null
  # Only tokenizer and configuration files are needed by the HTTP client.
  gcloud storage rsync -r --exclude '.*\.(safetensors|bin|pt|pth)$' "$MODEL_URI" "$MODEL"
  deadline=$((SECONDS + 10800))
  for host in "$PREFILL_HOST" "$DECODE_HOST"; do
    until curl -fsS --max-time 5 "http://${host}:8000/health" >/dev/null 2>&1; do
      if (( SECONDS >= deadline )); then echo "Startup timeout: $host"; exit 1; fi
      sleep 10
    done
  done
  python3 examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8100 \
    --prefiller-hosts "$PREFILL_HOST" --prefiller-ports 8000 \
    --decoder-hosts "$DECODE_HOST" --decoder-ports 8000 \
    >"$ARTIFACTS_DIR/proxy.log" 2>&1 &
  proxy_pid=$!
  timeout 120 bash -c 'until curl -fsS --max-time 5 http://127.0.0.1:8100/healthcheck >/dev/null; do sleep 2; done'
  echo SERVERS_READY
  python3 scripts/vllm/integration/check_qwen38_mtp.py \
    --prefill-url "http://${PREFILL_HOST}:8000" --decode-url "http://${DECODE_HOST}:8000" \
    --proxy-url http://127.0.0.1:8100 --tokenizer "$MODEL" --output-dir "$ARTIFACTS_DIR"
  exit 0
fi

: "${ROLE_HOST_PREFIX:?}" "${ROLE_HOST_DOMAIN:?}" "${POD_IP:?}"
INDEX="${JOB_COMPLETION_INDEX:?}"
use_mp_backend
export VLLM_HOST_IP="$POD_IP"
HEAD_HOST="${ROLE_HOST_PREFIX}0.${ROLE_HOST_DOMAIN}"
HEAD_IP=$(resolve "$HEAD_HOST" 1800)
export TPU_RAIDEN_ADVERTISE_HOST="$HEAD_HOST"
export TPU_RAIDEN_ENGINE_ID="${ROLE_HOST_DOMAIN}-${role}"
export TPU_KV_TRANSFER_NAMESPACE="${ROLE_HOST_DOMAIN//[^a-zA-Z0-9_]/_}_${role}"
export TPU_RAIDEN_JOB_NAME="$role"
export TPU_RAIDEN_RESHARD_IMPL=store TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_DST_SHARDS=32 TPU_RAIDEN_TRANSFER_PARALLELISM=32
export TPU_RAIDEN_RESHARD_PORT_BASE=28000 TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1 TPU_RAIDEN_POOL_TAGS_PER_LAYER=1
export RAIDEN_DISABLE_SINGLETON_WORKER=1 TPU_GDN_CONV_QK_PAIR_LAYOUT=1
export TPU_KV_TRANSFER_PORT=9100 TPU_SIDE_CHANNEL_PORT=9600 TPU_KV_SHM_POOL_GB=8
export TPU_P2P_WAIT_PULL_TIMEOUT=600 TPU_PREMAPPED_BUFFER_SIZE=17179869184
export USE_MOE_FUSED_EP_KERNEL=0 RAGGED_GATHER_REDUCE_VERSION=v3
export USE_BATCHED_RPA_LONGCTX=1 SKIP_JAX_PRECOMPILE=1 TPU_PARALLEL_PRECOMPILE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled
export VLLM_ENGINE_READY_TIMEOUT_S=10800 TORCH_DIST_TIMEOUT=1800
export XLA_FLAGS="${XLA_FLAGS:-} --xla_enable_hlo_sharding_v3=false"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset MOE_REQUANTIZE_BLOCK_SIZE

args=("$MODEL" --served-model-name qwen38-mtp --nnodes 4 --node-rank "$INDEX"
  --master-addr "$HEAD_IP" --master-port 29500 --distributed-executor-backend mp
  --enable-expert-parallel --enable-ep-weight-filter --disable-custom-all-reduce
  --load-format runai_streamer --dtype bfloat16 --trust-remote-code
  --seed 42 --generation-config vllm --max-model-len 40960 --max-num-seqs 32
  --block-size 384 --attention-backend CUSTOM --kv-cache-dtype fp8
  --mamba-ssm-cache-dtype bfloat16 --no-enable-prefix-caching
  --no-disable-hybrid-kv-cache-manager --language-model-only
  --host 0.0.0.0 --port 8000 --no-enable-log-requests
  --additional-config '{"tpu_online_fp8":{"enabled":true,"weight_scheme":"per_channel","targets":["model.layers.*.self_attn.qkv_proj","model.layers.*.self_attn.o_proj","model.layers.*.linear_attn.in_proj_qkvz","model.layers.*.linear_attn.in_proj_ba","model.layers.*.linear_attn.out_proj","model.layers.*.mlp.shared_expert.gate_up_proj","model.layers.*.mlp.shared_expert.down_proj"],"exclude":["model.layers.*.mlp.experts*"]}}')
if [[ "$INDEX" != 0 ]]; then args+=(--headless); fi
if [[ "$role" == prefill ]]; then
  export VLLM_PP_LAYER_PARTITION=23,24,24,21
  export TPU_PP_DYNAMIC_CHUNKS=1 TPU_PP_LAZY_PUSH=0 TPU_MOE_COLLECTION_CHUNK_SIZE=4096
  args+=(--pipeline-parallel-size 4 --tensor-parallel-size 1 --prefill-context-parallel-size 8
    --no-async-scheduling --max-num-batched-tokens 16384 --long-prefill-token-threshold 16384
    --gpu-memory-utilization 0.95 --cp-kv-cache-interleave-size 128
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":14579,"kv_connector_extra_config":{"destination_decode_context_parallel_size":8}}'
    --compilation-config '{"compile_sizes":[512,1024,2048]}')
else
  export TPU_MOE_COLLECTION_CHUNK_SIZE=65536
  args+=(--tensor-parallel-size 32 --decode-context-parallel-size 8
    --async-scheduling --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":14579}')
fi

# Run:AI reads local ranges and finalizes online-FP8 layers during weight loading.
# The checkpoint copy uses the pod's ephemeral disk, outside the shared caches.
gcloud storage rsync -r "$MODEL_URI" "$MODEL"
echo "SERVE_START role=$role host=$INDEX model=$MODEL"
exec vllm serve "${args[@]}"
