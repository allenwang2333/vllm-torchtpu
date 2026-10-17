#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dev only: one Qwen3.5 PP engine with the disagg prefill engine's model
# arguments but no KV connector, then the prefix-cache correctness smoke
# against it directly.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

PREFILL_PP="${PREFILL_PP:-4}"
PORT="${PORT:-8400}"
RUN_DIR="${RUN_ROOT:-perf_eval_results/qwen35_pp_standalone}/run"
mkdir -p "${RUN_DIR}/logs"

export PYTHONPATH="${repo_root}/src:${repo_root}:${PYTHONPATH:-}"
export JAX_PLATFORMS=tpu,cpu PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax
export VLLM_XLA_CHECK_RECOMPILATION=0 USE_MOE_SPARSE_CORE=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 VLLM_TPU_BLOCK_MAJOR_KV=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_scoped_vmem_limit_kib=65536"
export ONEHOT_MOE_PERMUTE_THRESHOLD=1024
export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback TPU_RAGGED_GATHER_IMPL=fallback
export VLLM_SERVER_DEV_MODE=1
sb=""
for ((i = 0; i < PREFILL_PP; i++)); do sb+="${sb:+,}localhost:$((29500 + i))"; done
export TORCH_TPU_SLICEBUILDER_ADDRESSES="${sb}"

python -m vllm.entrypoints.openai.api_server --port "${PORT}" --host 127.0.0.1 \
  --model Qwen/Qwen3.5-35B-A3B-FP8 --served-model-name Qwen3.5-35B-A3B-FP8 \
  --trust-remote-code --seed 42 --max-model-len 262144 --enable-expert-parallel \
  --disable-custom-all-reduce --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --language-model-only --max-num-batched-tokens 16384 --max-num-seqs 64 \
  --no-disable-hybrid-kv-cache-manager --attention-backend CUSTOM \
  --mamba-cache-mode align --prefix-cache-retention-interval 0 \
  --enable-prompt-tokens-details --no-enable-log-requests --enable-prefix-caching \
  --no-async-scheduling \
  --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096]}' \
  --tensor-parallel-size 1 --pipeline-parallel-size "${PREFILL_PP}" --block-size 768 \
  >"${RUN_DIR}/logs/engine.log" 2>&1 &
engine_pid=$!

ready=0
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then ready=1; break; fi
  kill -0 "${engine_pid}" 2>/dev/null || break
  sleep 15
done
if [[ "${ready}" != 1 ]]; then
  echo "PP_STANDALONE_FAIL: engine did not come up"
  tail -40 "${RUN_DIR}/logs/engine.log"
  kill "${engine_pid}" 2>/dev/null
  exit 1
fi
grep -m1 -oP 'kv lcm block sizes \K[0-9]+' "${RUN_DIR}/logs/engine.log" | sed 's/^/engine block size /'

python3 scripts/vllm/integration/smoke_prefix_cache_correctness.py \
  --host 127.0.0.1 --port "${PORT}" --model Qwen3.5-35B-A3B-FP8 \
  2>&1 | tee "${RUN_DIR}/logs/correctness.log"
rc=${PIPESTATUS[0]}
kill "${engine_pid}" 2>/dev/null
wait "${engine_pid}" 2>/dev/null
if [[ "${rc}" == 0 ]]; then echo "PP_STANDALONE_OK"; else echo "PP_STANDALONE_FAIL rc=${rc}"; fi
exit "${rc}"
