#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Qwen3.5-397B PCP=8 full-graph CoreHalt and verify the vLLM
# splitting_ops mitigation. Each mode must run on a fresh TPU runner: a
# RuntimeUnexpectedCoreHalt can leave the TPU runtime unusable for a later run.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

mode="${1:-${GDN_SPLITTING_MODE:-split}}"
case "${mode}" in
  split)
    default_expected_outcome=healthy
    ;;
  full)
    default_expected_outcome=core_halt
    ;;
  *)
    echo "Usage: $0 {split|full}" >&2
    exit 2
    ;;
esac

expected_outcome="${EXPECTED_OUTCOME:-${default_expected_outcome}}"
case "${expected_outcome}" in
  healthy|core_halt) ;;
  *)
    echo "EXPECTED_OUTCOME must be healthy or core_halt, got: ${expected_outcome}" >&2
    exit 2
    ;;
esac

python_bin="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "Could not find an executable Python interpreter." >&2
  exit 2
fi

model_dir="${MODEL_DIR:-${repo_root}/tests/fixtures/models/qwen3_5_397b_a17b_fp8}"
if [[ ! -f "${model_dir}/config.json" ]]; then
  echo "Qwen3.5-397B config fixture not found: ${model_dir}/config.json" >&2
  exit 2
fi

compilation_config="$("${python_bin}" - "${mode}" <<'PY'
import json
import sys

mode = sys.argv[1]
splitting_ops = []
if mode == "split":
    splitting_ops = [
        "pallas::gdn_attention_pcp_"
        f"language_model_model_layers_{layer}_linear_attn"
        for layer in range(60)
        if layer % 4 != 3
    ]

config = {
    "compile_sizes": [512],
    "use_inductor_graph_partition": False,
    "splitting_ops": splitting_ops,
    # Piecewise mode otherwise enables this CUDA-only fusion pass in vLLM.
    "pass_config": {"fuse_rope_kvcache_cat_mla": False},
    "inductor_compile_config": {
        "enable_auto_functionalized_v2": False,
        "size_asserts": False,
        "alignment_asserts": False,
        "scalar_asserts": False,
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
)"

if [[ "${PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "${compilation_config}"
  exit 0
fi

run_root="${RUN_ROOT:-/tmp/qwen35_397b_pcp8_compile_repro}"
run_suffix="${GITHUB_RUN_ID:-manual}_${GITHUB_RUN_ATTEMPT:-1}_$$_$(date +%s)"
run_dir="${RUN_DIR:-${run_root}/${mode}_${run_suffix}}"
server_log="${run_dir}/server.log"
summary_file="${run_dir}/summary.txt"
port="${PORT:-8000}"
startup_timeout_seconds="${STARTUP_TIMEOUT_SECONDS:-1800}"
poll_interval_seconds="${POLL_INTERVAL_SECONDS:-5}"

mkdir -p "${run_dir}" "${run_dir}/vllm_cache" \
  "${run_dir}/xla_cache" "${run_dir}/torchinductor_cache"
: >"${server_log}"

server_pid=""
tail_pid=""
cleanup() {
  # Target the whole setsid process group even if its API-server leader has
  # already exited; CoreHalt can otherwise leave worker children behind.
  if [[ -n "${server_pid}" ]]; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 -- "-${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
  fi
  if [[ -n "${tail_pid}" ]]; then
    kill "${tail_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

server_args=(
  --host 127.0.0.1
  --port "${port}"
  --model "${model_dir}"
  --served-model-name Qwen3.5-397B-A17B-FP8-dummy
  --trust-remote-code
  --skip-tokenizer-init
  --dtype bfloat16
  --quantization fp8
  --generation-config vllm
  --load-format dummy
  --tensor-parallel-size 1
  --data-parallel-size 1
  --prefill-context-parallel-size 8
  --cp-kv-cache-interleave-size 256
  --enable-expert-parallel
  --distributed-executor-backend mp
  --disable-custom-all-reduce
  --block-size 256
  --num-gpu-blocks-override 128
  --kv-cache-dtype bfloat16
  --mamba-cache-dtype bfloat16
  --mamba-ssm-cache-dtype bfloat16
  --no-enable-prefix-caching
  --language-model-only
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --max-model-len 256000
  --max-num-batched-tokens 4096
  --max-num-seqs 1
  --enable-chunked-prefill
  --disable-chunked-mm-input
  --no-async-scheduling
  --compilation-config "${compilation_config}"
)

{
  echo "mode=${mode}"
  echo "expected_outcome=${expected_outcome}"
  echo "model_dir=${model_dir}"
  echo "compilation_config=${compilation_config}"
  printf 'command='
  printf '%q ' "${python_bin}" -m vllm.entrypoints.openai.api_server \
    "${server_args[@]}"
  printf '\n'
} | tee "${run_dir}/launch_config.txt"

tier2_name="qwen397b_pcp8_compile_repro_${mode}_${GITHUB_RUN_ID:-local}_${GITHUB_RUN_ATTEMPT:-1}"
setsid env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${repo_root}/src:${repo_root}:${PYTHONPATH:-}" \
  VLLM_PLUGINS=torchtpu \
  VLLM_TARGET_DEVICE=tpu \
  MODEL_IMPL_TYPE=vllm \
  JAX_PLATFORMS=tpu,cpu \
  PJRT_DEVICE=TPU \
  TPU_BACKEND_TYPE=jax \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false \
  TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${tier2_name}" \
  TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${run_dir}/xla_cache/torch_tpu_tier3" \
  TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
  TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0 \
  RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd \
  USE_MOE_SPARSE_CORE=0 \
  RAY_USAGE_STATS_ENABLED=0 \
  VLLM_CACHE_ROOT="${run_dir}/vllm_cache" \
  VLLM_XLA_CACHE_PATH="${run_dir}/xla_cache" \
  TORCHINDUCTOR_CACHE_DIR="${run_dir}/torchinductor_cache" \
  LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 \
  "${python_bin}" -m vllm.entrypoints.openai.api_server \
  "${server_args[@]}" >"${server_log}" 2>&1 &
server_pid=$!

# Stream the server log to the CI console while retaining a complete artifact.
tail --pid="${server_pid}" -n +1 -F "${server_log}" &
tail_pid=$!

observed_outcome=timeout
deadline=$((SECONDS + startup_timeout_seconds))
core_halt_pattern='RuntimeUnexpectedCoreHalt'

while ((SECONDS < deadline)); do
  if grep -Eiq "${core_halt_pattern}" "${server_log}"; then
    observed_outcome=core_halt
    break
  fi
  health_code="$(curl -fsS -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
  if [[ "${health_code}" == "200" ]]; then
    observed_outcome=healthy
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    # Worker tracebacks can reach the shared log just after the API-server
    # leader exits. Give them a short flush window before classifying the exit.
    sleep 2
    if grep -Eiq "${core_halt_pattern}" "${server_log}"; then
      observed_outcome=core_halt
    else
      observed_outcome=process_exit
    fi
    break
  fi
  sleep "${poll_interval_seconds}"
done

printf 'mode=%s\nexpected_outcome=%s\nobserved_outcome=%s\n' \
  "${mode}" "${expected_outcome}" "${observed_outcome}" \
  | tee "${summary_file}"

if [[ "${expected_outcome}" == "healthy" ]]; then
  if [[ "${observed_outcome}" != "healthy" ]]; then
    echo "Split compilation did not become healthy." >&2
    exit 1
  fi
  if grep -Eiq "${core_halt_pattern}" "${server_log}"; then
    echo "Split compilation reached health but also logged a CoreHalt." >&2
    exit 1
  fi
  if ! grep -q 'subgraph_45' "${server_log}"; then
    echo "Split mode became healthy without evidence of all 46 graph pieces." >&2
    exit 1
  fi
  echo "QWEN35_397B_PCP8_SPLIT_COMPILE_OK=1" | tee -a "${summary_file}"
else
  if [[ "${observed_outcome}" != "core_halt" ]]; then
    echo "Full-graph mode did not reproduce RuntimeUnexpectedCoreHalt." >&2
    exit 1
  fi
  echo "QWEN35_397B_PCP8_FULL_GRAPH_CORE_HALT_REPRODUCED=1" \
    | tee -a "${summary_file}"
fi

echo "RUN_DIR=${run_dir}"
