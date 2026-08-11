#!/usr/bin/env bash
# ==============================================================================
# Compilation Cache E2E Test (Raw TPU Execution)
# ==============================================================================
# Verifies cold compilation and warm cache hit consistency for vllm-torchtpu.
# Usage: ./scripts/vllm/integration/run_compilation_cache_e2e.sh [model_name]
# Example: ./scripts/vllm/integration/run_compilation_cache_e2e.sh Qwen/Qwen2.5-0.5B
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_DIR}"

RUN_ID="cache_test_$$"
CACHE_DIR="/tmp/vllm_xla_cache_${RUN_ID}"
mkdir -p "${CACHE_DIR}"

export VLLM_XLA_CACHE_PATH="${CACHE_DIR}/xla_cache"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_cache"
export VLLM_TARGET_DEVICE="tpu"

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B}}"
TIER3_DIR="${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3"

# Helper function: kill orphaned processes holding TPU locks
clean_tpu_processes() {
  echo "=== 🧹 Cleaning orphaned TPU processes... ==="
  for d in /proc/[0-9]*; do
    if [ -f "$d/cmdline" ]; then
      pid=$(basename "$d")
      cmd=$(tr "\0" " " < "$d/cmdline" 2>/dev/null || true)
      if echo "$cmd" | grep -qiE "python|vllm|enginecore"; then
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    fi
  done
  rm -rf /tmp/libtpu* /tmp/tpu_logs* || true
}

# --- 1. Cold Compilation Run ---
echo "================================================================="
echo "=== 🏃 Run 1: Cold Compilation (${MODEL})"
echo "================================================================="
clean_tpu_processes

rm -rf "${VLLM_XLA_CACHE_PATH}" "${VLLM_CACHE_ROOT}"
mkdir -p "${VLLM_XLA_CACHE_PATH}" "${VLLM_CACHE_ROOT}"

python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 --num-gpu-blocks-override 47376 --seed 0 2>&1 | tee "${CACHE_DIR}/cold_run.log"

sync && sleep 10

if grep -q "Compiling FX graph" "${CACHE_DIR}/cold_run.log"; then
  echo "✅ Cold run triggered compilation as expected."
else
  echo "❌ Cold run did not trigger compilation!"
  exit 1
fi

if [ -z "$(find "${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3" -type f -print -quit 2>/dev/null)" ]; then
  echo "❌ No files found in tier3 cache directory (${VLLM_XLA_CACHE_PATH}/torch_tpu_tier3)!"
  exit 1
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -name "*.bin" -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_cold.txt" || true
else
  touch "${CACHE_DIR}/tier3_cold.txt"
fi

# --- 2. Warm Compilation Cache Hit Run ---
echo "================================================================="
echo "=== 🏃 Run 2: Warm Compilation (Cache Hit Check)"
echo "================================================================="
clean_tpu_processes

python3 examples/offline_inference.py --model "${MODEL}" \
  --max-model-len 256 --max-num-batched-tokens 256 --num-gpu-blocks-override 47376 --seed 0 2>&1 | tee "${CACHE_DIR}/warm_run.log"

sync && sleep 10

if grep -q -E "Directly load|Using cache directory" "${CACHE_DIR}/warm_run.log"; then
  echo "✅ Warm run log confirmed cache hit (Directly load from cache)."
else
  echo "❌ Warm run did not log cache hit!"
  exit 1
fi

if [[ -d "${TIER3_DIR}" ]]; then
  find "${TIER3_DIR}" -name "*.bin" -exec md5sum {} \; | sort > "${CACHE_DIR}/tier3_after.txt" || true
else
  touch "${CACHE_DIR}/tier3_after.txt"
fi

echo "=== 🔍 Verifying Zero File & MD5 Differences (Cold vs Warm) ==="
if diff -u "${CACHE_DIR}/tier3_cold.txt" "${CACHE_DIR}/tier3_after.txt"; then
  echo "✅ Tier-3 cache files and MD5sums are 100% identical between Cold and Warm runs!"
else
  echo "❌ Tier-3 cache files or MD5sums changed between Cold and Warm runs!"
  exit 1
fi

echo "================================================================="
echo "🎉 Compilation Cache E2E Test Passed!"
echo "Logs saved in: ${CACHE_DIR}"
echo "================================================================="
