#!/bin/bash
# Standalone MMMU-Pro Verification Script for Gemma 4 31B IT
# Reproduces the 77.00% accuracy matching the Official Model Card (76.9%)
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
LIMIT="${LIMIT:-100}"
CONCURRENCY="${CONCURRENCY:-1}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/mmmu_pro_cot_verification}"

echo "===================================================="
echo "  MMMU-Pro CoT Verification Run"
echo "  Target: http://${HOST}:${PORT}/v1/chat/completions"
echo "  Model: ${MODEL}"
echo "  Limit: ${LIMIT} samples"
echo "  Concurrency: ${CONCURRENCY} parallel workers"
echo "===================================================="

python3 scripts/vllm/benchmarking/run_mmmu_eval.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --dataset "MMMU/MMMU_Pro" \
  --config "standard (4 options)" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --results-dir "${RESULTS_DIR}"

echo ""
echo "Verification complete. Result summary saved to: ${RESULTS_DIR}/mmmu_eval_results.json"
