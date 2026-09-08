#!/bin/bash
# [branch-local] Per-host runner for the PR-332 matmul-reduce-scatter ring
# kernel benchmark on a multi-host slice (run_multihost.sh --backend torchrun
# launches this on every host; the benchmark itself calls
# jax.distributed.initialize() and every process runs the sweep, process 0
# prints the table).
set -uo pipefail
OUT="${MMRS_OUT:-/perf_eval_results/mmrs}"
mkdir -p "$OUT"
cd /root/torchtpu-vllm
# jax.distributed.initialize() with no arguments discovers the coordinator and
# the slice through the TPU VM metadata server; the tt container env disables
# that lookup (torch_tpu does not need it), so re-enable it for this JAX run.
unset TPU_SKIP_MDS_QUERY
python -c "import jax; jax.distributed.initialize(); print('[mmrs] jax dist ok:', jax.process_index(), '/', jax.process_count(), jax.device_count(), 'devices')" 2>&1 | grep "\[mmrs\]" || echo "[mmrs] jax.distributed.initialize probe failed"
echo "[mmrs] node=${NODE_RANK:-?} host=$(hostname) k=${MMRS_K:-7168} n=${MMRS_N:-28672} m=${MMRS_M:-1024,2048,4096,8192}"
python -c "import jax; print('[mmrs] jax', jax.__version__)" 2>/dev/null || true
timeout 2400 python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
    --k "${MMRS_K:-7168}" --n "${MMRS_N:-28672}" --m "${MMRS_M:-1024,2048,4096,8192}" \
    --reps "${MMRS_REPS:-20}" --warmup 5 --profile-dir "$OUT/profile" \
    2>&1 | tee "$OUT/node${NODE_RANK:-0}.log"
echo "[mmrs] node=${NODE_RANK:-?} rc=${PIPESTATUS[0]}"
exit 0
