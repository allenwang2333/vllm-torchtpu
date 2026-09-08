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
# The JAX profiler plugin in this image crashes (PLUGIN_Profiler_Api size
# mismatch), so time the calls by wall clock instead of an xprof trace.
export MMRS_WALLCLOCK=1
python -c "import jax; jax.distributed.initialize(); print('[mmrs] jax dist ok:', jax.process_index(), '/', jax.process_count(), jax.device_count(), 'devices')" 2>&1 | grep "\[mmrs\]" || echo "[mmrs] jax.distributed.initialize probe failed"
echo "[mmrs] node=${NODE_RANK:-?} host=$(hostname) k=${MMRS_K:-7168} n=${MMRS_N:-28672} m=${MMRS_M:-1024,2048,4096,8192}"
python -c "import jax; print('[mmrs] jax', jax.__version__)" 2>/dev/null || true
# Sweep 1: Woosuk's hidden size (k = 7168) with an output block that divides it.
echo "[mmrs] sweep k=7168 bk_out=1792"
MMRS_BK_OUT=1792 timeout 1200 python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
    --k 7168 --n "${MMRS_N:-28672}" --m "${MMRS_M:-1024,2048,4096,8192}" \
    --reps "${MMRS_REPS:-20}" --warmup 5 \
    2>&1 | tee "$OUT/node${NODE_RANK:-0}_k7168.log"
echo "[mmrs] node=${NODE_RANK:-?} sweep k=7168 rc=${PIPESTATUS[0]}"
# Sweep 2: the PR's own shape (k = 8192, default blocks) as a sanity point.
echo "[mmrs] sweep k=8192 default blocks"
timeout 1200 python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
    --k 8192 --n "${MMRS_N:-28672}" --m "${MMRS_M:-1024,2048,4096,8192}" \
    --reps "${MMRS_REPS:-20}" --warmup 5 \
    2>&1 | tee "$OUT/node${NODE_RANK:-0}_k8192.log"
echo "[mmrs] node=${NODE_RANK:-?} sweep k=8192 rc=${PIPESTATUS[0]}"
exit 0
