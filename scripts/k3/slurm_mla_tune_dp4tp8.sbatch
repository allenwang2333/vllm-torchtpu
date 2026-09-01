#!/usr/bin/env bash
#SBATCH --job-name=mla-tune-dp4tp8
#SBATCH --partition=ici-2x2x4-b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=2:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# Batch Q2 stage 2: MLA tuning for the DP=4 x TP=8 geometry. At TP=8 each
# rank carries actual_num_q_heads=12 (96/8) vs 3 at TP=32, and every shipped
# entry is keyed at 3, so job 382's server missed all 10 keys (batched_decode
# and mixed at compile buckets 1,2,4,8,2048), 768 misses each.
#
# Key geometry (quoted from job 382's server logs):
#   actual_num_q_heads=12, lkv=512, r=64, kv=fp8_e4m3fn, q=bf16,
#   page_size_per_kv_packing=4, kv_packing=4, max_num_seqs=8, pages_per_seq=640
#
# Node layout (one tuner per host; this libtpu stack grants the TPU to ~one
# client process per host, job 63):
#   node 0: batched_decode, tools/kernel/tuner/mla_kernel_tuner.py (the
#           upstream-port sweep). Decode inputs do not depend on
#           max_num_tokens, so one 72-config sweep covers all five buckets and
#           emits one entry per bucket. bench_kv_len=8192 matches the sweep
#           workload (isl=8192) rather than the 10240-token page capacity.
#   node 1: mixed buckets 1,2   (scripts/mla_mixed_prefill_tuner.py, 16 cands)
#   node 2: mixed buckets 4,8   (same small-tile list)
#   node 3: mixed bucket 2048   (17 candidates around the seqs=32 8192-bucket
#           neighbourhood, minus the (512,256,qs=2) CompileTimeScopedVmemOom
#           landmine family)
set -euo pipefail
: "${SLURM_JOB_ID:?submit with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/results/mla-tune-dp4tp8}"
DRIVER="${REPO_DIR}/scripts/k3/mla_tune_node.sh"
PY="${REPO_DIR}/.venv/bin/python"
mkdir -p "${OUT_DIR}"

# Small-tile cross product for the decode-sized mixed buckets (same list the
# seqs=32 bucket-4 tune used).
for t in 1 2 4 8; do
  cat > "${OUT_DIR}/candidates-${t}.txt" <<'EOF'
16,1,1,60
16,2,1,60
16,4,1,60
16,8,1,60
32,1,1,60
32,2,1,60
32,4,1,60
32,8,1,60
64,1,1,60
64,2,1,60
64,4,1,60
64,8,1,60
128,1,1,60
128,2,1,60
128,4,1,60
128,8,1,60
EOF
done
# Prefill bucket 2048 at 12 heads: the q operand is 4x the TP=32 tune's, so
# the big-bq/small-q_split corner is the OOM risk; (512,256) is only probed
# with q_split>=4.
cat > "${OUT_DIR}/candidates-2048.txt" <<'EOF'
512,64,2,60
512,64,4,60
512,64,8,60
512,128,2,60
512,128,4,60
512,128,8,60
512,256,4,60
512,256,8,60
256,64,1,60
256,64,2,60
256,64,4,60
256,128,1,60
256,128,2,60
256,128,4,60
128,64,1,60
128,128,1,60
128,128,2,60
EOF

export MAX_NUM_SEQS=8 PAGES_PER_SEQ=640 TOTAL_PAGES=2112 NUM_HEADS=12

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "mla-tune-dp4tp8: job ${SLURM_JOB_ID} on ${HOSTS[*]}"
echo "decode -> ${HOSTS[0]}; mixed 1,2 -> ${HOSTS[1]}; mixed 4,8 -> ${HOSTS[2]}; mixed 2048 -> ${HOSTS[3]}; out=${OUT_DIR}"

PIDS=()
# node 0: batched_decode sweep (standalone single-host topology; without
# these, jax init waits forever for the other three hosts).
srun --overlap -N1 -n1 -w "${HOSTS[0]}" --unbuffered --export=ALL \
  env PYTHONPATH="${REPO_DIR}/src" JAX_PLATFORMS= PYTHONUNBUFFERED=1 \
      TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
      TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
      LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false" \
  "${PY}" -u "${REPO_DIR}/tools/kernel/tuner/mla_kernel_tuner.py" \
      --mla_actual_num_q_heads 12 --mla_actual_lkv_dim 512 --mla_actual_r_dim 64 \
      --mla_kv_dtype float8_e4m3fn --mla_q_dtype bfloat16 \
      --mla_page_size_per_kv_packing 4 --mla_kv_packing 4 \
      --mla_max_num_seqs 8 --mla_pages_per_seq 640 --mla_total_num_pages 2112 \
      --max_num_tokens 1,2,4,8,2048 --bench_kv_len 8192 \
      --out "${OUT_DIR}/decode-sweep.json" \
  > "${OUT_DIR}/decode-sweep.log" 2>&1 &
PIDS+=($!)

# nodes 1-3: mixed buckets via the shard driver (SHARDS=1: one TPU client per
# host).
srun --overlap -N1 -n1 -w "${HOSTS[1]}" --unbuffered --export=ALL \
  env REPO_DIR="${REPO_DIR}" OUT_DIR="${OUT_DIR}" TOKENS=1,2 SHARDS=1 \
  bash "${DRIVER}" > "${OUT_DIR}/mixed-1-2.log" 2>&1 &
PIDS+=($!)
srun --overlap -N1 -n1 -w "${HOSTS[2]}" --unbuffered --export=ALL \
  env REPO_DIR="${REPO_DIR}" OUT_DIR="${OUT_DIR}" TOKENS=4,8 SHARDS=1 \
  bash "${DRIVER}" > "${OUT_DIR}/mixed-4-8.log" 2>&1 &
PIDS+=($!)
srun --overlap -N1 -n1 -w "${HOSTS[3]}" --unbuffered --export=ALL \
  env REPO_DIR="${REPO_DIR}" OUT_DIR="${OUT_DIR}" TOKENS=2048 SHARDS=1 \
  bash "${DRIVER}" > "${OUT_DIR}/mixed-2048.log" 2>&1 &
PIDS+=($!)

rc=0
for pid in "${PIDS[@]}"; do wait "${pid}" || rc=1; done

echo "== tune complete (rc=${rc}) =="
echo "--- decode winner (paste-ready entries at the end of decode-sweep.log):"
grep -A4 "^  best:" "${OUT_DIR}/decode-sweep.log" || tail -5 "${OUT_DIR}/decode-sweep.log"
for t in 1 2 4 8 2048; do
  echo "--- winner-${t}.json:"
  cat "${OUT_DIR}/winner-${t}.json" 2>/dev/null || echo "MISSING"
done
exit "${rc}"
