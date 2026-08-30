#!/usr/bin/env bash
#SBATCH --job-name=mla-tune-seqs32
#SBATCH --partition=ici-2x2x4-c
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

# MLA mixed-prefill tuning for the K3 fast recipe: max_num_seqs=32,
# pages_per_seq=640 (max_model_len=10240 / block_size=16), compile buckets
# 8192 and 4. The existing pages_per_seq=640 entries were tuned at
# max_num_seqs=64, and max_num_seqs is part of TuningKey, so a seqs=32 server
# misses every one of them; these two buckets are what COMPILE_SIZES=4,8192
# compiles.
#
# Candidates are seeded from the seqs=64 winners' neighbourhood rather than
# the full 69-candidate sweep: bucket 8192 gets the q_split probes around
# (bq=512, bkvp=64..512) plus the seqs=64 winners of the 4096/8192/16384
# buckets; bucket 4 gets the small-tile cross product around its seqs=64
# winner (bq=64, bkvp=2, q_split=1).
#
# Bucket 8192 runs on node 0, bucket 4 on node 1, both serial (SHARDS=1):
# this libtpu stack grants the TPU to only ~one client process per host (job
# 63), so intra-host parallelism just falls back to CPU.
set -euo pipefail
: "${SLURM_JOB_ID:?submit with sbatch}"

REPO_DIR="${KIMI_K3_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/results/mla-tune-seqs32}"
DRIVER="${REPO_DIR}/scripts/k3/mla_tune_node.sh"
mkdir -p "${OUT_DIR}"

# bq,bkvp,q_split,vmem_mib -- one per line.
cat > "${OUT_DIR}/candidates-8192.txt" <<'EOF'
512,256,8,60
512,256,4,60
512,256,2,60
512,128,8,60
512,128,4,60
512,128,2,60
512,64,8,60
512,64,4,60
512,64,2,60
256,256,8,60
256,128,4,60
256,64,2,60
512,512,8,60
256,512,8,60
EOF
cat > "${OUT_DIR}/candidates-4.txt" <<'EOF'
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

export REPO_DIR OUT_DIR DRIVER
export MAX_NUM_SEQS=32 PAGES_PER_SEQ=640 TOTAL_PAGES=2112

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
echo "mla-tune-seqs32: job ${SLURM_JOB_ID} on ${HOSTS[*]}"
echo "8192 -> ${HOSTS[0]}, 4 -> ${HOSTS[1]}; out=${OUT_DIR}"

srun --overlap -N1 -n1 -w "${HOSTS[0]}" --unbuffered --export=ALL \
  env TOKENS=8192 SHARDS=1 bash "${DRIVER}" &
P8192=$!
srun --overlap -N1 -n1 -w "${HOSTS[1]}" --unbuffered --export=ALL \
  env TOKENS=4 SHARDS=1 bash "${DRIVER}" &
P4=$!
rc=0
wait "${P8192}" || rc=1
wait "${P4}" || rc=1

echo "== tune complete (rc=${rc}) =="
for t in 8192 4; do
  echo "--- winner-${t}.json:"
  cat "${OUT_DIR}/winner-${t}.json" 2>/dev/null || echo "MISSING"
done
exit "${rc}"
