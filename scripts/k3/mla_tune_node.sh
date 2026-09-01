#!/usr/bin/env bash
# Tune the MLA v2 mixed-prefill kernel for one or more token buckets on a
# single TPU host. Candidates are sharded across this host's TPU devices, one
# tuner subprocess per device (TPU_VISIBLE_DEVICES), so a 69-candidate sweep
# costs ceil(69/SHARDS) serial candidates instead of 69.
#
# Env:
#   REPO_DIR     repository root (default: two levels up from this script)
#   OUT_DIR      where shard + merged JSON results go (required)
#   TOKENS       comma-separated token buckets to tune (default: 8192)
#   KV_TOKENS    optional --kv-tokens override applied to every bucket
#                (used for the 16384 bucket: q=16384 over 10240-token pages)
#   SHARDS       concurrent subprocesses, one per device (default: 8)
#   NUM_HEADS    per-rank query heads -> --num-heads (default: 3, TP=32)
#   MAX_NUM_SEQS tuning-key geometry (default: 64)
#   PAGES_PER_SEQ tuning-key geometry (default: 640)
#   TOTAL_PAGES  tuner cache size (default: 2112; only seq 0 is populated)
#
# Candidate list: ${OUT_DIR}/candidates-<tokens>.txt wins over
# ${OUT_DIR}/candidates.txt, which is generated from the tuner's default
# 69-candidate sweep if neither exists. One "BQ,BKVP,Q_SPLIT,VMEM_MIB" per
# line. Pre-seed a per-bucket file to tune a neighbourhood instead of the
# full sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${OUT_DIR:?set OUT_DIR}"
TOKENS="${TOKENS:-8192}"
KV_TOKENS="${KV_TOKENS:-}"
SHARDS="${SHARDS:-8}"
NUM_HEADS="${NUM_HEADS:-3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PAGES_PER_SEQ="${PAGES_PER_SEQ:-640}"
TOTAL_PAGES="${TOTAL_PAGES:-2112}"

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=""
# Single-host libtpu topology: without these, jax init on a multi-host slice
# waits forever for the other hosts to join.
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"
TUNER="${REPO_DIR}/scripts/mla_mixed_prefill_tuner.py"

mkdir -p "${OUT_DIR}"
CANDIDATES="${OUT_DIR}/candidates.txt"
if [[ ! -s "${CANDIDATES}" ]]; then
  "${REPO_DIR}/.venv/bin/python" - "${TUNER}" > "${CANDIDATES}" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1].rsplit("/", 1)[0])
from mla_mixed_prefill_tuner import _default_candidates
for c in _default_candidates():
    print(",".join(map(str, c)))
EOF
fi

run_bucket() {
  local tokens="$1"
  shift
  local extra=("$@")
  local candidates="${OUT_DIR}/candidates-${tokens}.txt"
  [[ -s "${candidates}" ]] || candidates="${CANDIDATES}"
  local num_candidates
  num_candidates=$(wc -l < "${candidates}")
  local args=(--tokens="${tokens}"
              --num-heads="${NUM_HEADS}"
              --max-num-seqs="${MAX_NUM_SEQS}"
              --pages-per-seq="${PAGES_PER_SEQ}"
              --total-pages="${TOTAL_PAGES}")
  echo "bucket ${tokens}: ${num_candidates} candidates from ${candidates} (shards=${SHARDS})"
  local pids=()
  local start
  start=$(date +%s)
  for shard in $(seq 0 $((SHARDS - 1))); do
    local shard_args=()
    while read -r cand; do
      shard_args+=(--candidate="${cand}")
    done < <(awk -v s="${shard}" -v n="${SHARDS}" 'NR % n == s' "${candidates}")
    [[ ${#shard_args[@]} -gt 0 ]] || continue
    TPU_VISIBLE_DEVICES="${shard}" \
      "${REPO_DIR}/.venv/bin/python" "${TUNER}" \
      "${args[@]}" "${extra[@]}" "${shard_args[@]}" \
      --output="${OUT_DIR}/tune-${tokens}-shard${shard}.json" \
      > "${OUT_DIR}/tune-${tokens}-shard${shard}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  local elapsed=$(( $(date +%s) - start ))
  echo "bucket ${tokens}: ${elapsed}s wall (rc=${rc})"
  TOKENS="${tokens}" OUT_DIR="${OUT_DIR}" "${REPO_DIR}/.venv/bin/python" - <<'EOF'
import glob, json, os
tokens, out_dir = os.environ["TOKENS"], os.environ["OUT_DIR"]
rows = []
for path in glob.glob(f"{out_dir}/tune-{tokens}-shard*.json"):
    rows.extend(json.loads(open(path).read()))
valid = [r for r in rows if r.get("status") == "success"]
valid.sort(key=lambda r: r["median_ms"])
by_status = {}
for r in rows:
    by_status[r["status"]] = by_status.get(r["status"], 0) + 1
print(f"bucket {tokens}: {len(rows)} results, by_status={by_status}")
for rank, r in enumerate(valid[:5], 1):
    print(f"  {rank}. median={r['median_ms']:.3f}ms bq={r['bq']} bkvp={r['bkvp']} "
          f"q_split={r['q_split']} vmem={r['vmem_mib']}MiB")
json.dump(rows, open(f"{out_dir}/tuned-{tokens}.json", "w"), indent=2)
if valid:
    json.dump(valid[0], open(f"{out_dir}/winner-{tokens}.json", "w"), indent=2)
else:
    raise SystemExit(f"no successful candidate for bucket {tokens}")
EOF
}

echo "tuning tokens=[${TOKENS}] shards=${SHARDS} out=${OUT_DIR}"
IFS=',' read -ra token_list <<< "${TOKENS}"
for t in "${token_list[@]}"; do
  extra=()
  [[ -n "${KV_TOKENS}" ]] && extra+=(--kv-tokens="${KV_TOKENS}")
  run_bucket "${t}" "${extra[@]}"
done
