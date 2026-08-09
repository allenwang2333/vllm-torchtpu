#!/bin/bash
# Find a serving config that survives startup + compile, using DUMMY WEIGHTS.
#
# Startup failures so far (KV budget guard, RuntimeBufferAllocationFailure) are
# shape/memory problems: they cannot depend on weight values. Dummy weights skip
# the safetensors load and the MoE FP4 stage -- ~11 of the ~14 min per attempt --
# so the whole grid costs about what one real-weight attempt does.
#
# The winning config MUST then be re-run with real weights before any accuracy
# number is recorded. Random weights produce fluent nonsense; this model has
# already burned weeks on exactly that failure mode (all-zero MoE experts).
#
#   scripts/dsv4/search_serving_config.sh [outdir]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/home/patemotter_google_com/workspace/eval_results/cfgsearch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# util  seqs  batched_tokens   -- ordered most-desirable first (match the
# reference's shape where possible), then progressively cheaper.
ARMS=(
    "0.67 128 1024"
    "0.67  64 1024"
    "0.67  32 1024"
    "0.66  32  512"
    "0.70  32  512"
    "0.67  16  512"
)

echo "### dummy-weight config search -- results are about STARTUP ONLY ###"
for arm in "${ARMS[@]}"; do
    read -r util seqs toks <<< "$arm"
    tag="util${util}_seqs${seqs}_tok${toks}"
    log="$OUT/dummy_$tag.log"
    echo "===== ARM $tag ($(date +%H:%M:%S)) ====="
    LOAD_FORMAT=dummy GPU_MEM_UTIL="$util" MAX_NUM_SEQS="$seqs" \
        MAX_NUM_BATCHED_TOKENS="$toks" DSV4_DUMP_KV_PLAN=1 SERVER_TIMEOUT=900 \
        timeout 1200 bash "$SCRIPT_DIR/serve_dsv4.sh" "$log"
    rc=$?
    grep -vE "Worker_TP[1-7]" "$log" 2>/dev/null | grep -oE \
        "num_kv_cache_groups=[0-9]+|prealloc billed total=[0-9]+ \(.*GiB\)|GPU KV cache size: [0-9,]+ tokens|total_hbm_avail_gb=[0-9.]+GiB|ValueError: To serve|RuntimeBufferAllocationFailure|RESOURCE_EXHAUSTED" \
        | sort -u | sed 's/^/    /'
    if [[ $rc -eq 0 ]]; then
        echo "===== ARM $tag SERVED (dummy weights) ====="
        echo "$tag" > "$OUT/WINNER"
        echo "util=$util seqs=$seqs toks=$toks" >> "$OUT/WINNER"
    else
        echo "===== ARM $tag FAILED rc=$rc ====="
    fi
    # shellcheck source=./dsv4_env.sh disable=SC1091
    source "$SCRIPT_DIR/dsv4_env.sh"
    dsv4_kill_server
    [[ $rc -eq 0 ]] && break
done

if [[ -f "$OUT/WINNER" ]]; then
    echo "### WINNER (startup only, dummy weights): $(tr '\n' ' ' < "$OUT/WINNER")"
    echo "### Re-run with REAL weights before recording any accuracy number."
else
    echo "### NO ARM SERVED -- see $OUT"
fi
