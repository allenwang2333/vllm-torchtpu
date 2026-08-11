#!/bin/bash
# One unattended chain: try each config arm in order, fully instrumented, and
# stop at the first that serves. On success, run the MMLU-Pro smoke, then the
# full sweep. Every arm dumps the whole KV plan (group structure, per-spec
# geometry, billed startup budget) and the memory statistics, so a failure is
# diagnosable from its log without another run.
#
#   scripts/dsv4/chain_mmlu_bringup.sh [outdir]
#
# Arms vary the two levers that can matter here, in order:
#   util  -- KV pool size. Floor is set by vLLM's inflated startup guard;
#            ceiling by leaving enough HBM for compile/activation buffers.
#   seqs  -- batch width. Activation/compile scratch scales with this, not
#            with the KV pool, so it is the lever once util is boxed in.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${WORKSPACE:-$HOME/workspace}/eval_results/chain_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# util  seqs  batched_tokens
ARMS=(
    "0.67 128 1024"
    "0.67  64 1024"
    "0.66  32  512"
    "0.70  32  512"
)

summarize() {  # $1 = log
    local log="$1"
    echo "--- summary: $(basename "$log") ---"
    grep -vE "Worker_TP[1-7]" "$log" 2>/dev/null | grep -oE \
        "num_kv_cache_groups=[0-9]+|group [0-9]+ spec_type=[A-Za-z]+ layers=[0-9]+ layer_tuples=[0-9A-Za-z]+|prealloc billed total=[0-9]+ \(.*GiB\)|GPU KV cache size: [0-9,]+ tokens|total_hbm_[a-z_]*=[0-9.]*GiB|ValueError: To serve[^.]*\.|RuntimeBufferAllocationFailure|RESOURCE_EXHAUSTED" \
        | sort -u | head -40
}

# PREFLIGHT: fail in seconds, not after a 14-minute model load. Every startup
# spent discovering that the eval harness cannot run is a startup wasted --
# `lm_eval`'s API models need the `[api]` extra (tenacity), and a missing
# tokenizer path fails just as late.
    # shellcheck source=./dsv4_env.sh disable=SC1091
source "$SCRIPT_DIR/dsv4_env.sh"
if ! "$VENV/bin/python" - <<'PY'
import sys
try:
    import tenacity  # noqa: F401  (lm_eval[api] extra)
    from lm_eval.models.openai_completions import LocalCompletionsAPI  # noqa: F401
except Exception as exc:
    print(f"PREFLIGHT FAIL: {exc}")
    sys.exit(1)
print("preflight ok: lm_eval api model importable")
PY
then
    echo "PREFLIGHT FAILED -- fix the eval harness before burning a server start"
    exit 1
fi

WINNER=""
for arm in "${ARMS[@]}"; do
    read -r util seqs toks <<< "$arm"
    tag="util${util}_seqs${seqs}"
    log="$OUT/server_$tag.log"
    echo "===== ARM $tag ($(date +%H:%M:%S)) ====="
    GPU_MEM_UTIL="$util" MAX_NUM_SEQS="$seqs" MAX_NUM_BATCHED_TOKENS="$toks" \
        DSV4_DUMP_KV_PLAN=1 SERVER_TIMEOUT=1800 \
        timeout 2400 bash "$SCRIPT_DIR/serve_dsv4.sh" "$log"
    rc=$?
    summarize "$log"
    if [[ $rc -eq 0 ]]; then
        echo "===== ARM $tag SERVED ====="
        WINNER="$tag"
        echo "util=$util seqs=$seqs toks=$toks" > "$OUT/WINNER"
        break
    fi
    echo "===== ARM $tag FAILED (rc=$rc), next arm ====="
    # shellcheck source=./dsv4_env.sh disable=SC1091
    source "$SCRIPT_DIR/dsv4_env.sh"
    dsv4_kill_server
done

if [[ -z "$WINNER" ]]; then
    echo "ALL ARMS FAILED -- see $OUT"
    exit 1
fi

# Server from the winning arm is up. Smoke first, then the full sweep; both
# reuse this one server so the ~10 min load is paid once.
# shellcheck source=./dsv4_env.sh disable=SC1091
source "$SCRIPT_DIR/dsv4_env.sh"
eval_args=(
    --model local-completions
    --model_args "model=$SERVED_MODEL_NAME,base_url=http://localhost:$PORT/v1/completions,num_concurrent=32,max_gen_toks=4096,timeout=72000,max_length=8192,tokenizer_backend=huggingface,tokenizer=$MODEL_PATH"
    --tasks mmlu_pro --num_fewshot 5 --batch_size auto --seed 42
)
echo "===== SMOKE (limit=10) $(date +%H:%M:%S) ====="
API_KEY=EMPTY "$VENV/bin/lm_eval" "${eval_args[@]}" --limit 10 \
    --output_path "$OUT/smoke" 2>&1 | tee "$OUT/lm_eval_smoke.log"

echo "===== FULL SWEEP $(date +%H:%M:%S) ====="
API_KEY=EMPTY "$VENV/bin/lm_eval" "${eval_args[@]}" \
    --output_path "$OUT/full" 2>&1 | tee "$OUT/lm_eval_full.log"

dsv4_kill_server
echo "===== CHAIN DONE ($WINNER) -- results in $OUT ====="
