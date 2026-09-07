#!/bin/bash
# Kimi-K3 accuracy vs client concurrency (issue #841).
# One server boot (config kimi-k3-tp32-ep-concsweep), then lm_eval at each
# concurrency in ascending order, for each task, in the nightly's eval form
# (local-chat-completions, plain chat, thinking off). Engine drained between
# points. Every point is uploaded as soon as it is scored.
#
#   k3_eval_conc_sweep.sh --results-dir DIR [--conc-list 1,2,4,...]
#                         [--tasks mmlu_pro,gsm8k] [--gsm8k-limit 128]
#                         [--mmlu-limit 10] [--ctx 16384] [--max-num-seqs 256]
#                         [--port 8000] [--gcs-root gs://bucket/prefix]
#
# Results (per-point lm_eval output, samples, summary.md) land in --results-dir,
# which the pipeline uploads as Buildkite artifacts; --gcs-root additionally
# copies each point to <gcs-root>/<build number>/ as soon as it is scored.
#
# Question counts grow with concurrency so that the requested concurrency is
# really reached: gsm8k limit = max(--gsm8k-limit, 2*conc); mmlu_pro limit
# (per subject, 14 subjects) = max(--mmlu-limit, ceil(2*conc/14)).
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-concsweep"
RESULTS_DIR=""
CONC_LIST="1,2,4,8,16,32,64,128,256"
TASKS="mmlu_pro,gsm8k"
GSM8K_LIMIT=128
MMLU_LIMIT=10
PORT="${PORT:-8000}"
TEMPLATE_KWARGS='{"thinking": false}'
GCS_ROOT=""
export K3_SWEEP_CTX="${K3_SWEEP_CTX:-16384}"
export K3_SWEEP_MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --conc-list)   CONC_LIST="$2"; shift 2 ;;
        --tasks)       TASKS="$2"; shift 2 ;;
        --gsm8k-limit) GSM8K_LIMIT="$2"; shift 2 ;;
        --mmlu-limit)  MMLU_LIMIT="$2"; shift 2 ;;
        --ctx)         export K3_SWEEP_CTX="$2"; shift 2 ;;
        --max-num-seqs) export K3_SWEEP_MAX_NUM_SEQS="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        --gcs-root)    GCS_ROOT="$2"; shift 2 ;;
        --config)      CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$RESULTS_DIR" ] || { echo "ERROR: --results-dir is required"; exit 1; }
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.md"
[ -n "$GCS_ROOT" ] && GCS_ROOT="${GCS_ROOT%/}/${BUILDKITE_BUILD_NUMBER:-local}"

log() { echo "[k3-concsweep] $*"; }

cleanup() {
    log "cleanup"
    bash "$SCRIPT_DIR/cleanup_server.sh"
    if [ -f "$RESULTS_DIR/server.log" ]; then
        log "===== server.log: error context ====="
        tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' \
            | grep -v -E 'ActorHandle|ray_executor_v2.py:566|Failed to kill|client_mode_hook|auto_init_hook|_raylet|ray/_private' \
            | grep -n -E -B3 -A25 'Traceback|Error|Exception|Killed|OOM|halt' | tail -120 | cut -c1-400
        log "===== server.log: last 30 non-blank lines ====="
        tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' | tail -30 | cut -c1-300
    fi
    [ -f "$SUMMARY" ] && { log "===== summary ====="; cat "$SUMMARY"; }
    if [ -n "$GCS_ROOT" ] && command -v gsutil >/dev/null 2>&1; then
        log "uploading results to $GCS_ROOT"
        gsutil -q -m cp -r "$RESULTS_DIR" "$GCS_ROOT/" 2>&1 | tail -3 || log "WARNING: GCS upload failed"
    fi
}
trap cleanup EXIT INT TERM

save_now() {
    [ -n "$GCS_ROOT" ] && command -v gsutil >/dev/null 2>&1 || return 0
    for x in "$@"; do
        [ -e "$x" ] || continue
        gsutil -q -m cp -r "$x" "$GCS_ROOT/$(basename "$RESULTS_DIR")/" 2>&1 | tail -2 || log "WARNING: upload of $x failed"
    done
}

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# Engine drain between points: vllm:num_requests_running + waiting == 0, 3 polls.
wait_server_idle() {
    local timeout=1800 t0 idle=0 n
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$timeout" ]; do
        n=$(curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^vllm:num_requests_(running|waiting)/{s+=$NF} END{print s+0}')
        if [ "${n%.*}" = "0" ]; then idle=$((idle+1)); [ "$idle" -ge 3 ] && return 0; else idle=0; fi
        sleep 10
    done
    log "WARNING: engine did not drain within ${timeout}s"; return 1
}

# ---------------------------------------------------------------- boot
log "===== boot: config=$CONFIG_NAME concs=${CONCS[*]} tasks=${TASK_LIST[*]} gsm8k_limit>=$GSM8K_LIMIT mmlu_limit>=$MMLU_LIMIT ctx=$K3_SWEEP_CTX max_num_seqs=$K3_SWEEP_MAX_NUM_SEQS ====="
bash "$SCRIPT_DIR/cleanup_server.sh"
T_BOOT=$(date +%s)
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT" \
    || { log "FATAL: server boot / smoke bench failed"; exit 1; }
server_alive || { log "FATAL: server not answering after boot"; exit 1; }
log "server up after $(( $(date +%s) - T_BOOT )) s"
MODEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")
save_now "$RESULTS_DIR/config.json"

{
    echo "# K3 accuracy vs eval concurrency (build ${BUILDKITE_BUILD_NUMBER:-local}, config $CONFIG_NAME, ctx $K3_SWEEP_CTX, max-num-seqs $K3_SWEEP_MAX_NUM_SEQS, thinking off, plain chat)"
    echo
    echo "| conc | task | questions | metric | score | stderr | wall s | errors |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

# ---------------------------------------------------------------- sweep
score_of() {   # score_of <output_dir> <task> -> "metric score stderr"
    python3 - "$1" "$2" << 'PY'
import glob, json, os, sys
d, task = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
if not files:
    print("none nan nan"); sys.exit()
r = json.load(open(files[-1]))["results"][task]
for k in ("exact_match,custom-extract", "exact_match,flexible-extract", "exact_match,strict-match"):
    if k in r:
        print(k, r[k], r.get(k.replace("exact_match", "exact_match_stderr"), float("nan")))
        break
else:
    print("none nan nan")
PY
}

FAIL=0
for c in "${CONCS[@]}"; do
    for task in "${TASK_LIST[@]}"; do
        if [ "$task" = "gsm8k" ]; then
            limit=$GSM8K_LIMIT; [ $((2*c)) -gt "$limit" ] && limit=$((2*c))
            n_q=$limit
            gen_kwargs="{\"chat_template_kwargs\": $TEMPLATE_KWARGS, \"max_gen_toks\": 1024}"
        else
            limit=$MMLU_LIMIT; [ $(( (2*c + 13) / 14 )) -gt "$limit" ] && limit=$(( (2*c + 13) / 14 ))
            n_q=$((limit * 14))
            gen_kwargs="{\"chat_template_kwargs\": $TEMPLATE_KWARGS}"
        fi
        out="$RESULTS_DIR/${task}_c${c}"; mkdir -p "$out"
        log "===== conc=$c task=$task limit=$limit (~$n_q questions) ====="
        server_alive || { log "FATAL: server died before conc=$c $task"; FAIL=1; break 2; }
        T0=$(date +%s)
        lm_eval --tasks "$task" --seed "0,1234,None,1234" --output_path "$out" --log_samples --limit "$limit" \
            --model local-chat-completions \
            --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=$c,max_retries=3,timeout=3600" \
            --apply_chat_template --gen_kwargs "$gen_kwargs" > "$out/lm_eval.log" 2>&1
        rc=$?
        wall=$(( $(date +%s) - T0 ))
        errs=$(grep -c -iE 'error|exception|timed out' "$out/lm_eval.log" || true)
        read -r metric score stderr <<< "$(score_of "$out" "$task")"
        log "conc=$c task=$task rc=$rc $metric=$score ±$stderr wall=${wall}s log-errors=$errs"
        [ "$rc" -ne 0 ] && { log "lm_eval failed; last 40 log lines:"; tail -40 "$out/lm_eval.log" | cut -c1-300; }
        echo "| $c | $task | $n_q | $metric | $score | $stderr | $wall | rc=$rc/$errs |" >> "$SUMMARY"
        save_now "$out" "$SUMMARY"
        wait_server_idle || true
    done
done

log "===== summary ====="
cat "$SUMMARY"
server_alive || { log "server is DEAD at the end of the sweep"; FAIL=1; }
exit $FAIL
