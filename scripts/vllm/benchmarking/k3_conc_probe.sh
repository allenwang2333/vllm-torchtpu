#!/bin/bash
# Concurrency probe driver (issue #845): boot one server, then run
# conc_probe.py at each N in --conc-list, checking after every N whether the
# server is still alive. Server variants come from --env KEY=VAL (exported
# before the boot; the Ray executor copies the driver env to every worker) and
# --extra-serve-args (appended last, overrides the config's flags).
#
#   k3_conc_probe.sh --results-dir DIR [--config kimi-k3-tp32-ep-concsweep]
#       [--conc-list 128,160,192,256] [--ctx 16384] [--max-num-seqs 256]
#       [--max-tokens 1500] [--env KEY=VAL]... [--extra-serve-args "..."]
#       [--gcs-root gs://bucket/prefix] [--port 8000]
set -uo pipefail
CONFIG_NAME="kimi-k3-tp32-ep-concsweep"
RESULTS_DIR=""; CONC_LIST="128,160,192,256"; MAX_TOKENS=1500; PORT="${PORT:-8000}"; GCS_ARG=""
export K3_SWEEP_CTX="${K3_SWEEP_CTX:-16384}" K3_SWEEP_MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
ENVS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --conc-list) CONC_LIST="$2"; shift 2 ;;
        --ctx) export K3_SWEEP_CTX="$2"; shift 2 ;;
        --max-num-seqs) export K3_SWEEP_MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --env) ENVS+=("$2"); shift 2 ;;
        --extra-serve-args) export K3_EXTRA_SERVE_ARGS="$2"; shift 2 ;;
        --gcs-root) GCS_ARG="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "[k3-conc-probe] unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$RESULTS_DIR" ] || { echo "[k3-conc-probe] ERROR: --results-dir is required"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3_SWEEP_TAG="k3-conc-probe"
# shellcheck source=scripts/vllm/benchmarking/k3_sweep_lib.sh
source "$SCRIPT_DIR/k3_sweep_lib.sh"
k3_check_conc_list "$CONC_LIST" || exit 1
mkdir -p "$RESULTS_DIR"
k3_set_gcs_root "$GCS_ARG"
for kv in "${ENVS[@]}"; do
    [[ "$kv" == *=* ]] || { log "ERROR: --env expects KEY=VAL, got '$kv'"; exit 1; }
    export "${kv?}"
    log "server env: $kv"
done
# For configs that read the harness names instead of the K3_SWEEP_* knobs.
export MAX_MODEL_LEN="$K3_SWEEP_CTX" MAX_NUM_SEQS="$K3_SWEEP_MAX_NUM_SEQS"

cleanup() {
    log "cleanup"
    bash "$SCRIPT_DIR/cleanup_server.sh"
    k3_dump_server_log_context
    if [ -n "$GCS_ROOT" ] && command -v gsutil >/dev/null 2>&1; then
        gsutil -q -m cp -r "$RESULTS_DIR" "$GCS_ROOT/" 2>&1 | tail -3 || log "WARNING: GCS upload to $GCS_ROOT failed"
    fi
}
trap cleanup EXIT INT TERM

log "===== boot: config=$CONFIG_NAME concs=$CONC_LIST ctx=$K3_SWEEP_CTX max_num_seqs=$K3_SWEEP_MAX_NUM_SEQS max_tokens=$MAX_TOKENS env=[${ENVS[*]:-}] extra=[${K3_EXTRA_SERVE_ARGS:-}] ====="
k3_boot_server "$CONFIG_NAME" || exit 1
MODEL=$(k3_config_json_get model)
save_now "$RESULTS_DIR/config.json"
grep -m1 -oE 'vllm serve.*' "$RESULTS_DIR/server.log" | cut -c1-600 | sed 's/^/[k3-conc-probe] serve cmd: /' || true

FAIL=0
IFS=',' read -r -a CONCS <<< "$CONC_LIST"
for n in "${CONCS[@]}"; do
    server_alive || { log "FATAL: server dead before N=$n"; FAIL=1; break; }
    pre0=$(k3_engine_counters)
    log_lines0=$(wc -l < "$RESULTS_DIR/server.log" 2>/dev/null || echo 0)
    python3 "$SCRIPT_DIR/conc_probe.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --conc-list "$n" \
        --max-tokens "$MAX_TOKENS" --out "$RESULTS_DIR/probe_n${n}.json" 2>&1 | tee "$RESULTS_DIR/probe_n${n}.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && FAIL=1
    # Second reading of the peak batch, from the engine's own 10 s stats lines
    # written during this point ("Running: X reqs").
    peak_log=$(tail -n +"$((log_lines0 + 1))" "$RESULTS_DIR/server.log" 2>/dev/null | tr '\r' '\n' | grep -oE 'Running: [0-9]+ reqs' | grep -oE '[0-9]+' | sort -n | tail -1)
    if server_alive; then
        log "N=$n done rc=$rc; engine before=[$pre0] after=[$(k3_engine_counters)]; peak Running in server.log during the point: ${peak_log:-n/a}"
    else
        log "N=$n: SERVER DIED during or after the burst (rc=$rc)"
        FAIL=1; break
    fi
    save_now "$RESULTS_DIR/probe_n${n}.json" "$RESULTS_DIR/probe_n${n}.log" "$RESULTS_DIR/server.log"
    wait_server_idle 600 || true
done
log "===== summary ====="
grep -h '^\[conc-probe\] N=' "$RESULTS_DIR"/probe_n*.log 2>/dev/null || true
[ "$FAIL" = "0" ] || log "probe finished with failures; exiting 1"
exit $FAIL
