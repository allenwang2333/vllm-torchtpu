#!/bin/bash
# Shared helpers for the Kimi-K3 sweep drivers (k3_eval_conc_sweep.sh,
# k3_agentx_sweep.sh). Source it after setting:
#   K3_SWEEP_TAG   log tag, e.g. "k3-eval-sweep"
#   RESULTS_DIR    where the harness and the sweep write results
#   PORT           the vLLM server port
# Everything here talks to the server the caller booted with
# run_benchmarks.sh --keep-alive on 127.0.0.1:$PORT.

log() { echo "[${K3_SWEEP_TAG:-k3-sweep}] $*"; }

# k3_check_conc_list "1,2,4" -> fails with a clear message on a non-integer.
k3_check_conc_list() {
    local c
    IFS=',' read -r -a _concs <<< "$1"
    for c in "${_concs[@]}"; do
        [[ "$c" =~ ^[1-9][0-9]*$ ]] || { log "ERROR: bad concurrency '$c' in '$1' (positive integers, comma separated)"; return 1; }
    done
}

# k3_set_gcs_root gs://bucket/prefix -> GCS_ROOT=gs://bucket/prefix/<build number>,
# or GCS_ROOT="" (no upload) when the argument is empty.
k3_set_gcs_root() {
    GCS_ROOT=""
    [ -n "${1:-}" ] || return 0
    GCS_ROOT="${1%/}/${BUILDKITE_BUILD_NUMBER:-local}"
}

# save_now <path>... : copy each path to $GCS_ROOT/<results dir name>/ right
# away. A cancelled build never runs the exit trap, so every point is saved as
# soon as it is written, not only at exit. No-op without GCS_ROOT or gsutil.
save_now() {
    [ -n "${GCS_ROOT:-}" ] && command -v gsutil >/dev/null 2>&1 || return 0
    local x
    for x in "$@"; do
        [ -e "$x" ] || continue
        gsutil -q -m cp -r "$x" "$GCS_ROOT/$(basename "$RESULTS_DIR")/" 2>&1 | tail -2 \
            || log "WARNING: upload of $x to $GCS_ROOT failed"
    done
}

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# Number of requests the engine holds (running + waiting), from /metrics.
# The anchor stops vllm:num_requests_waiting_by_reason{...} from matching too.
k3_engine_queue_depth() {
    curl -s "http://127.0.0.1:$PORT/metrics" \
        | awk '/^vllm:num_requests_(running|waiting)( |\{)/{s+=$NF} END{print s+0}'
}

# wait_server_idle [timeout_s]: return when running+waiting has been 0 on 3
# polls 10 s apart, so the next point starts on a drained engine.
wait_server_idle() {
    local timeout="${1:-1800}" t0 idle=0 n
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$timeout" ]; do
        n=$(k3_engine_queue_depth)
        if [ "${n%.*}" = "0" ]; then
            idle=$((idle + 1)); [ "$idle" -ge 3 ] && return 0
        else
            idle=0; log "drain: running+waiting=$n"
        fi
        sleep 10
    done
    log "WARNING: engine did not drain within ${timeout}s (running+waiting=$n)"
    return 1
}

# k3_boot_server <config> : cleanup, then run_benchmarks.sh --keep-alive
# (boots the server, runs the config's bench as a smoke test, leaves the server
# up). Returns 1 with a FATAL line when the boot or the smoke bench fails.
k3_boot_server() {
    local config="$1" t0 rc
    bash "$K3_SWEEP_SCRIPT_DIR/cleanup_server.sh"
    t0=$(date +%s)
    "$K3_SWEEP_SCRIPT_DIR/run_benchmarks.sh" --config "$config" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "FATAL: run_benchmarks.sh --config $config failed (rc=$rc): server boot or smoke bench failed; see server.log context below"
        return 1
    fi
    if ! server_alive; then
        log "FATAL: server not answering on port $PORT after run_benchmarks.sh returned 0"
        return 1
    fi
    log "server up after $(( $(date +%s) - t0 )) s (incl. the config's smoke bench)"
}

# k3_config_json_get <key> : a value from the harness's $RESULTS_DIR/config.json.
k3_config_json_get() {
    python3 -c 'import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
        "$RESULTS_DIR/config.json" "$1"
}

# k3_dump_server_log_context : error context + tail of server.log, for the
# exit trap, so a failed boot can be diagnosed from the job log alone.
k3_dump_server_log_context() {
    [ -f "$RESULTS_DIR/server.log" ] || return 0
    log "===== server.log: error context ====="
    tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' \
        | grep -v -E 'ActorHandle|ray_executor_v2.py|Failed to kill|client_mode_hook|auto_init_hook|_raylet|ray/_private' \
        | grep -n -E -B3 -A25 'Traceback|Error|Exception|NotImplemented|Killed|OOM|halt' | tail -150 | cut -c1-400
    log "===== server.log: last 40 non-blank lines ====="
    tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' | tail -40 | cut -c1-300
}

K3_SWEEP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
