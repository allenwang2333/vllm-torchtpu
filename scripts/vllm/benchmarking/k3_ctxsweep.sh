#!/bin/bash
# [branch-local, not for merge] Run the harness context-length sweep for
# Kimi-K3 on the v7x-32 pod and push the results dir to GCS (the multihost
# wrapper's artifact copy-back has come up empty for these steps).
#
# Usage (inside the head container, via run_multihost.sh):
#   bash ./scripts/vllm/benchmarking/k3_ctxsweep.sh [--config NAME] [--results-dir DIR]
set -uo pipefail
CONFIG_NAME="kimi-k3-tp32-ep-agentx-ctxsweep"
RESULTS_DIR="/perf_eval_results/k3-ctxsweep"
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)      CONFIG_NAME="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --max-model-len) export MAX_MODEL_LEN="$2"; shift 2 ;;
        --kv)            export KV_CACHE_DTYPE="$2"; shift 2 ;;
        --isl-osl)       export K3_ISL_OSL_OVERRIDE="$2"; shift 2 ;;
        --extra-serve-args) export K3_EXTRA_SERVE_ARGS="$2"; shift 2 ;;
        *) echo "[k3-ctxsweep] unknown arg: $1"; exit 1 ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"
# shellcheck disable=SC2317  # invoked via trap
cleanup() {
    bash "$SCRIPT_DIR/cleanup_server.sh"
    if [ -f "$RESULTS_DIR/server.log" ]; then
        echo "[k3-ctxsweep] ===== server.log: error context ====="
        tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' \
            | grep -v -E 'ActorHandle|ray_executor_v2.py:566|Failed to kill|client_mode_hook|auto_init_hook|_raylet|ray/_private' \
            | grep -n -E -B3 -A25 'Traceback|Error|Exception|NotImplemented|Killed|OOM|halt' | tail -120 | cut -c1-400
    fi
    echo "[k3-ctxsweep] disk at exit:"; df -h / /tmp /perf_eval_results 2>/dev/null
    if command -v gsutil >/dev/null 2>&1; then
        gcs_dst="gs://cuiq-inference-test/k3/agentx/${BUILDKITE_BUILD_NUMBER:-local}/"
        echo "[k3-ctxsweep] uploading results to $gcs_dst"
        gsutil -q -m cp -r "$RESULTS_DIR" "$gcs_dst" 2>&1 | tail -3 || echo "[k3-ctxsweep] WARNING: GCS upload failed"
    fi
}
trap cleanup EXIT INT TERM
echo "[k3-ctxsweep] config=$CONFIG_NAME results=$RESULTS_DIR max_model_len=${MAX_MODEL_LEN:-config} kv=${KV_CACHE_DTYPE:-config} isl_osl=${K3_ISL_OSL_OVERRIDE:-config} extra=${K3_EXTRA_SERVE_ARGS:-}"; df -h / /tmp 2>/dev/null
bash "$SCRIPT_DIR/cleanup_server.sh"
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR"
rc=$?
echo "[k3-ctxsweep] run_benchmarks rc=$rc"
echo "[k3-ctxsweep] ===== summary ====="
python3 - "$RESULTS_DIR" << 'PYEOF'
import glob, json, os, sys
print("| isl | osl | conc | completed | TTFT med ms | TPOT med ms | out tok/s |")
print("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "isl*_osl*_c*.json"))):
    try:
        r = json.load(open(f)); n = os.path.basename(f)[:-5]
        print(f"| {n} | | | {r.get('completed')} | {r.get('median_ttft_ms',0):.0f} | {r.get('median_tpot_ms',0):.2f} | {r.get('output_throughput',0):.2f} |")
    except Exception as e:
        print(f"| {f} | parse error {e} |")
PYEOF
exit "$rc"
