#!/bin/bash
# Ask the engine, once per feature, whether Kimi-K3 can have it.
#
# Several K3 features are described in the tree as "off" when they are in fact
# *refused* at engine start, and at least one (--async-scheduling) is off with
# no code reason at all. The difference matters: "off" is a tuning choice,
# "refused" is a missing feature with an owner. This runs one short server per
# knob and records which it is, so the answer is a log line rather than a
# reading of the source.
#
# A refusal is a RESULT, not a failure: the driver always exits 0 and prints
#   K3PROBE name=<probe> status=OK|REFUSED|ERROR ...
# The Buildkite step asserts that every probe reported, not that every probe
# passed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME=""; RESULTS_DIR=""; PROBE_NAME=""; PORT=8000
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)      CONFIG_NAME="$2"; shift 2 ;;
    --name)        PROBE_NAME="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$CONFIG_NAME" ] || { echo "ERROR: --config is required"; exit 2; }
PROBE_NAME="${PROBE_NAME:-$CONFIG_NAME}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/k3_probe_$PROBE_NAME}"
mkdir -p "$RESULTS_DIR"

trap 'bash "$SCRIPT_DIR/cleanup_server.sh" >/dev/null 2>&1' EXIT INT TERM
bash "$SCRIPT_DIR/cleanup_server.sh" >/dev/null 2>&1

"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" --port "$PORT"
rc=$?

server_log="$RESULTS_DIR/server.log"
if [ "$rc" = "0" ]; then
  python3 - "$PROBE_NAME" "$RESULTS_DIR" <<'PY'
import glob, json, os, sys
name, root = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(root, "isl*.json"))):
    d = json.load(open(f))
    print(f"K3PROBE name={name} status=OK cell={os.path.basename(f)[:-5]} "
          f"completed={d.get('completed')} "
          f"ttft_med_ms={d.get('median_ttft_ms', 0):.1f} "
          f"tpot_med_ms={d.get('median_tpot_ms', 0):.3f} "
          f"tpot_p99_ms={d.get('p99_tpot_ms', 0):.3f} "
          f"total_tps={d.get('total_token_throughput', 0):.1f}")
PY
  exit 0
fi

# Non-zero: separate "the engine said no" from "something else broke".
reason=$(grep -m1 -oE '(NotImplementedError|ValueError|AssertionError|RuntimeError)[:(].*' \
           "$server_log" 2>/dev/null | head -c 400)
if [ -n "$reason" ]; then
  echo "K3PROBE name=$PROBE_NAME status=REFUSED reason=${reason}"
else
  echo "K3PROBE name=$PROBE_NAME status=ERROR reason=$(tail -3 "$server_log" 2>/dev/null | tr '\n' ' ' | head -c 400)"
fi
exit 0
