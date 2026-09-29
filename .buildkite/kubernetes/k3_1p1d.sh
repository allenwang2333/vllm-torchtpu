#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One pod of the Kimi-K3 1P1D test. Prefill and decode each get a 2x2x4 slice,
# four hosts, and run one Ray engine across it.
#
#   k3_1p1d.sh prefill|decode          every pod of the role's slice
#   k3_1p1d.sh --head prefill|decode   what multihost_entry.sh runs on index 0
#
# Both engines serve the nightly kimi-k3-tp32-ep config with TPURaidenConnector
# as producer or consumer. The prefill head then fronts the pair with the toy
# proxy, sends a few completions through it and exits, and the JobSet's success
# policy tears decode down.
#
# Environment, from manifests/workloads/kimi-k3-1p1d.yaml: POD_IP,
# DECODE_HOST (decode's index 0), and what multihost_entry.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
PORT=8000
PROXY_PORT=8100

head=0
if [[ "${1:-}" == "--head" ]]; then
  head=1
  shift
fi
role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 [--head] prefill|decode" >&2; exit 2 ;;
esac

if [[ "$head" == "0" ]]; then
  # Set before Ray starts on every host: the engine's workers are Ray actors on
  # all four, and they inherit the raylet's environment, not the head's.
  export VLLM_HOST_IP="${POD_IP}"
  export TPU_KV_RESHARD_TRANSPORT=raiden
  export TPU_RAIDEN_RESHARD_IMPL=store
  export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
  export TPU_RAIDEN_KIMIK3_ADMISSION=1
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export TPU_RAIDEN_TRANSFER_PARALLELISM=8
  export TPU_KV_TRANSFER_PORT=9100
  export TPU_SIDE_CHANNEL_PORT=9600
  export TPU_KV_SHM_POOL_GB=8
  export TPU_P2P_WAIT_PULL_TIMEOUT=600
  export TPU_PREMAPPED_BUFFER_SIZE=17179869184
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-14400}"
  # Identity and ports, which must differ so the two engines do not collide.
  case "$role" in
    prefill)
      export TPU_KV_TRANSFER_NAMESPACE=k3_prefill
      export TPU_RAIDEN_ENGINE_ID=k3-prefill-engine
      export TPU_RAIDEN_JOB_NAME=prefill
      export TPU_RAIDEN_RESHARD_PORT_BASE=27000
      export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
      ;;
    decode)
      export TPU_KV_TRANSFER_NAMESPACE=k3_decode
      export TPU_RAIDEN_ENGINE_ID=k3-decode-engine
      export TPU_RAIDEN_JOB_NAME=decode
      export TPU_RAIDEN_RESHARD_PORT_BASE=28000
      export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
      ;;
  esac
  exec bash "$HERE/multihost_entry.sh" bash "$HERE/k3_1p1d.sh" --head "$role"
fi

cd "$REPO_DIR"
run_dir="/tmp/k3_1p1d/${role}"
mkdir -p "$run_dir"

kv_role=kv_producer
[[ "$role" == "decode" ]] && kv_role=kv_consumer
# No spaces: run_benchmarks.sh word-splits the command it builds.
kv_config="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${kv_role}\",\"kv_port\":14579}"

serve() {
  EXTRA_SERVE_ARGS="--kv-transfer-config ${kv_config}" \
    bash scripts/vllm/benchmarking/run_benchmarks.sh \
    --config kimi-k3-tp32-ep --serve-only --port "$PORT" --results-dir "$run_dir"
}

# The server writes to a file; follow it here so the pod's log carries it.
touch "$run_dir/server.log"
tail -n +1 -F "$run_dir/server.log" 2>/dev/null | sed -u "s/^/[${role} server] /" &
tail_pid=$!

if [[ "$role" == "decode" ]]; then
  serve
  exit $?
fi

serve &
serve_pid=$!
proxy_pid=""
cleanup() {
  [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null || true
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
  kill "$tail_pid" 2>/dev/null || true
}
trap cleanup EXIT

# wait_ready <url> <what> [pid]: poll until the URL answers 200, failing early
# when the named process has died.
wait_ready() {
  local url="$1" what="$2" pid="${3:-}"
  local deadline=$((SECONDS + 250 * 60))
  echo "--- Waiting for ${what} at ${url}"
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 "$url" || true)" == "200" ]]; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: ${what} exited before it was ready"
      return 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "ERROR: ${what} not ready after 250 minutes"
      return 1
    fi
    sleep 30
  done
  echo "${what} is ready"
}

wait_ready "http://127.0.0.1:${PORT}/health" "prefill engine" "$serve_pid"
wait_ready "http://${DECODE_HOST}:${PORT}/health" "decode engine"

echo "--- Starting the proxy on port ${PROXY_PORT}"
python3 examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port "$PROXY_PORT" \
  --prefiller-hosts 127.0.0.1 --prefiller-ports "$PORT" \
  --decoder-hosts "$DECODE_HOST" --decoder-ports "$PORT" \
  >"$run_dir/proxy.log" 2>&1 &
proxy_pid=$!
if ! timeout 120 bash -c "until curl -sf -o /dev/null http://127.0.0.1:${PROXY_PORT}/healthcheck; do sleep 2; done"; then
  echo "ERROR: proxy not ready"
  tail -50 "$run_dir/proxy.log" || true
  exit 1
fi

echo "--- Completions through the proxy"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}/v1/completions" python3 - <<'PY'
import json
import os
import sys
import urllib.request

# Prompts whose greedy continuation is not in doubt, so a wrong or empty answer
# means the decode engine did not get the prefill's KV, not that the model
# phrased something differently.
cases = [
    ("The capital of France is", "paris"),
    ("1, 2, 3, 4, 5, 6,", "7"),
    ("The chemical symbol for gold is", "au"),
]
failed = 0
for prompt, expected in cases:
    body = json.dumps({
        "model": "moonshotai/Kimi-K3",
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(os.environ["PROXY_URL"], data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            text = json.load(response)["choices"][0]["text"]
    except Exception as error:  # noqa: BLE001 - report every case
        print(f"FAIL {prompt!r}: {error}")
        failed += 1
        continue
    ok = expected in text.lower()
    print(f"{'PASS' if ok else 'FAIL'} {prompt!r} -> {text!r}")
    failed += not ok
sys.exit(1 if failed else 0)
PY
