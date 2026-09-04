#!/bin/bash
# [branch-local, not for merge] Kimi-K3 InferenceX AgentX trace replay
# (multi-turn agentic-coding sessions, aiperf `--scenario
# inferencex-agentx-mvp`) against the tt benchmark harness's kept-alive
# server on the v7x-32 pod.
#
# Usage (inside the head container, via run_multihost.sh):
#   bash ./scripts/vllm/benchmarking/k3_agentx.sh \
#       [--config kimi-k3-tp32-ep-agentx] [--results-dir DIR] \
#       [--conc 4] [--max-model-len 131072] [--duration 1200] [--fast 1] \
#       [--ix-ref <InferenceX git ref>]
#
# --fast 1 => AIPERF_EXPERIMENTAL_FAST: 20 min profiling window, one warmup
# request per lane (bring-up mode). --fast 0 => the canonical 30 min window
# (DURATION honoured) and 10 warmup requests per lane.
#
# The aiperf client filters out traces whose peak input+output exceeds
# --max-model-len, so a short server context replays a subset of the corpus
# (the client logs how many traces survived). That is NOT a valid AgentX
# submission (canonical runs use the model's native 1M context) — it is the
# TPU bring-up ladder.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-agentx"
RESULTS_DIR="/perf_eval_results/k3-agentx"
CONC=4
MAX_MODEL_LEN_ARG=131072
DURATION=1800
FAST=1
IX_REF="main"
PORT=8000

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --conc)          CONC="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN_ARG="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --fast)          FAST="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        *) echo "[k3-agentx] unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"
trap 'echo "[k3-agentx] cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM

# Server geometry knobs consumed by the config (env-overridable there).
export MAX_MODEL_LEN="$MAX_MODEL_LEN_ARG"
export MAX_NUM_SEQS=$((2 * CONC)); [ "$MAX_NUM_SEQS" -lt 4 ] && export MAX_NUM_SEQS=4
export PORT

echo "[k3-agentx] config=$CONFIG_NAME conc=$CONC max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS duration=$DURATION fast=$FAST ix_ref=$IX_REF"
echo "[k3-agentx] host: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, python $(python3 --version 2>&1), uv $(uv --version 2>&1 || echo missing)"

# ---------------------------------------------------------------- InferenceX
# The official orchestration (benchmarks/benchmark_lib.sh) owns the aiperf
# venv, the trace corpus download, the exact aiperf argv and the result
# aggregation; reuse it verbatim so the numbers are produced the way the
# dashboard produces them.
IX_DIR=/tmp/InferenceX
if [ ! -d "$IX_DIR/utils/aiperf/src" ]; then
    rm -rf "$IX_DIR"
    git clone --depth 1 --branch "$IX_REF" --recurse-submodules --shallow-submodules \
        https://github.com/SemiAnalysisAI/InferenceX "$IX_DIR" \
        || { echo "[k3-agentx] FATAL: cannot clone InferenceX@$IX_REF"; exit 1; }
fi
echo "[k3-agentx] InferenceX $(git -C "$IX_DIR" rev-parse --short HEAD), aiperf $(git -C "$IX_DIR/utils/aiperf" rev-parse --short HEAD)"

# ------------------------------------------------- server (harness bring-up)
set -e
bash "$SCRIPT_DIR/cleanup_server.sh"
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT"
set +e

MODEL=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model'])")
MODEL_URI=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model_uri'] or '')")
TOK_DIR=""
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
    if [ -n "$TOK_DIR" ] && [ -f "$TOK_DIR/config.json" ]; then
        echo "[k3-agentx] client tokenizer: $TOK_DIR"
    else
        echo "[k3-agentx] WARNING: pulled dir '$TOK_DIR' unusable; client will resolve $MODEL on the hub"
        TOK_DIR=""
    fi
fi

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# Single-request probes: plain chat, then a chat turn carrying a tool
# definition + tool result (the shape every AgentX turn has), so a template
# / parser failure shows up here as a 4xx instead of as a 100% error rate
# after a 20-minute replay.
sanity_probe() {
    echo "[k3-agentx] --- probe: /v1/chat/completions ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]; m = c.get('message', {})
print('[k3-agentx] chat content:', repr(m.get('content')), '| reasoning:', repr((m.get('reasoning_content') or '')[:120]), '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
print('[k3-agentx] error:', r.get('error')) if 'error' in r else None
" || echo "[k3-agentx] chat probe FAILED"
    echo "[k3-agentx] --- probe: chat with tools + tool result ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- << JSON | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'error' in r:
    print('[k3-agentx] tool probe ERROR:', json.dumps(r['error'])[:600])
else:
    c = r.get('choices', [{}])[0]; m = c.get('message', {})
    print('[k3-agentx] tool probe content:', repr((m.get('content') or '')[:200]), '| tool_calls:', json.dumps(m.get('tool_calls'))[:300], '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
"
{"model":"$MODEL","max_tokens":128,"temperature":0,
 "tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
 "messages":[{"role":"system","content":"You are a coding agent."},
             {"role":"user","content":"Read README.md and tell me its first line."},
             {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},
             {"role":"tool","tool_call_id":"call_1","content":"# Hello World\nThis is a test repo."}]}
JSON
}
sanity_probe

# ------------------------------------------------------------- aiperf replay
export INFMAX_CONTAINER_WORKSPACE="$IX_DIR"
export AIPERF_RUNTIME_DIR=/tmp/inferencex-agentic
# The image ships python 3.12; let uv reuse it instead of downloading 3.11.
export AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.12}"
export KV_OFFLOADING=none
unset KV_OFFLOAD_BACKEND
export CONC DURATION MODEL
export MODEL_PREFIX=kimik3
export TP=32 EP_SIZE=32 PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
export FRAMEWORK=vllm RUNNER_TYPE=tpu7x-32 PRECISION=fp8 SPEC_DECODING=none
export IMAGE="${BUILDKITE_COMMIT:-local}"
export RESULT_FILENAME="agentx_k3_tpu7x-32_c${CONC}_ctx${MAX_MODEL_LEN}"
export RESULT_DIR="$RESULTS_DIR/agentx"
export AGENTIC_OUTPUT_DIR="$RESULTS_DIR"
export ENABLE_AGENTX_POWER=0        # no nvidia-smi here
export AIPERF_EXPERIMENTAL_FAST="$FAST"
mkdir -p "$RESULT_DIR"

# shellcheck source=/dev/null
source "$IX_DIR/benchmarks/benchmark_lib.sh"

resolve_trace_source            # aiperf venv + corpus download (HF, public)
# The K3 tokenizer's custom code (tokenization_kimi.py) imports tiktoken;
# blobfile is what tiktoken's loader wants for local .model files.
UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" -q blobfile tiktoken \
    || echo "[k3-agentx] WARNING: blobfile install failed"

build_replay_cmd "$RESULT_DIR"
if [ -n "$TOK_DIR" ]; then
    REPLAY_CMD="${REPLAY_CMD/--tokenizer $MODEL/--tokenizer $TOK_DIR}"
fi
echo "[k3-agentx] replay cmd: $REPLAY_CMD"

if ! server_alive; then
    echo "[k3-agentx] FATAL: server down before replay"; exit 1
fi
run_agentic_replay_and_write_outputs "$RESULT_DIR"
rc=$?
echo "[k3-agentx] replay+aggregate rc=$rc"

# ------------------------------------------------------------------ summary
AGG="$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json"
if [ -f "$AGG" ]; then
    echo "[k3-agentx] ===== AgentX aggregate ($AGG) ====="
    python3 - "$AGG" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
def walk(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{prefix}{k}.")
    elif isinstance(o, (int, float, str, bool)) or o is None:
        s = str(o)
        if len(s) < 120:
            print(f"{prefix[:-1]} = {s}")
walk(d)
PYEOF
else
    echo "[k3-agentx] no aggregate JSON at $AGG"
    find "$RESULT_DIR" -maxdepth 2 -print 2>/dev/null | head -40
fi
echo "[k3-agentx] server log tail:"
tail -40 "$RESULTS_DIR/server.log"
exit "$rc"
