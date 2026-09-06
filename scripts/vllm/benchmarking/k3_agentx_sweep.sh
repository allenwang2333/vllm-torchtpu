#!/usr/bin/env bash
# [k3-agentx branch-local] Kimi-K3 InferenceX AgentX sweep on the v7x-32 pod:
# boot the server ONCE, then run (1) a GSM8K accuracy probe on the fresh server,
# (2) the AgentX trace replay at each concurrency in --conc-list, ascending,
# waiting for the engine to drain between points (same shape as InferenceX's
# benchmarks/multi_node/agentic_srt.sh: aiperf's per-invocation cache-bust
# marker keeps the points' KV keyspaces disjoint, so one live server is valid).
#
# Mirrors the InferenceX B300 vLLM recipe's client contract; server geometry is
# the tt K3 AgentX config (kimi-k3-tp32-ep-agentx.sh). Not for merge.
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-agentx"
RESULTS_DIR="/perf_eval_results/k3-agentx-sweep"
CONC_LIST="1,2,4,8"
MAX_MODEL_LEN_ARG=131072
DURATION=1800
FAST=1
IX_REF="main"
APC=0
KV_DTYPE="fp8"
SPEC="none"            # none | dspark  (dspark needs PR #702 in the branch)
SPEC_TOKENS=7
SPEC_AL=3.84
GSM8K_LIMIT=100        # 0 = skip
GSM8K_CONC=8
PORT=8000

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --conc-list)     CONC_LIST="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN_ARG="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --fast)          FAST="$2"; shift 2 ;;
        --ix-ref)        IX_REF="$2"; shift 2 ;;
        --apc)           APC="$2"; shift 2 ;;
        --kv)            KV_DTYPE="$2"; shift 2 ;;
        --spec)          SPEC="$2"; shift 2 ;;
        --spec-tokens)   SPEC_TOKENS="$2"; shift 2 ;;
        --spec-al)       SPEC_AL="$2"; shift 2 ;;
        --gsm8k-limit)   GSM8K_LIMIT="$2"; shift 2 ;;
        --gsm8k-conc)    GSM8K_CONC="$2"; shift 2 ;;
        *) echo "[k3-sweep] unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a CONCS <<< "$CONC_LIST"
MAX_CONC=1
for c in "${CONCS[@]}"; do
    [[ "$c" =~ ^[1-9][0-9]*$ ]] || { echo "[k3-sweep] bad concurrency '$c'"; exit 1; }
    [ "$c" -gt "$MAX_CONC" ] && MAX_CONC=$c
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"

cleanup() {
    echo "[k3-sweep] cleanup"
    bash "$SCRIPT_DIR/cleanup_server.sh"
    if [ -f "$RESULTS_DIR/server.log" ]; then
        echo "[k3-sweep] ===== server.log: error context ====="
        tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' \
            | grep -v -E 'ActorHandle|ray_executor_v2.py:566|Failed to kill|client_mode_hook|auto_init_hook|_raylet|ray/_private' \
            | grep -n -E -B3 -A25 'Traceback|Error|Exception|NotImplemented|Killed|OOM|halt' | tail -150 | cut -c1-400
        echo "[k3-sweep] ===== server.log: last 40 non-blank lines ====="
        tr '\r' '\n' < "$RESULTS_DIR/server.log" | grep -v '^\s*$' | tail -40 | cut -c1-300
    fi
    echo "[k3-sweep] results dir:"; find "$RESULTS_DIR" -maxdepth 3 2>/dev/null | head -60
    echo "[k3-sweep] disk at exit:"; df -h / /tmp /perf_eval_results /dev/shm 2>/dev/null
    rm -rf /dev/shm/.k3_agentx_scratch
    if command -v gsutil >/dev/null 2>&1; then
        gcs_dst="gs://cuiq-inference-test/k3/agentx/${BUILDKITE_BUILD_NUMBER:-local}/$(basename "$RESULTS_DIR")"
        echo "[k3-sweep] uploading results to $gcs_dst"
        gsutil -q -m cp -r "$RESULTS_DIR" "$(dirname "$gcs_dst")/" 2>&1 | tail -3 || echo "[k3-sweep] WARNING: GCS upload failed"
    fi
}
trap cleanup EXIT INT TERM

# ---- Server geometry --------------------------------------------------------
export MAX_MODEL_LEN="$MAX_MODEL_LEN_ARG"
export MAX_NUM_SEQS=$((2 * MAX_CONC)); [ "$MAX_NUM_SEQS" -lt 4 ] && export MAX_NUM_SEQS=4
export KV_CACHE_DTYPE="$KV_DTYPE"
export PORT
if [ "$APC" = "1" ]; then
    export ENABLE_PREFIX_CACHING=true
    export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
    export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-prompt-tokens-details"
else
    export ENABLE_PREFIX_CACHING=false
fi
if [ "$SPEC" = "dspark" ]; then
    # Synthetic acceptance, as InferenceX pins it (golden AL per draft length).
    export ASYNC_SCHEDULING=false
    export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--speculative-config {\"method\":\"dspark\",\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_TOKENS,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SPEC_AL}"
fi

echo "[k3-sweep] config=$CONFIG_NAME concs=${CONCS[*]} max_num_seqs=$MAX_NUM_SEQS ctx=$MAX_MODEL_LEN kv=$KV_DTYPE apc=$APC spec=$SPEC/$SPEC_TOKENS/$SPEC_AL duration=$DURATION fast=$FAST gsm8k_limit=$GSM8K_LIMIT ix_ref=$IX_REF"
echo "[k3-sweep] host: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, python $(python3 --version 2>&1), uv $(uv --version 2>&1 || echo missing)"
echo "[k3-sweep] disk (container view):"; df -h / /tmp /root/.cache /perf_eval_results /dev/shm 2>/dev/null

# ---- Client scratch (venv on root disk: /dev/shm is noexec) -----------------
SCRATCH=/tmp/.k3_agentx_scratch
DATA_SCRATCH="$SCRATCH"
shm_avail=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${shm_avail:-}" ] && [ "$shm_avail" -gt $((20 * 1024 * 1024)) ]; then
    DATA_SCRATCH=/dev/shm/.k3_agentx_scratch
fi
mkdir -p "$SCRATCH" "$DATA_SCRATCH"
export TMPDIR="$DATA_SCRATCH/tmp"; mkdir -p "$TMPDIR"
export AIPERF_UV_CACHE_DIR="$DATA_SCRATCH/uv-cache"
IX_DIR="$SCRATCH/InferenceX"
if [ ! -d "$IX_DIR/utils/aiperf/src" ]; then
    rm -rf "$IX_DIR"
    git clone --depth 1 --branch "$IX_REF" --recurse-submodules --shallow-submodules \
        https://github.com/SemiAnalysisAI/InferenceX "$IX_DIR" \
        || { echo "[k3-sweep] FATAL: cannot clone InferenceX@$IX_REF"; exit 1; }
fi
echo "[k3-sweep] InferenceX $(git -C "$IX_DIR" rev-parse --short HEAD), aiperf $(git -C "$IX_DIR/utils/aiperf" rev-parse --short HEAD)"

# ---- Boot the server once ---------------------------------------------------
set -e
bash "$SCRIPT_DIR/cleanup_server.sh"
T_BOOT0=$(date +%s)
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT"
set +e
echo "[k3-sweep] server up in $(( $(date +%s) - T_BOOT0 )) s (incl. 4-prompt 8k/1k warmup bench)"

MODEL=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model'])")
MODEL_URI=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model_uri'] or '')")
TOK_DIR=""
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
    if [ -n "$TOK_DIR" ] && [ -f "$TOK_DIR/config.json" ]; then
        echo "[k3-sweep] client tokenizer: $TOK_DIR"
    else
        echo "[k3-sweep] WARNING: pulled dir '$TOK_DIR' unusable; client will resolve $MODEL on the hub"
        TOK_DIR=""
    fi
fi

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# Engine drain between points: vllm:num_requests_running + waiting == 0, 3 polls.
wait_server_idle() {
    local timeout="${AIPERF_DRAIN_TIMEOUT_SECONDS:-1800}" t0 idle=0 n
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$timeout" ]; do
        n=$(curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^vllm:num_requests_(running|waiting)/{s+=$NF} END{print s+0}')
        echo "[k3-sweep] drain: running+waiting=$n"
        if [ "${n%.*}" = "0" ]; then idle=$((idle+1)); [ "$idle" -ge 3 ] && return 0; else idle=0; fi
        sleep 10
    done
    echo "[k3-sweep] WARNING: engine did not drain within ${timeout}s"; return 1
}

sanity_probe() {
    echo "[k3-sweep] --- probe: /v1/chat/completions ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]; m = c.get('message', {})
print('[k3-sweep] chat content:', repr(m.get('content')), '| reasoning:', repr((m.get('reasoning_content') or '')[:120]), '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
print('[k3-sweep] error:', r.get('error')) if 'error' in r else None
" || echo "[k3-sweep] chat probe FAILED"
    echo "[k3-sweep] --- probe: chat with tools + tool result ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- << JSON | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'error' in r:
    print('[k3-sweep] tool probe ERROR:', json.dumps(r['error'])[:600])
else:
    c = r.get('choices', [{}])[0]; m = c.get('message', {})
    print('[k3-sweep] tool probe content:', repr((m.get('content') or '')[:200]), '| tool_calls:', json.dumps(m.get('tool_calls'))[:300], '| finish:', c.get('finish_reason'), '| usage:', r.get('usage'))
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

# ---- (1) GSM8K on the fresh server (before any ignore_eos load) -------------
if [ "$GSM8K_LIMIT" -gt 0 ]; then
    echo "[k3-sweep] ===== GSM8K limit=$GSM8K_LIMIT num_concurrent=$GSM8K_CONC ====="
    EVAL_DIR="$RESULTS_DIR/gsm8k"; mkdir -p "$EVAL_DIR"
    T0=$(date +%s)
    # The server runs --reasoning-parser kimi_k3 (as the B300 recipe does), so the
    # answer arrives in reasoning_content; stock lm_eval reads only content and
    # scores 0 (build 9). InferenceX's sitecustomize patch + gsm8k.yaml handle it.
    PATCH_DIR=$(mktemp -d); cp "$IX_DIR/utils/evals/patches/lm_eval_sitecustomize.py" "$PATCH_DIR/sitecustomize.py"
    PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    lm_eval --tasks "$IX_DIR/utils/evals/gsm8k.yaml" --limit "$GSM8K_LIMIT" --seed "0,1234,None,1234" \
        --model local-chat-completions --apply_chat_template --log_samples \
        --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=$GSM8K_CONC,timeout=1800,tokenized_requests=False,max_length=$MAX_MODEL_LEN" \
        --gen_kwargs "max_tokens=4096,temperature=0,top_p=1" \
        --output_path "$EVAL_DIR" 2>&1 | tail -25
    echo "[k3-sweep] gsm8k wall $(( $(date +%s) - T0 )) s"
    python3 - "$EVAL_DIR" << 'PYEOF'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True):
    r = next(iter(json.load(open(f))["results"].values()), {})
    print("[k3-sweep] GSM8K:", {k: round(v, 4) for k, v in r.items() if isinstance(v, (int, float)) and "exact_match" in k})
PYEOF
    wait_server_idle
fi

# ---- (2) AgentX replay, one concurrency at a time, ascending ----------------
export INFMAX_CONTAINER_WORKSPACE="$IX_DIR"
export AIPERF_RUNTIME_DIR="$SCRATCH/inferencex-agentic"
export HF_HOME="$DATA_SCRATCH/hf_home"
export AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.12}"
export KV_OFFLOADING=none
unset KV_OFFLOAD_BACKEND
export DURATION MODEL
export MODEL_PREFIX=kimik3
export TP=32 EP_SIZE=32 PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
export FRAMEWORK=vllm RUNNER_TYPE=tpu7x-32 PRECISION=fp4
export SPEC_DECODING="$([ "$SPEC" = "dspark" ] && echo mtp || echo none)"
export IMAGE="${BUILDKITE_COMMIT:-local}"
export AGENTIC_OUTPUT_DIR="$RESULTS_DIR"
export ENABLE_AGENTX_POWER=0
export AIPERF_EXPERIMENTAL_FAST="$FAST"
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"

source "$IX_DIR/benchmarks/benchmark_lib.sh"
resolve_trace_source            # aiperf venv + corpus download (HF, public)
UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" -q blobfile tiktoken \
    || echo "[k3-sweep] WARNING: blobfile install failed"

BASE_RESULT_FILENAME="agentx_k3_tpu7x-32_ctx${MAX_MODEL_LEN}_kv${KV_DTYPE}_apc${APC}_spec${SPEC}"
declare -a DONE_CONCS=()
for idx in "${!CONCS[@]}"; do
    c="${CONCS[$idx]}"
    export CONC="$c"
    export RESULT_FILENAME="${BASE_RESULT_FILENAME}_c${c}"
    export RESULT_DIR="$RESULTS_DIR/conc_${c}"
    mkdir -p "$RESULT_DIR"
    echo "[k3-sweep] ===== AgentX conc $c ($((idx+1))/${#CONCS[@]}) ====="
    if ! server_alive; then echo "[k3-sweep] FATAL: server down before conc $c"; break; fi
    build_replay_cmd "$RESULT_DIR"
    if [ -n "$TOK_DIR" ]; then
        REPLAY_CMD="${REPLAY_CMD/--tokenizer $MODEL/--tokenizer $TOK_DIR}"
    fi
    echo "[k3-sweep] replay cmd: $REPLAY_CMD"
    T0=$(date +%s)
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    echo "[k3-sweep] conc $c: replay+aggregate rc=$? wall=$(( $(date +%s) - T0 )) s"
    DONE_CONCS+=("$c")
    if [ "$idx" -lt $(( ${#CONCS[@]} - 1 )) ]; then wait_server_idle; fi
done

# ---- Summary ----------------------------------------------------------------
echo "[k3-sweep] ===== summary (ctx=$MAX_MODEL_LEN kv=$KV_DTYPE apc=$APC spec=$SPEC fast=$FAST) ====="
python3 - "$RESULTS_DIR" "$BASE_RESULT_FILENAME" "${DONE_CONCS[@]}" << 'PYEOF'
import json, os, sys
root, base, concs = sys.argv[1], sys.argv[2], sys.argv[3:]
WANT = ["output_tput_per_gpu", "output_tput_tps", "p90_intvty", "p90_full_response_intvty",
        "p90_e2e_norm_intvty", "mean_ttft", "p90_ttft", "mean_tpot", "mean_input_tokens",
        "mean_output_tokens_actual", "total_requests_completed", "theoretical_cache_hit_rate"]
def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and isinstance(v, (int, float)): out.append(v)
            find(v, key, out)
    elif isinstance(o, list):
        for v in o: find(v, key, out)
print("| conc | " + " | ".join(WANT) + " |")
print("|" + "---|" * (len(WANT) + 1))
for c in concs:
    f = os.path.join(root, f"{base}_c{c}.json")
    if not os.path.exists(f):
        print(f"| {c} | (no aggregate: {os.path.basename(f)}) |"); continue
    d = json.load(open(f)); row = []
    for k in WANT:
        vals = []; find(d, k, vals)
        row.append(f"{vals[0]:.4g}" if vals else "-")
    print(f"| {c} | " + " | ".join(row) + " |")
    errs = []; find(d, "error_categories", errs)
PYEOF
echo "[k3-sweep] B300 reference (out tok/s/chip @ P90 intvty): c1 15.1@203, c2 18.6@133, c4 23.4@121, c8 40.6@64, c16 58.4@32, c24 66.3@22, c48 75.7@12.5, c70 87.6@7.9"
exit 0
