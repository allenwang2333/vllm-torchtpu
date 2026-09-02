#!/bin/bash
# [branch-local, not for merge] Kimi-K3 8k/1k concurrency sweep driven by the
# InferenceX (SemiAnalysis) benchmark client, with a per-concurrency GSM8K
# accuracy check, against the tt benchmark harness's kept-alive server.
#
# Usage (inside the head container, via run_multihost.sh):
#   bash ./scripts/vllm/benchmarking/k3_ix_sweep.sh \
#       [--config kimi-k3-tp32-ep-ix] [--results-dir DIR] \
#       [--concs "8 16 32 64 128"] [--gsm8k-limit 200] [--profile-conc 0]
#
# --profile-conc > 0 switches to profile mode: no sweep, one profiled
# InferenceX run at that concurrency (use with the -ix-profile config).
set -uo pipefail

CONFIG_NAME="kimi-k3-tp32-ep-ix"
RESULTS_DIR="/perf_eval_results/k3-ix"
SWEEP_CONCS="8 16 32 64 128"
GSM8K_LIMIT=200
PROFILE_CONC=0
SKIP_BENCH=0
PORT=8000

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG_NAME="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --concs)        SWEEP_CONCS="$2"; shift 2 ;;
        --gsm8k-limit)  GSM8K_LIMIT="$2"; shift 2 ;;
        --profile-conc) PROFILE_CONC="$2"; shift 2 ;;
        --skip-bench)   SKIP_BENCH=1; shift ;;
        *) echo "[k3-ix-sweep] unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/sweep_summary.md"
trap 'echo "[k3-ix-sweep] cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM

echo "[k3-ix-sweep] config=$CONFIG_NAME concs='$SWEEP_CONCS' gsm8k_limit=$GSM8K_LIMIT profile_conc=$PROFILE_CONC"

# ---------------------------------------------------------------- InferenceX
IX_DIR=/tmp/InferenceX
if [ ! -d "$IX_DIR" ]; then
    git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX "$IX_DIR" \
        || { echo "[k3-ix-sweep] FATAL: cannot clone InferenceX"; exit 1; }
fi
LM_EVAL_PATCH_DIR=$(mktemp -d)
cp "$IX_DIR/utils/evals/patches/lm_eval_sitecustomize.py" "$LM_EVAL_PATCH_DIR/sitecustomize.py"
# Pin lm-eval to the ref InferenceX's sitecustomize patch targets (their
# harness does the same force-reinstall). The image ships 0.5.0.dev1, whose
# response objects the patch mis-parses into empty strings -> gsm8k scored
# 0.0 on both filters (dev build 346). --no-deps: runtime deps are present.
pip install -q --no-deps --force-reinstall \
    "git+https://github.com/EleutherAI/lm-evaluation-harness.git@b315ef3b05176acc9732bb7fdec116abe1ecc476" \
    || echo "[k3-ix-sweep] WARNING: lm-eval pin install failed; using image version"
python3 -c "import lm_eval; print('[k3-ix-sweep] lm_eval', lm_eval.__version__)" || true

# ------------------------------------------------- server (harness bring-up)
set -e
bash "$SCRIPT_DIR/cleanup_server.sh"
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" --keep-alive --port "$PORT"
set +e

MODEL=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model'])")
MODEL_URI=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/config.json'))['model_uri'] or '')")
TOK_ARGS=()
if [ -n "$MODEL_URI" ]; then
    TOK_DIR=$(python3 -c "import sys; from vllm.transformers_utils.runai_utils import ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)" "$MODEL_URI" 2>/dev/null | tail -1)
    if [ -n "$TOK_DIR" ] && [ -f "$TOK_DIR/config.json" ]; then
        TOK_ARGS=(--tokenizer "$TOK_DIR")
        echo "[k3-ix-sweep] client tokenizer: $TOK_DIR"
    else
        echo "[k3-ix-sweep] WARNING: pulled dir '$TOK_DIR' unusable; client will resolve $MODEL on the hub"
    fi
fi

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# Single-request generation-quality probes: plain completion (no chat
# template) vs chat completion. Discriminates broken chat-template handling
# from genuine garbage generation (gsm8k scored 0.0 with 1-token junk
# responses in dev builds 346/348 while the content-blind bench passed).
sanity_probe() {
    echo "[k3-ix-sweep] --- sanity probe: pulled dir contents ---"
    if [ -n "${TOK_DIR:-}" ]; then
        find "$TOK_DIR" -maxdepth 1 -printf '%s\t%p\n' 2>/dev/null | head -20
        python3 - "$TOK_DIR" << 'PYEOF'
import json, os, sys
d = sys.argv[1]
tc = os.path.join(d, "tokenizer_config.json")
if os.path.exists(tc):
    cfg = json.load(open(tc))
    ct = cfg.get("chat_template")
    print(f"[k3-ix-sweep] tokenizer_config: keys={sorted(cfg)[:12]} chat_template_len={len(ct) if isinstance(ct,str) else type(ct).__name__}")
else:
    print("[k3-ix-sweep] no tokenizer_config.json in pulled dir")
PYEOF
    fi
    echo "[k3-ix-sweep] --- sanity probe: /v1/completions (no template) ---"
    curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); c=r.get('choices',[{}])[0]; print('[k3-ix-sweep] completion text:', repr(c.get('text'))); print('[k3-ix-sweep] finish_reason:', c.get('finish_reason'))" \
        || echo "[k3-ix-sweep] completions probe FAILED"
    echo "[k3-ix-sweep] --- sanity probe: /v1/chat/completions ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r.get('choices', [{}])[0]
m = c.get('message', {})
print('[k3-ix-sweep] chat content:', repr(m.get('content')))
print('[k3-ix-sweep] chat reasoning_content:', repr((m.get('reasoning_content') or '')[:200]))
print('[k3-ix-sweep] finish_reason:', c.get('finish_reason'))
" || echo "[k3-ix-sweep] chat probe FAILED"
}

run_ix_bench() {
    # run_ix_bench <concurrency> <num_prompts> <num_warmups> <result_stem> [--profile]
    local c="$1" np="$2" warm="$3" stem="$4"; shift 4
    mkdir -p "$RESULTS_DIR/ix"
    python3 "$IX_DIR/utils/bench_serving/benchmark_serving.py" \
        --model "$MODEL" \
        --backend vllm \
        --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random \
        --random-input-len 8192 --random-output-len 1024 \
        --random-range-ratio 1.0 \
        --num-prompts "$np" --max-concurrency "$c" \
        --request-rate inf --ignore-eos \
        --num-warmups "$warm" \
        "${TOK_ARGS[@]}" --trust-remote-code \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$RESULTS_DIR/ix" \
        --result-filename "$stem.json" \
        "$@" 2>&1 | tee "$RESULTS_DIR/ix/$stem.log"
    return "${PIPESTATUS[0]}"
}

run_gsm8k() {
    # run_gsm8k <concurrency>
    local c="$1"
    local out="$RESULTS_DIR/gsm8k_c$c"
    PYTHONPATH="$LM_EVAL_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
        --tasks "$IX_DIR/utils/evals/gsm8k.yaml" \
        --output_path "$out" --log_samples \
        --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=$c,timeout=1800,tokenized_requests=False,max_length=9216" \
        --gen_kwargs "max_tokens=4096,temperature=0,top_p=1" \
        --limit "$GSM8K_LIMIT" 2>&1 | tee "$out.log"
    local rc="${PIPESTATUS[0]}"
    # Echo two sample responses into the job log so a zero score is
    # diagnosable without waiting for artifacts (empty vs truncated vs
    # format mismatch).
    python3 - "$out" << 'PYEOF'
import glob, json, sys
files = sorted(glob.glob(sys.argv[1] + "/**/samples_gsm8k*.jsonl", recursive=True))
if files:
    with open(files[-1]) as fh:
        for i, line in enumerate(fh):
            if i >= 2: break
            s = json.loads(line)
            resp = (s.get("resps") or [[""]])[0][0]
            print(f"[k3-ix-sweep] sample {i}: filtered={s.get('filtered_resps')} resp_len={len(resp)} resp_tail={resp[-300:]!r}")
else:
    print("[k3-ix-sweep] no samples jsonl found under", sys.argv[1])
PYEOF
    return "$rc"
}

sanity_probe

# ------------------------------------------------------------- profile mode
if [ "$PROFILE_CONC" -gt 0 ]; then
    echo "[k3-ix-sweep] PROFILE MODE at concurrency $PROFILE_CONC"
    # The harness's built-in bench already ran one profiled session on the
    # 4-prompt warmup traffic; drop those traces so the dir holds only the
    # real session below (phase tracking resets per /start_profile session).
    if [ -d "$RESULTS_DIR/profile" ]; then
        rm -rf "${RESULTS_DIR:?}/profile"/*
    fi
    c=$PROFILE_CONC
    np=$((2 * c)); [ "$np" -lt 16 ] && np=16
    run_ix_bench "$c" "$np" 4 "ix_profiled_8k1k_c$c" --profile
    rc=$?
    echo "[k3-ix-sweep] profiled bench rc=$rc"
    nx=$(find "$RESULTS_DIR/profile" -name '*.xplane.pb' 2>/dev/null | wc -l)
    echo "[k3-ix-sweep] xplane traces: $nx"
    find "$RESULTS_DIR/profile" -name '*.xplane.pb' 2>/dev/null | head -40
    [ "$nx" -gt 0 ] || { echo "[k3-ix-sweep] FATAL: profiler armed but no xplane traces"; exit 1; }
    exit 0
fi

# --------------------------------------------------------------- sweep mode
{
    echo "# K3 8k/1k InferenceX sweep — $(date -u +%Y-%m-%dT%H:%MZ)"
    echo ""
    echo "| conc | bench | out tok/s | med TTFT ms | med TPOT ms | completed | gsm8k strict |"
    echo "|---|---|---|---|---|---|---|"
} > "$SUMMARY"

for c in $SWEEP_CONCS; do
    if ! server_alive; then
        echo "[k3-ix-sweep] server DOWN before conc=$c; aborting remaining cells"
        echo "| $c | SERVER DOWN | - | - | - | - | - |" >> "$SUMMARY"
        break
    fi
    np=$((4 * c)); [ "$np" -lt 32 ] && np=32
    warm=$((2 * c)); [ "$warm" -gt 32 ] && warm=32
    echo "[k3-ix-sweep] ===== conc=$c num_prompts=$np ====="
    if [ "$SKIP_BENCH" = "1" ]; then
        bench_rc=99
        echo "[k3-ix-sweep] bench skipped (--skip-bench)"
    else
        run_ix_bench "$c" "$np" "$warm" "ix_8k1k_c$c"
        bench_rc=$?
    fi

    bench_row="- | - | - | -"
    if [ $bench_rc -eq 0 ]; then
        bench_row=$(python3 - "$RESULTS_DIR/ix/ix_8k1k_c$c.json" << 'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    print(f"{r.get('output_throughput', 0):.1f} | {r.get('median_ttft_ms', 0):.0f} | {r.get('median_tpot_ms', 0):.2f} | {r.get('completed', '?')}")
except Exception as e:
    print(f"parse-error {e} | - | - | -")
PYEOF
)
    fi

    gsm8k_cell="-"
    if server_alive; then
        run_gsm8k "$c"
        gsm8k_rc=$?
        if [ $gsm8k_rc -eq 0 ]; then
            gsm8k_cell=$(python3 - "$RESULTS_DIR/gsm8k_c$c" << 'PYEOF'
import json, glob, sys
try:
    f = sorted(glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True))[-1]
    m = json.load(open(f))["results"]["gsm8k"]
    print(f"{m.get('exact_match,strict-match', '?')}")
except Exception as e:
    print(f"err:{e}")
PYEOF
)
        else
            gsm8k_cell="FAILED(rc=$gsm8k_rc)"
        fi
    else
        gsm8k_cell="SERVER DOWN"
    fi

    if [ $bench_rc -eq 0 ]; then bench_state=ok
    elif [ $bench_rc -eq 99 ]; then bench_state=skipped
    else bench_state="FAILED(rc=$bench_rc)"; fi
    echo "| $c | $bench_state | $bench_row | $gsm8k_cell |" >> "$SUMMARY"
    echo "[k3-ix-sweep] conc=$c bench=$bench_state gsm8k=$gsm8k_cell"
done

echo "[k3-ix-sweep] ===== SUMMARY ====="
cat "$SUMMARY"
exit 0
