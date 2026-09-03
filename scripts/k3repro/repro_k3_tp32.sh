#!/usr/bin/env bash
# Reproduce the Kimi-K3 TP32 serving numbers on a 32-device TPU v7x pod.
#
# Standalone: no Buildkite, no in-repo benchmark harness. Everything the run
# needs is in this file. Run it inside the serving container on the Ray HEAD
# host of an already-formed 32-device cluster.
#
# RIG THIS WAS MEASURED ON
#   Kimi-K3 full checkpoint (1561 GB), TP32 + EP32, real weights
#   v7x-32 = 16 chips x 2 cores = 32 devices, spread over 4 hosts, Ray over ssh
#   ISL/OSL 8192/1024, greedy
#
# RESULT (build 360, 3 repeats per cell, agreeing to <0.1%)
#   conc 1   TTFT 6053.3 ms   TPOT 27.98 ms   out 29.5 tok/s
#   conc 2   TTFT 9064.1 ms   TPOT 33.52 ms   out 47.2 tok/s
#   conc 4   TTFT 15126.1 ms  TPOT 35.51 ms   out 62.8 tok/s
#
# NOTE ON WHAT THIS CONFIG IS
#   These are a colleague's flags, reproduced as closely as we can. Two
#   knowingly differ:
#     - TPU_LATENT_PROJ_INTRA_CHIP_TP is NOT set. It needs PR #448's code,
#       which is not on main.
#     - block-size is 16 (their value). Our own measurements put 256 at
#       -42.6% TTFT, so this is faithful rather than fast.
set -uo pipefail

MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
SERVED_NAME="${SERVED_NAME:-moonshotai/Kimi-K3}"
PORT="${PORT:-8000}"
OUT="${OUT:-/tmp/k3repro}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
REPS="${REPS:-3}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"
mkdir -p "$OUT"

# --- environment ------------------------------------------------------------
export TPU_MOE_HIERARCHICAL_EP=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_TORCHTPU_KDA_FUSED=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=33554432 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=33554432"

# LIBTPU_INIT_ARGS is consumed AFTER the HLO is emitted, so torch_tpu's tier-2
# cache in /dev/shm is NOT keyed on it. A warm cache silently reuses an
# executable built with different flags. Clear it or the flags are a no-op.
rm -rf /dev/shm/torch_tpu_cache 2>/dev/null || true

# --- serve ------------------------------------------------------------------
vllm serve "$MODEL_URI" \
  --served-model-name "$SERVED_NAME" --port "$PORT" \
  --tensor-parallel-size=32 --data-parallel-size=1 --enable-expert-parallel \
  --enable-ep-weight-filter --load-format runai_streamer \
  --model-loader-extra-config '{"memory_limit":17179869184}' \
  --max-model-len=10240 --max-num-batched-tokens=8192 --max-num-seqs=32 \
  --block-size 16 --compilation-config '{"compile_sizes":[4,8,16,32,8192]}' \
  --kv-cache-dtype=fp8 --gpu-memory-utilization=0.82 \
  --async-scheduling --no-enable-prefix-caching \
  --attention-backend CUSTOM --trust-remote-code --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

echo "waiting for the server (first boot streams the checkpoint; allow ~25 min)"
for _ in $(seq 1 3600); do
  curl -fsS -o /dev/null "http://localhost:$PORT/health" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died -- root cause:"; grep -aE 'Error|Traceback' "$OUT/server.log" | head -20; exit 1; }
  sleep 5
done
echo "server up"

# vLLM resolves an object-storage URI into a local directory; tokenize from it
# rather than re-resolving the name on the HF hub.
TOK=$(ls -d /root/.cache/vllm/assets/model_streamer/* 2>/dev/null | head -1)
[ -n "$TOK" ] || TOK="$SERVED_NAME"

# --- benchmark --------------------------------------------------------------
for C in $CONCURRENCIES; do
  # num_prompts must exceed the concurrency or the run is starved and every
  # cell above the prompt count returns an identical, meaningless number.
  # Build 360 sent 6 prompts at every concurrency; its c>=8 cells are void.
  N=$(( C * 4 )); [ "$N" -lt 8 ] && N=8
  for R in $(seq 1 "$REPS"); do
    echo "--- concurrency=$C rep=$R prompts=$N"
    vllm bench serve --backend vllm --model "$SERVED_NAME" --tokenizer "$TOK" \
      --trust-remote-code --dataset-name random \
      --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0.0 \
      --num-prompts "$N" --max-concurrency "$C" --ignore-eos --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$OUT" \
      --result-filename "isl${ISL}_osl${OSL}_c${C}_r${R}.json" 2>&1 | tail -18
  done
done

echo; echo "=== summary ==="
python3 - "$OUT" <<'PY'
import json,glob,os,sys,statistics as st
rows={}
for f in sorted(glob.glob(os.path.join(sys.argv[1],'*.json'))):
    d=json.load(open(f)); c=d.get('max_concurrency')
    rows.setdefault(c,[]).append((d['median_ttft_ms'],d['median_tpot_ms'],d['output_throughput'],d['completed']))
print(f"{'conc':>5} {'reps':>4} {'TTFT ms':>10} {'TPOT ms':>9} {'out tok/s':>10} {'done':>5}")
for c in sorted(rows, key=lambda x:(x is None, x)):
    v=rows[c]
    print(f"{str(c):>5} {len(v):>4} {st.median(x[0] for x in v):>10.1f} "
          f"{st.median(x[1] for x in v):>9.2f} {st.median(x[2] for x in v):>10.1f} {v[0][3]:>5}")
PY
