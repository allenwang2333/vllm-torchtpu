#!/bin/bash
# Repeated-restart A/B matrix for cheap serving knobs (no code changes).
#
# One invocation = one agent = one host, so every cell it produces is directly
# comparable. Each cell is a FULL server lifetime: fresh `vllm serve`, N
# benchmark repeats, optional completion capture and gsm8k, then teardown.
# That separates the two variances an A/B needs to be judged against:
#
#   within-server  = spread of the N repeats inside one cell
#   across-restart = spread of the cell means for one arm
#
# Arms are given as an ordered list so a caller can write A,B,B,A: with the
# arms interleaved, monotone drift over the job cannot alias onto the arm
# (see lesson 7df107efaae9).
#
# Usage:
#   k3_knob_matrix.sh --config kimi-k3-sliced8-tp8-ep --results-dir DIR \
#       --arms base,flagset,flagset,base [--reps 3] [--probe] [--gsm8k 100]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="kimi-k3-sliced8-tp8-ep"
RESULTS_DIR=""
ARMS=""
REPS=3
PORT=8000
PROBE=0
GSM8K_LIMIT=0
GSM8K_CELLS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)       CONFIG_NAME="$2"; shift 2 ;;
    --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
    --arms)         ARMS="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --probe)        PROBE=1; shift ;;
    --gsm8k)        GSM8K_LIMIT="$2"; shift 2 ;;
    --gsm8k-cells)  GSM8K_CELLS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$ARMS" ] || { echo "ERROR: --arms is required"; exit 2; }
RESULTS_DIR="${RESULTS_DIR:-/tmp/k3_knob_matrix}"
mkdir -p "$RESULTS_DIR"

# The four flags from gh/agent/k3-b1-tpot-optimization-backup's launcher
# (scripts/k3/launch_kimi_k3_tp32_ep.sh), measured there one at a time on
# dummy weights at TP32/ISL=1 and never A/B'd together.
F_LAYER_SCHED="--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
F_ASYNC_MERGER="--xla_tpu_enable_async_collective_merger=true"
F_DECOMPOSE_AR="--xla_tpu_decompose_all_reduce_bidirectional_communication=true"
F_MULTI_OVERLAP="--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true"

# Prints the `KEY=VALUE` overrides that define an arm, one per line.
arm_env() {
  case "$1" in
    base)     ;;
    flagset)  echo "LIBTPU_INIT_ARGS=$F_LAYER_SCHED $F_ASYNC_MERGER $F_DECOMPOSE_AR $F_MULTI_OVERLAP" ;;
    f_sched)  echo "LIBTPU_INIT_ARGS=$F_LAYER_SCHED" ;;
    f_merge)  echo "LIBTPU_INIT_ARGS=$F_ASYNC_MERGER" ;;
    f_decomp) echo "LIBTPU_INIT_ARGS=$F_DECOMPOSE_AR" ;;
    f_overlap) echo "LIBTPU_INIT_ARGS=$F_MULTI_OVERLAP" ;;
    fp8kv)    echo "KV_CACHE_DTYPE=fp8" ;;
    *) echo "__UNKNOWN_ARM__" ;;
  esac
}

IFS=',' read -r -a arm_list <<< "$ARMS"
for arm in "${arm_list[@]}"; do
  if [ "$(arm_env "$arm")" = "__UNKNOWN_ARM__" ]; then
    echo "ERROR: unknown arm '$arm'"; exit 2
  fi
done

trap 'echo "--- matrix cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM

fail=0
cell=0
for arm in "${arm_list[@]}"; do
  cell=$((cell + 1))
  cell_id=$(printf 'cell%02d_%s' "$cell" "$arm")
  cell_dir="$RESULTS_DIR/$cell_id"
  mkdir -p "$cell_dir"
  echo "--- $cell_id ($cell/${#arm_list[@]}) : $(date -u +%H:%M:%S)"

  bash "$SCRIPT_DIR/cleanup_server.sh"

  # Subshell so an arm's exports cannot leak into the next cell.
  (
    while IFS= read -r kv; do
      [ -n "$kv" ] && export "${kv?}"
    done < <(arm_env "$arm")
    export BENCHMARK_WARMUP_RUNS=$((REPS - 1))
    echo "[arm] LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS:-}' KV_CACHE_DTYPE='${KV_CACHE_DTYPE:-}'" \
      | tee "$cell_dir/arm.txt"

    bash "$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" \
      --results-dir "$cell_dir" --keep-alive --port "$PORT" || exit 1

    model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' \
      "$cell_dir/config.json")

    if [ "$PROBE" = "1" ]; then
      python3 "$SCRIPT_DIR/k3_completion_probe.py" --model "$model" \
        --base-url "http://localhost:$PORT" --out "$cell_dir/completions.json" \
        2>&1 | tee "$cell_dir/probe.log"
    fi

    if [ "$GSM8K_LIMIT" != "0" ] && \
       { [ -z "$GSM8K_CELLS" ] || [[ ",$GSM8K_CELLS," == *",$cell,"* ]]; }; then
      # /v1/completions, not /v1/chat/completions: K3 ships no chat template
      # and the hand-written one scores a 0.927 model at 0.0034.
      lm_eval --model local-completions \
        --model_args "model=${model},base_url=http://localhost:$PORT/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,timeout=5000,max_length=8192,max_gen_toks=256" \
        --tasks gsm8k --num_fewshot 5 --limit "$GSM8K_LIMIT" \
        --output_path "$cell_dir/gsm8k" 2>&1 | tee "$cell_dir/gsm8k.log"
    fi
  )
  cell_status=$?
  [ "$cell_status" -eq 0 ] || { echo "::error::$cell_id failed ($cell_status)"; fail=1; }

  bash "$SCRIPT_DIR/cleanup_server.sh"
  # Keep the artifact bundle small: the interesting parts of a 190 KB
  # server.log are the config echo, the KV sizing and any error.
  if [ -f "$cell_dir/server.log" ]; then
    grep -nE 'KV cache|kv_cache|num_gpu_blocks|Maximum concurrency|init engine|ERROR|Traceback|tuned|TuningKey|fallback|page_size|kv_packing' \
      "$cell_dir/server.log" > "$cell_dir/server.grep.txt" 2>/dev/null
    gzip -f "$cell_dir/server.log"
  fi
done

echo "--- summary"
python3 "$SCRIPT_DIR/k3_knob_summary.py" --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/summary.txt"
exit "$fail"
