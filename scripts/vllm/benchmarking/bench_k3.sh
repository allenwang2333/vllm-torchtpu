#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Ephemeral-server benchmark runner for Kimi-K3 across 4 hosts (TP=32).
#
# run_benchmarks.sh launches ONE local server and has no concept of headless
# ranks, so it cannot drive a model that spans 4 hosts. This runner keeps its
# disciplines -- max-concurrency, pinned temperature, discarded warmup, JSON
# per cell, fixed seed -- and adds the multi-host launch.
#
# The server is EPHEMERAL: serve -> wait -> benchmark -> evaluate -> exit.
# The process ends, so the pods terminate and the pool frees itself. No
# watchdog, no manual delete.
#
# Rank 0 serves and measures. Ranks 1..N-1 run `vllm serve --headless` with
# identical parallel args and rendezvous over torch.distributed.
#
# Usage: bench_k3.sh --config <name> [--rank N] [--nnodes N] [--leader HOST]
#                    [--results-dir DIR] [--record-identity] [--skip-eval]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME=""; RANK="${JOB_COMPLETION_INDEX:-0}"; NNODES=4; LEADER=""
LEADER_ADDR=""
RESULTS_DIR="/tmp/out/bench"; PORT=8000
RECORD_IDENTITY=0; SKIP_EVAL=0; EXTRA_SERVE=""; CAPTURE=0; CAPTURE_HLO=0
CAPTURE_DIR="/tmp/profiles"; HLO_DIR="/tmp/hlo"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)        CONFIG_NAME="$2"; shift 2 ;;
    --rank)          RANK="$2"; shift 2 ;;
    --nnodes)        NNODES="$2"; shift 2 ;;
    --leader)        LEADER="$2"; shift 2 ;;
    --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --record-identity) RECORD_IDENTITY=1; shift ;;
    --capture)       CAPTURE=1; shift ;;   # xprof trace
    --capture-hlo)   CAPTURE=1; CAPTURE_HLO=1; shift ;;  # ...and the HLO dump
    --capture-dir)   CAPTURE_DIR="$2"; shift 2 ;;
    --no-frontier)   USE_FRONTIER=0; shift ;;
    --skip-eval)     SKIP_EVAL=1; shift ;;
    --extra-serve)   EXTRA_SERVE="$2"; shift 2 ;;   # the knob under test
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$CONFIG_NAME" ] || { echo "ERROR: --config is required"; exit 2; }
LEADER_ADDR="${LEADER:-127.0.0.1}"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
# Stack: every SUPPORTED lever so far, applied before the knob under test, so
# each experiment measures a new lever on top of the current best rather than
# against a bare baseline. --no-frontier opts out for a clean-baseline run.
if [ "${USE_FRONTIER:-1}" = "1" ] && [ -f "$SCRIPT_DIR/configs/kimi-k3-frontier.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/configs/kimi-k3-frontier.sh"
  [ -n "${FRONTIER_ARGS:-}" ] && echo "[frontier] $FRONTIER_ARGS"
  # Exported before the server starts: the kernel-selection env vars are read
  # at import and at compile time, so setting them later selects nothing.
  if [ -n "${FRONTIER_ENV:-}" ]; then
    echo "[frontier-env] $FRONTIER_ENV"
    # shellcheck disable=SC2163
    for kv in $FRONTIER_ENV; do export "$kv"; done
  fi
fi
mkdir -p "$RESULTS_DIR"

SERVE=(--served-model-name "$MODEL"
  --tensor-parallel-size "$TENSOR_PARALLELISM"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --attention-backend "${ATTENTION_BACKEND:-CUSTOM}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.7}"
  --trust-remote-code)
[ "${ENABLE_EP:-false}" = "true" ] && SERVE+=(--enable-expert-parallel)
[ -n "${KV_CACHE_DTYPE:-}" ]       && SERVE+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "${COMPILE_SIZES:-}" ]        && SERVE+=(--compilation-config "{\"compile_sizes\":${COMPILE_SIZES}}")
[ -n "${CHAT_TEMPLATE:-}" ]        && SERVE+=(--chat-template "$CHAT_TEMPLATE")
if [ "$NNODES" -gt 1 ]; then
  SERVE+=(--nnodes "$NNODES" --node-rank "$RANK" --data-parallel-backend mp
          --master-addr "$LEADER" --master-port 29500)
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_SERVE_ARGS:-}" ] && SERVE+=($EXTRA_SERVE_ARGS)
# shellcheck disable=SC2206
[ -n "${FRONTIER_ARGS:-}" ] && SERVE+=($FRONTIER_ARGS)
# the knob under test goes LAST so it can override a frontier setting
# shellcheck disable=SC2206
[ -n "$EXTRA_SERVE" ] && SERVE+=($EXTRA_SERVE)

# ---- profile capture. Armed BEFORE the server starts because --xla_dump_to is
# read at compile time and the profiler dir is a serve-time flag. The bench
# passes below stay unarmed; only the short capture pass at the end is armed, so
# the reported numbers are never profiling-polluted (~+11% TTFT / -4% tput).
# Only the named ranks arm the profiler. Arming all four on the full model kills
# the pod: each rank buffers its own trace in the same host memory that is
# already holding a share of 1.42 TiB of weights, and the serialise on
# /stop_profile spikes it further. Measured twice -- the log reaches "CAPTURE --
# decode window" and the pod vanishes with no traceback. Rank 0 is what the
# analysis reads anyway (v001 captured rank 0 alone).
CAPTURE_RANKS="${CAPTURE_RANKS:-0}"
case " $CAPTURE_RANKS " in *" $RANK "*) CAPTURE_THIS_RANK=1 ;; *) CAPTURE_THIS_RANK=0 ;; esac
if [ "$CAPTURE" = "1" ] && [ "$CAPTURE_THIS_RANK" = "0" ]; then
  echo "rank $RANK: capture enabled for ranks [$CAPTURE_RANKS]; not arming here"
  CAPTURE=0
fi

if [ "$CAPTURE" = "1" ]; then
  mkdir -p "$CAPTURE_DIR"
  # The HLO dump is opt-in and off by default, because it does not fit. A 12L
  # model produced 7356 modules / 469 MB gzipped; as uncompressed text at 93
  # layers that is tens of GB against 47 GB of node ephemeral storage shared
  # with the weight cache. A run that fills the disk dies mid-benchmark and
  # takes the xprof trace -- the thing actually wanted -- down with it.
  if [ "$CAPTURE_HLO" = "1" ]; then
    mkdir -p "$HLO_DIR"
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$HLO_DIR --xla_dump_hlo_as_text"
  fi
  SERVE+=(--profiler-config.profiler=torch
          --profiler-config.torch_profiler_dir="$CAPTURE_DIR"
          --profiler-config.ignore_frontend=true
          --profiler-config.torch_profiler_with_stack=false)
fi

TARGET="${MODEL_URI:-$MODEL}"
[ -n "${MODEL_URI:-}" ] && SERVE+=(--load-format runai_streamer)

# ---- headless ranks: serve, then follow the leader out.
#
# `vllm serve --headless` never returns, so an exec here leaves the rank running
# after the leader finishes. With completions=N the JobSet then never completes
# and the slice keeps its chips until somebody deletes it. Measured once: rank 0
# Completed while ranks 1-3 held 16 chips for 20 minutes.
#
# Instead: run headless in the background and poll the leader. The leader is up
# once its port answers; when it stops answering after having been up, the run
# is over and this rank exits too.
if [ "$RANK" != "0" ]; then
  echo "rank $RANK: headless, following leader $LEADER_ADDR"
  vllm serve "$TARGET" "${SERVE[@]}" --headless > "$RESULTS_DIR/worker.log" 2>&1 &
  WORKER_PID=$!
  LEADER_WAS_UP=0
  WAITED=0
  # A leader that never comes up at all must still release this rank. Without a
  # bound the latch below waits forever, the rank keeps its chips, and in a
  # multi-arm run the NEXT arm's rendezvous then fails with "3/4 clients
  # joined" -- one dead arm silently kills every arm after it. Measured once.
  LEADER_WAIT_MAX="${LEADER_WAIT_MAX:-2700}"
  while true; do
    sleep 20; WAITED=$((WAITED + 20))
    kill -0 "$WORKER_PID" 2>/dev/null || { echo "rank $RANK: worker exited"; break; }
    if curl -sf --max-time 5 "http://${LEADER_ADDR}:${PORT}/health" >/dev/null 2>&1; then
      LEADER_WAS_UP=1
    elif [ "$LEADER_WAS_UP" = "1" ]; then
      echo "rank $RANK: leader went away, shutting down"
      kill -TERM "$WORKER_PID" 2>/dev/null || true
      sleep 10; kill -9 "$WORKER_PID" 2>/dev/null || true
      break
    elif [ "$WAITED" -ge "$LEADER_WAIT_MAX" ]; then
      echo "rank $RANK: leader never came up within ${LEADER_WAIT_MAX}s, giving up"
      echo "rank $RANK: tail of worker.log ----"
      tail -40 "$RESULTS_DIR/worker.log" 2>/dev/null
      echo "rank $RANK: ---- end worker.log"
      kill -TERM "$WORKER_PID" 2>/dev/null || true
      sleep 10; kill -9 "$WORKER_PID" 2>/dev/null || true
      break
    fi
  done
  echo "rank $RANK: done"
  exit 0
fi

# ---- leader
# The server is a background child holding this script's stdout. Without an
# explicit kill it survives the script, `tail`/`grep` downstream never see EOF,
# and a caller running arms in sequence wedges forever. Measured once: arm A
# finished its cells, then the sweep sat for 30 minutes with arm B never
# starting. Trap covers the error paths too.
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM

echo "[cmd] vllm serve $TARGET ${SERVE[*]} --host 0.0.0.0 --port $PORT"
vllm serve "$TARGET" "${SERVE[@]}" --host 0.0.0.0 --port "$PORT" \
  > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!

T0=$SECONDS
MAXWAIT=$(( ${SERVER_READY_WAIT_MIN:-40} * 60 ))
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  (( SECONDS - T0 > MAXWAIT )) && { echo "TIMEOUT after $((SECONDS-T0))s"; tail -120 "$RESULTS_DIR/server.log"; exit 1; }
  # On multi-host the leader's traceback ends at "WorkerProc initialization
  # failed ... in a background process", which names no cause; the cause is in
  # this rank's own worker log. Print both or the failure is undiagnosable
  # after the pod is collected.
  kill -0 "$SERVER_PID" 2>/dev/null || {
    echo "!!! server died"
    # The tail is the LAST frame of a re-raise chain and names nothing. The
    # worker's real error is printed much earlier and has already scrolled
    # past 120 lines by the time the leader notices. Show the first error
    # block as well, which is where the cause actually is.
    echo "---- first error block in server.log ----"
    grep -nE "Error|Exception|Traceback|Killed|out of memory|OOM|RESOURCE_EXHAUSTED|Aborted" \
      "$RESULTS_DIR/server.log" 2>/dev/null | head -40
    echo "---- server.log tail ----"
    tail -120 "$RESULTS_DIR/server.log"
    [ -f "$RESULTS_DIR/worker.log" ] && {
      echo "---- worker.log (this rank) ----"
      tail -60 "$RESULTS_DIR/worker.log"; }
    echo "---- resources ----"
    df -h /tmp 2>/dev/null | tail -1
    free -g 2>/dev/null | head -2
    exit 1; }
  (( (SECONDS - T0) % 180 == 0 )) && echo "[$((SECONDS-T0))s] booting..."
  sleep 10
done
READY_S=$((SECONDS-T0)); echo "SERVER READY after ${READY_S}s"

bench () {   # $1=isl $2=osl $3=concurrency $4=label
  local isl=$1 osl=$2 conc=$3 label=$4
  # Dataset selection. The default `random` generates independent prompts, which
  # is right for measuring steady-state decode but makes prefix caching
  # unmeasurable by construction: there is no shared prefix to hit, so the
  # feature reads as null whether or not it works.
  #
  # `prefix_repetition` builds NUM_PREFIXES distinct prefixes and reuses each
  # across num_prompts/NUM_PREFIXES requests, so every request after the first
  # of its group should hit. isl is split prefix/suffix -- the suffix is the
  # unique part, so a long prefix and short suffix is the agentic shape where
  # caching is worth the most.
  local DS=()
  case "${BENCH_DATASET:-random}" in
    prefix_repetition)
      local pfx="${PREFIX_LEN:-$(( isl - ${SUFFIX_LEN:-256} ))}"
      DS=(--dataset-name prefix_repetition
          --prefix-repetition-prefix-len "$pfx"
          --prefix-repetition-suffix-len "${SUFFIX_LEN:-256}"
          --prefix-repetition-output-len "$osl"
          --prefix-repetition-num-prefixes "${NUM_PREFIXES:-4}") ;;
    *)
      DS=(--dataset-name random --random-input-len "$isl"
          --random-output-len "$osl" --random-range-ratio 0.0) ;;
  esac
  vllm bench serve --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --trust-remote-code \
    "${DS[@]}" \
    --num-prompts $(( conc * ${PROMPTS_PER_CONC:-5} )) \
    --max-concurrency "$conc" --request-rate inf \
    --temperature "${BENCHMARK_TEMPERATURE:-0}" --ignore-eos --seed 42 \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99" \
    --save-result --result-filename "$RESULTS_DIR/${label}.json" 2>&1 | tail -26
}

for cell in ${ISL_OSL_CONFIGS}; do
  isl="${cell%%:*}"; osl="${cell##*:}"
  for conc in ${CONCURRENCY_OPTIONS}; do
    echo "########## WARMUP isl=$isl osl=$osl c=$conc (discarded) ##########"
    bench "$isl" "$osl" "$conc" "warmup_isl${isl}_osl${osl}_c${conc}" >/dev/null 2>&1
    echo "########## BENCH isl=$isl osl=$osl c=$conc ##########"
    bench "$isl" "$osl" "$conc" "isl${isl}_osl${osl}_c${conc}"
    # The verdict on a prefix-caching arm is the hit rate, not the latency: a
    # latency win with a 0% hit rate is something else moving.
    grep -o "Prefix cache hit rate: [0-9.]*%" "$RESULTS_DIR/server.log" 2>/dev/null \
      | tail -1 | sed "s/^/HITRATE isl${isl}_osl${osl}_c${conc} /"
  done
done

# ---- Capture pass: ARMED, short, numbers discarded. Two bracketed windows so
# prefill and decode land in separate xprof runs (they have different bucket
# profiles and one merged window hides both).
if [ "$CAPTURE" = "1" ]; then
  CISL="${CAPTURE_ISL:-8192}"; COSL="${CAPTURE_OSL:-1024}"
  if [ "${CAPTURE_PREFILL:-1}" = "1" ]; then
    echo "########## CAPTURE — prefill window ##########"
    curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" >/dev/null; sleep 2
    bench "$CISL" 1 1 "capture_prefill" >/dev/null 2>&1
    sleep 2; curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null; sleep 30
    echo "CAPTURE after prefill: $(du -sh "$CAPTURE_DIR" 2>/dev/null | cut -f1)"
  fi

  echo "########## CAPTURE — decode window ##########"
  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" >/dev/null; sleep 2
  bench 128 "$COSL" 1 "capture_decode" >/dev/null 2>&1
  sleep 2; curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null; sleep 30

  if [ "$CAPTURE_HLO" = "1" ]; then
    tar czf "$CAPTURE_DIR/hlo.tar.gz" -C "$(dirname $HLO_DIR)" "$(basename $HLO_DIR)" 2>/dev/null && rm -rf "$HLO_DIR"
  fi
  df -h /tmp 2>/dev/null | tail -1 | sed "s/^/CAPTURE disk /"
  cp "$RESULTS_DIR/server.log" "$CAPTURE_DIR/" 2>/dev/null
  echo "CAPTURE local $(du -sh "$CAPTURE_DIR" 2>/dev/null | cut -f1) in $CAPTURE_DIR"
  # Upload is best-effort: a pod without a GCS-writable SA must not fail the run.
  UPLOADED=0
  if [ -n "${CAPTURE_GCS:-}" ]; then
    if gsutil -q -m cp -r "$CAPTURE_DIR"/* "$CAPTURE_GCS/rank${RANK}/" 2>&1 | tail -3; then
      echo "CAPTURE uploaded $CAPTURE_GCS/rank${RANK}/"; UPLOADED=1
    else
      echo "CAPTURE upload FAILED -- artifacts remain in the pod at $CAPTURE_DIR"
    fi
  fi
  # Egress is not guaranteed: these nodes' service account carries
  # devstorage.READ_ONLY, so a pod cannot write to any bucket regardless of
  # bucket IAM. That is a node-pool scope, not something a policy fixes. When
  # the upload does not happen, hold the pod open so the artifacts can be
  # pulled with `kubectl cp` -- otherwise the pod is collected and a capture
  # that succeeded is lost anyway. Measured once: 13 GB captured, 0 retrieved.
  if [ "$UPLOADED" = "0" ] && [ "${CAPTURE_HOLD_S:-0}" != "0" ]; then
    echo "CAPTURE HOLDING pod open ${CAPTURE_HOLD_S}s for retrieval:"
    echo "  kubectl cp $(hostname):${CAPTURE_DIR} ./profiles-rank${RANK}"
    sleep "$CAPTURE_HOLD_S"
    echo "CAPTURE hold elapsed"
  fi
fi

# ---- Tier 0: output identity (own probe -- vllm bench serve --save-result
# strips the per-request lists, so the result JSON carries no generated text)
echo "########## TIER 0 — output identity ##########"
IDBASE="$SCRIPT_DIR/baselines/identity/${CONFIG_NAME}.json"
# FAIL CLOSED. The old form left IDENTITY_FAIL unset on the happy path and the
# summary printed "${IDENTITY_FAIL:-0}", so a probe that never ran -- or that
# died -- still reported a pass. That default masked a real failure twice: on
# v022, and again on v055e/v055f where the probe errored on BOTH cells and
# identity_fail=0 printed anyway. Start at 1; only an explicit success clears it.
IDENTITY_FAIL=1
if [ "$RECORD_IDENTITY" = "1" ]; then
  IDENTITY_MODE=record
  if python3 "$SCRIPT_DIR/identity_probe.py" --record --model "$MODEL" \
      --port "$PORT" --out "$IDBASE"; then
    IDENTITY_FAIL=0
  fi
else
  IDENTITY_MODE=compare
  if [ ! -s "$IDBASE" ]; then
    # --compare against a reference that does not exist is not a pass, and it
    # is not a probe failure either: it means nobody ever recorded one. Say so.
    echo "identity: NO REFERENCE at $IDBASE -- record one from the BASELINE config first"
    IDENTITY_MODE=absent
  elif python3 "$SCRIPT_DIR/identity_probe.py" --compare --model "$MODEL" \
      --port "$PORT" --out "$IDBASE"; then
    IDENTITY_FAIL=0
  fi
fi

# ---- Tier 1: accuracy, against the same server
if [ "$SKIP_EVAL" != "1" ] && [ -n "${EVAL_TASKS:-}" ]; then
  echo "########## TIER 1 — accuracy (${EVAL_TASKS}) ##########"
  bash "$SCRIPT_DIR/run_eval_flow.sh" --config "$CONFIG_NAME" \
       --results-dir "$RESULTS_DIR" --host 127.0.0.1 --port "$PORT" --run-lm-eval \
       2>&1 | tail -40
fi

echo "########## SUMMARY ##########"
echo "ready_s=$READY_S"
for f in "$RESULTS_DIR"/isl*_osl*_c*.json; do
  python3 -c "
import json,sys
d=json.load(open('$f'))
print('$(basename "$f" .json)', '|',
      'out_tok_s=%.2f' % d.get('output_throughput',0),
      'tpot_ms=%.2f' % d.get('median_tpot_ms',0),
      'ttft_ms=%.2f' % d.get('median_ttft_ms',0),
      'completed=%s' % d.get('completed'))"
done
# Report the MODE alongside the result. "identity_fail=0" from a `record` run
# means "a reference was written", NOT "the output was checked" -- reading the
# two as the same thing is what made the false pass invisible.
echo "identity_mode=${IDENTITY_MODE:-none} identity_fail=${IDENTITY_FAIL}"
if [ "${IDENTITY_MODE:-none}" != "compare" ]; then
  echo "identity: NO SEMANTICS CHECK WAS PERFORMED (mode=${IDENTITY_MODE:-none})"
fi
cleanup_server
echo "########## BENCH_K3_DONE ##########"
