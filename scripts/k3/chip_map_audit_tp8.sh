#!/usr/bin/env bash
# Single-host TP8 version of the chip-map audit. Two uses:
#
#   1. Validate the instrumentation before spending a 4-node allocation --
#      the TP32 job is the real question, and finding a bug in the audit hook
#      there costs four nodes instead of one.
#   2. Establish the known-good baseline. On one host the rank->device map is
#      already known to be the identity (see .claude/ENVIRONMENT.md), so this
#      arm MUST print PASS on both checks. If it does not, the audit is wrong,
#      not the topology.
#
# Dummy weights: the split is computed during MoE construction, before any
# weight is read, so nothing needs the checkpoint beyond config and tokenizer.
#
# Usage:
#   srun --partition=dev-2x2x1 -w tpu-rack1-w-014 -N1 -n1 --exclusive \
#        --time=00:30:00 scripts/k3/chip_map_audit_tp8.sh
set -uo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
MODEL=${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}
NUM_EXPERTS=${NUM_EXPERTS:-256}
TP=${TP:-8}
PORT=${PORT:-8021}
SP=${SP:-$REPO/.runtime-cache/chip-map-audit-tp8}
AUDIT_DIR="$SP/chip-map"
SERVER_LOG="$SP/server.log"
AUDIT_TIMEOUT=${AUDIT_TIMEOUT:-1800}

rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"

MODEL_DIR="models--${MODEL//\//--}"
if   [ -d "/mnt/scratch/hf-hub/$MODEL_DIR/snapshots" ]; then export HF_HUB_CACHE=/mnt/scratch/hf-hub
elif [ -d "/mnt/scratch/hf-hub/hub/$MODEL_DIR/snapshots" ]; then export HF_HUB_CACHE=/mnt/scratch/hf-hub/hub
else export HF_HUB_CACHE=/mnt/lustre/hf-models/hub; fi
export HF_HOME="$(dirname "$HF_HUB_CACHE")"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

export PATH="$REPO/.venv/bin:$PATH"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE=tpu
export JAX_PLATFORMS=""
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TORCH_TPU_TOPOLOGY=2,2,1,2
unset TPU_MULTIHOST_BACKEND || true

export TPU_MOE_HIERARCHICAL_EP=1
export TPU_CHIP_MAP_AUDIT_DIR="$AUDIT_DIR"

export HOME="$SP/home"
export XDG_CACHE_HOME="$SP/xdg-cache"
export VLLM_CACHE_ROOT="$SP/vllm"
export VLLM_CONFIG_ROOT="$SP/vllm-config"
export HF_MODULES_CACHE="$SP/hf_modules"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TF_CPP_MIN_LOG_LEVEL=1
export TPU_STDERR_LOG_LEVEL=3
export TPU_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$HF_MODULES_CACHE"

SERVER_PID=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  pkill -9 -f 'VLLM::Worker' 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "== $(hostname -s): $MODEL at TP=$TP, dummy weights"
echo "== audit dir: $AUDIT_DIR"

vllm serve "$MODEL" \
  --tensor-parallel-size="$TP" \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --load-format=dummy \
  --max-model-len=2048 \
  --max-num-batched-tokens=512 \
  --max-num-seqs=4 \
  --block-size=16 \
  --num-gpu-blocks-override=1024 \
  --attention-backend=CUSTOM \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --gpu-memory-utilization=0.80 \
  --port="$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "== waiting up to ${AUDIT_TIMEOUT}s for $TP rank audit files"
START=$(date +%s)
while :; do
  COUNT=$(find "$AUDIT_DIR" -name 'rank-*.json' 2>/dev/null | wc -l)
  [ "$COUNT" -ge "$TP" ] && { echo "== all $TP ranks reported after $(( $(date +%s) - START ))s"; break; }
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "== server exited with $COUNT/$TP files; tail:"; tail -40 "$SERVER_LOG"; break
  fi
  if [ $(( $(date +%s) - START )) -gt "$AUDIT_TIMEOUT" ]; then
    echo "!! timeout with $COUNT/$TP files; tail:"; tail -40 "$SERVER_LOG"; break
  fi
  sleep 5
done
cleanup; SERVER_PID=""

echo
grep -m1 "Chip topology:" "$SERVER_LOG" || echo "   (chip topology not logged)"
echo
python "$REPO/scripts/kernels/collect_chip_map_audit.py" \
  "$AUDIT_DIR" --num-experts "$NUM_EXPERTS"
exit $?
