#!/usr/bin/env bash
# One arm of the VLLM_RPA_DECODE_BATCH_SIZE correctness gate.
# Runs as one fan-out shard, pinned to one chip by tpu-fanout.sh.
set -u
BS="$1"
REPO=/root/cloud-devkit/vllm-torchtpu
NPZDIR=/tmp/rpa_gate
mkdir -p "$NPZDIR"
cd "$REPO" || exit 90

# Explicit, identical in every arm, so it cannot confound the comparison.
export VLLM_RPA_SKIP_EMPTY_PASS=0
export VLLM_RPA_DECODE_BATCH_SIZE="$BS"

echo "shard arm VLLM_RPA_DECODE_BATCH_SIZE=$BS"
echo "  TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-<unset>} TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-<unset>}"
echo "  VLLM_RPA_SKIP_EMPTY_PASS=$VLLM_RPA_SKIP_EMPTY_PASS"

python3 scripts/check_batched_rpa_decode_batch_size.py \
    --emit "$NPZDIR/arm_b$BS.npz"
rc=$?
echo "emit rc=$rc for arm $BS"
exit $rc
