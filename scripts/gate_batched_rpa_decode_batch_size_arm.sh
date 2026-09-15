#!/usr/bin/env bash
# One arm of the VLLM_RPA_DECODE_BATCH_SIZE correctness gate.
# Runs as one fan-out shard, pinned to one chip by tpu-fanout.sh.
#
# The shard emits the arm at every --num-live in NLIVE. The env var is read at
# JAX trace time, so the arm cannot change inside a process; --num-live can,
# at the cost of one compile each, so the sweep belongs here rather than in
# more shards than the pod has chips.
set -u
BS="$1"
NLIVE="${2:-64 63 33 7}"
REPO=/root/cloud-devkit/vllm-torchtpu
NPZDIR=/tmp/rpa_gate
mkdir -p "$NPZDIR"
cd "$REPO" || exit 90

# Explicit, identical in every arm, so it cannot confound the comparison.
export VLLM_RPA_SKIP_EMPTY_PASS=0
export VLLM_RPA_DECODE_BATCH_SIZE="$BS"

echo "shard arm VLLM_RPA_DECODE_BATCH_SIZE=$BS  num_live sweep: $NLIVE"
echo "  TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-<unset>} TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-<unset>}"
echo "  VLLM_RPA_SKIP_EMPTY_PASS=$VLLM_RPA_SKIP_EMPTY_PASS"

fail=0
for n in $NLIVE; do
  echo
  echo "--- arm $BS, num_live=$n ---"
  python3 scripts/check_batched_rpa_decode_batch_size.py \
      --num-live "$n" --emit "$NPZDIR/arm_b${BS}_n${n}.npz"
  rc=$?
  echo "emit rc=$rc for arm $BS num_live=$n"
  [ "$rc" = 0 ] || fail=$((fail + 1))
done
echo "shard arm $BS: $fail failed emit(s)"
exit $fail
