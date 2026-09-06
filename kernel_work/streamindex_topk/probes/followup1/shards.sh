#!/bin/bash
# Fan-out driver for the four step-3 follow-up probes. One shard per TPU chip.
#
#   shard 0  a2_gate.py       gate option A2 (clamp w <- max(w,0), keep ReLU)
#                             against the float64 ReLU oracle, five regimes.
#                             HOST CPU float64 numpy -- it does not touch the
#                             TPU; A2's accuracy is a property of the
#                             arithmetic and the input distribution, not of the
#                             kernel. The chip it is given goes unused.
#   shard 1  head_ladder.py   NUM_HEADS ladder in the vpu_mxu_strike mirror:
#                             split the 3.858 us head-sum bundle into a term
#                             linear in NUM_HEADS and a constant term.
#                             DECOMPOSITION, NOT AN OPTIMISATION -- a smaller
#                             NUM_HEADS computes a different function.
#   shard 2  bucket_ladder.py option B: is section 5c's linearity in live width
#                             real, and what does bucketing actually cost.
#                             Includes the direct P=1/P=2 read-back.
#   shard 3  bkv844_rerun.py  re-run the (8,4,4) vs (2,2,2) comparison at
#                             mixed_prod that came back at 491.23% spread.
#
# Shard 2 is the long pole (14 full streamindex compiles); shard 0 is the
# shortest. That is unbalanced but there is nothing to trade -- the four probes
# are the four questions, and splitting shard 2 across chips would put two
# arms of one A/B on two different chips, which is exactly what an A/B must
# not do.
#
# Nothing here writes to kernel_work/streamindex_topk/instruments/. Every
# constant that moves is moved by rebinding a module attribute from the probe
# script, and restored in a finally.
set -u
SHARD="$1"
D=/workspace/kernel_work/streamindex_topk/probes/followup1
OUT="${CDK_OUTPUT_DIR:-/tmp}/followup1"
mkdir -p "$OUT"

# Cap host threads: four shards share 16 CPUs. Shard 0 is pure host numpy, so
# this matters most to it. Shards 1-3 are timing measurements on the TPU, where
# the host is only dispatching -- but keep the cap uniform so no shard can
# starve another's dispatch thread.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "===== shard $SHARD ====="
date -u
python3 -c "import numpy, jax, sys; print(numpy.__version__, jax.__version__, sys.version)"
echo "TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-unset} TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-unset}"

case "$SHARD" in
  0)
    echo "--- shard 0: A2 accuracy gate, five regimes, host CPU ---"
    python3 "$D/a2_gate.py" \
      --regimes decode_prod,mixed_prod,mixed_32k,mixed_64k,decode_64k \
      --token-chunk 64,64,32,16,16 \
      --out "$OUT/a2_gate.json"
    ;;
  1)
    echo "--- shard 1: NUM_HEADS ladder (decomposition, not an optimisation) ---"
    python3 "$D/head_ladder.py" \
      --heads 4,8,16,32,64 \
      --reps 30 --batch 8 --warmup 5 \
      --json "$OUT/head_ladder.json"
    ;;
  2)
    echo "--- shard 2: option B width ladder + bucket cost + P read-back ---"
    python3 "$D/bucket_ladder.py" \
      --kv 8192,16384,24576,32768,49152,65536 \
      --early-exit true \
      --json "$OUT/bucket_ladder.json"
    ;;
  3)
    echo "--- shard 3: (2,2,2) vs (8,4,4) at mixed_prod, paired ---"
    python3 "$D/bkv844_rerun.py" \
      --mode both --blocks 60 --reps 40 --early-exit false \
      --json "$OUT/bkv844_rerun.json"
    ;;
  *)
    echo "bad shard: $SHARD"; exit 2 ;;
esac
rc=$?
date -u
echo "shard $SHARD exit rc=$rc"
exit $rc
