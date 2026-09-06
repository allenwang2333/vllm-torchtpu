#!/bin/bash
# Step 4: the COMPOSED candidate against the untouched incumbent. Four shards,
# one per TPU chip, one job.
#
#   shard 0  PERF, shipping bucket, the two regimes the A/B shape is made of.
#            `mixed_64k` is the A/B's own shape and carries 85.7-87.9% of
#            indexer time there; `mixed_confound` is the short-request-inside-a
#            -long-window case. mixed_64k is also re-run at early_exit=True,
#            the checked-in call-site value, to check the win survives it.
#   shard 1  PERF, shipping bucket, the other four step-1 regimes.
#   shard 2  CORRECTNESS at the shapes with REAL UNWRITTEN LANES. This is the
#            shard the recommendation rests on: step 3's poison arm bound
#            bucket == kv_len and therefore had zero unwritten lanes to catch
#            anything on. Run at both (2,2,2) and Option E's (4,2,4).
#   shard 3  CORRECTNESS at the remaining regimes and at the masked-store tail,
#            then the `exact` PERF ceiling at mixed_confound, then the
#            per-bucket compile cost that step 3 left `unmodelled`.
#
# Every arm of every comparison runs in the SAME PROCESS on the SAME chip,
# through the READ-ONLY instruments. Nothing here writes to
# kernel_work/streamindex_topk/instruments/; the two constants that move
# (`NUM_KV_PAGES_PER_BLOCK` for Option E, `inputs.build` for Option A's split
# cache) are moved by rebinding a module attribute and restored in a finally.
#
# Shards 2 and 3 are the long poles: a float64 oracle at a 64k-wide regime is
# minutes of host numpy. They are ordered cheapest-and-most-important first so
# that a shard that runs out of wall clock loses the least.
set -u
SHARD="$1"
D=/workspace/kernel_work/streamindex_topk/final
OUT="${CDK_OUTPUT_DIR:-/tmp}/final"
mkdir -p "$OUT"

# Four shards share 16 CPUs. The oracle shards are host-numpy-heavy, so keep
# the cap uniform rather than letting one shard starve another's dispatch.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "===== shard $SHARD ====="
date -u
python3 -c "import numpy, jax, sys; print(numpy.__version__, jax.__version__, sys.version)"
echo "TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-unset} TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-unset}"
git -C /root/cloud-devkit/vllm-torchtpu rev-parse HEAD 2>/dev/null || true

rc=0
case "$SHARD" in
  0)
    echo "--- shard 0: PERF static bucket, mixed_64k + mixed_confound ---"
    python3 "$D/run_perf.py" \
      --regimes mixed_64k,mixed_confound \
      --bucket-mode static \
      --early-exit-also mixed_64k \
      --reps 10 \
      --json "$OUT/perf_shard0.json"
    rc=$?
    ;;
  1)
    echo "--- shard 1: PERF static bucket, the other four step-1 regimes ---"
    python3 "$D/run_perf.py" \
      --regimes decode_prod,mixed_prod,mixed_32k,decode_64k \
      --bucket-mode static \
      --reps 10 \
      --json "$OUT/perf_shard1.json"
    rc=$?
    ;;
  2)
    echo "--- shard 2: CORRECTNESS where unwritten lanes ACTUALLY EXIST ---"
    echo "--- (bucket > kv_len; the configuration step 3 never oracled) ---"
    python3 "$D/run_correctness.py" \
      --regimes unwritten_decode_8000,unwritten_mixed_24576,mixed_confound \
      --bucket-modes static \
      --tuples '2,2,2;4,2,4' \
      --require-unwritten unwritten_decode_8000,unwritten_mixed_24576,mixed_confound \
      --json "$OUT/correctness_unwritten.json"
    rc=$?
    ;;
  3)
    echo "--- shard 3a: CORRECTNESS, masked-store tail at a shipping bucket ---"
    python3 "$D/run_correctness.py" \
      --regimes tail_static_mixed_8300,mask_mixed_8000,mask_decode_8000 \
      --bucket-modes static,kvlen \
      --tuples '2,2,2' \
      --json "$OUT/correctness_tail.json"
    rc=$?
    echo "--- shard 3b: CORRECTNESS, the production and 64k regimes ---"
    python3 "$D/run_correctness.py" \
      --regimes decode_prod,mixed_prod,decode_64k \
      --bucket-modes static \
      --tuples '2,2,2' \
      --json "$OUT/correctness_regimes.json"
    rc=$(( rc + $? ))
    echo "--- shard 3c: PERF at the UNREACHABLE exact-bucket ceiling ---"
    python3 "$D/run_perf.py" \
      --regimes mixed_confound \
      --bucket-mode exact \
      --reps 10 \
      --json "$OUT/perf_exact_ceiling.json"
    rc=$(( rc + $? ))
    echo "--- shard 3d: compile cost per distinct static bucket ---"
    python3 "$D/run_recompile.py" \
      --regime mixed_64k \
      --json "$OUT/recompile.json"
    rc=$(( rc + $? ))
    ;;
  *)
    echo "bad shard: $SHARD"; exit 2 ;;
esac
date -u
echo "shard $SHARD exit rc=$rc"
exit $rc
