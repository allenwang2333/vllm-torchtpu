#!/bin/bash
# PREFLIGHT for the composed streamindex_topk candidate (B + A + E), before a
# pair of multi-hour GLM-5.2 end-to-end runs.  Four shards, one per TPU chip.
#
#   shard 0  BASELINE.  The EXISTING repo test for this kernel, unmodified,
#            against the image's own incumbent `src/`.  If this is not green
#            the gate is not usable and a green changed-arm proves nothing.
#   shard 1  BASELINE.  The EXISTING repo tests that pin the KV cache FORMAT
#            (`test_dsv4_indexer_op.py` asserts head_size == 256;
#            `test_tpu_indexer.py` asserts k_cache.head_dim == 129), plus the
#            MLA/KV-cache-util and DSA-gather neighbours, plus the pure
#            arithmetic that decides whether the shipping bucket lands on the
#            sparsecore P = 1 side at max_model_len = 66560.
#   shard 2  CHANGED.  Option A's split layout against the REPO'S OWN
#            independent NumPy oracle, at the smallest Option-A-legal page
#            size, over all three call-site tuples -- with two negative
#            controls that must fail.
#   shard 3  CHANGED.  The same, at the PRODUCTION page_size 1024, at the
#            shipping bucket 65536 (i.e. bucket > kv_len, the configuration
#            that leaves real unwritten lanes), and at Option E's (4,2,4).
#
# Shards 0 and 1 are the baseline arm; 2 and 3 are the changed arm.  They run
# in one job on one pod so the image, the pod and the wall clock are shared.
set -u
SHARD="$1"
P=/workspace/kernel_work/streamindex_topk/probes/preflight
REPO=/root/cloud-devkit/vllm-torchtpu
OUT="${CDK_OUTPUT_DIR:-/tmp}/preflight"
mkdir -p "$OUT"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "===== preflight shard $SHARD ====="
date -u
python3 -c "import vllm_torchtpu, jax, numpy, sys; print('vllm_torchtpu', vllm_torchtpu.__file__); print('jax', jax.__version__, 'numpy', numpy.__version__, sys.version)"
python3 -c "from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m; print('incumbent kernel:', m.__file__)"
echo "TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-unset} TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-unset}"
git -C "$REPO" rev-parse HEAD 2>/dev/null || true

rc=0
case "$SHARD" in
  0)
    echo "--- shard 0: TRIAD IN SITU, prefill_only (early exit FIRES), all ---"
    echo "--- four arms: in situ AND standalone, so the gap between the    ---"
    echo "--- two ratios is measured rather than assumed                   ---"
    python3 "$P/pf_writer_triad.py" \
      --tokens 1024 --seq-len 1024 --heads 64 --topk 2048 --bucket 65536 \
      --hlo-dir "$OUT/hlo_ee" --json "$OUT/triad_earlyexit.json"
    rc=$?
    ;;
  1)
    echo "--- shard 1: TRIAD IN SITU, scoring kernel RUNS (seq_len > topk) ---"
    python3 "$P/pf_writer_triad.py" \
      --tokens 1024 --seq-len 65536 --heads 64 --topk 2048 --bucket 65536 \
      --hlo-dir "$OUT/hlo_long" --json "$OUT/triad_long.json"
    rc=$?
    ;;
  2)
    echo "--- shard 2: is the win driven by T (rows written) or by the    ---"
    echo "--- buffer size?  Sweep T at a fixed buffer.  If the win tracks  ---"
    echo "--- the buffer and not T, the mechanism is the 415x amplified    ---"
    echo "--- round trip, which is what the profile says it is.            ---"
    for T in 128 1024 4096; do
      python3 "$P/pf_writer_triad.py" \
        --tokens $T --seq-len 1024 --heads 64 --topk 2048 --bucket 65536 \
        --reps 20 --json "$OUT/triad_T$T.json" || rc=$(( rc + 1 ))
    done
    ;;
  3)
    echo "--- shard 3: does the win track the buffer?  Halve num_blocks.  ---"
    python3 "$P/pf_writer_triad.py" \
      --num-blocks 208 --tokens 1024 --seq-len 1024 --heads 64 --topk 2048 \
      --bucket 65536 --reps 20 --json "$OUT/triad_nb208.json"
    rc=$?
    echo "--- shard 3b: and at the incumbent lkv_dim if it were 128-tight ---"
    echo "--- i.e. how much of A's win is just deleting the 127 pad bytes ---"
    python3 "$P/pf_writer_triad.py" \
      --num-blocks 415 --tokens 4096 --seq-len 1024 --heads 64 --topk 2048 \
      --bucket 65536 --reps 20 --json "$OUT/triad_T4096.json" || \
      rc=$(( rc + 1 ))
    ;;
  *)
    echo "bad shard: $SHARD"; exit 2 ;;
esac
date -u
echo "preflight shard $SHARD exit rc=$rc"
exit $rc
