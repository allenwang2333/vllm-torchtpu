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
    echo "--- shard 0 BASELINE: the existing repo kernel test, unmodified ---"
    cd "$REPO" || exit 2
    python3 -m pytest tests/kernels/test_dsv4_streamindex_topk.py \
      -q -rA -p no:cacheprovider --timeout=2400 \
      -m "not nightly and not multichip" > "$OUT/shard0_pytest.log" 2>&1
    rc=$?
    tail -20 "$OUT/shard0_pytest.log"
    echo "shard0: existing kernel test rc=$rc"
    grep -E 'collected|passed|failed|error' "$OUT/shard0_pytest.log" | tail -5
    echo
    echo "--- shard 0b BASELINE: the two EXISTING tests that Option A would ---"
    echo "--- break: a REAL TPU KV-cache allocate/insert/gather path, and   ---"
    echo "--- the deviceless golden that pins dsa.idx page_shape/row_bytes  ---"
    python3 -m pytest \
      tests/layers/common/test_sparse_mla_attention.py \
      tests/distributed/test_raiden_layout_fingerprint.py \
      -q -rA -p no:cacheprovider --timeout=2400 \
      -m "not nightly and not multichip" > "$OUT/shard0b_pytest.log" 2>&1
    r2=$?
    tail -40 "$OUT/shard0b_pytest.log"
    echo "shard0b: sparse-MLA + raiden-fingerprint rc=$r2"
    grep -E 'collected|passed|failed|skipped|error' "$OUT/shard0b_pytest.log" \
      | tail -5
    echo "--- the golden values Option A changes, quoted from the test ---"
    grep -n -B2 -A3 '"dsa.idx"' tests/distributed/test_raiden_layout_fingerprint.py
    rc=$(( rc + r2 ))
    ;;
  1)
    echo "--- shard 1a BASELINE: the repo tests that PIN THE CACHE FORMAT ---"
    cd "$REPO" || exit 2
    python3 -m pytest \
      tests/layers/vllm/test_dsv4_indexer_op.py \
      tests/layers/vllm/test_tpu_indexer.py \
      tests/kernels/test_mla_kv_cache_utils.py \
      -q -rA -p no:cacheprovider --timeout=1800 \
      -m "not nightly and not multichip" > "$OUT/shard1_pytest.log" 2>&1
    rc=$?
    echo "shard1a: format-pinning tests rc=$rc"
    grep -E 'collected|passed|failed|error' "$OUT/shard1_pytest.log" | tail -5
    echo "--- shard 1b: the two configurations whose ORACLE arm went red ---"
    echo "--- with the boundary-tie diagnostics that say why            ---"
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 4 --tokens 1 --heads 2 --k 512 --decode \
      --bkv-p 2,2,2 --json "$OUT/fmt_ps1024_decode_222.json"
    rc=$(( rc + $? ))
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 4 --tokens 2 --heads 2 --k 512 \
      --bkv-p 2,2,2 --json "$OUT/fmt_ps1024_mixed_live.json"
    rc=$(( rc + $? ))
    ;;
  2)
    # THE WRITER TRIAD, IN SITU.  The profiled baseline
    # (j-074e570b-13e3-4ab7-bb0c) puts reshape-in + scatter_fusion +
    # reshape-out at mla_attention_op.py:104 at 773.48 us per indexer layer,
    # 9.8x the whole scoring kernel, and paid on the early-exit path too.
    # Option A's writer half targets exactly that line, and the 3.342x in hand
    # is from a STANDALONE store probe, which by construction cannot show a
    # triad that exists only because the scatter and the Pallas call want
    # different layouts on the same donated buffer.  So: both, side by side.
    echo "--- shard 2a: TRIAD IN SITU, prefill_only (early exit fires) ---"
    python3 "$P/pf_writer_triad.py" \
      --tokens 1024 --seq-len 1024 --heads 64 --topk 2048 --bucket 65536 \
      --hlo-dir "$OUT/hlo_ee" --json "$OUT/triad_earlyexit.json"
    rc=$?
    echo "--- shard 2b: TRIAD IN SITU, scoring kernel actually runs ---"
    echo "--- seq_len 65536 > topk, so early exit does NOT fire      ---"
    python3 "$P/pf_writer_triad.py" \
      --tokens 1024 --seq-len 65536 --heads 64 --topk 2048 --bucket 65536 \
      --hlo-dir "$OUT/hlo_long" --json "$OUT/triad_long.json" || \
      echo "shard2b: long-context arm did not run (see error above)"
    echo "--- shard 2c: TRIAD IN SITU at a smaller head count ---"
    python3 "$P/pf_writer_triad.py" \
      --tokens 1024 --seq-len 1024 --heads 32 --topk 2048 --bucket 65536 \
      --json "$OUT/triad_h32.json" || true
    ;;
  3)
    echo "--- shard 3a CHANGED: ps1024 DECODE, (3,2,2) = SECOND CALL SITE ---"
    echo "--- deepseek_v4_indexer.py:94; never measured by the exercise    ---"
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 6 --tokens 1 --heads 2 --k 512 --decode \
      --bkv-p 3,2,2 --json "$OUT/fmt_ps1024_decode_322.json"
    rc=$?
    echo "--- shard 3b CHANGED: PRODUCTION ps 1024, SHIPPING bucket 65536 ---"
    echo "--- bucket >> kv_len, so real unwritten lanes are in the buffer ---"
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 4 --tokens 2 --heads 2 --k 512 \
      --bkv-p 2,2,2 --bucket 65536 \
      --json "$OUT/fmt_ps1024_mixed_bucket65536.json"
    rc=$(( rc + $? ))
    echo "--- shard 3c CHANGED: ps 1024, bucket 65536, Option E (4,2,4) ---"
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 8 --tokens 2 --heads 2 --k 512 \
      --bkv-p 4,2,4 --bucket 65536 \
      --json "$OUT/fmt_ps1024_mixed_bucket65536_424.json"
    rc=$(( rc + $? ))
    echo "--- shard 3d ABSOLUTE ANCHOR: k == S_valid, the repo test's own ---"
    echo "--- regime, where the top-k is the whole valid set and ranking  ---"
    echo "--- ties cannot decide the comparison                           ---"
    python3 "$P/pf_format_contract.py" \
      --page-size 1024 --num-pages 2 --tokens 2 --heads 2 --k 2048 \
      --bkv-p 2,2,2 --json "$OUT/fmt_anchor_k_eq_svalid.json"
    rc=$(( rc + $? ))
    ;;
  *)
    echo "bad shard: $SHARD"; exit 2 ;;
esac
date -u
echo "preflight shard $SHARD exit rc=$rc"
exit $rc
