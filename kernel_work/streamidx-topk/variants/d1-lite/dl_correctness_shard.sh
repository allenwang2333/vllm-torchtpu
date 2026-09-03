#!/bin/bash
# D1-lite correctness shard.
#
#   1. the small-shape equivalence grid (`test_d1lite_equivalence.py`), k=64,
#      both early-exit regimes, every hoist mode;
#   2. the incumbent's own unmodified kernel test, as the guard that the
#      mapped-in kernels are the ones under test;
#   3. the production-configuration check (`correctness_prod_d1lite.py`):
#      batch 64, mml 9216, page 1024, k=2048, ctx 1024 / 2048 / 8704.
#
# No `set -e`: each step's rc is recorded and reported.
export DL_VARIANT=${DL_VARIANT:-/root/d1lite}
export DL_INSTRUMENTS=${DL_INSTRUMENTS:-/root/instruments}
export PYTHONPATH="$DL_VARIANT:$DL_INSTRUMENTS:$PYTHONPATH"
OUT=${CDK_OUTPUT_DIR:-/tmp}/d1lite
LABEL=${TPU_FANOUT_SHARD:-3}
PY_SYS=${PY_SYS:-python}
SRC=${DL_SRC:-/root/cloud-devkit/vllm-torchtpu}
mkdir -p "$OUT"
declare -A RC

bash "$DL_VARIANT/dl_readback.sh" "$PY_SYS"

echo "===== D1LITE/$LABEL STEP grid BEGIN ====="
$PY_SYS -m pytest -q --no-header -p no:cacheprovider --timeout=1800 \
    "$DL_VARIANT/test_d1lite_equivalence.py" 2>&1 | tail -120
RC[grid]=${PIPESTATUS[0]}
echo "===== D1LITE/$LABEL STEP grid END rc=${RC[grid]} ====="

echo "===== D1LITE/$LABEL STEP incumbent_kernel_test BEGIN ====="
if [ -f "$SRC/tests/kernels/test_dsv4_streamindex_topk.py" ]; then
  # cd into the checkout so tests/conftest.py and the marker declarations in
  # pyproject.toml resolve.
  ( cd "$SRC" && $PY_SYS -m pytest -q --no-header -p no:cacheprovider \
      --timeout=1800 -m "not nightly and not multichip" \
      tests/kernels/test_dsv4_streamindex_topk.py 2>&1 | tail -40; \
    exit ${PIPESTATUS[0]} )
  RC[incumbent_kernel_test]=$?
else
  echo "[d1lite] $SRC/tests/kernels/test_dsv4_streamindex_topk.py not present"
  RC[incumbent_kernel_test]=97
fi
echo "===== D1LITE/$LABEL STEP incumbent_kernel_test END rc=${RC[incumbent_kernel_test]} ====="

echo "===== D1LITE/$LABEL STEP prod BEGIN ====="
$PY_SYS "$DL_VARIANT/correctness_prod_d1lite.py"
RC[prod]=$?
echo "===== D1LITE/$LABEL STEP prod END rc=${RC[prod]} ====="

echo "===== D1LITE/$LABEL TABLE ====="
FAIL=0
for k in "${!RC[@]}"; do
  printf "%-24s rc=%s\n" "$k" "${RC[$k]}"
  [ "$k" = "incumbent_kernel_test" ] || [ "${RC[$k]}" = "0" ] || FAIL=$((FAIL + 1))
done
echo "===== D1LITE/$LABEL failed_steps=$FAIL ====="
exit $FAIL
