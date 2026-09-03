#!/bin/bash
# Composed-candidate correctness shard.
#
#   1. `readback.sh` -- the md5 gate.  Refuse to run if the mapped-in kernels
#      are not the expected ones; a correctness pass against the image's stale
#      kernel proves nothing about the one that will be measured.
#   2. the incumbent's own unmodified kernel test, as an independent check that
#      the mapped-in `src/` is healthy.  Informational: it is not this
#      candidate's test, and its absence or failure says something about the
#      checkout, not about the composition.
#   3. `correctness_composed.py` -- `np.array_equal` of every composed arm
#      against a separately-constructed incumbent call, over a grid spanning
#      the MIXED launch (where Option C runs), the production decode point
#      (where it must not), and both fast-path-firing contexts.
#
#   $@ = optional perf specs.  If any are given they are handed to
#        `perf_shard.sh` after the correctness steps, so this shard's chip is
#        not idle for the rest of the job.  The shard's exit code is the sum of
#        both halves' failures, so neither half can hide the other.
#
# No `set -e`: each step's rc is recorded and reported.
export CX_VARIANT=${CX_VARIANT:-/root/kw/variants/composed}
export CX_INSTRUMENTS=${CX_INSTRUMENTS:-/root/kw/instruments}
export PYTHONPATH="$CX_VARIANT:$CX_INSTRUMENTS:$PYTHONPATH"
OUT=${CDK_OUTPUT_DIR:-/tmp}/composed
LABEL=${TPU_FANOUT_SHARD:-3}
PY_SYS=${PY_SYS:-python}
SRC=${CX_SRC:-/root/cloud-devkit/vllm-torchtpu}
mkdir -p "$OUT"
declare -A RC

echo "===== CX/$LABEL STEP readback BEGIN ====="
bash "$CX_VARIANT/readback.sh" "$PY_SYS"
RC[readback]=$?
echo "===== CX/$LABEL STEP readback END rc=${RC[readback]} ====="
if [ "${RC[readback]}" != "0" ]; then
  echo "===== CX/$LABEL ABORT: provenance gate failed, refusing to verify ====="
  exit 90
fi

echo "===== CX/$LABEL STEP incumbent_kernel_test BEGIN ====="
if [ -f "$SRC/tests/kernels/test_dsv4_streamindex_topk.py" ]; then
  # cd into the checkout so tests/conftest.py and the marker declarations in
  # pyproject.toml resolve.
  ( cd "$SRC" && $PY_SYS -m pytest -q --no-header -p no:cacheprovider \
      --timeout=1800 -m "not nightly and not multichip" \
      tests/kernels/test_dsv4_streamindex_topk.py 2>&1 | tail -40; \
    exit ${PIPESTATUS[0]} )
  RC[incumbent_kernel_test]=$?
else
  echo "[cx] $SRC/tests/kernels/test_dsv4_streamindex_topk.py not present"
  RC[incumbent_kernel_test]=97
fi
echo "===== CX/$LABEL STEP incumbent_kernel_test END rc=${RC[incumbent_kernel_test]} ====="

echo "===== CX/$LABEL STEP composed_equivalence BEGIN ====="
$PY_SYS "$CX_VARIANT/correctness_composed.py"
RC[composed_equivalence]=$?
echo "===== CX/$LABEL STEP composed_equivalence END rc=${RC[composed_equivalence]} ====="

echo "===== CX/$LABEL CORRECTNESS TABLE ====="
FAIL=0
for k in "${!RC[@]}"; do
  printf "%-24s rc=%s\n" "$k" "${RC[$k]}"
  # `incumbent_kernel_test` is informational -- see the header.
  [ "$k" = "incumbent_kernel_test" ] || [ "${RC[$k]}" = "0" ] || FAIL=$((FAIL + 1))
done
echo "===== CX/$LABEL correctness_failed_steps=$FAIL ====="

PERF_FAIL=0
if [ $# -gt 0 ]; then
  echo "===== CX/$LABEL HANDOFF to perf_shard.sh with $# specs ====="
  bash "$CX_VARIANT/perf_shard.sh" "$@"
  PERF_FAIL=$?
  echo "===== CX/$LABEL perf_failed_specs=$PERF_FAIL ====="
fi

TOTAL=$((FAIL + PERF_FAIL))
echo "===== CX/$LABEL total_failures=$TOTAL (correctness=$FAIL perf=$PERF_FAIL) ====="
exit $TOTAL
