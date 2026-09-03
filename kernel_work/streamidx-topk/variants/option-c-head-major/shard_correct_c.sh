#!/bin/bash
# Option C correctness shard.  One process, one arm, because gate 1 rebinds a
# module attribute before pytest collects the suite and two arms in one
# interpreter would race that binding.
#
#   $1 = mode   pytest | recall | all
#   $2 = arm    incumbent | head_minor | head_major | head_major_hbm
#              (ignored by `recall`, which runs all three)
#
# `all` = gate 1 on incumbent (the control that must reproduce 13 passed), then
# on head_minor (the identity check on the copy), then on head_major (the
# gate), then gate 2's recall.  Each pytest arm is a fresh interpreter.
#
# `pytest` is not in the image, so it is installed here.  Nothing else about
# the environment is touched.
#
# No `set -e`: a failing gate is the result and must still print its tail.
V=${V:-/root/kw/variants/option-c-head-major}
OUT=${CDK_OUTPUT_DIR:-/tmp}/optc
mkdir -p "$OUT"
PY=${PY:-python}
MODE=${1:-pytest}
ARM=${2:-head_major}

echo "===== OPTC CORRECT ENV mode=$MODE arm=$ARM ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V/streamindex_topk_c.py"

$PY -m pytest --version >/dev/null 2>&1 || {
  echo "--- installing pytest ---"
  uv pip install --system -q pytest || pip install -q pytest
}
$PY -m pytest --version

TESTF=${TESTF:-/root/cloud-devkit/vllm-torchtpu/tests/kernels/test_dsv4_streamindex_topk.py}
echo "--- test file ---"
md5sum "$TESTF"
# The suite must be byte-identical to the branch's.  Expect
# md5=becf9f23ace9424a7d67cd3b34de9afd at c3a18c95.

run_pytest() {
  echo "===== OPTC GATE1 BEGIN arm=$1 ====="
  $PY "$V/run_correctness_c.py" --mode pytest --arm "$1" --test-file "$TESTF" 2>&1 | tail -80
  echo "===== OPTC GATE1 END arm=$1 rc=${PIPESTATUS[0]} ====="
}

run_recall() {
  echo "===== OPTC GATE2 BEGIN ====="
  $PY "$V/run_correctness_c.py" --mode recall \
      --arms incumbent,head_minor,head_major \
      --json "$OUT/recall.json" 2>&1 | tail -60
  echo "===== OPTC GATE2 END rc=${PIPESTATUS[0]} ====="
}

case "$MODE" in
  pytest) run_pytest "$ARM" ;;
  recall) run_recall ;;
  all)
    run_pytest incumbent
    run_pytest head_minor
    run_pytest head_major
    run_recall
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
exit 0
