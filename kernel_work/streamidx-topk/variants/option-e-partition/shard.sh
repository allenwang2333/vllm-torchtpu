#!/bin/bash
# Option E shard entry point.  One argument: the shard name.
#
# Exists so the `cdk` RUN_CMD stays free of nested quoting -- `RUN_CMD` is a
# single-quoted YAML scalar that may not contain a single quote, and pytest's
# `-m "not nightly and not multichip"` needs double quotes of its own.
#
# No `set -e`: a failed arm is a result and must not take the shard's log with
# it.  The last command in each branch is the one whose exit code is reported.

ARM=$1
KW=${KW:-/root/kw}
VD=$KW/variants/option-e-partition
REPO=/root/cloud-devkit/vllm-torchtpu
PY=${PY:-python}

export PYTHONPATH="$VD:$PYTHONPATH"

echo "===== option-e shard $ARM ====="
$PY -c "import vllm_torchtpu; print(\"vllm_torchtpu.__file__:\", vllm_torchtpu.__file__)"
$PY -c "from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as m; print(\"sparsecore_topk.__file__:\", m.__file__)"
$PY -c "import sys; sys.path.insert(0, \"$VD\"); import partition_v2 as v; print(\"partition_v2:\", v.__file__)"

case "$ARM" in
  pytest-optione)
    cd "$REPO" || exit 9
    OPTION_E=1 $PY -m pytest -v --timeout=900 -p no:cacheprovider \
      -p option_e_patch -m "not nightly and not multichip" \
      tests/kernels/test_dsv4_streamindex_topk.py
    ;;
  pytest-control)
    cd "$REPO" || exit 9
    OPTION_E=0 $PY -m pytest -v --timeout=900 -p no:cacheprovider \
      -p option_e_patch -m "not nightly and not multichip" \
      tests/kernels/test_dsv4_streamindex_topk.py
    ;;
  *)
    cd "$REPO" || exit 9
    $PY "$VD/probe_option_e.py" --arm "$ARM"
    ;;
esac
RC=$?
echo "===== option-e shard $ARM rc=$RC ====="
exit $RC
