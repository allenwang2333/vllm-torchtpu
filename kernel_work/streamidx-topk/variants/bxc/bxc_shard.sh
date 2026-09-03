#!/bin/bash
# B x C shard entry point.  One argument: the arm (plan | correct | perf | all).
#
# Exists so the `cdk` RUN_CMD stays free of nested quoting -- RUN_CMD is a
# single-quoted YAML scalar that may not contain a single quote.
#
# No `set -e`: a failed arm is a result and must not take the rest of the log
# with it.  The combined rc is accumulated and reported at the end.
#
# ONE unpinned process on the whole 4-chip pod.  These are timings, and a
# fanned-out capture is contended -- the same reason `option-c-head-major`'s
# `numbkv_c.sh` refuses to fan out.
#
# Every arm first reads back the md5 of the kernel modules it will import.  An
# earlier job in this exercise silently ran the *image's* older kernel behind a
# green JobStatus.

ARM=${1:-all}
KW=${KW:-/root/kw}
VD=$KW/variants/bxc
REPO=/root/cloud-devkit/vllm-torchtpu
PY=${PY:-python}
TRIALS=${TRIALS:-9}

echo "===== bxc shard $ARM ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<PYEOF
import hashlib, pathlib
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
for m in (streamindex_topk, sparsecore_topk):
    h = hashlib.md5(pathlib.Path(m.__file__).read_bytes()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
PYEOF
md5sum "$VD"/*.py "$KW"/variants/option-c-head-major/streamindex_topk_c.py \
       "$KW"/variants/option-c-head-major/stage_a_c.py \
       "$KW"/variants/option-b-pipeline/pipeline.py "$KW"/probes/_stage_a.py

cd "$REPO" || exit 9
RC=0

run_arm () {
  echo "----- bxc arm $1 BEGIN -----"
  $PY "$VD/probe_bxc.py" --arm "$1" --trials "$TRIALS" 2>&1
  local r=$?
  echo "----- bxc arm $1 END rc=$r -----"
  [ $r -ne 0 ] && RC=$((RC + 1))
  return $r
}

case "$ARM" in
  all)
    run_arm plan
    run_arm correct
    run_arm perf
    ;;
  *)
    run_arm "$ARM"
    ;;
esac

echo "===== bxc shard $ARM combined rc=$RC ====="
exit $RC
