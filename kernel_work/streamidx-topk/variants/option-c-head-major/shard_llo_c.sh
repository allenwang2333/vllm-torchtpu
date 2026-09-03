#!/bin/bash
# Option C LLO shard: compile the REAL stage-A program for one or more arms
# under the `/xprof-llo` dump-side LIBTPU_INIT_ARGS, one arm per process
# (libtpu reads LIBTPU_INIT_ARGS once at load time; two dump flags aimed at one
# directory interleave into an unreadable mess).
#
#   $@ = arm names, run sequentially in this shard
#
# Dumps land in /tmp (node-local, gigabytes).  Only the three passes the reader
# parses are copied back to $CDK_OUTPUT_DIR:
#   *41-post-DCE-3   opcode census / rotate attribution (pre-RA, SSA)
#   *53-post-ra      spill stores, fill loads, spill slots  <- the whole point
#   *70-schedule-analysis_final_bundles   scheduled bundle count
#
# Compile-time analysis only.  Nothing here is a timing, and a fanned-out
# shard's wall clock is contended anyway.
#
# No `set -e`: a failed arm is a result and must not take the others with it.
V=${V:-/root/kw/variants/option-c-head-major}
OUT=${CDK_OUTPUT_DIR:-/tmp}/optc
mkdir -p "$OUT"
PY=${PY:-python}

echo "===== OPTC ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
# The image at ec4bcecf predates this kernel, so `src` is --map-dir'd in over
# its checkout.  A mapping that did not take is otherwise silent and the
# `incumbent` control would quietly be the image's older kernel.  Expect
# md5=1e221f10cbf3573aa0d2ae78c3463532 for the incumbent at c3a18c95.
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V/streamindex_topk_c.py"

for ARM in "$@"; do
  echo "===== OPTC ARM $ARM BEGIN ====="
  $PY "$V/dump_llo_c.py" --arm "$ARM" --outdir "/tmp/optcdump/$ARM" 2>&1 | tail -60
  RC=${PIPESTATUS[0]}
  echo "===== OPTC ARM $ARM END rc=$RC ====="
  mkdir -p "$OUT/pass/$ARM"
  find "/tmp/optcdump/$ARM/jf" \( -name "*StreamIdx*41-post-DCE-3.txt" \
       -o -name "*StreamIdx*53-post-ra.txt" \
       -o -name "*StreamIdx*70-schedule-analysis_final_bundles.txt" \) \
       2>/dev/null | while read -r f; do
    cp "$f" "$OUT/pass/$ARM/" 2>/dev/null
  done
  echo "--- kept for $ARM ---"
  ls -la "$OUT/pass/$ARM" 2>/dev/null
  echo "--- scheduled bundles for $ARM ---"
  head -4 "$OUT/pass/$ARM"/*70-schedule-analysis_final_bundles.txt 2>/dev/null
  echo "--- spill lines for $ARM ---"
  grep -c "_spill" "$OUT/pass/$ARM"/*53-post-ra.txt 2>/dev/null
  du -sh "/tmp/optcdump/$ARM" 2>/dev/null
done
exit 0
