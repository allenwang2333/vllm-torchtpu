#!/bin/bash
# LLO shard for the Option F / decode follow-up.  Same mechanism as
# `shard_llo_c.sh` -- one arm per process under the `/xprof-llo` dump-side
# LIBTPU_INIT_ARGS -- with two additions:
#
#   $1  = phase, `prefill` or `decode`
#   $2..= arm names, run sequentially in this shard
#
# `decode` compiles the DECODE launcher (bq_sz=1, seq_batch_size=4), which is
# the program RESULTS.md 7.3's +59.62% belongs to and which no earlier dump
# covered.  The arm names now include the Option F ones, whose program name
# carries a `-bf16` / `-f32` suffix so the dumps stay distinguishable.
#
# Compile-time analysis only.  Nothing here is a timing.
# No `set -e`: a failed arm is a result and must not take the others with it.
V=${V:-/root/kw/variants/option-c-head-major}
PHASE=${1:?usage: shard_llo_f.sh <prefill|decode> <arm>...}
shift
OUT=${CDK_OUTPUT_DIR:-/tmp}/optf/$PHASE
mkdir -p "$OUT"
PY=${PY:-python}

echo "===== OPTF LLO ENV phase=$PHASE ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
$PY - <<'PYEOF'
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

for ARM in "$@"; do
  echo "===== OPTF ARM $PHASE/$ARM BEGIN ====="
  # Do NOT pipe this through `tail`.  The first run did, with `tail -80`, and
  # both decode arms emitted exactly 80 `MATMUL` lines -- which silently ate
  # the `phase`/`geometry` and `arm ... ok ... finite_frac` header lines that
  # are the only in-job evidence the right program ran.  See RESULTS.md 10.3.
  # Keep the header, and cap only the MATMUL flood, which is what is long.
  $PY "$V/dump_llo_c.py" --arm "$ARM" --phase "$PHASE" \
      --outdir "/tmp/optfdump/$PHASE/$ARM" 2>&1 \
    | awk '/\[C-dump\] MATMUL/ {m++; if (m>24) next} {print}'
  RC=${PIPESTATUS[0]}
  echo "===== OPTF ARM $PHASE/$ARM END rc=$RC ====="
  mkdir -p "$OUT/$ARM"
  find "/tmp/optfdump/$PHASE/$ARM/jf" \( -name "*StreamIdx*41-post-DCE-3.txt" \
       -o -name "*StreamIdx*53-post-ra.txt" \
       -o -name "*StreamIdx*70-schedule-analysis_final_bundles.txt" \) \
       2>/dev/null | while read -r f; do
    cp "$f" "$OUT/$ARM/" 2>/dev/null
  done
  echo "--- kept for $PHASE/$ARM ---"
  ls -la "$OUT/$ARM" 2>/dev/null
  echo "--- scheduled bundles ---"
  head -4 "$OUT/$ARM"/*70-schedule-analysis_final_bundles.txt 2>/dev/null
  echo "--- vrot.slane / vxpose / vpack counts ---"
  for OP in vrot.slane vxpose vpack vmatmul vadd; do
    printf '%-12s %s\n' "$OP" \
      "$(grep -ao "$OP" "$OUT/$ARM"/*41-post-DCE-3.txt 2>/dev/null | wc -l)"
  done
  echo "--- spill lines ---"
  grep -c "_spill" "$OUT/$ARM"/*53-post-ra.txt 2>/dev/null
done
exit 0
