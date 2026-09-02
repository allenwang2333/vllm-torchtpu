#!/bin/bash
# R3 shard: compile one or more arms of the rotate-attribution A/B under the
# `/xprof-llo` dump-side LIBTPU_INIT_ARGS, one arm per process (libtpu reads
# LIBTPU_INIT_ARGS once at load time, and two dump flags pointed at one path
# interleave into an unreadable mess).
#
#   $@ = arm names, run sequentially in this shard
#
# Dumps go to /tmp (node-local scratch, gigabytes), NOT $CDK_OUTPUT_DIR.  Only
# the one pass the reader needs is copied back.
#
# Compile-only work: no profiler, so this runs on the image's pinned jax and
# cannot hit the PLUGIN_Profiler_Api SIGSEGV.
#
# No `set -e`: a failed arm is a result and must not take the others with it.
P=${P:-/root/kw/probes}
OUT=${CDK_OUTPUT_DIR:-/tmp}/r3
mkdir -p "$OUT"
PY=${PY:-python}

echo "===== R3 ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"

for ARM in "$@"; do
  echo "===== R3 ARM $ARM BEGIN ====="
  $PY "$P/probe_r3_rotate_ab.py" --arm "$ARM" --outdir "/tmp/r3dump/$ARM" \
      2>&1 | tail -60
  RC=${PIPESTATUS[0]}
  echo "===== R3 ARM $ARM END rc=$RC ====="
  # keep only the pass the reader parses, plus the post-ra allocation table
  mkdir -p "$OUT/pass/$ARM"
  find "/tmp/r3dump/$ARM/jf" \( -name "*R3Score*41-post-DCE-3.txt" \
       -o -name "*R3Score*53-post-ra.txt" \
       -o -name "*R3Score*70-schedule-analysis_final_bundles.txt" \
       -o -name "*R3Score*67-pre-delay_hlo-static-per-bundle-utilization.txt" \
       -o -name "*R3Score*69-final_hlo-static-per-bundle-utilization.txt" \) \
       2>/dev/null | while read -r f; do
    cp "$f" "$OUT/pass/$ARM/" 2>/dev/null
  done
  echo "--- kept for $ARM ---"
  ls -la "$OUT/pass/$ARM" 2>/dev/null
  echo "--- scheduled bundles for $ARM ---"
  head -4 "$OUT/pass/$ARM"/*70-schedule-analysis_final_bundles.txt 2>/dev/null
  du -sh "/tmp/r3dump/$ARM" 2>/dev/null
done
exit 0
