#!/bin/bash
# Option C measurement 1: prefill stage-A device time, all four arms.
#
# ONE unpinned process on the whole 4-chip pod, deliberately -- not fanned out.
# `probes/stage-b-ksweep.md` establishes that a fanned-out capture is contended
# and yields ratios only, and these numbers have to be comparable in absolute
# microseconds against the 335.1 us stage-A figure and the 280 us pass mark.
#
# Device time comes from the chain-slope fit, not from a wall clock: `time_arm`
# enqueues asynchronously and blocks once, so its per-call figure is
# max(device, host), and a dispatch floor is a clamp rather than an addend --
# subtracting one is invalid.  R calls are chained inside a single jit through
# the aliased `scores_init` -> output dependency, which is a genuine data
# dependency the algebraic simplifier cannot fold.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}
TRIALS=${TRIALS:-9}

echo "===== OPTC TIMING ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTC TIMING BEGIN trials=$TRIALS ====="
# NOT piped through `tail`: `emit_json` writes the machine-readable payload to
# stdout, and truncating it would leave only the prose summary to recompute
# from.  A copy also lands in $CDK_OUTPUT_DIR/probes.
$PY "$V/bench_headmajor.py" --trials "$TRIALS" \
    --arms incumbent,head_minor,head_major,head_major_hbm 2>&1
echo "===== OPTC TIMING END rc=$? ====="

echo "===== OPTC TIMING ARTIFACTS ====="
find "${CDK_OUTPUT_DIR:-/tmp}" -name "*option_c*" -o -name "*.json" 2>/dev/null | head -20
exit 0
