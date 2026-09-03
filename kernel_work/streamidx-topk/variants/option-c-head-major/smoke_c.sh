#!/bin/bash
# Option C smoke: compile and run every arm once, no timing.
#
# Its whole job is to find a Mosaic compile failure -- most plausibly the
# "Slice sizes along tiled dimensions must be divisible by 128" constraint on
# the `head_major_hbm` strided q copy -- for the price of one short job rather
# than inside the timing job, where a dead arm would cost the whole run.
#
# It also prints the arm-vs-arm numeric agreement, so a `head_minor` that has
# drifted from the incumbent shows up here first.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}

echo "===== OPTC SMOKE ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTC SMOKE BEGIN ====="
$PY "$V/bench_headmajor.py" --no-chain --trials 1 \
    --arms incumbent,head_minor,head_major,head_major_hbm 2>&1 | tail -120
echo "===== OPTC SMOKE END rc=${PIPESTATUS[0]} ====="
exit 0
