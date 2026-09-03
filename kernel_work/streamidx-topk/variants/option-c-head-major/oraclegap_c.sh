#!/bin/bash
# Option C follow-up 2: are the f64 oracle's 10 disagreements with the SHIPPING
# kernel near-ties, or a real defect?
#
# Not a timing -- safe to fan out, safe to share a pod.  Three seeds, so a
# verdict does not rest on one draw.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}
SEEDS=${SEEDS:-0,1,2}
OUT=${CDK_OUTPUT_DIR:-/tmp}/optc

mkdir -p "$OUT"
echo "===== OPTC ORACLEGAP ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sb
for mod in (m, sb):
    print("MODULE", mod.__file__, "md5=" +
          hashlib.md5(open(mod.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTC ORACLEGAP BEGIN seeds=$SEEDS ====="
$PY "$V/probe_oracle_gap.py" --seeds "$SEEDS" \
    --arms incumbent,head_major --json "$OUT/oracle_gap.json" 2>&1
echo "===== OPTC ORACLEGAP END rc=$? ====="
exit 0
