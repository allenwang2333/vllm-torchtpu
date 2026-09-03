#!/bin/bash
# Which MXU pass sets stage A's score error, and what dtype the operands are.
#
# NOT a timing -- it is a numerical comparison against a numpy f64 reference,
# so it is safe to share a pod with other shards.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}
SEEDS=${SEEDS:-0,1,2}

echo "===== OPTF PRECISION ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<'PYEOF'
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTF PRECISION BEGIN seeds=$SEEDS ====="
$PY "$V/probe_precision.py" --seeds "$SEEDS" 2>&1
echo "===== OPTF PRECISION END rc=$? ====="
exit 0
