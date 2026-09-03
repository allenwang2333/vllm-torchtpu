#!/bin/bash
# Option F: bf16 score-matmul operands, chain slope at prefill and decode.
#
# ONE unpinned process on the whole 4-chip pod -- these are timings and a
# fanned-out capture is contended.  Nothing else may share this job.
#
# 2 shapes x 4 arms x 4 chain lengths = 32 Mosaic compiles; expect a compile
# tail longer than the measurement.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}
TRIALS=${TRIALS:-7}
ARMS=${ARMS:-incumbent,f_bf16,head_major,f_bf16_hm}

echo "===== OPTF ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<'PYEOF'
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTF BEGIN trials=$TRIALS arms=$ARMS ====="
$PY "$V/bench_optionf.py" --trials "$TRIALS" --arms "$ARMS" 2>&1
echo "===== OPTF END rc=$? ====="
exit 0
