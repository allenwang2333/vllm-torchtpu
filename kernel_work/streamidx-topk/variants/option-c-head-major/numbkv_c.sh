#!/bin/bash
# Option C follow-up 1: chain slope vs num_bkv, prefill + decode.
#
# ONE unpinned process on the whole 4-chip pod, like `timing_c.sh` and for the
# same reason: these are timings and a fanned-out capture is contended.  Nothing
# else may share this job.
#
# The sweep is 7 prefill points x 2 arms + 1 decode point x 2 arms = 16 arms,
# each of which compiles 4 chain lengths, so expect a long Mosaic compile tail.
V=${V:-/root/kw/variants/option-c-head-major}
PY=${PY:-python}
TRIALS=${TRIALS:-7}
# num_bkv = 1,2,3,4,5,8.  ctx 32768 (num_bkv=16) is omitted deliberately: it is
# deep in the amortised regime the 9216 point already establishes, and it costs
# a 128 MB score buffer and four more Mosaic compiles to learn nothing new.
CTX=${CTX:-2048,4096,6144,8192,9216,16384}

echo "===== OPTC NUMBKV ENV ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
$PY - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as m
print("MODULE", m.__file__, "md5=" +
      hashlib.md5(open(m.__file__, "rb").read()).hexdigest())
PYEOF
md5sum "$V"/*.py

echo "===== OPTC NUMBKV BEGIN trials=$TRIALS ctx=$CTX ====="
# Not piped through `tail`; `emit_json` puts the machine-readable payload on
# stdout and the per-point tables are what the write-up recomputes from.
$PY "$V/bench_numbkv.py" --trials "$TRIALS" --prefill-ctx "$CTX" \
    --arms incumbent,head_major --decode-ctx 8704 2>&1
echo "===== OPTC NUMBKV END rc=$? ====="
exit 0
