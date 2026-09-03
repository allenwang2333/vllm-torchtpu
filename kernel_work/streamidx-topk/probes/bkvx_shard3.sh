#!/bin/bash
# Job 3: close the last gap in the f64 oracle's coverage.
#
# Jobs j-6a35b078 and j-ec3b2a64 checked bkv_p in {1,2,3,4,5,8,9,10,16} against
# the f64 numpy top-k oracle and found the emitted index set byte-identical at
# every one of them.  Section G then showed the recommendation is `p = P` for
# every ctx bucket, which promotes p = 6 (ctx 6144) and p = 7 (ctx 7168) from
# "swept for timing" to "would actually be deployed" -- and those two are the
# only values in 1..9 the oracle never ran.  This job runs them, so the
# correctness statement covers the whole deployable range rather than most of
# it.
#
# Section B only; the timing sections are all already in hand.
P=${P:-/root/probes}
export BDT_INSTRUMENTS=${BDT_INSTRUMENTS:-/root/instruments}
PY_SYS=${PY_SYS:-python}
echo "===== BKVX3 ENV $(date -u) ====="
$PY_SYS - <<'PYEOF'
import hashlib, os, sys
sys.path.insert(0, os.environ.get("BDT_INSTRUMENTS", "/root/instruments"))
sys.path.insert(0, "/root/probes")
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import bench_streamidx_topk as B, _stage_a as SA, jax
for m in (streamindex_topk, sparsecore_topk, B, SA):
    print(f"MODULE {m.__file__} md5={hashlib.md5(open(m.__file__,'rb').read()).hexdigest()}")
print("EXPECT streamindex_topk.py     md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py      md5=8bc9e25b8701a53a97704564b9fbc658")
print("EXPECT bench_streamidx_topk.py md5=e8e492d13e59c4a1675cc7a98c5a0813")
print("EXPECT _stage_a.py             md5=eea6a51bb40a2bbf7cf5b39571753b62")
print("device", jax.devices()[0].device_kind, "n", len(jax.devices()))
PYEOF
echo "===== BKVX3 STEP oracle BEGIN $(date -u +%H:%M:%S) ====="
timeout "${ORACLE_TIMEOUT:-3600}" $PY_SYS "$P/probe_bkvp_extended.py" \
    --name bkvp-extended-oracle67 --skip A,A2,C,D,E,F,G \
    --p-max 9 --oracle-p 6,7
echo "===== BKVX3 STEP oracle END rc=$? $(date -u +%H:%M:%S) ====="
find "${CDK_OUTPUT_DIR:-/tmp}" -name "bkvp-extended*.json" | head
exit 0
