#!/bin/bash
# Provenance read-back, shared by every D1-lite shard.
#
# The pinned image's kernels are STALE: its `streamindex_topk.py` differs from
# the checkout and it has no `sparsecore_topk.py` at all. Both are supplied by
# `--map-dir <checkout>/src`. A green `JobStatus` is not evidence that the
# mapped-in code is what ran -- an earlier job in this exercise was green with
# every shard at rc=1 because the image's own `src` shadowed the mapped-in
# kernels on PYTHONPATH.
#
# So: import the modules the way the probe will import them, print the file
# that actually resolved and its md5, and print the variant module's md5 too.
#
# Expected, and asserted by the caller reading this back:
#   streamindex_topk.py  md5=1e221f10cbf3573aa0d2ae78c3463532
#   sparsecore_topk.py   md5=8bc9e25b8701a53a97704564b9fbc658
PY=${1:-python}
echo "===== D1LITE PROVENANCE ====="
$PY -c "import jax, jaxlib; print(f'jax {jax.__version__} jaxlib {jaxlib.__version__}')"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu) " || true
$PY - <<PYEOF
import hashlib, os, sys
sys.path.insert(0, os.environ.get("DL_VARIANT", "/root/d1lite"))
sys.path.insert(0, os.environ.get("DL_INSTRUMENTS", "/root/instruments"))
import vllm_torchtpu
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import streamindex_topk_d1lite as DL
import bench_streamidx_topk as B
print("PKG vllm_torchtpu", vllm_torchtpu.__file__)
for m in (streamindex_topk, sparsecore_topk, DL, B):
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
print("EARLY_EXIT_IN_SIG",
      "enable_early_exit" in open(streamindex_topk.__file__).read())
print("KERNEL_IDENTITY d1lite.kernel is streamindex_topk.kernel:",
      DL.kernel is streamindex_topk.kernel)
print("HELPER_IDENTITY d1lite._effective_row_lengths is incumbent's:",
      DL._effective_row_lengths is streamindex_topk._effective_row_lengths)
print("BENCH_CONSTS k=%d cr=%d page=%d bkvp=%s nqpb=%s mml=%d" % (
    B.K_IDX, B.COMPRESSION_RATIO, B.PAGE_SIZE, B.NUM_KV_PAGES_PER_BLOCK,
    B.NUM_QUERIES_PER_BLOCK, B.PROD_MAX_MODEL_LEN))
PYEOF
echo "===== D1LITE PROVENANCE END ====="
