#!/bin/bash
# Provenance read-back, shared by every Option D job.
#
# R2's first job (`j-adddca06`) was green with all four shards at rc=1 because
# the image's own `src` shadowed the mapped-in kernels on PYTHONPATH, and the
# probe silently measured the image's older kernel. A green `JobStatus` is not
# evidence that your code ran. So: import the modules the way the probe will
# import them, print the file that actually resolved and its md5, and print the
# variant's own md5 too.
#
# Expected, and asserted by the caller reading this back:
#   streamindex_topk.py  md5=1e221f10cbf3573aa0d2ae78c3463532   (== R2, == HEAD)
#   sparsecore_topk.py   md5=8bc9e25b8701a53a97704564b9fbc658   (== R2, == HEAD)
PY=${1:-python}
echo "===== OPTD PROVENANCE ====="
$PY -c "import jax, jaxlib; print(f'jax {jax.__version__} jaxlib {jaxlib.__version__}')"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu) " || true
$PY - <<PYEOF
import hashlib, os, sys
sys.path.insert(0, os.environ.get("D_VARIANT", "/root/optd"))
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import streamidx_hoist
for m in (streamindex_topk, sparsecore_topk, streamidx_hoist):
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
print("EARLY_EXIT_IN_SIG",
      "enable_early_exit" in open(streamindex_topk.__file__).read())
print("KERNEL_IDENTITY streamidx_hoist.kernel is streamindex_topk.kernel:",
      streamidx_hoist.kernel is streamindex_topk.kernel)
PYEOF
echo "===== OPTD PROVENANCE END ====="
