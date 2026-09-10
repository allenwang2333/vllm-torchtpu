"""Price the seven in-body bitcasts against emitted SparseCore code.

STATIC COMPILE-TIME ANALYSIS ONLY.

`--xla_sc_dump_bundles_to` (a LIBTPU_INIT_ARGS flag, not an XLA_FLAG) makes the
SparseCore backend write its scheduled VLIW bundles to disk at compile time. It
works on a host with **no TPU**, so the two arms' bundle counts are comparable
for zero cluster time.

The comparison is deliberately NOT incumbent-vs-v12 as compiled from the real
call site: those two modules also differ by a 40 MiB TensorCore fusion, which
would contaminate any program-size number. Both arms here are handed an input
array of their own dtype directly, so the ONLY difference between the two SC
programs is the seven `arith.bitcast`s.

Run: uv_venv/bin/python3 sc_bundles.py
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
VARIANT = HERE / "sparsecore_topk.py"
PY = HERE.parents[3] / "uv_venv" / "bin" / "python3"

CHILD = r'''
import os, sys
_d = sys.argv[2]
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_sc_dump_bundles_to={_d} --xla_sc_dump_mlir_to={_d} "
    f"--xla_sc_dump_llvm_ir_to={_d}")
import jax, jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
_T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import SingleDeviceSharding
pltpu.get_tpu_info = lambda: _T
import importlib.util
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
sh = SingleDeviceSharding(dev)
B, N, K = 1024, 10240, 2048
s = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
l = jax.ShapeDtypeStruct((B,), jnp.int32, sharding=sh)
c = jax.jit(lambda a, b: m._sc_topk_direct(a, K, b),
            out_shardings=sh).lower(s, l).compile()
print("generated_code_size_in_bytes",
      c.memory_analysis().generated_code_size_in_bytes)
print("compiled ok")
'''


def run(tag, src):
    out = HERE / f"scdump-{tag}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    child = HERE / "_child_sc.py"
    child.write_text(CHILD)
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    r = subprocess.run([str(PY), str(child), str(src), str(out)],
                       env=env,
                       capture_output=True,
                       text=True,
                       timeout=3600)
    print(f"--- {tag}: rc={r.returncode}")
    for ln in r.stdout.strip().splitlines()[-2:]:
        print(f"    {ln}")
    if r.returncode:
        print(r.stderr[-2000:])
    return out


BUNDLE_RE = re.compile(r"^\s*\{", re.M)


def summarize(tag, out):
    files = sorted(p for p in out.rglob("*") if p.is_file())
    print(f"\n===== {tag}: {len(files)} dump file(s) =====")
    for p in files:
        txt = p.read_text(errors="replace")
        print(f"  {p.name[:90]}  {p.stat().st_size} B  {len(txt.splitlines())} lines")
    return files


if __name__ == "__main__":
    for tag, src in (("incumbent", INCUMBENT), ("variant", VARIANT)):
        summarize(tag, run(tag, src))
