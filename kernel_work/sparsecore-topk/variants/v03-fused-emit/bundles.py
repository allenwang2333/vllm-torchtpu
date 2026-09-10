"""Dump SCHEDULED SparseCore VLIW bundles for ONE arm, host-only, no TPU.

Usage: bundles.py <base|v3> <dumpdir>

The bundle flags are LIBTPU_INIT_ARGS, read once at libtpu load -- they must be
in the environment BEFORE `import jax`, and XLA rejects them by name if they go
in XLA_FLAGS.  Paired with XLA_FLAGS=--xla_dump_to for buffer assignment.

Single-arm by construction so no jit cache can alias the two arms; identity is
asserted three ways (source path, source md5, kernel name in the HLO).
"""
import hashlib
import os
import re
import sys

ARM, DUMP = sys.argv[1], sys.argv[2]
os.makedirs(DUMP, exist_ok=True)

# ---- MUST precede `import jax`.  LIBTPU_INIT_ARGS, not XLA_FLAGS. -----------
os.environ["LIBTPU_INIT_ARGS"] = " ".join((
    os.environ.get("LIBTPU_INIT_ARGS", ""),
    f"--xla_sc_dump_bundles_to={DUMP}",
    f"--xla_sc_dump_llvm_ir_to={DUMP}",
    f"--xla_sc_dump_mlir_to={DUMP}",
)).strip()
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           f" --xla_dump_to={DUMP}").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax._src.pallas.mosaic import tpu_info as _tpu_info  # noqa: E402

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

import importlib.util  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PATH = {
    "base": os.path.join(ROOT, "src", "vllm_torchtpu", "kernels",
                         "deepseek_v4", "sparsecore_topk.py"),
    "v3": os.path.join(HERE, "sparsecore_topk.py"),
}[ARM]
md5 = hashlib.md5(open(PATH, "rb").read()).hexdigest()
print(f"ARM={ARM}\nPATH={PATH}\nMD5={md5}")
print(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}")

spec = importlib.util.spec_from_file_location("sctopk_arm", PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["sctopk_arm"] = mod
spec.loader.exec_module(mod)
assert mod.__file__ == PATH, mod.__file__

DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)
B, N, K = 1024, 10240, 2048

hlo = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l),
              out_shardings=SH).lower(
                  jax.ShapeDtypeStruct((B, N), jnp.float32),
                  jax.ShapeDtypeStruct((B, ), jnp.int32)).compile().as_text()
names = sorted(set(re.findall(r"sc_topk_direct_[A-Za-z0-9_]+", hlo)))
print(f"KERNEL NAMES IN HLO: {names}")
assert names, "no kernel name in HLO -- arm identity unverifiable"
if ARM == "v3":
    assert any("_v3fused" in x for x in names), names
else:
    assert not any("_v3fused" in x for x in names), names

files = sorted(os.listdir(DUMP))
print(f"\n{len(files)} dump files:")
for f in files:
    print(f"  {os.path.getsize(os.path.join(DUMP, f)):10d}  {f}")
