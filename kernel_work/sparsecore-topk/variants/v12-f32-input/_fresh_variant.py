
import hashlib, importlib.util, pathlib, re, sys
import jax, jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info
_T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1
from jax.experimental import topologies
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import SingleDeviceSharding
pltpu.get_tpu_info = lambda: _T
DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)
B, N, K = 1024, 10240, 2048

path = pathlib.Path(sys.argv[1])
print("source_path", path)
print("source_md5", hashlib.md5(path.read_bytes()).hexdigest())
spec = importlib.util.spec_from_file_location("arm_under_test", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

# Challenge 1: prove the symbol we compile is the one this file defines, and
# that exactly one arm module is resident in this process.
fn = m._sc_topk_direct
print("bound_symbol_module", fn.__module__, "id", hex(id(fn)))
resident = [k for k, v in sys.modules.items()
            if getattr(v, "__file__", None) and
            pathlib.Path(v.__file__).name == "sparsecore_topk.py"]
print("resident_sparsecore_topk_modules", resident)
assert len(resident) == 1, resident
print("jit_cache_size_before", jax._src.pjit._cpp_pjit_cache_shard_count()
      if hasattr(jax._src.pjit, "_cpp_pjit_cache_shard_count") else "n/a")

s = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=SH)
l = jax.ShapeDtypeStruct((B,), jnp.int32, sharding=SH)
c = jax.jit(lambda a, b: m._sc_topk_direct(a, K, b),
            out_shardings=SH).lower(s, l).compile()
txt = c.as_text()
e = re.search(r"^ENTRY .*?^}", txt, re.S | re.M).group(0)
body = [ln for ln in e.splitlines()[1:-1] if ln.strip()]
print("ENTRY_instructions", len(body))
for pat in ("fusion(", "bitcast-convert(", " copy(", "copy-start(",
            "bitcast(", "custom-call"):
    print(f"ENTRY_count {pat!r}", sum(pat in ln for ln in body))
ma = c.memory_analysis()
for f in ("temp_size_in_bytes", "argument_size_in_bytes",
          "output_size_in_bytes", "generated_code_size_in_bytes"):
    print("mem", f, getattr(ma, f, None))
print("ENTRY_BEGIN")
print(e)
print("ENTRY_END")
