
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
