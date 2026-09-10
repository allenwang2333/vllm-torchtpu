
import os, pathlib, sys
# LIBTPU_INIT_ARGS is read by libtpu once, at load time -- so it must be set
# before `import jax`. These are NOT XLA_FLAGS; XLA rejects them by name.
# (xprof-llo SKILL.md:109-111)
_d = sys.argv[2]
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={_d} --xla_jf_dump_llo_text=true "
    f"--xla_mosaic_dump_to={_d}")
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
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("m", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
sh = SingleDeviceSharding(dev)
B, N, K = 1024, 10240, 2048
s = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
l = jax.ShapeDtypeStruct((B,), jnp.int32, sharding=sh)
jax.jit(lambda a, b: m._sc_topk_direct(a, K, b),
        out_shardings=sh).lower(s, l).compile()
print("compiled ok")
