
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

# Challenge 2: the control must be refused by the SparseCore ALLOCATOR, not by
# the MLIR layer. 8 MiB of VMEM scratch against a 512 KiB SC tile, with a real
# DMA to HBM so the scratch cannot be dead-code-eliminated.
def kernel(o_hbm, big, sems):
    cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0], sems.at[0])
    cp.start()
    cp.wait()

def build():
    return pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
        mesh=plsc.VectorSubcoreMesh(
            num_cores=_T.sparse_core.num_cores,
            num_subcores=_T.sparse_core.num_subcores,
            core_axis_name="core", subcore_axis_name="subcore"),
        scratch_types=(pltpu.VMEM((2 * 1024 * 1024,), jnp.int32),
                       pltpu.SemaphoreType.DMA((1,))),
        name="aot_negctl_oversized_vmem")()

try:
    jax.jit(build, out_shardings=SH).lower().compile()
    print("CONTROL_RESULT COMPILED -- CONTROL FAILED")
except Exception as exc:
    print("CONTROL_RESULT REFUSED")
    print("CONTROL_EXC_TYPE", type(exc).__name__)
    print("CONTROL_FULL_TEXT_BEGIN")
    print(str(exc))
    print("CONTROL_FULL_TEXT_END")
